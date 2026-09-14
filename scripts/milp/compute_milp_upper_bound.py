"""MILP upper bound on EXACTLY the same scenario as the RL/eval episodes.

The environment is built identically to ``scripts/baselines/run_baselines.py`` (same
validation profile from ``configs/global.yaml``: start mode, activation
window, episode length, seeds and start_seed). For every evaluation seed the
env is reset with the same arguments the baselines use, so the episode starts
at the same ``random_day`` slot with the same request stream; the MILP then
solves that exact episode's window offline with full future knowledge and
reports the strict success-rate upper bound (``-HiGHS dual bound``), which any
policy executed on that same episode cannot exceed.

Usage (mirrors run_baselines.py):
    python scripts/milp/compute_milp_upper_bound.py                  # seeds from global.yaml
    python scripts/milp/compute_milp_upper_bound.py --seeds 7,8,9    # explicit seeds
    python scripts/milp/compute_milp_upper_bound.py --episode-start-mode fixed
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore", message=".*iCCP.*")

from qkd_rl.core.config import (ConfigValidator, deep_merge, load_config)
from qkd_rl.env.factory import load_default_config, build_env_from_config


def _validation_profile() -> dict:
    """Parse the validation block of global.yaml (mirrors
    qkd_rl.evaluation.test_protocol.load_validation_profile but without
    importing the evaluation package, which pulls in matplotlib)."""
    import yaml
    raw = yaml.safe_load((ROOT / "configs" / "global.yaml").read_text(encoding="utf-8")) or {}
    ev = (raw.get("global") or {}).get("validation") or {}
    window = ev.get("window") or {}
    start_day = int(window.get("start_day", 330))
    end_day = int(window.get("end_day", start_day + 1))
    seeds = [int(s) for s in (ev.get("request_seeds", []) or ev.get("seeds", []) or [])]
    return {
        "window_start_day": start_day,
        "window_end_day": end_day,
        "episode_steps": int(ev.get("episode_steps") or 240),
        "seeds": seeds,
        "start_seed": int(ev.get("start_seed", 0)),
        "start_mode": str(ev.get("start_mode", "random_day")),
    }


def _build_config(profile: dict, start_mode: str, episode_steps: int) -> dict:
    """Same config shape as build_validation_env_config (run_baselines.py)."""
    start_day = profile["window_start_day"]
    end_day = profile["window_end_day"]
    window_days = max(0, end_day - start_day)
    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "global.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = start_mode
    config["env"]["episode_steps"] = episode_steps
    if start_mode == "random_day":
        config["env"]["activation_window_start_day"] = start_day
        config["env"]["activation_window_end_day"] = end_day
        config["env"]["activation_window_days"] = window_days
        config["scenario"]["time_limit"]["days"] = end_day + max(1, math.ceil(episode_steps / 1440))
    else:
        config["scenario"]["time_limit"]["days"] = end_day
    ConfigValidator().validate(config)
    return config


def main():
    parser = argparse.ArgumentParser(description="MILP upper bound on the same scenario as the RL/eval episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (ignored if --seeds given).")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (default: global.yaml request_seeds).")
    parser.add_argument("--episode-start-mode", type=str, default=None,
                        help="Override start mode (e.g. fixed); default: validation start_mode (random_day).")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--rel-gap", type=float, default=0.0,
                        help="HiGHS mip_rel_gap (0.0 proves optimality; >0 stops earlier, bound stays valid).")
    parser.add_argument("--out", type=str, default="outputs/eval/milp_ub")
    parser.add_argument("--max-requests", type=int, default=512)
    parser.add_argument("--max-paths", type=int, default=256,
                        help="Path candidates per request; <100 silently underestimates the bound.")
    parser.add_argument("--max-hops", type=int, default=6)
    parser.add_argument("--window-steps", type=int, default=None,
                        help="MILP horizon in slots. Default = the full episode. A single "
                             "full-episode window (240 slots) does NOT converge in a "
                             "practical time limit: status stays at time-limit and the "
                             "printed bound is only a loose LP/dual bound. Use 30-60 for "
                             "a converged window optimum (tight bound, but a receding-"
                             "horizon approximation of the global optimum).")
    args = parser.parse_args()

    profile = _validation_profile()
    start_mode = args.episode_start_mode or profile["start_mode"]
    episode_steps = int(profile["episode_steps"])
    window_steps = args.window_steps or episode_steps
    seeds = []
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = profile["seeds"]
    if not seeds:
        seeds = [7]
    print(f"profile: mode={start_mode} days={profile['window_start_day']}->{profile['window_end_day']} "
          f"episode_steps={episode_steps} seeds={seeds} start_seed={profile['start_seed']}", flush=True)

    config = _build_config(profile, start_mode, episode_steps)
    env = build_env_from_config(config)

    from qkd_rl.baselines.receding_horizon_milp import RecedingHorizonMILPPolicy

    class _Light: pass

    per_seed = []
    for seed in seeds:
        # Identical reset to run_baselines.py's env_builder.
        obs = env.reset(seed=seed, start_seed=int(profile["start_seed"]) + seed)
        t0 = int(obs.state.t)
        print(f"seed={seed}: episode starts t={t0} (day {t0/1440:.1f})", flush=True)

        pol = RecedingHorizonMILPPolicy(
            config, window_steps=window_steps, max_requests=args.max_requests,
            max_paths_per_request=args.max_paths, max_path_hops=args.max_hops,
            time_limit_s=args.time_limit, mip_rel_gap=args.rel_gap,
            # Single-window ideal bound: no cross-window stocking incentive.
            final_inventory_weight=0.0,
        )
        # Inject the env's generator: same RNG position -> the preview equals
        # the requests the env will actually generate on this episode, and the
        # O(t) fast-forward is skipped.
        pol._ensure_init(obs, gen=env.request_generator)

        state = _Light()
        state.t = t0
        state.pending_requests = list(obs.state.pending_requests)
        state.qkp_snapshot = env.qkp.snapshot() if hasattr(env, "qkp") else {}
        state.qkp_capacity = {}
        if hasattr(env, "qkp"):
            for eid in obs.physical_edge_ids:
                try:
                    state.qkp_capacity[eid] = env.qkp.get_capacity(eid)
                except Exception:
                    pass
        state.edge_windows = obs.state.edge_windows
        holder = _Light()
        holder.state = state
        holder.node_ids = obs.node_ids
        holder.physical_edge_ids = obs.physical_edge_ids

        w0 = time.perf_counter()
        outcome = pol.solve_window(holder)
        solve_s = time.perf_counter() - w0

        # Denominator = the remaining demand of the requests the MILP actually
        # modelled (after dropping pathless requests). The bound is valid for
        # these servable requests; using the raw arrived demand (which includes
        # pathless/unsupported requests) would systematically understate it.
        modeled = outcome.modeled_amount
        sr_ub = outcome.upper_bound_amount / max(1e-9, modeled) if modeled > 0 else 0.0
        feasible_sr = outcome.served_amount / max(1e-9, modeled) if modeled > 0 else 0.0
        print(f"  solve={solve_s:.1f}s status={outcome.status} gap={outcome.mip_gap:.4f} "
              f"modeled={modeled:,.0f} ub_served={outcome.upper_bound_amount:,.0f} "
              f"SUCCESS_RATE_UPPER_BOUND={sr_ub:.4f} feasible_sr={feasible_sr:.4f}", flush=True)
        if outcome.status != 0:
            print(f"  WARNING: status={outcome.status} -> optimality NOT proven; "
                  f"SUCCESS_RATE_UPPER_BOUND is only a dual/LP bound, not a tight optimum. "
                  f"Use --window-steps 30-60 (and/or a larger --time-limit) for a converged "
                  f"window optimum.", flush=True)

        per_seed.append({
            "seed": seed, "start_step": t0, "window_steps": window_steps,
            "modeled_amount": modeled, "total_served_upper": outcome.upper_bound_amount,
            "total_served_feasible": outcome.served_amount,
            "success_rate_upper_bound": sr_ub, "success_rate_feasible": feasible_sr,
            "converged": outcome.status == 0, "mip_gap": outcome.mip_gap,
            "solve_s": solve_s, "status": outcome.status,
        })

    mean_ub = sum(r["success_rate_upper_bound"] for r in per_seed) / len(per_seed)
    print(f"\nMEAN SUCCESS_RATE_UPPER_BOUND over {len(per_seed)} episode(s) = {mean_ub:.4f}", flush=True)

    summary = {
        "profile": {"start_mode": start_mode, "window_start_day": profile["window_start_day"],
                    "window_end_day": profile["window_end_day"], "episode_steps": episode_steps,
                    "start_seed": profile["start_seed"]},
        "milp": {"time_limit_s": args.time_limit, "mip_rel_gap": args.rel_gap,
                 "max_requests": args.max_requests, "max_paths": args.max_paths,
                 "max_hops": args.max_hops, "window_steps": window_steps},
        "episodes": per_seed,
        "mean_success_rate_upper_bound": mean_ub,
    }
    out_dir = Path(ROOT) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()