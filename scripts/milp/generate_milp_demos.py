"""Generate GLOBAL god's-eye-view MILP demos that carry per-slot gold link
choices for RL supervision (behaviour cloning).

Each episode is solved ONCE with full temporal information (a single open-loop
window over the whole segment, final_inventory_weight=0), producing the optimal
per-slot directed link plan. The env then executes that single plan to collect
the real observations a policy would see, and every trajectory step stores the
MILP's gold ``(tx_target, rx_source)`` link decision for that observation. The
demo's ``success_rate`` is the global plan's SR on modelled demand (the target
quality gate); the executed SR is reported separately but is NOT the RL target.

Episodes are drawn from the SAME validation profile as RL eval
(configs/global.yaml → global.validation: window days, request seeds, start_seed,
episode length). Each episode runs in its own env instance and is fully
independent; numbering continues off files already on disk and never
overwrites. Every demo records its environment (time range, request seed,
start_seed, episode length, MILP horizon) so later RL guiding can replay the
same conditions.

Usage:
    python scripts/generate_milp_demos.py                       # global-view validation demos
    python scripts/generate_milp_demos.py --window-steps 60     # smaller global horizon
    python scripts/generate_milp_demos.py --append --out outputs/milp_demos
"""
from __future__ import annotations
import os, sys, time, json, random, math, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from qkd_rl.core.config import ConfigValidator
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.baselines.receding_horizon_milp import RecedingHorizonMILPPolicy

# Load test_protocol without going through qkd_rl.evaluation (whose package
# __init__ pulls in matplotlib, unavailable in a headless run).
import importlib.util
_tp_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp)
build_validation_env_config = _tp.build_validation_env_config
load_validation_profile = _tp.load_validation_profile
resolve_seeds = _tp.resolve_seeds


def _strip_obs(obs, actions):
    return {
        "node_features": obs.node_features,
        "edge_features": obs.edge_features,
        "edge_index": obs.edge_index,
        "edge_ids": obs.edge_ids,
        "node_ids": obs.node_ids,
        "physical_edge_ids": obs.physical_edge_ids,
        "action_candidates": obs.action_candidates,
        "action_masks": obs.action_masks,
        "raw_action_masks": obs.raw_action_masks,
        "milp_actions": actions,
        "t": int(obs.state.t),
    }


def _run_episode(ep_num: int, seed: int, start_seed: int, config: dict, args, out_dir: Path) -> dict | None:
    """Solve and record one validation episode. Returns the demo dict or None.

    Runs with its own env instance so a stuck episode (killed by the caller's
    timeout) can never corrupt the next one. Each episode is independent.
    """
    env = build_env_from_config(config)
    if args.no_switch_cost:
        env.config.setdefault("env", {}).setdefault("switch_cost", {})["enabled"] = False
        print("  [debug] env switch_cost DISABLED", flush=True)
    obs = env.reset(seed=seed, start_seed=start_seed)
    t0 = int(obs.state.t)
    print(f"ep {ep_num}: seed={seed} start_seed={start_seed} reset t={t0} (day {t0/1440:.1f})", flush=True)

    # GLOBAL open-loop: solve the WHOLE segment once with full future
    # information (上帝视角) and execute that single plan without re-solving.
    # ``replan_every_steps=None`` + ``window_steps=steps`` => one solve at reset,
    # plan covers every slot of the episode, so each step's gold link choice
    # comes from the global optimum (best plan found).
    switch_decay = args.switch_decay if args.switch_decay is not None else 0.5
    window_steps = args.window_steps if args.window_steps else args.steps
    pol = RecedingHorizonMILPPolicy(
        config, window_steps=window_steps, max_requests=args.max_requests,
        max_paths_per_request=args.max_paths, max_path_hops=args.max_hops,
        time_limit_s=args.time_limit, replan_every_steps=None,
        switch_decay=switch_decay, single_path=args.single_path,
        final_inventory_weight=0.0,
    )
    # Use the env's request generator so the MILP previews the EXACT same
    # requests that env.step() will generate; passing it to _ensure_init also
    # skips the O(t) RNG fast-forward.
    pol._ensure_init(obs, gen=env.request_generator)

    trajectory = []
    total_served = 0.0
    total_failed = 0.0
    n_solves = 0
    global_srs: list[float] = []
    env_meta = {
        "seed": seed,
        "start_seed": start_seed,
        "episode_start_step": t0,
        "episode_start_day": t0 / 1440.0,
        "episode_steps": args.steps,
        "window_steps": window_steps,
        "window_start_day": config["env"].get("activation_window_start_day", 0),
        "window_end_day": config["env"].get("activation_window_end_day", 0),
        "replan_every": 0,
        "switch_decay": switch_decay,
        "max_paths": args.max_paths,
        "max_hops": args.max_hops,
        "time_limit_s": args.time_limit,
    }
    wall0 = time.perf_counter()
    for step in range(args.steps):
        actions, scores = pol.act(obs)
        # The single global solve at reset gives the (best-found) whole-plan
        # optimum on the modelled demand; report it as the plan's SR. The demo's
        # value is its per-slot gold link choices, which come from this plan.
        if pol._plan_t0 == int(obs.state.t) and pol.last_outcome is not None:
            modeled = pol.last_outcome.modeled_amount
            g_sr = pol.last_outcome.served_amount / max(1e-9, modeled) if modeled > 0 else 0.0
            global_srs.append(g_sr)
            if n_solves == 0:
                print(f"  [t={int(obs.state.t)}] GLOBAL plan SR={g_sr:.4f} "
                      f"(served={pol.last_outcome.served_amount:,.0f}/{modeled:,.0f}) "
                      f"status={pol.last_outcome.status} gap={pol.last_outcome.mip_gap:.4f} "
                      f"solve={pol.last_outcome.solve_time_s:.1f}s", flush=True)
            n_solves += 1
        trajectory.append(_strip_obs(obs, actions))
        obs, reward, term, trunc, info = env.step(actions, scores)
        total_served += info.get("served_keys", 0.0)
        total_failed += info.get("failed_keys", 0.0)
        if term or trunc:
            break
    elapsed = time.perf_counter() - wall0
    arrived = float(env.metrics.arrived_keys)
    executed_sr = total_served / max(1e-9, arrived) if arrived > 0 else 0.0
    plan_sr = sum(global_srs) / len(global_srs) if global_srs else 0.0
    print(f"  -> plan SR={plan_sr:.4f} (exec={executed_sr:.4f} failed={total_failed:,.0f} "
          f"arrived={arrived:,.0f} n_solves={n_solves} ({elapsed:.0f}s)", flush=True)

    return {
        "file": f"episode_{ep_num:04d}.pt",
        "episode": ep_num,
        "seed": seed,
        "start_seed": start_seed,
        "episode_start_step": t0,
        "steps": len(trajectory),
        # success_rate = the GLOBAL plan's SR on modelled demand (single
        # god's-eye solve). This is the target quality gate for the demos;
        # per-slot gold link choices in 'trajectory' are what supervisors use.
        "success_rate": plan_sr,
        "ideal_sr": plan_sr,
        "executed_sr": executed_sr,
        "served_keys": total_served,
        "failed_keys": total_failed,
        "arrived_keys": arrived,
        "env": env_meta,
        "trajectory": trajectory,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate MILP demos + real executed MILP SR for RL guidance/comparison.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of validation episodes/seeds to generate. Default: all profile seeds.")
    parser.add_argument("--append", action="store_true",
                        help="Continue from existing files instead of overwriting.")
    parser.add_argument("--steps", type=int, default=240,
                        help="Segment length in slots (must match RL episode length).")
    parser.add_argument("--window-steps", type=int, default=None,
                        help="MILP horizon in slots. Default = the full segment length "
                             "(GLOBAL god's-eye open-loop): one solve decides every slot's "
                             "links up front. Override with a smaller value only to make it "
                             "tractable (then it is a constricted global window, not a "
                             "per-step receding solver).")
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="HiGHS per-solve time limit in seconds. With switch_decay "
                             "0.5 (env-fair) the switch linearization makes large windows "
                             "slow: keep this generous so HiGHS finds the TRUE global "
                             "optimum instead of a near-zero incumbent on timeout.")
    parser.add_argument("--max-requests", type=int, default=512)
    parser.add_argument("--max-paths", type=int, default=64,
                        help="Path candidates per request. NB small values (8) drop most "
                             "routable demand as 'pathless' and cap the MILP's optimal SR "
                             "at ~0.2; 64 covers ~100% of demand and the optimal SR reaches "
                             "~0.95 on the same served/arrived metric as RL. Keep max-hops=6.")
    parser.add_argument("--max-hops", type=int, default=6)
    parser.add_argument("--switch-decay", type=float, default=None,
                        help="Generation factor for newly activated links in the MILP. "
                             "Default 0.5 (matches the env switch_cost rate_decay_factor, "
                             "so the MILP upper bound is env-fair). The switch linearization "
                             "branches harder on large windows — set --time-limit generously. "
                             "Use 1.0 only for a pure no-switch upper bound (fast but not "
                             "env-fair).")
    parser.add_argument("--min-sr", type=float, default=0.0,
                        help="Drop episodes whose GLOBAL plan SR is below this.")
    parser.add_argument("--no-switch-cost", action="store_true",
                        help="DEBUG: disable the env's switch-cost decay "
                             "(env switch_cost.enabled=False).")
    parser.add_argument("--single-path", action="store_true",
                        help="Model the env's 'one path per request per slot' rule in the MILP.")
    parser.add_argument("--out", type=str, default="outputs/milp_demos")
    args = parser.parse_args()

    profile = load_validation_profile(ROOT / "configs" / "global.yaml")
    n_episodes = args.episodes if args.episodes is not None else len(resolve_seeds(profile))
    seeds = resolve_seeds(profile)[:n_episodes]
    if not seeds:
        raise SystemExit("No validation request seeds configured in configs/global.yaml (global.validation.request_seeds).")

    # Same env config as RL eval / baselines (fixed validation episodes).
    config = build_validation_env_config(
        profile,
        include_baselines=False,
        episode_steps=args.steps,
        start_mode=profile["start_mode"],
    )
    ConfigValidator().validate(config)
    start_seed_base = int(profile.get("start_seed", 0))
    print(f"profile: days {profile['window_start_day']}->{profile['window_end_day']} "
          f"mode={profile['start_mode']} steps={args.steps} seeds={seeds} "
          f"start_seed={start_seed_base}", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.json"

    existing = 0
    if args.append:
        # Number continuation off the files actually on disk, so a skipped/stuck
        # episode can never overwrite an existing demo.
        for f in sorted(out_dir.glob("episode_*.pt")):
            try:
                existing = max(existing, int(f.stem.rsplit("_", 1)[-1]) + 1)
            except ValueError:
                continue
        if existing == 0 and metadata_path.exists():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            existing = int(meta.get("total_episodes", 0))
        print(f"Appending: {existing} episodes exist.")

    results = []
    n_skipped = 0
    n_stuck = 0
    for ep_idx, seed in enumerate(seeds):
        ep_num = existing + ep_idx
        start_seed = start_seed_base + seed
        # Wall-clock budget per episode: the single global solve may use the
        # whole time limit; a stuck episode is skipped after this budget.
        window_steps = args.window_steps if args.window_steps else args.steps
        budget = 120.0 + window_steps * (args.time_limit / max(1, window_steps)) + args.time_limit

        holder: dict = {}
        t = threading.Thread(
            target=lambda: holder.update(result=_run_episode(ep_num, seed, start_seed, config, args, out_dir)),
            daemon=True,
        )
        t.start()
        t.join(timeout=budget)
        if t.is_alive():
            print(f"ep {ep_num}: STUCK after {budget:.0f}s, skipping (add --append to continue numbering)", flush=True)
            n_stuck += 1
            n_skipped += 1
            continue
        demo = holder.get("result")
        if demo is None:
            n_skipped += 1
            continue
        if demo["success_rate"] < args.min_sr:
            print(f"  !! GLOBAL plan SR {demo['success_rate']:.4f} < --min-sr {args.min_sr}, episode skipped", flush=True)
            n_skipped += 1
            continue
        torch.save(demo, out_dir / f"episode_{ep_num:04d}.pt")
        results.append({k: demo[k] for k in
                        ("file", "episode", "seed", "steps", "success_rate", "ideal_sr",
                         "executed_sr", "served_keys", "failed_keys", "arrived_keys", "env")})

    if args.append and metadata_path.exists():
        try:
            old_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            prev_episodes = list(old_meta.get("all_episodes", []))
        except Exception:
            prev_episodes = []
        all_episodes = prev_episodes + results
    else:
        all_episodes = results
    total = len(all_episodes)
    meta = {"total_episodes": total, "steps_per_episode": args.steps,
            "profile": {k: profile.get(k) for k in
                        ("window_start_day", "window_end_day", "episode_steps", "start_mode", "start_seed")},
            "window_steps": args.window_steps, "time_limit_s": args.time_limit,
            "max_requests": args.max_requests, "max_paths": args.max_paths,
            "max_hops": args.max_hops, "replan_every": 0,
            "switch_decay": args.switch_decay, "min_sr": args.min_sr,
            "skipped": n_skipped, "stuck": n_stuck, "all_episodes": all_episodes}
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone. Total global-view MILP demos: {total} (skipped {n_skipped}, stuck {n_stuck}) -> {out_dir}/")
    if all_episodes:
        m = sum(e["success_rate"] for e in all_episodes) / len(all_episodes)
        print(f"MEAN GLOBAL plan SR over {len(all_episodes)} demo(s) = {m:.4f}")


if __name__ == "__main__":
    main()