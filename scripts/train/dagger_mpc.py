"""MPC 教师 DAgger 采集（2026-10-03，opt-in 新脚本，不改已有文件）。

在 run 自己的训练任务上（env.reset(seed=s)，与 PPO rollout 相同；种子须与留出 1000-1029 不相交）
逐步：教师 = 在线滚动 LP（scripts/milp/mpc_online.MPCPolicy，只用在线可得信息）给出动作并记录为
标签（_obs_record(obs) + arcs，格式同 collect_pg_phased_trajectories）；以概率 beta 执行教师动作，
否则执行学生（--student 检查点）动作 —— 学生走到的状态也由教师打标签（DAgger）。
可直接 supervised_train_pg_phased.py --data-dir <out> --task-env 训练。
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qkd_rl.core.config import deep_merge, load_config  # noqa: E402
from qkd_rl.env.action_space import NodeActionSpace  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from scripts.milp.fluid_upper_bound import parse_seeds  # noqa: E402
from scripts.milp.mpc_online import MPCPolicy  # noqa: E402
from scripts.train.collect_pg_phased_trajectories import _obs_record  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--student", default=None)
    ap.add_argument("--beta", type=float, default=1.0, help="每步执行教师动作的概率（1 = 纯教师示范）")
    ap.add_argument("--forecast", choices=("perfect", "persist"), default="perfect")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    cfg = load_config([Path(a.run) / "experiment_config.yaml"])
    for name in a.overlay:
        p = Path(name)
        cfg = deep_merge(cfg, load_config([p if p.exists() else ROOT / "configs" / name]))
    T = int(cfg["env"]["episode_steps"])
    scfg = cfg.get("env", {}).get("switch_cost", {})
    sd = float(scfg.get("rate_decay_factor", 0.5)) if scfg.get("enabled", True) else 1.0
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy = None
    if a.student:
        import torch
        from qkd_rl.model_zoo import build_model
        from qkd_rl.rl.algos.checkpoint import load_checkpoint
        from qkd_rl.rl.algos.policy import DETERMINISTIC_RESOLVER_MODES, MAPPOPolicy
        torch.set_num_threads(1)
    idle = NodeActionSpace.IDLE
    for seed in parse_seeds(a.seeds):
        if 1000 <= seed <= 1029:
            raise SystemExit("seed overlaps held-out test seeds 1000-1029")
        f = out / f"day_{seed:05d}.pkl.gz"
        if f.exists():
            continue
        env = build_env_from_config(cfg)
        obs = env.reset(seed=seed)
        if a.student and policy is None:
            model = build_model(env.action_resolver.action_space, cfg)
            model.load_state_dict(load_checkpoint(Path(a.student), "cpu").model_state)
            model.eval()
            policy = MAPPOPolicy(model, "cpu")
            mode = str((cfg.get("action_resolver", {}) or {}).get("mode", "mutual_choice"))
            needs = mode in ("priority_matching", "max_weight_matching")
        teacher = MPCPolicy(env, int(obs.state.t), T, sd, a.horizon, True, forecast=a.forecast)
        rng = random.Random(seed)
        records, n_teacher, mism = [], 0, 0
        done = False
        while not done:
            t_acts = teacher.act(obs)
            arcs = [(u, tx) for u, (tx, _rx) in t_acts.items() if tx != idle]
            rec = _obs_record(obs)
            rec["arcs"] = arcs
            records.append(rec)
            if policy is None or rng.random() < a.beta:
                obs, _r, te, tr, _i = env.step(t_acts, None)
                n_teacher += 1
                if set(map(tuple, env.last_matched_arcs)) != set(arcs):
                    mism += 1
            else:
                import torch
                with torch.no_grad():
                    o = policy.act_batched([obs], deterministic=True, build_scores=needs,
                                           use_edge_arrays=not needs)[0]
                obs, _r, te, tr, _i = env.step(
                    o.actions, o.action_scores, edge_scores=o.edge_scores,
                    expected_matched_edges=None if mode in DETERMINISTIC_RESOLVER_MODES
                    else list(o.matched_edges or []))
            done = te or tr
        sr = env.metrics.episode_summary().get("success_rate")
        tmp = f.with_suffix(".tmp")
        with gzip.open(tmp, "wb") as fh:
            pickle.dump(records, fh, protocol=4)
        tmp.rename(f)
        row = {"seed": seed, "beta": a.beta, "student": a.student, "forecast": a.forecast,
               "success_rate": sr, "steps": len(records), "teacher_steps": n_teacher,
               "arc_mismatch_steps": mism, "solve_s": round(teacher.solve_s, 1)}
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
