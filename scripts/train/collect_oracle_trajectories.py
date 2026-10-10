"""先知教师轨迹（2026-10-03，opt-in 新脚本，不改任何已有文件）。

在 run 自己的训练任务（run 的 experiment_config.yaml + overlay，训练窗口，env.reset(seed=s)
与 PPO rollout 相同）上逐回合采集 BC 轨迹，格式与 collect_pg_phased_trajectories.py 相同
（day_XXXX.pkl.gz：_obs_record(obs) + 实际执行的 arcs），可直接
``supervised_train_pg_phased.py --data-dir`` 训练。
--teacher oracle：先跑一遍收集该回合真实到达请求 → 求含切换开销的流体 LP → 把激活解
  四舍五入成双端口动作执行（与 scripts/milp/oracle_exec.py 同一实现）。教师用了未来信息
  （privileged teacher / learning by cheating），学生只看到普通观测。
--teacher expert：同一 env 上的 PG-Phased 专家（对照：分离「教师更强」与「采集任务对齐」）。
种子必须与留出测试种子（1000-1029）不相交。
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qkd_rl.core.config import deep_merge, load_config  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from scripts.milp.fluid_upper_bound import parse_seeds, rollout_collect, solve_bound  # noqa: E402
from scripts.milp.oracle_exec import oracle_actions  # noqa: E402
from scripts.train.collect_pg_phased_trajectories import _obs_record  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--teacher", choices=("oracle", "expert"), default="oracle")
    ap.add_argument("--bucket", type=int, default=1)
    ap.add_argument("--keep-bonus", type=float, default=0.05)
    ap.add_argument("--fill", action="store_true", help="先知动作补满空闲端口（见 oracle_exec.oracle_actions）")
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
    for seed in parse_seeds(a.seeds):
        if 1000 <= seed <= 1029:
            raise SystemExit("seed overlaps held-out test seeds 1000-1029")
        f = out / f"day_{seed:04d}.pkl.gz"
        if f.exists():
            continue
        plan: dict[int, dict[str, float]] = defaultdict(dict)
        info = {}
        if a.teacher == "oracle":
            env = build_env_from_config(cfg)
            obs = env.reset(seed=seed)
            arrivals, _sr, t0 = rollout_collect(env, obs, False)
            r = solve_bound(env, t0, T, arrivals, a.bucket, switch_decay=sd, return_activation=True)
            for eid, tau, val in r.pop("activation", []):
                plan[tau][eid] = val
            info = {"ub_success": r["ub_success"], "lp_status": r["lp_status"]}
        env2 = build_env_from_config(cfg)
        obs = env2.reset(seed=seed)
        expert = None
        if a.teacher == "oracle":
            assert int(obs.state.t) == t0, "episode start mismatch"
            ends = {e.edge_id: (e.src, e.dst) for e in env2.routing.edges}
        else:
            from qkd_rl.baselines.path_greedy import PathScoreGreedy
            from qkd_rl.baselines.serve_probe import ServeProbe
            expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                                     principles=False, router=ServeProbe(env2))
        records = []
        done = False
        while not done:
            if expert is None:
                acts, scores = oracle_actions(obs, plan, t0, ends, a.keep_bonus, a.fill), None
            else:
                acts, scores = expert.act(obs)
            obs_next, _r, term, trunc, _i = env2.step(acts, scores)
            rec = _obs_record(obs)
            rec["arcs"] = list(env2.last_matched_arcs)
            records.append(rec)
            obs = obs_next
            done = term or trunc
        sr = env2.metrics.episode_summary().get("success_rate")
        tmp = f.with_suffix(".tmp")
        with gzip.open(tmp, "wb") as fh:
            pickle.dump(records, fh, protocol=4)
        tmp.rename(f)
        row = {"seed": seed, "teacher": a.teacher, "fill": a.fill, "success_rate": sr, "steps": len(records), **info}
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
