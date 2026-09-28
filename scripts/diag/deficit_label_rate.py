"""测 deficit / new 标签正例率（专家策略 rollout，训练配置），用于定 pos_weight=(1-p)/p。
用法：python scripts/diag/deficit_label_rate.py load_x20.yaml 11 12   （LOAD_OVERLAY 可为 none）"""
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from scripts.experiment import resolve_experiment  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402
from qkd_rl.rl.algos.rollout_buffer import RolloutBuffer, RolloutStep  # noqa: E402

load = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]]
extra = [load] if load != "none" else []
cfg = resolve_experiment("v8", 42, 1, ["env_full_demand_v2.yaml"] + extra + ["v8_bias.yaml", "aux_deficit.yaml"], "v8_rate")
K = int(cfg["model"]["actor"]["usage_forecast"]["horizon"])
thr = float(cfg["model"]["actor"]["usage_forecast"]["deficit_threshold"])
tot = {"new": [0, 0], "deficit": [0, 0]}
for seed in seeds:
    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed)
    expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True, principles=False, router=ServeProbe(env))
    obs_list = []
    done = False
    while not done:
        obs_list.append(obs)
        actions, scores = expert.act(obs)
        obs, _r, term, trunc, _info = env.step(actions, scores)
        done = term or trunc
    for mode in ("new", "deficit"):
        buf = RolloutBuffer(0.99, 0.95, torch.device("cpu"))
        buf.usage_horizon = K
        buf.usage_label_mode = mode
        steps = [RolloutStep(obs=o, actions={}, log_probs={}, entropies={}, value=torch.zeros(()),
                             reward=0.0, terminated=False, truncated=False) for o in obs_list]
        buf.steps.extend(steps)
        buf.finish_episode(0.0)
        pos = sum(float(s.usage_label.sum()) for s in steps)
        n = sum(int(s.usage_label.numel()) for s in steps)
        tot[mode][0] += pos; tot[mode][1] += n
        print(f"{load} seed={seed} {mode}: p={pos / max(1, n):.4f} (n={n}, steps={len(steps)})", flush=True)
for mode, (pos, n) in tot.items():
    p = pos / max(1, n)
    print(f"TOTAL {load} {mode}: p={p:.4f} (1-p)/p={(1 - p) / max(p, 1e-9):.2f}  thr={thr:g}", flush=True)
