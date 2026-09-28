"""改进项 4c：决策时间 vs 网络规模（2026-09-28）。单线程，只计策略决策（不含 env.step）。
规模：scenario.active_nodes smart 模式每类 N ∈ {10,15,20,25,30}（实际节点数以输出为准）。
用法：python scripts/diag/time_vs_size.py OUT.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from scripts.experiment import resolve_experiment  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.model_zoo import build_model  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402

torch.set_num_threads(1)
out_path = Path(sys.argv[1])
if out_path.exists():
    raise FileExistsError(out_path)
ARMS = {"lite": ["no_attention.yaml", "v8_bias.yaml"], "full": ["v8_bias.yaml"]}
WARM, MEAS = 5, 40
rows = []
for n in (10, 15, 20, 25, 30):
    for arm, extra in list(ARMS.items()) + [("expert", ["no_attention.yaml"])]:
        cfg = resolve_experiment("v8", 42, 1, ["env_full_demand_v2.yaml", "load_x20.yaml"] + extra, "timing")
        cfg["scenario"]["active_nodes"].update(gs_count=n, hap_count=n, sat_count=n)
        env = build_env_from_config(cfg)
        obs = env.reset(seed=1000)
        if arm == "expert":
            pol = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True, principles=False, router=ServeProbe(env))
        else:
            torch.manual_seed(0)
            model = build_model(env.action_resolver.action_space, cfg)
            model.eval()
            pol = MAPPOPolicy(model)
        times, n_phys = [], []
        for k in range(WARM + MEAS):
            t0 = time.perf_counter()
            if arm == "expert":
                actions, scores = pol.act(obs)
            else:
                with torch.no_grad():
                    step = pol.act(obs, deterministic=True)
            dt = time.perf_counter() - t0
            if k >= WARM:
                times.append(dt)
                n_phys.append(len(obs.physical_edge_ids))
            if arm == "expert":
                obs, _r, term, trunc, _i = env.step(actions, scores)
            else:
                obs, _r, term, trunc, _i = env.step(step.actions, expected_matched_edges=step.matched_edges)
            if term or trunc:
                obs = env.reset(seed=1001)
        r = {"arm": arm, "per_type": n, "n_nodes": len(env.scenario.nodes), "n_links": len(env.scenario.edges),
             "mean_active_phys_edges": float(np.mean(n_phys)), "ms_mean": 1e3 * float(np.mean(times)),
             "ms_p95": 1e3 * float(np.percentile(times, 95))}
        rows.append(r)
        print(json.dumps(r), flush=True)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
