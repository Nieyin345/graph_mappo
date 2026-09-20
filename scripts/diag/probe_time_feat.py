"""核实时间特征：std 只有 0.0007 ⟹ 可能恒为常数（真 bug）？

我的审计探针发现节点特征第 15/16 列 std 极小（0.0007 / 0.0004）。
若真是时间特征（sin/cos of minute-of-day），240 步里它**应该**变化很大
（sin 走过约 1/6 个周期）。恒为常数说明有 bug。

本探针直接打印每步的时间值。
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config  # noqa: E402


def main():
    prof = _tp.load_validation_profile(str(REPO / "configs" / "global.yaml"))
    cfg = _tp.build_validation_env_config(prof)
    env = build_env_from_config(cfg)
    obs = env.reset(seed=100, start_seed=100)

    print("=" * 92)
    print("时间特征逐列（前 3 列之后的最右几列 = 时间 sin/cos）")
    print("=" * 92)

    rows = []
    for k in range(0, 12):
        nf = np.asarray(obs.node_features, dtype=np.float64)
        if nf.ndim == 2 and nf.shape[0] > 0:
            rows.append(nf[0].copy())   # 第一行 = 第一个节点
        # 推进
        from qkd_rl.baselines.path_greedy import PathScoreGreedy
        from qkd_rl.baselines.serve_probe import ServeProbe
        exp = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                              principles=False, router=ServeProbe(env))
        acts, scores = exp.act(obs)
        obs, _r, term, trunc, _i = env.step(acts, scores)
        if term or trunc:
            break

    M = np.array(rows)
    print(f"\n  采样 {len(M)} 步 × {M.shape[1]} 维（第一个节点）")
    print(f"\n  {'列':>4}" + "".join(f"{'步'+str(i):>11}" for i in range(len(M))))
    for c in range(M.shape[1]):
        vals = "".join(f"{M[i, c]:>11.6f}" for i in range(len(M)))
        flag = ""
        if M[:, c].std() < 1e-3:
            flag = "   ★ 近乎恒定"
        print(f"  {c:>4}{vals}{flag}")

    print("\n  逐列跨步 std：")
    for c in range(M.shape[1]):
        print(f"    列 {c:>3}: std={M[:, c].std():.6f}  范围[{M[:, c].min():.4f}, {M[:, c].max():.4f}]")

    print(f"\n  环境配置 include_time_features = "
          f"{cfg['features']['node'].get('include_time_features')}")
    print(f"  minute_of_day_sin_cos = "
          f"{cfg['features']['node'].get('time_features', {}).get('minute_of_day_sin_cos')}")
    print(f"  episode_start_mode = {cfg['env'].get('episode_start_mode')}")


if __name__ == "__main__":
    main()
