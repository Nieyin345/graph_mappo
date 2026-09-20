"""复核：边特征里 `available_now` 那一列是否恒为 1？

## 为什么单列查

上一版审计把 44 列一起看，输出太长且受 ssh 断线影响。
本版只查**一列**，跑得快、结论明确。

## 列的定位（读代码，不猜）

`_build_physical_edge_rows_vectorized`（graph_builder.py:685-795）的列序：
  static_rows（link_type_one_hot ×4 + ...）+ dyn（available_now, rate_now,
  rate_future×6, available_future×6, rate_delta, rate_mean, rate_max,
  last_activated, relay_importance, req_hop×4, qkp_capacity_left,
  on_pending_path）

与其猜列号，**直接按名字找**：`graph_builder` 里 `cols.append` 的顺序
就是 dyn 的顺序。本探针打印**每一列的 min/max/唯一值个数**，
让"哪一列恒为常数"自己现形（不依赖我对列号的推断）。
"""
import importlib.util
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
from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402


def main():
    prof = _tp.load_validation_profile(str(REPO / "configs" / "global.yaml"))
    cfg = _tp.build_validation_env_config(prof)
    env = build_env_from_config(cfg)
    obs = env.reset(seed=100, start_seed=100)
    exp = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                          principles=False, router=ServeProbe(env))

    PHYS, DEM = [], []
    done, k = False, 0
    while not done and k < 120:          # 120 步够了，跑快些
        ef = np.asarray(obs.edge_features, dtype=np.float64)
        npd = len(obs.physical_edge_ids)
        if ef.ndim == 2 and npd > 0:
            PHYS.append(ef[:npd])
            if ef.shape[0] > npd:
                DEM.append(ef[npd:])
        acts, scores = exp.act(obs)
        obs, _r, term, trunc, _i = env.step(acts, scores)
        done = term or trunc
        k += 1

    for label, chunks in (("物理边行", PHYS), ("需求边行", DEM)):
        if not chunks:
            continue
        M = np.vstack(chunks)
        print("=" * 88)
        print(f"{label}  shape={M.shape}  —— 逐列唯一值个数（=1 ⟹ 恒定，死重）")
        print("=" * 88)
        print(f"  {'列':>4}{'唯一值':>8}{'min':>13}{'max':>13}{'std':>11}")
        for c in range(M.shape[1]):
            col = M[:, c]
            nuniq = len(np.unique(col))
            flag = "  ★ 恒定" if nuniq == 1 else ""
            print(f"  {c:>4}{nuniq:>8}{col.min():>13.5g}{col.max():>13.5g}"
                  f"{col.std():>11.4g}{flag}")

        const = [c for c in range(M.shape[1]) if len(np.unique(M[:, c])) == 1]
        print(f"\n  ⟹ 恒定列：{const}")
        for c in const:
            print(f"      列 {c}: 恒 = {M[0, c]:.6g}")


if __name__ == "__main__":
    main()
