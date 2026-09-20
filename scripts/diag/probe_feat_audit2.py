"""特征审计（**修正版**）：物理边行与需求边行**分开**审。

## 上一版的方法缺陷

上一版把 `obs.edge_features` 的**全部行** vstack 在一起。但那个数组里混了
**物理边行**（1978 条）与**需求边行**（~150 条），每个行类型在对方的列段上是
**零填充**。⟹ "后 16 列 95%+ 是 0" 是**我自己造成的假象**，不是特征的问题。

本版用 `num_physical_directed` 切分，分别审。

## 同时：把列名对上（读代码给的行组装顺序）

物理边行 = static_rows + dyn + zeros(demand_dim)
需求边行 = zeros(physical_dim) + demand 列
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


def audit(M, label):
    print(f"\n{'='*92}\n{label}  shape={M.shape}\n{'='*92}")
    ncol = M.shape[1]

    print("\n  ① 逐位相同的列对（**排除全 0/全常数对**，那是零填充不是 bug）：")
    live = [c for c in range(ncol) if M[:, c].std() > 0]
    dup = [(i, j) for ii, i in enumerate(live) for j in live[ii+1:]
           if np.array_equal(M[:, i], M[:, j])]
    if dup:
        for i, j in dup:
            print(f"      ★ 列 {i} ≡ 列 {j}   (值域 [{M[:,i].min():.4g}, {M[:,i].max():.4g}])")
    else:
        print("      （无非平凡重复）")
    print(f"     （全 0 列 {ncol - len(live)} 个，是物理/需求行段的零填充）")

    print("\n  ② 方差为 0 的常数列：")
    const = np.where(M.std(0) == 0)[0]
    if len(const):
        for c in const:
            print(f"      ★ 列 {c}: 恒 = {M[0,c]:.6g}")
    else:
        print("      （无）")

    print("\n  ③ 饱和（>90% 落在 0 或 1）：")
    any_sat = False
    for c in range(ncol):
        col = M[:, c]
        z = float(np.mean(np.abs(col) < 1e-6))
        o = float(np.mean(np.abs(col - 1.0) < 1e-6))
        if z > 0.9 or o > 0.9:
            any_sat = True
            print(f"      列 {c:>3}: {'全0' if z>o else '全1'} {max(z,o):.1%}"
                  f"   值域[{col.min():.4g}, {col.max():.4g}]")
    if not any_sat:
        print("      （无）")


def main():
    prof = _tp.load_validation_profile(str(REPO / "configs" / "global.yaml"))
    cfg = _tp.build_validation_env_config(prof)
    env = build_env_from_config(cfg)
    obs = env.reset(seed=100, start_seed=100)
    exp = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                          principles=False, router=ServeProbe(env))

    NODE, PHYS, DEM = [], [], []
    done, k = False, 0
    npd = 0
    while not done and k < 240:
        nf = np.asarray(obs.node_features, dtype=np.float64)
        ef = np.asarray(obs.edge_features, dtype=np.float64)
        # 用两个 id 列表的长度做切分，不猜字段名
        npd = len(obs.physical_edge_ids)
        if nf.ndim == 2:
            NODE.append(nf)
        if ef.ndim == 2 and npd > 0:
            PHYS.append(ef[:npd])
            if ef.shape[0] > npd:
                DEM.append(ef[npd:])
        acts, scores = exp.act(obs)
        obs, _r, term, trunc, _i = env.step(acts, scores)
        done = term or trunc
        k += 1

    audit(np.vstack(NODE), "节点特征")
    if PHYS:
        audit(np.vstack(PHYS), "物理边行")
    if DEM:
        audit(np.vstack(DEM), "需求边行")

    # 维数信息
    print(f"\n  num_physical_directed = {npd}  ⟹ 需求边行 = {44-npd if False else '见上'}")
    print(f"  物理/需求切分点 = {npd}")


if __name__ == "__main__":
    main()
