"""核一件事：请求生成器采样到的 GS 对里，有多少在拓扑上**根本不可达**。

### 为什么这是个真问题

`request.py:258`：`self._pairs = list(combinations(sorted(gs_ids), 2))`
—— **没有可达性过滤**，435 个 GS 对全进采样池。
但服务需要一条**物理路径**（`routing.hop_distance`）。若某对 GS 分处不同
连通分量，那对请求**结构上永远服务不了**，却照样进 `arrived` 分母
⟹ `success_rate` 有一个与策略无关的**天花板**。

### 判据

| # | 判据 | 期望 |
|---|---|---|
| 1 | 全对可达 ⟹ 本问题不存在，探针应报 `0` | 未知，看实测 |
| 2 | 若存在不可达对：它们的占比 × (1) 应能对上"专家缺口"的量级 | 解释力检验 |
| 3 | ★ 正对照：**互换两个已知可达节点的 id** 应不改变可达对数 | 证明算法不是恒报满 |

★ 判据 3 重要：一个"数连通分量"的脚本很容易写成恒返回 1 个分量
（比如忘了初始化 union-find），那样它会报"全可达"而**看起来完全合法**。

用法（服务器上，走克隆）：
    OMP_NUM_THREADS=2 /opt/qkd/venv/bin/python -u /tmp/probe_unreachable_pairs.py
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys
from pathlib import Path

CLONE = Path("/tmp/gm_probe")
sys.path.insert(0, str(CLONE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def components(edges: list[tuple[str, str]], nodes: list[str]) -> dict[str, int]:
    """无向图连通分量（朴素 union-find，路径压缩）。"""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return {n: find(n) for n in nodes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_full_rl.yaml")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location(
        "_tp", CLONE / "qkd_rl" / "evaluation" / "test_protocol.py")
    _tp = importlib.util.module_from_spec(spec)
    sys.modules["_tp"] = _tp
    spec.loader.exec_module(_tp)

    fails: list[str] = []
    print("=" * 78)
    print("核：请求生成器采样的 GS 对里有多少在拓扑上不可达")
    print("=" * 78)

    # 与 factory.build_env_from_config 同一入口构造 scenario
    profile = _tp.load_validation_profile(CLONE / a.config)
    cfg = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=60,
        start_mode=profile["start_mode"])

    from qkd_rl.data.scenario_builder import ScenarioBuilder  # noqa: E402
    mode = cfg["scenario"].get("mode", "small")
    sb = ScenarioBuilder(cfg)
    scenario = sb.build_full() if mode == "full" else sb.build_small()

    gs = sorted(n.node_id for n in scenario.nodes if n.node_type.value == "gs")
    edges = [(e.src, e.dst) for e in scenario.edges]
    print(f"\n场景: mode={mode}  节点 {len(scenario.nodes)}  边 {len(edges)}  "
          f"GS {len(gs)}")

    comp = components(edges, [n.node_id for n in scenario.nodes])
    grp: dict[int, list[str]] = {}
    for g in gs:
        grp.setdefault(comp[g], []).append(g)
    sizes = sorted((len(v) for v in grp.values()), reverse=True)
    print(f"GS 分布在 {len(grp)} 个连通分量里，各分量 GS 数 = {sizes}")

    # ---------------- 判据 1：不可达对 ----------------
    total_pairs = 0
    unreachable: list[tuple[str, str]] = []
    for u, v in itertools.combinations(gs, 2):
        total_pairs += 1
        if comp[u] != comp[v]:
            unreachable.append((u, v))
    frac = len(unreachable) / total_pairs if total_pairs else 0.0
    print(f"\n[1] GS 对总数 {total_pairs}，其中不可达 {len(unreachable)}"
          f"（{frac:.4%}）")
    if unreachable:
        print(f"    不可达对示例（前 8）：{unreachable[:8]}")
        # 哪些 GS 是"孤岛"
        iso = [g for g in gs if sum(1 for g2 in gs if comp[g2] != comp[g]) == len(gs) - len(grp[comp[g]])]
        iso = [g for g in gs if len(grp[comp[g]]) <= 2]
        print(f"    只与 ≤2 个 GS 同分量的节点：{iso}")
        print(f"    ⟹ 这部分请求**结构上不可服务**，却照样进 `arrived` 分母")
        print(f"    ⟹ `success_rate` 天花板 ≈ {1 - frac:.4%}（上界）")
    else:
        print("    全对可达 ⟹ 本问题**不存在**")

    # ---------------- 判据 3：正对照 ----------------
    print("\n[3] 正对照：把两个 GS 的 id 对调，可达对数必须不变")
    if len(grp) < 2:
        print("    ⊘ 只有一个连通分量，无法构造跨分量对调 —— 正对照不适用")
        print("      （★ 这本身就是证据：算法报的是 1 个分量，不是因为代码恒返回 1）")
    else:
        print("    ⊘ 跳过：需要人造图才能构造，下面用合成的三节点图单独测")

    # 合成正对照：2 个分量，必须报出 2
    fake_nodes = ["a", "b", "c", "d"]
    fake_edges = [("a", "b"), ("c", "d")]
    fc = components(fake_edges, fake_nodes)
    n_fake = len({fc[n] for n in fake_nodes})
    print(f"    合成图 a-b / c-d ⟹ 分量数 {n_fake}（期望 2）")
    if n_fake != 2:
        fails.append(f"连通分量算法在合成图上返回 {n_fake} 个分量（期望 2）"
                     f"⟹ 算法恒真，判据 1 的结论不可信")
    else:
        print("    ✓ 算法在合成图上正确区分 2 个分量 ⟹ 判据 1 有判别力")

    # ---------------- 判据 2：解释力 ----------------
    print("\n[2] 解释力量级：不可达对占比 vs 专家在验证侧的缺口")
    print(f"    专家验证侧成功率 ≈ 0.6979 ⟹ 缺口 ≈ 0.3021")
    print(f"    不可达对占比 = {frac:.4%}")
    if frac > 0:
        ratio = frac / 0.3021
        print(f"    占比 / 缺口 = {ratio:.4f} ⟹ "
              f"{'同一量级，值得追' if ratio > 0.2 else '差 5 倍以上，**不足以解释缺口**'}")
    # ★ 单靠这一项解释不了缺口 —— 服务是**部分服务**（routing 逐跳瓶颈），
    #   而且不可达只是"必然失败"的一部分。这里只做量级核对，不下因果结论。

    print("\n" + "=" * 78)
    if fails:
        print(f"✗ {len(fails)} 条不通过：")
        for f in fails:
            print(f"    - {f}")
    else:
        print("✓ 探针自洽（合成正对照通过）")
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
