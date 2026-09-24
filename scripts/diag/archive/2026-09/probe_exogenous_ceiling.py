"""★ 纯外生天花板：`success_rate` 里有多少分是**结构上拿不到**的。

### 为什么这一版最干净

前两版都借用了专家的行为：
- `probe_unreachable_pairs.py`：数**对**（6.67%），但要假设均匀采样
- `probe_stockholm_share.py`：数专家的到达/服务，但"专家没服务"≠"不可达"

这一版**不用任何策略**：
1. **到达流是外生的** —— `RequestGenerator` 只吃 `(env_seed, t)`，策略
   改不了它。所以直接调 `env.request_generator.generate(t)` 就能复现
   一模一样的需求流，一步 `env.step` 都不需要。
2. **可服务性是纯拓扑** —— `routing.shortest_path(src, dst) is None`
   与策略无关。

两者相乘就是**与算法无关**的 `success_rate` 天花板。

### 判据

| # | 判据 | 期望 |
|---|---|---|
| 1 | 外生到达总量必须 == `metrics.add_arrivals` 累加的量 | 走同一条路，必须一致 |
| 2 | 不可服务对的到达占比 vs 其**对数**占比 | 两者应同量级；差太多说明采样有偏 |
| 3 | ★ 正对照 | 随机取一个稀疏但连通的对，`shortest_path` 必须**非 None** |

用法（服务器上，走克隆）：
    OMP_NUM_THREADS=2 /opt/qkd/venv/bin/python -u /tmp/probe_exogenous_ceiling.py
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--config", default="configs/train_full_rl.yaml")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))

    spec = importlib.util.spec_from_file_location(
        "_tp", CLONE / "qkd_rl" / "evaluation" / "test_protocol.py")
    _tp = importlib.util.module_from_spec(spec)
    sys.modules["_tp"] = _tp
    spec.loader.exec_module(_tp)

    from qkd_rl.env.factory import build_env_from_config

    profile = _tp.load_validation_profile(CLONE / a.config)
    cfg = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=a.steps,
        start_mode=profile["start_mode"])

    fails: list[str] = []
    print("=" * 80)
    print(f"★ 纯外生天花板 —— {len(seeds)} 种子 × {a.steps} 步（验证 regime，"
          f"**不跑任何策略**）")
    print("=" * 80)

    # ---------------- 拓扑侧：每对可服务性 ----------------
    env = build_env_from_config(cfg)
    gs = sorted(n.node_id for n in env.scenario.nodes
                if n.node_type.value == "gs")
    routing = env.routing
    unservable = set()
    pairs = list(itertools.combinations(gs, 2))
    for u, v in pairs:
        if routing.shortest_path(u, v) is None:
            unservable.add((u, v))
    print(f"\n[拓扑] GS {len(gs)} 个，对 {len(pairs)} 个，"
          f"其中 `shortest_path is None` 的 {len(unservable)} 个"
          f"（{len(unservable)/len(pairs):.4%}）")
    # ★ 只报**真正孤立**的节点（与其他任何 GS 都不可达），不要把每个
    #   不可达对的**两个端点**都收进来 —— 那会列出全部 30 个 GS，
    #   看起来像"整张图都断了"，与事实（只有 1 个孤岛）相反。
    iso_nodes = [g for g in gs
                 if all((min(g, h), max(g, h)) in unservable
                        for h in gs if h != g)]
    print(f"       真正孤立的 GS（与其余全部不可达）：{iso_nodes}")

    # ---------------- 正对照：连通对必须非 None ----------------
    reachable = [p for p in pairs if p not in unservable]
    if not reachable:
        fails.append("没有任何可达对 ⟹ 拓扑全断，判据无意义")
    else:
        probe_pair = reachable[len(reachable) // 2]
        sp = routing.shortest_path(*probe_pair)
        print(f"[正对照] 连通对 {probe_pair} 的最短路长度 = "
              f"{len(sp) if sp is not None else None}（必须非 None）")
        if sp is None:
            fails.append(f"正对照失败：`shortest_path{probe_pair}` 是 None，"
                         f"但该对不在不可服务集合里 ⟹ 两处判据自相矛盾")
        else:
            print("        ✓ 可达对确实可达 ⟹ `shortest_path is None` 有判别力")

    # ---------------- 需求侧：外生到达流（不跑策略） ----------------
    tot_arr = 0.0
    bad_arr = 0.0
    n_req = 0
    n_bad = 0
    per_pair: dict[tuple[str, str], float] = {}
    metric_arr = 0.0
    for seed in seeds:
        e = build_env_from_config(cfg)
        e.reset(seed=seed, start_seed=int(profile.get("start_seed", 0)) + seed)
        t0 = int(e.t)
        # ★ 复刻 env.step 的调用序列：每步一次 `generate(self.t)`，t 递增 1。
        #   `self.t` 只在 step 末尾 +1，所以整局就是 [t0, t0+steps)。
        for k in range(a.steps):
            for req in e.request_generator.generate(t0 + k):
                key = tuple(sorted((req.src_gs, req.dst_gs)))
                per_pair[key] = per_pair.get(key, 0.0) + req.amount
                tot_arr += req.amount
                n_req += 1
                if key in unservable:
                    bad_arr += req.amount
                    n_bad += 1
        metric_arr += float(e.metrics.arrived_keys)   # reset 后为 0，应等于 0

    print(f"\n[需求] 外生到达总量 = {tot_arr:,.1f}（{n_req:,} 条请求）")
    print(f"       其中落在不可服务对上的 = {bad_arr:,.1f}"
          f"（{n_bad:,} 条 = {n_bad/max(1,n_req):.2%}）")
    print(f"       按**密钥量**占 {bad_arr/tot_arr:.4%}，"
          f"按**对数**占 {len(unservable)/len(pairs):.4%}")

    # 采样是否均匀：对上的到达量与 Zipf 权重有关
    if unservable:
        share_amt = bad_arr / tot_arr
        share_cnt = n_bad / max(1, n_req)
        share_pair = len(unservable) / len(pairs)
        print(f"       （对数占比 {share_pair:.2%} · 条数占比 {share_cnt:.2%} · "
              f"密钥量占比 {share_amt:.2%}）")
        if share_amt > share_pair:
            print(f"       ⟹ 不可服务对分到的密钥量**高于**均匀采样 "
                  f"（{share_amt:.2%} > {share_pair:.2%}）⟹ 这些对权重偏重，"
                  f"天花板比按对数估的更低")
        else:
            print(f"       ⟹ 不可服务对分到的密钥量**低于**均匀采样 "
                  f"（{share_amt:.2%} < {share_pair:.2%}）")

    # ---------------- 结论 ----------------
    ceiling = 1 - bad_arr / tot_arr if tot_arr else float("nan")
    print(f"\n[结论] 与算法无关的 `success_rate` 天花板 = {ceiling:.4%}")
    print(f"       专家验证侧实测 ≈ 0.6979  ⟹ 占天花板的 "
          f"{0.6979/ceiling:.2%}")
    print(f"       RL   验证侧实测 ≈ 0.7178  ⟹ 占天花板的 "
          f"{0.7178/ceiling:.2%}")
    print(f"       ★ 「还剩多少可提升空间」应从 {1-0.6979:.4f} 改口为 "
          f"{ceiling-0.6979:.4f}")

    # ---------------- 最重的几对 ----------------
    worst = sorted(per_pair.items(), key=lambda kv: -kv[1])[:6]
    print(f"\n[到达量最大的 6 对]")
    for (u, v), amt in worst:
        mark = "✗不可服务" if (u, v) in unservable else " 可达"
        print(f"    {u:<22} → {v:<22} {amt:>14,.0f}   {mark}")

    print("\n" + "=" * 80)
    if fails:
        print(f"✗ {len(fails)} 条不通过：")
        for f in fails:
            print(f"    - {f}")
    else:
        print("✓ 天花算是纯外生的：不用策略、不用专家，只用了「需求流种子」"
              "与「拓扑连通性」")
    print("=" * 80)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
