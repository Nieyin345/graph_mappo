"""把「Stockholm 孤岛」从**结构事实**换算成**指标效应**。

### 为什么必须换算

`probe_unreachable_pairs.py` 给出「29/435 个 GS 对不可达 = 6.67%」——
那是**对**的占比，前提是均匀采样。而 `success_rate = served/arrived` 的分母
是**密钥量**，采样权重（Zipf / 时区权重）会改变实际占比。
**6.67% 不是指标上的效应量**，拿它直接解释专家缺口是拿错了尺子。

### 数据来源

`RequestHistoryTracker.events`：`(t, pair, kind, amount)` 扁平表，
`kind ∈ {arrived, served, failed}`，`pair = tuple(sorted((src_gs, dst_gs)))`。
—— 这是**唯一**能拿到逐请求归属的入口，比我先前猜的那些属性名靠谱。

★ 交叉核对：`Σ(events 里 kind=='arrived' 的 amount)` 必须等于
  `metrics.arrived_keys`。对不上就说明我读错了列 ⟹ 直接报红。

### 正对照

同时统计一个**正常节点**（Berlin）。若它与 Stockholm 的占比同量级，
说明我的标记法坏了（把正常请求也标了进去），而不是 Stockholm 真的占比高。

用法（服务器上，走克隆）：
    OMP_NUM_THREADS=8 /opt/qkd/venv/bin/python -u /tmp/probe_stockholm_share.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

CLONE = Path("/tmp/gm_probe")
sys.path.insert(0, str(CLONE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ISLAND = "Stockholm"      # 孤立 GS（link_registry 里度 = 0）
CONTROL = "Berlin"        # 正常 GS（正对照）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-102")
    ap.add_argument("--steps", type=int, default=60)
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
    from qkd_rl.baselines.path_greedy import PathScoreGreedy
    from qkd_rl.baselines.serve_probe import ServeProbe

    profile = _tp.load_validation_profile(CLONE / a.config)
    cfg = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=a.steps,
        start_mode=profile["start_mode"])

    fails: list[str] = []
    print("=" * 80)
    print("把 Stockholm 孤岛换算成指标效应（专家，验证 regime）")
    print("=" * 80)

    tot = {"arr": 0.0, "srv": 0.0, "metric_arr": 0.0, "metric_srv": 0.0,
           "n_arr": 0, "n_srv": 0}
    by = {ISLAND: {"arr": 0.0, "n_arr": 0, "srv_hits": 0, "srv_amt": 0.0},
          CONTROL: {"arr": 0.0, "n_arr": 0, "srv_hits": 0, "srv_amt": 0.0}}

    for seed in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=seed,
                        start_seed=int(profile.get("start_seed", 0)) + seed)
        expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2),
                                 phased=True, principles=False,
                                 router=ServeProbe(env))
        for _ in range(a.steps):
            actions, scores = expert.act(obs)
            obs, _r, term, trunc, info = env.step(actions, scores)
            if term or trunc:
                break
        s = env.metrics.episode_summary()
        tot["metric_arr"] += float(s["arrived_keys"])
        tot["metric_srv"] += float(s["served_keys"])

        for _t, pair, kind, amt in env.request_history.events:
            if kind == "arrived":
                tot["arr"] += amt
                tot["n_arr"] += 1
                for name in (ISLAND, CONTROL):
                    if name in pair:
                        by[name]["arr"] += amt
                        by[name]["n_arr"] += 1
            elif kind == "served":
                tot["srv"] += amt
                tot["n_srv"] += 1
                for name in (ISLAND, CONTROL):
                    if name in pair:
                        by[name]["srv_hits"] += 1
                        by[name]["srv_amt"] += amt

    # ---------------- 交叉核对：events 的 arrived == metrics 的 arrived ----------------
    print(f"\n[核对] Σ(events.arrived) = {tot['arr']:,.3f}   "
          f"metrics.arrived_keys = {tot['metric_arr']:,.3f}   "
          f"差 {abs(tot['arr']-tot['metric_arr']):.6f}")
    if abs(tot["arr"] - tot["metric_arr"]) > 1e-6:
        fails.append("events 的到达量与 metrics 的对不上 ⟹ 我读错了列，"
                     "下面所有占比作废")
    else:
        print("       ✓ 两条独立路径给出同一个到达量 ⟹ 归属统计可信")

    r_tot = tot["metric_srv"] / tot["metric_arr"] if tot["metric_arr"] else float("nan")
    print(f"\n全局: arrived={tot['metric_arr']:,.0f}  served={tot['metric_srv']:,.0f}"
          f"  success_rate={r_tot:.4f}   到达请求数={tot['n_arr']:,}"
          f"   服务事件数={tot['n_srv']:,}")

    print(f"\n{'节点':<12}{'到达请求':>10}{'arrived':>16}{'占比':>10}"
          f"{'服务事件':>10}{'服务密钥':>16}")
    print("-" * 80)
    for name in (ISLAND, CONTROL):
        b = by[name]
        share = b["arr"] / tot["arr"] if tot["arr"] else float("nan")
        print(f"{name:<12}{b['n_arr']:>10,}{b['arr']:>16,.0f}{share:>9.2%}"
              f"{b['srv_hits']:>10,}{b['srv_amt']:>16,.0f}")
    print("-" * 80)

    sh_i = by[ISLAND]["arr"] / tot["arr"] if tot["arr"] else float("nan")
    sh_c = by[CONTROL]["arr"] / tot["arr"] if tot["arr"] else float("nan")

    # ---------------- 判据 ----------------
    if by[ISLAND]["n_arr"] == 0:
        fails.append(f"从没见到涉 {ISLAND} 的到达 ⟹ 要么它不在采样池里，"
                     f"要么标记法坏了")
    elif by[ISLAND]["srv_hits"] > 0 or by[ISLAND]["srv_amt"] > 0:
        fails.append(f"涉 {ISLAND} 的请求**被服务过**"
                     f"（事件 {by[ISLAND]['srv_hits']} 条 / "
                     f"{by[ISLAND]['srv_amt']:,.0f} 密钥）⟹ "
                     f"「孤岛」判据被实测**证伪**，必须解释矛盾后才能引用")
    else:
        print(f"\n✓ 涉 {ISLAND} 的请求：{by[ISLAND]['n_arr']} 条到达，"
              f"**0 条被服务** ⟹ 孤岛判据与实测一致")
        oh = 1 - sh_i
        print(f"\n[效应] 涉 {ISLAND} 的到达占比 r_a = {sh_i:.4%}")
        print(f"       ⟹ 这部分**结构上拿不到分** ⟹ `success_rate` 天花板 "
              f"≈ {oh:.4%}")
        print(f"       专家的 {r_tot:.4f} ÷ {oh:.4f} = **{r_tot/oh:.4f}**"
              f"（去掉天花板后的相对水平）")
        print(f"\n[对照] 涉 {CONTROL} 的到达占比 = {sh_c:.4%}，"
              f"服务事件 {by[CONTROL]['srv_hits']:,} 条 —— "
              f"{'正常，标记法有判别力' if by[CONTROL]['srv_hits'] > 0 else '★ 控制组也没被服务过，标记法可疑'}")

    print("\n" + "=" * 80)
    if fails:
        print(f"✗ {len(fails)} 条不通过：")
        for f in fails:
            print(f"    - {f}")
    else:
        print("✓ 换算完成")
    print("=" * 80)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
