"""判决探针：卡住的请求，**补上「能激活但空」的跳**之后能服务吗？（只读）

### 它要回答的最后一个开放问题

前三个探针把方向收敛到了这里：

| 探针 | 实测 | 排除掉的 |
|---|---|---|
| `probe_v1_aim` | 标出的边 100% 有存量、挡路空跳 **0%** 被标 | v1 的实现 |
| `probe_bottleneck` | 89.4% 在挂请求**真的**没有有存量通路 | "其实有路，只是没走" |
| `probe_block_anatomy` | 挡路空跳只有 **0.76%** 在观测物理边表里 | 加特征指出它们 |
| `probe_reach` | 卡住请求 12.7% 在 VIS 上有路、17.8% 在 VIS∪POS、**78.1% 在全集上** | 可见性不是唯一约束 |

剩下的唯一可能：**这些跳是"能激活的"（`available` 为真），只是我们没给它生成密钥。**

区分两个【动作空间】——代码里它们是**两个不同的量**：

- **观测的物理边列表**（`obs.physical_edge_ids`，~167 条）—— 模型**看得见**、
  能打分的边。这是 `graph_builder` 从速率数据里挑出来的。
- **`edge_windows[e].available[0]`**（掩码用的那个）—— 边此刻是否**可用**，
  决定它**能不能被激活**（`masks.py` 的 `mask_unavailable_edges`）。

`probe_reach` 量到每一步「观测物理边数」和「物理可用边数」**中位数都是 167**
—— 但那是**中位数相等**，不证明**同一批边**。本探针把这两个集合**逐边**比。

### 判决逻辑

对每一步、每条卡住的请求：

1. 它的**几何最短路**上，每一跳分成三类：
   - `已有存量`（在 `qkp.positive` 里）—— 不需要生成
   - `★ 可激活但空`（在掩码允许的边里、`level == 0`）—— **"补上就能解锁"的跳**
   - `掩码挡死` —— 生成了也没用，改不了

   ★ 分类**只按掩码的真正判据**（`available`），**不猜 `available` 与
   `rate >= min_link_rate` 的关系** —— `rate_provider.py:586 _available_from`
   说明当 `availability_source != "rate"` 时，`available=True` **不蕴含**
   速率达标（LOS 可能才是判据）。速率达标与否只在 `可激活但空` 内部再分一层。
2. **反事实**：假设所有「可激活但空」的跳**都有密钥**，
   这条请求能不能被服务？（在 `有存量 ∪ 可激活空` 上 BFS）
3. 若解锁了，那条路每跳能生成多少（`rate × slot_seconds`）够不够它的需求量

判据：

- 若「可激活但空」占比高、且反事实解锁比例高 ⟹ **找到了真瓶颈**：
  动作空间里有这些边、模型也看得见，但**没有任何东西告诉它"提前给这几跳生成"**
  ⟹ 该做的是**奖励/机制**（把生成对准需求量），不是加观测特征。
- 若「可激活但空」几乎不存在（都被掩码挡死）⟹ 动作空间本身就不含解锁所需
  的那几跳 ⟹ 该放宽掩码 / 改可见性规则。
- 若反事实解锁比例很低 ⟹ 补上也没用，卡住是**需求本身在这个时刻物理不可达**
  ⟹ 唯一的路是**更早地预生成**（跨多步规划），那是另一类机制改动。

用法（服务器上）：
    /opt/qkd/venv/bin/python -u /tmp/probe_unlock.py --seeds 100-104
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config          # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy      # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe           # noqa: E402


def med(xs):
    v = [x for x in xs if x == x]
    return st.median(v) if v else float("nan")


def build_adj(edge_ids, eid2pair):
    adj: dict[str, list[str]] = {}
    for e in edge_ids:
        p = eid2pair.get(e)
        if p is None:
            continue
        u, v = p
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj


def reachable(adj, src, dst):
    if src not in adj or dst not in adj:
        return False
    seen = {src}
    q = deque([src])
    while q:
        n = q.popleft()
        if n == dst:
            return True
        for m in adj.get(n, ()):
            if m not in seen:
                seen.add(m)
                q.append(m)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-104")
    ap.add_argument("--max-steps", type=int, default=0,
                    help=">0 时截断步数（冒烟测试用）")
    ap.add_argument("--config", default="configs/train_full_rl.yaml")
    ap.add_argument("--out", default="/tmp/probe_unlock.json")
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))

    profile = _tp.load_validation_profile(ROOT / a.config)
    steps = int(profile["episode_steps"])
    if a.max_steps > 0:
        steps = min(steps, a.max_steps)
    start_seed = int(profile.get("start_seed", 0))
    cfg = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])

    # ★ `min_link_rate` **不在** `features.action` 下 —— 它在
    #   `rate_provider.rate.min_link_rate`（`factory.py:64` 传给 MaskBuilder）。
    #   按 `features.action` 读会**静默取 0**，把「被掩码砍掉」并进
    #   「可激活但空」—— 正是本项目反复踩的那个静默回退。所以：
    #   **不读配置，读对象**（`env.mask_builder` 上的真实取值），
    #   再断言它与配置路径一致；不一致就退出，绝不静默。
    _probe_env = build_env_from_config(cfg)
    mb = _probe_env.mask_builder
    min_rate = float(mb.min_link_rate)
    mask_avail = bool(mb._mask_available)
    mask_rate = bool(mb._mask_rate)
    cfg_rate = float(cfg["rate_provider"]["rate"]["min_link_rate"])
    if abs(min_rate - cfg_rate) > 1e-12:
        print(f"★★ MaskBuilder.min_link_rate={min_rate} 与配置 "
              f"{cfg_rate} 不符 ⟹ 输出不可信，退出")
        return 2
    if not mask_avail or not mask_rate:
        print(f"★★ 掩码没开（available={mask_avail}, rate={mask_rate}）"
              "⟹ 本探针的分类前提不成立，退出")
        return 2

    print(f"解锁潜力探针（只读）—— {len(seeds)} 种子 × {steps} 步")
    print(f"  窗口 {profile['window_start_day']}-{profile['window_end_day']}")
    print(f"  min_link_rate = {min_rate}（读自 env.mask_builder，与配置一致）")
    print(f"  mask_unavailable_edges = {mask_avail}，mask_below_min_rate = {mask_rate}")
    print()

    # —— 计数 ——
    n_blk = 0                       # 卡住条次
    hop_cls = {"pos": 0, "live_empty": 0, "dead_unavail": 0}
    hop_rate = {"okrate": 0, "lowrate": 0}   # 只对 live_empty 细分
    n_fillable_req = 0              # 卡住请求里，几何路上至少有一跳「可激活但空」
    n_unlocked = 0                  # 反事实：全填上之后能服务
    unlocked_min_gen, unlocked_rem = [], []
    every_valid_empty = []          # 每条卡住请求几何路上「可激活但空」的跳数
    n_blk_zero_fillable = 0
    # 观测集 vs 可用集 的重叠
    both_cnt = vis_cnt = av_cnt = 0
    # 反事实时那条路的每跳生成量（geometric 单位：rate*slot）
    hop_gen_amounts = []
    # ★ 对质用：同一条请求在三种邻接定义下分别可达吗
    n_pos_only = n_avail_only = n_vis_only = 0
    pos_cnt_hist, avail_cnt_hist, vempty_cnt_hist = [], [], []
    witness = None
    n_empty_geo = 0

    for seed in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=seed, start_seed=start_seed + seed)
        expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2),
                                 phased=True, principles=False,
                                 router=ServeProbe(env))
        slot_sec = float(getattr(env.scenario, "slot_seconds", 1.0) or 1.0)
        eid2pair = getattr(env.routing, "_eid2pair", None)
        if eid2pair is None:
            eid2pair = {e.edge_id: (e.src, e.dst) for e in env.routing.edges}
            env.routing._eid2pair = eid2pair
        all_ids = [e.edge_id for e in env.routing.edges]

        for _ in range(steps):
            obs = env._build_observation()
            qkp = env.qkp
            state = env._state_and_masks(env.t)[0]
            windows = state.edge_windows
            pos = set(qkp.positive)
            vis = set(obs.physical_edge_ids)

            # 全边表的 available / rate 一次取（向量化，避免 1978 次物化）
            blocks = getattr(windows, "blocks", None)
            valid_empty_all: set[str] = set()
            avail_all: set[str] = set()
            rate_of: dict[str, float] = {}
            if blocks is not None:
                link_ids = env.rate_provider.edge_link_ids(all_ids)
                rate_col = np.asarray(blocks[0])[0][link_ids]
                avail_col = np.asarray(blocks[1])[0][link_ids]
                for i, eid in enumerate(all_ids):
                    rate_of[eid] = float(rate_col[i])
                    if bool(avail_col[i]):
                        avail_all.add(eid)
                        if eid not in pos:
                            valid_empty_all.add(eid)
                            hop_gen_amounts.append(float(rate_col[i]) * slot_sec)
            else:                                     # 退化：逐边物化
                for eid in all_ids:
                    w = windows[eid]
                    rate_of[eid] = float(w.rates[0])
                    if bool(w.available[0]):
                        avail_all.add(eid)
                        if eid not in pos:
                            valid_empty_all.add(eid)
                            hop_gen_amounts.append(float(w.rates[0]) * slot_sec)

            both_cnt += len(vis & avail_all)
            vis_cnt += len(vis)
            av_cnt += len(avail_all)
            pos_cnt_hist.append(len(pos))
            avail_cnt_hist.append(len(avail_all))
            vempty_cnt_hist.append(len(valid_empty_all))

            # ★ 三种邻接定义，同一条请求上当场对质
            adj_pos = build_adj(pos, eid2pair)
            adj_avail = build_adj(avail_all, eid2pair)
            adj_vis = build_adj(vis, eid2pair)
            adj_filled = build_adj(pos | valid_empty_all, eid2pair)

            def classify(e):
                """对掩码忠实的三分类：不猜 available 与 rate 的关系。"""
                if e in pos:
                    return "pos"
                if e not in avail_all:
                    return "dead_unavail"
                return "live_empty"

            for req in env.requests.get_pending():
                if env.routing._find_positive_path(req, qkp) is not None:
                    continue
                n_blk += 1
                geo = env.routing.shortest_path(req.src_gs, req.dst_gs) or []
                if not geo:
                    n_empty_geo += 1
                n_fill = 0
                for e in geo:
                    c = classify(e)
                    hop_cls[c] += 1
                    if c == "live_empty":
                        n_fill += 1
                        if rate_of.get(e, 0.0) >= min_rate:
                            hop_rate["okrate"] += 1
                        else:
                            hop_rate["lowrate"] += 1
                every_valid_empty.append(n_fill)
                if n_fill == 0:
                    n_blk_zero_fillable += 1
                else:
                    n_fillable_req += 1
                r_pos = reachable(adj_pos, req.src_gs, req.dst_gs)
                r_av = reachable(adj_avail, req.src_gs, req.dst_gs)
                r_vis = reachable(adj_vis, req.src_gs, req.dst_gs)
                n_pos_only += bool(r_pos)
                n_avail_only += bool(r_av)
                n_vis_only += bool(r_vis)
                if reachable(adj_filled, req.src_gs, req.dst_gs):
                    n_unlocked += 1
                    rem = max(0.0, req.amount - req.served_amount)
                    unlocked_rem.append(rem)
                    lv = []
                    for e in geo:
                        if e in pos:
                            lv.append(qkp.get_level(e))
                        else:
                            lv.append(rate_of.get(e, 0.0) * slot_sec)
                    if lv:
                        unlocked_min_gen.append(min(lv))
                    if witness is None and not r_vis:
                        witness = {
                            "seed": seed, "t": int(env.t),
                            "src": str(req.src_gs), "dst": str(req.dst_gs),
                            "geo_len": len(geo),
                            "|pos|": len(pos), "|avail|": len(avail_all),
                            "|vis|": len(vis),
                            "vis==avail": vis == avail_all,
                            "geo_hops_in_avail": sum(1 for e in geo if e in avail_all),
                        }

            acts, scores = expert.act(obs)
            env.step(acts, scores)
        print(f"  seed {seed} 完成")

    def f(x, n):
        return x / n if n else float("nan")

    print()
    print("=" * 74)
    print("卡住请求的几何最短路：每一跳属于哪一类？")
    print("=" * 74)
    tot_hops = sum(hop_cls.values())
    for k, label in (("pos", "已有存量（不用生成）"),
                     ("live_empty", "★ 可激活但空（补上就解锁）"),
                     ("dead_unavail", "掩码挡死（生成了也没用）")):
        print(f"  {label:<34} {hop_cls[k]:>8}  ({f(hop_cls[k], tot_hops):.4f})")
    le = hop_cls["live_empty"] or 1
    print(f"      其中速率达标   {hop_rate['okrate']:>7} "
          f"({hop_rate['okrate']/le:.4f})")
    print(f"           速率不达标 {hop_rate['lowrate']:>7} "
          f"({hop_rate['lowrate']/le:.4f})")
    print()
    print(f"  卡住请求总条次                      {n_blk}")
    print(f"    几何路上至少有一跳『可激活但空』   {n_fillable_req}  "
          f"({f(n_fillable_req, n_blk):.4f})")
    print(f"    一跳都补不了的                     {n_blk_zero_fillable}  "
          f"({f(n_blk_zero_fillable, n_blk):.4f})")
    print(f"    每条几何路上『可激活但空』跳数 p50 {med(every_valid_empty):.0f}")
    print()
    print("=" * 74)
    print("反事实：把『可激活但空』的跳全填上，能服务吗？")
    print("=" * 74)
    print(f"  解锁的卡住请求                      {n_unlocked}  "
          f"({f(n_unlocked, n_blk):.4f})")
    print(f"  解锁后那条路的瓶颈跳生成量 p50      {med(unlocked_min_gen):.4f}")
    print(f"  解锁请求的需求量 p50                {med(unlocked_rem):.4f}")
    print(f"  全表『可激活但空』跳的生成量 p50    {med(hop_gen_amounts):.4f}"
          "   （rate × slot_seconds）")
    print()
    print("=" * 74)
    print("【观测集】与【可用集】是不是同一批边？")
    print("=" * 74)
    print(f"  每步观测物理边量(累计) {vis_cnt}，可用边量(累计) {av_cnt}，"
          f"交集 {both_cnt}")
    print(f"  观测 ⊂ 可用 的比例    {f(both_cnt, vis_cnt):.4f}")
    print(f"  每步 |pos| p50 {med(pos_cnt_hist):.0f}，"
          f"|avail| p50 {med(avail_cnt_hist):.0f}，"
          f"|avail\\pos| p50 {med(vempty_cnt_hist):.0f}")
    print()
    print("=" * 74)
    print("★ 对质：同一条卡住请求，三种邻接定义下分别可达吗？")
    print("=" * 74)
    print(f"  A  只在【有存量子图】上可达            {n_pos_only:>7}  "
          f"({f(n_pos_only, n_blk):.4f})   ← probe_reach 的『VIS∪POS』近亲")
    print(f"  B  只在【掩码可用子图】上可达          {n_avail_only:>7}  "
          f"({f(n_avail_only, n_blk):.4f})   ← 与 n_unlocked 应恒等")
    print(f"  C  只在【观测物理边子图】上可达        {n_vis_only:>7}  "
          f"({f(n_vis_only, n_blk):.4f})   ← probe_reach 的『VIS』")
    print(f"  几何路为空的请求                       {n_empty_geo}")
    if witness:
        print(f"  样例（既可达又 VIS 不可达）: {json.dumps(witness, ensure_ascii=False)}")
    print()
    if abs(n_pos_only - n_unlocked) > 0.001 * max(1, n_blk) and n_pos_only > n_unlocked:
        print("  ★★ 矛盾：只在【有存量】上可达的比在【可用】上可达的**还多**")
        print("     ⟹ 有存量 ⊄ 可用 —— 掩码把这批有存量的边挡在动作空间外。")
        print()

    un = f(n_unlocked, n_blk)
    ve = f(hop_cls["live_empty"], tot_hops)
    de = f(hop_cls["dead_unavail"], tot_hops)
    if un >= 0.25:
        verdict = (
            f"★★ **卡住不是物理不可达**：{un:.1%} 的卡住请求在**掩码可用子图**上"
            "明明有路。而它们的**几何最短路**上『可激活但空』的跳只有 "
            f"{ve:.1%} —— 因为 `routing.shortest_path` 走的是**注册拓扑**"
            f"（{len(all_ids)} 条边），其中 {de:.1%} 的跳**当前不可用**。"
            "⟹ **几何路与可行动的路是两张不同的图**，"
            "「关键跳 0 次碰」是**结构必然**，不是策略挑错。"
            "⟹ 修法：把『该给谁生成』对准**可用子图上的路**，"
            "不是注册拓扑的最短路。")
    elif ve < 0.10:
        verdict = (
            f"**几何路上没有可生成的跳**：只有 {ve:.1%} 是『可激活但空』，"
            f"其余 {de:.1%} 被掩码挡死；且只有 {un:.1%} 的卡住请求"
            "在可用子图上有路 ⟹ 这批请求本步确实不可达。")
    else:
        verdict = (
            f"**中间情形**：{ve:.1%} 的几何跳可生成、{un:.1%} 的卡住请求"
            "在可用子图上有路 ⟹ 两条路都要看，先确认可用子图上的路是否够好。")
    print("  ⟹ " + verdict)

    Path(a.out).write_text(json.dumps({
        "seeds": seeds, "steps": steps, "min_link_rate": min_rate,
        "n_blk": n_blk, "hop_class": hop_cls, "tot_hops": tot_hops,
        "hop_rate_within_live": hop_rate,
        "frac_live_empty": ve, "frac_dead_unavail": de,
        "n_fillable_req": n_fillable_req, "n_blk_zero_fillable": n_blk_zero_fillable,
        "n_unlocked": n_unlocked, "frac_unlocked": un,
        "median_unlocked_min_gen": med(unlocked_min_gen),
        "median_unlocked_rem": med(unlocked_rem),
        "median_hop_gen_amount": med(hop_gen_amounts),
        "median_live_empty_per_path": med(every_valid_empty),
        "vis_cnt": vis_cnt, "av_cnt": av_cnt, "both_cnt": both_cnt,
        "frac_vis_in_avail": f(both_cnt, vis_cnt),
        "n_pos_only": n_pos_only, "n_avail_only": n_avail_only,
        "n_vis_only": n_vis_only, "n_empty_geo": n_empty_geo,
        "frac_avail_only": f(n_avail_only, n_blk),
        "frac_vis_only": f(n_vis_only, n_blk),
        "median_pos_size": med(pos_cnt_hist),
        "median_avail_size": med(avail_cnt_hist),
        "median_avail_minus_pos": med(vempty_cnt_hist),
        "witness": witness,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
