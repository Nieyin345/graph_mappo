"""天花板在哪：给专家**追加**「为卡住的请求激活通路」的动作，看 SR 能涨多少。

## 为什么用专家测（而不是先训模型）

`PathScoreGreedy`（专家）自己就是分阶段的（`path_greedy.py:432-482`）：
阶段1 若现有存量已能服务某请求就**跳过**（不为它激活）；服务不了才按
2 跳→3 跳**认领整条路**（`_try_claim`，全有或全无）；最后 Step 2 再按
V3 分数把剩余端口**填满**。实测专家 SR = 0.6979，结构上限 0.8645 ⟹ 还有
16.7 个点。

而实测（`.tmp/probe_canon_stock.py`）：
  · 79.32% 的待服务请求，其规范路**几乎每一跳都没货**（平均 0.21/2.18 跳有货）
  · 4614 个样本里，规范路 100% 有货的次数 = **0**
  · `|qkp.positive|` 全程只有 260–655 / 1978 条边有存量
  · 服务缺口 F1（连"整局激活过的边并集"都连不通）= 52.2% 密钥量
  · 挡路的**空跳**里上一步被激活过的 = **0.00%**（v1 配置注释里记的）

假设：**激活预算打在了高费率边上，而需求所在的通路没被激活。**

## 怎么做（★ 关键：只占用专家**留空**的端口）

★ 我第一版是直接**覆写** `dual`，那是**混淆**：专家 Step 2 已经按分数填满了
  端口，覆写会挤掉它的激活，还可能把专家已认领的**整条路打残**（残留半条
  被 `_greedy_match` 照样激活）⟹ K>0 变差会因为错的原因。
  （`confound-can-hide-in-the-section-you-didnt-diff` 的同族：自变量不纯。）

这一版：**只占 IDLE 端口**⟹ 纯追加，不动专家任何已有认领。

## 三态判据

① 基线必须复现 0.6979 附近（专家确定性、无 BLAS）
② `K=0` 与**完全不调用 augment** 必须**逐位相同** ⟹ 附加逻辑本身无副作用
③ K 增大时 SR 上升 ⟹ 方向成立；下降 ⟹ 激活预算不是瓶颈
   （但见上：本版的追加是纯增量，若还降，只能是"多激活反而有害"）
★ 同时报告**实际追加条数/步**（端口被占满时会 < K）—— 只报 K 而不报
  实际生效数是 `gate-must-print-its-inputs` 那一类错。

用法（服务器，-u）：
    python3 -u probe_expert_plus_claim.py --seeds 100-104 --ks 0,5,15,30
"""
import argparse
import importlib.util
import os
import statistics as st
import sys
from collections import Counter, deque
from pathlib import Path

_here = sys.path[0] if sys.path else ""
if _here and _here not in ("", "."):
    sys.path[:] = [p for p in sys.path if p != _here]

REPO = Path(os.environ.get("GM_REPO", "/opt/qkd/graph_mappo"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qkd_rl.env.factory import build_env_from_config              # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy          # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe               # noqa: E402
from qkd_rl.env.action_space import NodeActionSpace               # noqa: E402

IDLE = NodeActionSpace.IDLE


def _tp():
    spec = importlib.util.spec_from_file_location(
        "gm_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def avail_path(routing, obs, src, dst, max_hops):
    """在**本步可用**的边上 BFS 一条 src->dst 的最短路（≤ max_hops）。

    返回 [(u, v, eid), ...]（有向、按路径顺序）或 None。
    ★ 判据用**可用性**而不是存量 —— 目的正是**新激活**这些边去生成存量。
    """
    if src == dst:
        return []
    windows = obs.state.edge_windows
    eid_by_pair = {}
    for e in routing.edges:
        eid_by_pair[(e.src, e.dst)] = e.edge_id
        eid_by_pair[(e.dst, e.src)] = e.edge_id
    parent = {src: None}
    q = deque([(src, 0)])
    while q:
        node, d = q.popleft()
        if d >= max_hops:
            continue
        for nbr, eid in routing.adj.get(node, ()):
            if nbr in parent:
                continue
            w = windows.get(eid)
            if w is None or not bool(w.available[0]):
                continue
            parent[nbr] = (node, eid)
            if nbr == dst:
                steps = []
                cur = dst
                while parent[cur] is not None:
                    prev, e = parent[cur]
                    steps.append((prev, cur, e))
                    cur = prev
                steps.reverse()
                return steps
            q.append((nbr, d + 1))
    return None


def augment(acts, obs, routing, router, k, stats):
    """给 K 条**服务不了**的请求，各追加激活一条**可用**通路。

    ★ 只占 IDLE 端口；任一跳端口被占就换下一条路/下一个请求。
    """
    if k <= 0:
        return
    pending = sorted(
        obs.state.pending_requests,
        key=lambda r: (r.deadline_t - obs.state.t, -(r.amount - r.served_amount)))
    n_add = 0
    for req in pending:
        if n_add >= k:
            break
        rem = max(0.0, req.amount - req.served_amount)
        if rem <= 1.0e-9:
            continue
        if router.has_path(req, rem):
            stats["already_ok"] += 1
            continue                      # 专家阶段1 也会跳过它
        stats["stuck"] += 1
        done = False
        for hops in (2, 3):
            steps = avail_path(routing, obs, req.src_gs, req.dst_gs, hops)
            if not steps:
                continue
            # 全有或全无：所有方向端口都必须空着
            ok = True
            for u, v, _e in steps:
                if acts[u][0] != IDLE or acts[v][1] != IDLE:
                    ok = False
                    break
            if not ok:
                continue
            for u, v, _e in steps:
                acts[u][0] = v
                acts[v][1] = u
            n_add += 1
            stats["added"] += 1
            stats["hops"] += len(steps)
            done = True
            break
        if not done:
            stats["blocked"] += 1
    stats["k_asked"] += k


def run(cfg, seed, start0, steps, k, augment_on=True):
    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start0 + seed)
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    router = expert.router
    stats = Counter()
    A = S = 0.0
    done = False
    n = 0
    while not done and n < steps:
        acts, scores = expert.act(obs)
        if augment_on:
            acts = {u: [acts[u][0], acts[u][1]] for u in acts}
            augment(acts, obs, env.routing, router, k, stats)
            acts = {u: tuple(acts[u]) for u in acts}
        obs, _r, term, trunc, info = env.step(acts, scores)
        n += 1
        A += float(info.get("arrived_keys", 0.0))
        S += float(info.get("served_keys", 0.0))
        done = term or trunc
    return (S / A if A else 0.0), A, S, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-104")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--ks", default="0,5,15,30")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])

    print("=" * 104)
    print(f"专家 + 追加「为卡住请求激活通路」  种子 {seeds[0]}–{seeds[-1]}  "
          f"{args.steps} 步  验证 regime")
    print("=" * 104)

    # ① 基线（完全不调用 augment）
    base = {}
    for s in seeds:
        sr, A, S, _ = run(cfg, s, start0, args.steps, 0, augment_on=False)
        base[s] = sr
        print(f"  基线(不 augment) seed {s}: SR={sr:.6f}  arrived={A:,.0f} "
              f"served={S:,.0f}")
    print(f"  基线 SR 均值 = {st.mean([base[s] for s in seeds]):.6f}  "
          f"（专家锚 0.6979 附近？）")
    print()

    # ② K=0 但调用 augment —— 必须逐位相同
    k0 = {}
    for s in seeds:
        sr, _A, _S, _ = run(cfg, s, start0, args.steps, 0, augment_on=True)
        k0[s] = sr
    nd = sum(1 for s in seeds if abs(base[s] - k0[s]) > 1e-12)
    print(f"  K=0 vs 基线 不同的种子：{nd}/{len(seeds)}  ⟸ 必须为 0")
    if nd:
        print("DECISION=AUGMENT_HAS_SIDE_EFFECT  ★ 附加逻辑本身改了行为 ⟹ 后面不可信")
        return 2
    print("  ✓ K=0 与基线逐位相同 ⟹ 附加逻辑无副作用")
    print()

    # ③ 各 K
    print("--- 各 K（逐种子配对 vs 基线）---")
    print(f"{'K':>5}{'SR 均值':>11}{'Δ vs 基线':>12}{'SD(Δ)':>9}"
          f"{'同向':>7}{'实际追加/步':>12}{'卡住/步':>10}{'端口挡/步':>11}")
    print("-" * 104)
    for k in [int(x) for x in args.ks.split(",")]:
        if k == 0:
            continue
        r = {}
        tot = Counter()
        for s in seeds:
            sr, _A, _S, stt = run(cfg, s, start0, args.steps, k, augment_on=True)
            r[s] = sr
            tot.update(stt)
        d = [r[s] - base[s] for s in seeds]
        n = len(d)
        ns = len(seeds) * args.steps
        print(f"{k:>5}{st.mean([r[s] for s in seeds]):>11.4f}{st.mean(d):>+12.4f}"
              f"{(st.stdev(d) if n > 1 else 0.0):>9.4f}"
              f"{f'{sum(1 for x in d if x > 0)}/{n}':>7}"
              f"{tot['added']/ns:>12.2f}{tot['stuck']/ns:>10.2f}"
              f"{tot['blocked']/ns:>11.2f}")

    print()
    print("=" * 104)
    print("判读")
    print("=" * 104)
    print("  K 增大而 SR 上升 ⟹ 激活**没打中需求通路**是真瓶颈，方向成立")
    print("  K 增大而 SR 不动 ⟹ 追加的激活**不改变**可服务量（通路本就服务不了）")
    print("  K 增大而 SR 下降 ⟹ 多激活**有害**（挤占或噪声）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
