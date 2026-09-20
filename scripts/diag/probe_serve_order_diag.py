"""钉死两件事：
  A. 第一个探针（probe_serve_order.py）的五种顺序为什么**全等** —— 是 patch 没生效，
     还是那些 key 真的排不出不同的顺序？（判据：key 被调用的计数 + 序列逐位比对）
  B. `hop_distance` 在待服务请求上的**分布** —— 如果它对所有请求都是同一个值，
     那"按跳数排序"这个方向**根本没有作用空间**，第一个探针的 0 就有了机制解释。

## 为什么这两件必须一起做

造反证（`probe_serve_order_falsify.py`）实测：
    reverse   Δ = −0.002867
    random    Δ = −0.022978      ⟸ 顺序**确实能**改 SR
    tie_only  Δ = +0.000000      ⟸ 但只翻 hop 的 tiebreak 时恰好不动

而第一个探针的 `small_first`（把 `remaining` 提到**第 2 键**）比 `tie_only`
的改动**强得多**，却也报 +0.0000 ⟹ **自相矛盾** ⟹ 第一个探针很可能是
`_KEY` 卡在初值（= `key_fifo`）上，五种顺序其实都在跑 FIFO。

★ 判据（三态）：
  ① 每个 key 的**调用计数**必须 > 0 —— 否则函数压根没被调
  ② 同一个 env 状态下，各 key 排出的 request_id 序列**必须不同**（至少一个）
  ③ 若 A 证明 patch 生效而序列仍相同 ⟹ 那些 key 在**本请求流下**确实等价
     （例如 `hop_distance` 是常数），这时 B 的分布就是解释。

用法（服务器，-u）：
    python3 -u probe_serve_order_diag.py --seed 100 --steps 120
"""
import argparse
import importlib.util
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

_here = sys.path[0] if sys.path else ""
if _here and _here not in ("", "."):
    sys.path[:] = [p for p in sys.path if p != _here]

REPO = Path(os.environ.get("GM_REPO", "/opt/qkd/graph_mappo"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qkd_rl.env.factory import build_env_from_config              # noqa: E402
from qkd_rl.env.request import RequestQueue, ServeResult          # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy          # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe               # noqa: E402


def _tp():
    spec = importlib.util.spec_from_file_location(
        "gm_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- 第一个探针用过的五个 key（逐字复制）+ 造反证用的两个 ----

def k_fifo(req, routing, t):
    return (req.deadline_t, req.arrival_t,
            routing.hop_distance(req.src_gs, req.dst_gs),
            max(0.0, req.amount - req.served_amount))


def k_short_first(req, routing, t):
    return (req.deadline_t,
            routing.hop_distance(req.src_gs, req.dst_gs),
            req.arrival_t,
            max(0.0, req.amount - req.served_amount))


def k_long_first(req, routing, t):
    return (req.deadline_t,
            -routing.hop_distance(req.src_gs, req.dst_gs),
            req.arrival_t,
            max(0.0, req.amount - req.served_amount))


def k_small_first(req, routing, t):
    return (req.deadline_t,
            max(0.0, req.amount - req.served_amount),
            routing.hop_distance(req.src_gs, req.dst_gs),
            req.arrival_t)


def k_cost_aware(req, routing, t):
    hop = routing.hop_distance(req.src_gs, req.dst_gs)
    rem = max(0.0, req.amount - req.served_amount)
    return (req.deadline_t, hop, rem, req.arrival_t)


def k_random(req, routing, t):
    return (hash(req.request_id) & 0xFFFFFF,)


KEYS = {
    "fifo": k_fifo,
    "short_first": k_short_first,
    "long_first": k_long_first,
    "small_first": k_small_first,
    "cost_aware": k_cost_aware,
    "random": k_random,
}

_KEY = k_fifo
_CALLS = Counter()
_SEQ: list[list[str]] = []
_LOG = False
_STOP = None


def serve_diag(self, qkp, routing, t):
    routing.prepare_serve(qkp)
    served, waiting, next_pending = [], [], []
    served_keys = 0.0
    serve_events, from_new_by_edge = [], {}

    def _prio(r):
        _CALLS[_KEY.__name__] += 1
        return _KEY(r, routing, t)

    ordered = sorted(self.pending, key=_prio)
    if _LOG and ordered:
        _SEQ.append([r.request_id for r in ordered])

    for req in ordered:
        served_now, from_new_map = routing.partial_consume_for_request(
            req, qkp, t, attributed=True)
        served_keys += served_now
        if served_now > 0:
            serve_events.append(
                (req.request_id, served_now, sum(from_new_map.values())))
            for edge_id, amount in from_new_map.items():
                from_new_by_edge[edge_id] = from_new_by_edge.get(edge_id, 0.0) + amount
            updated = replace(req, served_amount=req.served_amount + served_now)
            if updated.served_amount >= req.amount - 1.0e-9:
                served.append(updated)
                continue
            req = updated
        if t < req.deadline_t:
            waiting.append(req)
            next_pending.append(req)
        else:
            next_pending.append(req)

    self.pending = next_pending
    return ServeResult(
        served_requests=served, waiting_requests=waiting, failed_requests=[],
        served_keys=served_keys,
        waiting_keys=sum(max(0.0, r.amount - r.served_amount) for r in waiting),
        failed_keys=0.0, serve_events=serve_events,
        from_new_by_edge=from_new_by_edge)


def run(cfg, seed, start0, steps, order, log=False, stop_at=None, watch=None):
    global _KEY, _LOG, _STOP
    _KEY = KEYS[order]
    _LOG = log
    _STOP = stop_at
    RequestQueue.serve = serve_diag
    if log:
        _SEQ.clear()

    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start0 + seed)
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    hop_hist = Counter()
    rem_hist = Counter()
    A = S = 0.0
    done = False
    k = 0
    while not done and k < steps and (stop_at is None or k < stop_at):
        if watch is not None:
            for req in env.requests.pending:
                watch["hop"][env.routing.hop_distance(req.src_gs, req.dst_gs)] += 1
                watch["rem"][round(float(max(0.0, req.amount - req.served_amount)))] += 1
                watch["n"] += 1
        acts, scores = expert.act(obs)
        obs, _r, term, trunc, info = env.step(acts, scores)
        k += 1
        A += float(info.get("arrived_keys", 0.0))
        S += float(info.get("served_keys", 0.0))
        done = term or trunc
    _LOG = False
    return (S / A if A else 0.0), A, S, [list(x) for x in _SEQ], hop_hist, rem_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=240,
        start_mode=profile["start_mode"])

    print("=" * 100)
    print(f"A. 五种 key 到底排不排得出不同的顺序（seed {args.seed}，只跑 12 步采样）")
    print("=" * 100)

    seqs = {}
    for o in ("fifo", "short_first", "long_first", "small_first",
              "cost_aware", "random"):
        _CALLS.clear()
        sr, _A, _S, seq, _h, _r = run(cfg, args.seed, start0, args.steps, o,
                                      log=True, stop_at=12)
        seqs[o] = seq
        print(f"  {o:<13} key 被调用 {_CALLS[KEYS[o].__name__]:>6} 次  "
              f"记录排序 {len(seq)} 次")
        for i, ids in enumerate(seq[:2]):
            print(f"      t={i}: {ids}")

    n = min(len(v) for v in seqs.values())
    print()
    print(f"  以 fifo 为基准，前 {n} 次排序里序列不同的次数：")
    for o in seqs:
        if o == "fifo":
            continue
        d = sum(1 for i in range(n) if seqs["fifo"][i] != seqs[o][i])
        flag = "★ 不同" if d else "  相同"
        print(f"    {o:<13} {d:>3}/{n}  {flag}")

    # ---- B. hop_distance / remaining 分布 ----
    print()
    print("=" * 100)
    print("B. 待服务请求的 hop_distance 与 remaining 分布（采样窗口）")
    print("=" * 100)
    watch = {"hop": Counter(), "rem": Counter(), "n": 0}
    run(cfg, args.seed, start0, args.steps, "fifo", watch=watch, stop_at=120)
    tot = watch["n"] or 1
    print(f"  样本数 {tot:,}（每步遍历 pending）")
    print()
    print("  hop_distance 分布（GS-GS 静态跳数）：")
    for h, c in sorted(watch["hop"].items()):
        print(f"    {h:>6} 跳 : {c:>8,}  ({c/tot:>6.2%})")
    print()
    print(f"  不同 hop 取值个数 = {len(watch['hop'])}")
    print(f"  不同 remaining 取值个数 = {len(watch['rem'])}")
    print()

    print("=" * 100)
    print("判读")
    print("=" * 100)
    hop_uniq = len(watch["hop"])
    if hop_uniq <= 1:
        print(f"  ★ hop_distance **只有 {hop_uniq} 个取值** ⟹ 「按跳数排序」在本场景下")
        print("    **没有作用空间** —— 所有请求代价相同，cost-aware 的所有变体都退化")
        print("    成 FIFO。这解释了第一个探针里 short/long/cost_aware 全等于 fifo。")
    else:
        print(f"  hop_distance 有 {hop_uniq} 个取值 ⟹ 有作用空间，")
        print("    那第一个探针全等就**必须**用 bug 解释（看上面 A 的序列比对）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
