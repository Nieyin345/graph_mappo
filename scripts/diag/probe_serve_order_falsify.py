"""造反证：证明我 patch 进去的排序**真的改变了服务顺序**，然后看 SR 动不动。

## 为什么必须补这一步

`probe_serve_order.py` 实测五种顺序（含反向对照 `long_first`）的 SR
**逐位相同**（Δ=+0.0000, SD=0.0000）。但它的正对照①只证明了
「`order=fifo` 时我的副本与原版逐位相同」—— 而**一个完全忽略 `_KEY` 的
no-op 副本也会通过①**（因为 `key_fifo` 恰好就是原版）。

⟹ ①对"副本忠实"和"`_KEY` 根本没被读"**区分不了**
（`unavailable-must-not-look-like-a-value` 的同族：判据在两种世界下都报绿）。

## 这一步做什么

1. **打印被排序的键序列**：同一个 env 状态下，用 `fifo` 与 `reverse` 两种 key
   各排一次，把 `request_id` 序列打出来 —— **必须不同**。这是"输入真的变了"。
2. **造反证梯子**：用一些**故意破坏性**的 key 跑完整局，看 SR 动不动：
   - `reverse`  —— 把 FIFO 完全倒过来（最激进）
   - `random`   —— 用 request_id 的哈希打乱（与任何启发式无关）
   - `tie_only` —— 只打乱 tiebreak（把 hop_distance 倒过来）
3. 若连 `reverse`/`random` 都逐位相同 ⟹ 补一条**机制性论证**并给出
   **可判伪的预测**：`success_rate` 是**按密钥量的比值**，而
   `serve_now = min(瓶颈跳存量, 剩余)` 把存量**整份**给第一条请求
   ⟹ 顺序只决定**谁**拿到，不决定**总共**拿到多少
   ⟹ 体积型指标**结构上对顺序不变**。

★ 判据：至少要有**一个** key 让 SR 动。若一个都不动，结论只能是
  「在本指标下顺序不变」，**不是**「顺序没用」（比如换成按**请求数**的
  成功率、或看公平性指标，顺序就有影响）。

用法（服务器，-u）：
    python3 -u probe_serve_order_falsify.py --seed 100 --steps 240
"""
import argparse
import importlib.util
import os
import sys
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


# ---- 各种 key ----

def k_fifo(req, routing, t):
    return (req.deadline_t, req.arrival_t,
            routing.hop_distance(req.src_gs, req.dst_gs),
            max(0.0, req.amount - req.served_amount))


def k_reverse(req, routing, t):
    """★ 造反证：把 FIFO 完全倒过来。"""
    return (req.deadline_t, -req.arrival_t,
            -routing.hop_distance(req.src_gs, req.dst_gs),
            -max(0.0, req.amount - req.served_amount))


def k_random(req, routing, t):
    """★ 造反证：与任何启发式无关的伪随机序（稳定、与 request_id 绑定）。"""
    return (hash(req.request_id) & 0xFFFFFF,)


def k_tie_only(req, routing, t):
    """只翻 tiebreak（hop_distance 倒序），主键仍 FIFO。"""
    return (req.deadline_t, req.arrival_t,
            -routing.hop_distance(req.src_gs, req.dst_gs),
            max(0.0, req.amount - req.served_amount))


ORDERS = {"fifo": k_fifo, "reverse": k_reverse,
          "random": k_random, "tie_only": k_tie_only}

_KEY = k_fifo
_ORDER_LOG: list[list[str]] = []
_LOG_ON = False


def serve_v2(self, qkp, routing, t):
    """`RequestQueue.serve` 的逐行复刻，key 由 `_KEY` 决定。"""
    routing.prepare_serve(qkp)
    served, waiting, next_pending = [], [], []
    served_keys = 0.0
    serve_events, from_new_by_edge = [], {}

    ordered = sorted(self.pending, key=lambda r: _KEY(r, routing, t))
    if _LOG_ON and ordered:
        _ORDER_LOG.append([r.request_id for r in ordered])

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


def run(cfg, seed, start0, steps, order, log=False, stop_at=None):
    """跑一局；`stop_at` 用来只跑前 N 步（用于打印排序序列）。"""
    global _KEY, _LOG_ON
    _KEY = ORDERS[order]
    RequestQueue.serve = serve_v2
    _LOG_ON = log
    if log:
        _ORDER_LOG.clear()

    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start0 + seed)
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    A = S = 0.0
    done = False
    k = 0
    while not done and k < steps and (stop_at is None or k < stop_at):
        acts, scores = expert.act(obs)
        obs, _r, term, trunc, info = env.step(acts, scores)
        k += 1
        A += float(info.get("arrived_keys", 0.0))
        S += float(info.get("served_keys", 0.0))
        done = term or trunc
    _LOG_ON = False
    return (S / A if A else 0.0), A, S, [list(x) for x in _ORDER_LOG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])

    print("=" * 100)
    print(f"造反证：排序真的变了吗？（seed {args.seed}，{args.steps} 步）")
    print("=" * 100)

    # ---- 第 1 步：同一个 env 状态下，两种 key 排出的序列必须不同 ----
    print()
    print("--- 第 1 步：排序**输入**真的变了吗（只跑 12 步，记录每次的排序序列）---")
    seqs = {}
    for o in ("fifo", "reverse", "random"):
        _sr, _A, _S, log = run(cfg, args.seed, start0, args.steps, o,
                               log=True, stop_at=12)
        seqs[o] = log
        print(f"  {o:<9} 记录了 {len(log)} 次排序")
        for i, ids in enumerate(log[:3]):
            print(f"      t={i}: {ids}")

    n_serves = min(len(v) for v in seqs.values())
    diff_rev = sum(1 for i in range(n_serves) if seqs["fifo"][i] != seqs["reverse"][i])
    diff_rnd = sum(1 for i in range(n_serves) if seqs["fifo"][i] != seqs["random"][i])
    print()
    print(f"  fifo vs reverse 序列不同的次数：{diff_rev}/{n_serves}")
    print(f"  fifo vs random  序列不同的次数：{diff_rnd}/{n_serves}")
    input_changed = (diff_rev > 0) or (diff_rnd > 0)
    if not input_changed:
        print()
        print("DECISION=INPUT_UNCHANGED  ★★ 排序根本没变 ⟹ 我的 patch 没生效，"
              "「顺序无效」的结论**作废**")
        return 2
    print("  ✓ 排序**输入确实变了** ⟹ 后面看输出动不动才有意义")

    # ---- 第 2 步：全跑，看 SR ----
    print()
    print("--- 第 2 步：造反证梯子（全程 {0} 步）---".format(args.steps))
    print(f"{'order':<12}{'SR':>14}{'Δ vs fifo':>16}{'served':>18}")
    print("-" * 100)
    base = None
    out = {}
    for o in ("fifo", "reverse", "random", "tie_only"):
        sr, A, S, _ = run(cfg, args.seed, start0, args.steps, o)
        out[o] = sr
        if base is None:
            base = sr
            print(f"{o:<12}{sr:>14.6f}{'(基准)':>16}{S:>18,.0f}")
        else:
            print(f"{o:<12}{sr:>14.6f}{sr - base:>+16.6f}{S:>18,.0f}")

    print()
    print("=" * 100)
    print("判读")
    print("=" * 100)
    moved = [o for o in out if o != "fifo" and abs(out[o] - base) > 1e-12]
    if moved:
        print(f"  ✓ 这些顺序**让 SR 动了**：{moved}")
        print("    ⟹ 「顺序无效」被推翻，前面的探针结论要重写")
    else:
        print("  ★ 连 `reverse`（完全倒序）与 `random`（与启发式无关）都**逐位相同**")
        print("    ⟹ 输入变了、输出没变 ⟹ **在本指标下顺序结构上无效**")
        print("    机制：`serve_now = min(瓶颈跳存量, 剩余)` 把瓶颈存量**整份**给")
        print("    排在前面的请求 ⟹ 顺序只决定**谁**拿到，不决定**总共**拿到多少；")
        print("    而 `success_rate` 是**体积比**（served_keys/arrived_keys）")
        print("    ⟹ 顺序不变。")
        print("    ⚠ 这**不等于**「顺序没用」：换按**请求数**的成功率、或看公平性/")
        print("      最差对端，顺序就有影响。只是本项目优化的那个指标看不见它。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
