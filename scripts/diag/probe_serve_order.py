"""服务顺序探针：`serve()` 的排序键到底改不改总服务量？

背景：F2（27.9% 激活过但此刻通路断 = 时机）与 F3（12.5% 有通路却没服务 =
竞争/期限）合计 40% 是**纯调度**，是 RL 理论上该赢的地方。而
`qkd_rl/env/request.py:177-191` 的排序键是**硬编码**的：

    (deadline_t, arrival_t, hop_distance, remaining)

`deadline_steps` 是常数 ⟹ deadline_t = arrival_t + 常数 ⟹ 前两键同序 ⟹
**EDF ≡ FIFO**（代码注释自己承认了，:181-183）。而
`configs/env_small.yaml:70` 的 `routing.serve_order` 被
`config.py:215-216` 钉死成只接受 `earliest_deadline_first` ⟹
**这个旋钮只有一个合法值**，是个验证过但架空的开关。

★ 我要检验的机理假设（记忆 `serve-order-is-hardcoded-and-degenerate` 写的
「顺序直接决定总服务量」）：`partial_consume_for_request` 是**逐跳等量**
消费（`serve_now = min(hop_levels + [remaining])`，路径每一跳各砍掉同样多），
所以一条需求会同时消耗它路径上所有跳的存量。共享跳上的存量被谁先吃掉，
会影响**另一条需求**能拿到多少 ⟹ 我推不出「顺序无关」，也推不出「顺序有关」。
**推不出来就测。**

    ⟹ 本探针不写第二份实现。做法是**替换排序键**，`serve` 的其余部分原样执行。
       于是「同一个键」必须给出与原始**逐位相同**的服务量 —— 这就是正对照；
       正对照不过是恒真的（改错了也会"过"），所以它**不是**充分条件，
       但正对照**失败**足以证明这份接线不忠实。

跑专家（纯启发式、无模型、无 BLAS）⟹ 秒级、节点无关、可随意重复。

    /opt/qkd/venv/bin/python probe_serve_order.py --steps 240 --seeds 100-114
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
# ★ sys.path[0] 是脚本所在目录（.tmp/，有 547 个 .py 含 types.py）⟹ 必炸。
#   摘掉它，再把仓库根放回来。见记忆 `server-tmp-has-547-py-shadowing-stdlib`。
_here = sys.path[0] if sys.path else ""
if _here and _here not in ("", "."):
    sys.path[:] = [p for p in sys.path if p != _here]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.env import request as request_mod
from qkd_rl.env.request import RequestQueue
from qkd_rl.baselines.path_greedy import PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe

# ──────────────────────────────────────────────────────────────────────────
# 原始排序键：逐字抄自 request.py:177-191（**不许改动**，它就是正对照的锚）
def key_original(self, req, routing):
    remaining = max(0.0, req.amount - req.served_amount)
    return (req.deadline_t, req.arrival_t,
            routing.hop_distance(req.src_gs, req.dst_gs), remaining)


# 真正不同的顺序：**最短路径优先**。
# 依据：逐跳等量消费 ⟹ 服务 X 单位要占用路径上每一跳的 X 存量。
# 短路径占用更少的跳、更少的共享存量 ⟹ 同样的存量能服务更多需求。
# 这是唯一一条有**明确机理**的替代顺序，不是随便挑一个。
def key_shortest_first(self, req, routing):
    remaining = max(0.0, req.amount - req.served_amount)
    return (routing.hop_distance(req.src_gs, req.dst_gs),
            req.deadline_t, remaining, req.arrival_t)


# 小需求优先：把"能整条做完的"先做完，减少半成品对存量的占用。
def key_small_first(self, req, routing):
    remaining = max(0.0, req.amount - req.served_amount)
    return (remaining, req.deadline_t, req.arrival_t)


# 长路径优先（**是我原本的负对照**，实测**逐位相同 0/2 ⟹ 完全 no-op**）。
# 机理预测它该变差，实测不动 ⟹ 我的机理理解错了，而不是"没差别"。
def key_longest_first(self, req, routing):
    remaining = max(0.0, req.amount - req.served_amount)
    return (-routing.hop_distance(req.src_gs, req.dst_gs), req.deadline_t, remaining)


# ★★ 决定性对照：**完全倒序**（最晚期限优先）。
# 这是与原始顺序**最大程度不同**的合法顺序 —— 它把整个队列翻过来。
# 若连它都逐位相同 ⟹ 顺序**证明性地**不影响总服务量，
# 那么「F3 = 排队/优先级」这一半就整条关闭，不用再去搜顺序。
def key_reverse_edf(self, req, routing):
    remaining = max(0.0, req.amount - req.served_amount)
    return (-req.deadline_t, -req.arrival_t,
            -routing.hop_distance(req.src_gs, req.dst_gs), -remaining)


# ★ 另一个方向：**伪随机**顺序（按 request_id 的 md5，跨进程确定）。
# 用来回答「是不是**任何**顺序都动不了」——倒序只是"另一个确定的顺序"，
# 随机顺序能覆盖倒序碰不到的排列。用 md5 不用 `hash()`：后者受
# PYTHONHASHSEED 影响，会让探针本身不可复现。
def key_random(self, req, routing):
    h = hashlib.md5(req.request_id.encode("utf-8")).hexdigest()
    return (h,)


ORDERS = {
    "orig_edf_fifo": key_original,      # 正对照：必须与未打补丁逐位相同
    "shortest_first": key_shortest_first,
    "small_first": key_small_first,
    "longest_first": key_longest_first,
    "reverse_edf": key_reverse_edf,
    "random_md5": key_random,
}

# ★ 在任何打补丁之前抓住真身，作为 restore 的唯一来源。
_ORIG_SERVE = RequestQueue.serve


def install(order: str) -> None:
    """替换 `serve` 里的排序键，其余每一行原样保留。

    `_priority` 是 `serve` 的**闭包内函数**，没有模块级符号可替换。所以只能
    重写这一段源码。做法：`inspect.getsource` 只取 `serve` **这一个函数**的源码
    （不 exec 整个模块 —— 那会重复跑模块级代码），把 `_priority` 的函数体换成
    一行 `return __PROBE_KEY__(...)`，其余（:193 的循环、:194 的消费、
    :205-211 的重排队）一字不动。

    忠实性不靠"我觉得对"——由调用方断言：`orig_edf_fifo` 必须与**未打补丁**
    逐位相同。正对照通过不是充分条件（改错也可能凑巧相同），但**失败足以
    证明接线不忠实**。
    """
    src = textwrap.dedent(inspect.getsource(_ORIG_SERVE))
    # ★ 不猜缩进：定位到"def _priority"的行首，缩进从该行本身取出来。
    #   （第一版写死了 8 空格，而 dedent 后的实际缩进不是它 ⟹ ValueError。
    #    写死缩进等于把「源码格式」当成契约，那是会静默失效的假设。）
    try:
        # ★ 锚串只到 `def _priority(req`，**不含右括号**：真实签名是
        #   `def _priority(req: KeyRequest):`（带类型标注）。第一版写死了
        #   `def _priority(req):` ⟹ substring not found。把「签名长什么样」
        #   当契约就是会静默失效的假设。
        i = src.index("def _priority(req")
        a = src.rindex("\n", 0, i) + 1
        indent = src[a:i]
        j = src.index("for req in sorted(self.pending, key=_priority):")
        b = src.rindex("\n", 0, j) + 1
    except ValueError as exc:
        head = "\n".join(src.splitlines()[:6])
        raise RuntimeError(
            f"源码重写定位失败（{exc}）⟹ 接线坏了，不许静默继续。\n"
            f"dedent 后的前 6 行：\n{head}") from None
    if not indent.strip() == "":
        raise RuntimeError(f"_priority 行首不是缩进：{indent!r} ⟹ 定位错了")
    src = src[:a] + (
        f"{indent}def _priority(req):\n"
        f"{indent}    return __PROBE_KEY__(self, req, routing)\n\n"
    ) + src[b:]
    ns = dict(request_mod.__dict__)
    ns["__PROBE_KEY__"] = ORDERS[order]
    exec(compile(src, "<probe_serve_order>", "exec"), ns)  # noqa: S102
    if "serve" not in ns:
        raise RuntimeError("源码重写没产出 serve ⟹ 接线坏了，不许静默继续")
    RequestQueue.serve = ns["serve"]


def restore() -> None:
    RequestQueue.serve = _ORIG_SERVE


def run_seed(config, profile, seed, steps):
    env = build_env_from_config(config)
    obs = env.reset(seed=seed, start_seed=int(profile.get("start_seed", 0)) + seed)
    expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2),
                             phased=True, principles=False, router=ServeProbe(env))
    served = 0.0
    n = 0
    done = False
    while not done:
        actions, scores = expert.act(obs)
        obs, _r, terminated, truncated, info = env.step(actions, scores)
        served += float(info.get("served_keys", 0.0))
        n += 1
        done = terminated or truncated
    s = env.metrics.episode_summary()
    arrived = float(s.get("arrived_keys", 0.0))
    return {"seed": seed, "sr": served / arrived if arrived else 0.0,
            "served": served, "arrived": arrived, "steps": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "global.yaml"))
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--orders", default=",".join(ORDERS))
    ap.add_argument("--out", default="/tmp/serve_order_probe.json")
    args = ap.parse_args()

    def parse_seeds(spec):
        out = []
        for p in spec.split(","):
            p = p.strip()
            if "-" in p:
                a, b = p.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            elif p:
                out.append(int(p))
        return out

    profile = _tp.load_validation_profile(Path(args.config))
    steps = args.steps or int(profile["episode_steps"]) or 240
    seeds = parse_seeds(args.seeds)
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]

    print(f"profile: window {profile['window_start_day']}-{profile['window_end_day']}, "
          f"steps {steps}, start_mode {profile['start_mode']}, {len(seeds)} seeds")
    print(f"orders: {orders}")
    print()

    # ── 第一遍：**未打补丁**的基线（正对照的锚） ──
    t0 = time.perf_counter()
    baseline = {}
    restore()
    for seed in seeds:
        r = run_seed(config, profile, seed, steps)
        baseline[seed] = r
    base_mean = sum(v["sr"] for v in baseline.values()) / len(baseline)
    print(f"[未打补丁] mean success = {base_mean:.4f}   "
          f"({time.perf_counter()-t0:.1f}s)")

    results = {"_baseline": {str(k): v for k, v in baseline.items()}}
    verdicts = []
    for order in orders:
        restore()
        install(order)
        t1 = time.perf_counter()
        rows = {seed: run_seed(config, profile, seed, steps) for seed in seeds}
        mean = sum(v["sr"] for v in rows.values()) / len(rows)
        # 配对差（同种子）—— 配对不是可选（每种子跨度 0.29–0.88）
        diffs = [rows[s]["sr"] - baseline[s]["sr"] for s in seeds]
        md = sum(diffs) / len(diffs)
        same = sum(1 for d in diffs if d == 0.0)
        ident = all(rows[s]["served"] == baseline[s]["served"] for s in seeds)
        results[order] = {str(k): v for k, v in rows.items()}
        flag = ""
        if order == "orig_edf_fifo":
            flag = "  ★正对照 " + ("逐位相同 ✓" if ident else f"★不等！{same}/{len(seeds)} 个相同")
        verdicts.append((order, mean, md, same, ident))
        print(f"[{order:16s}] mean = {mean:.4f}  Δ配对 = {md:+.4f}  "
              f"逐位相同 {same}/{len(seeds)}{flag}   ({time.perf_counter()-t1:.1f}s)")

    restore()
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print("=" * 96)
    print("判读")
    print("=" * 96)
    ctrl = [v for v in verdicts if v[0] == "orig_edf_fifo"]
    if ctrl and not ctrl[0][4]:
        print("★ 正对照失败：同一个键的改写实现与原始**不逐位相同**")
        print("  ⟹ 这份接线不忠实，下面所有 Δ 都不可信。先修接线。")
        return 2
    if ctrl:
        print("✓ 正对照通过：改写实现与原实现逐位相同 ⟹ 接线忠实")
    else:
        print("⚠ 本次没跑正对照（orders 里没有 orig_edf_fifo）⟹ 无法排除接线漂移")
    print()
    for order, mean, md, same, ident in verdicts:
        if order == "orig_edf_fifo":
            continue
        n_un = len(seeds) - same
        print(f"  {order:16s} Δ配对 = {md:+.4f}  改动生效的种子 {n_un}/{len(seeds)}")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
