"""v2 决策探针：`on_pending_path` 对**被卡住的**请求是否真的静默？

## 假设（可否证）

`_compute_on_pending_path` 的 BFS 只在 **`qkp.positive`（有存量的边）子图**上找路，
找不到就 `continue` ⟹ **那条请求一条边都不标**。

⟹ 断言：**存在大量 pending 请求，它们在**全物理图**上可达，但在**有存量子图**上不可达**
  （= 被卡住、需要生成密钥才能服务），而 `on_pending_path` 对它们**全 0**。

若是 ⟹ v1 给的是"已经能服务的路径"的成员资格，
**对被卡的请求零信息** ⟹ 这解释了 v1 只有 +0.0154。
v2 应改成：在**全物理图**上找路，标出"这条请求**需要**哪些边有货"。

## 方法（不重写逻辑，避免重复实现漂移）

包装 `_compute_on_pending_path`：
  · 原样调一次拿**并集** `out`
  · 再对**每条请求单独**调一次（用一个只实现 `get_pending()` 的 shim 队列）
    ⟹ 拿到**逐请求**的标记，用的是**同一份真代码**

全图可达性用**现成的** `env.routing.shortest_path(src, dst) is not None`。

## 判据

① 正对照：任何**被标记**（逐请求 out 有 1）的请求，必须**全图可达**。
   违反 ⟹ 我的接线错了，停止解读。
② 自证：逐请求标记的并集 == 原样调用的并集（逐位）。
   不等 ⟹ shim 路径不等价，停止解读。
"""
import importlib.util
import os
import sys
from collections import deque
from pathlib import Path

_here = sys.path[0] if sys.path else ""
if _here and _here not in ("", "."):
    sys.path[:] = [p for p in sys.path if p != _here]

REPO = Path(os.environ.get("GM_REPO", "/opt/qkd/graph_mappo"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse                                                      # noqa: E402
from qkd_rl.env.factory import build_env_from_config                  # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy              # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe                   # noqa: E402


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


class ShimQueue:
    """只实现 `_compute_on_pending_path` 用到的那一个方法。"""

    def __init__(self, req):
        self._r = [req]

    def get_pending(self):
        return self._r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])

    # ---- 逐种子采集 ----
    tot_pend = 0          # pending 请求总数（按步累加）
    tot_marked = 0        # 其中 on_pending_path 逐请求**有标记**的
    tot_full = 0          # 其中**全物理图可达**的
    silent_fixable = 0    # ★ 全图可达、但有存量子图不可达（被卡住、特征静默）
    unreach = 0           # 全物理图都不可达（真·孤岛，谁都救不了）
    union_mismatch = 0    # 自证②失败次数
    ctrl_violation = 0    # 正对照①失败次数
    nsteps = 0
    per_seed = []

    for s in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=s, start_seed=start0 + s)
        builder = env.graph_builder      # env.py:42 已确认
        orig = builder._compute_on_pending_path

        s_pend = s_marked = s_full = s_silent = s_unreach = 0
        expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        n = 0
        done = False
        while not done and n < args.steps:
            real_pending = list(env.requests.get_pending())
            union = orig(env.requests)
            per_req = [orig(ShimQueue(r)) for r in real_pending]

            # 自证②：逐请求并集 == 原样调用
            merged = None
            for m in per_req:
                merged = m.copy() if merged is None else (merged + m)
            if merged is None:
                merged = union * 0.0
            if not (merged.shape == union.shape and
                    bool(((merged > 0) == (union > 0)).all())):
                union_mismatch += 1

            for r, m in zip(real_pending, per_req):
                marked = bool((m > 0).any())
                full_ok = env.routing.shortest_path(r.src_gs, r.dst_gs) is not None
                s_pend += 1
                s_marked += int(marked)
                s_full += int(full_ok)
                if marked and not full_ok:
                    ctrl_violation += 1        # 正对照①违反
                if full_ok and not marked:
                    s_silent += 1              # ★ 被卡住且特征静默
                if not full_ok:
                    s_unreach += 1

            acts, scores = expert.act(obs)
            obs, _r, term, trunc, _info = env.step(acts, scores)
            n += 1
            done = term or trunc

        nsteps += n
        tot_pend += s_pend
        tot_marked += s_marked
        tot_full += s_full
        silent_fixable += s_silent
        unreach += s_unreach
        per_seed.append((s, s_pend, s_marked, s_silent))

    print("=" * 100)
    print(f"on_pending_path 对「被卡住的请求」是否静默  种子 {seeds[0]}–{seeds[-1]}"
          f"  {nsteps} 步（专家策略下）")
    print("=" * 100)
    print(f"{'种子':>5}{'pending 累计':>14}{'有标记':>10}{'有标记占比':>12}"
          f"{'全图可达':>10}{'★被卡且静默':>14}{'静默占可达':>12}")
    print("-" * 100)
    for s, sp, sm, ss in per_seed:
        reach = sp - (0)          # 下面用总量算；这里逐种子只报前三列
        print(f"{s:>5}{sp:>14}{sm:>10}{(sm/sp if sp else 0):>11.2%}"
              f"{'—':>10}{ss:>14}"
              f"{(ss/max(sp - sm, 1)):>11.2%}")
    print("-" * 100)
    print(f"{'合计':>5}{tot_pend:>14}{tot_marked:>10}"
          f"{(tot_marked/tot_pend if tot_pend else 0):>11.2%}"
          f"{tot_full:>10}{silent_fixable:>14}"
          f"{(silent_fixable/max(tot_pend - tot_marked, 1)):>11.2%}")
    print("-" * 100)
    print()
    print(f"  全物理图不可达的请求（真孤岛，谁都救不了）：{unreach} "
          f"({unreach/max(tot_pend,1):.2%})")
    print()
    print("=" * 100)
    print("判据")
    print("=" * 100)
    print(f"  ① 正对照：被标记 ⟹ 必须全图可达。违反 {ctrl_violation} 次  "
          f"{'✓' if ctrl_violation == 0 else '✗ 接线错，停止解读'}")
    print(f"  ② 自证：逐请求并集 == 原样调用。不等 {union_mismatch} 次  "
          f"{'✓' if union_mismatch == 0 else '✗ shim 不等价，停止解读'}")
    print()
    if ctrl_violation or union_mismatch:
        print("DECISION=ABORT  接线/自证没过 ⟹ 不给结论")
        return 2
    frac = silent_fixable / max(tot_pend, 1)
    print(f"★ 被卡住**且**特征静默的请求占比 = {frac:.2%}（{silent_fixable}/{tot_pend}）")
    if frac > 0.10:
        print("  ⟹ 假设成立：on_pending_path 对相当一部分可救请求是**全 0** 的。")
        print("     v2 方向：把 BFS 从「有存量子图」换到「全物理图」，")
        print("     标出**这条请求需要哪些边有货**（而非哪些边现在有货）。")
        print("DECISION=V2_FULL_GRAPH")
    else:
        print("  ⟹ 假设**不成立**：静默比例很小 ⟹ on_pending_path 基本覆盖了可服务路径，")
        print("     缺口在别处 ⟹ 不要按这个方向改。")
        print("DECISION=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
