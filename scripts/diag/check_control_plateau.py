#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照臂（λ=0.95）在 u30 之后**还涨不涨**？—— 用本地镜像的 u30→u50 续跑回答。

### 为什么问这个
`gae90` 的判据只在 u25/u30 窗口过线（见 `check_gae90_window.py`）。
原计划让 gae90 与对照**成对**续跑到 u50，看 u25/u30 的正号是稳定平台还是涨落。
那条链死在节点上，**配对的 gae90 续跑没有本地副本**。

但本地**有**对照臂自己的 u30→u50 续跑（`server_results/outputs/ent01_s{42,43,44}_u30to50`）。
它单独算不了 Δ(u35..u50)，但能回答一个**前提问题**：

> 训练到 u30 之后，对照臂的成功率**还动不动**？

- 若 u35–u50 与 u25/u30 **统计上不可分** ⟹ 后半段是平台 ⟹
  「多训就会更高」这个前提很弱，u50 实验更可能得到"平台"。
- 若 u35–u50 **显著更高** ⟹ 存在"多训就涨"的系统漂移，
  那么任何晚期窗口的正号都要先扣掉它。

### ★★ 轮号规则（本项目已踩错三次，这里按实测写死）
**eval 的轮号 = 它前面那个非 eval 行的 `update` 值。**

- `eval_validation` 行**没有 `update` 键**（记忆 `eval-update-number-not-from-position`）。
- 续跑臂的 `update` 值是 **31…50**，**不是 1…20** ⟹ 训练器计数器**续着编**
  （`mappo_trainer.py:1319 target_updates = self.update_count + num_updates`），
  **不归零**。
- ⟹ 用"累计行数"当轮号，会把续跑臂的 u35–u50 误标成 u5–u20。
  我第一版就是这么错的，症状是"合并后轮号只到 u30"。

### 必须声明的 confound
u35 是续跑后的第一次 eval，而续跑是**新进程**。`load_checkpoint`
（`qkd_rl/rl/algos/mappo_trainer.py:1410`）**会恢复 Adam 矩**，所以"丢动量"
这个嫌疑已被代码排除；但**新进程的 RNG/环境种子是否与直接长跑逐位一致，
没有办法在本数据上验证**（u1 指纹法在这里用不了——续跑段没有 u1）。
⟹ **单边界的差异无法与"续跑伪影"分离**。因此本脚本只主张
"**统计上不可分**"，不主张"确实回落了"。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEEDS = ("42", "43", "44")


def read_run(path: Path):
    """→ {轮号: mean_success_rate}。轮号 = 前面那个非 eval 行的 `update` 值。"""
    out, last = {}, None
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if "update" in o:
                last = int(o["update"])
            elif "eval_validation" in o and last is not None:
                if last % 5 != 0:
                    print("  !! %s 的 eval 落在 update=%d（非 5 的倍数）⟹ 轮号规则存疑"
                          % (path.parent.name, last))
                out[last] = o["eval_validation"]["mean_success_rate"]
    return out


def t_two_sided_p(t, df):
    """双侧 p：数值积分 t 分布尾部（不查表、不手抄）。"""
    if df <= 0:
        raise ValueError("df 必须 > 0")
    nu = float(df)
    const = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                     - 0.5 * math.log(nu * math.pi))

    def f(x):
        return const * (1.0 + x * x / nu) ** (-(nu + 1) / 2)

    n, h, s, a = 20000, 1.0 / 20000, 0.0, abs(t)
    for i in range(n + 1):
        u = min(i * h, 1.0 - 1e-12)
        w = 1.0 if i in (0, n) else (4.0 if i % 2 else 2.0)
        s += w * f(a + u / (1.0 - u)) / (1.0 - u) ** 2
    return 2.0 * s * h / 3.0


def t_crit(df, p=0.05):
    lo, hi = 0.0, 500.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=None, help="outputs 目录（含 <arm>/metrics.jsonl）")
    ap.add_argument("--tail", default="u30to50", help="续跑臂的后缀")
    ap.add_argument("--prefix", default="ent01_s", help="臂名前缀")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    out_root = Path(args.outputs) if args.outputs else (repo / "server_results" / "outputs")

    full = {}
    for s in SEEDS:
        par = read_run(out_root / ("%s%s" % (args.prefix, s)) / "metrics.jsonl")
        con = read_run(out_root / ("%s%s_%s" % (args.prefix, s, args.tail)) / "metrics.jsonl")
        if par is None or con is None:
            print("  ! s%s 缺一侧（父 %s / 续 %s）" % (s, par is not None, con is not None))
            continue
        merged = dict(par)
        dup = sorted(set(par) & set(con))
        if dup:
            print("  ! s%s 父臂与续跑**轮号重叠** %s ⟹ 合并有歧义，跳过该种子" % (s, dup))
            continue
        merged.update(con)
        full[s] = merged

    if not full:
        print("!! 无可用轨迹（检查 --outputs / --tail / --prefix）")
        return 1

    us = sorted(set.intersection(*[set(d) for d in full.values()]))
    print("对照臂（λ=0.95）合并后共同轮号：%s" % us)
    print("  来源：父臂 %s<seed> ｜ 续跑 %s<seed>_%s" % (args.prefix, args.prefix, args.tail))
    print("\n  %-6s%s%12s" % ("u", "".join("%10s" % ("s" + s) for s in sorted(full)),
                              "均值"))
    for u in us:
        vs = [full[s][u] for s in sorted(full)]
        print("  u%-5d" % u + "".join("%10.4f" % v for v in vs)
              + "%12.4f" % statistics.mean(vs))

    # ── 窗口配对比较：u25/u30  vs  u35–u50 ──
    pre = [u for u in us if 25 <= u <= 30]
    post = [u for u in us if u > 30]
    if not pre or not post:
        print("\n!! 窗口不全（pre=%s post=%s）" % (pre, post))
        return 2

    print("\n" + "=" * 92)
    print("窗口配对比较（**每条种子各自在窗口内求均**，再对种子配对，df=n-1）")
    print("=" * 92)
    print("  u25/u30 窗口=%s ｜ u30 之后窗口=%s" % (pre, post))
    print("  %-6s%12s%12s%12s" % ("种子", "u25/u30", "u%d–u%d" % (post[0], post[-1]), "Δ(后−前)"))
    ds = []
    for s in sorted(full):
        a = statistics.mean([full[s][u] for u in pre])
        b = statistics.mean([full[s][u] for u in post])
        ds.append(b - a)
        print("  s%-5s%12.4f%12.4f%+12.4f" % (s, a, b, b - a))
    n = len(ds)
    df = n - 1
    m = statistics.mean(ds)
    sd = statistics.stdev(ds) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else float("inf")
    crit = t_crit(df) if df > 0 else float("inf")
    print("\n  n=%d ｜ df=%d ｜ Δ = %+.4f ｜ SD = %.4f ｜ t = %+.2f ｜ p = %.4f ｜ 临界 %.3f"
          % (n, df, m, sd, t, t_two_sided_p(t, df) if df > 0 else float("nan"), crit))
    print("  ⟹ **%s**" % ("过线（后半段确实变了）" if abs(t) > crit
                          else "未过线 ⟹ 后半段与 u25/u30 **统计上不可分**（平台）"))
    print("\n  ⚠ confound 声明：u35 是**续跑后**的第一次 eval，续跑是新进程。")
    print("    `load_checkpoint` 会恢复 Adam 矩（mappo_trainer.py:1410），故『丢动量』")
    print("    已被代码排除；但新进程的 RNG/环境种子是否与直接长跑逐位一致**无法")
    print("    在本数据上验证**（续跑段没有 u1 可做指纹）⟹ 单边界的差异不能与")
    print("    『续跑伪影』分离。所以本脚本只主张**统计上不可分**，不主张『确实回落了』。")
    print("    ⚠ 且这是**对照单臂**：Δ(u35..u50) 需要配对的 gae90 续跑才能算。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
