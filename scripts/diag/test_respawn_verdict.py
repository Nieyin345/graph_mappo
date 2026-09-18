#!/usr/bin/env python
"""变异测试：**导入真的 wake_parse，直接调真的 maybe_respawn**。

为什么不能像初稿那样把公式抄一遍再测：抄一遍测的是**我抄的那个副本**，
不是跑在生产上的那个函数。我第一版"修正"就是这样漏过去的 —— 抄出来一算
62.8 >= 55.98 放行，而我还以为改完会拒绝。**测副本等于没测**。

三处必须隔离（初稿漏了后两处，于是 2 项假失败）：
  1. `STATE` → 临时文件：否则测试自己写 hold/done，把真的自动补起关掉；
  2. `OUT`   → 临时目录：真的 `outputs/ent01_g999_s43_r2` 已存在（那个 run 被
     我停掉了，目录还在），函数会直接走"outputs 已存在"分支**提前返回**，
     后面所有内存判据都不执行 —— 测出来的全是那一句，不是判据；
  3. `subprocess.run` → 桩：**绝对不能让这个测试真的启动训练**。
     初稿没打桩，只是幸好每次都被"outputs 已存在"挡住了。挡住的原因是环境
     偶然状态，不是设计 —— 那种"侥幸没出事"不能留。

用法（服务器上，wake_parse.py 在 /tmp/）：
    cd /tmp && python test_respawn_verdict.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/tmp")
import wake_parse as wp   # noqa: E402

_tmpdir = Path(tempfile.mkdtemp())
wp.STATE = _tmpdir / "state.json"
wp.OUT = _tmpdir / "outputs"          # 空目录 → 不会撞上"已存在"分支
wp.OUT.mkdir()

LAUNCHED: list[str] = []


class _Stub:
    """替掉 subprocess.run，只记录不执行。"""

    @staticmethod
    def run(cmd, check=False, **kw):        # noqa: ANN001, ARG004
        LAUNCHED.append(cmd[-1] if isinstance(cmd, list) else cmd)


wp.subprocess = _Stub                          # type: ignore[assignment]

TOTAL = 125.4
NAMES_OTHER = ["vcoef1_s42", "vcoef1_s43", "vcoef1_s44"]   # 不含 g999

fails: list[str] = []


def check(label: str, cond: bool, detail: str) -> None:
    print(f"  {'✓' if cond else '✗'} {label}")
    print(f"      {detail}")
    if not cond:
        fails.append(label)


def call(avail: float, n_train: int, update: int, total: float = TOTAL) -> str:
    """每次调用前清空 STATE（否则上一次写的 done 会污染下一次）。"""
    wp.STATE.unlink(missing_ok=True)
    LAUNCHED.clear()
    return wp.maybe_respawn(avail, n_train, list(NAMES_OTHER), update, total)


print("=== 0. 前提：真的走到内存判据，而不是被别的分支挡掉 ===")
r0 = call(120.0, 0, 0)
check("干净 STATE + 空 OUT → 不看 hold/done/已存在",
      "被 hold" not in r0 and "已补过" not in r0 and "已存在" not in r0,
      f"返回值: {r0[:70]!r}")
check("余量 120G 很宽裕 → 放行", "已补起" in r0, f"返回值: {r0[:70]!r}")
check("且真的调用了启动（桩记录到 1 次）", len(LAUNCHED) == 1,
      f"记录到 {len(LAUNCHED)} 次")

print()
print("=== 1. 反例：复现 2026-09-18 17:42 的真实读数 ===")
r1 = call(62.8, 3, 4)
print(f"      真函数返回: {r1!r}")
check("真函数**拒绝**", "未补" in r1, "含'未补'即拒绝，含'已补起'即放行")
check("拒绝理由含绝对余量（= 视角 B 给的）", "余" in r1 and "底线" in r1,
      f"返回值: {r1[:80]!r}")
check("拒绝时**没有**启动任何东西", len(LAUNCHED) == 0,
      f"记录到 {len(LAUNCHED)} 次启动")

print()
print("=== 2. 视角 A 单独会放行（证明两个视角都需要）===")
n_ok = 3 * wp.RUN_GROWTH_GB * (wp.TARGET_UPDATES - 4) + wp.RUN_FULL_GB + wp.SAFETY_GB
check("视角 A 本身是放行的", 62.8 >= n_ok,
      f"62.8 >= {n_ok:.2f} → {'放行' if 62.8 >= n_ok else '拒绝'}")
check("所以真函数的拒绝不是视角 A 给的", "未补" in r1 and 62.8 >= n_ok,
      "→ 拒绝只能来自视角 B，这正是修正的要点")

print()
print("=== 3. 边界：机器空闲、只起 1 个 → 必须放行 ===")
r3 = call(120.0, 0, 0)
check("放行（否则判据是永远说不的摆设）", "已补起" in r3, f"返回值: {r3[:70]!r}")
check("放行时确实启动了 1 次", len(LAUNCHED) == 1, f"记录到 {len(LAUNCHED)} 次")

print()
print("=== 4. 单调性：在跑的越多，要求越高 ===")
needs = [call(1e9, k, 4) for k in range(2)]   # 余量极大 → 都放行，只为触发同一分支
print("      这一档靠手算核对（真函数在极大余量下都会放行）：")
vals = [k * wp.RUN_GROWTH_GB * (wp.TARGET_UPDATES - 4) + wp.RUN_FULL_GB + wp.SAFETY_GB
        for k in range(5)]
check("delta_need 随在跑数单调不减",
      all(b >= a for a, b in zip(vals, vals[1:])),
      "  ".join(f"n={k}:{v:.1f}" for k, v in enumerate(vals)))

print()
print("=== 5. 缺 total → 必须**拒绝**，不能静默降级成只看视角 A ===")
wp.STATE.unlink(missing_ok=True)
LAUNCHED.clear()
r5 = wp.maybe_respawn(62.8, 3, list(NAMES_OTHER), 4)   # 不传 total
check("不抛异常", True, f"返回值: {r5[:70]!r}")
check("拒绝（缺一半保护时不许放行）", "未补" in r5, f"返回值: {r5[:70]!r}")
check("且没启动", len(LAUNCHED) == 0, f"记录到 {len(LAUNCHED)} 次")
print("      为什么必须拒绝而不是跳过视角 B：62.8G 这个读数**恰是视角 A 会放行**")
print("      的那一档（62.8 >= 55.98）。静默降级 = 用被证伪的判据放行。")

print()
print("=== 6. hold 优先级：有 hold 时不许启动 ===")
wp.STATE.write_text(json.dumps({"hold": "测试用", "done": True}))
LAUNCHED.clear()
r6 = wp.maybe_respawn(200.0, 0, list(NAMES_OTHER), 0, TOTAL)
check("hold 拦住（哪怕余量 200G）", "被 hold" in r6, f"返回值: {r6[:60]!r}")
check("且没启动", len(LAUNCHED) == 0, f"记录到 {len(LAUNCHED)} 次")
wp.STATE.unlink(missing_ok=True)

print()
if fails:
    print(f"✗ {len(fails)} 项未通过：{fails}")
    raise SystemExit(1)
print("✓ 全部通过 —— 测的是**生产上的那个函数**，不是抄来的副本。")
print("  它在 17:42 的真实读数上拒绝（理由来自视角 B）、在机器空闲时放行，")
print("  拒绝时不会启动任何东西，hold 优先。")
