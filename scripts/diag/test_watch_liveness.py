# -*- coding: utf-8 -*-
"""`watch_liveness.py` 的**回归测试 + 造反证**（纯本地，不连服务器，不读 /proc）。

配对约定：`scripts/diag/test_launch_gate.py` 是对 `launch_g2.py` 的门的已知答案
断言；本文件是对 `watch_liveness.py` 的判据的已知答案断言。

## 为什么要造反证而不是"跑一遍没崩"

记忆 `never-run-code-path-hides-bugs`：长期为假的判据 = 一条从未执行的路径，
打开它之前要**假设里面有 bug**，并且修完要**造反证**（原版 vs 修复版同进程对打）。
所以本文件第 6 节把**修复前的实现**写出来当场跑，逐格证明新旧**给出不同答案**。
没有那一节，前面所有 "旧：…" 都只是散文。

## 它锁住的四个缺陷（2026-09-21 审计）

| # | 缺陷 | 症状 | 为什么危险 |
|---|------|------|-----------|
| 1 | 轮号 = 数**行数** | `metrics.jsonl` 每轮不止一行（`eval_validation` 行没有 `update` 键） | 续跑臂读少 ⟹ `u >= target` 完成判据偏慢/偏快，**静默** |
| 2 | `frozen` **只写不读** | 明细把 list 填好，结论段另用 `any(delta == 0)` 重算 | 打印说"已恢复"，退出码说"冻结" —— 自相矛盾 |
| 3 | `delta is None` **两边都不落** | 既不进 suspects 也不置 rc | **进程死了**与**一切正常**同码退出（exit 0） |
| 4 | 空集短路 | `runs and any(...)` ⟹ rc=0 | 打印「结论：全部健康（0 条 run 在算）」 |

★ 根因不是"某处算错了"，是**判据被内联在打印循环里、而 `rc` 又在结论段另算一遍**
⟹ 两者各自漂移。所以修复的主要动作是**把判据拎成纯函数 `classify()`**，
让"判据"这件事只有一个实现（同族：`cross-check-must-compare-same-population`）。

## 元教训（写测试时踩的，记下来免得再踩）

本文件自己迭代了四版才全绿，**四次的失败都在测试脚本里，不在被测代码**：
按字符串 split 分段（锚点写在注释里 ⟹ 切空）、锚点带 `\\n` 去匹配已按行拆开的
文本（永远 -1）、断言作用域开得太大（把 `def classify(` 自己数进"调用数"）。
**判据要能被自己检验**：所以下面每条"某某不得出现"都配了一条**正向**断言，
防止它在被切空的段上**空过**（vacuous pass）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import re
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")     # Windows 控制台是 GBK
except Exception:
    pass

HERE = Path(__file__).resolve().parent           # scripts/diag/
SRC = HERE / "watch_liveness.py"                 # ★ 相对定位，不硬编码中文路径
REPO = HERE.parent.parent

_ok = [True]


def check(label, got, want):
    good = (got == want)
    _ok[0] = _ok[0] and good
    print("  %s %-52s got=%r want=%r" % ("✓" if good else "✗", label, got, want))
    return good


# ============================================================ 夹具
tmp = Path(tempfile.mkdtemp(prefix="wl_"))
(tmp / "outputs").mkdir()


def write_run(name, updates, eval_every=5):
    """造一个 metrics.jsonl。`updates` 是轮号序列（续跑臂不从 1 开始）。"""
    d = tmp / "outputs" / name
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for u in updates:
        out.append(json.dumps({"update": u, "mean_success_rate": 0.8}))
        if u % eval_every == 0:
            out.append(json.dumps({"eval_validation": {"success": 0.7}}))
    (d / "metrics.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")
    return (d / "metrics.jsonl")


# 续跑臂的真实形态：update 从 6 编到 25（20 个训练行 + 4 个 eval 行 = 24 行）
P_CONT = write_run("cont_s42", range(6, 26))
# 从头跑的臂：30 训练 + 6 eval = 36 行
write_run("fresh_s42", range(1, 31))
(tmp / "outputs" / "empty_s42").mkdir()
(tmp / "outputs" / "empty_s42" / "metrics.jsonl").write_text("", encoding="utf-8")

# ★ 必须走**环境变量**。第一版写 `mod.ROOT = str(tmp)` —— 模块顶层的
#   `ROOT = os.environ.get("QKD_ROOT", ...)` 在 exec_module 时会把它覆盖掉
#   ⟹ 四条轮号断言全 -1（指向不存在的 /opt/qkd）。被测文件自己留了这条通路，
#   测试就该走它。
spec = importlib.util.spec_from_file_location("wl", SRC)
mod = importlib.util.module_from_spec(spec)
os.environ["QKD_ROOT"] = str(tmp)
spec.loader.exec_module(mod)
if mod.ROOT != str(tmp):
    raise SystemExit("QKD_ROOT 没生效：%r —— 后面的轮号断言会全部假 ✗" % mod.ROOT)

print("=" * 74)
print("watch_liveness.py 回归测试 ｜ 被测文件 %s" % SRC)
print("=" * 74)

# ============================================================ 1. 轮号
print()
print("1. 轮号 = 末个 `\"update\": N` 的值，**不是行数**（缺陷 1）")
check("续跑臂 6..25（24 行，真值 25）", mod.rounds_and_age("cont_s42")[0], 25)
check("从头跑 1..30（36 行，真值 30）", mod.rounds_and_age("fresh_s42")[0], 30)
check("不存在的臂 ⟹ -1", mod.rounds_and_age("nope_s42")[0], -1)
check("空文件 ⟹ 0（与 -1 区分：读得出但空 / 读不出）",
      mod.rounds_and_age("empty_s42")[0], 0)
check("年龄是数不是 None（文件在）",
      isinstance(mod.rounds_and_age("fresh_s42")[1], float), True)

# 半行（写到一半被读到）不得污染轮号 —— jsonl 的常见故障
(tmp / "outputs" / "torn_s42").mkdir()
(tmp / "outputs" / "torn_s42" / "metrics.jsonl").write_text(
    '{"update": 7}\n{"update": 8}\n{"upda', encoding="utf-8")
check("半行跳过、不猜（取到最后一个完整轮号）",
      mod.rounds_and_age("torn_s42")[0], 8)

# ============================================================ 2. 判据真值表
print()
print("2. `classify()` 真值表 —— 判据的**唯一实现**（缺陷 2/3）")
TT = [
    (None, None, "vanished", "首次采样没 tick 到 ⟹ 进程消失（旧：漏报）"),
    (0, None, "vanished", "复采窗口消失（旧：落进 else，谎报『已恢复』）"),
    (0, 0, "frozen", "两次连续 0 ⟹ 冻结（**唯一**该报冻结的格子）"),
    (0, 12345, None, "首次 0、二次在算 ⟹ 疑似停顿，**不报**"
                     "（旧：rc 用首次 delta==0 ⟹ exit 1，与自己的打印矛盾）"),
    (12345, None, None, "正常在算（复采不适用）"),
    (0, 7, None, "『已恢复』格的另一个二次读数"),
    (1, None, None, "边界：delta=1 也是在算"),
]
for d, d2, want, why in TT:
    check("classify(%-6s, %-5s) ⟹ %-8s" % (d, d2, want or "None"),
          mod.classify(d, d2, None)[0], want)
    print("       %s" % why)

print()
print("3. metrics 停滞是**独立**的一类（旧实现把它和 delta 混在一条 if/elif）")
k, f = mod.classify(12345, None, 450.0)
check("在算但 metrics 停滞 ⟹ 不报冻结，只给 ⚠", (k, "停滞" in f), (None, True))
check("冻结优先于停滞（两条件同时成立时报更严重的那个）",
      mod.classify(0, 0, 450.0)[0], "frozen")

# ============================================================ 4. 判据只有一处
print()
print("4. 判据必须**只有一处**（旧实现：明细一处、结论段另算一处）")
# ★ 只数**活代码**（先剥注释），否则注释里引用的旧代码会污染断言。
#   这是本文件的第 4 版写法 —— 前三版都栽在"测试自己做文本考古"上。
_lines = SRC.read_text(encoding="utf-8").splitlines()


def _find(pat, start=0):
    return next((i for i in range(start, len(_lines)) if pat in _lines[i]), -1)


def _strip(seg):
    return "\n".join(re.sub(r"#.*$", "", ln) for ln in seg)


i_cls = _find("def classify(")
i_nxt = next((i for i in range(i_cls + 1, len(_lines))
              if _lines[i].startswith("def ")), -1)
i_main = _find("def main(")
i_loop = _find("for r, d in rows:", i_main)      # [1] 段的明细打印循环
i_sec2 = _find("# [2]", i_main)
i_concl = _find("# [3] 结论", i_main)
# ★ 先用**正例**锚定切段，后面所有"不得出现"才有意义（防空过）
check("六个行号锚点都找得到（_lines 已按行拆开，锚点里不能带 \\n）",
      -1 in (i_cls, i_nxt, i_main, i_loop, i_sec2, i_concl), False)
check("锚点顺序自洽（cls < [1]循环 < [2] < [3]）",
      i_cls < i_nxt < i_main < i_loop < i_sec2 < i_concl, True)

cls_live = _strip(_lines[i_cls:i_nxt])
loop_live = _strip(_lines[i_loop:i_sec2])        # ★ 只取 [1]；[2] 的 ✗ 是另一回事
tail_live = _strip(_lines[i_concl:])
whole_live = _strip(_lines)
check("三段都非空（切空会让下面的断言全部空过）",
      min(len(cls_live), len(loop_live), len(tail_live)) > 50, True)

check("结论段按 frozen/vanished 判",
      'r.get("frozen") or r.get("vanished")' in tail_live, True)
check("结论段不得重算 delta", "delta" in tail_live, False)
check("结论段不得重算 any(...)", "any(" in tail_live, False)
check("classify 是唯一实现（def 一处 + 调用一处）",
      whole_live.count("classify("), 2)
check("打印循环只调用它", loop_live.count("classify("), 1)
check("打印循环里没有 delta == 0（δ 只活在 classify 里）",
      "delta == 0" in loop_live, False)
check("✗ 判据确实存在于 classify 里（防止下一条空过）", "✗" in cls_live, True)
check("明细打印循环里没有 ✗ 判据（判据已搬走）", "✗" in loop_live, False)

# ============================================================ 5. 空集第三态
print()
print("5. 空集是**第三态**（缺陷 4：旧实现说『全部健康（0 条 run 在算）』）")
check("空集分支返回 3", "return 3" in tail_live, True)
check("话术存在『没有可判读的对象』", "没有可判读的对象" in tail_live, True)
check("『不是健康也不是故障』说清楚", "既不是健康也不是故障" in tail_live, True)

# ============================================================ 6. 变异测试
print()
print("6. 变异测试：**把修复前的实现写出来跑**，必须与新实现给出不同答案")
print("   （没有这一节，前面所有 '旧：…' 都只是散文注释）")


def old_classify(delta, reconfirm_d2):
    """修复前的判据 —— 内联在打印循环里的那条 if/elif 链。"""
    if delta == 0:                    # ← 注意：不看 None
        return "frozen" if reconfirm_d2 == 0 else None
    return None


def old_rc(runs):
    """修复前结论段的 rc（用**首次** delta 判冻结）。"""
    rc = 0
    if runs and any(r.get("delta") == 0 for r in runs):
        rc |= 1
    return rc


def old_rounds(path):
    """修复前的轮号 —— 数带 update 的行数。"""
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines()
               if ln.strip() and '"update"' in ln)


for d, d2, why in [
    (None, None, "delta=None ⟹ 旧实现丢弃（进程死了当没事）"),
    (0, None, "二次窗口消失 ⟹ 旧实现谎报『已恢复』"),
    (12345, None, "正常在算 ⟹ 两边一致（这一格**不该**分叉）"),
]:
    o, n = old_classify(d, d2), mod.classify(d, d2, None)[0]
    if d == 12345:
        check("变异体与新版在此格一致（对照组）%s" % why, (o, n), (None, None))
    else:
        check("变异体在此格漏报/谎报，新版报 vanished：%s" % why,
              (o, n), (None, "vanished"))

check("旧 rc 在『首次 0、二次在算』上置 1（与它自己的明细打印矛盾）",
      old_rc([{"run": "x", "delta": 0}]), 1)
check("新判据在同一格不报 ⟹ dead 为空 ⟹ rc=0",
      mod.classify(0, 12345, None)[0], None)
check("旧 rc 在空集上给 0（= 健康）", old_rc([]), 0)

n_old = old_rounds(P_CONT)
check("变异体把续跑臂读成 %d（真值 25，少算 %d）" % (n_old, 25 - n_old), n_old, 20)
check("新实现对同一文件读 25", mod.rounds_and_age("cont_s42")[0], 25)

# ============================================================ 7. 语法
print()
print("7. 语法")
try:
    py_compile.compile(str(SRC), doraise=True, cfile=str(tmp / "wl.pyc"))
    check("py_compile 通过", True, True)
except py_compile.PyCompileError as e:
    check("py_compile 通过（%s）" % e, False, True)

print()
print("=" * 74)
print("全部通过 ✓" if _ok[0] else "**有失败 ✗**")
sys.exit(0 if _ok[0] else 1)
