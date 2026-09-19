# -*- coding: utf-8 -*-
"""审计：哪些配置叶子键**根本没有代码去读**？

动机是 actor_lr 那个 bug —— YAML 里写着、resolved_config.yaml 里也写着、
每个 run 都"用了"它，**但 load_checkpoint 恢复 param_groups 把它盖掉了**，
于是全部历史训练实际跑在 0.001 上。这类 bug 的可怕之处：
**配置看起来生效了**，任何只看 resolved_config 的自检都会说 ok。

这里做**静态**版本：把生效配置的每个叶子键，去源码里找有没有人读它。
找不到 → 该旋钮是**装饰品**，改它不会改变任何东西。

局限（必须写在结论里）：找到名字 ≠ 真的生效（可能是死代码、可能被
后续覆盖、也可能像 actor_lr 那样被别的机制盖掉）。所以本脚本只能
**证伪**（零引用 = 一定没用），不能**证实**。但"零引用"这一侧正是
actor_lr 那类 bug 的入口。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
SRC_DIRS = [ROOT / "qkd_rl", ROOT / "scripts"]

# 取一份真实生效的配置（ent01_s45 刚写的）
import yaml
rc = ROOT / "outputs" / "ent01_s45" / "resolved_config.yaml"
if not rc.exists():
    cands = sorted((ROOT / "outputs").glob("*/resolved_config.yaml"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    rc = cands[0]
print(f"用 {rc.relative_to(ROOT)} 作为生效配置样本\n")
cfg = yaml.safe_load(rc.read_text(encoding="utf-8", errors="replace"))


def leaves(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{prefix}{k}.")
    elif isinstance(o, list):
        yield prefix.rstrip("."), o
    else:
        yield prefix.rstrip("."), o


# 收集源码全文（只收 .py）
src = {}
for d in SRC_DIRS:
    for p in d.rglob("*.py"):
        try:
            src[p] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
allsrc = "\n".join(src.values())
print(f"扫描 {len(src)} 个 .py 文件，共 {len(allsrc):,} 字符\n")

# 按**父节**分组：cfg 的顶层节名
rows = []
for key, val in leaves(cfg):
    leaf = key.split(".")[-1]
    if leaf in ("", "null"):
        continue
    # 在源码里找这个键名（作为字符串或属性访问）
    pat = re.compile(rf"""["']{re.escape(leaf)}["']""")
    n_str = len(pat.findall(allsrc))
    # 也看 `cfg.get("x")` / `config["x"]` 这类已被上面覆盖；额外看裸标识符
    n_any = len(re.findall(rf"\b{re.escape(leaf)}\b", allsrc))
    rows.append((key, repr(val)[:38], n_str, n_any))

zero = [r for r in rows if r[2] == 0]
low = [r for r in rows if r[2] == 1]

print("=" * 100)
print(f"1. **零字符串引用**的叶子键（源码里没有任何 \"key\" 字面量）—— {len(zero)} 个")
print("=" * 100)
print(f"  {'配置路径':<52}{'值':<40}{'裸词频'}")
for key, v, ns, na in zero:
    flag = "  ← 裸词也没有？" if na == 0 else ""
    print(f"  {key:<52}{v:<40}{na}{flag}")

print()
print("=" * 100)
print(f"2. 只被引用 **1 次**的叶子键 —— {len(low)} 个（最可能是装饰品）")
print("=" * 100)
for key, v, ns, na in low:
    print(f"  {key:<52}{v:<40}")

print()
print("=" * 100)
print("3. 判读")
print("=" * 100)
print("  · 第 1 节里**裸词频也为 0** 的，是最硬的证据：源码里连这个标识符都没有。")
print("  · 但**名字没被引用的键未必无害** —— 有的键是容器（list/dict），")
print("    代码读的是它下面的子键或整个父节，所以父节名可能零引用。")
print("  · 真正要盯的是**标量旋钮**（lr / coef / size / 步数）零引用。")
