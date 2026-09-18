#!/usr/bin/env python
"""严格校验 configs/*.yaml 能被 yaml.safe_load 解析。

**为什么需要它**：`configs/train_ent01_off.yaml` 第一次启动失败，原因是我在
文件头的注释块里写了一段**没加 `#` 的命令行示例**，YAML 把它当成文档内容：

    yaml.parser.ParserError: expected '<document start>', but found '<scalar>'
      in "configs/train_ent01_off.yaml", line 27, column 1

这个错误**只在训练启动时才暴露**，而启动要走 ssh + 排队等内存，
代价是十几分钟。写一个能本地（服务器上）秒级跑的校验器，就能在提交前挡住。

同时检查两个容易踩的坑：
  1. **后覆盖前**：yaml 按顺序深合并，靠后的覆盖靠前的。`train_ent01_off.yaml`
     这类"把某一项改回去"的对照配置**必须排在它要覆盖的文件之后**，
     否则完全无效（而且不会报错，只是安静地按前者跑）。
  2. 顶层键是否落在已知集合里（防止拼错 `trainn:` 这种静默空操作）。

用法（服务器上）：python scripts/diag/check_configs.py
     指定文件：  python scripts/diag/check_configs.py configs/train_ent01_off.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

# 按 `configs/*.yaml` 顶层键的**实际并集**校准（19 个，`.tmp/top_keys.py` 列出）。
# 不校准的话警告会恒亮（第一版把 global/model/evaluation/train_profiles 四个
# 合法的键误报为"不认识"）——**一个总是触发的警告等于没有警告**。
KNOWN_TOP = {
    "action_resolver", "baselines", "env", "evaluation", "features", "global",
    "model", "project", "qkp", "rate_provider", "requests", "reward", "routing",
    "runtime", "scenario", "seed", "train", "train_profiles", "validation",
}


def rel(p: Path) -> str:
    """相对仓库根显示；仓库外的文件（如 /tmp/bad.yaml）直接给绝对路径。

    第一版直接 p.relative_to(ROOT)，传仓库外文件时抛 ValueError
    ——校验器反过来崩了，比它要挡的错误还难查。
    """
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def check(path: Path) -> list[str]:
    errs: list[str] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        # 报出行号，并把它附近的内容打出来 —— 行号是定位这类错误最快的线索
        mark = getattr(exc, "problem_mark", None)
        where = f" 第 {mark.line + 1} 行" if mark is not None else ""
        errs.append(f"YAML 解析失败{where}: {exc}")
        if mark is not None:
            lines = path.read_text(encoding="utf-8").splitlines()
            for i in range(max(0, mark.line - 3), min(len(lines), mark.line + 3)):
                flag = ">>" if i == mark.line else "  "
                errs.append(f"    {flag} {i + 1:4d} | {lines[i]}")
        return errs
    if data is None:
        errs.append("文件为空或被全部注释掉（safe_load 返回 None）")
        return errs
    if not isinstance(data, dict):
        errs.append(f"顶层不是映射，而是 {type(data).__name__}")
        return errs
    unknown = set(data) - KNOWN_TOP
    if unknown:
        # 只警告，不判错：有些 profile 可能有自己的段
        errs.append(f"[警告] 顶层键不认识（可能是拼写错误，会被静默忽略）: {sorted(unknown)}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="默认检查 configs/*.yaml 全部")
    args = ap.parse_args()

    files = [Path(f) for f in args.files] if args.files else sorted(CONFIGS.glob("*.yaml"))
    bad = 0
    for p in files:
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            print(f"!! 不存在: {p}")
            bad += 1
            continue
        errs = check(p)
        real_errs = [e for e in errs if not e.startswith("[警告]")]
        if real_errs:
            bad += 1
            print(f"✗ {rel(p)}")
            for e in errs:
                print(f"    {e}")
        else:
            tag = "  " + errs[0] if errs else ""
            print(f"✓ {rel(p)}{tag}")

    print(f"\n{len(files) - bad}/{len(files)} 通过")
    if bad:
        print("提示：注释块里写代码示例时每一行都要 '#'，否则 YAML 会把它当文档内容。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
