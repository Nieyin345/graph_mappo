"""建一份隔离副本 + 放进改动后的两个文件 + 跑探针。全程不碰活包。

为什么隔离：15 个并发 run/worker 还在跑，`/opt/qkd/graph_mappo/qkd_rl/`
是它们**正在用的包**。改活包 = 后续轮次的代码与已跑轮次不同，两条曲线报废。
验证"多记两个字段"这种低收益的事不值得这个风险。

成本很低：整个仓库排除 outputs 只有 725 MB，且只需拷源码
（worker 的 site-packages 在 venv 里，不复制）。副本在 /tmp（根分区还有 40G）。

跑法（服务器上）：
    /opt/qkd/venv/bin/python /tmp/mk_isolated_copy.py
    /opt/qkd/venv/bin/python /tmp/probe_metrics_keys.py /tmp/metrics_check
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

SRC = Path("/opt/qkd/graph_mappo")
DST = Path("/tmp/metrics_check")
SKIP = {"outputs", ".git", "__pycache__", ".venv", "node_modules", ".tmp"}

if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)

copied = 0
for root, dirs, files in os.walk(SRC):
    rel = Path(root).relative_to(SRC)
    if any(part in SKIP for part in rel.parts):
        dirs[:] = []
        continue
    dirs[:] = [d for d in dirs if d not in SKIP]
    (DST / rel).mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            shutil.copy2(Path(root) / f, DST / rel / f)
            copied += 1
        except OSError:
            pass
print(f"副本 {DST}：拷贝 {copied} 个文件（排除 {sorted(SKIP)}）")

# 把**改动后的**两份文件放进副本（scp 到 /tmp/ 的那一对）。
PAIRS = [
    ("/tmp/new_metrics.py", DST / "qkd_rl" / "env" / "metrics.py"),
    ("/tmp/new_mappo_trainer.py", DST / "qkd_rl" / "rl" / "algos" / "mappo_trainer.py"),
]
for src, dst in PAIRS:
    s = Path(src)
    if not s.exists():
        print(f"  !! 缺 {src} —— 副本里保留原版（探针会据此失败，是预期行为）")
        continue
    shutil.copy2(s, dst)
    print(f"  覆盖 {dst.relative_to(DST)}  <-  {src}")

# 副本里也清掉 __pycache__，否则可能加载到旧字节码。
n = 0
for p in DST.rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)
    n += 1
print(f"清掉 {n} 个 __pycache__ 目录")
