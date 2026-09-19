#!/usr/bin/env python
"""按路径比较节点与本地工作区文件，**两侧都做行尾归一化**。

为什么两边都要归一化：上一版只在本地侧归一化，节点侧用的是 `sha256sum` 的
原始哈希。于是"节点 CRLF / 本地 LF"这类纯换行差异被报成"训练代码不同"，
差点让我据此否决一次有效配对。**diff 一跑就知道是空差异** —— 这正是本项目
记过的"行尾符陷阱"，我按自己写的警告又踩了一次。

教训写进这里：**比较两个来源的哈希时，归一化必须对两侧同时施加**；
只归一化一侧等于在比较"内容"和"内容+换行约定"两个不同的量。

输出写文件不用控制台：Windows 控制台是 GBK，非 ASCII 字符会 UnicodeEncodeError
（上面那次崩就是），而崩掉的位置恰好在打印结论那一行 —— **结论行最容易丢**。

用法：python .tmp/hashcmp.py <node_hashes_norm.txt>
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = ["qkd_rl", "scripts/train", "configs"]
EXTS = {".py", ".yaml"}
REPORT = ROOT / ".tmp" / "hashcmp_report.txt"

SENSITIVE = [
    "qkd_rl/rl/algos/mappo_trainer.py",
    "qkd_rl/rl/algos/policy.py",
    "qkd_rl/rl/algos/gae.py",
    "qkd_rl/rl/algos/rollout_buffer.py",
    "qkd_rl/rl/algos/rollout_workers.py",
    "qkd_rl/env/env.py",
    "qkd_rl/env/action_resolver.py",
    "qkd_rl/env/reward.py",
    "qkd_rl/env/qkp.py",
    "qkd_rl/env/routing.py",
    "qkd_rl/env/metrics.py",
    "qkd_rl/evaluation/test_protocol.py",
    "configs/rl_algorithm.yaml",
    "configs/train_full_rl.yaml",
    "configs/train_ent01.yaml",
    "configs/features.yaml",
    "configs/global.yaml",
    "configs/graph_mappo.yaml",
    "configs/env_full.yaml",
    "configs/train_profiles.yaml",
    "configs/default.yaml",
]


def norm_digest(p: Path) -> str:
    data = p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def raw_digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def local_hashes() -> dict[str, str]:
    out = {}
    for base in PATHS:
        d = ROOT / base
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if not f.is_file() or f.suffix not in EXTS:
                continue
            if "/archive/" in str(f).replace("\\", "/"):
                continue
            out[str(f.relative_to(ROOT)).replace("\\", "/")] = norm_digest(f)
    return out


def main() -> int:
    node = {}
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            h, _, p = line.partition(" ")
            node[p.strip()] = h.strip()

    loc = local_hashes()
    differ = sorted(k for k in set(node) & set(loc) if node[k] != loc[k])
    only_node = sorted(set(node) - set(loc))
    only_local = sorted(set(loc) - set(node))

    L = []
    L.append(f"node={len(node)} local={len(loc)}")
    L.append(f"differ={len(differ)} only_node={len(only_node)} only_local={len(only_local)}")
    for k in differ:
        L.append(f"DIFF {k}")
        L.append(f"  node {node[k][:40]}")
        L.append(f"  locl {loc[k][:40]}")
    for k in only_node[:20]:
        L.append(f"ONLY_NODE {k}")
    for k in only_local[:20]:
        L.append(f"ONLY_LOCAL {k}")

    L.append("")
    L.append("--- training-critical files ---")
    bad = 0
    for f in SENSITIVE:
        a, b = node.get(f), loc.get(f)
        if a is None:
            s = "MISSING_ON_NODE"
        elif b is None:
            s = "MISSING_LOCAL"
        elif a == b:
            s = "SAME"
        else:
            s = "DIFFERENT"
            bad += 1
        L.append(f"{s:<16} {f}")

    # 额外的诊断：哪些文件的差异**纯粹来自行尾**
    eol_only = []
    for k in differ:
        lp = ROOT / k
        if lp.exists() and node.get(k) == hashlib.sha256(lp.read_bytes()).hexdigest():
            eol_only.append(k)
    L.append("")
    L.append(f"--- pure-EOL-only diffs: {len(eol_only)} ---")
    for k in eol_only[:20]:
        L.append(f"  {k}")

    L.append("")
    if bad == 0:
        L.append(f"RESULT: OK - all {len(SENSITIVE)} training-critical files identical")
        L.append("  new arms and ent01 are the same code -> paired comparison valid")
    else:
        L.append(f"RESULT: {bad} training-critical files differ -> cross-version, pairing INVALID")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    # 控制台只打 ASCII 摘要，避免 GBK 编码崩在结论行
    print(f"report -> {REPORT}")
    print(f"differ={len(differ)} pure_eol={len(eol_only)} critical_bad={bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
