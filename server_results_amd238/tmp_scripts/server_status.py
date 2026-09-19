#!/usr/bin/env python
"""把当前所有实验状态汇总成一行 JSON，供本地唤醒轮询。

输出写到 /tmp/qkd_status.json，内容：
  {
    "ts": "2026-09-19T12:00:00-06:00",
    "running": ["r9ext_s52"],          # 正在跑的 run（从进程命令行解析）
    "runs": {                           # 每个 run 的最新验证曲线
      "r9ext_s52": {"update": 12, "rows": 2, "vals": [[5,0.66],[10,0.68]]}
    },
    "finished": ["r8_base_s42"],        # 有 checkpoint_final 的
    "chain_alive": true
  }

本地唤醒者只看这个文件的哈希有没有变，变了就拉回来分析。
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
STATUS = Path("/tmp/qkd_status.json")
BC0 = 0.6483
EXPERT = 0.6980


def running_runs():
    """从进程命令行里解析正在跑的 run-name。"""
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                            text=True, timeout=10).stdout
    except Exception:
        return []
    out = []
    for line in ps.splitlines():
        if "train_graph_mappo.py" not in line or "bash -c" in line:
            continue
        m = re.search(r"--run-name\s+(\S+)", line)
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def summarize(run_dir: Path):
    """读 metrics.jsonl，返回 (最新轮, 验证曲线点)。"""
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return None
    last_update = 0
    vals = []
    try:
        for line in p.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" in row:
                last_update = max(last_update, int(row["update"]))
            if "eval_validation" in row:
                ev = row["eval_validation"]
                vals.append([last_update,
                             round(float(ev.get("mean_success_rate", 0.0)), 4)])
    except Exception:
        return None
    return {"update": last_update, "vals": vals,
            "done": (run_dir / "checkpoint_final.pt").exists()}


def main():
    runs = {}
    finished = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir() or not d.name.startswith(("r8_", "r9", "r1")):
            continue
        s = summarize(d)
        if s is None:
            continue
        runs[d.name] = s
        if s["done"]:
            finished.append(d.name)

    live = running_runs()
    status = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "running": live,
        "runs": runs,
        "finished": finished,
        "chain_alive": subprocess.run(
            ["pgrep", "-f", "chain_r9ext"], capture_output=True).returncode == 0,
        "refs": {"BC0": BC0, "expert": EXPERT},
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=1),
                      encoding="utf-8")

    # 人读的一行摘要
    print("== 状态 {} ==".format(status["ts"]))
    print("在跑: {}".format(", ".join(live) or "无"))
    print("接力守护: {}".format("在" if status["chain_alive"] else "**没了**"))
    for name, s in runs.items():
        tag = "完成" if s["done"] else "u{}".format(s["update"])
        curv = " ".join("{:.4f}".format(v) for _, v in s["vals"][-4:])
        best = max((v for _, v in s["vals"]), default=0.0)
        print("  {:<16} {:<6} 验证: {:<34} 最佳 {:.4f} (BC {:.4f}/专家 {:.4f})".format(
            name, tag, curv, best, BC0, EXPERT))


if __name__ == "__main__":
    main()
