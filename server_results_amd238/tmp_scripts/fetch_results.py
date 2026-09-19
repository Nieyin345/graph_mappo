#!/usr/bin/env python
"""把服务器上的「结果」（不是权重）抓回本地。

用户要求：只要好的结果，以及如何调的参数、如何优化的结构；不要权重。

抓什么：
  - 每轮标量指标（含 eval_validation 曲线）+ 诊断量
  - resolved_config 的关键差异项
  - rollout_debug 的最后几轮（免费的高价值诊断数据）
不抓：
  - *.pt 权重（12 MB/个，用户明确不要）
  - rollout_debug 全量（很大）

实现：远端汇总脚本 .tmp/collect_runs.py 先 scp 到 /tmp，再经 ssh 执行，
只把 JSON 结果传回来——避免把大文件拉过网络，也避免内联 python -c
（项目红线：Windows 三层转义 + GBK 编码）。

用法（本地）：
  python .tmp/fetch_results.py --runs all
  python .tmp/fetch_results.py --runs r8_base_s42,r9ext_s53 --tail-debug 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REMOTE = "/opt/qkd/graph_mappo"


def run(cmd: list[str], timeout: int = 180) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} 失败: {r.stderr[:400]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all",
                    help="逗号分隔的 run 名，或 all")
    ap.add_argument("--out", default="server_results")
    ap.add_argument("--tail-debug", type=int, default=3,
                    help="抓最后 N 轮 rollout_debug（0=不抓）")
    args = ap.parse_args()

    print("[1/3] 上传远端汇总脚本...", flush=True)
    run(["scp", "-q", str(HERE / "collect_runs.py"),
         f"qkd:/tmp/collect_runs.py"])

    print(f"[2/3] 汇总 {args.runs} ...", flush=True)
    remote_json = run(["ssh", "qkd",
                       f"cd {REMOTE} && /opt/qkd/venv/bin/python "
                       f"/tmp/collect_runs.py {args.runs}"], timeout=300)
    data = json.loads(remote_json)

    outdir = ROOT / args.out / "runs"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'run':<18}{'轮数':>5}{'状态':>6}  验证曲线（最近 5 点）"
          f"{'':>4}最佳")
    print("-" * 78)
    for name, rec in sorted(data.items()):
        v = rec.get("validation") or []
        best = rec.get("best_val")
        fin = "完成" if rec["finished"] else "未完"
        vs = " ".join(f"{a}:{b}" for a, b in v[-5:]) or "(无)"
        print(f"{name:<18}{rec['n_updates']:>5}{fin:>6}  {vs:<34}"
              f"{best if best is not None else '-'}")
        (outdir / f"{name}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")

    print("-" * 78)
    print(f"[3/3] 写入 {outdir}（{len(data)} 个 run）")

    if args.tail_debug > 0:
        ddir = ROOT / args.out / "debug"
        ddir.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in data:
            try:
                txt = run(["ssh", "qkd",
                           f"tail -n {args.tail_debug} "
                           f"{REMOTE}/outputs/{name}/rollout_debug.jsonl "
                           f"2>/dev/null || true"], timeout=60)
            except Exception:
                continue
            if txt.strip():
                (ddir / f"{name}.jsonl").write_text(txt, encoding="utf-8")
                n += 1
        print(f"      rollout_debug 尾部 → {ddir}（{n} 个）")


if __name__ == "__main__":
    main()
