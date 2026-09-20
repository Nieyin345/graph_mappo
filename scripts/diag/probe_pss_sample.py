"""采样所有训练臂的 PSS 与当前轮号，算增长率。

用途：11 条臂同时在跑，需判断会不会 OOM。
判据不能靠记忆里的"25 GiB 常数"——本节点实测 cmax 在 u27 只有 15.8 GiB，
与常数差 37%。要么常数过期，要么本配置更省。实测外推。
"""
import json
import sys
import time
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
OUT = REPO / "outputs"


def sample():
    """/proc 扫描 + 轮号。"""
    import subprocess
    r = subprocess.run(
        "ps -eo pid,comm,args | awk '$2 ~ /^python/'",
        shell=True, capture_output=True, text=True)
    rows = []
    for ln in r.stdout.splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 3:
            continue
        pid, comm, args = parts
        if "train_graph_mappo" not in args or "--run-name" not in args:
            continue
        toks = args.split()
        nm = toks[toks.index("--run-name") + 1]
        pss = 0.0
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    pss = int(line.split()[1]) / 1024.0 / 1024.0
                    break
        except Exception:
            pass
        upd = -1
        mf = OUT / nm / "metrics.jsonl"
        if mf.exists():
            last = None
            for line in mf.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if "update" in o:
                    last = o["update"]
            if last is not None:
                upd = int(last)
        rows.append((nm, pss, upd))
    return sorted(rows)


def meminfo():
    d = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        d[k.strip()] = float(v.split()[0]) / 1024.0 / 1024.0
    return d


def main():
    wait = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if wait:
        print(f"等 {wait}s 再采第二个点...")
        time.sleep(wait)

    rows = sample()
    mi = meminfo()
    tot = sum(r[1] for r in rows)
    print("=" * 88)
    print(f"{'臂':<20}{'PSS GiB':>10}{'u':>5}")
    print("=" * 88)
    for nm, pss, upd in rows:
        print(f"{nm:<20}{pss:>10.1f}{upd:>5}")
    print("-" * 88)
    print(f"{'合计':<20}{tot:>10.1f}   （{len(rows)} 条）")
    print(f"\n  MemTotal     {mi['MemTotal']:.1f} GiB")
    print(f"  MemAvailable {mi['MemAvailable']:.1f} GiB")
    print(f"  地板 17 ⟹ 余量 {mi['MemAvailable'] - 17:.1f} GiB")
    print(f"  均值 {tot/max(1,len(rows)):.1f} GiB/条")

    # 外推：若每条涨到 25 或 17
    for steady in (25.0, 20.0, 17.0):
        need = steady * len(rows)
        print(f"  若稳态 {steady:.0f} GiB/条 ⟹ 最终 {need:.0f} GiB  "
              f"{'★ OOM' if need > mi['MemTotal'] - 17 else '✓ 安全'}")

    json.dump({"rows": rows, "total": tot, "mem": mi},
              open("/tmp/pss_sample.json", "w"))


if __name__ == "__main__":
    main()
