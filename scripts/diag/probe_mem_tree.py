"""测**含 worker 的真实总内存** + 增长率（11 条臂并发是否安全）。

## 为什么不能只看父进程

上一版 `probe_pss_sample.py` 只抓到父进程（worker 的 cmdline 是
`python -c from multiprocessing.spawn import spawn_main ...`，**不含**
`train_graph_mappo`）⟹ 报 11 GiB/条，而记忆说 25 GiB。
差的那部分就是 8 个 worker。

## 本版做法

按**进程组**归并：worker 的 PPID 是父进程 PID ⟹ 以父为单位聚合整棵树。
采两次（间隔 N 秒）算增长率，外推到 u60。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
OUT = REPO / "outputs"


def pss_of(pid):
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return 0.0


def scan():
    """{run_name: (父PSS, worker合计, update)}"""
    # 所有 python 进程：pid ppid args
    r = subprocess.run(
        "ps -eo pid,ppid,comm,args", shell=True, capture_output=True, text=True)
    parents, workers = {}, []
    for ln in r.stdout.splitlines():
        parts = ln.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, comm, args = parts
        if not comm.startswith("python"):
            continue
        if "train_graph_mappo" in args and "--run-name" in args:
            toks = args.split()
            nm = toks[toks.index("--run-name") + 1]
            parents[int(pid)] = nm
        elif "multiprocessing" in args or "spawn_main" in args:
            workers.append((int(pid), int(ppid)))

    out = {}
    for pid, nm in parents.items():
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
        out[nm] = [pss_of(pid), 0.0, upd]

    for wpid, wppid in workers:
        nm = parents.get(wppid)
        if nm and nm in out:
            out[nm][1] += pss_of(wpid)
    return out


def memavail():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    raise RuntimeError("no MemAvailable")


def main():
    gap = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    a = scan()
    ma = memavail()
    print(f"=== 第 1 次 (MemAvailable {ma:.1f} GiB) ===")
    ta = sum(p + w for p, w, _ in a.values())
    for nm, (p, w, u) in sorted(a.items()):
        print(f"  {nm:<20} 父{p:5.1f} + worker{w:5.1f} = {p+w:5.1f} GiB   u={u}")
    print(f"  ---- 合计 {ta:.1f} GiB / {len(a)} 条  （均值 {ta/max(1,len(a)):.1f}）")

    print(f"\n等 {gap}s 采第二点...")
    time.sleep(gap)
    b = scan()
    mb = memavail()
    print(f"\n=== 第 2 次 (MemAvailable {mb:.1f} GiB) ===")
    tb = sum(p + w for p, w, _ in b.values())
    for nm, (p, w, u) in sorted(b.items()):
        prev = a.get(nm)
        d = (p + w) - (prev[0] + prev[1]) if prev else 0.0
        du = u - prev[2] if prev else 0
        print(f"  {nm:<20} {p+w:5.1f} GiB  Δ{d:+5.2f}  u={u} (Δ{du:+d})")
    print(f"  ---- 合计 {tb:.1f} GiB")

    print(f"\n  ★ MemAvailable {ma:.1f} → {mb:.1f}  (Δ{mb-ma:+.1f} GiB / {gap}s)")
    rounds = gap / 123.0     # 一轮约 123s
    if rounds > 0 and tb != ta:
        growth = (tb - ta) / rounds
        print(f"  ★ 增长 ≈ {growth:+.3f} GiB/轮  （{gap}s ≈ {rounds:.1f} 轮）")
        print(f"     外推到 u60（约 30 轮）: 合计 ≈ {tb + growth*30:.0f} GiB")
        print(f"     若均值到 25 GiB/条: {25*len(b):.0f} GiB   MemTotal 251")
    print(f"\n  地板 17 ⟹ 当前余量 {mb - 17:.1f} GiB")


if __name__ == "__main__":
    main()
