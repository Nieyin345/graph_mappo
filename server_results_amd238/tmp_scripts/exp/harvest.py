"""汇总 .tmp/exp/results.csv：按实验分组，丢弃预热轮，给稳态耗时对比。

用法：python .tmp/exp/harvest.py [warmup]   （warmup 默认 3）

为什么丢弃预热轮：corr_rss 实测同一个进程内 update_s 会爬升
u1=72.3 u2=77.5 u3=81.4 u4=82.1 u5=83.6 u6=83.9 u7=83.2 —— 前 3 轮在爬，
第 4~6 轮封顶在 ~83.5 s。拿首轮当"性能"会高估 16%。
"""
import csv
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent
CSV = ROOT / "results.csv"
WARMUP = 3

# 可选：只分析某个 tag（回填用）。CSV 里已有的行会被覆盖重写。
ONLY_TAG = None
args = sys.argv[1:]
if args and not args[0].isdigit():
    ONLY_TAG = args.pop(0)
if args:
    WARMUP = int(args[0])

if not CSV.exists():
    sys.exit(f"没有 {CSV}")

rows = []
with CSV.open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("update") in (None, "", "FAILED"):
            continue
        try:
            r["update"] = int(r["update"])
            r["update_s"] = float(r["update_s"])
            r["rollout_s"] = float(r["rollout_s"])
            r["success_rate"] = float(r["success_rate"])
        except (ValueError, KeyError):
            continue
        rows.append(r)

if not rows:
    sys.exit("结果表里没有可解析的行")

if ONLY_TAG:
    rows = [r for r in rows if r["tag"] == ONLY_TAG]
    if not rows:
        sys.exit(f"没有 tag={ONLY_TAG} 的行")

groups: dict[str, dict[int, dict]] = {}
for r in rows:
    g = groups.setdefault(r["tag"], {})
    u = r["update"]
    # 同一 tag 同名轮次取最后一次写入的（重跑时旧行还在表里）
    g[u] = r

out_tags = list(groups)

print(f"预热轮丢弃数：{WARMUP}（前 {WARMUP} 轮在爬升，不代表稳态）\n")
hdr = f"{'实验':<24} {'配置':<32} {'轮数':>4} {'rollout_s':>10} {'update_s':>10} {'一轮':>7} {'成功率':>8}"
print(hdr)
print("-" * len(hdr))

summary = []
for tag in out_tags:
    g = sorted(groups[tag].values(), key=lambda r: r["update"])
    steady = g[WARMUP:] or g          # 轮数不够就不丢
    u = [r["update_s"] for r in steady]
    ro = [r["rollout_s"] for r in steady]
    su = [r["success_rate"] for r in steady]
    cfg = (g[0].get("configs") or "(默认)")[:30]
    mu = statistics.mean(u)
    summary.append((tag, mu, statistics.mean(ro), mu + statistics.mean(ro)))
    print(f"{tag:<24} {cfg:<32} {len(steady):>4} "
          f"{statistics.mean(ro):>10.1f} {mu:>10.1f} "
          f"{mu + statistics.mean(ro):>7.1f} {statistics.mean(su):>8.4f}")

# 逐轮明细，用来看爬升形状
print("\n逐轮 update_s（看预热与封顶）：")
for tag in out_tags:
    g = sorted(groups[tag].values(), key=lambda r: r["update"])
    series = " ".join(f"{r['update_s']:.0f}" for r in g)
    print(f"  {tag:<24} {series}")

# 最优
if len(summary) > 1:
    best = min(summary, key=lambda s: s[3])
    base = summary[0]
    print(f"\n最快一轮：{best[0]}  {best[3]:.1f} s"
          f"（对比 {base[0]} 的 {base[3]:.1f} s，"
          f"{'快' if best[3] < base[3] else '慢'} {abs(best[3] / base[3] - 1):.1%}）")
