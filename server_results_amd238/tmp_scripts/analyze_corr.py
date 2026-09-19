"""把 corr_rss 实验的「每轮耗时」和「RSS 曲线」对齐，看两者是否同步封顶。

用法（本地，工作区根目录）：python .tmp/analyze_corr.py

输入：
  .tmp/corr_rss.rss       服务器采的 RSS，每行 "elapsed_s main_MB worker_MB"
  .tmp/corr_rss_metrics.jsonl   训练自己写的每轮指标（含 elapsed_s）
输出：对齐表 + 封顶判断
"""
import json
import pathlib
import sys

# 按 CLAUDE.md 的约定，脚本放 .tmp/ 下，用显式 utf-8 读中文路径无关但保持一致
ROOT = pathlib.Path(__file__).resolve().parent
RSS = ROOT / "corr_rss.rss"
MET = ROOT / "corr_rss_metrics.jsonl"

if not RSS.exists() or not MET.exists():
    sys.exit(f"缺输入文件：{RSS.exists()=} {MET.exists()=}")


def load_rss(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        # 容错：老格式可能是 4 列（多一个空的 upd 位），只取首列和最后两列
        if len(parts) < 3:
            continue
        try:
            t = float(parts[0])
            main = float(parts[-2])
            wrk = float(parts[-1])
        except ValueError:
            continue          # 表头
        rows.append((t, main, wrk))
    return rows


def load_metrics(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" not in d or "update_s" not in d:
            continue          # 可能是 eval 那一行
        out.append(d)
    return out


rss = load_rss(RSS)
met = load_metrics(MET)
if not rss or not met:
    sys.exit("解析后为空，检查输入文件格式")

# 每轮结束时刻：训练自己记的 elapsed_s 就是"本轮结束时的累计秒"
print(f"{'轮':>3} {'rollout_s':>10} {'update_s':>10} {'elapsed_s':>10} "
      f"{'main_MB':>9} {'workers_MB':>10} {'总RSS_GB':>9}")
print("-" * 72)

prev_main = None
deltas = []
for m in met:
    end = float(m["elapsed_s"])
    # 取该轮结束时刻之前最近的一次 RSS 采样
    cand = [r for r in rss if r[0] <= end]
    if not cand:
        continue
    t, main, wrk = cand[-1]
    total = (main + wrk) / 1024
    dm = "" if prev_main is None else f"{(main - prev_main):+.0f}"
    if prev_main is not None:
        deltas.append((m["update"], main - prev_main))
    prev_main = main
    print(f"{m['update']:>3} {m.get('rollout_s', 0):>10.1f} {m['update_s']:>10.1f} "
          f"{end:>10.1f} {main:>9.0f} {wrk:>10.0f} {total:>9.2f}   {dm}")

# 判断是否封顶
print()
if len(met) >= 6:
    u = [m["update_s"] for m in met]
    first, last = u[0], u[-1]
    print(f"update_s：首轮 {first:.1f}s → 末轮 {last:.1f}s  ({last / first - 1:+.1%})")
    half = len(u) // 2
    g1 = u[half - 1] - u[0]
    g2 = u[-1] - u[half]
    print(f"  前半段增幅 {g1:+.1f}s，后半段增幅 {g2:+.1f}s "
          f"→ {'仍在爬升（未封顶）' if g2 > g1 * 0.5 else '已趋缓/封顶'}")
else:
    print(f"只有 {len(met)} 轮，不足以判断封顶")

if deltas:
    print()
    print("每轮 main RSS 增量(MB)：", " ".join(f"{d:+.0f}" for _, d in deltas))
