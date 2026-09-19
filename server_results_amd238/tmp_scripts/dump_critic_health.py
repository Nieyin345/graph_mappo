"""对比训练侧标量：critic 到底有没有拟合上。

起因是读 real_base 的第 1 轮日志时看到

    V_std=0.020  R_std=0.39  corr(V,R)=-0.084

critic 预测值的标准差只有 0.02，而回报的标准差是 0.39 —— 也就是说 critic
输出的几乎是常数，和真实回报基本不相关（-0.084）。**如果它一直是这样，
优势函数 A = R - V 就等于 R 减一个常数，PPO 的"优势"退化成原始回报**，
而且 value_coef 那一项也学不到东西。

这里只看**训练侧**（metrics.jsonl 的 update 行），不碰验证。目的：
  1. V_std 是不是一直贴地？还是只在前几轮？
  2. corr(V,R) 有没有随训练升上去？
  3. 真实配置 vs 诊断配置，这个量的差别有多大？

用法（本地，读 server_results 里已经拉回来的 jsonl）：
    python .tmp/dump_critic_health.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_runs_root():
    """在节点上是 outputs/，在本地是拉回来的 server_results/runs/outputs/。

    两个都试，取先存在的 —— 这样同一个脚本两边都能直接跑，不用改路径。
    """
    for cand in (ROOT / "outputs", ROOT / "server_results" / "runs" / "outputs"):
        if cand.is_dir():
            return cand
    return ROOT / "outputs"


RUNS = find_runs_root()

# 想看的运行：诊断配置的基线 vs 真实配置的基线，外加两个变体做对照。
GROUPS = {
    "诊断 base": "r2_base",
    "诊断 mini512": "r2_mini512",
    "诊断 vcoef1": "r2_vcoef1",
    "真实 base": "real_base",
    "真实 mini512": "real_mini512",
    "真实 vcoef1": "real_vcoef1",
}

# jsonl 里的键名和日志打印的不一样（日志 corr(V,R) / V_std，jsonl
# value_return_corr / value_std）—— 第一版按日志名去查，整张表全是 "—"。
KEYS = [
    "value_std",
    "return_std",
    "value_return_corr",
    "mean_abs_advantage",
    "entropy",
    "critic_loss",
]


def load_updates(name):
    p = RUNS / name / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" in d:
            out.append(d)
    return out


def fmt(v):
    if v is None:
        return f"{'—':>10}"
    return f"{v:>10.3f}"


HEAD = {
    "value_std": "V_std",
    "return_std": "R_std",
    "value_return_corr": "corr(VR)",
    "mean_abs_advantage": "|A|",
    "entropy": "entropy",
    "critic_loss": "closs",
}

print(f"{'运行':<14}{'轮':>4}  " + "".join(f"{HEAD[k]:>10}" for k in KEYS))
print("-" * (18 + 10 * len(KEYS)))

for label, name in GROUPS.items():
    rows = load_updates(name)
    if not rows:
        print(f"{label:<14}  (没有记录)")
        continue
    # 首轮 / 中位轮 / 末轮 —— 只看这三个点，避免刷屏。
    idxs = sorted({0, len(rows) // 2, len(rows) - 1})
    for i in idxs:
        r = rows[i]
        tag = f"{label}" if i == idxs[0] else ""
        print(f"{tag:<14}{r.get('update', i + 1):>4}  "
              + "".join(fmt(r.get(k)) for k in KEYS))
    print()

print("读法：V_std 是 critic 预测值的标准差，R_std 是回报的标准差。")
print("两者同量级 = critic 跟得上；V_std 远小于 R_std 且 corr(VR) 贴 0 = 没拟合上。")
print("|A| 是优势的绝对均值 —— 它塌到 0 就意味着 PPO 没有梯度信号可用。")
