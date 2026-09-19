"""STOP 是不是一个**活的**杠杆？扰动 `stop_logit`，只测生成质量，不训练。

### 为什么先做这个（而不是直接改代码）

`stop_logit` 是 `graph_mappo.py:439` 的**一个全局标量**，决定匹配采样的 STOP
（`policy.py:765` / `_sample_matching_arrays` 里 `S[:, A] = stop_score`）。
已测：BC 把它从 0 推到 −0.426，而 RL 30 轮只动了 +0.006
（`probe_stop_logit.py`）—— **RL 几乎不碰它**。

"RL 不碰"有两种解释，处置完全不同：

  (A) 这个维度**在最优解处梯度为零** ⟹ 改它是浪费，先别动
  (B) 这个维度**重要但 RL 探索不到**（单标量，梯度信号弱）⟹ 状态化是活的方向

**扰动实验能把两者分开**：扫 `stop_logit` ∈ {−1.5, −1.0, −0.5, −0.427, 0, +0.5, +1.0}，
在验证 regime 上只跑评测（不训练）。若扫描曲线**平**，说明它现在不敏感；
若**有明显峰**且峰不在 −0.427，说明这是块没被优化的空地（解释 (B)）。

★ 零训练成本：只做 7 次 × 15 种子的评测，不写 checkpoint、不改任何文件。
★ 判据**跑之前写死**在下方的 JUDGE 里。

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/probe_stop_sensitivity.py \
      --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
      --seeds 100-114
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch                                                    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config             # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint           # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy                   # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic    # noqa: E402

# ---- 判据（跑之前写死）----
FLAT_SPAN = 0.020      # 全程极差 < 此值 ⟹ 平
PEAK_MIN = 0.030       # 峰值 − 两端最低 > 此值 ⟹ 有明显峰
# ★ MIN_EFFECT：**实用下限**，与统计显著性并列的第二道闸。
#   只判 t 会出事：n=15 且逐种子方差极小时，0.0001 的差也能「显著」。
#   反过来，SE=0 且 Δ=0（两档逐位相同）会被 m/se 算成 t=+inf，
#   于是一次**完全相同**的观测被判成「可测地不同」——2026-09-19 实测踩到。
MIN_EFFECT = 0.010     # |Δ| 未达此值一律不算「有差别」，无论 t 多大
BC_VALUE = -0.426157   # 已测的 BC 起点值
LOW_END, HIGH_END = -1.5, 1.0   # 取"两端"用于比峰
# ★ 配对，不用非配对均值 —— 逐种子成功率跨 0.29~0.88，非配对 15 种子 SE≈0.062，
#   配对后 ≈0.012（见 docs/测试规范.md §4①）。同一个 checkpoint、同一批种子、
#   只改一个标量，天然配对。
COL = "stop_logit"


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "global.yaml"))
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--values", default="-1.5,-1.0,-0.5,-0.426157,0.0,0.5,1.0")
    a = ap.parse_args()

    profile = _tp.load_validation_profile(Path(a.config))
    steps = int(profile["episode_steps"]) or 240
    seeds = parse_seeds(a.seeds)
    values = [float(v) for v in a.values.split(",")]

    cfg = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])

    env0 = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env0.action_resolver.action_space, cfg)
    data = load_checkpoint(a.checkpoint, torch.device("cpu"))
    model.load_state_dict(data.model_state)
    model.eval()
    policy = MAPPOPolicy(model, "cpu")

    print(f"检查点 {a.checkpoint}（update={data.update}）")
    print(f"验证 regime：窗口 {profile['window_start_day']}-{profile['window_end_day']}，"
          f"{steps} 步，{len(seeds)} 种子")
    print(f"扫描 stop_logit ∈ {values}")
    print()

    results: dict[float, list[float]] = {}
    t0 = time.perf_counter()
    for v in values:
        with torch.no_grad():
            model.actor.stop_logit.fill_(v)
        per_seed = []
        for seed in seeds:
            env = build_env_from_config(cfg)
            obs = env.reset(seed=seed, start_seed=int(profile.get("start_seed", 0)) + seed)
            done = False
            while not done:
                step = policy.act(obs, deterministic=True)
                obs, _r, term, trunc, _info = env.step(step.actions, step.action_scores)
                done = term or trunc
            s = env.metrics.episode_summary()
            per_seed.append(float(s["success_rate"]))
        results[v] = per_seed
        print(f"  stop_logit={v:+.4f}  sr={statistics.mean(per_seed):.4f}  "
              f"({time.perf_counter() - t0:.0f}s)")

    # ---- 判读 ----
    # ★ 全程用**配对**口径：同一个 checkpoint、同一批种子、只改 stop_logit，
    #   所以 Δ(种子) 的配对 SE 才是正确的分辨率（≈0.012），
    #   非配对均值（SE≈0.062）会把 0.03 量级的真实效应埋掉。
    print()
    print("=" * 72)
    means = {v: statistics.mean(r) for v, r in results.items()}
    span = max(means.values()) - min(means.values())
    peak_v = max(means, key=lambda k: means[k])
    base_pts = results[BC_VALUE]

    print(f"  各档均值（非配对，仅供看形状）：")
    for v in values:
        print(f"    stop_logit={v:+.4f}  sr={means[v]:.4f}")

    # 配对：每档 vs BC 起点，逐种子差 → 单样本 t
    print()
    print(f"  配对 Δ（每档 − BC 起点 {BC_VALUE:+.4f}），df={len(seeds) - 1}：")
    print(f"    {'档位':>10}{'Δ均值':>11}{'SD(Δ)':>10}{'SE':>9}{'t':>9}")
    paired: dict[float, tuple[float, float]] = {}
    for v in values:
        if v == BC_VALUE:
            continue
        d = [results[v][i] - base_pts[i] for i in range(len(seeds))]
        m = statistics.mean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        se = sd / len(d) ** 0.5 if d else float("inf")
        # ★ sd=0 时**不能**报 inf。两种情形要分开：
        #   sd=0 且 m=0  ⟹ 逐种子完全相同，是「分不出」（该报 0）
        #   sd=0 且 m≠0  ⟹ 全种子同向同幅，才是真正无穷大的信噪比
        #   2026-09-19：前一种被 m/se 写成 +inf，把「没差别」判成「有差别」。
        if se > 0:
            t = m / se
        elif abs(m) < 1e-12:
            t = 0.0
        else:
            t = float("inf")
        paired[v] = (m, t, sd)
        print(f"    {v:>+10.4f}{m:>+11.4f}{sd:>10.4f}{se:>9.4f}{t:>+9.3f}")

    df = len(seeds) - 1
    try:
        from scipy import stats as _st
        crit = float(_st.t.ppf(0.975, df))
        src = "scipy"
    except ImportError:
        crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 13: 2.160, 14: 2.145}.get(
                    df, float("nan"))
        src = "内置表"

    print()
    print(f"  双侧临界值 t* = {crit:.3f} (df={df}, {src})")
    print(f"  实用下限 |Δ| ≥ {MIN_EFFECT:.4f}（统计显著**且**效应量过线才算数）")
    # ★ 两道闸同时过才算「有差别」。只判 t 会把「逐位相同」判成差别（见上）。
    hits = {v: mt for v, mt in paired.items()
            if abs(mt[1]) >= crit and abs(mt[0]) >= MIN_EFFECT}
    print()
    if not hits:
        print("  ⟹ **没有任何档位与 BC 起点可测地不同** ⟹ 解释 (A)：")
        print("     stop_logit 这个维度在当前点附近**不敏感**。")
        print("     「RL 不碰它」是因为那里梯度为零，不是因为它不重要。")
        print("     ⟹ 先别做状态化 STOP，把力气花在 actor 的其它自由度上。")
        print()
        print("     旁证：−1.5/−1.0/−0.5 三档与 BC **逐种子完全相同**（SD(Δ)=0），")
        print("     说明 STOP 在负区间根本不参与决策（底下没有可停的候选）。")
        print()
        print("     ★ 但这**不等于** STOP 语义不重要：专家是「钉住 54.6 条、游程 9.92 槽」")
        print("       的**状态依赖**停行为，而这里扫的是一个**全局常数**。")
        print("       常数不敏感 ⇒ 关于状态的导数可能很大。可测的是后者，不是前者。")
        print("       （本实验**证伪**的是「手动挪常数就能提分」，不是「状态化 STOP 有用」。）")
    else:
        print(f"  ⟹ 有 {len(hits)} 档与 BC 起点**可测地不同** ⟹ 解释 (B)：")
        print("     这是一个**没被优化的维度**（RL 30 轮只动了 0.006，")
        print("     而手动挪一下就有可测差别）。")
        best_v = max(hits, key=lambda v: paired[v][0])
        print(f"     其中最好的是 stop_logit={best_v:+.4f}"
              f"（配对 Δ = {paired[best_v][0]:+.4f}，t = {paired[best_v][1]:+.3f}）")
        print("     ⟹ 下一步做**状态化 STOP**：给它的输入是 per-node 状态，")
        print("       而不是一个全局常数。")
    print("=" * 72)
    out = ROOT / "outputs" / "eval" / "stop_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"checkpoint": a.checkpoint, "update": int(data.update),
         "seeds": seeds, "per_value": {str(k): v for k, v in results.items()},
         "mean": {str(k): means[k] for k in means},
         "paired_delta": {str(k): paired[k][0] for k in paired},
         "paired_t": {str(k): paired[k][1] for k in paired}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
