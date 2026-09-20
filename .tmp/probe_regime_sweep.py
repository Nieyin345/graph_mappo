"""**机制扫描**：找到"冷启动瞬态已消除、但还没饱和"的那个键位。

## 为什么这是当前最高价值的一步

实测链条（官方种子 100–114，240 步）：

  ① 冷启动时**前 14 步服务量恒为 0**，而池子已堆到 1.96e8 条密钥
  ② `initial_level = 0`      ⟹ 专家 0.6979 / RL 0.7031   Δ=+0.0052 测不出
     `initial_level = 1e6`    ⟹ 专家 0.9176 / RL 0.9173   Δ=-0.0003 测不出
  ③ 拓扑天花板 0.9197；冷启动口径的结构上限 0.8645
  ④ 负载扫描：需求翻倍 SR 只掉 6 点，**没有饱和崖**；生成量是需求的 **153×**

⟹ **最大的单一杠杆是"池子起点"**（0 → 1e6 值 +22 点），远大于任何模型改动。
   而 `1e6` 处已经**饱和**（0.9176 ≈ 拓扑天花板 0.9197）⟹ 没有判别力。
   真正的键位在**两者之间**：冷启动瞬态已消除、但还没顶到天花板。

## 这一条做什么

只跑**专家**（不训练）⟹ 便宜（7 档 × 15 种子 × 240 步），且**与任何训练改动无关**。
一次一个因子（one-factor-at-a-time）：

  A. `qkp.initial_level`：0 / 1e4 / 3e4 / 1e5 / 3e5 / 1e6
  B. `requests.deadline_steps`：30（基线）/ 60 / 120
     —— 依据：池子要 **14 步**才凑齐第一条可用路径，而请求只活 **30 步**
        ⟹ 起局那批请求的存活期与预热期**重叠**，是第二个候补机制。

逐种子配对（同种子比同种子），报 Δ 与 t（df=14，临界 2.145）。

## 标出的决策规则（先写死，避免事后挑）

  DECISION = 满足「专家 SR ≤ 0.88」的**最大** initial_level
             （0.88 留出约 4 点余量在天花板 0.9197 之下）
  若无满足者 ⟹ 退到 0（保持现 regime）

用法（远程务必 -u）：
    python3 -u probe_regime_sweep.py --seeds 100-114 --steps 240
"""
import importlib.util
import os
import statistics as st
import sys
from pathlib import Path

_here = sys.path[0] if sys.path else ""
if _here and _here not in ("", "."):
    sys.path[:] = [p for p in sys.path if p != _here]

REPO = Path(os.environ.get("GM_REPO", "/opt/qkd/graph_mappo"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse                                                   # noqa: E402
from qkd_rl.env.factory import build_env_from_config              # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy          # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe               # noqa: E402

T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
          13: 2.160, 14: 2.145}

SATURATION_BAR = 0.88


def _tp():
    spec = importlib.util.spec_from_file_location(
        "gm_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def expert_sr(cfg, seed, start_seed, steps):
    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start_seed)
    ex = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                         principles=False, router=ServeProbe(env))
    A = S = G = 0.0
    n = 0
    done = False
    while not done and n < steps:
        acts, scores = ex.act(obs)
        obs, _r, term, trunc, info = env.step(acts, scores)
        A += float(info.get("arrived_keys", 0.0))
        S += float(info.get("served_keys", 0.0))
        G += float(info.get("generated_keys", 0.0))
        n += 1
        done = term or trunc
    return (S / A if A else 0.0), A, S, (G / n if n else 0.0)


def paired(base, exp, label):
    d = [a - b for a, b in zip(exp, base)]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    se = sd / n ** 0.5 if n > 1 else 0.0
    t = m / se if se else float("nan")
    crit = T_CRIT.get(n - 1)
    if crit is None:
        raise SystemExit(f"df={n-1} 无临界值表项 —— 不猜，直接停")
    sig = "**可分辨**" if abs(t) > crit else "测不出"
    same = sum(1 for x in d if x * m > 0)
    print(f"    {label:<28} ΔSR {m:+.4f}  SD {sd:.4f}  t {t:+.2f}  "
          f"(df={n-1} 临界 {crit})  {same}/{n} 同向  ⟹ {sig}")
    return m, t, crit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))

    def build(**over):
        cfg = tp.build_validation_env_config(
            profile, include_baselines=False, episode_steps=args.steps,
            start_mode=profile["start_mode"])
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
        if "il" in over:
            cfg.setdefault("qkp", {})
            cfg["qkp"] = dict(cfg["qkp"])
            cfg["qkp"]["initial_level"] = float(over["il"])
        if "dl" in over:
            cfg.setdefault("requests", {})
            cfg["requests"] = dict(cfg["requests"])
            cfg["requests"]["deadline_steps"] = int(over["dl"])
        return cfg

    base_cfg = build()
    print("=" * 100)
    print(f"机制扫描（**只跑专家**，种子 {seeds[0]}–{seeds[-1]}，{args.steps} 步）")
    print(f"基线：initial_level={base_cfg.get('qkp', {}).get('initial_level')}  "
          f"deadline_steps={base_cfg.get('requests', {}).get('deadline_steps')}  "
          f"amount_mean={base_cfg.get('requests', {}).get('amount_mean')}")
    print("=" * 100)

    base_sr = []
    for s in seeds:
        r, _a, _sv, _g = expert_sr(base_cfg, s, start0 + s, args.steps)
        base_sr.append(r)
    print(f"  基线 SR = {st.mean(base_sr):.4f}  SD {st.pstdev(base_sr):.4f}")
    print()

    print("  A. `qkp.initial_level`（池子起点）")
    results = {}
    for il in [0, 1e4, 3e4, 1e5, 3e5, 1e6]:
        cfg = build(il=il)
        srs = []
        for s in seeds:
            r, _a, _sv, _g = expert_sr(cfg, s, start0 + s, args.steps)
            srs.append(r)
        results[il] = srs
        m = st.mean(srs)
        tag = "（基线）" if il == 0 else ""
        print(f"    initial_level {il:>9,.0f}:  SR {m:.4f}  SD {st.pstdev(srs):.4f}{tag}")
        if il != 0:
            paired(base_sr, srs, f"  vs 基线 il={il:,.0f}")
    print()

    print("  B. `requests.deadline_steps`（请求存活期）")
    for dl in [60, 120]:
        cfg = build(dl=dl)
        srs = []
        for s in seeds:
            r, _a, _sv, _g = expert_sr(cfg, s, start0 + s, args.steps)
            srs.append(r)
        print(f"    deadline_steps {dl:>4}:  SR {st.mean(srs):.4f}  SD {st.pstdev(srs):.4f}")
        paired(base_sr, srs, f"  vs 基线 dl=30")

    # ---------- 决策（规则先写死，见 docstring）----------
    print()
    print("=" * 100)
    print("决策（规则预注册：满足 SR ≤ 0.88 的**最大** initial_level）")
    print("=" * 100)
    ok = [(il, st.mean(v)) for il, v in results.items() if st.mean(v) <= SATURATION_BAR]
    if ok:
        pick_il, pick_sr = max(ok, key=lambda x: x[0])
        print(f"  候选（SR ≤ {SATURATION_BAR}）：")
        for il, m in sorted(ok):
            print(f"    initial_level {il:>9,.0f}  SR {m:.4f}")
        print(f"  ⟹ 选 **initial_level = {pick_il:,.0f}**（SR {pick_sr:.4f}，"
              f"距天花板 0.9197 还有 {0.9197 - pick_sr:.4f} 余量）")
    else:
        pick_il, pick_sr = 0.0, st.mean(base_sr)
        print(f"  ⚠ 全部档位都超过 {SATURATION_BAR} ⟹ 退回基线 0")
    print(f"DECISION_INITIAL_LEVEL={pick_il:.0f}")
    print(f"DECISION_EXPERT_SR={pick_sr:.6f}")
    print(f"BASELINE_EXPERT_SR={st.mean(base_sr):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
