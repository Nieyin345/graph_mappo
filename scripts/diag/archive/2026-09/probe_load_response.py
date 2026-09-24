"""找「调度开始起作用」的工作点 —— 扫需求强度，量专家成功率怎么掉。

## 为什么必须做这一条

用户的诉求是「**RL 要超过启发式**」。但在**当前负载**下这件事**没有空间**：

    实测（官方种子 100–114，240 步，专家）
      到达 85,300 条/步      服务 73,000 条/步      ⟹ **约 14% 余量**

系统有余量时，**任何**合理策略都能把可服务的需求服务掉 ⟹ 排得好不好**无所谓**。
这正是 RL − 专家 ≈ 0 的机理（实测 Δ=+0.0052 / −0.0003，都测不出）。
**一个专家已经贴在天花板上的基准，没有判别力。**

调度只有在**系统被压紧**时才值钱：那时「先服务谁 / 密钥往哪条边放 / 要不要为
未来预留」才开始有对错之分，而贪心的专家会开始犯错。

## 这一条量什么

扫 `requests.amount_mean`（需求强度），固定其余一切，量专家策略的 SR：
    · 若 SR 在某个负载后**开始明显下滑** ⟹ 那里就是「调度起作用」的门槛
    · 若一直到很重的负载都不掉 ⟹ **供给根本没被压到**，问题在供给侧不在调度侧

同时报**每步到达 / 服务 / 容量余量**，用来判断压紧的是哪一环。

## 纪律

- 只跑**专家**，不训练 ⟹ 便宜（60 局），且**与任何训练改动无关**
- 逐种子配对：同一批种子在所有负载档上跑，看**每档相对基线**的变化
- 负载是**环境/机制**旋钮，不是调参（符合「结构→机制→奖励→策略→调参」的顺序）

用法（远程务必 -u）：
    python3 -u probe_load_response.py --means 100000,130000,160000,200000
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


def run(config, seed, start_seed, steps):
    env = build_env_from_config(config)
    obs = env.reset(seed=seed, start_seed=start_seed)
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    A = S = F = G = 0.0
    W = 0.0
    n = 0
    done = False
    while not done:
        acts, scores = expert.act(obs)
        obs, _r, term, trunc, info = env.step(acts, scores)
        A += float(info.get("arrived_keys", 0.0))
        S += float(info.get("served_keys", 0.0))
        F += float(info.get("failed_keys", 0.0))
        G += float(info.get("generated_keys", 0.0))
        W += float(info.get("waiting_keys", 0.0))
        n += 1
        done = term or trunc
    return {"sr": S / A if A else 0.0, "A": A, "S": S, "F": F, "G": G,
            "W_mean": W / n if n else 0.0, "n": n,
            "arr_rate": A / n if n else 0.0, "srv_rate": S / n if n else 0.0,
            "gen_rate": G / n if n else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--means", default="100000,130000,160000,200000")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    means = [float(x) for x in args.means.split(",")]

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))

    def build(amount_mean):
        cfg = tp.build_validation_env_config(
            profile, include_baselines=False, episode_steps=args.steps,
            start_mode=profile["start_mode"])
        cfg = dict(cfg)
        cfg["requests"] = dict(cfg.get("requests", {}))
        cfg["requests"]["amount_mean"] = float(amount_mean)
        # amount_max 要保持是均值的 5 倍（原配置口径），否则分布形状变了，
        # 那就同时动了两个东西，读不出是哪一个在起作用。
        cfg["requests"]["amount_max"] = float(amount_mean) * 5.0
        return cfg

    base = build(means[0])
    print("=" * 108)
    print(f"扫需求强度 `requests.amount_mean`（专家，{len(seeds)} 种子 × {args.steps} 步）")
    print("=" * 108)
    print(f"  基线口径：arrival_rate={base['requests'].get('arrival_rate')}  "
          f"deadline_steps={base['requests'].get('deadline_steps')}  "
          f"qkp.initial_level={base.get('qkp', {}).get('initial_level')}")
    print()
    print(f"{'amount_mean':>13}{'SR均值':>10}{'SD':>9}"
          f"{'到达/步':>12}{'服务/步':>12}{'生成/步':>14}{'余量%':>9}{'等待均值':>14}")
    res = {}
    for am in means:
        cfg = build(am)
        rows = [run(cfg, s, start0 + s, args.steps) for s in seeds]
        srs = [r["sr"] for r in rows]
        arr = st.mean([r["arr_rate"] for r in rows])
        srv = st.mean([r["srv_rate"] for r in rows])
        gen = st.mean([r["gen_rate"] for r in rows])
        marg = (1 - srv / arr) * 100 if arr else 0.0
        res[am] = {"srs": srs, "arr": arr, "srv": srv, "gen": gen}
        print(f"{am:>13,.0f}{st.mean(srs):>10.4f}{st.pstdev(srs):>9.4f}"
              f"{arr:>12,.0f}{srv:>12,.0f}{gen:>14,.0f}{marg:>8.1f}%"
              f"{st.mean([r['W_mean'] for r in rows]):>14,.0f}")

    print()
    print("=" * 108)
    print("逐档相对基线（逐种子配对）—— 判「压紧了没有」")
    print("=" * 108)
    b = res[means[0]]["srs"]
    for am in means:
        d = [x - y for x, y in zip(res[am]["srs"], b)]
        m = st.mean(d)
        sd = st.stdev(d) if len(d) > 1 else 0.0
        se = sd / len(d) ** 0.5 if d else 0.0
        t = m / se if se else float("nan")
        print(f"  amount_mean {am:>10,.0f}:  ΔSR = {m:+.4f}   "
              f"SD {sd:.4f}   t = {t:+.2f}   SR {st.mean(res[am]['srs']):.4f}")

    print()
    print("  判读：")
    print("   · 若 SR 在某档**断崖式下滑** ⟹ 那里就是调度开始起作用的负载")
    print("     ⟹ 把训练/验证负载设在那附近，RL 才有超越贪心专家的空间")
    print("   · 若 SR 一路平 ⟹ 供给没被压到，**再调奖励/结构也没用**，")
    print("     得先动供给侧（生成量/容量）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
