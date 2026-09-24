"""两件事一起量：① 天花板在**官方验证种子 100–114** 上是多少；
② 240 步里冷启动斜坡的机理（服务量为什么头 20 步几乎是 0）。

## ① 为什么要重测天花板

文档记的天花板 **0.9197** 是在 `probe_ceiling_clean.py` 里量的，那个脚本用
`seeds = 7-21`（`eval_expert.py` 的默认）。而正式验证协议用的是
`train_full_rl.yaml` 的 **request_seeds = [100..114]**。

同一批种子，专家实测 0.6979（100–114）vs 0.7140（7–21）——**两个不同的种子集**。
所以天花板也必须用 100–114 重算，才能与 0.6979 同口径比较。

★ 这个天花板**不用任何策略**：需求流外生（`request_generator.generate(t)`），
   可服务性是纯拓扑（`shortest_path is None`）。所以它**与训练无关**，结论对
   改动后的模型同样成立。（[[ab-pair-must-share-the-code-not-just-the-config]] 的教训：
   但**这只对天花板成立**，策略读数必须固定 regime 重测。）

## ② 为什么要量冷启动机理

P2 实测（seed 7，专家，h0=20）：240 步 SR 0.2741 → 1440 步 0.8196。
分段轨迹显示头 20 步只服务了 **1,320** 条密钥（到达 1,833,692）。
⟹ 「起局那个洞」值很多分，但**不知道为什么**第一个密钥要等那么久。

三个互斥假说，各指向完全不同的修法：
  H1 **生成速率受限** —— 每步能生成的密钥量有上限，池子要若干步才够
  H2 **激活滞后** —— 策略要先决定激活哪些边，头几步没选对
  H3 **路由/池子分配受限** —— 密钥生成在 A 边，需求落在 B 边，搬不过去

判据：看**每步的 generated 与 qkp 总水位**。
  · H1：generated 恒为正且接近某个上限，水位单调涨但很慢
  · H2：头几步 n_activated 低，之后才升上来
  · H3：generated 很高、水位很高，但 served 仍低 ⟹ 池子有货却服务不了

用法（远程务必 -u）：
    python3 -u probe_ceiling_official_seeds.py
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


def main():
    ap = argparse.ArgumentParser()
    # ★ 默认用**官方验证种子**，不是 eval_expert.py 的 7-21
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--traj-seed", type=int, default=100)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    _here_s = sys.path[0] if sys.path else ""

    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])

    print("=" * 104)
    print(f"① 天花板（官方验证种子 {seeds[0]}–{seeds[-1]}，{args.steps} 步，"
          f"start_seed0={start0}）")
    print("=" * 104)

    # ---- 拓扑：哪些 GS 对不可服务 ----
    e0 = build_env_from_config(cfg)
    routing = e0.routing
    gss = sorted(n.node_id for n in e0.scenario.nodes
                 if n.node_type.value == "gs")
    print(f"  [拓扑] GS 数 = {len(gss)}")

    import itertools
    unservable = set()
    reachable = []
    for a, b in itertools.combinations(gss, 2):
        if routing.shortest_path(a, b) is None:
            unservable.add(tuple(sorted((a, b))))
        else:
            reachable.append(tuple(sorted((a, b))))
    pairs = list(itertools.combinations(gss, 2))
    print(f"         可服务对 {len(reachable)} / {len(pairs)}；"
          f"不可服务 {len(unservable)}（{len(unservable)/len(pairs):.4%}）")
    if reachable:
        pr = reachable[len(reachable) // 2]
        sp = routing.shortest_path(*pr)
        print(f"  [正对照] 连通对 {pr} 最短路 = {len(sp) if sp else None}（须非 None）")

    # ---- 需求：外生，逐种子 ----
    tot = bad = 0.0
    per_seed = []
    for seed in seeds:
        e = build_env_from_config(cfg)
        e.reset(seed=seed, start_seed=start0 + seed)
        t0 = int(e.t)
        s = b = 0
        for k in range(args.steps):
            for req in e.request_generator.generate(t0 + k):
                key = tuple(sorted((req.src_gs, req.dst_gs)))
                s += req.amount
                if key in unservable:
                    b += req.amount
        tot += s
        bad += b
        per_seed.append((seed, s, b))

    print(f"\n  [需求] 外生到达总量 {tot:,.0f}")
    print(f"         落在不可服务对上 {bad:,.0f}")
    print(f"         按**密钥量**占 {bad/tot:.4%}   ⟹ 天花板 = **{1 - bad/tot:.4%}**")
    print(f"         按对数占 {len(unservable)/len(pairs):.4%}")
    print()
    print(f"  {'seed':>6}{'到达':>16}{'不可服务':>16}{'该种子占比':>12}")
    for seed, s, b in per_seed:
        print(f"  {seed:>6}{s:>16,.0f}{b:>16,.0f}{(b/s if s else 0):>11.3%}")
    sh = [b / s for _, s, b in per_seed if s]
    print(f"\n  逐种子占比：均值 {st.mean(sh):.4%}  SD {st.pstdev(sh):.4%}  "
          f"极差 {min(sh):.3%}..{max(sh):.3%}")

    # ---------- ② 冷启动机理 ----------
    print()
    print("=" * 104)
    print(f"② 冷启动斜坡机理（专家，seed {args.traj_seed}，前 30 步逐步）")
    print("=" * 104)
    e = build_env_from_config(cfg)
    obs = e.reset(seed=args.traj_seed, start_seed=start0 + args.traj_seed)
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(e))
    print(f"  {'步':>4}{'到达':>13}{'服务':>13}{'生成':>13}"
          f"{'池总水位':>16}{'激活边':>8}{'等待':>13}{'桶SR':>9}")
    A = S = 0.0
    for k in range(args.steps):
        acts, scores = expert.act(obs)
        obs, _r, term, trunc, info = e.step(acts, scores)
        a = float(info.get("arrived_keys", 0.0))
        s = float(info.get("served_keys", 0.0))
        g = float(info.get("generated_keys", 0.0))
        A += a
        S += s
        if k < 30:
            lvl = sum(e.qkp.levels.values())
            print(f"  {k:>4}{a:>13,.0f}{s:>13,.0f}{g:>13,.0f}{lvl:>16,.0f}"
                  f"{len(e.last_activated_edges):>8}{float(info.get('waiting_keys',0)):>13,.0f}"
                  f"{(s/a if a else 0):>9.4f}")
        if term or trunc:
            break
    print(f"\n  整局 {args.steps} 步：到达 {A:,.0f}  服务 {S:,.0f}  SR = {S/A if A else 0:.4f}")

    print()
    print("  判读：")
    print("   · H1 生成受限：`生成` 每步都贴近某个上限，水位单调但缓慢上涨")
    print("   · H2 激活滞后：头几步 `激活边` 明显偏低，之后才升")
    print("   · H3 池子有货服务不了：`生成` 与 `池总水位` 都高，但 `服务` 仍接近 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
