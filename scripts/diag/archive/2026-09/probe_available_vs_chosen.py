"""决定性问题：需求路径上那 41.7% 「从未被激活」的边，**能不能**被激活？

## 为什么这一条决定「RL 能不能赢」

实测（官方种子 100–114，240 步，专家，initial_level=0）：

    需求路径用到的不同边   48 条
    其中被激活过           28 条 (58.3%)
    从未被激活             20 条 (41.7%)      ← 就这一条决定了全部结论
    全网物理边             1,978 条
    覆盖全部 pending 路径所需边数  均值 10.9  最大 26
    实际激活边数                  均值 55.3  最大 65   ⟹ 端口预算**绰绰有余**（5 倍）

「从未被激活」有**两个互斥解释**，指向完全相反的下一步：

  E1 **不可用**（`window.available[0] == False`，即物理上不可见/速率低于门限）
     ⟹ 那些边**任何策略都激活不了** ⟹ 41.7% 是**环境的结构性质**
     ⟹ 「RL 赢不了专家」不是训练问题 ⟹ 该改的是**场景/链路生成**，不是模型

  E2 **可用但没被选中**（策略/专家的选边规则漏了）
     ⟹ **有可学的空间** ⟹ 该改的是**策略或奖励**（RL 真有机会赢）

判据就是 `available[0]` 的时间并集（`rate_provider.LazyEdgeWindows.available0`）。

## 附带量一条

被激活的需求路径边，**存量**是多少？与全网平均比。
若激活了但存量恒为 0 ⟹ 问题在**生成落点**而不是选边；
若存量正常 ⟹ 问题在**选边覆盖**。

## 纪律

- 只跑专家（不训练）⟹ 便宜，且与任何训练改动无关
- 两遍跑：第一遍收集需求路径边集，第二遍（同种子、确定性）对**完整边集**
  逐步 OR 可用性 ⟹ 避免"边在第 50 步才进集合、前 50 步的可用性没测到"的偏差
- 判据必须带**正对照**：随便挑一条已知可用的边，确认 `available0` 真的报 True

用法（远程务必 -u）：
    python3 -u probe_available_vs_chosen.py --seeds 100-103 --steps 240
"""
import importlib.util
import os
import statistics as st
import sys
from pathlib import Path

# ★ 服务器 /tmp 有 547 个 .py 会遮蔽标准库 ⟹ 先摘掉 sys.path[0] 再 import
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


def _rollout(env, obs, expert, steps):
    """跑一局，回调每步；返回 (到达, 服务)。"""
    A = S = 0.0
    done = False
    k = 0
    while not done and k < steps:
        acts, scores = expert.act(obs)
        obs, _r, term, trunc, info = env.step(acts, scores)
        A += float(info.get("arrived_keys", 0.0))
        S += float(info.get("served_keys", 0.0))
        k += 1
        yield k, info, env
        done = term or trunc
    # 收尾


def pass1(cfg, seed, start_seed, steps):
    """收集：需求路径边集 + 曾被激活的边集。"""
    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start_seed)
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    demand_edges = set()
    activated_ever = set()
    for _k, _info, env in _rollout(env, obs, expert, steps):
        for e in getattr(env, "last_activated_edges", []) or []:
            activated_ever.add(e)
        for req in env.requests.pending:
            path = env.routing.shortest_path(req.src_gs, req.dst_gs)
            if path:
                demand_edges.update(path)
    return demand_edges, activated_ever


def pass2(cfg, seed, start_seed, steps, demand_edges):
    """同种子重跑（确定性），对**完整**需求边集逐步 OR 可用性 + 采存量。"""
    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start_seed)
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    edges = sorted(demand_edges)
    ever_avail = {e: False for e in edges}
    # 只统计**被激活的**需求路径边的存量（用最大水位，避免抽到刚好被消耗完的时刻）
    lvl_max = {e: 0.0 for e in edges}
    activated_ever = set()
    all_lvl = []          # 全网所有边的水位（对照）
    ctrl_ok = 0           # 正对照：可用边数 > 0 的步数
    n_steps = 0
    for _k, _info, env in _rollout(env, obs, expert, steps):
        n_steps += 1
        for e in getattr(env, "last_activated_edges", []) or []:
            activated_ever.add(e)
        try:
            stt = env._build_state()
        except Exception:                                          # noqa: BLE001
            break
        wins = stt.edge_windows
        try:
            av = wins.available0(edges)
        except Exception:                                          # noqa: BLE001
            break
        for e, a in zip(edges, av):
            if bool(a):
                ever_avail[e] = True
        if bool(av.any()):
            ctrl_ok += 1
        for e in edges:
            l = env.qkp.get_level(e)
            if l > lvl_max[e]:
                lvl_max[e] = l
        all_lvl.append(sum(env.qkp.levels.values()))
    return ever_avail, lvl_max, activated_ever, ctrl_ok, n_steps, all_lvl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-103")
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])

    print("=" * 100)
    print("需求路径边的三分类：可用但没选 / 根本不可用 / 可选且选了")
    print("=" * 100)
    print(f"{'seed':>6}{'需求边':>8}{'可选且选':>10}{'可选但没选':>12}{'不可用':>9}"
          f"{'  SR':>9}{'正对照(可用边>0的步)':>24}")

    tot = {"need": 0, "chosen": 0, "avail_not": 0, "unavail": 0}
    lvl_chosen, lvl_unchosen = [], []
    ctrl_all_ok = True
    for s in seeds:
        de1, act1 = pass1(cfg, s, start0 + s, args.steps)
        av2, lvmax, act2, ctrl_ok, nsteps, all_lvl = pass2(
            cfg, s, start0 + s, args.steps, de1)
        # pass2 的激活集应当与 pass1 一致（确定性）；不一致就是不确定性，要报
        if act1 != act2:
            print(f"  ⚠ seed {s}: 两遍激活集不一致（{len(act1)} vs {len(act2)}）"
                  f" ⟹ 环境不确定，读数不可信")

        chosen = avail_not = unavail = 0
        for e in sorted(de1):
            if not av2.get(e, False):
                unavail += 1
            elif e in act2:
                chosen += 1
            else:
                avail_not += 1
        n = len(de1)
        tot["need"] += n
        tot["chosen"] += chosen
        tot["avail_not"] += avail_not
        tot["unavail"] += unavail
        if ctrl_ok == 0:
            ctrl_all_ok = False
        # SR
        env = build_env_from_config(cfg)
        obs = env.reset(seed=s, start_seed=start0 + s)
        ex = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
        A = S = 0.0
        for _k, info, env in _rollout(env, obs, ex, args.steps):
            A += float(info.get("arrived_keys", 0.0))
            S += float(info.get("served_keys", 0.0))
        print(f"{s:>6}{n:>8}{chosen:>10}{avail_not:>12}{unavail:>9}"
              f"{S/A if A else 0:>9.4f}{f'{ctrl_ok}/{nsteps}':>24}")
        for e in sorted(de1):
            if e in act2:
                lvl_chosen.append(lvmax[e])
            elif av2.get(e, False):
                lvl_unchosen.append(lvmax[e])

    n = tot["need"]
    print()
    print("=" * 100)
    print("判读")
    print("=" * 100)
    print(f"  需求路径边合计 {n}")
    print(f"    **可选且已被选**  : {tot['chosen']:>4}  ({tot['chosen']/n:>6.1%})")
    print(f"    **可选但没被选**  : {tot['avail_not']:>4}  ({tot['avail_not']/n:>6.1%})"
          f"   ⟹ E2：策略有空间")
    print(f"    **根本不可用**    : {tot['unavail']:>4}  ({tot['unavail']/n:>6.1%})"
          f"   ⟹ E1：谁都做不到")
    print()
    print(f"  正对照（`available0` 在需求边集上至少报一条 True 的步数>0）："
          f"{'✓ 有' if ctrl_all_ok else '✗ 全 False ⟹ 判据空真，读数不可信'}")
    print()
    if lvl_chosen:
        print(f"  已被选的需求路径边，水位最大值 均值 {st.mean(lvl_chosen):,.0f}"
              f"  中位 {st.median(lvl_chosen):,.0f}")
    if lvl_unchosen:
        print(f"  可选但没选的需求路径边，水位最大值 均值 {st.mean(lvl_unchosen):,.0f}"
              f"  中位 {st.median(lvl_unchosen):,.0f}")
    print()
    print("  结论怎么读：")
    print("   · 若「根本不可用」占大头 ⟹ **E1**：那些边谁也激活不了，")
    print("     41.7% 是**环境的结构性质**，不是策略漏选 ⟹ 「RL 打不过专家」")
    print("     不是训练问题，该改的是**场景/链路生成**（可用性），不是模型")
    print("   · 若「可选但没被选」占大头 ⟹ **E2**：有可学空间，改策略/奖励")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
