"""探针 H：失败请求的源-目的，在它**活着的那些槽里，到底有没有过"一路全通"**。

## 为什么换成这个口径

探针 G 量的是"物理速率够不够"（每条路径各跳的生命期生成量之和 ≥ 剩余量）。
但真正卡住一条请求的是另一件事：**路径上每一跳必须同时可用**。

这台环境里，一个槽能接哪些边由 `rate_provider.is_available(edge, t)` 决定
（GS↔LEO 卫星链路是分时通断的），而请求只有 30 个槽的寿命。所以"边在拓扑上
存在"和"边在这一刻通"是两回事 —— 探针 E 用 `hop_distance`（静态拓扑）判 A，
把这些"拓扑上通、时间上不通"的请求全都算进了"可达却没搬完"。

这个探针把那一层剥开：对每个槽 t，在"当时可用的边"构成的子图上算连通分量，
于是"src 与 dst 在 t 时刻是否存在一条全通的路径"就变成一次分量号比较 ——
**精确**，不需要枚举路径，也不受路径长度限制。

分类（对每条过期请求，按优先级）：

  A      静态拓扑就不连通（`routing.hop_distance == 10**6`）—— 探针 E 的 A
  B_avail 有路径，但生命期 [arrival, deadline) 里**没有任何一个槽**做到全通
          → 任何策略都服务不了，这条需求从出生就注定失败
  C      存在过全通的槽，最终没搬完 → 归策略（调度 / 竞争）

## 和探针 G 的关系

G 给"容量不够"的下界，H 给"时间窗口不合拍"的下界。两条都应该跑：
一条请求可能既容量不够又窗口不通；H 的判读里要按"先 B_avail 再 C"的顺序看，
所以 H 报出来的 C 是**扣掉窗口问题之后剩下的**，比 G 的 C 更硬。

用法（节点上，驱动 `.tmp/run_joint.sh`）：
    bash .tmp/run_joint.sh
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402


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


def slots_components(env, horizon: int) -> tuple[np.ndarray, dict[str, int]]:
    """每个槽一份"可用边子图"的连通分量编号。

    返回 ``comp[t][node_idx]``，节点索引由 ``node_idx`` 给出。一次 O(horizon·(V+E))
    的扫描替代"每条请求枚举路径"—— 分量号相等就等价于"存在一条全通路径"，
    而且不限制路径长度。
    """
    edges = env.routing.edges
    node_ids = sorted({n for e in edges for n in (e.src, e.dst)})
    node_idx = {n: i for i, n in enumerate(node_ids)}
    comp = np.full((horizon, len(node_ids)), -1, dtype=np.int32)
    rp = env.rate_provider

    for t in range(horizon):
        adj: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            if rp.is_available(e.edge_id, t):
                adj[e.src].append(e.dst)
                adj[e.dst].append(e.src)
        row = comp[t]
        cid = 0
        for nd in node_ids:
            i = node_idx[nd]
            if row[i] >= 0:
                continue
            row[i] = cid
            stack = [nd]
            while stack:
                u = stack.pop()
                for v in adj.get(u, ()):
                    j = node_idx[v]
                    if row[j] < 0:
                        row[j] = cid
                        stack.append(v)
            cid += 1
    return comp, node_idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--policy", default="expert", choices=["expert", "rl"])
    ap.add_argument("--seeds", default="7-18")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--tag", default="", help="输出文件名后缀，避免覆盖别的 checkpoint 的结果")
    ap.add_argument("--config", default=str(ROOT / "configs" / "var2_diag.yaml"))
    # 训练侧的 `evaluate_validation` 是 deterministic=True，这里默认**采样**。
    # 两者量级不同（采样）且比不了训练日志里的验证数字，所以必须显式可选。
    ap.add_argument("--deterministic", action="store_true",
                    help="RL 策略用确定性动作（与训练侧 evaluate_validation 同口径）")
    args = ap.parse_args()

    profile = _tp.load_validation_profile(Path(args.config))
    steps = args.steps or int(profile["episode_steps"]) or 240
    seeds = parse_seeds(args.seeds)
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])
    start_seed = int(profile.get("start_seed", 0))
    horizon = steps + 40

    act_mode = ("确定性" if (args.policy == "expert" or args.deterministic) else "采样")
    print(f"策略={args.policy}  动作={act_mode}  种子={seeds}  步数={steps}  "
          f"窗={profile['window_start_day']}-{profile['window_end_day']}  "
          f"模式={profile['start_mode']}")

    tot_arrived = tot_served = 0.0
    rows: list[dict] = []
    checked = False

    for seed in seeds:
        env = build_env_from_config(config)
        if args.policy == "expert":
            policy = PathScoreGreedy(
                weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                principles=False, router=ServeProbe(env))
            is_expert = True
        else:
            model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
            data = load_checkpoint(args.checkpoint, device="cpu")
            model.load_state_dict(data.model_state)
            model.eval()
            policy = MAPPOPolicy(model, "cpu")
            is_expert = False

        comp, node_idx = slots_components(env, horizon)
        # 自检：可用性是"第 0 天"的确定性函数，与请求种子无关。第二个种子时
        # 重算一次并比对，不同就说明这个假设不成立，后面的判读全部作废。
        if not checked and len(seeds) >= 2:
            env2 = build_env_from_config(config)
            comp2, _ = slots_components(env2, horizon)
            same = np.array_equal(comp, comp2)
            print(f"  自检：可用性子图跨种子一致 = {same}"
                  f"{'' if same else '  ← 假设不成立，H 的判读作废'}")
            checked = True

        conn_cache: dict[tuple[str, str], np.ndarray] = {}

        def connected(src: str, dst: str) -> np.ndarray:
            key = (src, dst)
            arr = conn_cache.get(key)
            if arr is None:
                arr = comp[:, node_idx[src]] == comp[:, node_idx[dst]]
                conn_cache[key] = arr
            return arr

        obs = env.reset(seed=seed, start_seed=start_seed + seed)
        queue = env.requests
        orig_expire = queue.expire

        def expire_patched(t: int, _rows=rows, _env=env):
            out = orig_expire(t)
            for req in out:
                rem = max(0.0, float(req.amount) - float(req.served_amount))
                hops = int(_env.routing.hop_distance(req.src_gs, req.dst_gs))
                reachable = hops < 10 ** 6
                a, d = int(req.arrival_t), int(req.deadline_t)
                avail = 0
                if reachable:
                    arr = connected(req.src_gs, req.dst_gs)
                    avail = int(arr[a:min(d, horizon)].sum())
                _rows.append({
                    "seed": seed, "src": req.src_gs, "dst": req.dst_gs,
                    "amount": float(req.amount), "remaining": rem,
                    "served_frac": float(req.served_amount) / max(1e-9, float(req.amount)),
                    "hops": hops if reachable else -1,
                    "lifetime": d - a, "avail_slots": avail,
                })
            return out

        queue.expire = expire_patched
        n = 0
        done = False
        while n < steps and not done:
            if is_expert:
                actions, scores = policy.act(obs)
            else:
                with torch.no_grad():
                    step = policy.act(obs, deterministic=args.deterministic)
                actions, scores = step.actions, step.action_scores
            obs, _r, term, trunc, _info = env.step(actions, action_scores=scores)
            n += 1
            done = term or trunc
        queue.expire = orig_expire

        s = env.metrics.episode_summary()
        tot_arrived += s["arrived_keys"]
        tot_served += s["served_keys"]
        print(f"  seed={seed} 成功率={s['success_rate']:.4f}  过期请求={len(rows)}（累计）")

    unreach = [r for r in rows if r["hops"] < 0]
    reach = [r for r in rows if r["hops"] >= 0]
    never = [r for r in reach if r["avail_slots"] == 0]
    had = [r for r in reach if r["avail_slots"] > 0]

    def amt(g: list[dict]) -> float:
        return sum(r["remaining"] for r in g)

    print(f"\n量加权成功率={tot_served / max(1e-9, tot_arrived):.4f}"
          f"  到达={tot_arrived:,.0f}  服务={tot_served:,.0f}")
    print("\n过期请求分类（占**到达量**的比例）：")
    for name, g in (("A 拓扑不可达", unreach),
                    ("B_avail 生命期内从未全通", never),
                    ("C 曾全通却没搬完", had)):
        print(f"  {name:<26} n={len(g):>4}  剩余量={amt(g):>14,.0f}  "
              f"{amt(g) / max(1e-9, tot_arrived):>7.2%}  "
              f"平均完成度={sum(r['served_frac'] for r in g) / max(1, len(g)):.3f}")
    print(f"  {'过期总量':<26} n={len(rows):>4}  剩余量={amt(rows):>14,.0f}  "
          f"{amt(rows) / max(1e-9, tot_arrived):>7.2%}")

    print("\n可达且过期：按'生命期内全通槽数'分层（看完成度是否随它上升）")
    print(f"{'全通槽数':>10}{'请求数':>8}{'剩余量':>14}{'平均完成度':>12}")
    buckets = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]
    for lo, hi in buckets:
        g = [r for r in reach if lo <= r["avail_slots"] <= hi]
        if not g:
            continue
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}")
        print(f"{label:>10}{len(g):>8}{amt(g):>14,.0f}"
              f"{sum(r['served_frac'] for r in g) / len(g):>12.3f}")

    out = Path(f"outputs/eval/joint_avail_{args.policy}{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"policy": args.policy, "deterministic": bool(args.deterministic),
               "checkpoint": args.checkpoint, "seeds": seeds, "steps": steps,
               "arrived": tot_arrived, "served": tot_served, "rows": rows},
              out.open("w", encoding="utf-8"), indent=2)
    print(f"\n已保存 {out}")
    print("JOINT_AVAIL_DONE")


if __name__ == "__main__":
    main()