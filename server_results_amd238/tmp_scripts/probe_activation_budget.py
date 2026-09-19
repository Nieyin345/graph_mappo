"""探针 O：预置库存"负担得起"吗？现在的策略到底有没有在钉住边？

## 背景（探针 N 修正版的结论）

探针 N 量出：C 桶（曾全通却没搬完）里 **2.05% 的到达量**属于"库存不够但物理够"
—— 把路径各跳**持续激活约 5 个槽**就能攒够，而它们有 13.7 个全通槽。
所以可争的那一格落在「**没预置**」这一侧。但还有两件事没量：

1. **预算**：`env._generate_keys` 只给**被激活的边**生成 key，而每槽能激活多少条边受
   双端口匹配约束（每节点 1 个发、1 个收，且 u->v 与 v->u 不能共存）→ 上界 = 节点数 90。
   一条路径要占掉**等于其跳数**的边。那么"同时在途的请求路径总跳数"和 90 比，
   是富余还是捉襟见肘？
2. **行为**：现在的策略有没有真的"钉住"边？如果绝大多数激活只持续 1 槽，
   那就是**纯反应式**，预置根本没发生 —— 这与探针 N 的 `bn@到达=0` 互为印证。

## 记什么（逐槽，与策略无关的量 + 策略的行为量）

    act        该槽实际激活的边数（= 上一槽 `env.last_activated_edges`）
    keep       act 与上一槽的交集（"钉住"的边）
    sw         act - keep（新切的边，也是 switch_count）
    live       该槽仍在途的请求数（`requests.get_pending()`）
    dem        在途请求的"路径跳数"之和（每槽可用边子图上的 BFS 最短路；不可达的跳过）
    unr        在途但**当槽物理不可达**的请求数（这些想要边也没法要）
    avail      该槽可用的边数

    表 1 每槽均值；表 2 在途数的分布 + dem/90 的比值；表 3 边激活游程长度分布。

用法（本机或节点）：
    python -u .tmp/probe_activation_budget.py --policy expert --seeds 7-18 --steps 240 --tag expert
    python -u .tmp/probe_activation_budget.py --policy rl --checkpoint outputs/xxx.pt --deterministic
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict, deque
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--policy", default="expert", choices=["expert", "rl"])
    ap.add_argument("--seeds", default="7-18")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--tag", default="expert")
    ap.add_argument("--config", default=str(ROOT / "configs" / "var2_diag.yaml"))
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    profile = _tp.load_validation_profile(Path(args.config))
    steps = args.steps or int(profile["episode_steps"]) or 240
    seeds = parse_seeds(args.seeds)
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])
    start_seed = int(profile.get("start_seed", 0))

    act_mode = ("确定性" if (args.policy == "expert" or args.deterministic) else "采样")
    print(f"策略={args.policy}  动作={act_mode}  种子={seeds}  步数={steps}")

    slots: list[dict] = []
    run_hist: dict[int, int] = defaultdict(int)   # 边激活游程长度 -> 次数
    tot_arrived = tot_served = 0.0

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

        n_nodes = len({n for e in env.routing.edges for n in (e.src, e.dst)})
        rp = env.rate_provider
        adj = env.routing.adj

        def hop_len(src: str, dst: str, t: int) -> int:
            """该槽可用边子图上 src->dst 的跳数（不可达返回 -1）。"""
            if src == dst:
                return 0
            seen = {src}
            dq = deque([(src, 0)])
            while dq:
                u, d = dq.popleft()
                for nb, eid in adj.get(u, ()):
                    if nb in seen or not rp.is_available(eid, t):
                        continue
                    if nb == dst:
                        return d + 1
                    seen.add(nb)
                    dq.append((nb, d + 1))
            return -1

        hop_cache: dict[tuple[str, str, int], int] = {}

        def cached_hop(src: str, dst: str, t: int) -> int:
            key = (src, dst, t)
            v = hop_cache.get(key)
            if v is None:
                v = hop_len(src, dst, t)
                hop_cache[key] = v
            return v

        obs = env.reset(seed=seed, start_seed=start_seed + seed)
        prev: set[str] = set()
        runs: dict[str, int] = {}
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
            t = n
            n += 1
            done = term or trunc

            cur = set(env.last_activated_edges)
            keep = len(cur & prev)
            # 边激活游程：连续被激活的槽数。
            for e in cur:
                runs[e] = runs.get(e, 0) + 1 if e in prev else 1
            for e in list(prev - cur):
                run_hist[runs.pop(e)] += 1
            prev = cur

            live = env.requests.get_pending()
            dem = unr = 0
            for req in live:
                h = cached_hop(req.src_gs, req.dst_gs, t)
                if h < 0:
                    unr += 1
                else:
                    dem += h
            avail = sum(1 for e in env.routing.edges if rp.is_available(e.edge_id, t))
            slots.append({
                "seed": seed, "t": t, "act": len(cur), "keep": keep,
                "sw": len(cur) - keep, "live": len(live), "dem": dem,
                "unr": unr, "avail": avail, "n_nodes": n_nodes,
            })
        for e, r in runs.items():
            run_hist[r] += 1

        s = env.metrics.episode_summary()
        tot_arrived += s["arrived_keys"]
        tot_served += s["served_keys"]
        print(f"  seed={seed} 成功率={s['success_rate']:.4f}")

    def mean(vals) -> float:
        vals = list(vals)
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\n量加权成功率={tot_served / max(1e-9, tot_arrived):.4f}"
          f"  槽数={len(slots)}  节点数={slots[0]['n_nodes']}")

    print("\n【表 1】每槽均值（专家 = 纯反应式的参照）")
    print(f"  {'act 激活边':>12}{'keep 钉住':>12}{'sw 新切':>10}{'live 在途':>12}"
          f"{'dem 需求跳数':>14}{'unr 不可达':>12}{'avail 可用边':>14}{'dem/90':>10}")
    print(f"  {mean(s['act'] for s in slots):>12.1f}{mean(s['keep'] for s in slots):>12.1f}"
          f"{mean(s['sw'] for s in slots):>10.1f}{mean(s['live'] for s in slots):>12.1f}"
          f"{mean(s['dem'] for s in slots):>14.1f}{mean(s['unr'] for s in slots):>12.1f}"
          f"{mean(s['avail'] for s in slots):>14.1f}"
          f"{mean(s['dem'] for s in slots) / slots[0]['n_nodes']:>10.2f}")

    print("\n【表 2】在途请求数分布，以及「需求边数 / 激活预算(=节点数)」的比值")
    print(f"  {'在途数区间':<14}{'槽数':>8}{'占比':>10}{'平均 dem':>12}{'dem/预算':>12}")
    bins = ((0, 0), (1, 3), (4, 6), (7, 10), (11, 15), (16, 25), (26, 10 ** 9))
    for lo, hi in bins:
        sel = [s for s in slots if lo <= s["live"] <= hi]
        if not sel:
            continue
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f">={lo}")
        print(f"  {label:<14}{len(sel):>8}{len(sel) / len(slots):>10.2%}"
              f"{mean(s['dem'] for s in sel):>12.1f}"
              f"{mean(s['dem'] for s in sel) / slots[0]['n_nodes']:>12.2f}")
    over = [s for s in slots if s["dem"] > s["n_nodes"]]
    print(f"  → 需求 > 预算（dem > 节点数）的槽：{len(over)}/{len(slots)} = "
          f"{len(over) / len(slots):.2%}")

    print("\n【表 3】边激活游程长度（同一条边连续被激活的槽数）分布")
    tot_runs = sum(run_hist.values())
    print(f"  {'游程(槽)':<12}{'次数':>10}{'占比':>10}")
    order = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]
    for lo, hi in order:
        k = sum(c for r, c in run_hist.items() if lo <= r <= hi)
        if k == 0:
            continue
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f">={lo}")
        print(f"  {label:<12}{k:>10}{k / max(1, tot_runs):>10.2%}")
    avg_run = (sum(r * c for r, c in run_hist.items()) / max(1, tot_runs))
    print(f"  平均游程={avg_run:.2f} 槽  总游程数={tot_runs}")
    print("  参照：探针 N 说「物理够」的那批要攒够约需连续激活 5 槽。")

    out = ROOT / "outputs" / "eval" / f"activation_budget_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "policy": args.policy, "deterministic": bool(args.deterministic),
        "checkpoint": args.checkpoint, "seeds": seeds, "steps": steps,
        "arrived": tot_arrived, "served": tot_served,
        "slots": slots, "run_hist": {str(k): v for k, v in run_hist.items()},
    }, open(out, "w", encoding="utf-8"))
    print(f"\n已保存 {out}")
    print("ACTBUDGET_DONE")


if __name__ == "__main__":
    main()