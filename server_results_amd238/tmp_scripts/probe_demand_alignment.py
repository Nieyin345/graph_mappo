"""探针 P：被激活的边，和"在途请求真正需要的那几条边"，重合多少？

## 为什么问这个（探针 O 的反常组合）

探针 O 量出专家：每槽激活 **60.8** 条边、其中 **54.6** 条是**从上一槽保持过来的**
（保持率 89.8%），边激活游程**平均 9.92 槽**；而在途请求的路径跳数加起来只有 **7.5**，
激活预算（节点数 89）**一次都没用满**。

也就是说：**预算远远够、边也一直在被长时间钉住** —— 但探针 N 同时测到 C 桶
86.66% 的到达量在源-目之间"连一条正库存路径都没有"。两句合起来只剩一个解释：

    钉住的**不是需求要的那些边**。

`env._generate_keys` 只给被激活的边生成 key，所以"货"被攒在 `act` 里；
而服务需要的是"某条路径的**每一跳**都为正"。若 `act` 与 `needs` 重合度低，
那么无论攒多少货、攒多久，请求都用不上。

## 记什么（逐槽）

    needs     在途每个请求"该走的路径"的边集合之并（每槽可用边子图上的 BFS 最短路）
    act       该槽实际激活的边集合（`env.last_activated_edges`）
    cov       重合度各项：|needs|、|needs ∩ act|、|act ∩ needs|、以及槽级覆盖率

    逐请求：H（该走几跳）、cov_hops（其中几跳被激活）、pos_hops（其中几跳有正库存）

用法：
    python -u .tmp/probe_demand_alignment.py --policy expert --seeds 7-18 --steps 240 --tag expert
    ... --policy rl --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt --deterministic
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
    reqs: list[dict] = []          # 逐请求的（H, cov_hops, pos_hops）
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

        rp = env.rate_provider
        adj = env.routing.adj
        qkp = env.qkp
        path_cache: dict[tuple[str, str, int], list[str] | None] = {}

        def path_edges(src: str, dst: str, t: int) -> list[str] | None:
            """该槽可用边子图上 src->dst 的最短路（跳数最少），返回边 id 列表。"""
            key = (src, dst, t)
            if key in path_cache:
                return path_cache[key]
            out: list[str] | None = None
            if src == dst:
                out = []
            else:
                parent: dict[str, tuple[str | None, str | None]] = {src: (None, None)}
                dq = deque([src])
                while dq and out is None:
                    u = dq.popleft()
                    for nb, eid in adj.get(u, ()):
                        if nb in parent or not rp.is_available(eid, t):
                            continue
                        parent[nb] = (u, eid)
                        if nb == dst:
                            p: list[str] = []
                            cur = dst
                            while parent[cur][0] is not None:
                                p.append(parent[cur][1])
                                cur = parent[cur][0]
                            out = p
                            break
                        dq.append(nb)
            path_cache[key] = out
            return out

        obs = env.reset(seed=seed, start_seed=start_seed + seed)
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

            act = set(env.last_activated_edges)
            needs: set[str] = set()
            live = 0
            unr = 0
            for req in env.requests.get_pending():
                pe = path_edges(req.src_gs, req.dst_gs, t)
                if pe is None:
                    unr += 1
                    continue
                live += 1
                needs.update(pe)
                cov = sum(1 for e in pe if e in act)
                pos = sum(1 for e in pe if qkp.get_level(e) > 1.0e-9)
                reqs.append({"seed": seed, "t": t, "H": len(pe),
                             "cov": cov, "pos": pos})
            slots.append({
                "seed": seed, "t": t, "act": len(act), "needs": len(needs),
                "inter": len(needs & act), "live": live, "unr": unr,
                "extra": len(act - needs),
            })
        s = env.metrics.episode_summary()
        tot_arrived += s["arrived_keys"]
        tot_served += s["served_keys"]
        print(f"  seed={seed} 成功率={s['success_rate']:.4f}")

    def mean(vals) -> float:
        vals = list(vals)
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\n量加权成功率={tot_served / max(1e-9, tot_arrived):.4f}"
          f"  槽数={len(slots)}  在途请求样本={len(reqs)}")

    print("\n【表 1】每槽：需求边集合 vs 实际激活边集合")
    print(f"  {'act 激活边':>12}{'needs 需要边':>14}{'重合':>10}{'覆盖率':>10}"
          f"{'act 里多余的边':>16}{'在途请求':>10}")
    print(f"  {mean(s['act'] for s in slots):>12.1f}"
          f"{mean(s['needs'] for s in slots):>14.1f}"
          f"{mean(s['inter'] for s in slots):>10.1f}"
          f"{mean(s['inter'] / max(1, s['needs']) for s in slots):>10.2%}"
          f"{mean(s['extra'] for s in slots):>16.1f}"
          f"{mean(s['live'] for s in slots):>10.1f}")

    print("\n【表 2】槽级覆盖率分布（needs 里有多大比例当槽真的被激活）")
    print(f"  {'覆盖率区间':<14}{'槽数':>8}{'占比':>10}{'平均 needs':>12}{'平均重合':>12}")
    for lo, hi in ((0.0, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        sel = [s for s in slots
               if (s["needs"] > 0)
               and (s["inter"] / s["needs"] > lo if lo > 0 else s["inter"] == 0)
               and (s["inter"] / s["needs"] <= hi)]
        if not sel:
            continue
        label = "覆盖率=0" if lo == hi == 0.0 else f"({lo:.0%},{hi:.0%}]"
        print(f"  {label:<14}{len(sel):>8}{len(sel) / len(slots):>10.2%}"
              f"{mean(s['needs'] for s in sel):>12.1f}"
              f"{mean(s['inter'] for s in sel):>12.1f}")

    print("\n【表 3】逐请求：该走几跳、其中几跳被激活、其中几跳有正库存")
    print(f"  {'路径被激活的比例':<18}{'请求数':>8}{'占比':>9}{'平均跳数':>10}"
          f"{'平均覆盖跳':>12}{'平均有库存跳':>14}")
    for label, pre in (("全部跳都激活", lambda r: r["cov"] == r["H"] and r["H"] > 0),
                       ("部分跳激活", lambda r: 0 < r["cov"] < r["H"]),
                       ("一跳都没激活", lambda r: r["cov"] == 0 and r["H"] > 0),
                       ("零跳路径(H=0)", lambda r: r["H"] == 0)):
        sel = [r for r in reqs if pre(r)]
        if not sel:
            continue
        print(f"  {label:<18}{len(sel):>8}{len(sel) / max(1, len(reqs)):>9.2%}"
              f"{mean(r['H'] for r in sel):>10.1f}"
              f"{mean(r['cov'] for r in sel):>12.1f}"
              f"{mean(r['pos'] for r in sel):>14.1f}")
    print("  → 若「一跳都没激活」占比很高，而「全部跳都激活」里也没几跳有货，")
    print("    则瓶颈不在攒得久不久，而在**激活的边根本不在需求的路径上**。")

    out = ROOT / "outputs" / "eval" / f"demand_alignment_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "policy": args.policy, "deterministic": bool(args.deterministic),
        "checkpoint": args.checkpoint, "seeds": seeds, "steps": steps,
        "arrived": tot_arrived, "served": tot_served,
        "slots": slots, "reqs": reqs,
    }, open(out, "w", encoding="utf-8"))
    print(f"\n已保存 {out}")
    print("ALIGN_DONE")


if __name__ == "__main__":
    main()