"""探针 Q：relay_importance 的"得分边集"，和需求真正要走的路径边集，重合多少？

## 为什么问这个（特征/奖励审计的落点）

审计发现三处共用同一个信号 `relay_importance`：
  1. 特征列 include_relay_importance（actor 看见的"这条边对需求多重要"）；
  2. dense 奖励 dense_generation_importance_weight=0.02（生成量 × importance）；
  3. **专家 PathScoreGreedy 的 dense_fill 第二遍**（空闲端口按 importance 排序填充，
     权重在排序键里排第 2 位）。

而 `compute_relay_importance` 只给 **total_hops = min(离src) + min(离dst) + 1 <= max_path_links(=3)**
的边打分。对诊断场景平均 6.2 跳的需求路径，路径中段任何边都有
min(离src) + min(离dst) ≈ H-1 >= 5 → total_hops ≈ H > 3 → **整条路径 0 分**；
只有"3 跳内的捷径边"（多数根本不在路径上）能得分。
若成立，三处信号合谋把生成引向"短中继捷径" —— 正是探针 P 量到的"钉错地方"。

## 记什么（与探针 P 同轨迹，专家档）

    imp        本槽策略看到的 relay_importance 非零边集（step 后 graph_builder.last_relay_importance）
    needs      在途请求 BFS 最短路的边集之并（可用性按 imp 同状态查询 env.t）
    表 1 每槽：|imp|、|needs|、|needs∩imp|、覆盖率
    表 2 逐(请求,槽)：H、路径上 importance>0 的跳数 imp_hops —— imp_hops=0 占比
    表 3 imp 得分边里有几条同时在需求路径上（错位方向：imp 里多少是"无需求"的）

用法：
    python -u .tmp/probe_relay_coverage.py --policy expert --seeds 7-18 --steps 240 --tag expert
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402


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
    ap.add_argument("--seeds", default="7-18")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--tag", default="expert")
    ap.add_argument("--config", default=str(ROOT / "configs" / "var2_diag.yaml"))
    args = ap.parse_args()

    profile = _tp.load_validation_profile(Path(args.config))
    steps = args.steps or int(profile["episode_steps"]) or 240
    seeds = parse_seeds(args.seeds)
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])
    start_seed = int(profile.get("start_seed", 0))
    print(f"策略=expert(确定性)  种子={seeds}  步数={steps}")

    slots: list[dict] = []
    reqs: list[dict] = []          # (H, imp_hops)
    tot_arrived = tot_served = 0.0

    for seed in seeds:
        env = build_env_from_config(config)
        policy = PathScoreGreedy(
            weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
            principles=False, router=ServeProbe(env))

        rp = env.rate_provider
        adj = env.routing.adj
        path_cache: dict[tuple[str, str, int], list[str] | None] = {}

        def path_edges(src: str, dst: str, t: int) -> list[str] | None:
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
            actions, scores = policy.act(obs)
            obs, _r, term, trunc, _info = env.step(actions, action_scores=scores)
            n += 1
            done = term or trunc

            # step 返回后：env.t 已 +1，graph_builder.last_relay_importance 是
            # 下一槽观测构建时的值（策略下一槽会看到的那个），pending 也是同状态。
            t_now = env.t
            imp = {k: v for k, v in (env.graph_builder.last_relay_importance or {}).items() if v > 0.0}
            imp_set = set(imp)
            needs: set[str] = set()
            live = 0
            unr = 0
            for req in env.requests.get_pending():
                pe = path_edges(req.src_gs, req.dst_gs, t_now)
                if pe is None:
                    unr += 1
                    continue
                live += 1
                needs.update(pe)
                ih = sum(1 for e in pe if imp.get(e, 0.0) > 0.0)
                reqs.append({"seed": seed, "t": t_now, "H": len(pe), "imp_hops": ih})
            slots.append({
                "seed": seed, "t": t_now, "imp": len(imp), "needs": len(needs),
                "inter": len(needs & imp_set), "live": live, "unr": unr,
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

    print("\n【表 1】每槽：imp 得分边集 vs needs 需求路径边集")
    print(f"  {'imp 得分边':>12}{'needs 需要边':>14}{'重合':>10}{'覆盖率':>10}"
          f"{'imp 里需求外的边':>18}{'在途(可达)':>12}")
    print(f"  {mean(s['imp'] for s in slots):>12.1f}"
          f"{mean(s['needs'] for s in slots):>14.1f}"
          f"{mean(s['inter'] for s in slots):>10.1f}"
          f"{mean(s['inter'] / max(1, s['needs']) for s in slots):>10.2%}"
          f"{mean(s['imp'] - s['inter'] for s in slots):>18.1f}"
          f"{mean(s['live'] for s in slots):>12.1f}")

    print("\n【表 2】逐(请求,槽)：路径上 importance>0 的跳数")
    for label, pre in (("整条路径有分", lambda r: r["H"] > 0 and r["imp_hops"] == r["H"]),
                       ("部分跳有分", lambda r: 0 < r["imp_hops"] < r["H"]),
                       ("一跳都没分", lambda r: r["imp_hops"] == 0 and r["H"] > 0),
                       ("零跳路径", lambda r: r["H"] == 0)):
        sel = [r for r in reqs if pre(r)]
        if not sel:
            continue
        print(f"  {label:<10}{len(sel):>8}{len(sel) / max(1, len(reqs)):>9.2%}"
              f"  平均H={mean(r['H'] for r in sel):.1f}"
              f"  平均有分跳={mean(r['imp_hops'] for r in sel):.1f}")

    # 交叉核对基准：专家 0.7801（探针 N/O/P 同协议）
    sr = tot_served / max(1e-9, tot_arrived)
    print(f"\n交叉核对：成功率 {sr:.4f} vs 基准 0.7801 → "
          f"{'一致' if abs(sr - 0.7801) < 5e-4 else '不一致，先查原因再用本探针结论'}")

    out = ROOT / "outputs" / "eval" / f"relay_coverage_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps({
        "tag": args.tag, "seeds": seeds, "steps": steps,
        "arrived": tot_arrived, "served": tot_served,
        "slots": slots, "reqs": reqs,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
