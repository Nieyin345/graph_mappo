"""探针 M：损失里有多少是**自己造成的拥塞**，有多少是**物理可用性**？

## 为什么问这个

从探针 H/K 得到的事实：训练后的 RL（确定性 0.8247）仍有 15.40% 的到达量过期，
拆成 A 5.32%（孤立节点，钉死）/ B_avail 6.83%（逐槽快照从未全通）/ C 3.25%（曾全通没搬完）。
探针 K 又显示 C 档的形状已经从"完全错过窗口"（BC 44.9% 的量完成度 <0.01）
变成"抓住窗口但搬不完"（RL 54.4% 的量完成度 0.25–0.50）。

这两种解释指向完全不同的下一步：

* 若损失主要是**竞争/排队**（那么多请求抢同一条瓶颈边，谁的 deadline 先到谁先拿，
  后来的就永远排队）→ 那"提升策略"的天花板其实取决于**需求侧**，
  策略再聪明也只能重分配，做机制（预置库存、更早排队）才有用。
* 若损失主要是**物理可用性**（那条边在那个窗口根本没有速率）→ 策略完全无能为力，
  只能改场景（拉长 deadline）。

## 怎么做

**单变量**：只缩放 `requests.arrival_rate`，其它一切不动。
负载从 1.0 降到 0.5 / 0.25 / 0.1，即"把需求抽稀、让排队消失但窗口不变"。

读数怎么解释：

* 降载后成功率**大涨**（趋近 `1 − A − B_avail` 的残余）→ 损失主体是拥塞，是**内生**的。
* 降载后成功率**基本不动** → 损失主体是可用性，是**外生**的。

A 档（孤立节点）和 B_avail（快照从未全通）在两边的定义下都与负载无关，
所以它们该是常量；**只有 C 档会随负载单调塌下去**，这就是判据。

用法（节点上）：
    cd /opt/qkd/graph_mappo && OMP_NUM_THREADS=2 /opt/qkd/venv/bin/python -u \
        .tmp/probe_load_sens.py --checkpoint outputs/r3_m512ent/checkpoint_final.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402

FACTORS = (1.0, 0.5, 0.25, 0.1)
UNREACHABLE_HOPS = 10 ** 6


def parse_seeds(spec: str) -> list[int]:
    return list(range(int(spec.split("-")[0]), int(spec.split("-")[1]) + 1))


def slots_components(env, horizon: int):
    """逐槽连通分量表（与探针 H 同一实现，保持口径一致）。"""
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


def run_factor(profile, factor: float, seeds: list[int], steps: int,
               checkpoint: str) -> dict:
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])
    base_rate = float(config["requests"]["arrival_rate"])
    config["requests"]["arrival_rate"] = base_rate * factor
    horizon = steps + 40

    tot_arrived = tot_served = 0.0
    rows: list[dict] = []

    for seed in seeds:
        env = build_env_from_config(config)
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
        data = load_checkpoint(checkpoint, device="cpu")
        model.load_state_dict(data.model_state)
        model.eval()
        policy = MAPPOPolicy(model, "cpu")

        comp, node_idx = slots_components(env, horizon)
        conn_cache: dict[tuple[str, str], np.ndarray] = {}

        def connected(src: str, dst: str) -> np.ndarray:
            key = (src, dst)
            arr = conn_cache.get(key)
            if arr is None:
                arr = comp[:, node_idx[src]] == comp[:, node_idx[dst]]
                conn_cache[key] = arr
            return arr

        queue = env.requests
        orig_expire = queue.expire

        def expire_patched(t: int, _rows=rows, _env=env, _seed=seed):
            out = orig_expire(t)
            for req in out:
                rem = max(0.0, float(req.amount) - float(req.served_amount))
                hops = int(_env.routing.hop_distance(req.src_gs, req.dst_gs))
                reachable = (hops < UNREACHABLE_HOPS
                             and req.src_gs in node_idx and req.dst_gs in node_idx)
                a, d = int(req.arrival_t), int(req.deadline_t)
                avail = 0
                if reachable:
                    avail = int(connected(req.src_gs, req.dst_gs)[a:min(d, horizon)].sum())
                _rows.append({
                    "seed": _seed, "src": req.src_gs, "dst": req.dst_gs,
                    "amount": float(req.amount), "remaining": rem,
                    "served_frac": float(req.served_amount) / max(1e-9, float(req.amount)),
                    "hops": hops if reachable else -1,
                    "lifetime": d - a, "avail_slots": avail,
                })
            return out

        queue.expire = expire_patched
        obs = env.reset(seed=seed, start_seed=seed)
        n = 0
        done = False
        while n < steps and not done:
            with torch.no_grad():
                step = policy.act(obs, deterministic=True)
            obs, _r, term, trunc, _info = env.step(
                step.actions, action_scores=step.action_scores)
            n += 1
            done = term or trunc
        queue.expire = orig_expire
        s = env.metrics.episode_summary()
        tot_arrived += s["arrived_keys"]
        tot_served += s["served_keys"]

    buckets = {"A": [0.0, 0], "B_avail": [0.0, 0], "C": [0.0, 0]}
    for r in rows:
        if r["hops"] < 0:
            k = "A"
        elif r["avail_slots"] == 0:
            k = "B_avail"
        else:
            k = "C"
        buckets[k][0] += r["remaining"]
        buckets[k][1] += 1
    return {
        "factor": factor, "arrival_rate": base_rate * factor,
        "arrived": tot_arrived, "served": tot_served,
        "success_pooled": tot_served / max(1e-9, tot_arrived),
        "buckets": buckets,
        "c_served_frac_mean": (
            sum(r["served_frac"] for r in rows if r["hops"] >= 0
                and r["avail_slots"] > 0)
            / max(1, sum(1 for r in rows if r["hops"] >= 0 and r["avail_slots"] > 0))),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",
                    default="outputs/r3_m512ent/checkpoint_final.pt")
    ap.add_argument("--seeds", default="7-18")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--config", default=str(ROOT / "configs" / "var2_diag.yaml"))
    ap.add_argument("--factors", default=",".join(str(f) for f in FACTORS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    profile = _tp.load_validation_profile(Path(args.config))
    seeds = parse_seeds(args.seeds)
    factors = [float(x) for x in args.factors.split(",")]

    print(f"checkpoint={args.checkpoint}")
    print(f"协议：窗={profile['window_start_day']}-{profile['window_end_day']}  "
          f"模式={profile['start_mode']}  步数={args.steps}  种子={seeds[0]}-{seeds[-1]}")
    print(f"负载因子：{factors}（只改 requests.arrival_rate，动作确定性）\n")

    out = []
    for f in factors:
        r = run_factor(profile, f, seeds, args.steps, args.checkpoint)
        out.append(r)
        b = r["buckets"]
        print(f"负载 × {f:<5} arrival_rate={r['arrival_rate']:<7.4f} "
              f"到达={r['arrived']:>14,.0f}  成功率={r['success_pooled']:.4f}")
        print(f"        过期拆解：A {b['A'][0] / max(1e-9, r['arrived']):>6.2%}"
              f"（{b['A'][1]:>4} 条）  B_avail {b['B_avail'][0] / max(1e-9, r['arrived']):>6.2%}"
              f"（{b['B_avail'][1]:>4} 条）  C {b['C'][0] / max(1e-9, r['arrived']):>6.2%}"
              f"（{b['C'][1]:>4} 条）  C 完成度={r['c_served_frac_mean']:.3f}")

    print(f"\n{'负载×':>8}{'到达量':>16}{'成功率':>10}{'A':>9}{'B_avail':>9}{'C':>9}{'C完成度':>10}")
    for r in out:
        b = r["buckets"]
        a = r["arrived"]
        print(f"{r['factor']:>8}{a:>16,.0f}{r['success_pooled']:>10.4f}"
              f"{b['A'][0] / max(1e-9, a):>9.2%}{b['B_avail'][0] / max(1e-9, a):>9.2%}"
              f"{b['C'][0] / max(1e-9, a):>9.2%}{r['c_served_frac_mean']:>10.3f}")
    print("\n判读：C 随负载塌下去 = 拥塞（内生）；三档都不动 = 可用性（外生）。")
    print("\nLOAD_SENS_DONE")

    p = Path(f"outputs/eval/load_sens{args.tag}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump([{k: v for k, v in r.items() if k != "rows"} for r in out],
              p.open("w", encoding="utf-8"), indent=2)
    print(f"已保存 {p}")


if __name__ == "__main__":
    main()