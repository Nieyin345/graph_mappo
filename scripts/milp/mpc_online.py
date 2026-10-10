"""在线滚动 LP（MPC）参考策略（2026-10-03，只读诊断，不改 env）。

每一步：用「当前各边库存 + 上一槽激活状态 + 当前待服务请求（剩余量、截止）」从现在起
求含切换开销的流体 LP（不使用任何未来请求信息），取第 0 槽的激活解，按 oracle_exec 的同一
四舍五入（度≤2 贪心 + 定向，--fill 补满空闲端口）执行。它只用在线可得信息 ⟹ 是一个
合法的在线基线 / 无作弊教师；与先知执行之差 = 「未来请求信息」的价值。
用法：python scripts/milp/mpc_online.py --run <run> --seeds 1000 [--overlay x.yaml] --fill --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.evaluation import test_protocol as tp  # noqa: E402
from scripts.milp.fluid_upper_bound import parse_seeds, solve_bound, task_config  # noqa: E402
from scripts.milp.oracle_exec import oracle_actions  # noqa: E402


class MPCPolicy:
    def __init__(self, env, t_start: int, T: int, sd: float, horizon: int = 60, fill: bool = True,
                 keep_bonus: float = 0.05, forecast: str = "perfect", lookback: int = 1440):
        self.env, self.t_start, self.T, self.sd = env, t_start, T, sd
        self.h, self.fill, self.kb = horizon, fill, keep_bonus
        self.ends = {e.edge_id: (e.src, e.dst) for e in env.routing.edges}
        self.n_solve = 0
        self.solve_s = 0.0
        # forecast="persist"（现实预报）：可见性按接触计划（轨道确定，真实），速率用持续性预报——
        # 当前可见的边取当前速率，当前不可见的边取它最近一次可见时的速率（回看 lookback 步）。
        self.forecast = forecast
        if forecast == "persist":
            eids = [e.edge_id for e in env.routing.edges]
            self.idx = env.rate_provider.edge_link_ids(eids)
            self.last = np.zeros(len(eids))
            seen = np.zeros(len(eids), dtype=bool)
            for k in range(1, lookback + 1):
                rate, av = self._blocks(t_start - k)
                new = av & ~seen
                self.last[new] = rate[new]
                seen |= av
                if seen.all():
                    break

    def _blocks(self, t):
        b = self.env.rate_provider.get_window_blocks(t)
        return (np.asarray(b[0][0][self.idx], dtype=float), np.asarray(b[1][0][self.idx], dtype=float) >= 0.5)

    def act(self, obs):
        t = int(obs.state.t)
        end = self.t_start + self.T
        pend = [SimpleNamespace(src_gs=r.src_gs, dst_gs=r.dst_gs,
                                amount=max(0.0, r.amount - r.served_amount), deadline_t=r.deadline_t)
                for r in self.env.requests.pending if r.amount - r.served_amount > 1e-9]
        plan = {}
        ro = None
        if self.forecast == "persist":
            rate, av = self._blocks(t)
            self.last[av] = rate[av]
            ro = self.last.copy()
        if pend:
            h = max(1, min(end - t, max(int(p.deadline_t) for p in pend) - t + 1, self.h))
            ts = time.perf_counter()
            r = solve_bound(self.env, t, h, pend, 1, switch_decay=self.sd, return_activation=True,
                            init_stock=dict(self.env.qkp.levels),
                            prev_active=set(obs.state.last_activated_edges), rate_override=ro)
            self.solve_s += time.perf_counter() - ts
            self.n_solve += 1
            plan = {0: {eid: a for eid, tau, a in r.get("activation", []) if tau == 0}}
        return oracle_actions(obs, plan, t, self.ends, self.kb, self.fill)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="1000")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--forecast", choices=("perfect", "persist"), default="perfect")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    config, vcfg = task_config(a.run, a.overlay)
    T = int(config["env"]["episode_steps"])
    scfg = config.get("env", {}).get("switch_cost", {})
    sd = float(scfg.get("rate_decay_factor", 0.5)) if scfg.get("enabled", True) else 1.0
    base = int(vcfg.get("start_seed", 0) or 0)
    rows = []
    for seed in parse_seeds(a.seeds):
        env = build_env_from_config(config)
        obs = env.reset(seed=seed, start_seed=base + seed)
        pol = MPCPolicy(env, int(obs.state.t), T, sd, a.horizon, a.fill, forecast=a.forecast)
        done = False
        while not done:
            obs, _r, te, tr, _i = env.step(pol.act(obs), None)
            done = te or tr
        row = {"seed": seed, "mpc_success": env.metrics.episode_summary().get("success_rate"),
               "n_solve": pol.n_solve, "solve_s": round(pol.solve_s, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task_fingerprint": tp.task_fingerprint(config), "horizon": a.horizon, "forecast": a.forecast,
                               "fill": a.fill, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
