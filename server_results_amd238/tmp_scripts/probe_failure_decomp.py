"""把"没服务上的 key"拆开：到底是没时间、没库存，还是路径根本没有。

背景（读代码读出来的机制，见 qkd_rl/env/request.py:154-220）：
``RequestQueue.serve`` **永远返回 failed_keys=0.0** —— 它只把没服务完的请求放回
pending，真正的失败全在 ``RequestQueue.expire``：``t >= req.deadline_t`` 时，
未服务完的**剩余量**计入 failed_keys（qkd_rl/env/metrics.py:38）。

所以一个请求失败只有一条路径：**在 deadline（到达 + 30 槽）之内没被完全服务**。
没有"拒服务"、没有"端口冲突丢弃"。

那么 0.78 → 0.18 的缺口必然是下面几件事之一：

  A. 路径缺口：源-目的之间压根没有 usable path（GS 对里有些对可能永远不可达），
     这类需求从出生就注定失败。
  B. 时间/库存缺口：路径存在，但配额速率 × 可用窗口 × 30 槽 < 需求量，
     即物理上搬不完。
  C. 调度缺口：物理上搬得完，但策略没在正确的时刻把正确的链路接上。

C 才是 RL 能改进的部分。这个脚本给 (A)/(B) 一个**量级**，好知道 C 还剩多少空间。

做法：不改环境，只在每条请求过期时记账 —— 记录它的需求量、路径是否存在、
路径上各跳在它存活期间的**可用性**与**速率**。用 monkey-patch 包一层
``RequestQueue.expire`` 和 ``RoutingPolicy.partial_consume_for_request``，
环境本身一行不动。

用法（节点上）：
    cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python -u .tmp/probe_failure_decomp.py \\
        --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \\
        --seeds 7-9 --steps 240
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
from qkd_rl.env.request import RequestQueue  # noqa: E402
from qkd_rl.env.routing import RoutingPolicy  # noqa: E402
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


class Ledger:
    """过期请求的分类账。"""

    def __init__(self):
        self.rows: list[dict] = []
        self.served_total = 0.0
        self.expired_total = 0.0
        self.arrived_total = 0.0
        # 路径缓存：同 (src,dst,age) 反复算没意义，一集里 GS 对数量有限。
        self._path_cache: dict[tuple[str, str, int], dict] = {}

    def note_arrival(self, amount: float) -> None:
        self.arrived_total += amount

    def note_served(self, amount: float) -> None:
        self.served_total += amount

    def note_expired(self, req, routing, t: int) -> None:
        remaining = max(0.0, req.amount - req.served_amount)
        self.expired_total += remaining
        key = (req.src_gs, req.dst_gs, req.arrival_t)
        info = self._path_cache.get(key)
        if info is None:
            info = self._path_info(req, routing)
            self._path_cache[key] = info
        self.rows.append({
            "req": req.request_id,
            "src": req.src_gs,
            "dst": req.dst_gs,
            "amount": float(req.amount),
            "remaining": remaining,
            "served_frac": float(req.served_amount) / max(1e-9, float(req.amount)),
            "lifetime": int(req.deadline_t - req.arrival_t),
            **info,
        })

    @staticmethod
    def _path_info(req, routing) -> dict:
        """路径在不在、几跳、跳数是多少。"""
        try:
            hops = int(routing.hop_distance(req.src_gs, req.dst_gs))
        except Exception:
            hops = -1
        # hop_distance 返回 inf/大数 表示不可达
        reachable = hops >= 0 and hops < 10 ** 6
        return {"hops": hops, "reachable": bool(reachable)}


def run_episode(env, policy, seed: int, steps: int, ledger: Ledger, start_seed: int = 0,
                is_expert: bool = False):
    ledger.rows.clear()
    ledger._path_cache.clear()
    # 和 eval_expert.py 完全一致的 reset 参数（随机日启动靠 start_seed + seed 决定），
    # 否则起点的场景和参照表对不上。
    obs = env.reset(seed=seed, start_seed=start_seed + seed)

    queue = env.requests
    routing = env.routing
    orig_expire = queue.expire
    orig_serve = queue.serve

    def expire_patched(t: int):
        expired = orig_expire(t)
        for req in expired:
            ledger.note_expired(req, routing, t)
        return expired

    def serve_patched(qkp, rt, t: int):
        result = orig_serve(qkp, rt, t)
        ledger.note_served(result.served_keys)
        return result

    queue.expire = expire_patched
    queue.serve = serve_patched

    n = 0
    done = False
    while n < steps and not done:
        if is_expert:
            actions, scores = policy.act(obs)
            obs, _r, term, trunc, _info = env.step(actions, scores)
        else:
            with torch.no_grad():
                step = policy.act(obs)
            obs, _r, term, trunc, _info = env.step(
                step.actions, action_scores=step.action_scores)
        n += 1
        done = term or trunc

    queue.expire = orig_expire
    queue.serve = orig_serve
    return env.metrics.episode_summary()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--policy", default="expert", choices=["expert", "rl"])
    ap.add_argument("--seeds", default="7-9")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--config", default=str(ROOT / "configs" / "global.yaml"))
    args = ap.parse_args()

    profile = _tp.load_validation_profile(Path(args.config))
    steps = args.steps or int(profile["episode_steps"]) or 240
    seeds = parse_seeds(args.seeds)
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])
    start_seed = int(profile.get("start_seed", 0))

    print(f"策略={args.policy}  种子={seeds}  步数={steps}  "
          f"窗={profile['window_start_day']}-{profile['window_end_day']}  "
          f"模式={profile['start_mode']}")
    print(f"{'种子':>5}{'成功率':>9}{'到达':>16}{'服务':>16}{'过期':>16}{'缺口':>9}"
          f"{'不可达请求占比':>15}{'不可达需求占比':>16}")

    all_rows: list[dict] = []
    tot_arr = tot_srv = tot_exp = tot_unreach = tot_unreach_amt = 0.0
    for seed in seeds:
        # 每个种子一个新 env —— 和 eval_expert.py 一致（ServeProbe 持 env 引用）。
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
        led = Ledger()
        s = run_episode(env, policy, seed, steps, led, start_seed, is_expert)
        arr = s["arrived_keys"]
        srv = s["served_keys"]
        exp = led.expired_total
        unreach_rows = [r for r in led.rows if not r["reachable"]]
        unreach_amt = sum(r["remaining"] for r in unreach_rows)
        gap = 1.0 - s["success_rate"]
        tot_arr += arr
        tot_srv += srv
        tot_exp += exp
        tot_unreach += len(unreach_rows)
        tot_unreach_amt += unreach_amt
        all_rows.extend(led.rows)
        frac_unreach_n = len(unreach_rows) / max(1, len(led.rows))
        frac_unreach_a = unreach_amt / max(1e-9, exp)
        print(f"{seed:>5}{s['success_rate']:>9.4f}{arr:>16,.0f}{srv:>16,.0f}{exp:>16,.0f}"
              f"{gap:>9.4f}{frac_unreach_n:>12.3f}{frac_unreach_a:>16.3f}")

    print()
    print(f"合计 到达={tot_arr:,.0f} 服务={tot_srv:,.0f} 过期={tot_exp:,.0f}")
    print(f"整体成功率={tot_srv / max(1e-9, tot_arr):.4f}  缺口={1 - tot_srv / max(1e-9, tot_arr):.4f}")
    print(f"过期请求数={len(all_rows)}  其中不可达={int(tot_unreach)}"
          f"（{tot_unreach / max(1, len(all_rows)):.1%}）")
    print(f"不可达请求占过期量={tot_unreach_amt / max(1e-9, tot_exp):.1%}")

    # 按 hop 数看：服务了多少比例。hops 越大越难。
    print()
    print("按跳数分层（过期请求的完成度）：")
    by_hop: dict[int, list[dict]] = {}
    for r in all_rows:
        by_hop.setdefault(r["hops"], []).append(r)
    print(f"{'跳数':>6}{'过期数':>9}{'过期量占比':>12}{'平均完成度':>12}")
    for h in sorted(by_hop):
        rows = by_hop[h]
        amt = sum(r["remaining"] for r in rows)
        done = sum(r["served_frac"] for r in rows) / len(rows)
        print(f"{h:>6}{len(rows):>9}{amt / max(1e-9, tot_exp):>12.1%}{done:>12.3f}")

    out = Path("outputs/eval/failure_decomp.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"policy": args.policy, "seeds": seeds, "steps": steps,
               "overall_success": tot_srv / max(1e-9, tot_arr),
               "rows": all_rows}, out.open("w", encoding="utf-8"), indent=2)
    print(f"\n已保存 {out}")


if __name__ == "__main__":
    main()
