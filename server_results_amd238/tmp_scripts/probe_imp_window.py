"""验证实验：放开 relay_importance 的窗口（max_path_links / hop_decay），专家档对照。

## 依据（探针 Q，成功率 0.7801 交叉核对一致）

72.47% 的（请求,槽）路径一跳都拿不到 importance 分（max_path_links=3 对平均 6.4 跳的
路径全盲）；imp 得分边每槽仅 1.6 条、对 needs 覆盖率 7.09%。
该信号被特征列 / dense 奖励 / 专家 dense_fill 三方共用。

## 改法

三方里专家侧的参数是 compute_dynamic_relay_importance 的**函数默认参数**（不走 config），
所以本脚本 monkey-patch `path_greedy.compute_dynamic_relay_importance`；同时改
env.config 的 relay_cfg（对应特征列与 dense 奖励的来源 graph_builder）。
训练侧若验证有效，正式改动 = features.yaml 两行 + greedy_relay_diffusion 默认值两行。

用法：
    python -u .tmp/probe_imp_window.py --max_path_links 8 --hop_decay 1.0 --tag mp8d10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.baselines import path_greedy as _pg  # noqa: E402
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
    ap.add_argument("--max_path_links", type=int, default=8)
    ap.add_argument("--hop_decay", type=float, default=1.0)
    ap.add_argument("--seeds", default="7-18")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--tag", default="mp8d10")
    ap.add_argument("--config", default=str(ROOT / "configs" / "var2_diag.yaml"))
    args = ap.parse_args()

    # --- 1) 专家侧参数在 GreedyRelayDiffusionPolicyV3 实例属性上（不走 config）---
    # PathScoreGreedy._v3_scorer 懒创建，先造一个再改 max_path_links / hop_decay。
    profile = _tp.load_validation_profile(Path(args.config))
    steps = args.steps or int(profile["episode_steps"]) or 240
    seeds = parse_seeds(args.seeds)
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])
    start_seed = int(profile.get("start_seed", 0))
    relay_cfg = config["features"]["edge"].setdefault("relay_importance", {})
    relay_cfg["max_path_links"] = args.max_path_links
    relay_cfg["hop_decay_factor"] = args.hop_decay
    print(f"tag={args.tag}  max_path_links={args.max_path_links}  "
          f"hop_decay={args.hop_decay}  种子={seeds}  步数={steps}")

    slots: list[dict] = []
    tot_arrived = tot_served = 0.0
    for seed in seeds:
        env = build_env_from_config(config)
        policy = _pg.PathScoreGreedy(
            weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
            principles=False, router=ServeProbe(env))
        # 直接构造 v3 scorer（与 _v3_scores 同权重）并覆盖窗口参数
        policy._v3_scorer = _pg.GreedyRelayDiffusionPolicyV3(
            rate_weight=1.0, importance_weight=10.0, completion_weight=1.0,
            keep_weight=0.5, switch_weight=0.2)
        policy._v3_scorer.max_path_links = args.max_path_links
        policy._v3_scorer.hop_decay_factor = args.hop_decay
        rp = env.rate_provider
        adj = env.routing.adj
        cache: dict[tuple[str, str, int], list[str] | None] = {}

        def path_edges(src: str, dst: str, t: int) -> list[str] | None:
            key = (src, dst, t)
            if key in cache:
                return cache[key]
            out: list[str] | None = None
            if src != dst:
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
            cache[key] = out
            return out

        obs = env.reset(seed=seed, start_seed=start_seed + seed)
        n = 0
        done = False
        while n < steps and not done:
            actions, scores = policy.act(obs)
            obs, _r, term, trunc, _info = env.step(actions, action_scores=scores)
            n += 1
            done = term or trunc
            t_now = env.t
            imp = {k for k, v in (env.graph_builder.last_relay_importance or {}).items() if v > 0.0}
            act = set(env.last_activated_edges)
            needs: set[str] = set()
            live = unr = 0
            for req in env.requests.get_pending():
                pe = path_edges(req.src_gs, req.dst_gs, t_now)
                if pe is None:
                    unr += 1
                    continue
                live += 1
                needs.update(pe)
            slots.append({
                "seed": seed, "t": t_now, "act": len(act), "needs": len(needs),
                "inter_an": len(act & needs), "imp": len(imp),
                "inter_in": len(imp & needs), "live": live, "unr": unr,
            })
        s = env.metrics.episode_summary()
        tot_arrived += s["arrived_keys"]
        tot_served += s["served_keys"]
        print(f"  seed={seed} 成功率={s['success_rate']:.4f}")

    def mean(vals) -> float:
        vals = list(vals)
        return sum(vals) / len(vals) if vals else float("nan")

    sr = tot_served / max(1e-9, tot_arrived)
    print(f"\n量加权成功率={sr:.4f}  （对照 0.7801）")
    print("\n【表 1】每槽均值")
    print(f"  {'act 激活':>8}{'needs':>8}{'act∩needs':>10}{'act 覆盖率':>11}"
          f"{'imp 得分':>9}{'imp∩needs':>10}{'imp 覆盖率':>11}")
    print(f"  {mean(s['act'] for s in slots):>8.1f}{mean(s['needs'] for s in slots):>8.1f}"
          f"{mean(s['inter_an'] for s in slots):>10.1f}"
          f"{mean(s['inter_an'] / max(1, s['needs']) for s in slots):>11.2%}"
          f"{mean(s['imp'] for s in slots):>9.1f}"
          f"{mean(s['inter_in'] for s in slots):>10.1f}"
          f"{mean(s['inter_in'] / max(1, s['needs']) for s in slots):>11.2%}")

    out = ROOT / "outputs" / "eval" / f"imp_window_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": args.tag, "max_path_links": args.max_path_links,
        "hop_decay": args.hop_decay, "seeds": seeds, "steps": steps,
        "arrived": tot_arrived, "served": tot_served, "slots": slots,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
