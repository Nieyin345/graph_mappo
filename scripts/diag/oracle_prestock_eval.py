"""诊断：如果策略确切知道该给哪些边"提前囤货"，成功率能涨多少？（2026-09-27）

动机：失败解剖（实验日志 §11.12）里最大的可控损失是"没提前囤货"（2× 3.1%、3× 4.9%，按连通性估计的上限）；
但把 120 步接触前瞻当特征喂给模型（batch-8）毫无效果。要么是 RL 学不会用（信用分配 ~17 步），
要么是这块损失本身就不可兑现（端口/容量竞争）。本脚本回答后者：给出**真实**的囤货标签，看能涨多少。

囤货标签（与策略无关，只用真实到达序列 + 接触计划）：
  对每个请求 r（到达 a_r、截止 d_r），V_r = [a_r, d_r) 内可见过的边。若 src–dst 在 V_r 上不连通，
  在 V_r ∪ Vis[a_r−H, a_r) 上用 0-1 BFS 找"V_r 外的边最少"的路径 P_r，D_r = P_r \\ V_r
  （= 必须在到达前囤好货的边）。于是在 t ∈ [a_r−H, a_r) 时，D_r ∩ Vis(t) 的边标为"该囤"。
  可见性直接取 rate provider 的全边可见性（接触计划的真值）。

评测：与 oracle_forecast_eval 同一循环，把预测头输出换成 标签→±3 的 logit，偏置照常
tanh → 图内 z-score → × bias_weight（--weight 可扫）。另设 shuffled（图内打乱）对照。

用法：
  python scripts/diag/oracle_prestock_eval.py --run outputs/experiments/v8/load2x_b7_noattbias_s42 \\
      --seeds 100-114 --weights 0.2,0.5,1.0 --out outputs/eval/oracle_prestock_xxx.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "diag"))

from oracle_forecast_eval import ORACLE_LOGIT, parse_seeds, run_episode  # noqa: E402
from qkd_rl.core.config import load_config  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.evaluation.test_protocol import run_validation_env_config, task_fingerprint  # noqa: E402
from qkd_rl.model_zoo import build_model  # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402


def min_outside_path(adj, src, dst, usable, inside):
    """0-1 BFS：只走 usable，最小化 inside 之外的边数；返回边列表或 None。"""
    best = {src: 0}
    parent = {src: (None, None)}
    dq = deque([(0, src)])
    while dq:
        c, u = dq.popleft()
        if c > best.get(u, 1 << 30):
            continue
        if u == dst:
            path = []
            while parent[u][0] is not None:
                path.append(parent[u][1])
                u = parent[u][0]
            return path[::-1]
        for v, e in adj.get(u, ()):
            if e not in usable:
                continue
            w = 0 if e in inside else 1
            if c + w < best.get(v, 1 << 30):
                best[v] = c + w
                parent[v] = (u, e)
                (dq.appendleft if w == 0 else dq.append)((c + w, v))
    return None


class ArrivalLog:
    def __init__(self, env):
        self.reqs = []
        orig = env.requests.add_arrivals

        def add(requests, _orig=orig):
            self.reqs.extend(requests)
            return _orig(requests)

        env.requests.add_arrivals = add
        self.env = env

    def detach(self):
        del self.env.requests.add_arrivals


def prestock_labels(env, t_list, reqs, horizon):
    """→ list（按步）of set(edge_id)：该步可见且"该囤"的边。另返回统计。"""
    t0, T = t_list[0], len(t_list)
    all_ids = list(env.routing.edges)
    all_ids = [e.edge_id for e in all_ids]
    idx = env.rate_provider.edge_link_ids(all_ids)
    av = env.rate_provider.get_avail_lookahead(t0 - 1, T + 60)[:, idx]  # 行 = t0 .. t0+T+59
    vis = [set(e for e, ok in zip(all_ids, av[i]) if ok) for i in range(av.shape[0])]
    adj = env.routing.adj
    labels = [set() for _ in range(T)]
    stats = {"requests": 0, "life_connected": 0, "needs_prestock": 0, "no_path": 0,
             "needs_prestock_amount": 0.0, "amount": 0.0}
    for r in reqs:
        a, d = int(r.arrival_t) - t0, int(r.deadline_t) - t0
        stats["requests"] += 1
        stats["amount"] += float(r.amount)
        life = set()
        for i in range(max(0, a), min(d, len(vis))):
            life |= vis[i]
        if min_outside_path(adj, r.src_gs, r.dst_gs, life, life) is not None:
            stats["life_connected"] += 1
            continue
        lo = max(0, a - horizon)
        before = set()
        for i in range(lo, max(lo, a)):
            before |= vis[i]
        path = min_outside_path(adj, r.src_gs, r.dst_gs, life | before, life)
        if path is None:
            stats["no_path"] += 1
            continue
        deficit = [e for e in path if e not in life]
        stats["needs_prestock"] += 1
        stats["needs_prestock_amount"] += float(r.amount)
        for i in range(lo, min(max(lo, a), T)):
            for e in deficit:
                if e in vis[i]:
                    labels[i].add(e)
    return labels, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--weights", default="0.2,0.5,1.0")
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"{out_path} 已存在（outputs 只增不改）")
    torch.set_num_threads(args.threads)
    run = Path(args.run)
    config = load_config([run / "experiment_config.yaml"])
    seeds = parse_seeds(args.seeds)
    vcfg = config.get("validation", {}) or {}
    env_cfg = run_validation_env_config(config, vcfg)
    val_seeds = [int(s) for s in (vcfg.get("request_seeds") or [])]
    env_cfg["seed"]["env_seed"] = val_seeds[0] if val_seeds else seeds[0]
    start_base = int(vcfg.get("start_seed", 0) or 0)
    env = build_env_from_config(env_cfg)
    model = build_model(env.action_resolver.action_space, config)
    model.load_state_dict(load_checkpoint(run / "checkpoint_final.pt", "cpu").model_state)
    model.eval()
    if model.actor.usage_forecast is None:
        raise SystemExit("该检查点没有预测头（需要 v8 类带偏置的模型）")
    w_train = float(model.actor.usage_bias_weight)
    policy = MAPPOPolicy(model, "cpu")
    from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
    trainer = MAPPOTrainer(env, policy, config, run, device="cpu")
    needs_scores, resolver_mode = trainer._needs_edge_scores, trainer.resolver_mode
    weights = [float(x) for x in args.weights.split(",")]
    res = {"normal": []}
    for w in weights:
        res[f"oracle_w{w}"] = []
        res[f"shuffled_w{w}"] = []
    label_stats = []
    for seed in seeds:
        model.actor.usage_bias_weight = w_train
        log = ArrivalLog(env)
        sr, trace, _g, _s = run_episode(env, policy, seed, start_base + seed, needs_scores, resolver_mode)
        log.detach()
        res["normal"].append(sr)
        # t_list：从 trace 长度与 env 起点推回（run_episode 从 reset 后的 env.t 开始逐步 +1）
        t_end = int(env.t)
        t_list = list(range(t_end - len(trace), t_end))
        labels, stats = prestock_labels(env, t_list, log.reqs, args.horizon)
        stats["label_pos_per_step"] = sum(len(x) for x in labels) / max(1, len(labels))
        label_stats.append(stats)
        gen = torch.Generator().manual_seed(seed)

        def oracle(t, obs, _labels=labels):
            lab = _labels[t] if t < len(_labels) else set()
            v = torch.tensor([1.0 if e in lab else 0.0 for e in obs.physical_edge_ids], dtype=torch.float32)
            return (2.0 * v - 1.0) * ORACLE_LOGIT

        def shuffled(t, obs, _o=oracle, _g=gen):
            v = _o(t, obs)
            return v[torch.randperm(v.numel(), generator=_g)] if v.numel() else v

        for w in weights:
            model.actor.usage_bias_weight = w
            for name, fn in ((f"oracle_w{w}", oracle), (f"shuffled_w{w}", shuffled)):
                sr_m, *_ = run_episode(env, policy, seed, start_base + seed, needs_scores, resolver_mode, fn)
                res[name].append(sr_m)
        model.actor.usage_bias_weight = w_train
        print(f"seed {seed}: " + "  ".join(f"{k} {v[-1]:.4f}" for k, v in res.items())
              + f"  | needs_prestock {stats['needs_prestock']}/{stats['requests']} pos/step {stats['label_pos_per_step']:.2f}",
              flush=True)
    mean = {k: sum(v) / len(v) for k, v in res.items()}
    print("mean: " + "  ".join(f"{k} {v:.4f}" for k, v in mean.items()))
    ref = run / "validation.json"
    if ref.exists():
        v = json.loads(ref.read_text(encoding="utf-8"))
        ref_map = dict(zip(v.get("seeds", []), v.get("per_seed_success", [])))
        diff = max(abs(a - ref_map[s]) for a, s in zip(res["normal"], seeds) if s in ref_map)
        print(f"normal vs validation.json 最大逐种子差 {diff:.2e}（应为 0）")
        mean["normal_vs_validation_maxdiff"] = diff
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "run": str(run), "seeds": seeds, "task_fingerprint": task_fingerprint(env_cfg), "horizon": args.horizon,
        "train_weight": w_train, "per_seed_success": res, "mean": mean, "label_stats": label_stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
