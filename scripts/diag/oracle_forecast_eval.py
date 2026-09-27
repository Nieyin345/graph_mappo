"""诊断：如果使用预测是完美的，成功率能涨多少？（2026-09-27）

在训练好的 v8 类检查点上，只在**评测**时把预测头的输出换掉，不重新训练：
  normal   : 原样（必须逐种子复现 validation.json，验证本脚本的评测循环）
  oracle   : 真实标签（与训练同一个 `RolloutBuffer._fill_usage_labels`，标签来自 normal
             那一遍的逐步 request_path_edge_mask）→ logit = ±3
  shuffled : 同一组标签在图内随机打乱（同分布、无信息）——排除"偏置分布变了"本身的效应
偏置照常走 tanh → 图内 z-score → × bias_weight（可用 --weight 覆盖）。

用法：
  python scripts/diag/oracle_forecast_eval.py --run outputs/experiments/v8/load2x_b7_noattbias_s42 \
      --seeds 100-114 --modes normal,oracle,shuffled [--weight 0.2] --out outputs/eval/oracle_xxx.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import load_config  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.evaluation.test_protocol import run_validation_env_config, task_fingerprint  # noqa: E402
from qkd_rl.model_zoo import build_model  # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint  # noqa: E402
from qkd_rl.rl.algos.policy import DETERMINISTIC_RESOLVER_MODES, MAPPOPolicy  # noqa: E402
from qkd_rl.rl.algos.rollout_buffer import RolloutBuffer  # noqa: E402

ORACLE_LOGIT = 3.0


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def oracle_labels(trace: list[tuple[list[str], list[bool] | None]], horizon: int, mode: str) -> list[dict[str, float]]:
    """与训练完全相同的标签实现（RolloutBuffer._fill_usage_labels），按边 id 返回。"""
    episode = [SimpleNamespace(obs=SimpleNamespace(physical_edge_ids=ids, request_path_edge_mask=m),
                               usage_label=None) for ids, m in trace]
    fake = SimpleNamespace(usage_horizon=horizon, usage_label_mode=mode)
    RolloutBuffer._fill_usage_labels(fake, episode)
    return [{eid: float(v) for eid, v in zip(step.obs.physical_edge_ids, step.usage_label.tolist())}
            for step in episode]


def run_episode(env, policy, seed, start_seed, needs_scores, resolver_mode, override=None):
    """与 MAPPOTrainer.evaluate_validation 同一循环（单环境）。override(t, obs) -> logits 或 None。"""
    head = policy.model.actor.usage_forecast
    orig_forward = head.forward
    obs = env.reset(seed=seed, start_seed=start_seed)
    trace: list[tuple[list[str], list[bool] | None]] = []
    t = 0
    done = False
    try:
        while not done:
            trace.append((list(obs.physical_edge_ids or []),
                          list(obs.request_path_edge_mask) if obs.request_path_edge_mask is not None else None))
            if override is not None:
                vec = override(t, obs)

                def _fwd(pair_emb, _vec=vec):
                    assert pair_emb.size(0) == _vec.numel(), (pair_emb.size(0), _vec.numel())
                    return _vec.to(pair_emb.device, pair_emb.dtype)
                head.forward = _fwd
            with torch.no_grad():
                out = policy.act_batched([obs], deterministic=True, build_scores=needs_scores,
                                         use_edge_arrays=not needs_scores)[0]
            obs, _r, te, tr, _i = env.step(
                out.actions, out.action_scores, edge_scores=out.edge_scores,
                expected_matched_edges=(None if resolver_mode in DETERMINISTIC_RESOLVER_MODES
                                        else list(out.matched_edges or [])))
            done = te or tr
            t += 1
    finally:
        head.forward = orig_forward
    s = env.metrics.episode_summary()
    return float(s.get("success_rate", 0.0)), trace, float(s.get("generated_keys", 0.0)), float(s.get("served_keys", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--modes", default="normal,oracle,shuffled")
    ap.add_argument("--weight", type=float, default=None, help="覆盖 bias_weight（默认用训练值）")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"{out_path} 已存在（outputs 只增不改）")
    torch.set_num_threads(args.threads)
    config = load_config([run / "experiment_config.yaml"])
    seeds = parse_seeds(args.seeds)
    vcfg = config.get("validation", {}) or {}
    env_cfg = run_validation_env_config(config, vcfg)
    val_seeds = [int(s) for s in (vcfg.get("request_seeds") or [])]
    env_cfg["seed"]["env_seed"] = val_seeds[0] if val_seeds else seeds[0]
    fp = task_fingerprint(env_cfg)
    start_base = int(vcfg.get("start_seed", 0) or 0)

    env = build_env_from_config(env_cfg)
    model = build_model(env.action_resolver.action_space, config)
    model.load_state_dict(load_checkpoint(run / "checkpoint_final.pt", "cpu").model_state)
    model.eval()
    if model.actor.usage_forecast is None:
        raise SystemExit("该检查点没有预测头")
    if args.weight is not None:
        model.actor.usage_bias_weight = float(args.weight)
    policy = MAPPOPolicy(model, "cpu")
    ucfg = config["model"]["actor"]["usage_forecast"]
    horizon, label_mode = int(ucfg.get("horizon", 8)), str(ucfg.get("label_mode", "used"))
    # needs_scores / resolver_mode 与 trainer 一致（evaluate_one 同样构造 trainer）
    from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
    trainer = MAPPOTrainer(env, policy, config, run, device="cpu")
    needs_scores, resolver_mode = trainer._needs_edge_scores, trainer.resolver_mode

    modes = args.modes.split(",")
    res = {m: [] for m in modes}
    extra = {m: {"generated": [], "served": []} for m in modes}
    for seed in seeds:
        gen = torch.Generator().manual_seed(seed)
        sr, trace, g, sv = run_episode(env, policy, seed, start_base + seed, needs_scores, resolver_mode)
        if "normal" in res:
            res["normal"].append(sr); extra["normal"]["generated"].append(g); extra["normal"]["served"].append(sv)
        labels = oracle_labels(trace, horizon, label_mode)

        def oracle(t, obs, _labels=labels):
            lab = _labels[t] if t < len(_labels) else {}
            v = torch.tensor([lab.get(e, 0.0) for e in obs.physical_edge_ids], dtype=torch.float32)
            return (2.0 * v - 1.0) * ORACLE_LOGIT

        def shuffled(t, obs, _o=oracle, _g=gen):
            v = _o(t, obs)
            return v[torch.randperm(v.numel(), generator=_g)] if v.numel() else v

        for m, fn in (("oracle", oracle), ("shuffled", shuffled)):
            if m in res:
                sr_m, _tr, g, sv = run_episode(env, policy, seed, start_base + seed, needs_scores, resolver_mode, fn)
                res[m].append(sr_m); extra[m]["generated"].append(g); extra[m]["served"].append(sv)
        print(f"seed {seed}: " + "  ".join(f"{m} {res[m][-1]:.4f}" for m in modes), flush=True)

    summary = {m: sum(v) / len(v) for m, v in res.items()}
    print("mean: " + "  ".join(f"{m} {summary[m]:.4f}" for m in modes))
    ref = run / "validation.json"
    if "normal" in res and ref.exists():
        v = json.loads(ref.read_text(encoding="utf-8"))
        ref_map = dict(zip(v.get("seeds", []), v.get("per_seed_success", [])))
        pairs = [(a, ref_map[s]) for a, s in zip(res["normal"], seeds) if s in ref_map]
        if pairs:
            diff = max(abs(a - b) for a, b in pairs)
            print(f"normal vs validation.json 最大逐种子差 {diff:.2e}（应为 0）")
            summary["normal_vs_validation_maxdiff"] = diff
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "run": str(run), "seeds": seeds, "task_fingerprint": fp, "weight": model.actor.usage_bias_weight,
        "horizon": horizon, "label_mode": label_mode, "oracle_logit": ORACLE_LOGIT,
        "per_seed_success": res, "mean": summary, "keys": extra,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
