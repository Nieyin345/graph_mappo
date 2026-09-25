"""Measure v3 BC action mismatch on the saved expert observations.

This is an offline diagnostic: it does not step an environment or change a
checkpoint. Run on the training server where ``day_*.pkl.gz`` is available.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.model_zoo import build_model
from qkd_rl.rl.algos.checkpoint import load_checkpoint
from qkd_rl.rl.algos.policy import MAPPOPolicy
from scripts.train.supervised_train_pg_phased import _obs_class, rebuild_obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--days", nargs="+", type=int, default=[0, 10, 20, 30, 40])
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(8)
    ckpt = load_checkpoint(args.checkpoint, args.device)
    config = ckpt.config
    env = build_env_from_config(config)
    model = build_model(env.action_resolver.action_space, config).to(args.device)
    model.load_state_dict(ckpt.model_state)
    model.eval()
    policy = MAPPOPolicy(model, device=args.device)
    obs_cls = _obs_class(str(config.get("experiment", {}).get("model", "v3")))

    counts = {key: 0 for key in (
        "steps", "expert_arcs", "filtered_arcs", "steps_with_filtered", "empty_score_maps",
        "missing_path_masks", "exact_sets", "expert_first_top1", "expert_first_available",
        "pred_arcs", "common_arcs", "expert_empty", "pred_empty", "false_stop",
        "missed_arcs_after_stop", "extra_arcs", "invalid_expert_sequence",
        "first_pick_in_expert", "steps_with_wrong_first_pick",
        "wrong_when_expert_feasible", "wrong_after_expert_exhausted",
        "feasible_expert_at_stop", "first_wrong_position_sum", "first_wrong_steps",
    )}
    per_day = []
    with torch.no_grad():
        for day in args.days:
            path = Path(args.data_dir) / f"day_{day:04d}.pkl.gz"
            with gzip.open(path, "rb") as fh:
                records = pickle.load(fh)
            selected = records[::args.stride][:args.limit]
            day_counts = {key: 0 for key in counts}
            for start in range(0, len(selected), 8):
                batch = selected[start : start + 8]
                obs = [rebuild_obs(rec, obs_cls) for rec in batch]
                output = model.batched_forward(obs, torch.device(args.device))
                for rec, edge_map in zip(batch, output.edge_score_maps):
                    c = day_counts
                    c["steps"] += 1
                    expert = [tuple(arc) for arc in rec["arcs"]]
                    c["expert_arcs"] += len(expert)
                    c["expert_empty"] += int(not expert)
                    c["missing_path_masks"] += int(not rec.get("pair_path_edge_masks"))
                    filtered = [arc for arc in expert if arc not in edge_map]
                    c["filtered_arcs"] += len(filtered)
                    c["steps_with_filtered"] += int(bool(filtered))
                    if not edge_map:
                        c["empty_score_maps"] += 1
                        continue
                    usable = [arc for arc in expert if arc in edge_map]
                    if usable:
                        c["expert_first_available"] += 1
                        first = usable[0]
                        best = max(edge_map, key=lambda arc: float(edge_map[arc]))
                        c["expert_first_top1"] += int(first == best)
                    try:
                        policy._matching_log_prob_entropy_fast(edge_map, usable)
                    except ValueError:
                        c["invalid_expert_sequence"] += 1
                        continue
                    _, predicted, _, _ = policy._sample_matching(edge_map, deterministic=True, build_scores=False)
                    e_set, p_set = set(usable), set(predicted)
                    if predicted:
                        c["first_pick_in_expert"] += int(predicted[0] in e_set)
                        c["steps_with_wrong_first_pick"] += int(predicted[0] not in e_set)
                    c["pred_arcs"] += len(predicted)
                    c["common_arcs"] += len(e_set & p_set)
                    c["exact_sets"] += int(e_set == p_set)
                    c["pred_empty"] += int(not predicted)
                    c["false_stop"] += int(bool(usable) and not predicted)
                    c["missed_arcs_after_stop"] += len(e_set - p_set)
                    c["extra_arcs"] += len(p_set - e_set)
                    chosen = []
                    first_wrong = None
                    for position, arc in enumerate(predicted):
                        feasible_expert = [
                            candidate for candidate in e_set - set(chosen)
                            if all(
                                candidate[0] != prev[0]
                                and candidate[1] != prev[1]
                                and {candidate[0], candidate[1]} != {prev[0], prev[1]}
                                for prev in chosen
                            )
                        ]
                        if arc not in e_set:
                            if feasible_expert:
                                c["wrong_when_expert_feasible"] += 1
                            else:
                                c["wrong_after_expert_exhausted"] += 1
                            if first_wrong is None:
                                first_wrong = position
                        chosen.append(arc)
                    c["first_wrong_steps"] += int(first_wrong is not None)
                    c["first_wrong_position_sum"] += first_wrong or 0
                    c["feasible_expert_at_stop"] += sum(
                        all(
                            candidate[0] != prev[0]
                            and candidate[1] != prev[1]
                            and {candidate[0], candidate[1]} != {prev[0], prev[1]}
                            for prev in chosen
                        )
                        for candidate in e_set - p_set
                    )
            per_day.append({"day": day, **day_counts})
            for key in counts:
                counts[key] += day_counts[key]
            print(json.dumps(per_day[-1], ensure_ascii=False), flush=True)
    print(json.dumps({"total": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
