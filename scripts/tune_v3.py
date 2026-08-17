"""Systematic tuning: vary relay_importance params AND v3 weights, find balance.

Overrides use a compact format:
    hop=0.5 cap=0.5 minscar=0.1 typebonus=off imp=20 keep=1.0

Keys:
  relay_importance: hop_decay_factor, capacity_strength, min_scarcity,
    link_type_bonus (on/off), max_path_links
  v3: importance, rate, completion, keep, switch, wait

Each run: 1 episode x 240 steps, seed 7, validation window.
"""
from __future__ import annotations
import os, sys, csv, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml


def main():
    ov = {}
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            ov[k.strip()] = v.strip()

    # --- mutate features.yaml relay_importance ---
    fp = ROOT / "configs" / "features.yaml"
    feat = yaml.safe_load(fp.read_text(encoding="utf-8"))
    ri = feat["features"]["edge"]["relay_importance"]
    if "hop" in ov: ri["hop_decay_factor"] = float(ov["hop"])
    if "cap" in ov: ri["capacity_decay_strength"] = float(ov["cap"])
    if "minscar" in ov: ri["min_scarcity"] = float(ov["minscar"])
    if "path" in ov: ri["max_path_links"] = int(ov["path"])
    if "typebonus" in ov:
        if ov["typebonus"] == "off":
            ri.pop("link_type_bonus", None)
        else:
            ri["link_type_bonus"] = {"SAT-SAT": float(ov["typebonus"]),
                                     "GS-SAT": 1.2, "GS-HAP": 1.0, "HAP-SAT": 1.0}
    fp.write_text(yaml.safe_dump(feat, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # --- mutate baselines.yaml v3 ---
    bp = ROOT / "configs" / "baselines.yaml"
    bl = yaml.safe_load(bp.read_text(encoding="utf-8"))
    v3 = dict(bl["baselines"]["greedy_relay_diffusion_v3"])
    for k, v in ov.items():
        if k in ("importance", "rate", "completion", "keep", "switch", "wait"):
            v3[{"importance": "importance_weight", "rate": "rate_weight",
                "completion": "completion_weight", "keep": "keep_weight",
                "switch": "switch_weight", "wait": "wait_urgency_tau_ratio"}[k]] = v
    bl["baselines"]["greedy_relay_diffusion_v3"] = v3
    bp.write_text(yaml.safe_dump(bl, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"relay_importance: {dict(ri)}")
    print(f"v3: {dict(v3)}")

    from qkd_rl.core.config import (ConfigValidator, deep_merge, load_config)
    from qkd_rl.env.factory import load_default_config, build_env_from_config
    from qkd_rl.baselines.greedy_relay_diffusion import GreedyRelayDiffusionPolicyV3

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "global.yaml"]))
    config["env"]["episode_steps"] = 240
    config["env"]["activation_window_start_day"] = 330
    config["env"]["activation_window_end_day"] = 365
    config["env"]["activation_window_days"] = 35
    config["scenario"]["time_limit"]["days"] = 366
    ConfigValidator().validate(config)

    env = build_env_from_config(config)
    pol = GreedyRelayDiffusionPolicyV3(
        rate_weight=float(v3.get("rate_weight", 1.0)),
        importance_weight=float(v3.get("importance_weight", 10.0)),
        completion_weight=float(v3.get("completion_weight", 1.0)),
        keep_weight=float(v3.get("keep_weight", 0.5)),
        switch_weight=float(v3.get("switch_weight", 0.2)),
        hop_decay_factor=float(ri.get("hop_decay_factor", 0.25)),
        max_path_links=int(ri.get("max_path_links", 3)),
        wait_urgency_tau_ratio=float(v3.get("wait_urgency_tau_ratio", 0.8)),
        ignore_consumption=str(v3.get("ignore_consumption", "False")).lower() == "true",
        include_stocked_unavailable=str(v3.get("include_stocked_unavailable", "True")).lower() == "true",
    )
    obs = env.reset(seed=7, start_seed=1007)
    steps = 0
    t0 = time.perf_counter()
    while steps < 240:
        actions, scores = pol.act(obs)
        obs, reward, term, trunc, info = env.step(actions, scores)
        steps += 1
        if term or trunc:
            break
    s = env.metrics.episode_summary()
    sr = s["success_rate"]
    print(f"SENTINEL: SR={sr:.4f} served={s['served_keys']:,.0f} arrived={s['arrived_keys']:,.0f} ({time.perf_counter()-t0:.0f}s)")
    csvp = ROOT / "outputs" / "v3_tuning.csv"
    header = ["params", "sr", "served", "arrived", "failed", "steps"]
    write_header = not csvp.exists()
    with open(csvp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([json.dumps(ov), sr, s["served_keys"], s["arrived_keys"], s["failed_keys"], steps])


if __name__ == "__main__":
    main()