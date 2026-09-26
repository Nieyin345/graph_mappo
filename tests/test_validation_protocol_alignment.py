"""The canonical validation protocol must be ONE thing, not three.

The failure this prevents (measured 2026-09-26):

Three places defined "the validation window" and they disagreed:

    configs/global.yaml validation        seeds [7..11],   5 episodes   <- fed the
                                                                          trainer's
                                                                          eval_validation
                                                                          AND every baseline
    configs/train_full_rl.yaml validation seeds [100..114], 15 episodes
    scripts/eval/frozen_eval.py WINDOWS   seeds [100..114], 15 episodes  <- the only one
                                                                          used for cross-arm
                                                                          comparison

Same window (330-365), same 240 steps, same expert -- but the expert scores
0.8643 on seeds 7-11 and 0.7368 on seeds 100-114, a 12.75-point gap produced
entirely by which request streams are drawn (a fixed-seed sweep of the start day
from 30 to 360 is flat, so the day is not the cause). That made the trainer's
internal 0.86 and the frozen 0.76 look like two different regimes when they are
two different seed sets.

frozen_eval.py is frozen ("do not modify -- new protocol = new file"), so the
alignment is achieved by pointing global.yaml at the SAME protocol. This test
pins the two together: if either drifts, it fails.
"""

from __future__ import annotations

from pathlib import Path

from qkd_rl.evaluation.test_protocol import load_validation_profile
from scripts.eval.frozen_eval import EPISODE_STEPS, START_MODE, WINDOWS

ROOT = Path(__file__).resolve().parents[1]


def test_global_validation_matches_frozen_window_a():
    """global.yaml's default validation must equal frozen_eval window A."""
    prof = load_validation_profile(ROOT / "configs" / "global.yaml")
    spec = WINDOWS["A"]

    assert prof["window_start_day"] == spec["start_day"], (
        f"global.yaml 验证窗口起点 {prof['window_start_day']} != frozen 窗A "
        f"{spec['start_day']}"
    )
    assert prof["window_end_day"] == spec["end_day"], (
        f"global.yaml 验证窗口终点 {prof['window_end_day']} != frozen 窗A "
        f"{spec['end_day']}"
    )
    assert prof["seeds"] == list(spec["request_seeds"]), (
        "global.yaml 的验证请求种子 ≠ frozen 窗A —— 两组种子的专家水平实测相差 "
        "12.75 点（7–11 得 0.864 vs 100–114 得 0.737），对不上就等于在比两个"
        "不同的东西。\n"
        f"  global.yaml: {prof['seeds']}\n"
        f"  frozen 窗A : {list(spec['request_seeds'])}"
    )
    assert prof["episode_steps"] == EPISODE_STEPS, (
        f"global.yaml 验证回合长度 {prof['episode_steps']} != frozen {EPISODE_STEPS}"
    )
    assert prof["episodes"] == len(spec["request_seeds"]), (
        f"global.yaml episodes={prof['episodes']} != frozen 窗A 种子数 "
        f"{len(spec['request_seeds'])}"
    )
    assert prof["start_mode"] == START_MODE, (
        f"global.yaml start_mode={prof['start_mode']!r} != frozen {START_MODE!r}"
    )
