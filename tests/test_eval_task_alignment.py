"""RL 验证与专家评测必须在**同一个任务**上打分（2026-09-27 审计）。

实测失败：trainer 的验证继承训练 config（pairhot_v2 + arrival_rate），而
eval_expert 从 default + env_full 拼配置（legacy_v1 请求分布）。同样 15 个种子，
1× 下专家读数 0.7674（legacy）vs 0.7803（pairhot_v2）——发表的 RL−专家差
因此被高估 0.013。本测试钉住：

1. trainer 验证环境 config == test_protocol 唯一实现（逐字等价于旧逻辑）；
2. eval_expert --run 与 trainer 验证的任务指纹相同；
3. 不加 --run 的旧路径指纹**不同**（即：指纹能抓出这类错配）；
4. 叠加 pair_seed 配置会改变指纹，并真正改变请求配置。
"""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from qkd_rl.core.config import deep_merge, load_config
from qkd_rl.env.factory import load_default_config
from qkd_rl.evaluation import test_protocol as tp

ROOT = Path(__file__).resolve().parents[1]

VALIDATION = {
    "window": {"start_day": 330, "end_day": 365},
    "request_seeds": list(range(100, 115)),
    "episodes": 15,
    "episode_steps": 240,
    "start_mode": "random_day",
}


def _run_config(*overlays: str) -> dict:
    cfg = load_default_config(ROOT)
    for name in ("env_full.yaml",) + overlays:
        cfg = deep_merge(cfg, load_config([ROOT / "configs" / name]))
    cfg["validation"] = deepcopy(VALIDATION)
    return cfg


def _old_trainer_config(config: dict, vcfg: dict) -> dict:
    """冻结的旧 trainer._build_validation_envs 配置拼装（重构前逐字抄录）。"""
    window = vcfg.get("window", {}) or {}
    start_day = int(window.get("start_day", 0))
    end_day = int(window.get("end_day", 365))
    episode_steps = int(vcfg.get("episode_steps", 0) or 0)
    day_steps = int(config["env"].get("day_steps", 1440))
    out = deepcopy(config)
    out["env"]["episode_start_mode"] = str(vcfg.get("start_mode", "random_day"))
    out["env"]["episode_steps"] = episode_steps
    out["env"]["continuous"] = False
    out["env"]["activation_window_start_day"] = start_day
    out["env"]["activation_window_end_day"] = end_day
    out["env"]["activation_window_days"] = max(0, end_day - start_day)
    out["scenario"]["time_limit"]["days"] = end_day + max(1, math.ceil(episode_steps / day_steps))
    return out


def _trainer_validation_config(config: dict, monkeypatch) -> tuple[dict, list[int]]:
    """真调 MAPPOTrainer._build_validation_envs，截获它交给 env 工厂的 config。"""
    from qkd_rl.env import factory
    from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer

    captured: list[dict] = []
    monkeypatch.setattr(factory, "build_env_from_config",
                        lambda c: captured.append(deepcopy(c)) or object())
    stub = SimpleNamespace(config=config, validation_cfg=config["validation"],
                           _validation_envs=None)
    MAPPOTrainer._build_validation_envs(stub)
    assert captured, "trainer 没有构建验证环境"
    return captured[0], stub._validation_seeds


def _write_run(tmp_path: Path, config: dict) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "experiment_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return run


def test_trainer_validation_uses_shared_impl_equal_to_old_logic(monkeypatch):
    cfg = _run_config("env_full_demand_v2.yaml", "load_x20.yaml")
    got, seeds = _trainer_validation_config(cfg, monkeypatch)
    want = _old_trainer_config(cfg, cfg["validation"])
    got.get("seed", {}).pop("env_seed", None)
    want.get("seed", {}).pop("env_seed", None)
    assert got == want
    assert seeds == list(range(100, 115))


def test_expert_run_task_matches_trainer_validation(tmp_path, monkeypatch):
    from scripts.eval.eval_expert import build_expert_task

    cfg = _run_config("env_full_demand_v2.yaml", "load_x20.yaml")
    trainer_cfg, trainer_seeds = _trainer_validation_config(cfg, monkeypatch)
    expert_cfg, expert_seeds, _ = build_expert_task(run=str(_write_run(tmp_path, cfg)))

    assert expert_seeds == trainer_seeds
    assert expert_cfg["requests"]["distribution_version"] == "pairhot_v2"
    assert expert_cfg["requests"]["arrival_rate"] == trainer_cfg["requests"]["arrival_rate"]
    assert tp.task_fingerprint(expert_cfg) == tp.task_fingerprint(trainer_cfg), (
        "专家 --run 与 RL 验证不在同一任务上——配对无效")


def test_legacy_expert_path_is_a_different_task(tmp_path, monkeypatch):
    """旧路径（无 --run）= legacy_v1：指纹必须不同，否则指纹抓不住这次的错配。"""
    from scripts.eval.eval_expert import build_expert_task

    cfg = _run_config("env_full_demand_v2.yaml")
    trainer_cfg, _ = _trainer_validation_config(cfg, monkeypatch)
    legacy_cfg, _, _ = build_expert_task(seeds_spec="100-114")
    assert legacy_cfg["requests"].get("distribution_version", "legacy_v1") == "legacy_v1"
    assert tp.task_fingerprint(legacy_cfg) != tp.task_fingerprint(trainer_cfg)


def test_pair_seed_overlay_changes_task(tmp_path):
    from scripts.eval.eval_expert import build_expert_task

    run = _write_run(tmp_path, _run_config("env_full_demand_v2.yaml"))
    base, _, _ = build_expert_task(run=str(run))
    held, seeds, _ = build_expert_task(run=str(run), seeds_spec="1000-1002",
                                       overlays=["pair_seed1.yaml"])
    assert base["requests"]["pair_seed"] == 0
    assert held["requests"]["pair_seed"] == 1
    assert seeds == [1000, 1001, 1002]
    assert tp.task_fingerprint(held) != tp.task_fingerprint(base)


def test_evaluate_one_refuses_to_overwrite_validation_json(tmp_path):
    from scripts.experiment import evaluate_one

    run = _write_run(tmp_path, _run_config("env_full_demand_v2.yaml"))
    with pytest.raises(ValueError, match="--tag"):
        evaluate_one(run, seeds=[1000, 1001])


def test_rh_milp_preview_equals_env_arrivals_without_perturbing_env():
    """RH-MILP 的先知预览必须等于 env 实际到达，且不能推进 env 自己的请求流。"""
    from qkd_rl.env.factory import build_env_from_config
    from scripts.eval.eval_expert import build_policy

    cfg = tp.run_validation_env_config(_run_config("env_full_demand_v2.yaml"), VALIDATION)
    env = build_env_from_config(cfg)
    obs = env.reset(seed=101, start_seed=101)
    t0 = int(obs.state.t)
    pol = build_policy("rh_milp", env, obs, cfg)
    assert pol._gen is not env.request_generator
    preview = [(t, r.src_gs, r.dst_gs, round(r.amount, 6)) for t, r in pol._preview_requests(t0, 15)]

    seen = []
    orig = env.request_generator.generate

    def rec(t):
        out = orig(t)
        seen.extend((t, r.src_gs, r.dst_gs, round(r.amount, 6)) for r in out)
        return out

    env.request_generator.generate = rec
    for _ in range(15):
        env.step({}, None)
    assert preview, "15 槽内应有到达"
    assert preview == seen
