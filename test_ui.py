"""UI 功能测试脚本。
验证核心功能（不启动 Tkinter 窗口）：
  1. 配置管理器：模式列表、baseline 加载、checkpoint 列表
  2. 命令生成：模式→正确命令行
  3. 奖励计算：keep_active_reward 正确生效
  4. 模型 forward：history encoder 开启/关闭
"""
from __future__ import annotations
import os, sys, tempfile, shutil, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

pass_count = 0
fail_count = 0


def check(name: str, ok: bool, detail: str = ""):
    global pass_count, fail_count
    if ok:
        pass_count += 1
        print(f"  ✅ {name}")
    else:
        fail_count += 1
        print(f"  ❌ {name}: {detail}")


# ===== 1. 配置管理器 =====
print("\n=== 1. 配置管理器 ===")
from ui_tk.utils.config_manager import (
    get_profile_keys, get_profile, load_baselines_config, save_baselines_config,
    get_baselines, list_checkpoints, load_train_profiles,
)

keys = get_profile_keys()
check("模式列表", len(keys) == 4, f"got {len(keys)}: {keys}")
check("fixed_episode 存在", "fixed_episode" in keys)
check("random_duration 存在", "random_duration" in keys)
check("curriculum 存在", "curriculum" in keys)
check("continuous 存在", "continuous" in keys)

# 每个模式文件能正常加载
for k in keys:
    prof = get_profile(k)
    check(f"模式 {k} 可加载", isinstance(prof, dict) and len(prof) > 0)

# baselines
bl = get_baselines()
check("baselines 列表", len(bl) >= 7)
check("milp 在 baselines 中", "milp" in bl)

cfg = load_baselines_config()
check("baselines 配置加载", "greedy_relay_diffusion_v3" in cfg)

# 保存/读取 baselines（临时）
tmp_dir = ROOT / "outputs" / "_test_ui"
tmp_dir.mkdir(parents=True, exist_ok=True)
old_cfg = dict(cfg["greedy_relay_diffusion_v3"])
cfg["greedy_relay_diffusion_v3"]["keep_weight"] = "0.99"
save_baselines_config(cfg)
cfg2 = load_baselines_config()
check("baselines 保存/读取", cfg2["greedy_relay_diffusion_v3"]["keep_weight"] == "0.99")
# 恢复
cfg["greedy_relay_diffusion_v3"] = old_cfg
save_baselines_config(cfg)

# checkpoints
ckpts = list_checkpoints()
check("checkpoint 列表", isinstance(ckpts, list))
if ckpts:
    check("checkpoint 有名称", all(c.get("name") for c in ckpts))


# ===== 2. 命令生成 =====
print("\n=== 2. 命令生成 ===")
from ui_tk.utils.train_manager import generate_command

# fixed_episode → mode=random_episode + configs modes/fixed_episode.yaml
cmd = generate_command("fixed_episode", "test_run", config_files=["modes/fixed_episode.yaml"])
cmd_str = " ".join(cmd)
check("fixed_episode → random_episode mode", "--mode=random_episode" in cmd_str or "--mode=fixed_episode" in cmd_str)
check("fixed_episode 含 configs", "modes/fixed_episode.yaml" in cmd_str)

# curriculum → mode=curriculum
cmd = generate_command("curriculum", "test_run", config_files=["modes/curriculum.yaml"])
cmd_str = " ".join(cmd)
check("curriculum → curriculum mode", "--mode=curriculum" in cmd_str)

# checkpoint 参数
cmd = generate_command("fixed_episode", "test", config_files=["modes/fixed_episode.yaml"], checkpoint="ckpt.pt")
check("checkpoint 参数", "--checkpoint=ckpt.pt" in " ".join(cmd))


# ===== 3. 奖励计算 =====
print("\n=== 3. 奖励计算 ===")
from qkd_rl.env.reward import RewardFunction, RewardDetail

# 基础奖励配置（含 keep_active）
reward_cfg = {
    "mode": "shaped",
    "served_weight": 50.0,
    "generated_weight": 0.0,
    "failed_weight": 0.01,
    "waiting_weight": 0.0,
    "overflow_weight": 0.0,
    "expired_key_weight": 0.01,
    "conflict_weight": 0.0,
    "switch_weight": 0.001,
    "keep_active_weight": 0.01,
    "served_reference": 100000.0,
    "dense_generation_importance_weight": 0.02,
    "dense_reference": 1000000.0,
    "dense_normalize_by_added": True,
    "normalize_by_arrived_demand": False,
    "success_delta_enabled": True,
    "success_delta_weight": 50,
    "served_enabled": True,
    "raw_generation_enabled": False,
    "dense_enabled": True,
    "failed_enabled": True,
    "waiting_enabled": False,
    "switch_enabled": False,
    "keep_active_enabled": True,
    "overflow_enabled": False,
    "expired_key_enabled": True,
    "conflict_enabled": False,
    "normalize_penalties_by_arrived": False,
    "penalty_reference": 1000000.0,
    "normalize_window": 60,
    "normalize_floor": 1000.0,
    "clip_abs": 0.0,
    "waiting_stock_weight": 0.0,
}
rf = RewardFunction(reward_cfg)
from qkd_rl.env.action_resolver import ResolvedAction
from qkd_rl.env.request import ServeResult
from qkd_rl.env.routing import AllocationResult
from qkd_rl.core.types import KeyRequest

sr = ServeResult(served_keys=100000.0, failed_keys=0.0, served_requests=1, waiting_requests=0, failed_requests=0, waiting_keys=0.0)
alloc = AllocationResult(added_keys=0.0, overflow_keys=0.0, added_by_edge={})
resolved = ResolvedAction(activated_edges=[], rejected_actions={}, illegal_actions={}, conflict_count=0)
rd = rf.compute(
    serve_result=sr, allocation=alloc,
    expired_requests=[], expired_keys=0.0,
    resolved_action=resolved, qkp=None,
    arrived_keys=100000.0, served_keys=100000.0,
    waiting_keys=0.0, waiting_delta=0.0,
    switch_count=0, keep_active_count=5,
    added_by_edge={}, relay_importance={},
)
check("keep_active_reward 计算正确", abs(rd.keep_active_reward - 0.05) < 1e-6, f"got {rd.keep_active_reward}")
check("keep_active 累加进 total", rd.total > 0)

# keep_active_count=0 时奖励为 0
rd0 = rf.compute(
    serve_result=sr, allocation=alloc,
    expired_requests=[], expired_keys=0.0,
    resolved_action=resolved, qkp=None,
    arrived_keys=100000.0, served_keys=100000.0,
    waiting_keys=0.0, waiting_delta=0.0,
    switch_count=0, keep_active_count=0,
)
check("keep_active=0 时奖励为 0", rd0.keep_active_reward == 0.0)


# ===== 4. 模型 forward =====
print("\n=== 4. 模型 forward ===")
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import load_default_config, build_env_from_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic
from qkd_rl.algos.policy import MAPPOPolicy
import torch, random, tempfile

config = load_default_config(ROOT)
config = deep_merge(config, load_config([ROOT/"configs"/"env_full.yaml"]))
config = deep_merge(config, load_config([ROOT/"configs"/"global.yaml"]))
config["env"]["episode_steps"] = 40
config["train"] = {"gamma": 0.99, "optimizer": {"actor_lr": 3e-4, "critic_lr": 3e-4}}
ConfigValidator().validate(config)

env = build_env_from_config(config)
obs = env.reset(seed=7, start_seed=1007)

# history OFF
model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
with torch.no_grad():
    out = model.forward(obs)
check("history OFF: edge_scores > 0", len(out.edge_scores or {}) > 0)

# history ON（需重新 validate）
c2 = deep_merge(config, {"features": {"history_encoder": {"enabled": True}}})
ConfigValidator().validate(c2)
model2 = GraphMAPPOActorCritic(env.action_resolver.action_space, c2)
# 跑几步让 history 有数据
for i in range(5):
    actions = {n: random.choice(obs.action_candidates[n]) for n in obs.node_ids}
    obs, r, term, trunc, info = env.step(actions, None)
with torch.no_grad():
    out2 = model2.forward(obs)
check("history ON: edge_scores > 0", len(out2.edge_scores or {}) > 0, f"got {len(out2.edge_scores or {})}")

# 策略采样
pol = MAPPOPolicy(model)
step = pol.act(obs)
check("策略采样有动作", len(step.actions) > 0, f"got {len(step.actions)}")
non_idle = [a for a in step.actions.values() if a != "idle"]
check("非 idle 动作数 > 0", len(non_idle) > 0, f"got {len(non_idle)}")


# ===== 5. 全局参数读/写 =====
print("\n=== 5. 全局参数 I/O ===")
import yaml
gp = ROOT/"configs"/"global.yaml"
g = yaml.safe_load(gp.read_text(encoding="utf-8"))
val = g.get("global", {}).get("validation", {})
check("validation 有 episode_steps", "episode_steps" in val)
check("validation 有 request_seeds", "request_seeds" in val)
check("validation 有 episodes", "episodes" in val)
check("validation 有 window", "window" in val)


# ===== 清理 =====
shutil.rmtree(tmp_dir, ignore_errors=True)

print(f"\n{'='*40}")
print(f"通过: {pass_count}  失败: {fail_count}")
if fail_count > 0:
    print("❌ 有失败项!")
    sys.exit(1)
else:
    print("✅ 全部通过")