"""验证 episode_summary 新增的密钥效率维度**真的在动**，而不是恒为 0。

为什么必须有这个探针：这次改动的**失败模式是静默的** ——
`episode_summary()` 原先不返回 `generated_keys`，而调用方写的是
`summary.get("generated_keys", 0.0)`。两边对不上时不会报错，只会一路记 0，
事后看上去"这个指标一直是 0"跟"策略真的一把密钥都不生成"无法区分。
所以判据不能是"脚本跑通了"，必须是**数值非零且与独立来源一致**。

三个判据：
  1. 新键存在（generated_keys / waiting_keys_mean / key_efficiency）
  2. generated_keys 非零，且与逐步累加的 `last["generated_keys"]` 之和一致
  3. waiting_keys_mean 非零且量级合理（存量均值，不该是 0 也不该是总量的量级）

跑法（服务器上）：
    /opt/qkd/venv/bin/python /tmp/probe_metrics_keys.py [ROOT]

ROOT 默认 /opt/qkd/graph_mappo。**在一份 /tmp 下的隔离副本上跑**，
不要动在跑的 run 正在用的包 —— 两个 run 还在跑，改活包风险不对称：
收益只是多记一个字段，代价可能是两条正在跑的曲线报废。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.baselines.path_greedy import PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe

STEPS = 60          # 够看到非零，又不至于占内存
SEED = 100

profile = _tp.load_validation_profile(ROOT / "configs" / "global.yaml")
config = _tp.build_validation_env_config(
    profile, include_baselines=False, episode_steps=STEPS,
    start_mode=profile["start_mode"])

env = build_env_from_config(config)
obs = env.reset(seed=SEED, start_seed=int(profile.get("start_seed", 0)) + SEED)
expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                         principles=False, router=ServeProbe(env))

# 独立来源：从**逐步 info** 自己累加一份，最后与 episode_summary 对照。
gen_manual = 0.0
wait_manual = 0.0
n = 0
done = False
while not done and n < STEPS:
    actions, scores = expert.act(obs)
    obs, _r, terminated, truncated, info = env.step(actions, scores)
    gen_manual += float(info.get("generated_keys", 0.0))
    wait_manual += float(info.get("waiting_keys", 0.0))
    n += 1
    done = terminated or truncated

sm = env.metrics.episode_summary()

print("=" * 64)
print(f"步数 n = {n}")
print("-" * 64)
print(f"{'键':<24}{'summary':>20}{'逐步累加':>20}")
for key, manual in (("generated_keys", gen_manual),):
    have = key in sm
    print(f"{key:<24}{sm.get(key, float('nan')):>20,.2f}{manual:>20,.2f}"
          f"   {'有' if have else '★缺'}")

print(f"{'served_keys':<24}{sm.get('served_keys', 0.0):>20,.2f}{'':>20}")
print(f"{'waiting_keys_mean':<24}{sm.get('waiting_keys_mean', 0.0):>20,.4f}"
      f"{wait_manual / max(1, n):>20,.4f}")
print(f"{'key_efficiency':<24}{sm.get('key_efficiency', 0.0):>20,.4f}")
print("-" * 64)

fails = []
for k in ("generated_keys", "waiting_keys_mean", "key_efficiency"):
    if k not in sm:
        fails.append(f"缺键 {k}")
if sm.get("generated_keys", 0.0) <= 0:
    fails.append("generated_keys 为 0（静默失败！）")
if sm.get("waiting_keys_mean", 0.0) <= 0:
    fails.append("waiting_keys_mean 为 0")
if abs(sm.get("generated_keys", 0.0) - gen_manual) > 1e-6 * max(1.0, gen_manual):
    fails.append(f"generated_keys 与逐步累加不一致: "
                 f"{sm.get('generated_keys')} vs {gen_manual}")
if abs(sm.get("waiting_keys_mean", 0.0) - wait_manual / max(1, n)) > 1e-6 * max(1.0, wait_manual):
    fails.append("waiting_keys_mean 与逐步累加/步数 不一致")
if sm.get("key_efficiency", 0.0) <= 0:
    fails.append("key_efficiency 为 0（served 或 generated 有一边没记）")

print("=" * 64)
if fails:
    print("FAIL")
    for f in fails:
        print("  ✗ " + f)
    sys.exit(1)
print("PASS：新键存在、非零、且与逐步累加的独立来源一致")
