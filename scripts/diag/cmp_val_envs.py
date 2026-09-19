#!/usr/bin/env python
"""两条验证路径构造出的 env 配置**真的一样吗**？

`ent01`（entropy_coef=0.01，30 轮）看起来在 u25/u30 **超过专家**
（跨训练种子 t=+4.09 / +3.03，平台合并 t=+3.46）。但专家与 RL 的验证环境是
**两条不同的代码路径**造出来的：

- 专家：`scripts/eval/eval_expert.py` → `test_protocol.build_validation_env_config()`
  （`load_default_config` 起底 → 叠 `env_full.yaml` → 强制 `rate_provider=h5`）
- RL  ：`MAPPOTrainer._build_validation_envs()` → `copy.deepcopy(self.config)`
  （**训练用的完整链**，再改若干字段）

**若两者造出的环境不同，那个"超过"就是两个不同任务上的数，结论作废。**
（本项目记录里"假的配对"就是这一类：数算对了，两侧口径不同。）

做法：把 trainer 送给 `build_env_from_config` 的那份 config **截获**下来
（monkeypatch 成记录器，**不真的建环境**，所以不吃内存、不占服务器），
与专家路径的 config 递归逐字段比。

用法（服务器上）：python /tmp/cmp_val_envs.py
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))

# ---------- 公共：递归 diff ----------
def diff(a, b, path="") -> list[tuple[str, object, object]]:
    out: list[tuple[str, object, object]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append((f"{path}.{k}", "<缺>", b[k]))
            elif k not in b:
                out.append((f"{path}.{k}", a[k], "<缺>"))
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if a != b:
            out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))
    return out


# ---------- 1. 专家那侧 ----------
_tp_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp)

# 专家的 JSON 名为 expert_seeds100_240 —— 那 15 个种子**不是**从 profile 读来的：
# `load_validation_profile` 只看 `global.validation`，而 train_full_rl.yaml 的
# `validation:` 是顶层键，读进去 seeds=[]。所以当初是
#   eval_expert.py --seeds 100-114 --steps 240
# 用命令行给的。**环境配置与种子列表无关**，所以 profile 取 global.yaml 即可。
prof_path = ROOT / "configs" / "global.yaml"
prof = _tp.load_validation_profile(prof_path)
prof = dict(prof, seeds=list(range(100, 115)))     # 环境不受影响，仅为打印口径

print(f"专家侧 profile: {prof_path.relative_to(ROOT)}"
      f"（种子是 --seeds 100-114 从命令行给的，环境配置与种子无关）")
print(f"  steps={prof['episode_steps']}  start_mode={prof['start_mode']}  "
      f"window={prof['window_start_day']}-{prof['window_end_day']}")
print()

expert_cfg = _tp.build_validation_env_config(
    prof, include_baselines=False,
    episode_steps=int(prof["episode_steps"]),
    start_mode=prof["start_mode"])

# ---------- 2. RL 那侧：喂**该 run 真正用过的** self.config ----------
# 首选 outputs/<run>/resolved_config.yaml —— 那是训练器落盘的 `self.config`，
# **就是** ent01 三个种子实际吃的配置。用它比"自己重建一遍"强：重建要么漏了
# 某个 --configs 项，要么 argparse 默认值对不上，两者都会造假差异。
import qkd_rl.env.factory as _factory                                  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer                 # noqa: E402
from qkd_rl.core.config import load_config                             # noqa: E402

RUN = "ent01_s42"
rc_path = ROOT / "outputs" / RUN / "resolved_config.yaml"
if rc_path.exists():
    base = load_config([rc_path])
    print(f"RL 侧：用 {rc_path.relative_to(ROOT)}（该 run 实际落盘的 self.config）")
else:
    import scripts.train.train_graph_mappo as T                        # noqa: E402

    cli = argparse.Namespace(
        mode="random_episode",
        configs=["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"],
        seed=42, num_updates=30, run_name=RUN, device=None)
    base = T.build_config(cli)
    print(f"RL 侧：{RUN} 无 resolved_config.yaml，改由 build_config 重建（次优）")

CAPTURED: list[dict] = []
_real_build = _factory.build_env_from_config


def _capture(config):
    CAPTURED.append(copy.deepcopy(config))
    return None                     # 不真建环境：不吃内存、不占服务器


_factory.build_env_from_config = _capture
# trainer 里是函数内 `from qkd_rl.env.factory import build_env_from_config`，
# 只改 factory 模块属性即可，函数内的 import 每次都会重新取到它。


class _Stub:
    """只为调用真实方法而存在的壳：_build_validation_envs 只用这三个属性。"""
    config = base
    validation_cfg = base.get("validation", {}) or {}
    _validation_envs = None


MAPPOTrainer._build_validation_envs(_Stub.__new__(_Stub))
_factory.build_env_from_config = _real_build

if not CAPTURED:
    raise SystemExit("!! 没截获到 config —— 说明 _build_validation_envs 没走 factory")
print(f"  validation_cfg.request_seeds = {_Stub.validation_cfg.get('request_seeds')}")
print(f"  seed.env_seed = {CAPTURED[0]['seed'].get('env_seed')}")
print(f"  env.episode_steps = {CAPTURED[0]['env'].get('episode_steps')}")
print(f"  env.activation_window = "
      f"{CAPTURED[0]['env'].get('activation_window_start_day')}-"
      f"{CAPTURED[0]['env'].get('activation_window_end_day')}")
print()

# ---------- 3. 逐字段比 ----------
d = diff(expert_cfg, CAPTURED[0])

# 只关心"会不会改变验证结果"的字段；纯训练超参不同不影响环境
#
# 注意：diff() 拼出的 path 带**前导点**（`"" + "." + "env"` → `.env.continuous`）。
# 第一版直接 startswith("env.") 于是**一条都没匹配上**，把所有差异全归到
# "不影响环境"，打出假的"完全一致"。教训：分类函数自己也可能是 bug 源，
# 分类完后要看两侧计数是否合理（这里 23 处差异不可能全是训练超参）。
ENV_RELEVANT = ("env", "scenario", "rate_provider", "seed", "network",
                "qkd", "topology", "requests")
irrelevant = [x for x in d if x[0].lstrip(".").split(".")[0] not in ENV_RELEVANT]
relevant = [x for x in d if x[0].lstrip(".").split(".")[0] in ENV_RELEVANT]

print("=== 环境相关字段的差异（这些会改变验证结果）===")
if not relevant:
    print("  （无）—— 两条路径造出的验证环境**完全一致**")
else:
    for k, a, b in relevant:
        print(f"  {k}")
        print(f"      专家 = {a!r}")
        print(f"      RL   = {b!r}")
print()
print(f"=== 训练超参差异（不影响环境，共 {len(irrelevant)} 处，略）===")
for k, a, b in irrelevant[:12]:
    print(f"  {k}: {a!r} → {b!r}")
if len(irrelevant) > 12:
    print(f"  ...（另有 {len(irrelevant) - 12} 处）")
print()

print("=== 结论 ===")
if not relevant:
    print("  两条路径在验证窗口 / 回合长度 / 起始模式 / 时间上限 / 速率源 上**逐字段一致**")
    print("  → ent01 与专家的对比是**同一任务上的**，'超过' 的结论口径成立。")
else:
    print("  差异（逐条判定是否影响行为）：")
    print("   - .env.continuous        专家侧键缺失 → env.py:47 `.get(...,False)` → False；")
    print("                            RL 侧显式 False。**同值，只是键在不在**。")
    print("   - .env.episode_start_day 专家侧缺失 → env.py:109 `.get(...,-1)` → -1；")
    print("                            RL 侧显式 -1。env.py:110 `if >=0` → 两边都不覆盖。**同效**。")
    print("   - .seed.env_seed         只在 factory.py 建 RequestGenerator 时用一次；")
    print("                            env.py:64 `reset(seed=..)` 会 `request_generator.seed(seed)`")
    print("                            **覆盖**它。两边 reset 都传同一个种子 → 请求流相同。")
    print("  ⇒ 三者都是**表象差异**（键在不在 / 被覆盖的默认值）。但'推理'不是证明，")
    print("     下面**真建两个环境跑一次**，逐比特比。")

# ---------- 4. 实证：真建两边环境，**全部 15 个种子** + 走满一回合比服务量 ----------
print()
print("=== 4. 实证等价 ===")
import numpy as np                                                 # noqa: E402
from qkd_rl.env.factory import build_env_from_config as _build     # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy           # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe                # noqa: E402


def _flat(o):
    """把 GraphObservation 摊平成 {名字: ndarray}。"""
    out = {}
    for k, v in (o.__dict__ if hasattr(o, "__dict__") else {}).items():
        if isinstance(v, np.ndarray):
            out[k] = v
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], np.ndarray):
            out[k] = np.concatenate([np.asarray(x).ravel() for x in v])
        elif isinstance(v, (int, float, bool)):
            out[k] = np.asarray([v])
    return out


SEEDS = list(range(100, 115))

# --- 4a. 15 个种子逐个 reset，比观测 ---
n_bad = 0
for s in SEEDS:
    env_e = _build(expert_cfg)
    env_r = _build(CAPTURED[0])
    # 两侧调用方（eval_expert.py:87 / mappo_trainer.py:1034）都传
    # reset(seed=s, start_seed=0+s)
    oe = _flat(env_e.reset(seed=s, start_seed=s))
    orr = _flat(env_r.reset(seed=s, start_seed=s))
    for k in sorted(set(oe) | set(orr)):
        a, b = oe.get(k), orr.get(k)
        if a is None or b is None or a.shape != b.shape or not np.array_equal(a, b):
            n_bad += 1
            print(f"  ✗ seed {s} 字段 {k} 不等")
    if env_e.t != env_r.t:
        n_bad += 1
        print(f"  ✗ seed {s} 起始 t: {env_e.t} vs {env_r.t}")
    del env_e, env_r
print(f"  4a. {len(SEEDS)} 个种子 × 全部观测字段 + 起始 t："
      f"{'✓ 全部逐比特相同' if n_bad == 0 else f'✗ {n_bad} 处不等'}")

# --- 4b. 同一条专家轨迹走满 240 步，比服务成功率（**正是被测量的那个量**）---
print("  4b. 专家策略走满 240 步，比 success_rate / served / arrived：")
_ap = ROOT / "outputs" / "eval" / "expert_seeds100_240.json"
ARCHIVED: dict[int, float] = {}
if _ap.exists():
    _d = json.loads(_ap.read_text(encoding="utf-8"))
    ARCHIVED = {int(a): float(b) for a, b in zip(_d["seeds"], _d["success"])}
mismatch: list = []
for s in SEEDS[:5]:
    res = {}
    for tag, cfg in (("专家侧", expert_cfg), ("RL侧", CAPTURED[0])):
        env = _build(cfg)
        obs = env.reset(seed=s, start_seed=s)
        pol = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                              principles=False, router=ServeProbe(env))
        done = False
        while not done:
            # 照抄 scripts/eval/eval_expert.py:96-98 的调用形状
            actions, scores = pol.act(obs)
            obs, _r, term, trunc, _i = env.step(actions, scores)
            done = term or trunc
        sm = env.metrics.episode_summary()
        res[tag] = (round(float(sm.get("success_rate", 0.0)), 12),
                    sm.get("served_keys"), sm.get("arrived_keys"))
        del env
    ok = res["专家侧"] == res["RL侧"]
    if not ok:
        mismatch.append(s)
    # 顺带做**正向对照**：专家侧的 success_rate 必须等于归档 JSON 里的值。
    # 否则说明我这套 walk 本身没复现出当初的测量，那么"两边相同"也不足为证
    # （可能两边都以同样的方式错了）。
    arch = ARCHIVED.get(s)
    tag = "—"
    if arch is not None:
        tag = "✓复现" if abs(res["专家侧"][0] - arch) < 1e-9 else f"✗≠归档{arch:.4f}"
    print(f"    seed {s}: 专家侧 {res['专家侧']}  RL侧 {res['RL侧']}  "
          f"{'✓' if ok else '✗'}  归档对照 {tag}")

print()
print("=== 最终结论 ===")
if n_bad == 0 and not mismatch:
    print("  **实证一致**：两条路径造出的验证环境在全部 15 个种子上观测逐比特相同，")
    print("  且同一条专家轨迹的服务成功率完全相同。")
    print("  → ent01 与专家的对比是**同一任务上的比较**，'超过' 的口径成立。")
else:
    print("  !! 实证不一致 → 'ent01 超过专家' 必须撤回或改述。")


