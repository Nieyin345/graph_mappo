"""★ 决定性造反证：A1 修复版 vs 原版，**同进程对打**。

### 为什么不能只跑「修复后」的探针

`probe_a1_wiring.py` 在**未修复**的服务器代码上已经跑出红态：
`encoder.activation=gelu` 而 actor/critic **仍是 ReLU** —— A1 实证成立。
但那只证明「原版坏」，没证明「修复版好且惰性」。

而且我**不能**把修复 sync 到服务器去测：peer 正在跑 hist32
（记忆 `chain-must-not-be-resynced`：分波链在跑时 sync 会让 A/B 变伪 A/B）。
所以这里用**两不碰**的办法：

1. **不碰服务器代码**：在场内用两份 `__init__` 的**忠实复刻**对比
2. **但复刻必须自证忠实**：拿**服务器上真实的（未修复的）类**去校验
   `old_variant` 复刻 —— 若两者不一致，说明我的复刻是错的，全部结论作废

★ 这一步是关键：没有它，"我复刻的 old 版"只是**我以为是**的 old 版
（记忆 `never-run-code-path-hides-bugs`：断言必须能被造反证）。

### 三条判据

| # | 判据 | 期望 |
|---|---|---|
| A | 复刻忠实性 | `old_variant` 复刻 与 服务器真实类 **逐位相同**；`fixed_variant` 与之**不同** |
| B | 惰性（当前 relu/0.0） | `fixed_variant` 与 `old_variant` 的模块序列**相同**、state_dict **逐位相同** ⟹ 暖启动安全 |
| C | 旋钮能动（gelu/0.5） | `fixed_variant` 变 GELU + 出现 Dropout；`old_variant` **仍是 ReLU**（这就是 bug） |

用法（服务器上）：
    /opt/qkd/venv/bin/python -u /tmp/probe_a1_confront.py
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))

import torch                                                        # noqa: E402
from torch import nn                                                # noqa: E402

from qkd_rl.rl.models.graph_mappo import (                          # noqa: E402
    SharedNodeActor, GlobalCritic, build_mlp)


def build_cfg(extra: dict | None = None):
    from scripts.train import train_graph_mappo as tgm              # noqa: E402
    args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", seed=7, num_updates=1,
        run_name="probe_a1", device="cpu")
    cfg = tgm.build_config(args)
    if extra:
        for dotted, val in extra.items():
            node = cfg
            for p in dotted.split(".")[:-1]:
                node = node.setdefault(p, {})
            node[dotted.split(".")[-1]] = val
    return cfg


# --------------------------------------------------------------------------
# 两份复刻：只有「从哪里读 activation/dropout」这一处不同
# --------------------------------------------------------------------------
def _actor_common(m, hidden_dim, config, activation, dropout):
    """两个复刻共用的尾部（与真实类逐字对应）。

    ★ `stop_logit = nn.Parameter(torch.zeros(()))` 必须复刻 —— 它是真实类
    state_dict 里的一个键（`:446`）。`torch.zeros` **不消耗 RNG**，所以
    在它之前播种的效果与真实类一致（播种只影响 `build_mlp` 的 `nn.Linear`）。
    """
    m.mode = config.get("mode", "mixed")
    m.edge_scorer = build_mlp(hidden_dim * 3,
                              list(config["actor"]["edge_scorer_hidden_dims"]),
                              1, activation, dropout)
    m.idle_scorer = build_mlp(hidden_dim,
                              list(config["actor"]["idle_scorer_hidden_dims"]),
                              1, activation, dropout)
    m.temperature = float(config["actor"].get("temperature", 1.0))
    m.stop_logit = nn.Parameter(torch.zeros(()))
    return m


def fixed_actor(hidden_dim, config, invalid_logit_value):
    """修复版：从 config['encoder'] 读（与磁盘上改动后的源码逐字对应）。"""
    enc = config.get("encoder", {})
    return _actor_common(nn.Module(), hidden_dim, config,
                         enc.get("activation", "relu"),
                         float(enc.get("dropout", 0.0)))


def old_actor(hidden_dim, config, invalid_logit_value):
    """原版：从 config **顶层**读（磁盘改动前的行为）。"""
    return _actor_common(nn.Module(), hidden_dim, config,
                         config.get("activation", "relu"),
                         float(config.get("dropout", 0.0)))


def fixed_critic(hidden_dim, config):
    m = nn.Module()
    enc = config.get("encoder", {})
    activation = enc.get("activation", "relu")
    dropout = float(enc.get("dropout", 0.0))
    m.pooling = config["critic"].get("pooling", "mean")
    vdim = hidden_dim * 3 + 3 if m.pooling == "typed_mean" else hidden_dim * 2
    m.value_head = build_mlp(vdim, list(config["critic"]["hidden_dims"]),
                             1, activation, dropout)
    return m


def old_critic(hidden_dim, config):
    m = nn.Module()
    activation = config.get("activation", "relu")
    dropout = float(config.get("dropout", 0.0))
    m.pooling = config["critic"].get("pooling", "mean")
    vdim = hidden_dim * 3 + 3 if m.pooling == "typed_mean" else hidden_dim * 2
    m.value_head = build_mlp(vdim, list(config["critic"]["hidden_dims"]),
                             1, activation, dropout)
    return m


def seq(mod) -> list[str]:
    return [type(x).__name__ for x in mod]


def act_of(mod) -> str:
    for x in mod:
        if isinstance(x, (nn.ReLU, nn.GELU, nn.Tanh)):
            return type(x).__name__
    return "None"


def build_seeded(fn, *a):
    torch.manual_seed(20260921)
    return fn(*a)


def main() -> int:
    a = argparse.ArgumentParser().parse_args()
    fails: list[str] = []

    cfg = build_cfg()
    mcfg = copy.deepcopy(cfg["model"])
    hidden = int(mcfg["encoder"]["hidden_dim"])
    ilv = float(mcfg["distribution"]["invalid_logit_value"])
    print("=" * 76)
    print("A1 同进程对打：修复版 vs 原版（不碰服务器代码）")
    print(f"  encoder.activation={mcfg['encoder'].get('activation')!r}  "
          f"encoder.dropout={mcfg['encoder'].get('dropout')!r}")
    print(f"  model 顶层有 activation 吗: {'activation' in mcfg}  "
          f"有 dropout 吗: {'dropout' in mcfg}")
    print("=" * 76)

    # ---------------- 判据 A：复刻忠实性 ----------------
    # ★ 忠实性的正确判据是「真实类 == old 复刻」，**在多个配置下都成立**。
    #   我第一版写成「fix 复刻必须与真实类不同」是**错的** —— 当前配置是
    #   relu/0.0，此时 fix ≡ old 本就是判据 B 要证的事，「相同」是必然的。
    #   一个判据在正确世界和错误世界里返回同一个值，它就是坏的（读数异常
    #   先怪方法，别怪对象）。判别力来自**换配置**：gelu 下 old 必须仍等于
    #   真实类（复刻跟着配置走），而 fix 必须与之不同。
    print("\n[A] 复刻忠实性 —— 拿服务器上**真实的**（未修复的）类校验 old 复刻")
    real_a = build_seeded(SharedNodeActor, hidden, copy.deepcopy(mcfg), ilv)
    old_a = build_seeded(old_actor, hidden, copy.deepcopy(mcfg), ilv)
    fix_a = build_seeded(fixed_actor, hidden, copy.deepcopy(mcfg), ilv)
    sd_real, sd_old, sd_fix = real_a.state_dict(), old_a.state_dict(), fix_a.state_dict()
    same_real_old = all(torch.equal(sd_real[k], sd_old[k]) for k in sd_real)
    same_real_fix = all(torch.equal(sd_real[k], sd_fix[k]) for k in sd_real)
    print(f"    relu 配置下 真实类 vs old 复刻 : "
          f"{'逐位相同 ✓' if same_real_old else '✗ 不同'}")
    if not same_real_old:
        fails.append("old 复刻与服务器真实类不同 ⟹ 复刻不忠实，全部结论作废")

    # gelu 配置下的忠实性（判别力所在）
    gcfg0 = copy.deepcopy(cfg["model"])
    gcfg0["encoder"]["activation"] = "gelu"
    gcfg0["encoder"]["dropout"] = 0.5
    real_ag = build_seeded(SharedNodeActor, hidden, copy.deepcopy(gcfg0), ilv)
    old_ag = build_seeded(old_actor, hidden, copy.deepcopy(gcfg0), ilv)
    fix_ag = build_seeded(fixed_actor, hidden, copy.deepcopy(gcfg0), ilv)
    g_real_old = all(torch.equal(real_ag.state_dict()[k], old_ag.state_dict()[k])
                     for k in real_ag.state_dict())
    # ★ gelu+dropout 下不能用 state_dict 逐位比 —— 开 dropout 会让
    #   `nn.Dropout` 插进序列，`edge_scorer.2` 从 Linear 变 Dropout，
    #   **键名会平移**。此处要比的是「结构」，所以用模块序列。
    g_real_old = (seq(real_ag.edge_scorer) == seq(old_ag.edge_scorer)
                  and all(torch.equal(real_ag.state_dict()[k], old_ag.state_dict()[k])
                          for k in real_ag.state_dict()))
    print(f"    gelu 配置下 真实类 vs old 复刻 : "
          f"{'逐位相同 ✓（复刻跟着配置走）' if g_real_old else '✗ 不同'}")
    print(f"    gelu 配置下 真实类激活={act_of(real_ag.edge_scorer)}  "
          f"fix 复刻激活={act_of(fix_ag.edge_scorer)}  "
          f"（真实类**仍是 ReLU**，正是未修复的表现）")
    if not g_real_old:
        fails.append("gelu 配置下 old 复刻与真实类不同 ⟹ 复刻不忠实")
    if act_of(real_ag.edge_scorer) != "ReLU":
        fails.append("gelu 配置下真实类竟不是 ReLU ⟹ 服务器可能已修，对打无意义")
    if act_of(fix_ag.edge_scorer) != "GELU":
        fails.append("gelu 配置下 fix 复刻没变成 GELU ⟹ 修法不成立")

    # ---------------- 判据 B：惰性（当前 relu/0.0） ----------------
    print("\n[B] 惰性（当前 relu/0.0）—— 暖启动安全的依据")
    print(f"    old 序列: {seq(old_a.edge_scorer)}")
    print(f"    fix 序列: {seq(fix_a.edge_scorer)}")
    same_seq = seq(old_a.edge_scorer) == seq(fix_a.edge_scorer)
    same_sd = all(torch.equal(sd_old[k], sd_fix[k]) for k in sd_old)
    same_keys = set(sd_old) == set(sd_fix)
    print(f"    模块序列相同: {same_seq}   state_dict 键集相同: {same_keys}   "
          f"逐位相同: {same_sd}")
    if not (same_seq and same_sd and same_keys):
        fails.append("当前配置下 fix 与 old 不等价 ⟹ **不是**惰性修复，必须 A/B 才能上线")
    else:
        print("    ✓ 当前配置下逐位相同 ⟹ 可安全 sync（不污染 peer 的 hist32、暖启动无损）")

    # critic 侧同样
    real_c = build_seeded(GlobalCritic, hidden, copy.deepcopy(mcfg))
    old_c = build_seeded(old_critic, hidden, copy.deepcopy(mcfg))
    fix_c = build_seeded(fixed_critic, hidden, copy.deepcopy(mcfg))
    c_same = all(torch.equal(old_c.state_dict()[k], fix_c.state_dict()[k])
                 for k in old_c.state_dict())
    c_real_old = all(torch.equal(real_c.state_dict()[k], old_c.state_dict()[k])
                     for k in real_c.state_dict())
    print(f"    critic: 真实类 vs old 复刻 {'逐位相同 ✓' if c_real_old else '✗ 不同'}"
          f"，old vs fix {'逐位相同 ✓' if c_same else '✗ 不同'}")
    if not c_real_old:
        fails.append("critic 的 old 复刻不忠实")
    if not c_same:
        fails.append("critic 在当前配置下 fix 与 old 不等价 ⟹ 非惰性")

    # ---------------- 判据 C：旋钮能动 ----------------
    print("\n[C] 旋钮能动（encoder.activation=gelu, dropout=0.5）")
    gcfg = copy.deepcopy(cfg["model"])
    gcfg["encoder"]["activation"] = "gelu"
    gcfg["encoder"]["dropout"] = 0.5
    go_a = build_seeded(old_actor, hidden, copy.deepcopy(gcfg), ilv)
    gf_a = build_seeded(fixed_actor, hidden, copy.deepcopy(gcfg), ilv)
    gf_c = build_seeded(fixed_critic, hidden, copy.deepcopy(gcfg))
    print(f"    old actor  激活={act_of(go_a.edge_scorer):<5} 序列={seq(go_a.edge_scorer)}")
    print(f"    fix actor  激活={act_of(gf_a.edge_scorer):<5} 序列={seq(gf_a.edge_scorer)}")
    print(f"    fix critic 激活={act_of(gf_c.value_head):<5} 序列={seq(gf_c.value_head)}")
    if act_of(gf_a.edge_scorer) != "GELU":
        fails.append("gelu 配置下 fix 的 actor 仍非 GELU")
    if act_of(gf_c.value_head) != "GELU":
        fails.append("gelu 配置下 fix 的 critic 仍非 GELU")
    if not any(isinstance(x, nn.Dropout) for x in gf_a.edge_scorer):
        fails.append("dropout=0.5 下 fix 的 actor 没有 Dropout")
    if act_of(go_a.edge_scorer) != "ReLU":
        fails.append("gelu 配置下 old 的 actor 竟不是 ReLU —— 反证不成立")
    else:
        print("    ✓ old 在 gelu 配置下**仍死死钉在 ReLU** ⟹ 这就是那个静默 bug")

    print("\n" + "=" * 76)
    if fails:
        print(f"✗ 对打未通过，{len(fails)} 条：")
        for f in fails:
            print(f"    - {f}")
    else:
        print("✓ 对打通过：复刻忠实、修复惰性、旋钮能动")
    print("=" * 76)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
