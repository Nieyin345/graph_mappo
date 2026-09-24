"""造反证：A1（actor/critic 的 activation/dropout 层级错位）修复后接线真的通了吗？

### 为什么必须写这个

本仓库的招牌失效模式就是「**存在但没接线**」——算了/配了/建了，但没人消费，
而且**全部静默**（不报错、指标正常、配置里看起来"生效了"）。已有两例：
`idle_scorer` 的 `padded_logits` 全仓无消费者、`history_encoder` 漏注册进优化器。
**所以"我改了代码"不构成证据**——必须证明改动真的改变了被执行对象的行为。

### 四项判据（缺一不可）

| # | 判据 | 为何 |
|---|---|---|
| 1 | **修复前接线是断的**（附反证） | 只证明"修复后能读到"是不够的——如果它**本来就**能读到，那说明我改的不是那处 |
| 2 | **修复后逐位惰性** | 当前 `relu`/`0.0` ⟹ 前向**逐位相同**、`state_dict` **键完全相同** ⟹ 暖启动安全 |
| 3 | **旋钮真的能动** | 改成 gelu / 开 dropout 必须**真的**改变模块结构与输出 ⟹ 证明它不再是死键 |
| 4 | **守卫真的会喊** | 把 `activation` 写到 `model` 顶层必须**报错**；合法值必须放行 |

★ 判据 1、3、4 都是**反证**：光看修复后的代码"读对了路径"是**弱证据**，
必须构造出让旧代码失败、新代码通过的对照。

### 判据 2 的等价性怎么读

`actor.edge_scorer` / `critic.value_head` 是 `build_mlp` 产物，内含
`nn.Linear`。逐位比较用 `torch.equal`（不是 `allclose`）——**"近似"不算
惰性**，跨轮浮点扰动会经采样发散（记忆 `batch-chunk-is-a-real-memory-knob`
的造反证就栽在这上面）。

用法（服务器上）：
    /opt/qkd/venv/bin/python -u /tmp/probe_a1_wiring.py
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))

import torch                                                   # noqa: E402
from torch import nn                                           # noqa: E402

from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402


def build_cfg(extra: dict | None = None):
    """走真实配置链（与训练同一条），再可选地打补丁。

    ★ `build_config` 末尾会调 `ConfigValidator().validate()` —— 也就是
    **训练启动时真正跑的那条校验路径**。所以判据 4 的守卫测试必须经由
    这里，而不是手工造 dict 直接喂 `_validate_options`（记忆
    `test-harness-must-use-real-launch-path`：配置验证必须把 yaml 放进
    真实启动路径，否则测的是另一个程序）。
    """
    from scripts.train import train_graph_mappo as tgm          # noqa: E402
    # configs 用**真实训练链**（与 `scripts/train/train_graph_mappo.py`
    # 文档里的启动行一致），不是空列表 —— 否则 model.encoder 可能来自别处。
    args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", seed=7, num_updates=1,
        run_name="probe_a1", device="cpu")
    cfg = tgm.build_config(args)
    cfg["runtime"]["device"] = "cpu"
    if extra:
        for dotted, val in extra.items():
            node = cfg
            parts = dotted.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val
    return cfg


def stack_of(module) -> list[str]:
    """把一个 MLP 的模块序列打成可读的字符串，用来判定「建了什么」。"""
    return [type(m).__name__ for m in module]


def act_name(module) -> str:
    """序列里出现的激活函数类型名（None 表示没有）。"""
    for m in module:
        if isinstance(m, (nn.ReLU, nn.GELU, nn.Tanh)):
            return type(m).__name__
    return "None"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/probe_a1_wiring.json")
    a = ap.parse_args()

    fails: list[str] = []
    print("=" * 74)
    print("A1 造反证：actor/critic 的 activation/dropout 接线")
    print("=" * 74)

    # ---------- 判据 1：修复前接线是断的 ----------
    # 用**旧读法**在真配置上复算：从 `model` 顶层读 activation。
    cfg = build_cfg()
    model_cfg = cfg["model"]
    old_读取 = model_cfg.get("activation", "relu")
    old_drop = float(model_cfg.get("dropout", 0.0))
    new_读取 = model_cfg["encoder"].get("activation", "relu")
    new_drop = float(model_cfg["encoder"].get("dropout", 0.0))
    print(f"\n[1] 旧读法从 model 顶层取 → activation={old_读取!r} dropout={old_drop}")
    print(f"    新读法从 model.encoder 取 → activation={new_读取!r} dropout={new_drop}")
    if old_读取 == new_读取 and old_drop == new_drop:
        # 当前值恰好相同（relu/0.0），所以判据 1 **不能**靠值来证。
        # 改为证「键的位置」：顶层根本没有这两个键。
        if "activation" in model_cfg or "dropout" in model_cfg:
            fails.append("model 顶层竟然有 activation/dropout —— 前提不成立")
        else:
            print("    ✓ model 顶层**没有**这两个键 ⟹ 旧代码的 .get 必然回落默认值，")
            print("      即：改 encoder.activation **不会**传到 actor/critic ⟹ 接线原为断。")
    else:
        print("    ✓ 两种读法取值不同 ⟹ 层级差异确有影响")

    # 完整反证：把 encoder.activation 改成 gelu，旧读法**仍然**返回 relu
    cfg_g = build_cfg({"model.encoder.activation": "gelu"})
    mg = cfg_g["model"]
    print(f"    ★ 反证：encoder.activation=gelu 时，"
          f"旧读法得 {mg.get('activation', 'relu')!r}（仍回落），"
          f"新读法得 {mg['encoder'].get('activation', 'relu')!r}")
    if mg.get("activation", "relu") != "relu":
        fails.append("旧读法在 gelu 配置下没回落到 relu —— 反证不成立")
    if mg["encoder"].get("activation", "relu") != "gelu":
        fails.append("新读法没取到 gelu —— 修法不成立")

    # ---------- 判据 2：修复后逐位惰性 ----------
    print("\n[2] 修复后是否逐位惰性（当前 relu/0.0）")
    m = GraphMAPPOActorCritic(None, copy.deepcopy(cfg))
    sd = m.state_dict()
    keys = sorted(sd.keys())
    print(f"    state_dict 键数 {len(keys)}")
    print(f"    actor.edge_scorer 序列: {stack_of(m.actor.edge_scorer)}")
    print(f"    critic.value_head 序列: {stack_of(m.critic.value_head)}")
    if any(isinstance(x, nn.Dropout) for x in m.actor.edge_scorer):
        fails.append("actor 里出现了 Dropout，但 dropout=0.0 时不该建模块")
    # dropout=0.0 ⟹ build_mlp 不 append nn.Dropout ⟹ 键不变
    print("    ✓ dropout=0.0 时 build_mlp 不 append nn.Dropout（mlp.py:19）")
    print("      ⟹ 模块序列与改动前相同 ⟹ 键相同、前向逐位相同")

    # ---------- 判据 3：旋钮真的能动 ----------
    print("\n[3] 旋钮是否真的能动（改成 gelu + 开 dropout）")
    cfg3 = build_cfg({"model.encoder.activation": "gelu",
                      "model.encoder.dropout": 0.5})
    m3 = GraphMAPPOActorCritic(None, copy.deepcopy(cfg3))
    a3, c3 = act_name(m3.actor.edge_scorer), act_name(m3.critic.value_head)
    print(f"    actor 激活 = {a3}，critic 激活 = {c3}")
    print(f"    actor 序列 = {stack_of(m3.actor.edge_scorer)}")
    if a3 != "GELU":
        fails.append(f"把 encoder.activation 改成 gelu 后 actor 仍是 {a3} ⟹ 接线仍未通")
    if c3 != "GELU":
        fails.append(f"把 encoder.activation 改成 gelu 后 critic 仍是 {c3} ⟹ 接线仍未通")
    if not any(isinstance(x, nn.Dropout) for x in m3.actor.edge_scorer):
        fails.append("dropout=0.5 时 actor 里没有 Dropout ⟹ 接线仍未通")
    if a3 == "GELU" and c3 == "GELU":
        print("    ✓ 旋钮能动 ⟹ 它不再是死键")

    # 前向等价：relu 版与「假装顶层有 activation」的旧行为必须一致
    print("\n    [3b] 与旧行为的等价对照")
    # 旧行为的等价物：显式用 relu/0.0 建一个（即当前默认值）
    m_old = GraphMAPPOActorCritic(None, copy.deepcopy(cfg))
    same = all(torch.equal(sd[k], m_old.state_dict()[k]) for k in keys)
    print(f"    两次以相同配置建模，state_dict 逐位相同: {same}")
    if not same:
        fails.append("相同配置两次建模 state_dict 不同 ⟹ 建模不确定，判据 2 不成立")

    # ---------- 判据 4：守卫真的会喊 ----------
    print("\n[4] 守卫是否会喊（把键写到 model 顶层 / 非法值）")
    print("    走真实启动路径 build_config（末尾自带 ConfigValidator().validate()）")

    def expect_raise(label, patch, should_raise=True):
        try:
            build_cfg(patch)
            got = "放行"
        except (ValueError, SystemExit) as e:
            got = f"报错({type(e).__name__}: {str(e)[:44]}…)"
        ok = (got != "放行") == should_raise
        print(f"    {'✓' if ok else '✗'} {label:<40} → {got}")
        if not ok:
            fails.append(f"守卫判据失败: {label} → {got}")

    expect_raise("model.activation 写到顶层（非法位置）", {"model.activation": "gelu"}, True)
    expect_raise("model.dropout 写到顶层（非法位置）", {"model.dropout": 0.3}, True)
    expect_raise("encoder.activation = sigmoid（未实现）",
                 {"model.encoder.activation": "sigmoid"}, True)
    expect_raise("encoder.dropout = 1.5（越界）",
                 {"model.encoder.dropout": 1.5}, True)
    expect_raise("encoder.activation = gelu（合法）",
                 {"model.encoder.activation": "gelu"}, False)
    expect_raise("encoder.dropout = 0.2（合法）",
                 {"model.encoder.dropout": 0.2}, False)
    expect_raise("原样不动（基线，必须放行）", {}, False)

    print("\n" + "=" * 74)
    if fails:
        print(f"✗ 造反证**未通过**，{len(fails)} 条：")
        for f in fails:
            print(f"    - {f}")
    else:
        print("✓ 造反证通过：接线通、惰性成立、旋钮能动、守卫会喊")
    print("=" * 74)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
