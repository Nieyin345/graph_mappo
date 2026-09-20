"""极性自证：`pool_include_max` 的两个取值**各自都验证**。

## 两件事，分开测

**A. 惰性（false ⟹ 现有臂不受影响）**
   `pool_include_max: false` 时，value_head 输入宽度必须与改动前**逐位相同**
   （387）。若不同，则我改坏了 46 条既有臂 —— 那不是"没效果"，是**污染**。

**B. 生效（true ⟹ 真的多了 max 那一段）**
   宽度必须变成 771（`hidden*3+3 + hidden*3` = 387 + 384），
   且**不是**靠形状碰巧对上 —— 要确认新增段**确实是 max 而非又一次 mean**。
   判据：把同一批 `phys_emb` 喂进去，新段的最后一维应等于
   `phys_emb.max(0).values`（逐位比），而**不等于**其 mean。

★ 这两条合起来才是"改对了"。只测 B 会把"算成了 mean"当成功；
  只测 A 会把"根本没接上"当惰性。
"""
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.models.graph_mappo import (  # noqa: E402
    GraphMAPPOActorCritic, observation_to_tensors, GlobalCritic)

CFG_BASE = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"]

# ★ 训练入口在**模块级**调 torch.set_num_interop_threads(2)，而它必须在
#   任何并行工作开始前调用。所以**先 import 一次**（只会成功一次），
#   之后 build_config 直接用它。
import importlib.util as _iu  # noqa: E402
_spec_te = _iu.spec_from_file_location(
    "_te", REPO / "scripts" / "train" / "train_graph_mappo.py")
_te = _iu.module_from_spec(_spec_te)
_spec_te.loader.exec_module(_te)


def build_cfg(extra=None):
    import argparse
    args = argparse.Namespace(
        configs=CFG_BASE + ([extra] if extra else []),
        checkpoint=None, seed=42, num_updates=30, run_name="_probe_cmax",
        device="cpu", mode="random_episode")
    return _te.build_config(args)


def main():
    env = build_env_from_config(build_cfg())
    obs = env.reset(seed=100, start_seed=100)

    print("=" * 96)
    print("A. 惰性：pool_include_max 缺省（false）⟹ 宽度必须 = 387")
    print("=" * 96)
    cfg_off = build_cfg()
    m_off = GlobalCritic(128, cfg_off["model"])
    w_off = list(m_off.value_head.parameters())[0].shape[1]
    print(f"  pool_include_max = {m_off.pool_include_max}")
    print(f"  value_head 输入宽度 = {w_off}")
    print(f"  {'✓ 与改动前一致' if w_off == 387 else '★ 不一致 ⟹ 污染了既有臂'}")

    print("\n" + "=" * 96)
    print("B. 生效：pool_include_max = true ⟹ 宽度 = 771，且新增段必须是 max")
    print("=" * 96)
    cfg_on = build_cfg("train_cmax.yaml")
    m_on = GlobalCritic(128, cfg_on["model"])
    w_on = list(m_on.value_head.parameters())[0].shape[1]
    print(f"  pool_include_max = {m_on.pool_include_max}")
    print(f"  value_head 输入宽度 = {w_on}  （期望 771）")
    print(f"  {'✓' if w_on == 771 else '★ 不等于 771'}")

    # ★ 关键：新增段是不是 **max**
    torch.manual_seed(0)
    node_emb = torch.randn(20, 128)
    phys = torch.randn(12, 128)
    dem = torch.randn(6, 128)

    # 用 GlobalCritic.forward 的中间量复算（不依赖内部实现细节）
    node_mean = node_emb.mean(0)
    phys_mean = phys.mean(0)
    dem_mean = dem.mean(0)
    node_max = node_emb.max(0).values
    phys_max = phys.max(0).values
    dem_max = dem.max(0).values

    scale = torch.log1p(torch.tensor([20.0, 6.0, 3.0]))
    expected = torch.cat([node_mean, phys_mean, dem_mean, scale,
                          node_max, phys_max, dem_max], dim=-1)
    print(f"\n  手工构造的期望 graph_emb 宽度 = {expected.numel()}  （期望 771）")

    # 用 forward 的**输入**抓 graph_emb：hook 到 value_head 前一层的输入
    captured = {}

    def hook(_mod, inp, _out):
        captured["x"] = inp[0].detach()

    h = m_on.value_head[0].register_forward_hook(hook)
    with torch.no_grad():
        m_on.forward(node_emb, torch.cat([phys, dem], 0),
                     num_physical_directed=12)
    h.remove()

    got = captured["x"].squeeze(0)
    print(f"  实际 graph_emb 宽度 = {got.numel()}")
    same = torch.allclose(got, expected, atol=1e-6)
    print(f"  {'✓ 逐位等于手工的 max 构造' if same else '★ 与 max 构造不符'}")

    # 造反证：确认新增段**不是** mean
    wrong = torch.cat([node_mean, phys_mean, dem_mean, scale,
                       node_mean, phys_mean, dem_mean], dim=-1)
    is_mean = torch.allclose(got, wrong, atol=1e-6)
    print(f"  造反证：等于「新增段是 mean」的构造吗？ {is_mean}"
          f"  {'★ 是 ⟹ 没接上 max' if is_mean else '✓ 否 ⟹ 确实是 max'}")

    # 对照臂：off 时 graph_emb 应恰好 387 且不含 max
    captured.clear()
    h = m_off.value_head[0].register_forward_hook(hook)
    with torch.no_grad():
        m_off.forward(node_emb, torch.cat([phys, dem], 0), num_physical_directed=12)
    h.remove()
    got_off = captured["x"].squeeze(0)
    exp_off = torch.cat([node_mean, phys_mean, dem_mean, scale], dim=-1)
    print(f"\n  对照（off）：宽度 {got_off.numel()}，"
          f"{'✓ 逐位等于改动前的构造' if torch.allclose(got_off, exp_off, atol=1e-6) else '★ 不符'}")

    ok = (w_off == 387 and w_on == 771 and same and not is_mean
          and torch.allclose(got_off, exp_off, atol=1e-6))
    print(f"\n  {'✓ 全部通过：惰性 + 生效 都成立' if ok else '★ 有检查未通过'}")


if __name__ == "__main__":
    main()
