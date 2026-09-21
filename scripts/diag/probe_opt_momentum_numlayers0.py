"""修正版：`num_layers: 0` 时 Adam 动量真的保留了吗？

## 上一版的三个缺陷（自曝）

① **忘了装回去**：`_upgrade_optimizer_state_for_model` 只**返回** state，
   调用方要赋 `self.optimizer.state = st`。我没赋 ⟹ 查到的是空 `o3.state`。
② **空真**：所有计数三项全 0（0/0/0）说明那段循环**一次都没进分支**，
   而不是"全部相同"。这正是记忆 `unavailable-must-not-look-like-a-value` 的坑。
③ 因此上一版打印的"✓ Adam 动量没丢"是**无证据的**。

## 本版

① 真的赋 `opt.state = st`
② **打印输入**（有多少个参数对象、多少个带矩），让"0"能被区分成
   "确实没有" 还是 "没查到"
③ 造反证：`num_layers=3` 时共享参数必须有矩（否则测法本身坏了）
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "_te", REPO / "scripts" / "train" / "train_graph_mappo.py")
_te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_te)

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import (  # noqa: E402
    _upgrade_state_dict_for_model, _upgrade_optimizer_state_for_model,
    build_param_groups)

BASE = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"]
CKPT = REPO / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"


def build(layers, name):
    cfg = _te.build_config(argparse.Namespace(
        configs=BASE, mode="random_episode", run_name=name,
        num_updates=30, seed=42, checkpoint=None, device="cpu"))
    cfg["model"]["encoder"]["num_layers"] = layers
    return cfg


def make(layers):
    cfg = build(layers, f"o{layers}")
    env = build_env_from_config(cfg)
    m = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    groups, _, _ = build_param_groups(
        m, cfg["train"]["optimizer"]["actor_lr"],
        cfg["train"]["optimizer"]["critic_lr"])
    return m, torch.optim.Adam(groups)


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    opt_state = ck.get("optimizer_state")
    print("=" * 90)
    print("输入自证（不打印输入的空真判据不能用）")
    print("=" * 90)
    print(f"  checkpoint: {CKPT.name}")
    print(f"  optimizer_state 存在: {opt_state is not None}")
    if opt_state is None:
        print("  ★ 没有 optimizer_state ⟹ 测不了"); return
    print(f"  checkpoint 顶层键: {sorted(ck.keys())}")

    results = {}
    for L in (3, 0):
        m, opt = make(L)
        n_params = len(list(m.parameters()))
        st, msgs = _upgrade_optimizer_state_for_model(m, opt, opt_state)
        print(f"\n{'='*90}")
        print(f"num_layers={L}   模型参数 {n_params} 个")
        print("=" * 90)
        print(f"  返回 state: {'非空' if st else '**None**'}")
        for msg in msgs[:4]:
            print(f"    {msg}")
        if st:
            opt.state = st          # ★★ 上一版漏了这一步
        # 自证：装完后有多少参数带矩
        with_mom = sum(1 for p in m.parameters() if p in opt.state)
        print(f"  ★ 装完后**带矩的参数**数: {with_mom} / {n_params}")

        # 逐名字比
        named = dict(m.named_parameters())
        shared = [k for k in named if not k.startswith("encoder.layers.")]
        snap = {}
        for name in shared:
            s = opt.state.get(named[name])
            if s and "exp_avg" in s:
                snap[name] = s["exp_avg"].clone()
        results[L] = snap
        print(f"  ★ 其中**非 layers 的共享参数**带矩: {len(snap)} / {len(shared)}")

    # 关键比较
    print(f"\n{'='*90}")
    print("判决")
    print("=" * 90)
    a, b = results.get(3, {}), results.get(0, {})
    print(f"  L=3 共享参数带矩: {len(a)}    L=0 共享参数带矩: {len(b)}")
    if not a:
        print("  ★ L=3 都没有矩 ⟹ **测法本身坏了**（造反证失败），结论无效")
        return
    common = sorted(set(a) & set(b))
    same = [k for k in common if torch.equal(a[k], b[k])]
    diff = [k for k in common if not torch.equal(a[k], b[k])]
    only3 = sorted(set(a) - set(b))
    print(f"  两边都有矩的共享参数: {len(common)}")
    print(f"    逐位相同: {len(same)}")
    print(f"    **不同**: {len(diff)}  {diff[:4]}")
    print(f"    仅 L=3 有（L=0 丢了）: {len(only3)}  {only3[:4]}")
    print()
    if only3 or diff:
        print("  ★ **共享部分的 Adam 动量有丢失** ⟹ §3.3 第 0 条的否决**仍成立**")
    else:
        print("  ✓ **共享部分的 Adam 动量逐位保留** ⟹ 否决**已过期**，`num_layers:0` 是单变量")


if __name__ == "__main__":
    main()
