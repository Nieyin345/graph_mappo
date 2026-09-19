"""`max_grad_norm` 到底有没有在起作用 —— 直接量裁剪前后的梯度范数。

### 为什么查这个

项目已记的病症是「**actor 被优势塌缩冻住**」：critic 漂移 1.43 / actor 0.047
（30 倍），kl≈0.001，clip 从不激活。已试过的三刀（降 critic_lr、改终止语义、
削 idle_scorer）都没解决；而 `probe_adv_ablation` 三臂共同给出第三条读数：
**关掉归一化、改全局、加大 minibatch —— 三条不同方向的改动都在把 kl 变小**。
⟹ 要找的是**放大更新**的杠杆，而不是继续调保守化旋钮。

`max_grad_norm = 0.5`（`check_knob_coverage.py` 报的「**从未扫过**」之一）
是一个天然的候选：它是**唯一一个能把梯度缩小的乘性闸门**，而且从 0.5 起
只可能往上调（放大）。但**前提是它真的在裁**。

### 判据（跑之前写死）

  · 裁剪**几乎从不触发**（裁前范数绝大多数 < max_norm）
      ⟹ `max_grad_norm` 不是杠杆，调它没用 ⟹ **不要**为它排 run。
  · 裁剪**频繁触发**（裁前范数中位数 >= max_norm，或触发率 > 30%）
      ⟹ 它是**活跃的**限制器 ⟹ 放大更新有机制支撑，值得排一臂。

★ 这是**零训练成本**探针：复用同一份 rollout 做 1 个 update，只装个 spy。

### 为什么这条能省下大量成本

`minibatch` 那条预测已经被证伪一次（见 docs/训练诊断记录.md）。
这次先把「闸门是否真的在闸」量出来，再决定要不要花 3 个 run —— 顺序反过来
就是拿 3 个 run 去赌一个没量过的前提。

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/probe_grad_clip_binding.py --max-norm 0.5
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch                                                    # noqa: E402
import torch.nn.utils as tu                                     # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_tgm", ROOT / "scripts" / "train" / "train_graph_mappo.py")
tgm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tgm)

from qkd_rl.env.factory import build_env_from_config            # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint          # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer          # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy                  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402


def total_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
    return sq ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-norm", type=float, default=0.5,
                    help="要检验的阈值；0 表示用配置里的现值")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1, seed=a.seed,
        checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    cfg["runtime"]["device"] = "cpu"
    cfg["train"]["n_rollout_workers"] = 1
    cfg["train"]["ppo"]["batch_chunk"] = 64
    max_norm = a.max_norm or float(
        cfg["train"]["ppo"].get("max_grad_norm", 0.5))
    cfg["train"]["ppo"]["max_grad_norm"] = max_norm

    torch.manual_seed(a.seed)
    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    data = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu")
    model.load_state_dict(data.model_state)
    out = Path("/tmp/grad_clip_probe")
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, cfg, out, device="cpu")

    policy_ids = {id(p) for p in trainer._policy_params}
    value_ids = {id(p) for p in trainer._value_params}
    rec: dict[str, list[tuple[float, float]]] = {"policy": [], "value": []}

    orig = tu.clip_grad_norm_

    def spy(parameters, mn, *args, **kw):
        params = list(parameters)
        ids = {id(p) for p in params}
        pre = total_norm(params)
        ret = orig(parameters, mn, *args, **kw)
        if ids & policy_ids:
            rec["policy"].append((pre, float(mn)))
        elif ids & value_ids:
            rec["value"].append((pre, float(mn)))
        return ret

    tu.clip_grad_norm_ = spy
    try:
        buf = trainer.collect_rollout()
        stats = trainer.update(buf)
    finally:
        tu.clip_grad_norm_ = orig

    print("=" * 78)
    print("梯度裁剪是否在起作用（零训练成本：1 个 rollout + 1 个 update）")
    print("=" * 78)
    print(f"  max_grad_norm = {max_norm}   种子 {a.seed}   配置链 "
          f"{' '.join(tgm_args.configs)}")
    print(f"  本次 update 的 kl = {float(stats.kl):.5f}")
    print()

    verdict_lines = []
    for name, label in (("policy", "actor+encoder"), ("value", "critic")):
        rows = rec[name]
        if not rows:
            print(f"  {label}: 没有捕获到（该组本步没走裁剪）")
            continue
        pres = [r[0] for r in rows]
        clipped = sum(1 for p, m in rows if p > m)
        rate = 100.0 * clipped / len(rows)
        med = statistics.median(pres)
        print(f"  {label}（{len(rows)} 次 step）")
        print(f"    裁前范数: 中位数 {med:.4f}  最小 {min(pres):.4f}  "
              f"最大 {max(pres):.4f}")
        print(f"    触发裁剪: {clipped}/{len(rows)} = **{rate:.1f}%**"
              f"   （阈值 {max_norm}）")
        # 裁后恒等于 max_norm（超阈值时），所以「裁前中位数 / 阈值」就是倍数
        print(f"    中位数 / 阈值 = {med / max_norm:.3f}"
              f"   （>1 ⟹ 一半以上的步都被裁）")
        print()
        verdict_lines.append((label, rate, med / max_norm))

    print("=" * 78)
    if not verdict_lines:
        print("  没有捕获到任何裁剪调用 ⟹ **不做判读**（可能 spy 没挂上）。")
    else:
        hot = [v for v in verdict_lines if v[1] > 30.0 or v[2] >= 1.0]
        if hot:
            for label, rate, ratio in hot:
                print(f"  ⟹ **{label} 的裁剪是活跃的**（触发 {rate:.1f}%，"
                      f"中位/阈值 {ratio:.2f}）")
            print("     ⟹ 放大更新有机制支撑：`max_grad_norm` 往上调**可能**有效。")
            print("        ★ 但「可能有效」不是「有效」—— 仍要 ≥3 个训练种子配对。")
        else:
            for label, rate, ratio in verdict_lines:
                print(f"  ⟹ {label} 裁剪**几乎不触发**（{rate:.1f}%，"
                      f"中位/阈值 {ratio:.2f}）")
            print("     ⟹ `max_grad_norm` **不是杠杆**，调它没用。")
            print("        **不要**为它排 run —— 这条把它从候选中划掉。")
        print()
        print("  读法提醒：本轮只有 1 个 update，样本量小；但「触发率」是"
              "每步一个观测，")
        print(f"    步数 = {'/'.join(str(len(rec[k])) for k in ('policy', 'value'))}"
              "（policy/critic），比种子数大两个量级。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
