"""判决性检验：RL 训练到底有没有贡献？

问题：所有臂的验证轨迹是 u5 0.6491 → u30 0.6982，而专家是 0.6979。
      **但从没有人测过 BC 起点本身**（`supervised_pg_phased_latest.pt`）。
      如果 BC 起点已经在 0.70 附近，那 30 轮 RL 是在原地打转。

做法：用**和臂完全相同的**运行路径（`build_config` + `MAPPOTrainer`）
      载入 BC checkpoint，调 `evaluate_validation()` —— 与臂内部
      u5/u10/.../u30 那 6 个点走的是**同一个函数、同一批种子**
      （test_protocol 的 15 种子 100–114，天 330–365，240 步）。

对照量（都已存在，只读）：
  BC 起点            ← 本脚本测
  专家               0.6979
  ent01_rerun u30    ≈0.6982
  v2_bottleneck u30  ≈0.6930

判读：
  BC ≈ 0.70 且 ≈ u30  ⟹ **30 轮 RL 零贡献**，应在**起点**上找出路
  BC 明显 < 0.70      ⟹ RL 确实学到了东西，继续在 RL 侧优化

★ 摘掉 sys.path[0]：服务器 `.tmp/` 有 547 个 .py（含 `types.py`），
  从那里启动必然循环导入。见记忆 `server-tmp-has-547-py-shadowing-stdlib`。
"""
import os
import sys
import sys as _sys

_here = _sys.path[0] if _sys.path else ""
if _here and _here not in ("", "."):
    _sys.path[:] = [p for p in _sys.path if p != _here]

import argparse
import json
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("OMP_NUM_THREADS", "8")   # ★ 与臂一致，否则不可比
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
import importlib.util  # noqa: E402

# ---- 复用训练入口的 build_config，保证配置链一模一样 ----
_spec = importlib.util.spec_from_file_location(
    "_train_entry", REPO / "scripts" / "train" / "train_graph_mappo.py")
_train_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_entry)

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="要评测的 .pt；不给就是 BC 起点")
    ap.add_argument("--label", default="bc_start")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    ckpt_path = REPO / ckpt
    if not ckpt_path.exists():
        raise SystemExit(f"起点不存在：{ckpt_path} ⟹ 不许兜底，致命退出")

    # ---- 配置链：与 ent01_rerun / v2_bottleneck 臂逐项相同 ----
    cfg_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"],
        checkpoint=None, seed=42, num_updates=30,
        run_name="_probe_bc_eval", device="cpu", mode="random_episode",
    )
    config = _train_entry.build_config(cfg_args)

    seed = int(config["seed"]["global_seed"])
    torch.manual_seed(seed)
    device = "cpu"
    config.setdefault("runtime", {})
    config["runtime"]["num_threads"] = 8

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, device)

    out_dir = REPO / "outputs" / "_probe_bc_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out_dir, device=device)
    trainer.load_checkpoint(ckpt_path)

    print("=" * 96)
    print(f"BC 起点验证读数 —— {args.label}")
    print("=" * 96)
    print(f"  checkpoint : {ckpt}")
    print(f"  OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  验证协议   : ", end="")
    v = config.get("validation", {})
    print(f"天 {v.get('window',{}).get('start_day')}–{v.get('window',{}).get('end_day')}"
          f"  {v.get('episode_steps')} 步  {len(v.get('request_seeds',[]))} 种子"
          f"  start_mode={v.get('start_mode')}")

    res = trainer.evaluate_validation()
    seeds = res.get("seeds", [])
    per = res.get("per_seed_success", [])
    mean = res.get("mean_success_rate", 0.0)

    print(f"\n  seeds = {seeds}")
    print(f"  per_seed_success = {[round(x, 4) for x in per]}")
    print(f"\n  ★ mean_success_rate = {mean:.6f}")
    print(f"  mean_served_keys    = {res.get('mean_served_keys'):,.0f}")

    payload = {"label": args.label, "checkpoint": ckpt, "seeds": seeds,
               "per_seed_success": per, "mean_success_rate": mean,
               "mean_served_keys": res.get("mean_served_keys"),
               "mean_key_efficiency": res.get("mean_key_efficiency")}
    out = Path(args.out or f"/tmp/probe_bc_eval_{args.label}.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
