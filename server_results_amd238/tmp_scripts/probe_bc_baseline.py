"""给 BC 权重取一个"起点"基线，而且要和 r2_* 各变体**同一套协议**。

为什么不直接用 run_baselines.py：那个走的是 `MAPPOTrainer.evaluate` —— 种子循环、
步数来源、是否确定性都不一定和训练器里的 `evaluate_validation` 一致。而 r2 各变体
metrics.jsonl 里记的是 `eval_validation`。协议不同就没法按种子配对，
"涨没涨"这个判断会立刻退回到 0.06 的不配对噪声里。

所以这里直接复用训练器自己的 `evaluate_validation`，配置也走同一个 `build_config`，
保证和 `--configs rl_algorithm.yaml train_diag_fast.yaml` 逐字一致。

用法（节点上）：
    cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python .tmp/probe_bc_baseline.py \
        [checkpoint] [输出 json]
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rl.train_graph_mappo import build_config  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
)
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/eval/bc_diag_perseed.json"


def main() -> None:
    # 与 screen_r2.sh 里训练用的命令行完全同构，只把 --num-updates 设成 0
    # （根本不进训练循环，下面直接调验证）。
    args = Namespace(
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        mode="random_episode",
        run_name="bc_baseline_eval",
        num_updates=0,
        seed=None,
        checkpoint=None,
        device="cpu",
    )
    config = build_config(args)

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, "cpu")

    out_dir = ROOT / "outputs" / "bc_baseline_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out_dir, device="cpu")
    trainer.load_checkpoint(Path(CKPT))

    episodes = int(config["validation"].get("episodes", 12) or 12)
    result = trainer.evaluate_validation(num_episodes=episodes)

    payload = {
        "checkpoint": CKPT,
        "episodes": episodes,
        "seeds": result["seeds"],
        "per_seed_success": result["per_seed_success"],
        "mean_success_rate": result["mean_success_rate"],
        "mean_reward": result["mean_reward"],
    }
    Path(OUT).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"checkpoint   : {CKPT}")
    print(f"种子         : {payload['seeds']}")
    print(f"逐种子成功率 : {[round(v, 4) for v in payload['per_seed_success']]}")
    print(f"均值         : {payload['mean_success_rate']:.4f}")
    print(f"已写         : {OUT}")
    print("BC_BASELINE_DONE")


if __name__ == "__main__":
    main()
