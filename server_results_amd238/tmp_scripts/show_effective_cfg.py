"""打印真实训练合并后的有效配置（直接调用训练的 build_config，不重复实现）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "_tg", ROOT / "scripts" / "train" / "train_graph_mappo.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> int:
    argv = sys.argv[1:] or ["--configs", "rl_algorithm.yaml", "train_full_rl.yaml"]
    sys.argv = ["train_graph_mappo.py", *argv]
    config = mod.build_config(mod.parse_args())

    t = config["train"]
    p = t.get("ppo", {})
    print("模式: random_episode   --configs:", " ".join(argv))
    print()
    for k in (
        "episodes_per_update",
        "n_rollout_workers",
        "rollout_batch",
        "rollout_steps",
        "num_updates",
        "gamma",
    ):
        print(f"  train.{k:<22} = {t.get(k)}")
    for k in ("epochs", "minibatch_size", "batch_chunk", "entropy_coef", "value_coef"):
        print(f"  train.ppo.{k:<18} = {p.get(k)}")
    print()
    print(f"  >>> 有效 batch_chunk = {p.get('batch_chunk')}   (512 慢 / 64 快, 见 train_mappo.yaml 实测)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
