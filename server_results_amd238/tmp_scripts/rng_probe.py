"""构造模型时的 RNG 消耗是否一致？

动机：训练加载 checkpoint 时有一段会被忽略的逻辑 ——

    self.model.load_state_dict(data.model_state)
    if data.config.get("reward") != self.config.get("reward"):
        self.model.critic.value_head.apply(_reset_module)   # <- 从全局 RNG 重新随机初始化

（mappo_trainer.py:1150-1157）。这一步在两个版本里都会执行，但**如果两版模型构造时
消耗掉的随机数个数不同**，走到这里时全局 RNG 的位置就不同，价值头拿到的就是完全
不同的随机权重 —— 那是数量级级别的差异，足以让训练轨迹系统性分叉，而且完全可复现。

它同时能解释另外两件已经确认的事：
  * 相同权重下前后向几乎逐位相同（我的 grad_probe 用的是不带重初始化的原始加载器，
    没走这条路径）；
  * 确定性评估与 critic 无关，所以同一权重下新旧模型的评估逐位相同。

本脚本只回答一件事：构造完模型后，价值头的初始权重是否相同。相同 -> 这条排除。

用法：
    python rng_probe.py --repo <repo> --out <hash.pt>
    python rng_probe.py --compare <a.pt> <b.pt>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch


def build(repo: Path, device: str):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts" / "rl"))

    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
    from train_graph_mappo import build_config

    ns = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        mode="random_episode",
        run_name="rng_probe",
        num_updates=None,
        seed=None,
        checkpoint=None,
        device=device,
    )
    config = build_config(ns)
    torch.manual_seed(int(config["seed"]["global_seed"]))
    torch.set_float32_matmul_precision("high")
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    return model


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def probe(args) -> None:
    repo = Path(args.repo).resolve()
    model = build(repo, args.device)

    after_ctor = torch.get_rng_state().clone()

    info = {
        "repo": str(repo),
        "rng_after_ctor": digest(after_ctor),
        "value_head": [digest(p) for p in model.critic.value_head.parameters()],
        "actor_first": [digest(p) for p in list(model.actor.parameters())[:3]],
        "encoder_first": [digest(p) for p in list(model.encoder.parameters())[:3]],
        "n_params": sum(1 for _ in model.parameters()),
    }
    torch.save(info, args.out)

    print(f"REPO={repo}")
    print(f"  n_params            = {info['n_params']}")
    print(f"  rng_after_ctor      = {info['rng_after_ctor']}")
    print(f"  value_head digests  = {info['value_head']}")
    print(f"  actor[:3] digests   = {info['actor_first']}")
    print(f"  encoder[:3] digests = {info['encoder_first']}")
    print("RNG_PROBE_DONE")


def compare(args) -> None:
    a = torch.load(args.compare[0], map_location="cpu")
    b = torch.load(args.compare[1], map_location="cpu")
    print(f"A = {a['repo']}")
    print(f"B = {b['repo']}")
    for key in ("n_params", "rng_after_ctor", "value_head", "actor_first", "encoder_first"):
        same = a[key] == b[key]
        flag = "一致" if same else "**不同**"
        print(f"  {key:<20} {flag}")
        if not same:
            print(f"      A={a[key]}")
            print(f"      B={b[key]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", nargs=2, metavar=("A.pt", "B.pt"))
    args = ap.parse_args()
    if args.compare:
        compare(args)
    else:
        probe(args)


if __name__ == "__main__":
    main()
