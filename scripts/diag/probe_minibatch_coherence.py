"""minibatch 大小 → 梯度一致性：检验一条**由探针推导出的**预测。

### 推论链（不是盲扫）

`probe_adv_ablation.py` 否掉了「归一化」这一支，并把病因指到更上游：

  真实优势已趋零（|A| 1.09→0.91）⟹ 弱信号
  而 buffer 被切成**45 个跨回合随机子集**（minibatch 256，11520/256）
  ⟹ 每个子集各自指向不同方向（cos 在第 4 个 minibatch 就翻号）

**预测**：若病因是「子集太多、每个太小」，那么**加大 minibatch**
（子集变少变大）应当让**相邻梯度更一致**。

| minibatch_size | 子集数 | 预测 |
|---|---|---|
| 256（现状） | 45 | 基线 |
| 512 | 22 | 一致性↑ |
| 1024 | 11 | 一致性↑↑ |

这**不是**在扫一个随机旋钮 —— `minibatch_size` 是
`check_knob_coverage.py` 报的「从未扫过」之一，而本探针给的是
**有机制、可证伪**的检验。且 `configs/train_safe_mini512.yaml` 已预注册。

### 判据（跑之前写死）

  · 相邻 cos 随 minibatch 增大而**单调上升** ⟹ 预测成立，mini512 有机制支撑
  · 不升（或反而下降） ⟹ 预测被证伪，**别拿 mini512 去跑 30 轮**

★ 零训练成本：每档只做 1 个 update，不写 checkpoint。

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/probe_minibatch_coherence.py --sizes 256,512,1024
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch                                                    # noqa: E402
import torch.nn.utils as tu                                     # noqa: E402

_tgm_spec = importlib.util.spec_from_file_location(
    "_tgm", ROOT / "scripts" / "train" / "train_graph_mappo.py")
tgm = importlib.util.module_from_spec(_tgm_spec)
_tgm_spec.loader.exec_module(tgm)

from qkd_rl.env.factory import build_env_from_config            # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint          # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer          # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy                  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402


def run_arm(size: int, cfg: dict, seed: int = 42) -> dict:
    import copy
    c = copy.deepcopy(cfg)
    c["train"]["ppo"]["minibatch_size"] = size

    torch.manual_seed(seed)
    env = build_env_from_config(c)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, c)
    policy = MAPPOPolicy(model, "cpu")
    data = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu")
    model.load_state_dict(data.model_state)
    out = Path(f"/tmp/minib_coherence_{size}")
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, c, out, device="cpu")

    recorded = []
    policy_ids = {id(p) for p in trainer._policy_params}
    orig_clip = tu.clip_grad_norm_

    def spy(parameters, max_norm, *a, **kw):
        params = list(parameters)
        flat = [p.grad.detach().reshape(-1).clone() for p in params
                if p.grad is not None]
        if flat:
            ids = {id(p) for p in params}
            if ids & policy_ids:
                recorded.append(torch.cat(flat))
        return orig_clip(parameters, max_norm, *a, **kw)

    tu.clip_grad_norm_ = spy
    try:
        buf = trainer.collect_rollout()
        stats = trainer.update(buf)
    finally:
        tu.clip_grad_norm_ = orig_clip

    if len(recorded) < 2:
        return {"size": size, "nb": len(recorded), "adj": float("nan"),
                "drift": float("nan"), "kl": float("nan")}
    adj = [float(torch.dot(recorded[i], recorded[i + 1]) /
                 (recorded[i].norm() * recorded[i + 1].norm()).clamp_min(1e-12))
           for i in range(len(recorded) - 1)]
    first = recorded[0]
    drift = [float(torch.dot(g, first) /
                   (g.norm() * first.norm()).clamp_min(1e-12)) for g in recorded]
    return {"size": size, "nb": len(recorded),
            "adj": sum(adj) / len(adj), "adj_min": min(adj),
            "drift": drift, "kl": float(stats.kl)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="256,512,1024")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1, seed=42,
        checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    cfg["runtime"]["device"] = "cpu"
    cfg["train"]["n_rollout_workers"] = 1
    cfg["train"]["ppo"]["batch_chunk"] = 64

    print("minibatch 大小 → actor 梯度一致性（每档 1 个 update，同种子）")
    print(f"配置链：{' '.join(tgm_args.configs)}（+ 逐档改 minibatch_size）")
    print()
    print(f"  {'minibatch':>10}{'子集数':>8}{'相邻cos':>10}{'最小相邻':>10}{'kl':>10}"
          f"   {'vs 首个（前 8 个）':<40}")
    print("  " + "-" * 78)
    res = []
    for s in sizes:
        r = run_arm(s, cfg)
        res.append(r)
        head = " ".join(f"{v:+.2f}" for v in r["drift"][:8]) if r["drift"] else "—"
        print(f"  {s:>10}{r['nb']:>8}{r['adj']:>+10.4f}"
              f"{r.get('adj_min', float('nan')):>+10.4f}{r['kl']:>10.5f}   {head:<40}")

    print()
    print("=" * 80)
    ok = [r for r in res if r["nb"] >= 2]
    if len(ok) >= 2:
        adjs = [r["adj"] for r in ok]
        rising = all(adjs[i] < adjs[i + 1] for i in range(len(adjs) - 1))
        print(f"  相邻 cos：{['%.4f' % v for v in adjs]}")
        if rising:
            print("  ⟹ **单调上升** ⟹ 预测成立：子集太多太小确实是反号的一个原因，")
            print("     `minibatch_size` 加大有机制支撑 ⟹ 可用 train_safe_mini512.yaml 跑 30 轮。")
        else:
            print("  ⟹ **没有单调上升** ⟹ 预测被证伪：")
            print("     加大 minibatch 不改善一致性 ⟹ **不要**拿 mini512 去跑 30 轮。")
    else:
        print("  可判读的档位不足（每档都要 nb>=2），**不做判读**。")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
