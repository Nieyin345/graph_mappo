"""奖励对动作的敏感度 —— **分项**版。

动机：奖励由多项组成（served / failed / storage / dense generation / expired /
keep_active ...）。总奖励对动作有反应（探针 B 已证），但**是哪几项在动**决定了梯度
方向是否指向"服务得更多"。如果主项 served 对动作不敏感、敏感的是几个小项，
策略就会去优化小项，成功率自然不涨。

做法与探针 B 相同（同 seed reset、同动作同步走到第 t 步、然后分叉），
区别是记录 `info['reward_detail']`，对每个分项算：
  * mean                       该项的量级
  * 局内 std（同状态换动作）     动作能推动多少
  * 局间 std（同动作换状态）     局面影响多少
  * 局内/mean                   该项对动作的相对敏感度

用法：
    python .tmp/probe_reward_components.py [--trials 8] [--branches 5,20,60] [--samples 6]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "rl"))


def numeric_items(d) -> dict[str, float]:
    """把 dataclass / dict 摊平成 {名字: 数值}。

    `info['reward_detail']` 是 `qkd_rl.env.reward.RewardDetail` 这个 dataclass，
    不是 dict —— 直接当 dict 取键会静默拿到空结果。
    """
    import dataclasses

    if dataclasses.is_dataclass(d) and not isinstance(d, type):
        d = dataclasses.asdict(d)
    elif not isinstance(d, dict) and hasattr(d, "__dict__"):
        d = vars(d)

    out: dict[str, float] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif dataclasses.is_dataclass(v) or isinstance(v, dict):
            for k2, v2 in numeric_items(v).items():
                out[f"{k}.{k2}"] = v2
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="/opt/qkd/graph_mappo/outputs/supervised_pg_phased/supervised_pg_phased_latest.pt",
    )
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--branches", default="5,20,60")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.rl.algos.checkpoint import load_checkpoint
    from qkd_rl.rl.algos.policy import MAPPOPolicy
    from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
    from train_graph_mappo import build_config

    ns = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode",
        run_name="probe_components",
        num_updates=None,
        seed=None,
        checkpoint=None,
        device=args.device,
    )
    config = build_config(ns)
    torch.manual_seed(int(config["seed"]["global_seed"]))
    torch.set_float32_matmul_precision("high")

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    data = load_checkpoint(args.checkpoint, device=args.device)
    model.load_state_dict(data.model_state)
    model.eval()
    policy = MAPPOPolicy(model, args.device)

    K = args.samples
    envs = [build_env_from_config(config) for _ in range(K)]
    branches = [int(x) for x in str(args.branches).split(",")] or [0]

    # per-component: 每个状态一组观测（K 个动作）
    per_state: list[dict[str, list[float]]] = []
    sample_detail_keys: list[str] | None = None
    dbg = [False]

    for t_branch in branches:
        for trial in range(args.trials):
            seed = args.seed + trial
            obs_list = [e.reset(seed=seed) for e in envs]
            for _ in range(t_branch):
                with torch.no_grad():
                    s = policy.act(obs_list[0], build_scores=False)
                for k, e in enumerate(envs):
                    obs_list[k], _r, _te, _tr, _i = e.step(
                        s.actions, s.action_scores,
                        edge_scores=s.edge_scores,
                        expected_matched_edges=list(s.matched_edges or []),
                    )
            with torch.no_grad():
                sampled = policy.act_batched(
                    [obs_list[0]] * K, build_scores=False, use_edge_arrays=True
                )

            comp: dict[str, list[float]] = defaultdict(list)
            for k in range(K):
                _o, r, _te, _tr, info = envs[k].step(
                    sampled[k].actions, sampled[k].action_scores,
                    edge_scores=sampled[k].edge_scores,
                    expected_matched_edges=list(sampled[k].matched_edges or []),
                )
                comp["__total__"].append(float(r))
                detail = info.get("reward_detail")
                if detail is not None:
                    if not dbg[0]:
                        # 第一次见到就把原始类型/内容打出来 —— 它不一定是个普通 dict。
                        dbg[0] = True
                        print(f"[debug] reward_detail 类型 = {type(detail)}")
                        print(f"[debug] 内容 = {repr(detail)[:400]}")
                        print(f"[debug] 解析出的数值项 = {sorted(numeric_items(detail))}")
                    items = numeric_items(detail)
                    if sample_detail_keys is None and items:
                        sample_detail_keys = sorted(items)
                    for kk, vv in items.items():
                        comp[kk].append(vv)
            per_state.append(dict(comp))

    if not per_state:
        print("没采到样本")
        return

    if sample_detail_keys:
        print(f"reward_detail 的分项: {sample_detail_keys}")
    print()

    keys = ["__total__"] + [k for k in per_state[0] if k != "__total__"]
    # 只保留在绝大多数状态里都出现的分项
    n_states = len(per_state)
    keys = [k for k in keys if sum(1 for s in per_state if k in s) >= n_states * 0.9]

    print("=" * 100)
    print(f"{'分项':<28}{'均值':>14}{'局内std':>12}{'局间std':>12}{'局内/均值':>12}{'局间/局内':>12}")
    print("=" * 100)
    rows = []
    for k in keys:
        # 每个状态内：K 个动作 → 均值与 std
        state_means, state_stds = [], []
        all_vals = []
        for s in per_state:
            if k not in s:
                continue
            v = torch.tensor(s[k], dtype=torch.float64)
            state_means.append(float(v.mean()))
            state_stds.append(float(v.std()) if len(s[k]) > 1 else 0.0)
            all_vals.extend(s[k])
        if not state_means:
            continue
        sm = torch.tensor(state_means, dtype=torch.float64)
        ss = torch.tensor(state_stds, dtype=torch.float64)
        mean = float(sm.mean())
        w = float(ss.mean())          # 局内：动作引起
        b = float(sm.std()) if len(state_means) > 1 else 0.0   # 局间：局面引起
        rel = w / abs(mean) if mean else float("nan")
        snr = b / w if w > 0 else float("inf")
        rows.append((k, mean, w, b, rel, snr))

    rows.sort(key=lambda r: -abs(r[1]))
    for k, mean, w, b, rel, snr in rows:
        nm = "总奖励" if k == "__total__" else k
        print(f"{nm:<28}{mean:>14.6f}{w:>12.6f}{b:>12.6f}"
              f"{rel:>12.4f}{snr:>12.3f}")

    print()
    print("判读：")
    print("  * 局内/均值 大 = 动作能显著推动该项；小 = 该项基本由局面决定，动作改不动。")
    print("  * 若**量级最大**的项（served）局内/均值很小，而梯度主要由局内大的小项贡献，")
    print("    策略就会去优化小项而不是'服务得更多'—— 这是成功率不涨的直接机制。")
    print("PROBE_COMPONENTS_DONE")


if __name__ == "__main__":
    main()
