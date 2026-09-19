#!/usr/bin/env python
"""探针：`_matching_log_prob_entropy_arrays` 的候选唯一性 + 去掉 flatnonzero 的等价性/提速。

**为什么要先问"唯一性"**：函数里每个已匹配弧都要
    cand = np.flatnonzero((src_code == pos) & (dst_code == dst_pos)); arc_pos = int(cand[0])
即"取第一个 src/dst 都相符的候选"。若候选表里 (src,dst) **不重复**，那
`cand` 恒为单元素、`cand[0]` 就是该配对的固定下标，与 `alive` 无关 ——
于是整段可以用一张预计算好的查找表替掉，省下每次 2 次比较 + 1 次
flatnonzero（实测每步约 55 次调用，1440 步共 79,451 次）。

**判据（先测再改）**：
  1. 唯一性：在真实 obs 上统计 (src,dst) 重复。若真有重复，快速实现
     与现实现的数学就不等价，必须放弃。
  2. 逐位等价：对大量 (obs, matching) 组合比 `torch.equal`。这是硬门槛——
     这个函数直接进 PPO 的 ratio，改错了会静默改变训练结果。
  3. 提速：同负载交替 A/B。**注意本机在跑 4 个训练，计时噪声大，只作参考**，
     真正的判据是 (1)(2)。

用法（服务器上）：
  OMP_NUM_THREADS=4 /opt/qkd/venv/bin/python /tmp/probe_matching_fast.py --obs 60
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402


def fast_arrays(policy, src_arr, dst_arr, scores, node_pos, matched_edges,
                src_code, dst_code, pair_code):
    """本函数的候选下标改为查表版（其余逐行照抄现实现）。"""
    n_arcs = int(len(src_arr))
    if n_arcs == 0:
        return (torch.zeros((), dtype=torch.float32, device=policy.device),
                torch.zeros((), dtype=torch.float32, device=policy.device))
    temperature = float(policy.model.actor.temperature)
    if temperature != 1.0:
        scores = scores / temperature
        stop_score = policy.model.actor.stop_logit / temperature
    else:
        stop_score = policy.model.actor.stop_logit

    alive = np.ones(n_arcs, dtype=bool)
    rows = np.empty((len(matched_edges) + 1, n_arcs), dtype=np.float32)
    choices: list[int] = []
    n_rows = 0
    for arc in matched_edges:
        pos = node_pos.get(arc[0])
        if pos is None:
            raise ValueError(f"Stored matching arc {arc!r} is not in the observation.")
        dst_pos = node_pos.get(arc[1])
        if dst_pos is None:
            raise ValueError(f"Stored matching arc {arc!r} dst not in the observation.")
        # 查表代替 np.flatnonzero(...)[0]：唯一性成立时两者恒等。
        arc_pos = policy._probe_lookup.get((pos, dst_pos), -1)
        if arc_pos < 0:
            raise ValueError(f"Stored matching arc {arc!r} is not available for evaluation.")
        if not alive[arc_pos]:
            raise ValueError(f"Stored matching arc {arc!r} is not available for evaluation.")
        rows[n_rows] = alive
        choices.append(arc_pos)
        n_rows += 1
        alive &= ~((src_code == src_code[arc_pos])
                   | (dst_code == dst_code[arc_pos])
                   | (pair_code == pair_code[arc_pos]))
    if alive.any():
        rows[n_rows] = alive
        choices.append(n_arcs)
        n_rows += 1
    rows = rows[:n_rows]
    if n_rows == 0:
        return (torch.zeros((), dtype=torch.float32, device=policy.device),
                torch.zeros((), dtype=torch.float32, device=policy.device))
    A = torch.from_numpy(rows).to(device=policy.device)
    logits = (A * scores.unsqueeze(0)).masked_fill(A == 0.0, float("-inf"))
    logits = torch.cat([logits, stop_score.reshape(1).expand(logits.size(0), 1)], dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    choices_t = torch.tensor(choices, dtype=torch.long, device=policy.device).unsqueeze(-1)
    mean_lp = logp.gather(-1, choices_t).mean()
    safe = torch.where(torch.isfinite(logits), logp, torch.zeros_like(logp))
    mean_entropy = -(safe.exp() * safe).sum(dim=-1).mean()
    return mean_lp, mean_entropy


def maximal_matching(src_arr, dst_arr, node_ids):
    used_tx, used_rx, used_pair, out = set(), set(), set(), []
    for s, d in zip(src_arr.tolist(), dst_arr.tolist()):
        pair = (s, d) if s <= d else (d, s)
        if s in used_tx or d in used_rx or pair in used_pair:
            continue
        used_tx.add(s); used_rx.add(d); used_pair.add(pair)
        out.append((node_ids[s], node_ids[d]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", type=int, default=60)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    _spec = importlib.util.spec_from_file_location(
        "_tgm", ROOT / "scripts" / "train" / "train_graph_mappo.py")
    tgm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(tgm)
    cfg = tgm.build_config(argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"], mode="random_episode",
        run_name=None, num_updates=1, seed=42, checkpoint=None, device="cpu"))
    cfg["runtime"]["device"] = "cpu"
    cfg["env"]["episode_steps"] = 240
    cfg["train"]["episode_steps_fixed"] = True
    cfg["train"]["n_rollout_workers"] = 1

    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")

    print(f"OMP={os.environ.get('OMP_NUM_THREADS')} torch={torch.get_num_threads()}")
    print(f"收集 {args.obs} 个真实 obs...", flush=True)

    cases = []          # (src, dst, scores, node_pos, matched, src_code, dst_code, pair_code)
    dup_total = 0
    dup_obs = 0
    for k in range(args.obs):
        obs = env.reset(seed=1000 + k)
        out = policy.model.batched_forward([obs], policy.device, want_edge_maps=True)
        edge_scores = out.edge_score_maps[0]
        src_arr, dst_arr, scores = out.edge_arrays[0]
        n = int(len(src_arr))
        if n == 0:
            continue
        node_ids = list(obs.node_ids)
        node_pos = {nid: i for i, nid in enumerate(node_ids)}
        src_code = np.asarray(src_arr, dtype=np.int64)
        dst_code = np.asarray(dst_arr, dtype=np.int64)
        n_nodes = max(1, len(node_pos))
        pair_code = np.minimum(src_code, dst_code) * n_nodes + np.maximum(src_code, dst_code)

        # 唯一性统计
        keys = src_code.astype(np.int64) * (n_nodes + 1) + dst_code
        n_uniq = len(np.unique(keys))
        if n_uniq != n:
            dup_total += n - n_uniq
            dup_obs += 1

        # 配对：真实采样出来的 + 极长可行匹配（压力更大）
        sampled = list(policy.act_batched([obs], deterministic=True,
                                          build_scores=True)[0].matched_edges or [])
        maxim = maximal_matching(src_arr, dst_arr, node_ids)
        for m in (sampled, maxim):
            if not m:
                continue
            cases.append((src_arr, dst_arr, scores, node_pos, m,
                          src_code, dst_code, pair_code, edge_scores))

    print(f"可用 (obs, matching) 组合 {len(cases)} 个")
    print(f"=== 1. 候选唯一性 ===")
    print(f"  有重复 (src,dst) 的 obs: {dup_obs} / {args.obs}，重复条目合计 {dup_total}")
    if dup_total:
        print("  ** 存在重复 → 查表法与现实现不等价，本优化必须放弃 **")
    else:
        print("  无重复 → cand 恒为单元素，查表法与现实现数学等价（待下面逐位验证）")

    print(f"\n=== 2. 逐位等价（torch.equal，硬门槛）===")
    lp_bad = ent_bad = 0
    maxdiff = 0.0
    for (src_arr, dst_arr, scores, node_pos, m,
         src_code, dst_code, pair_code, edge_scores) in cases:
        policy._probe_lookup = {}
        for i in range(len(src_code)):
            policy._probe_lookup.setdefault((int(src_code[i]), int(dst_code[i])), i)
        ref_lp, ref_ent = policy._matching_log_prob_entropy_arrays(
            src_arr, dst_arr, scores, node_pos, m)
        new_lp, new_ent = fast_arrays(policy, src_arr, dst_arr, scores, node_pos, m,
                                      src_code, dst_code, pair_code)
        if not torch.equal(ref_lp, new_lp):
            lp_bad += 1
            maxdiff = max(maxdiff, abs(ref_lp.item() - new_lp.item()))
        if not torch.equal(ref_ent, new_ent):
            ent_bad += 1
            maxdiff = max(maxdiff, abs(ref_ent.item() - new_ent.item()))
    print(f"  log_prob 不等: {lp_bad} / {len(cases)}")
    print(f"  entropy  不等: {ent_bad} / {len(cases)}")
    print(f"  最大绝对差: {maxdiff:.3e}")
    if lp_bad or ent_bad:
        print("  ** 不等价，不能改 **")
    else:
        print("  → 逐位相同，可以改（仍要补一个唯一性断言 + 回归测试）")

    print(f"\n=== 3. 提速（同负载交替，取 min；**当前机器在跑 4 个训练，噪声大**）===")
    # 对每个 case 预建 lookup，让两边只差"取下标的方式"
    def build_lookup(src_code, dst_code):
        lut = {}
        for i in range(len(src_code)):
            lut.setdefault((int(src_code[i]), int(dst_code[i])), i)
        return lut

    ref_t, new_t = [], []
    for rep in range(args.reps):
        t0 = time.perf_counter()
        for (src_arr, dst_arr, scores, node_pos, m, src_code, dst_code, pair_code, _e) in cases:
            policy._matching_log_prob_entropy_arrays(src_arr, dst_arr, scores, node_pos, m)
        ref_t.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for (src_arr, dst_arr, scores, node_pos, m, src_code, dst_code, pair_code, _e) in cases:
            policy._probe_lookup = build_lookup(src_code, dst_code)
            fast_arrays(policy, src_arr, dst_arr, scores, node_pos, m,
                        src_code, dst_code, pair_code)
        new_t.append(time.perf_counter() - t0)
        print(f"  rep{rep}  现实现 {ref_t[-1]:6.2f}s   查表版 {new_t[-1]:6.2f}s"
              f"   {ref_t[-1]/new_t[-1]:.3f}x", flush=True)

    print(f"\n  min:  现实现 {min(ref_t):.2f}s  查表版 {min(new_t):.2f}s"
          f"  → {min(ref_t)/min(new_t):.3f}x")
    print(f"  median: 现实现 {statistics.median(ref_t):.2f}s"
          f"  查表版 {statistics.median(new_t):.2f}s")
    print(f"\n  注：查表版这里把 build_lookup 也算进耗时了（真实实现里它是一次性的），"
          f"所以上面对比是**保守**的。")


if __name__ == "__main__":
    main()
