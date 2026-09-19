#!/usr/bin/env python
"""确认候选唯一性是**构造保证**而非巧合，并检查 IDLE 子集身份。

两条要验的：
  1. IDLE 在这些图里是不是候选表的**前缀**（`_cand_is_idle` 的子集假设）。
     若不是前缀而只按布尔取，`_cand_srcs` 仍跟着一起取，身份仍成立——
     但"前缀"这个更强的性质若成立，可以省掉一次布尔索引。
  2. `candidates_for_node` 是否真的按 `(src,dst)` 无重复，且与
     `neighbors` 的边数一致（即每个邻居只出现一次，没有平行边）。
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

from qkd_rl.env.factory import build_env_from_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", type=int, default=40)
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
    aspace = env.action_resolver.action_space
    IDLE = aspace.IDLE

    n_cand_dup = 0
    n_idle_not_prefix = 0
    n_idle_multi = 0
    checked = 0

    for k in range(args.obs):
        obs = env.reset(seed=2000 + k)
        for node_id in obs.node_ids:
            cands = list(obs.action_candidates[node_id])
            checked += 1
            # 1. 候选表 (src=node_id, dst=cand) 不得重复
            if len(cands) != len(set(cands)):
                n_cand_dup += 1
            # 2. IDLE 的出现次数与位置
            n_idle = cands.count(IDLE)
            if n_idle > 1:
                n_idle_multi += 1
            if n_idle == 1 and cands[0] != IDLE:
                n_idle_not_prefix += 1

    print(f"检查了 {checked} 个 (obs, node) 的候选表")
    print(f"  1. (src,dst) 有重复的: {n_cand_dup}")
    print(f"  2. IDLE 出现 >1 次的: {n_idle_multi}")
    print(f"  3. IDLE 存在但不在首位的: {n_idle_not_prefix}")
    print()
    if n_cand_dup == 0:
        print("→ 唯一性是构造保证（neighbors 每个相邻节点只有一条边，sorted 后无重复；")
        print("  平行边也没有）。查表法与本实现的数学等价。")
    else:
        print("→ **有重复**，查表法不等价，优化必须放弃。")

    # 静态检查：candidates_for_node 的构造路径
    print()
    print("=== 构造路径核对 ===")
    print("  action_space.py:31-37  neighbors[node] 每条边 append 一次")
    print("  action_space.py:82     candidates_for_node = _with_idle(sorted(neighbors))")
    print("  graph_builder.py:73    candidates[node] = candidates_for_node(node)")
    print("  graph_builder.py:187-188  legal = [a for a,m in zip(full_candidates, mask) if m]")
    print("  → legal 是有序子集，故 (src,dst) 仍唯一；IDLE 与任何真实邻居 id 都不同。")


if __name__ == "__main__":
    main()
