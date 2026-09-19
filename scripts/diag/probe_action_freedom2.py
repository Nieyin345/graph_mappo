"""actor 的动作空间还剩多少自由度 —— 两个**直接**口径，不用代理量。

### 为什么重写（v1 的判据是错的，留作教训）

`probe_action_freedom.py` v1 用「命中弧最低分 − stop_score」当 STOP 是否生效的
代理，并用 **P1** 判「一次都没生效」。实测：

    余量 <= 0 的样本: 33/11520 = 0.29%
    P1 = +0.3962

**判据打印了「STOP 一次都没生效」，而事实是 0.29% 的样本擦到了边界。**
0.29% 的事件落在 P1 以下，**P1 这个分位数原理上看不见它** —— 判据的粒度
比被检验的现象粗，于是从数据里"制造"出了一个更强的结论。

更根本的是，那个量**本身也不是** STOP 是否被选中：余量 <= 0 只说明 STOP
**有竞争力**（Gumbel 噪声下低分弧仍可能赢），不说明 STOP 赢了。
拿代理量当结论，正是本项目反复踩的同一类错误。

### v2 的两个口径都是直接观测量

**口径 A —— STOP 到底有没有参与决策？（扰动法，最直接）**
在同一份分数、同一份 Gumbel 上，把 `stop_logit` 分别设成
`{原值, −1e9, +1e9}` 重跑采样：

  · 三者匹配**逐位相同** ⟹ STOP **从未进入过任何一次决策**
  · 只有 −1e9 相同     ⟹ STOP 影响过决策（原值下偶尔被选中）
  · +1e9 立即全停      ⟹ 反向对照，确认这个方法**能**检测到差异（否则是死仪器）

★ `+1e9` 那一档是**仪器自检**：它必须产生不同的（大幅缩水的）匹配。
  若连它都不变，说明扰动没生效，**全部读数作废**。

**口径 B —— 匹配在"被迫"还是"选择"下结束？**
采样循环的两条终止路径：
  (i) 没有可行弧了（`finished |= ~alive[:, :A].any()`）→ **被迫**结束
  (ii) 选中 STOP（`finished |= live & (k == A)`）    → **主动**结束
从返回的匹配反推：按 `kill` 规则（同 src / 同 dst / 同 pair 都被杀）算出
**结束时还剩多少可行弧**。剩 0 ⟹ 被迫；剩 >0 ⟹ 主动放弃。

### 判据（跑之前写死，且**不用分位数**）

  A. 若 {原值, −1e9} 逐位相同且 {+1e9} 不同
       ⟹ STOP 不可达：弃权这条出口形式上有、**实际一次没用过**
  A'. 若 {原值} 与 {−1e9} 不同
       ⟹ STOP **确实会被选中**（罕见），"不可达"说法**不成立**
  B. 可行弧剩 0 的图占比
       ⟹ 若接近 100%：匹配**数量由端口拓扑决定**，策略只定**选哪几条**
       ⟹ 若很低：策略主动放弃了很多可行弧，数量**由它决定**

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/probe_action_freedom2.py --steps 40
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

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_tgm", ROOT / "scripts" / "train" / "train_graph_mappo.py")
tgm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tgm)

from qkd_rl.env.factory import build_env_from_config            # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint          # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer          # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy                  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402


def feasible_left(src, dst, ids, matched_ids) -> int:
    """按采样器的 kill 规则，算匹配结束时还剩多少条**可行**弧。

    `policy.py` 里一条弧被杀掉当且仅当：与已选弧**同 src**、**同 dst**、
    或**同 pair**（对端不同规则）。所以「还剩可行弧」= 三个集合都没碰到。
    """
    used_src, used_dst, used_pair = set(), set(), set()
    pos = {n: i for i, n in enumerate(ids)}
    for a, b in matched_ids:
        used_src.add(a)
        used_dst.add(b)
        used_pair.add((a, b))
        used_pair.add((b, a))
    left = 0
    for j in range(len(src)):
        u, v = ids[src[j]], ids[dst[j]]
        if u in used_src or v in used_dst or (u, v) in used_pair:
            continue
        left += 1
    return left


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
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

    torch.manual_seed(a.seed)
    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    data = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu")
    model.load_state_dict(data.model_state)
    out = Path("/tmp/action_freedom2")
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, cfg, out, device="cpu")

    rec = {"left": [], "n_matched": [], "n_arcs": [], "n_steps": 0,
           "n_graphs_per_step": []}
    # A 口径：三档 stop_score 下的匹配
    variants: dict[str, list] = {"real": [], "neg": [], "pos": []}
    stop_param = model.actor.stop_logit
    real_stop = float(stop_param.detach())

    orig = MAPPOPolicy._sample_matching_arrays

    def snapshot(self, src_list, dst_list, score_list, node_id_list,
                 deterministic):
        """存/恢复所有采样用到的 RNG，保证三档看到**同一份 Gumbel**。"""
        gen = getattr(self, "_sample_gen", None)
        gens = getattr(self, "_sample_gens_batched", None)
        saved = gen.get_state() if gen is not None else None
        saved_multi = [g.get_state() for g in gens] if gens else None

        def restore():
            if gen is not None and saved is not None:
                gen.set_state(saved)
            if gens and saved_multi is not None:
                for g, st in zip(gens, saved_multi):
                    g.set_state(st)

        return restore

    def wrapped(self, src_list, dst_list, score_list, node_id_list,
                deterministic=False):
        restore = snapshot(self, src_list, dst_list, score_list, node_id_list,
                           deterministic)

        # --- 口径 A：三档，同一份 Gumbel ---
        for key, val in (("real", real_stop), ("neg", -1e9), ("pos", 1e9)):
            with torch.no_grad():
                stop_param.fill_(val)
            r = orig(self, src_list, dst_list, score_list, node_id_list,
                     deterministic)
            variants[key].append([tuple(e) for res in r for e in res[1]])
            # ★ 每档之后必须恢复 RNG 状态，否则下一档的 Gumbel 不同
            #   （这样三档的差异才只归因于 stop_score）
            if key != "pos":
                restore()
        # 还原真实 stop_logit，再做真正返回给训练器的那一次
        with torch.no_grad():
            stop_param.fill_(real_stop)
        restore()
        res = orig(self, src_list, dst_list, score_list, node_id_list,
                   deterministic)

        rec["n_steps"] += 1
        rec["n_graphs_per_step"].append(len(res))
        for g, r in enumerate(res):
            matched = [tuple(e) for e in r[1]]
            ids = list(node_id_list[g])
            src = [int(v) for v in src_list[g]]
            dst = [int(v) for v in dst_list[g]]
            rec["n_arcs"].append(len(src))
            rec["n_matched"].append(len(matched))
            rec["left"].append(feasible_left(src, dst, ids, matched))
        return res

    MAPPOPolicy._sample_matching_arrays = wrapped
    try:
        trainer.collect_rollout()
    finally:
        MAPPOPolicy._sample_matching_arrays = orig
        with torch.no_grad():
            stop_param.fill_(real_stop)
    del trainer

    def same(x: list, y: list) -> bool:
        return len(x) == len(y) and all(p == q for p, q in zip(x, y))

    # 仪器自检：+1e9 必须产生不同的匹配
    instrument_ok = not all(
        same(variants["pos"][i], variants["real"][i])
        for i in range(len(variants["real"])))

    print("=" * 84)
    print("actor 动作自由度 · v2（直接观测量，零训练成本：1 个 rollout）")
    print("=" * 84)
    print(f"  种子 {a.seed}   决策步 {rec['n_steps']}   "
          f"图-步 {len(rec['n_matched'])}   stop_logit = {real_stop:+.4f}")
    print()

    n = len(variants["real"])
    n_real_ne_neg = sum(1 for i in range(n)
                        if not same(variants["real"][i], variants["neg"][i]))
    n_real_ne_pos = sum(1 for i in range(n)
                        if not same(variants["real"][i], variants["pos"][i]))
    # ★ 分母：`variants[key]` 是**每步**的扁平表（一步含 G 个图），
    #   所以「每图命中弧数」= 总和 ÷ (步数 × 每步图数)，**不是** ÷ 步数。
    #   首版写成 ÷ rec['n_steps']，把 60.0 印成 0.3 —— 与
    #   「分母用了 ${#SEEDS}（字符串长度）」是同一类错误。
    n_graphs = sum(rec["n_graphs_per_step"]) or 1
    mean_arcs = {k: sum(len(m) for m in v) / n_graphs
                 for k, v in variants.items()}
    print("  ── A. STOP 参不参与决策（扰动 stop_logit，同一份 Gumbel）──")
    print(f"     真实值 vs −1e9  : **{n_real_ne_neg}/{n}** 步不同")
    print(f"     真实值 vs +1e9  : **{n_real_ne_pos}/{n}** 步不同  ← 仪器自检，须 >0")
    print(f"     每图命中弧数: 真实 {mean_arcs['real']:.2f}   "
          f"−1e9 {mean_arcs['neg']:.2f}   +1e9 {mean_arcs['pos']:.2f}")
    print()

    left = rec["left"]
    nz = sum(1 for v in left if v == 0)
    print("  ── B. 匹配是怎么结束的（结束时还剩几条可行弧）──")
    print(f"     剩 0 条（**被迫**结束）: {nz}/{len(left)} = "
          f"{100 * nz / max(1, len(left)):.1f}%")
    if left:
        print(f"     剩余条数 中位 {statistics.median(left):.0f}   "
              f"均值 {statistics.mean(left):.1f}  最大 {max(left)}")
    print(f"     命中弧数  中位 {statistics.median(rec['n_matched']):.0f}   "
          f"候选弧数 中位 {statistics.median(rec['n_arcs']):.0f}")
    print()

    print("=" * 84)
    print("判读（判据在文件头写死；不用分位数）")
    print("=" * 84)
    if not instrument_ok:
        print("  ✗ 仪器自检失败：+1e9 没能改变任何匹配 ⟹ 扰动没生效。")
        print("    **全部读数作废**，先修探针。")
    else:
        print(f"  仪器自检 ✓（+1e9 在 {n_real_ne_pos}/{n} 步上改变了匹配）")
        print()
        if n_real_ne_neg == 0:
            print("  A ⟹ 真实值与 −1e9 **逐位相同** ⟹ STOP **从未进入过任何一次决策**")
            print("      弃权这条出口形式上有、实际一次没用过。")
        elif n_real_ne_neg < 0.05 * n:
            print(f"  A ⟹ STOP **确实会被选中**，但极罕见（{n_real_ne_neg}/{n} = "
                  f"{100 * n_real_ne_neg / n:.1f}% 的步）。")
            print("      ⟹ 「不可达」**不成立**，应记为「近乎不可达」。")
        else:
            print(f"  A ⟹ STOP **真的在用**（{n_real_ne_neg}/{n} 步不同）"
                  f"⟹ 「STOP 是死出口」**不成立**。")
            print("      ★ 但注意差异**落在哪**：把 stop_score 压到 −1e9 时每图命中")
            print(f"        只剩 {mean_arcs['neg']:.1f} 条（真实 {mean_arcs['real']:.1f}），"
                  f"说明 STOP 就是那个")
            print("        「继续匹配 vs 收手」的总闸。真实值下的差异是**换了一批**")
            print("        匹配（谁占哪条边），不是「用不用 STOP」—— 见口径 B。")
        print()
        frac = nz / max(1, len(left))
        if frac > 0.9:
            print(f"  B ⟹ {100 * frac:.1f}% 的图在**没有可行弧时**才结束 ⟹ 匹配**数量**")
            print("      由端口拓扑决定，策略只决定**选哪几条**（排序）。")
        elif frac > 0.5:
            print(f"  B ⟹ {100 * frac:.1f}% 被迫结束，其余是主动放弃 ⟹ 数量**部分**可控。")
        else:
            print(f"  B ⟹ 只有 {100 * frac:.1f}% 被迫结束 ⟹ 策略**经常主动放弃**可行弧")
            print("      ⟹ 数量由策略决定，「自由度薄」的说法**不成立**。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
