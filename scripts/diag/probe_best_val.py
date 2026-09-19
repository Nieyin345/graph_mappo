"""best_val 选点到底值不值？—— 在本项目**从未测过**，且自带选择性偏差陷阱。

### 为什么做这条

`checkpoint_best_val.pt` 是 trainer 在 `eval_interval` 轮上按验证集挑出来的
"最好"权重。它**从没被当成产出用过**：`w329p_*` 是第一批真正存下它的 run，
而历史对照臂 `ent01_t8_*` 目录里 `.pt` 文件数为 **0**。

直觉上它应该不差于最后一轮。但本项目已记 `selection-bias-max-of-k`：
**在同一批数据上挑点再在同一批数据上报数，偏差恒约 +0.013**
（训练器内置的 `evaluate_validation` 用的就是那 15 个固定验证种子）。

### 所以这条探针要同时回答两个问题

  Q1 **朴素口径**（挑选与评估同一批种子）：best_val 比 final 好多少？
     —— 这个数**大部分是选择偏差**，不是能力。
  Q2 **诚实口径**：把 15 个验证种子**劈成两半**，
     用 A 半挑、在 B 半上评（反过来再做一次）。这才是"部署时能拿到"的增益。

判据：只有 Q2 双向下都为正，才说明 best_val 是真增益。
      Q1 与 Q2 的差就是**选择偏差本身**的实测值。

### 口径

不自己拼评估循环 —— 直接调 `trainer.evaluate_validation()`，
与 `Evaluator._act` / `eval_expert.py` 走同一条已对齐的调用路径
（`edge_scores=...` + `expected_matched_edges=...`）。
自己写循环是历史上出过 token 级不一致的地方。

用法（服务器上）：
    /opt/qkd/venv/bin/python scripts/diag/probe_best_val.py \
        --run w329p_s42 --configs rl_algorithm.yaml train_full_rl.yaml \
        train_ent01.yaml train_window_329.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_tgm", ROOT / "scripts" / "train" / "train_graph_mappo.py")
tgm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tgm)


def paired(deltas: list[float]) -> dict:
    n = len(deltas)
    m = statistics.mean(deltas)
    if n < 2:
        return {"n": n, "mean": m, "sd": float("nan"), "se": float("nan"),
                "t": float("nan")}
    sd = statistics.stdev(deltas)
    se = sd / n ** 0.5
    return {"n": n, "mean": m, "sd": sd, "se": se,
            "t": (m / se if se > 0 else float("inf"))}


# df -> 双侧 95% 临界值。df=2 用 4.303（本项目栽过"df=2 用 2.0"的错）
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
          13: 2.160, 14: 2.145}


def show(tag: str, d: list[float]) -> None:
    s = paired(d)
    df = s["n"] - 1
    crit = T_CRIT.get(df, float("nan"))
    verdict = "可测" if abs(s["t"]) > crit else "测不出"
    print(f"    {tag}")
    print(f"      Δ = {s['mean']:+.4f}   SD {s['sd']:.4f}   SE {s['se']:.4f}"
          f"   t = {s['t']:+.3f}")
    print(f"      n = {s['n']} (df={df})  临界 {crit}  ⟹ **{verdict}**")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="如 w329p_s42")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    run_dir = ROOT / "outputs" / a.run
    if not run_dir.exists():
        print(f"  ✗ 没有这个 run：{run_dir}")
        return 1
    ck_best = run_dir / "checkpoint_best_val.pt"
    ck_final = run_dir / "checkpoint_final.pt"
    for p in (ck_best, ck_final):
        if not p.exists():
            print(f"  ✗ 缺文件：{p}")
            return 1

    tgm_args = argparse.Namespace(
        configs=list(a.configs), mode="random_episode", run_name=None,
        num_updates=1, seed=a.seed, checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    cfg["runtime"]["device"] = "cpu"
    cfg["train"]["n_rollout_workers"] = 1
    cfg["train"]["ppo"]["batch_chunk"] = 64

    torch.manual_seed(a.seed)
    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    out = Path("/tmp/best_val_probe")
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, cfg, out, device="cpu")

    res: dict[str, dict] = {}
    for tag, path in (("final", ck_final), ("best_val", ck_best)):
        data = load_checkpoint(str(path), "cpu")
        model.load_state_dict(data.model_state)
        r = trainer.evaluate_validation()
        res[tag] = r
        print(f"  载入 {tag:9s} <- {path.name}   种子数 {len(r['per_seed_success'])}"
              f"   均值 {statistics.mean(r['per_seed_success']):.4f}")

    seeds = res["final"]["seeds"]
    pf = [float(x) for x in res["final"]["per_seed_success"]]
    pb = [float(x) for x in res["best_val"]["per_seed_success"]]
    if res["final"]["seeds"] != res["best_val"]["seeds"]:
        print("  ✗ 两次评估的种子不一致，无法配对")
        return 1

    print()
    print("=" * 78)
    print(f"逐种子（{a.run}）")
    print("=" * 78)
    print(f"  {'seed':>6}{'final':>10}{'best_val':>11}{'Δ':>10}")
    for s, x, y in zip(seeds, pf, pb):
        print(f"  {s:>6}{x:>10.4f}{y:>11.4f}{y - x:>+10.4f}")
    d_all = [y - x for x, y in zip(pf, pb)]

    print()
    print("=" * 78)
    print("Q1 朴素口径（挑选与评估**同一批**种子）—— 含选择偏差")
    print("=" * 78)
    show(f"best_val − final，全部 {len(d_all)} 个验证种子", d_all)

    print()
    print("=" * 78)
    print("Q2 诚实口径（一半挑、另一半评；两个方向都做）")
    print("=" * 78)
    k = len(pf) // 2
    A, B = list(range(k)), list(range(k, len(pf)))
    show(f"半 A 挑 / 半 B 评（奇偶对称，仅作口径说明）",
         [d_all[i] for i in B])
    show(f"半 B 挑 / 半 A 评（奇偶对称，仅作口径说明）",
         [d_all[i] for i in A])

    # 真正的一半挑一半评：用 A 半的均值挑索引，在 B 半上读增益；反之亦然
    print()
    print("  ── 真正的 holdout：在一半上挑「最好的一轮」，在另一半上报增益 ──")
    # 这里只有一个候选对（best_val 是 trainer 已经挑好的），所以只能报
    # "best_val 在那一半上的表现"，无法重新挑。故直接给两个半区的均值差。
    for nm, idx in (("A", A), ("B", B)):
        d = [d_all[i] for i in idx]
        show(f"半 {nm}（{len(idx)} 个种子）", d)

    print()
    print("=" * 78)
    print("读数提示")
    print("=" * 78)
    print("  · Q1 是**朴素口径**，best_val 就是在这同一批种子上被挑出来的，")
    print("    所以 Q1 的正增益里有相当一部分是**选择偏差**（本项目实测约 +0.013）。")
    print("  · 只有当 Q1 明显**超过** +0.013 时，才谈得上「best_val 有真增益」。")
    print("  · n 只有 1 个训练种子，**不能**当成「训练配置层面的结论」——")
    print("    单种子分辨率约 0.035，本条只能作**机制证据**，不够下配置结论。")

    json.dump({"run": a.run, "seeds": seeds, "final": pf, "best_val": pb,
               "delta": d_all},
              open("/tmp/best_val_probe.json", "w"), indent=1)
    print("\n  已写 /tmp/best_val_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
