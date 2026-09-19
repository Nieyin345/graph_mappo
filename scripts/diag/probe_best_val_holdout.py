"""best_val 是真增益还是选择偏差？—— 用**不相交种子**做诚实对照。

### 设计（这是本项目 `selection-bias-max-of-k` 那条记忆的直接检验）

训练器的验证集是固定的 15 个请求种子 `[100..114]`，
`checkpoint_best_val.pt` 就是**在这 15 个种子上**、于 u5/10/15/20/25/30
六个检查点里取验证均值最大的那个存下来的。

所以「在 100..114 上 best_val 比 final 好多少」这个数**按构造必然 ≥ 0**，
它把「六选一的最大值」和「最后一轮」相比，**大部分是选择偏差**。

诚实口径：换一组**训练时没见过的**请求种子 `[200..214]`，在同一验证窗口
（330–365 天、240 步、random_day）上重评两个 checkpoint。
  · 增益在不相交种子上**仍在** ⟹ 是真增益
  · 增益塌到 0 或转负      ⟹ 是选择偏差
两者之差就是**选择偏差本身的实测值**。

### 实测背景（2026-09-19）

`w329p_*` 三臂的验证曲线末段各不相同，正好构成三种情形：

| 臂 | argmax | 末轮 | best_val 与 final |
|----|--------|------|-------------------|
| s42 | u30 | u30 | sha256 相同 ⟹ 同一份 |
| s43 | u25 | u30 | 不同，末轮在跌 |
| s44 | u20 | u30 | 不同，末轮跌得更多 |

「平均」口径下 u25→u30 还在涨，但**逐臂看末轮未必在涨** ——
平均掩盖了逐臂的见顶。这条探针就是要把这件事量出来。

用法（服务器上）：
    /opt/qkd/venv/bin/python scripts/diag/probe_best_val_holdout.py \
        --run w329p_s43 --configs rl_algorithm.yaml train_full_rl.yaml \
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

T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
          13: 2.160, 14: 2.145}


def report(tag: str, d: list[float]) -> dict:
    n = len(d)
    m = statistics.mean(d)
    print(f"    {tag}")
    if n < 2:
        print(f"      Δ = {m:+.4f}   n={n}（太少，不作判读）")
        return {"n": n, "mean": m}
    sd = statistics.stdev(d)
    se = sd / n ** 0.5
    print(f"      Δ = {m:+.4f}   SD {sd:.4f}   SE {se:.4f}")
    if sd == 0.0:
        # ★ 不能报"t=inf ⟹ 可测"：SD=0 时 t 无定义，两个 checkpoint
        #   在**每一个**种子上都完全相同 ⟹ 是"没有差异"，不是"差异显著"。
        #   本项目已栽过 `0/0 → +inf` 一次（详见记忆 thresholds-and-transcribed-numbers）。
        print("      ⚠ SD = 0：**每个种子上的差都恰好为 0** ⟹ 两策略逐位相同，"
              "t 无定义，**不报显著性**")
        return {"n": n, "mean": m, "sd": 0.0}
    t = m / se
    df = n - 1
    crit = T_CRIT.get(df, float("nan"))
    print(f"      t = {t:+.3f}   (df={df}, 临界 {crit})   ⟹ "
          f"**{'可测' if abs(t) > crit else '测不出'}**")
    return {"n": n, "mean": m, "sd": sd, "se": se, "t": t, "df": df, "crit": crit}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", default="200-214",
                    help="不相交的请求种子范围，逗号或 - 分隔")
    a = ap.parse_args()

    run_dir = ROOT / "outputs" / a.run
    ck = {"final": run_dir / "checkpoint_final.pt",
          "best_val": run_dir / "checkpoint_best_val.pt"}
    for k, p in ck.items():
        if not p.exists():
            print(f"  ✗ 缺 {k}: {p}")
            return 1

    # 解析 holdout 种子
    hs: list[int] = []
    for part in a.holdout.split(","):
        part = part.strip()
        if "-" in part:
            x, y = part.split("-", 1)
            hs.extend(range(int(x), int(y) + 1))
        elif part:
            hs.append(int(part))
    print(f"  不相交请求种子：{hs}")

    # 两份配置：训练验证集（100..114，被污染）与 holdout（干净）
    def cfg_with_seeds(seeds: list[int]) -> dict:
        args = argparse.Namespace(
            configs=list(a.configs), mode="random_episode", run_name=None,
            num_updates=1, seed=a.seed, checkpoint=None, device="cpu")
        c = tgm.build_config(args)
        c["runtime"]["device"] = "cpu"
        c["train"]["n_rollout_workers"] = 1
        c["train"]["ppo"]["batch_chunk"] = 64
        c.setdefault("validation", {})
        c["validation"]["request_seeds"] = list(seeds)
        c["validation"]["episodes"] = len(seeds)
        return c

    out: dict[str, dict[str, list[float]]] = {}
    for setname, seeds in (("训练验证集 100..114", None), ("不相交 200..214", hs)):
        use = seeds
        if use is None:
            # 用配置里原本的种子（就是 100..114）
            c = cfg_with_seeds([100, 101, 102, 103, 104, 105, 106, 107, 108,
                                109, 110, 111, 112, 113, 114])
        else:
            c = cfg_with_seeds(use)

        torch.manual_seed(a.seed)
        env = build_env_from_config(c)
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, c)
        policy = MAPPOPolicy(model, "cpu")
        outp = Path("/tmp/best_val_holdout")
        outp.mkdir(parents=True, exist_ok=True)
        trainer = MAPPOTrainer(env, policy, c, outp, device="cpu")

        out[setname] = {}
        for tag, path in ck.items():
            data = load_checkpoint(str(path), "cpu")
            model.load_state_dict(data.model_state)
            r = trainer.evaluate_validation()
            ps = [float(x) for x in r["per_seed_success"]]
            out[setname][tag] = ps
            print(f"  [{setname}] {tag:9s} 种子数 {len(ps)}  "
                  f"均值 {statistics.mean(ps):.4f}")

    print()
    print("=" * 78)
    print(f"{a.run}：best_val − final")
    print("=" * 78)
    res = {}
    for setname in out:
        pf = out[setname]["final"]
        pb = out[setname]["best_val"]
        d = [y - x for x, y in zip(pf, pb)]
        print(f"\n  ── {setname} ──")
        res[setname] = report("best_val − final", d)

    print()
    print("=" * 78)
    print("判读")
    print("=" * 78)
    trained = res.get("训练验证集 100..114", {}).get("mean")
    held = res.get("不相交 200..214", {}).get("mean")
    if trained is not None and held is not None:
        print(f"  训练验证集上（被污染的）增益：{trained:+.4f}")
        print(f"  不相交种子上的增益（诚实的）：{held:+.4f}")
        print(f"  ⟹ **选择偏差的实测值 ≈ {trained - held:+.4f}**")
        if held > 0:
            print("  诚实口径仍为正 ⟹ best_val 有**真**增益（至少在单训练种子上）。")
        else:
            print("  诚实口径 ≤ 0 ⟹ 那个增益**基本是选择偏差**，"
                  "部署时拿不到。")

    json.dump(out, open("/tmp/best_val_holdout.json", "w"), indent=1)
    print("\n  已写 /tmp/best_val_holdout.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
