"""判别：`storage` 奖励项在当前权重下到底有没有产生梯度？

### 为什么先做这个（而不是先改代码）

架构审查给出两条**互斥**的假设，都指向「结构」但修法完全不同：

- **H_predictive**：策略**不会为未来攒密钥**（奖励无前瞻维）⟹ 该改的是**奖励/动作空间**
- **H_pragmatic**：策略**可以**通过「端口接满」来等效攒密钥（`add_keys` 接满就够，
  `ttl=1e6` 永不失效，服务走任意通路）⟹ 该改的是**决策头/归因**

判别只需要一个信号：**策略饱和后有没有在持续地做「生成」这件事**。

- 若 `generated` 稳定 > 0 ⟹ **H_pragmatic**：模型在学「多生成」
  ⟹ 而奖励对生成量**无感**（`raw_generation_weight: 0`）⟹ **行为在漂移但没人看守**
- 若 `generated → 0` ⟹ 策略主动停止生成 ⟹ 需再查（可能是端口/可行性约束）

### ★ 别用「分位数」判「从未发生」

本项目在分位数上栽过：一个只出现 0.1% 的事件，在按轮聚合的分位数里会被读成 0。
判据一律用**事件计数**：出现过几轮、占比多少、什么时候停止。

### 数据来源

`outputs/<run>/rollout_debug.jsonl` —— 每轮一行，**训练时免费写下的**，不需要重跑。
它含 `mean_reward_generated` 等分项。**该文件是原始记录，不是生成物。**

用法：
    python analyze_generated.py                     # 全扫，给汇总
    python analyze_generated.py --run ent01_rerun_s42   # 单条详查
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

ROOT = "/opt/qkd/graph_mappo"

# 候选分项名（不同版本可能只写了其中一部分，缺的不报错，只标「无此字段」）
KEYS = (
    "mean_reward_generated",
    "mean_reward_storage",
    "mean_reward_served",
    "mean_reward_total",
    "mean_success_rate",
)


def load(run: str):
    p = os.path.join(ROOT, "outputs", run, "rollout_debug.jsonl")
    if not os.path.exists(p):
        return None
    rows = []
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except ValueError:
            continue
    return rows


def series(rows, key):
    """取某字段的逐轮序列，并记录「有该字段」的轮数（区分 0 与 缺失）。"""
    vals, present = [], 0
    for o in rows:
        if key in o and o[key] is not None:
            present += 1
            vals.append(float(o[key]))
    return vals, present


def report_run(run: str, verbose: bool) -> dict:
    rows = load(run)
    if not rows:
        return {"run": run, "n": 0}
    out = {"run": run, "n": len(rows)}

    for key in KEYS:
        vals, present = series(rows, key)
        if present == 0:
            out[key] = None
            continue
        nz = [v for v in vals if v != 0.0]
        out[key] = {
            "present": present,
            "nonzero_rounds": len(nz),
            "first": vals[0],
            "last": vals[-1],
            "mean": st.mean(vals),
            "max": max(vals),
        }

    # ★ 核心判据：generated 的**事件计数**，不是分位数
    g, gp = series(rows, "mean_reward_generated")
    if gp:
        nz = [v for v in g if v != 0.0]
        out["generated_verdict"] = {
            "nonzero_rounds": len(nz),
            "total_rounds": len(g),
            "frac": len(nz) / len(g),
            # 「最后一次非零在第几轮」——若早期非零、后期归零，是**主动停止**的信号
            "last_nonzero_at": (max(i for i, v in enumerate(g) if v != 0.0)
                                if nz else None),
        }
    if verbose:
        print("  %-24s 轮数=%d" % (run, len(rows)))
        for key in KEYS:
            s = out.get(key)
            if s is None:
                print("    %-26s （无此字段）" % key)
            else:
                print("    %-26s 首=%.3e 末=%.3e 均=%.3e 峰=%.3e  非零轮=%d/%d"
                      % (key, s["first"], s["last"], s["mean"], s["max"],
                         s["nonzero_rounds"], s["present"]))
        gv = out.get("generated_verdict")
        if gv:
            print("    ⟹ generated 非零 %d/%d 轮（%.1f%%），最后一次非零在第 %s 轮"
                  % (gv["nonzero_rounds"], gv["total_rounds"],
                     100.0 * gv["frac"], gv["last_nonzero_at"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="单条 run 详查")
    ap.add_argument("--limit", type=int, default=0, help="只扫前 N 条（0=全扫）")
    a = ap.parse_args()

    if a.run:
        report_run(a.run, verbose=True)
        return 0

    odir = os.path.join(ROOT, "outputs")
    runs = sorted(d for d in os.listdir(odir)
                  if os.path.exists(os.path.join(odir, d, "rollout_debug.jsonl")))
    if a.limit:
        runs = runs[:a.limit]
    print("扫到 %d 条有 rollout_debug.jsonl 的 run" % len(runs))

    rep = [report_run(r, verbose=False) for r in runs]
    rep = [r for r in rep if r.get("n")]

    have_g = [r for r in rep if r.get("mean_reward_generated")]
    print("\n=== 有 generated 字段的：%d 条 ===" % len(have_g))
    if have_g:
        # ★ 事件计数式汇总
        always_nz = [r for r in have_g
                     if r["mean_reward_generated"]["nonzero_rounds"]
                     == r["mean_reward_generated"]["present"]]
        some_nz = [r for r in have_g
                   if 0 < r["mean_reward_generated"]["nonzero_rounds"]
                   < r["mean_reward_generated"]["present"]]
        never_nz = [r for r in have_g
                    if r["mean_reward_generated"]["nonzero_rounds"] == 0]
        print("  全程非零（一直在生成）: %d 条" % len(always_nz))
        print("  部分轮非零            : %d 条" % len(some_nz))
        print("  **从不非零**          : %d 条" % len(never_nz))
        if always_nz:
            ms = [r["mean_reward_generated"]["mean"] for r in always_nz]
            print("  全程非零那批的均值: %.4e ~ %.4e（中位 %.4e）"
                  % (min(ms), max(ms), st.median(ms)))
        if never_nz:
            print("  从不非零的例子: %s" % ", ".join(r["run"] for r in never_nz[:8]))
        if some_nz:
            print("  部分非零的例子（看「最后一次非零在第几轮」）:")
            for r in some_nz[:8]:
                print("    %-26s 非零 %d/%d，末次在第 %s 轮"
                      % (r["run"],
                         r["mean_reward_generated"]["nonzero_rounds"],
                         r["mean_reward_generated"]["present"],
                         r["generated_verdict"]["last_nonzero_at"]))

    # storage 项对照
    have_s = [r for r in rep if r.get("mean_reward_storage")]
    print("\n=== 有 storage 字段的：%d 条 ===" % len(have_s))
    if have_s:
        nz = [r for r in have_s if r["mean_reward_storage"]["nonzero_rounds"] > 0]
        print("  至少一轮非零: %d 条" % len(nz))
        if nz:
            ms = [r["mean_reward_storage"]["mean"] for r in nz]
            print("  均值范围: %.3e ~ %.3e" % (min(ms), max(ms)))

    # 若两者都有，报比值 —— 这是「storage 有没有量级」的直接读数
    both = [r for r in rep if r.get("mean_reward_storage") and r.get("mean_reward_served")]
    if both:
        print("\n=== storage / served 的量级比 ===")
        ratios = []
        for r in both:
            sv = r["mean_reward_served"]["mean"]
            if sv:
                ratios.append(r["mean_reward_storage"]["mean"] / sv)
        if ratios:
            print("  中位 %.3e  范围 %.3e ~ %.3e  （n=%d）"
                  % (st.median(ratios), min(ratios), max(ratios), len(ratios)))
            print("  ⟹ 若比值 ≪ float32 梯度噪声，则 storage 项**从未产生有效梯度**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
