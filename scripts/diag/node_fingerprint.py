#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""u1 硬件指纹：判断每个 run 跑在哪台机器上，并揪出跨节点的对照。

### 为什么需要这个
2026-09-20 一天内连续推翻三条结论（"RL 胜专家 0.86 点"、九臂有获胜臂、
`entropy_coef` 超过专家 p=0.0074），**机理完全相同：换了机器却继续当同一个
实验在用**。跨节点的臂看起来完全正常（文件在、配置对、曲线平滑、指标好看）。

### 判据
**第一次梯度更新之前的那个 rollout（u1）** 只依赖 权重 + 种子 + 代码 + 硬件，
与该臂自己的超参**无关**，也**不随线程数变**（32 线程探针验证逐位相同）。
所以 u1 是天然的、与实验变量正交的硬件指纹。它就在 `metrics.jsonl` 第一行
（`mean_success_rate`），读它成本为零。

**用法**：
    python scripts/diag/node_fingerprint.py                    # 扫 outputs/
    python scripts/diag/node_fingerprint.py --outputs DIR
    python scripts/diag/node_fingerprint.py --ref ref.json     # 对照参考表

### 怎么读输出
- **每个 (种子, u1) 是一个"硬件簇"**。同一种子下出现多个簇 ⟹ 有跨节点搬运。
- 最大簇 = 大多数臂所用的机器（通常是本节点）。
- 落单的臂会列出**与簇内同种子臂的配置差异**：若差异只有 `run_name`，
  那它 u1 不同**无法用配置解释** ⟹ 硬件不同（这是决定性判据）。
- 退出码 1 表示发现了跨节点混用。

相关记忆：`arms-must-not-be-carried-across-nodes`、`entropy-coef-0-01-lead`、
`cloudlab-node-address-is-ephemeral`。规范见 `docs/测试规范.md` §4⑧。
"""
import argparse
import glob
import io
import json
import os
import re
import sys
from collections import defaultdict

# 已记账的参考指纹（u1 的 median across seeds 不成立——每个种子有自己的值），
# 这里只存"节点标签 -> 该节点见过的 u1 集合"。用于给簇自动命名。
KNOWN = {
    "旧节点 amd238 (Zen 2)": [0.857759, 0.859823, 0.856640, 0.859354, 0.859164],
    "本节点 clnode316 (Zen 3)": [0.8725097087102971, 0.874097, 0.871008],
}


def u1_of(path, tol=1e-9):
    """读 metrics.jsonl 第一行的 mean_success_rate（第一次梯度步之前的 rollout）。"""
    try:
        with io.open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    o = json.loads(ln)
                except ValueError:
                    continue
                v = o.get("mean_success_rate")
                if isinstance(v, (int, float)):
                    return float(v)
    except OSError:
        pass
    return None


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        kk = pre + "." + k if pre else k
        if isinstance(v, dict):
            out.update(flat(v, kk))
        else:
            out[kk] = v
    return out


def cfg_of(run_dir):
    p = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.exists(p):
        return None
    try:
        import yaml
        return flat(yaml.safe_load(io.open(p, encoding="utf-8")))
    except Exception:
        return None


def label_of(u1):
    for name, vals in KNOWN.items():
        for v in vals:
            if abs(u1 - v) < 1e-6:
                return name
    return "未知节点"


def main():
    ap = argparse.ArgumentParser(description="u1 硬件指纹 / 跨节点对照检测")
    ap.add_argument("--outputs", default=None,
                    help="outputs 目录（默认依次试 outputs/ 与 /opt/qkd/graph_mappo/outputs）")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="判定 u1 相同的容差（默认 1e-6；逐位相同用 1e-9）")
    ap.add_argument("--quiet", action="store_true", help="只打结论行")
    A = ap.parse_args()

    roots = [A.outputs] if A.outputs else ["outputs", "/opt/qkd/graph_mappo/outputs"]
    root = next((r for r in roots if r and os.path.isdir(r)), None)
    if root is None:
        print("!! 找不到 outputs 目录（试过 %s）" % roots)
        return 2

    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*_s[0-9]*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        m = re.search(r"_s(\d+)$", name)
        if not m:
            continue
        u1 = u1_of(os.path.join(d, "metrics.jsonl"))
        if u1 is None:
            continue
        runs.append((name, int(m.group(1)), u1, d))

    if not runs:
        print("!! %s 下没有可读的 run" % root)
        return 2

    # 按种子聚簇
    by_seed = defaultdict(list)
    for name, seed, u1, d in runs:
        by_seed[seed].append((name, u1, d))

    mixed = []
    if not A.quiet:
        print("=" * 78)
        print("u1 硬件指纹   根目录 = %s   （%d 个 run）" % (root, len(runs)))
        print("=" * 78)

    for seed in sorted(by_seed):
        items = by_seed[seed]
        clusters = []
        for name, u1, d in items:
            for c in clusters:
                if abs(c["u1"] - u1) < A.tol:
                    c["runs"].append((name, d))
                    break
            else:
                clusters.append({"u1": u1, "runs": [(name, d)]})
        clusters.sort(key=lambda c: -len(c["runs"]))
        if len(clusters) <= 1:
            continue
        # 先分类：每个落单簇的 u1 差异**能否被配置解释**
        #
        # ⚠ 这里最容易出假警：若拿一个"改了旋钮的臂"当参照，它的 actor_lr 之类
        #   必然不同，于是每个落单簇都被误报成硬件差异。**假警比漏报更坏**，
        #   因为它留下"检查过"的假信心（记忆 whitelist-from-docs-not-from-keys）。
        #
        # 正确做法：参照取**差异最少的多簇臂**——即与被检臂最像的那条。
        #   若最小差异只剩 run_name/output 这类**不影响 rollout 的键**，才能判硬件。
        COSMETIC = ("project.run_name", "project.output_dir", "project.name")
        judged = []
        for c in clusters[1:]:
            nm, dd = c["runs"][0]
            cc = cfg_of(dd)
            best = None
            for rn, rd in clusters[0]["runs"]:
                rc = cfg_of(rd)
                if not rc or not cc:
                    continue
                diff = sorted(k for k in set(rc) | set(cc)
                              if rc.get(k, "<缺>") != cc.get(k, "<缺>"))
                if best is None or len(diff) < len(best[1]):
                    best = (rn, diff)
            if best is None:
                judged.append((c, nm, None, None, False))
                continue
            rn, diff = best
            subst = [k for k in diff if k not in COSMETIC]
            judged.append((c, nm, rn, subst, not subst))

        unexplained = [j for j in judged if j[4]]
        if unexplained:
            mixed.append((seed, clusters))

        if A.quiet:
            continue
        print()
        nlab = "**有跨节点混用**" if unexplained else "（差异均可由配置解释，**非硬件**）"
        print("### 种子 %d：%d 个簇 ⟹ %s" % (seed, len(clusters), nlab))
        for i, c in enumerate(clusters):
            tag = "（多数）" if i == 0 else "★ 落单"
            print("   簇%d %s  u1=%.12f  节点=%s  n=%d"
                  % (i + 1, tag, c["u1"], label_of(c["u1"]), len(c["runs"])))
            print("        %s" % " ".join(n for n, _ in c["runs"][:8]))
        for c, nm, rn, subst, unexp in judged:
            if rn is None:
                print("       %s：配置读不到，无法判定" % nm)
            elif unexp:
                print("       %s：vs %s **实质差异 0 项**（只剩 run_name）"
                      "⟹ 配置解释不了 u1，**硬件不同（决定性）**" % (nm, rn))
            else:
                print("       %s：vs %s 实质差异 %d 项（%s）⟹ u1 差异**可由配置解释**"
                      % (nm, rn, len(subst), "、".join(subst[:3])))

    if not mixed:
        if A.quiet:
            print("u1 指纹：每个种子都只有 1 个簇 ⟹ 无跨节点混用（%d run）" % len(runs))
        else:
            print()
            print("### 结论")
            print("每个种子都只有 **1 个簇** ⟹ 全部 run 同节点，**无跨节点混用**。")
            seeds = sorted(by_seed)
            if seeds:
                print("    节点 = %s" % label_of(by_seed[seeds[0]][0][1]))
                print("    逐种子 u1：%s"
                      % " ".join("s%d=%.6f" % (s, by_seed[s][0][1]) for s in seeds))
    else:
        if A.quiet:
            print("u1 指纹：**%d 个种子出现多簇 ⟹ 有跨节点混用**" % len(mixed))
        else:
            print()
            print("### 结论：**发现跨节点混用**（%d 个种子）" % len(mixed))
            print("    ⟹ 跨簇的「臂 vs 臂」比较含硬件偏置（已记账 **+0.0197**）；")
            print("      跨簇的「臂 vs 专家」偏置**不抵消**（专家锚节点无关）。")
            print("      见 docs/测试规范.md §4⑧ 与 docs/训练诊断记录.md「★★★ 重标定终判」。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
