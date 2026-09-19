#!/usr/bin/env python
"""系统性挖掘：对每个配置字段，找出「只在它上面有差异」的 run 对，算结果差。

动机：entropy_coef 0.001→0.01 这个结论是手工 diff 两个 run 挖出来的。
历史 outputs/ 里有几十个 run，应该把这件事自动化——**答案可能已经在
跑过的实验里，只是没人做两两对比**。重新跑实验比挖历史贵几个数量级。

方法：
  1. 读所有 run 的 resolved_config，展平成 {字段: 值}
  2. 读所有 run 的验证曲线，取「共同评估轮数」内的均值/最佳值
  3. 对每个字段 f，遍历所有 run 对 (A,B)：
       - 若两者除 f 外**所有配置字段相同** → 是干净的单变量对照
       - 同类 run 对按 f 的取值分组，比较结果
  4. 只报告「同种子」的对照（种子相同 → 配对，分辨率更高）

局限（必须显式说明，避免过度解读）：
  - 只对**同种子**的对做配对汇总；不同种子的对只列为参考
  - 单变量对可能很少，样本小时不做显著性判断，只列原始数字
  - 结果取「各 run 验证曲线的共同轮数区间」，避免轮数不同造成的偏差

用法（服务器上）：
  python /tmp/mine_config_effects.py --metric best --same-seed-only
  python /tmp/mine_config_effects.py --metric curve --min-pairs 2
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from collections import defaultdict

import yaml

R = "/opt/qkd/graph_mappo/outputs"

# 不参与「配置相同」判断的字段（run 身份、路径、时间戳）
IGNORE_PREFIX = ("project.run_name", "project.output_dir")
IGNORE_EXACT = {
    "project.run_name", "project.output_dir", "project.name",
    "runtime.device", "runtime.num_threads", "runtime.torch_num_threads",
}


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, val in d.items():
            out.update(flat(val, f"{prefix}{k}."))
    elif isinstance(d, list):
        out[prefix[:-1]] = json.dumps(d, sort_keys=True, ensure_ascii=False)
    else:
        out[prefix[:-1]] = d
    return out


def load_cfg(name):
    p = os.path.join(R, name, "resolved_config.yaml")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        c = yaml.safe_load(fh) or {}
    f = flat(c)
    return {k: v for k, v in f.items()
            if k not in IGNORE_EXACT
            and not any(k.startswith(p2) for p2 in IGNORE_PREFIX)}


def load_curve(name):
    """返回 {update: val} 的验证曲线。"""
    p = os.path.join(R, name, "metrics.jsonl")
    if not os.path.exists(p):
        return {}
    out, upd = {}, 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" in r:
                upd = max(upd, int(r["update"]))
            if "eval_validation" in r:
                out[upd] = float(r["eval_validation"].get("mean_success_rate", 0))
    return out


def seed_of(cfg):
    return (cfg.get("seed.global_seed"), cfg.get("seed.env_seed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["best", "curve", "final"],
                    default="best")
    ap.add_argument("--same-seed-only", action="store_true")
    ap.add_argument("--min-pairs", type=int, default=1)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    names = [os.path.basename(d) for d in sorted(glob.glob(R + "/*"))
             if os.path.isdir(d)]
    cfgs, curves = {}, {}
    for n in names:
        c = load_cfg(n)
        cv = load_curve(n)
        if c and cv:                    # 只保留有验证曲线的
            cfgs[n] = c
            curves[n] = cv

    print(f"有验证曲线的 run：{len(cfgs)} 个\n")

    def outcome(n):
        cv = curves[n]
        if args.metric == "best":
            return max(cv.values())
        if args.metric == "final":
            return cv[max(cv)]
        return sum(cv.values()) / len(cv)

    # 找单变量对照对
    pairs_by_field = defaultdict(list)
    for a, b in itertools.combinations(sorted(cfgs), 2):
        ca, cb = cfgs[a], cfgs[b]
        keys = set(ca) | set(cb)
        diff = [k for k in keys if ca.get(k, "<缺>") != cb.get(k, "<缺>")]
        if len(diff) != 1:
            continue
        f = diff[0]
        same_seed = seed_of(ca) == seed_of(cb) and seed_of(ca)[0] is not None
        if args.same_seed_only and not same_seed:
            continue
        pairs_by_field[f].append((a, b, ca.get(f), cb.get(f), same_seed))

    print(f"找到 {sum(len(v) for v in pairs_by_field.values())} 个单变量对照对，"
          f"分布在 {len(pairs_by_field)} 个字段上\n")

    # 按「|Δ结果| 的均值」排序字段
    report = []
    for f, pairs in pairs_by_field.items():
        if len(pairs) < args.min_pairs:
            continue
        ds = []
        for a, b, va, vb, ss in pairs:
            d = outcome(b) - outcome(a)
            ds.append((abs(d), d, a, b, va, vb, ss))
        ds.sort(reverse=True)
        mean_abs = sum(x[0] for x in ds) / len(ds)
        report.append((mean_abs, len(ds), f, ds))

    report.sort(reverse=True)

    print(f"{'字段':<38}{'对数':>5}{'平均|Δ|':>10}  最大的一对")
    print("-" * 100)
    for mean_abs, n, f, ds in report[:args.top]:
        _, d, a, b, va, vb, ss = ds[0]
        tag = "★配对" if ss else "  "
        print(f"{f:<38}{n:>5}{mean_abs:>10.4f}  "
              f"{a}({va}) → {b}({vb})  Δ{d:+.4f} {tag}")

    print("\n\n=== 同种子配对对照明细（分辨率最高，优先看）===")
    shown = 0
    for mean_abs, n, f, ds in report:
        ss_pairs = [x for x in ds if x[6]]
        if not ss_pairs:
            continue
        print(f"\n{f}")
        for _, d, a, b, va, vb, _ in ss_pairs[:4]:
            print(f"    {a:<20} ({f.split('.')[-1]}={va})  →  "
                  f"{b:<20} (={vb})   Δ{d:+.4f}")
        shown += 1
        if shown >= 12:
            break


if __name__ == "__main__":
    main()
