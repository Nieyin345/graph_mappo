#!/usr/bin/env python
"""启动前最后一道核对：每个臂相对 ent01 基线，**真的只差一个字段**吗？

为什么要写它：本轮三个臂的全部效力都建立在"唯一变量"上。而配置链是
default → env_full → profile → global → --configs 依次深合并，**后面覆盖前面**，
一个文件少写一层就可能让某个字段悄悄回落到别的值（本项目已经发生过：
`train_full_rl.yaml` 不设 train.ppo，于是整段回落到 rl_algorithm.yaml，
把调过的 entropy_coef/batch_chunk 全盖掉了）。

做法：直接调 `build_config()`（训练入口自己的函数），对每个臂构造完整配置，
与基线逐字段比对。**不是读 yaml 猜**，是走训练真正的构造路径。

用法（服务器上）：python /tmp/verify_arms.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))

from scripts.train.train_graph_mappo import build_config  # noqa: E402


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def cfg_for(configs, run_name, seed=42):
    args = argparse.Namespace(
        configs=configs,
        mode="random_episode",
        run_name=run_name,
        num_updates=30,
        seed=seed,
        checkpoint="outputs/supervised_pg_phased/supervised_pg_phased_latest.pt",
        device="cpu",
    )
    return flatten(build_config(args))


BASE_CFGS = ["rl_algorithm.yaml", "train_full_rl.yaml"]


ARMS = {
    "ent01 (基线)": BASE_CFGS + ["train_ent01.yaml"],
    "vcoef1":      BASE_CFGS + ["train_ent01.yaml", "train_safe_vcoef1.yaml"],
    "ep2":         BASE_CFGS + ["train_ent01.yaml", "train_safe_ep2.yaml"],
    "mini512":     BASE_CFGS + ["train_ent01.yaml", "train_safe_mini512.yaml"],
    "runt (锚点)": BASE_CFGS + ["train_ent01.yaml"],
}

# 这些字段本来就该随 run 名/臂不同，不算"变量"
IGNORE = {"project.run_name"}

built = {}
for name, cfgs in ARMS.items():
    try:
        built[name] = cfg_for(cfgs, f"probe_{name.split()[0]}")
    except Exception as e:  # noqa: BLE001
        print(f"!! {name} 构造失败: {type(e).__name__}: {e}")
        raise SystemExit(1)

base = built["ent01 (基线)"]
print(f"基线字段数: {len(base)}")
print()

EXPECTED = {
    "vcoef1":  {"train.ppo.value_coef": 1.0},
    "ep2":     {"train.ppo.epochs": 2},
    "mini512": {"train.ppo.minibatch_size": 512},
    "runt (锚点)": {},
}

ok = True
for name, exp in EXPECTED.items():
    cur = built[name]
    keys = set(base) | set(cur)
    diffs = {}
    for k in keys:
        if k in IGNORE:
            continue
        if base.get(k, "<缺>") != cur.get(k, "<缺>"):
            diffs[k] = (base.get(k, "<缺>"), cur.get(k, "<缺>"))
    print(f"=== {name} ===")
    if not diffs:
        print("  与 ent01 逐字段完全一致")
    for k, (a, b) in sorted(diffs.items()):
        mark = "  ← 预期" if exp.get(k) == b else "  ★ 意外差异"
        print(f"  {k}: {a} → {b}{mark}")
    # 判据：差异集合必须恰好等于预期集合。
    #
    # 注意这里**不能**用 for/else：上一版写成
    #     for k, v in exp.items():
    #         if diffs[k][1] != v: ok = False
    #     else:                      # ← 这个 else 挂在 for 上，不是 if
    #         if set(diffs) == set(exp): print("唯一变量成立")
    # for/else 的 else 在**循环正常结束**时执行（即没有 break），与 if 无关；
    # 当时它恰好打印了"成立"，但那是巧合而不是判断 —— 一个总说"成立"的
    # 校验器比没有校验器更糟，因为它会把配置链错误放行到 2 小时的训练里。
    mismatched = set(diffs) != set(exp) or any(diffs[k][1] != v for k, v in exp.items())
    if mismatched:
        print(f"  ✗ 差异字段 {sorted(diffs)} / 预期 {sorted(exp)}，或取值不符")
        ok = False
    else:
        print("  ✓ 唯一变量成立")
    print()

print("=" * 60)
if ok:
    print("全部臂的唯一变量成立 → 可以启动")
    raise SystemExit(0)
print("有臂不是唯一变量 → **不要启动**，先修配置链")
raise SystemExit(1)
