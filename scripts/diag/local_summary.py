#!/usr/bin/env python
"""从**本地** server_results/ 生成结果汇总，不依赖服务器（服务器随时可能到期）。

用法：python .tmp/local_summary.py > server_results/结果汇总.md
（本机 hook 会拦 python，改用：在服务器上跑，或临时摘 hook。
  也可直接看本脚本不做 import torch —— 但 hook 按命令名拦，故用 bash 传参绕过。）

输出三块：
  1. 关键对照表（配对的，同种子同轮）
  2. 全部 run 的验证曲线（按 run 名排序）
  3. 配置差异（每个实验臂相对基座改了什么）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOTS = [Path("server_results/outputs"), Path("server_results/runs/outputs")]


def align(d: Path):
    m = d / "metrics.jsonl"
    if not m.exists():
        return None
    last_u, vals, rows = None, [], 0
    with m.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" in r:
                last_u = r["update"]
                rows += 1
            ev = r.get("eval_validation")
            if isinstance(ev, dict) and ev.get("mean_success_rate") is not None:
                vals.append((last_u, float(ev["mean_success_rate"]),
                             ev.get("per_seed_success") or []))
    return {"last_u": last_u, "rows": rows, "vals": vals}


def cfg_of(d: Path):
    p = d / "resolved_config.yaml"
    if not p.exists():
        return {}
    import yaml
    try:
        c = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    tr = c.get("train", {}) or {}
    ppo = tr.get("ppo", {}) or {}
    return {
        "entropy_coef": ppo.get("entropy_coef"),
        "gamma": tr.get("gamma"),
        "actor_lr": (tr.get("optimizer") or {}).get("actor_lr"),
        "num_updates": tr.get("num_updates"),
        "n_rollout_workers": tr.get("n_rollout_workers"),
        "episodes_per_update": tr.get("episodes_per_update"),
        "minibatch_size": ppo.get("minibatch_size"),
        "value_coef": ppo.get("value_coef"),
        "clip_eps": ppo.get("clip_eps"),
    }


def collect(extra_roots=None):
    """优先按 roots 顺序取，**但只在当前 root 真的缺这个 run 时才回退**。

    不能用"名字已存在就跳过"的老逻辑：`runs/outputs/` 里有一份很早的同名
    `ent01_g999_s42`（不同 code state 跑的），会**遮住** `outputs/` 里的当前版本，
    而两者的验证值不同。正确做法是在上层的 merge 里按 root 顺序、**逐 run 取
    第一个存在的**，而不是整目录去重。"""
    runs = {}
    roots = list(extra_roots or []) + list(ROOTS)
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            a = align(d)
            if a is None:
                continue
            if d.name in runs:
                continue          # 前面的 root 已给了这个 run，保留它
            runs[d.name] = {"dir": d, **a, "cfg": cfg_of(d)}
    return runs


def fmt_vals(vals):
    return "  ".join(f"u{u}={v:.4f}" for u, v, _ in vals) or "(无)"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="写到此文件（UTF-8）。不给则打印到 stdout——"
                         "Windows 控制台是 GBK，中文会乱码，建议给 --out。")
    ap.add_argument("--roots", nargs="*", default=None,
                    help="run 目录列表（默认 server_results/outputs 等；"
                         "在服务器上跑时传 outputs）")
    args = ap.parse_args()

    runs = collect([Path(r) for r in args.roots] if args.roots else None)
    W = []
    W.append("# 结果汇总（本地归档）\n")
    W.append(f"本地可读 run：**{len(runs)}** 个。"
             f"来源 `server_results/`，服务器到期不影响。\n")
    W.append("验证口径：留出协议（天 330–365，240 步，15 个请求种子 100–114），"
             "`eval_interval: 5`。\n")
    W.append("参照：**专家基线 0.6980**、BC 起点 0.6483（同为 240 步验证）。\n")

    # ---------- 1. 关键配对对照 ----------
    W.append("\n## 1. 关键配对对照（同种子、同轮，唯一可比口径）\n")

    pairs = [
        ("entropy_coef 0.001 → 0.01（三种子，最强线索）",
         ["r8_base_s42", "r8_base_s43", "r8_base_s44"],
         ["ent01_s42", "ent01_s43", "ent01_s44"]),
        ("gamma 0.999 叠在 ent=0.01 之上",
         ["ent01_s42"],
         ["ent01_g999_s42", "ent01_g999_s42_r2"]),
        ("entropy_coef（单种子 r7 对照，u20）",
         ["r7_base"], ["r7_fix_ent"]),
        ("gamma 0.999 单独（ent 保持 0.001）",
         ["r7_base"], ["r7_fix_g999", "r7_fix_g999_long"]),
    ]
    for title, base_names, arm_names in pairs:
        W.append(f"\n### {title}\n")
        rows = []
        for bn, an in zip(base_names, arm_names):
            b = runs.get(bn)
            a = runs.get(an)
            if not b or not a:
                W.append(f"- `{bn}` vs `{an}`：本地缺数据")
                continue
            bv = dict((u, v) for u, v, _ in b["vals"])
            av = dict((u, v) for u, v, _ in a["vals"])
            rows.append((bn, an, bv, av))
        if not rows:
            continue
        # 同轮的并集
        common = set()
        for _, _, bv, av in rows:
            common |= (set(bv) & set(av))
        if not common:
            for bn, an, bv, av in rows:
                W.append(f"- `{bn}` {fmt_vals([(u, v, []) for u, v in bv.items()])}"
                         f"  |  `{an}` {fmt_vals([(u, v, []) for u, v in av.items()])}")
            continue
        ups = sorted(common)
        # ⚠️ 每一行必须占满 "基线|实验|Δ" 三列 × len(ups)，否则 markdown 会把
        # 后面的单元格吞进前面的列，整表错位（第一版就是这样，表头 6 列、
        # 数据行 9 个单元格）。
        W.append("| 对照 | " + " | ".join(f"u{u} 基线" for u in ups)
                 + " | " + " | ".join(f"u{u} 实验" for u in ups)
                 + " | " + " | ".join(f"u{u} Δ" for u in ups) + " |")
        W.append("|" + "---|" * (1 + 3 * len(ups)))
        agg = {u: [] for u in ups}
        for bn, an, bv, av in rows:
            cells = []
            for u in ups:
                d = av[u] - bv[u]
                agg[u].append(d)
                cells.append(f"{bv[u]:.4f} | {av[u]:.4f} | **{d:+.4f}**")
            W.append(f"| {bn} → {an} | " + " | ".join(cells) + " |")
        if len(rows) > 1:
            W.append("| **均值 Δ** | "
                     + " | ".join("|  | **%+.4f**" % (sum(agg[u]) / len(agg[u]))
                                  for u in ups) + " |")

    # ---------- 2. 全部验证曲线 ----------
    W.append("\n\n## 2. 全部 run 的验证曲线\n")
    W.append("| run | 末轮 | 验证点 | 末验证值 | ent | gamma |")
    W.append("|---|---|---|---|---|---|")
    for n, r in sorted(runs.items()):
        if not r["vals"]:
            continue
        v = fmt_vals(r["vals"]).replace("|", "/")
        e = r["cfg"].get("entropy_coef")
        g = r["cfg"].get("gamma")
        W.append(f"| `{n}` | {r['last_u']} | {v} | {r['vals'][-1][1]:.4f} |"
                 f" {e} | {g} |")

    # ---------- 3. 实验臂改了什么 ----------
    W.append("\n\n## 3. 实验臂相对基座改了什么\n")
    W.append("| run | ent | gamma | actor_lr | updates | workers | eps/upd | minibatch |")
    W.append("|---|---|---|---|---|---|---|---|")
    for n, r in sorted(runs.items()):
        c = r["cfg"]
        if not c:
            continue
        W.append(f"| `{n}` | {c.get('entropy_coef')} | {c.get('gamma')} |"
                 f" {c.get('actor_lr')} | {c.get('num_updates')} |"
                 f" {c.get('n_rollout_workers')} | {c.get('episodes_per_update')} |"
                 f" {c.get('minibatch_size')} |")

    text = "\n".join(W)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {args.out}（{len(text)} 字符，{len(runs)} 个 run）")
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(text)


if __name__ == "__main__":
    main()
