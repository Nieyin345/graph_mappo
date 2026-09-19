#!/usr/bin/env python
"""远端汇总器：读 outputs/<run>/ 的标量结果，打成一个 JSON 到 stdout。

被 .tmp/fetch_results.py 经 scp 送到服务器后执行。**不导出权重**。

用法（服务器上）：
  python /tmp/collect_runs.py all
  python /tmp/collect_runs.py r8_base_s42,r9ext_s53
"""
from __future__ import annotations

import glob
import json
import os
import sys

R = "/opt/qkd/graph_mappo/outputs"

ENV_KEYS = ("episode_steps", "activation_window_start_day",
            "activation_window_end_day", "activation_window_days",
            "episode_start_mode")
TRAIN_KEYS = ("episodes_per_update", "n_rollout_workers",
              "episode_steps_fixed", "seed")
PPO_KEYS = ("epochs", "minibatch_size", "batch_chunk", "clip_eps",
            "entropy_coef", "value_coef", "max_grad_norm", "target_kl",
            "lr", "normalize_advantages")
REWARD_KEYS = ("mode", "served_weight", "served_reference", "failed_weight",
               "keep_active_weight", "storage_weight", "success_delta_enabled")
DIAG_KEYS = ("kl", "entropy", "mean_abs_advantage", "value_return_corr",
             "actor_grad_norm", "critic_grad_norm", "n_minibatches")


def num(x):
    try:
        return round(float(x), 6)
    except (TypeError, ValueError):
        return None


def render_table(data: dict) -> str:
    """人类可读的汇总表，打到 stderr（stdout 保持纯 JSON）。"""
    lines = []
    lines.append(f"{'run':<18}{'轮数':>5}{'状态':>6}  验证曲线(最近5点)  最佳")
    lines.append("-" * 80)
    for name in sorted(data):
        rec = data[name]
        v = rec.get("validation") or []
        best = rec.get("best_val")
        fin = "完成" if rec["finished"] else "未完"
        vs = " ".join(f"{a}:{b}" for a, b in v[-5:]) or "(无)"
        bs = f"{best:.4f}" if best is not None else "-"
        lines.append(f"{name:<18}{rec['n_updates']:>5}{fin:>6}  {vs:<30} {bs}")

    # 训练/验证耗时对照（诊断内存墙是否复发）
    lines.append("")
    lines.append("每轮耗时（rollout_s / update_s）：首轮 vs 末轮")
    lines.append("-" * 80)
    for name in sorted(data):
        rec = data[name]
        h = rec.get("timing_head") or []
        t = rec.get("timing_tail") or []
        fmt = lambda x: (f"{x[0]:.0f}/{x[1]:.0f}" if x and x[0] and x[1] else "?")
        lines.append(f"{name:<18} {fmt(h[0] if h else None)}  →  "
                     f"{fmt(t[-1] if t else None)}")
    return "\n".join(lines)


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "all"
    if spec == "all":
        names = [os.path.basename(d) for d in sorted(glob.glob(R + "/*"))
                 if os.path.isdir(d)]
    else:
        names = [n.strip() for n in spec.split(",") if n.strip()]

    out = {}
    for n in names:
        d = os.path.join(R, n)
        if not os.path.isdir(d):
            continue

        rows = []
        f = os.path.join(d, "metrics.jsonl")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        rec = {
            "name": n,
            "n_updates": len(rows),
            "finished": os.path.exists(os.path.join(d, "checkpoint_final.pt")),
        }

        vals, upd = [], 0
        for r in rows:
            if "update" in r:
                upd = max(upd, int(r["update"]))
            if "eval_validation" in r:
                ev = r["eval_validation"]
                vals.append([upd, num(ev.get("mean_success_rate", 0.0))])
        rec["validation"] = vals
        rec["best_val"] = max((v for _, v in vals), default=None)
        rec["train_success"] = [num(r.get("mean_success_rate")) for r in rows]

        t = [[num(r.get("rollout_s")), num(r.get("update_s"))] for r in rows]
        rec["timing_head"] = t[:3]
        rec["timing_tail"] = t[-3:]

        for k in DIAG_KEYS:
            rec[k] = [num(r.get(k)) for r in rows if r.get(k) is not None]

        try:
            import yaml
        except ImportError:
            yaml = None
        cf = os.path.join(d, "resolved_config.yaml")
        if yaml and os.path.exists(cf):
            with open(cf, encoding="utf-8") as fh:
                c = yaml.safe_load(fh) or {}
            tr = c.get("train", {}) or {}
            env = c.get("env", {}) or {}
            rec["cfg_env"] = {k: env.get(k) for k in ENV_KEYS}
            rec["cfg_train"] = {k: tr.get(k) for k in TRAIN_KEYS}
            rec["cfg_ppo"] = {k: (tr.get("ppo") or {}).get(k) for k in PPO_KEYS}
            rw = c.get("reward") or env.get("reward") or {}
            rec["cfg_reward"] = ({k: rw.get(k) for k in REWARD_KEYS} if rw
                                 else None)

        out[n] = rec

    if "--json" in sys.argv:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(render_table(out))


if __name__ == "__main__":
    main()
