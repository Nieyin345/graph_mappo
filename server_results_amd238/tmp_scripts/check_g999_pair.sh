#!/usr/bin/env bash
# g999_s42_r2 与 ent01_s42 是否同起点？（同 BC 检查点 + 同 seed → 请求流逐位相同）
cd /opt/qkd/graph_mappo || exit 1

/opt/qkd/venv/bin/python - <<'PY'
import json, yaml
from pathlib import Path
OUT = Path("/opt/qkd/graph_mappo/outputs")

def cfg(run):
    p = OUT / run / "resolved_config.yaml"
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

keys = ["entropy_coef", "gamma", "num_updates", "episodes_per_update",
        "rollout_steps"]

for run in ("ent01_s42", "ent01_g999_s42_r2"):
    c = cfg(run) or {}
    g, tr = c.get("global", {}) or {}, c.get("train", {}) or {}
    ppo = tr.get("ppo", {}) or {}
    v = c.get("validation", {}) or {}
    print(f"--- {run}")
    print(f"    seed(global)      = {g.get('seed')}")
    print(f"    env_seed          = {(c.get('env',{}) or {}).get('seed')}")
    print(f"    request_seed      = {(g.get('training',{}) or {}).get('request_seed')}")
    print(f"    entropy_coef      = {ppo.get('entropy_coef')}")
    print(f"    gamma             = {ppo.get('gamma', tr.get('gamma'))}")
    print(f"    init_checkpoint   = {tr.get('init_checkpoint') or tr.get('checkpoint')}")
    print(f"    episodes_per_upd  = {tr.get('episodes_per_update')}")
    print(f"    val request_seeds = {v.get('request_seeds')}")

# 逐种子对照：同 seed 同起点时，逐请求种子配对才有意义
print()
print("=== 逐种子（验证实例）配对：ent01_g999_s42_r2 − ent01_s42 ===")
def pts(run):
    out = []
    for line in (OUT / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        ev = r.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success"):
            out.append((int(ev.get("seeds", [0])[0]) if ev.get("seeds") else None,
                        list(ev["per_seed_success"]), list(ev.get("seeds") or [])))
    return out

a, b = pts("ent01_g999_s42_r2"), pts("ent01_s42")
if len(a) == len(b):
    import statistics
    print(f"{'轮':>4}{'g999':>9}{'ent01':>9}{'Δ':>9}{'SE':>8}{'t':>7}{'胜/负':>8}")
    for i, (ga, sb, sda) in enumerate(a):
        gb, sc, sdb = b[i]
        if sda != sdb:
            print(f"  第{i}点种子集不同，跳过")
            continue
        d = [sb[k] - sc[k] for k in range(len(sb))]
        m = sum(d) / len(d)
        se = statistics.stdev(d) / len(d) ** 0.5
        w = sum(1 for x in d if x > 0)
        print(f"{i*5+5:>4}{sum(sb)/len(sb):>9.4f}{sum(sc)/len(sc):>9.4f}"
              f"{m:>+9.4f}{se:>8.4f}{m/se:>+7.2f}{w:>5}/{len(d)-w:<3}")
else:
    print(f"  验证点数不同：g999={len(a)} ent01={len(b)}")
PY
