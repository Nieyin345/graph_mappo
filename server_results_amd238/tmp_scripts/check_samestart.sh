#!/usr/bin/env bash
# 核对 ent01_s42 与 ent01_g999_s42_r2 是否同起点（同一 BC 检查点 + 同一 seed）。
cd /opt/qkd/graph_mappo || exit 1

for r in ent01_s42 ent01_g999_s42_r2; do
  echo "########## $r"
  echo "--- 日志头部（启动配置）---"
  head -25 "/tmp/$r.log" 2>/dev/null | grep -iE "checkpoint|seed|gamma|config|loaded|resume" | head -12
  echo "--- 日志里的 checkpoint 行 ---"
  grep -iE "loading checkpoint|loaded checkpoint|init_checkpoint|resuming" "/tmp/$r.log" 2>/dev/null | head -5
  echo
done

echo "########## 两臂的 base_seed 是否逐位一致（前 3 个验证点的请求流）"
/opt/qkd/venv/bin/python - <<'PY'
import yaml
from pathlib import Path
OUT = Path("/opt/qkd/graph_mappo/outputs")
for run in ("ent01_s42", "ent01_g999_s42_r2"):
    c = yaml.safe_load((OUT / run / "resolved_config.yaml").read_text(encoding="utf-8"))
    def dig(d, *ks):
        for k in ks:
            d = (d or {}).get(k) if isinstance(d, dict) else None
        return d
    print(f"{run}:")
    for label, path in [
        ("global.seed", ("global", "seed")),
        ("global.training.request_seed", ("global", "training", "request_seed")),
        ("env.seed", ("env", "seed")),
        ("scenario.seed", ("scenario", "seed")),
        ("seed (top)", ("seed",)),
        ("train.seed", ("train", "seed")),
    ]:
        print(f"    {label:<30} = {dig(c, *path)}")
    # 把整个顶层里所有含 'seed' 的键都倒出来
    def walk(d, pre=""):
        if isinstance(d, dict):
            for k, v in d.items():
                if "seed" in str(k).lower() and not isinstance(v, (dict, list)):
                    print(f"    [walk] {pre}{k} = {v}")
                walk(v, f"{pre}{k}.")
        elif isinstance(d, list):
            for i, v in enumerate(d[:2]):
                walk(v, f"{pre}{i}.")
    walk(c)
PY
