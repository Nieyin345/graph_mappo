#!/usr/bin/env bash
# 唤醒链的服务器端：每隔一段时间把状态写到 /tmp/qkd_status.json，
# 并在**验证曲线出现新点**或**任何 run 跑完**时打印一行事件。
#
# 本地用 Monitor 盯 stdout 的每一行 —— 每行 = 一次唤醒。
#
# 为什么要这个而不是简单 sleep：训练 30 轮 × 3 种子、每 5 轮一次验证，
# 一轮约 190s，新验证点大约每 16 分钟出现一次；跑完一个种子约 1.6 小时。
# 只在"有新信息"时唤醒，避免空转。
set -u
cd /opt/qkd/graph_mappo

STATE=/tmp/qkd_watch_state.json
PREV_VALS=""
PREV_DONE=""

echo "[$(date -Is)] 监视启动"

while true; do
  /opt/qkd/venv/bin/python /tmp/server_status.py > /tmp/qkd_status.txt 2>&1

  # 提取当前所有验证点数与已完成集合
  CUR=$(/opt/qkd/venv/bin/python - <<'PY' 2>/dev/null
import json
from pathlib import Path
s = json.loads(Path("/tmp/qkd_status.json").read_text(encoding="utf-8"))
vals = {k: [v[1] for v in d["vals"]] for k, d in s["runs"].items()}
print(json.dumps({"vals": vals, "done": sorted(s["finished"]),
                  "running": s["running"], "chain": s["chain_alive"]}))
PY
)

  if [ -n "$PREV_VALS" ] && [ "$CUR" != "$PREV_VALS" ]; then
    # 有新东西 —— 让本地醒一次
    echo "EVENT $(date -Is)"
    /opt/qkd/venv/bin/python /tmp/server_status.py 2>&1 | sed 's/^/  /'
    # 完成事件单独标出来
    NEWDONE=$(/opt/qkd/venv/bin/python - "$PREV_VALS" "$CUR" <<'PY' 2>/dev/null
import json, sys
a = json.loads(sys.argv[1]); b = json.loads(sys.argv[2])
new = set(b["done"]) - set(a["done"])
print(" ".join(sorted(new)))
PY
)
    if [ -n "$NEWDONE" ]; then
      echo "FINISHED $NEWDONE"
    fi
  fi
  PREV_VALS="$CUR"

  # 训练全停了且接力守护也没了 → 退出监视
  if ! pgrep -f "chain_r9ext|run_one_seed|screen_r9ext" >/dev/null 2>&1; then
    echo "ALLDONE $(date -Is) 所有训练与守护都已结束"
    /opt/qkd/venv/bin/python /tmp/server_status.py 2>&1 | sed 's/^/  /'
    exit 0
  fi
  sleep 150
done
