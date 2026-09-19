#!/usr/bin/env bash
# 接力：等冒烟结束 -> 核对健康 -> 自动启动第七浪 -> 回报。
# 冒烟不健康就不启动第七浪（省一晚机时）。
set -uo pipefail
SSH="C:/Windows/System32/OpenSSH/ssh.exe"
SCP="C:/Windows/System32/OpenSSH/scp.exe"
OPTS="-o BatchMode=yes -o ConnectTimeout=15"
R="cd /opt/qkd/graph_mappo"

echo "=== 等冒烟结束 $(date +%H:%M:%S) ==="
deadline=$(( $(date +%s) + 2400 ))
while $SSH $OPTS qkd 'pgrep -f "smoke_tfi[x]" >/dev/null' 2>/dev/null; do
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "!! 冒烟超时"; exit 1; fi
    sleep 20
done
echo "冒烟结束 $(date +%H:%M:%S)"

echo
echo "=== 冒烟结果 ==="
$SSH $OPTS qkd "$R && cat /tmp/smoke_tfix.out 2>/dev/null; echo '--- metrics ---'; grep -c . outputs/smoke_tfix/metrics.jsonl 2>/dev/null"

echo
echo "=== 健康核对 ==="
VERDICT=$($SSH $OPTS qkd "$R && /opt/qkd/venv/bin/python - <<'PY'
import json, pathlib, math
p = pathlib.Path('outputs/smoke_tfix/metrics.jsonl')
rows = []
if p.exists():
    for line in p.open(encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(d, dict) and 'update' in d: rows.append(d)
print(f'轮数={len(rows)}')
bad = []
for r in rows:
    for k in ('actor_loss','critic_loss','mean_reward','mean_abs_advantage','update_s'):
        v = r.get(k)
        if not isinstance(v,(int,float)) or math.isnan(v) or math.isinf(v):
            bad.append(f\"update{r.get('update')}: {k}={v}\")
print('异常值:', bad if bad else '无')
tr = [r for r in rows if isinstance(r.get('mean_abs_advantage'),(int,float))]
if tr:
    print('mean_abs_advantage:', ' '.join(f\"{r['mean_abs_advantage']:.3f}\" for r in tr))
    print('update_s:', ' '.join(f\"{r['update_s']:.1f}\" for r in tr))
ok = len(rows) >= 2 and not bad
print('VERDICT=' + ('OK' if ok else 'BAD'))
PY" 2>/dev/null | tail -8)
echo "$VERDICT"

if echo "$VERDICT" | grep -q "VERDICT=OK"; then
    echo
    echo "=== 冒烟健康，启动第七浪 ==="
    $SSH $OPTS qkd "$R && setsid nohup bash .tmp/screen_r7.sh > /tmp/screen_r7.out 2>&1 < /dev/null & sleep 3; pgrep -f 'screen_r[7]' >/dev/null && echo '第七浪已启动' || echo '启动失败'"
    echo "=== 接力完成：第七浪在跑 $(date +%H:%M:%S) ==="
else
    echo
    echo "!!! 冒烟不健康，第七浪未启动 —— 需要人工检查 !!!"
    exit 1
fi
