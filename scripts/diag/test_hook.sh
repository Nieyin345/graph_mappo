#!/usr/bin/env bash
# 测 .claude/hooks/no_local_python.sh 的修订版：
#   · 该放行的轻量 python 任务 —— 必须放行（原版误拦了这类）
#   · 该拦的 torch / qkd_rl 任务 —— 一条都不能漏（这是安全底线）
#
# 判据：退出码 0 = 放行，2 = 拦截。
set -u
cd "D:/destop/work_space/learning_space/论文/QKD-SAGIN生产端调度/qkd_rl" || exit 1
HOOK=".claude/hooks/no_local_python.sh"

# 造两个测试文件，分别对应"轻量"和"重量"
cat > .tmp/_light_probe.py <<'PY'
import json, sys
print("light")
PY
cat > .tmp/_heavy_probe.py <<'PY'
import torch
from qkd_rl.env import QKDEnv
print("heavy")
PY

json_payload() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    printf '{"tool_input":{"command":"%s"}}' "$s"
}

pass=0; fail=0
t() {  # $1=期望退出码  $2=说明  $3=命令
    local want="$1" desc="$2" c="$3" rc
    json_payload "$c" | bash "$HOOK" >/dev/null 2>&1
    rc=$?
    if [ "$rc" = "$want" ]; then
        printf '  ✓ %-6s %s\n' "rc=$rc" "$desc"
        pass=$((pass+1))
    else
        printf '  ✗ 期望 rc=%s 实得 rc=%s   %s\n' "$want" "$rc" "$desc"
        printf '        命令: %s\n' "$c"
        fail=$((fail+1))
    fi
}

echo "jq 可用: $(command -v jq >/dev/null 2>&1 && echo 是 || echo 否（走 sed 退路）)"
echo
echo "=== 应当放行 (rc=0) —— 轻量 / 不跑本项目代码 ==="
t 0 "标准库脚本（原版误拦的就是这类）" 'python 模板库/代码/rules.py list'
t 0 "标准库脚本带重定向"                 'python 模板库/代码/rules.py get sci-layout'
t 0 "轻量 .py 文件（只 import json）"    'python .tmp/_light_probe.py'
t 0 "纯文本处理 -m json.tool"           'python -m json.tool .tmp/x.json'
t 0 "无 import 的内联片段"              'python -c "print(1+1)"'
t 0 "git 命令"                          'git status --short'
t 0 "ls"                                'ls -la'
t 0 "grep 提到 python（只是提及）"       'grep -rn "python" docs/测试规范.md'
t 0 "ps 查看（只是提及）"                'ps -W | grep python'
t 0 "git add 提到 .py（只是提及）"       'git add scripts/diag/push.sh'

echo
echo "=== 应当放行 (rc=0) —— ssh/scp/rsync 永远放行 ==="
t 0 "ssh 里跑 torch"                    'ssh qkd '"'"'cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python -m pytest tests/'"'"''
t 0 "scp 推送"                          'scp .tmp/foo.py qkd:/tmp/foo.py'
t 0 "rsync"                             'rsync -av qkd:/opt/qkd/graph_mappo/outputs ./outputs'
t 0 "ssh.exe 变量间接调用"               'SSH=/c/Windows/System32/OpenSSH/ssh.exe; "$SSH" -o BatchMode=yes qkd "python x.py"'

echo
echo "=== 应当拦截 (rc=2) —— 会拉起 torch / 本项目 env ==="
t 2 "pytest 裸调（原版拦，修订版必须仍拦）" 'pytest tests/ -x -q'
t 2 "python -m pytest"                   'python -m pytest tests/test_gae.py -q'
t 2 "内联 import torch"                  'python -c "import torch; print(1)"'
t 2 "内联 from qkd_rl"                   'python -c "from qkd_rl.env import QKDEnv"'
t 2 "训练入口脚本"                        'python scripts/train/train_graph_mappo.py --configs rl_algorithm.yaml'
t 2 "重量 .py（import torch/qkd_rl）"     'python .tmp/_heavy_probe.py'
t 2 "venv 绝对路径跑 pytest"              '/opt/qkd/venv/bin/python -m pytest tests/'
t 2 "-m qkd_rl 模块"                      'python -m qkd_rl.rl.algos.mappo_trainer'
t 2 "scripts/baselines 目录"              'python scripts/baselines/run_baselines.py'
t 2 "eval_expert 入口"                    'python scripts/eval/eval_expert.py --seeds 7-21'

echo
echo "========================"
printf '通过 %d  失败 %d\n' "$pass" "$fail"
rm -f .tmp/_light_probe.py .tmp/_heavy_probe.py
[ "$fail" = 0 ]
