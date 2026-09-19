#!/usr/bin/env bash
# 推脚本到服务器 **并先做语法检查**。
#
# ### 为什么需要这个
#
# 本项目已经**至少 6 次**因为「双引号字符串里又嵌了双引号」把脚本推上去后
# 才在服务器上报 SyntaxError，每次白跑一个来回。典型：
#
#     print("② 找"每决策可选弧数"的读数")      # ← 语法错
#     print(f"  分辨不出"单调"还是"过峰"。")   # ← 语法错
#
# 本地 hook 禁止跑 `python`（内存原因），所以本地查不了语法。
# 但 `ssh` 是放行的 —— 那就**把语法检查放到服务器上做**，在 scp 之后、
# 使用之前。这样一次往返内就能发现问题，而不是等到跑判读时。
#
# 用法：
#   bash scripts/diag/push.sh .tmp/foo.py                 # -> /tmp/foo.py
#   bash scripts/diag/push.sh .tmp/foo.py /tmp/bar.py     # 指定远端路径
#   bash scripts/diag/push.sh .tmp/foo.sh                 # .sh 用 bash -n
#   bash scripts/diag/push.sh configs/x.yaml              # .yaml 用 yaml.safe_load
#
# 退出码：0 = 推送且语法 OK；非 0 = 有问题（此时**不要**继续用它跑实验）。
set -u
HOST="${QKD_HOST:-qkd}"

if [ $# -lt 1 ]; then
  echo "用法: bash scripts/diag/push.sh <本地路径> [远端路径]" >&2
  exit 2
fi

SRC="$1"
if [ $# -ge 2 ]; then
  DST="$2"
else
  DST="/tmp/$(basename "$SRC")"
fi

if [ ! -f "$SRC" ]; then
  echo "!! 本地不存在: $SRC" >&2
  exit 2
fi

scp -q -o ConnectTimeout=25 -o BatchMode=yes "$HOST:$DST" /dev/null 2>/dev/null || true
if ! scp -q -o ConnectTimeout=25 -o BatchMode=yes "$SRC" "$HOST:$DST"; then
  echo "!! scp 失败" >&2
  exit 1
fi

# 按扩展名选检查方式，全部在服务器上做
case "$SRC" in
  *.py)
    OUT=$(ssh -o ConnectTimeout=30 -o BatchMode=yes "$HOST" \
      "/opt/qkd/venv/bin/python -m py_compile '$DST' && echo __OK__" 2>&1)
    ;;
  *.sh)
    OUT=$(ssh -o ConnectTimeout=30 -o BatchMode=yes "$HOST" \
      "bash -n '$DST' && echo __OK__" 2>&1)
    ;;
  *.yaml|*.yml)
    # 用一个小脚本而不是 python -c（本项目禁止内联命令）
    ssh -o ConnectTimeout=30 -o BatchMode=yes "$HOST" \
      "cat > /tmp/_push_yaml_check.py <<'PYEOF'
import sys, yaml
try:
    yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
    print('__OK__')
except Exception as e:
    print('YAMLERR:', e)
PYEOF
/opt/qkd/venv/bin/python /tmp/_push_yaml_check.py '$DST'" > /tmp/_push_out.txt 2>&1
    OUT=$(cat /tmp/_push_out.txt)
    ;;
  *)
    OUT="__OK__"
    ;;
esac

if printf '%s' "$OUT" | grep -q "__OK__"; then
  echo "OK  $SRC -> $HOST:$DST  （语法通过）"
  exit 0
fi

echo "!! 语法检查失败  $SRC -> $HOST:$DST" >&2
printf '%s\n' "$OUT" | head -12 | sed 's/^/    /' >&2
exit 1
