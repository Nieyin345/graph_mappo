#!/usr/bin/env bash
# 推送被拒：节点上有 6 个「未跟踪」文件会被快照覆盖。
# **先看目标再覆盖** —— 逐个比对节点内容与本地，确认没有节点独有改动，
# 再删掉它们让快照写回。若哪个文件真的不一样，必须停下来查清是谁改的。
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
WORK=.tmp/nodecheck
rm -rf "$WORK"; mkdir -p "$WORK"

FILES="configs/train_ent01.yaml configs/train_ent01_g999.yaml configs/train_ent01_off.yaml configs/train_window_329.yaml tests/test_eval_determinism.py"

echo "=== 逐文件比对（行尾归一化）==="
same=0; diffcnt=0
for f in $FILES; do
  ssh qkd "cat /opt/qkd/graph_mappo/$f 2>/dev/null" > "$WORK/n.tmp" 2>/dev/null
  if [ ! -s "$WORK/n.tmp" ]; then
    echo "  [节点缺失] $f"; diffcnt=$((diffcnt+1)); continue
  fi
  if [ ! -f "$f" ]; then
    echo "  [本地缺失] $f"; diffcnt=$((diffcnt+1)); continue
  fi
  hn=$(tr -d '\r' < "$WORK/n.tmp" | sha256sum | cut -d' ' -f1)
  hl=$(tr -d '\r' < "$f" | sha256sum | cut -d' ' -f1)
  if [ "$hn" = "$hl" ]; then
    echo "  [一致] $f"; same=$((same+1))
  else
    echo "  ★[不同] $f"
    diff <(tr -d '\r' < "$WORK/n.tmp") <(tr -d '\r' < "$f") | head -12 | sed 's/^/       /'
    diffcnt=$((diffcnt+1))
  fi
done

echo
echo "=== scripts/diag/ 目录 ==="
ssh qkd 'cd /opt/qkd/graph_mappo && find scripts/diag -type f 2>/dev/null | sort' > "$WORK/nd.txt" 2>/dev/null
find scripts/diag -type f 2>/dev/null | sort > "$WORK/ld.txt"
dn=$(comm -23 "$WORK/nd.txt" "$WORK/ld.txt" | wc -l)   # 只在节点有
echo "  节点 $(wc -l < "$WORK/nd.txt") 个 / 本地 $(wc -l < "$WORK/ld.txt") 个"
if [ "$dn" -gt 0 ]; then
  echo "  ★ 只在节点有 $dn 个（这些是真·节点独有，删前要看清）："
  comm -23 "$WORK/nd.txt" "$WORK/ld.txt" | sed 's/^/      /'
  diffcnt=$((diffcnt+dn))
else
  echo "  节点没有本地没有的文件"
fi
# 共同文件的差异
dcnt=0
while read -r f; do
  [ -f "$f" ] || continue
  hn=$(ssh qkd "cat /opt/qkd/graph_mappo/$f 2>/dev/null" | tr -d '\r' | sha256sum | cut -d' ' -f1)
  hl=$(tr -d '\r' < "$f" | sha256sum | cut -d' ' -f1)
  [ "$hn" != "$hl" ] && { echo "  ★[不同] $f"; dcnt=$((dcnt+1)); }
done < <(comm -12 "$WORK/nd.txt" "$WORK/ld.txt")
echo "  共有文件里内容不同的: $dcnt 个"
diffcnt=$((diffcnt+dcnt))

echo
if [ "$diffcnt" -eq 0 ]; then
  echo "RESULT: 全部一致（$same 个逐个比对通过）→ 可以安全删除节点上这批未跟踪文件，让快照写回"
  echo "  （本地是权威副本；节点 HEAD 落后只是 sync 的历史遗留）"
else
  echo "RESULT: 发现 $diffcnt 处差异 → **不要删**，先查清节点上是谁改的"
fi
