#!/usr/bin/env bash
# 按路径比训练代码/配置，**并把行尾归一化**。
#
# 为什么必须归一化：本机是 Windows，工作区文件可能是 CRLF，而服务器是 LF。
# 直接 sha256 会把"CRLF vs LF"报成"内容不同"，于是看上去像节点跑着另一版代码
# ——正是本项目记过的"行尾符陷阱"（scripts/diag/README.md）。不归一化就会
# 把一个换行符的差异读成版本差异，进而错误地否决一次有效配对。
set -u
cd "$(dirname "$0")/.." || exit 1

PATHS="qkd_rl scripts/train configs"

norm_local() {
  find $PATHS -type f \( -name "*.py" -o -name "*.yaml" \) -not -path "*/archive/*" 2>/dev/null \
    | sort | while read -r f; do
        printf '%s %s\n' "$(tr -d '\r' < "$f" | sha256sum | cut -c1-24)" "$f"
      done
}

ssh qkd "cd /opt/qkd/graph_mappo && find $PATHS -type f \\( -name '*.py' -o -name '*.yaml' \\) -not -path '*/archive/*' | sort | while read -r f; do printf '%s %s\n' \"\$(tr -d '\\r' < \"\$f\" | sha256sum | cut -c1-24)\" \"\$f\"; done" > /tmp/nh.norm 2>/dev/null
norm_local > /tmp/lh.norm

echo "节点 $(wc -l < /tmp/nh.norm) 个，本地 $(wc -l < /tmp/lh.norm) 个"
echo
echo "=== 差异（< 节点独有/不同，> 本地独有/不同）==="
if diff /tmp/nh.norm /tmp/lh.norm > /tmp/code_diff.txt 2>&1; then
  echo "  ✅ 完全一致（行尾归一化后）—— 新臂与 ent01 同代码，配对成立"
else
  cat /tmp/code_diff.txt | head -40
fi
echo
echo "=== 只看会影响训练数值的文件 ==="
for f in qkd_rl/rl/algos/mappo_trainer.py qkd_rl/rl/algos/policy.py \
         qkd_rl/env/env.py qkd_rl/env/action_resolver.py qkd_rl/env/reward.py \
         configs/rl_algorithm.yaml configs/train_full_rl.yaml configs/train_ent01.yaml \
         configs/features.yaml configs/global.yaml configs/graph_mappo.yaml \
         configs/env_full.yaml configs/train_profiles.yaml; do
  a=$(grep -F " $f" /tmp/nh.norm | cut -d' ' -f1)
  b=$(grep -F " $f" /tmp/lh.norm | cut -d' ' -f1)
  if [ -z "$a" ]; then s="节点缺"
  elif [ -z "$b" ]; then s="本地缺"
  elif [ "$a" = "$b" ]; then s="一致"
  else s="★不同"; fi
  printf "  %-44s 节点 %-26s 本地 %-26s %s\n" "$f" "${a:-—}" "${b:-—}" "$s"
done
