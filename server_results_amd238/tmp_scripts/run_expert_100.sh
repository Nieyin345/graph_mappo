#!/usr/bin/env bash
# 补测：专家在 RL 实际使用的验证种子上（100-114）的成功率。
#
# 为什么必须补：项目里所有"RL 超过专家"的结论，都是拿 RL 的 eval_validation
# （种子 100-114）去比专家文件（种子 7-21）—— 两批种子完全不相交。
# 按 docs/测试规范.md §4，配对不是可选项；种子集不同就是不可比。
# 本脚本给出唯一同口径的专家数字。
set -euo pipefail
cd /opt/qkd/graph_mappo

OUT=outputs/eval/expert_seeds100_240.json
if [ -f "$OUT" ]; then
    echo "已存在 $OUT，跳过"
    exit 0
fi

echo "开始：专家 @ 种子 100-114（留出窗口 330-365，240 步）"
date -u
/opt/qkd/venv/bin/python scripts/eval/eval_expert.py \
    --seeds 100-114 \
    --out "$OUT"
echo "完成"
date -u
