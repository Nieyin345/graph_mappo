#!/usr/bin/env bash
# 起 v2（activation relu→gelu）三臂。**内存门 + 一次性放行（all-or-none）**。
#
# ★★ 2026-09-20 修一个已经造成实际后果的 bug：
#   本脚本原用 `$i == "run-name"` 取 run 名，而真实 argv 字段是 **`--run-name`**
#   （两个连字符）。字段**等值**比较⟹零命中；正则 `/run-name/` 是**子串**匹配
#   ⟹命中。同一个 token 两种匹配方式，错的那种**静默返回空**。
#   后果：待涨量恒算成 0 ⟹ 门**静默偏松** ⟹ 一次超发两条臂，
#   机器掉到 MemAvailable=17.08 GiB（正好在地板上）而后续还要涨 ~20 GB
#   ⟹ 差一步就 OOM 掉已投入 2 小时的 hist32 臂。
#   （同族记忆：variable-name-is-not-the-config-key ④ / gate-must-print-its-inputs）
#
# ★ 因此本版加**自证**：取到的 run 数必须与"看到几条 python 本体"对得上，
#   对不上就 exit 2（第三态），**不许当成"没有在跑的 run"**。
#
# ★ 另一处修正：**all-or-none**。原先逐条判内存、逐条起，结果起了 2 条卡住第 3 条
#   ⟹ 三条臂跨两个代码快照 ⟹ 它们彼此**不是同一个程序**（本会话刚查完这个混淆，
#   见 docs/训练诊断记录.md）。要么三条一起起，要么一条都不起。
#
# 铁律（逐条都是踩过的坑）：
#  1. 判据 = `MemAvailable − Σ在跑run的待涨量 − N×本run稳态 ≥ 地板`
#     （[[mem-gate-must-count-warmup-growth]]：`free` 只报当前读数）
#  2. **不用 pgrep -f**：`bash -c "… python …"` 包装器内嵌 python 字面量
#     ⟹ 会匹配到包装器 ⟹ 恒判"在跑"。用 `ps -eo comm,args` 且 comm 以 python 开头。
#  3. **不用 bc**：数值判据用 awk；取值**为空必须停下**（第三态）。
#  4. 启动用 `timeout` 包住 ssh（setsid 后 ssh 不自己返回），rc=124 是**预期**的。
#  5. 启动**失败必须吵**：起完立刻验证进程真在。
#  6. **打印判据的全部输入**：恒真的门不报错。
set -u

GELU_CFG=train_v2_gelu.yaml
BASE_CFGS="rl_algorithm.yaml train_ent01.yaml"
SEEDS="42 43 44"
RUN_PREFIX=v2_gelu
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
PSS_NEW=25.0       # 本配置（minibatch 256、无 hist）稳态 ~25 GiB
GROWTH=0.21        # GiB/轮，实测多点线性拟合均值
TARGET_U=30        # 在跑的 run 按跑到 u30 估它的上限
FLOOR_GIB=17.0     # 绝对余量地板（respawn-guard-two-views）

N_ARM=$(printf '%s\n' $SEEDS | wc -l)

echo "=== 起 v2(gelu) ${N_ARM} 臂（all-or-none）：$SEEDS ==="
echo "    配置: $BASE_CFGS $GELU_CFG"
echo "    对照: ent01_rerun_s{42,43,44}（已存在，**不重起**）"
echo

# --- 去重：问世界，不问账本 ---
for s in $SEEDS; do
  name="${RUN_PREFIX}_s${s}"
  if ssh -o ConnectTimeout=20 qkd \
       "ps -eo comm,args | awk '\$1 ~ /^python/ && /--run-name ${name}\$/' | grep -q ."; then
    echo "  [$name] 已在跑 ⟹ 全部跳过（all-or-none，不补起半批）"; exit 0
  fi
done

# --- 内存门：一次 ssh 取回全部输入 ---
GATE=$(ssh -o ConnectTimeout=20 qkd bash -s <<'REMOTE'
set -u
awk '/MemAvailable/{printf "AVAIL %.2f\n", $2/1048576}' /proc/meminfo
# 正对照：先数"看到几条 python 本体"，再按字段取值 —— 两者必须一致
ps -eo comm,args | awk '$1 ~ /^python/ && /-u scripts\/train\/train_graph_mappo/ && /--run-name/' > /tmp/_gate_ps.txt
printf 'NPY %d\n' "$(wc -l < /tmp/_gate_ps.txt)"
# ★ 字段名是 --run-name（两个连字符）；等值比较，不用子串正则
awk '{ for (i = 1; i <= NF; i++) if ($i == "--run-name") { print "RUN " $(i+1); break } }' /tmp/_gate_ps.txt | sort -u
rm -f /tmp/_gate_ps.txt
REMOTE
)

AV=$(printf '%s\n' "$GATE" | awk '/^AVAIL/{print $2}')
NPY=$(printf '%s\n' "$GATE" | awk '/^NPY/{print $2}')
NRUN=$(printf '%s\n' "$GATE" | awk '/^RUN/{n++} END{print n+0}')

# ★ 第三态：空值必须停下
printf '%s' "$AV" | grep -qE '^[0-9]+(\.[0-9]+)?$' \
  || { echo "  ✗ 判据不可用：MemAvailable 读出 '$AV' ⟹ 停下（不当'条件不成立'）"; exit 2; }
printf '%s' "$NPY" | grep -qE '^[0-9]+$' \
  || { echo "  ✗ 判据不可用：python 本体计数读出 '$NPY' ⟹ 停下"; exit 2; }

# ★★ 自证：取到的 run 名数 必须 == 看到的 python 本体数
#   （这正是 2026-09-20 那个 bug 的形态：本体 5 条、取到 0 个名字，而门照常放行）
if [ "$NRUN" != "$NPY" ]; then
  echo "  ✗✗ 取数自证失败：看到 $NPY 条 python 本体，却只取到 $NRUN 个 run 名。"
  echo "      这说明 run 名提取坏了（字段名/正则不对）⟹ 待涨量会算成 0 ⟹ 门静默偏松。"
  echo "      拒绝启动（第三态），先修提取逻辑。原始行数：$(printf '%s' "$GATE" | wc -l)"
  exit 2
fi
echo "  [自证] python 本体 $NPY 条，取到 run 名 $NRUN 个 ⟹ 提取逻辑有效"

# 每条在跑 run 的当前轮数 → 待涨量
GROW_TOTAL=0
echo "  [判据输入] MemAvailable=${AV} GiB，本 run 需 ${PSS_NEW} GiB × ${N_ARM} 臂，地板 ${FLOOR_GIB} GiB"
for rn in $(printf '%s\n' "$GATE" | awk '/^RUN/{print $2}'); do
  u=$(ssh -o ConnectTimeout=20 qkd "grep -c '\"update\"' /opt/qkd/graph_mappo/outputs/${rn}/metrics.jsonl 2>/dev/null || echo 0")
  printf '%s' "$u" | grep -qE '^[0-9]+$' && : || u=0
  g=$(awk -v u="$u" -v G="$GROWTH" -v T="$TARGET_U" 'BEGIN{ d=T-u; if (d<0) d=0; printf "%.2f", G*d }')
  GROW_TOTAL=$(awk -v a="$GROW_TOTAL" -v b="$g" 'BEGIN{printf "%.2f", a+b}')
  printf '             在跑 %-18s u=%-3s ⟹ 待涨 %s GiB\n' "$rn" "$u" "$g"
done
printf '             待涨合计 %s GiB\n' "$GROW_TOTAL"

NEED=$(awk -v n="$PSS_NEW" -v k="$N_ARM" 'BEGIN{printf "%.2f", n*k}')
OK=$(awk -v a="$AV" -v g="$GROW_TOTAL" -v n="$NEED" -v f="$FLOOR_GIB" \
      'BEGIN{ printf "%d", (a - g - n >= f) }')
printf '             判据: %s − %s − %s = %s  vs 地板 %s ⟹ %s\n' \
  "$AV" "$GROW_TOTAL" "$NEED" \
  "$(awk -v a="$AV" -v g="$GROW_TOTAL" -v n="$NEED" 'BEGIN{printf "%.2f", a-g-n}')" \
  "$FLOOR_GIB" "$([ "$OK" = 1 ] && echo 放行 || echo 拒绝)"
[ "$OK" = "1" ] || { echo "  ⏸ 内存不够（须 ${N_ARM} 臂一起放行）⟹ 一条都不起，退出"; exit 0; }

# --- 启动（timeout 是预期行为，见铁律 4）---
for s in $SEEDS; do
  name="${RUN_PREFIX}_s${s}"
  timeout 25 ssh -o ConnectTimeout=20 qkd \
    "cd /opt/qkd/graph_mappo && setsid nohup /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
       --configs $BASE_CFGS $GELU_CFG \
       --checkpoint $CKPT --seed $s --num-updates 30 --run-name $name \
       > outputs/${name}.log 2>&1 < /dev/null &" || true
done

# --- ★ 启动后验证：失败必须吵 ---
sleep 25
FAIL=0
for s in $SEEDS; do
  name="${RUN_PREFIX}_s${s}"
  LIVE=$(ssh -o ConnectTimeout=20 qkd \
    "ps -eo comm,args | awk '\$1 ~ /^python/ && /--run-name ${name}\$/' | wc -l")
  if [ "$LIVE" -ge 1 ]; then
    echo "     ✓ $name 已起（python 本体 $LIVE 个）"
  else
    echo "     ✗✗ $name **启动失败**（无 python 本体）—— 最后 15 行日志："
    ssh -o ConnectTimeout=20 qkd "tail -15 /opt/qkd/graph_mappo/outputs/${name}.log 2>/dev/null"
    FAIL=1
  fi
done
[ "$FAIL" = 0 ] || { echo "  ⟹ 有臂没起来（[[failed-launch-must-be-loud]]）"; exit 1; }
echo "  ⟹ ${N_ARM} 臂全部在跑 ⟹ all-or-none 成立"
