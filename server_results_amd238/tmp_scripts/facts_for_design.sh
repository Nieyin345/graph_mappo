#!/usr/bin/env bash
# 事实核对：设计下一轮实验前必须确认的四件事。
#   A. ent01 三条臂（当前唯一的"修复后"三种子基线）到底用什么线程数/配置跑的
#   B. 它们的 u30 值（作为新臂的对照）
#   C. 生效配置里 value_coef / epochs / minibatch_size 各是多少
#   D. r4/r5 那两浪的真实配置筛选是不是**先于 d4bd79c 播种修复**（若是，它们
#      每臂只是"一次不受控的随机抽样"，按 §4⑦ 不能当结论）
cd /opt/qkd/graph_mappo || exit 1

echo "=== A. ent01 臂的启动命令（线程数从这里看）==="
for s in 42 43 44; do
  f=/tmp/ent01_s${s}.log
  if [ -f "$f" ]; then
    echo "  --- ent01_s${s} 头部 ---"
    head -6 "$f" | sed 's/^/    /' | cut -c1-150
  fi
done
echo
echo "  进程环境里有没有 OMP（若有 run 在跑）："
for pid in $(pgrep -f train_graph_mappo 2>/dev/null); do
  tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -E '^OMP_NUM_THREADS|^MKL_NUM_THREADS' | sed 's/^/    /'
done

echo
echo "=== A2. 启动脚本怎么写的（复现方法本身）==="
for f in /tmp/chain_ent01.sh /tmp/screen_r8_multiseed.sh; do
  [ -f "$f" ] && { echo "  --- $f ---"; grep -n "OMP\|num-updates\|seed\|run-name" "$f" | head -12 | sed 's/^/    /'; }
done

echo
echo "=== B. ent01 三条臂的 u30 验证值（对照基线）==="
for s in 42 43 44; do
  d=outputs/ent01_s${s}
  [ -d "$d" ] || { echo "  ent01_s${s}: 目录不存在"; continue; }
  # metrics.jsonl 里 eval_validation 是独立一行，靠前一条 update 行定位
  /opt/qkd/venv/bin/python - "$d" <<'PY' 2>/dev/null || echo "    (python 解析失败)"
import json,sys
d=sys.argv[1]; last=None; out=[]
for line in open(d+"/metrics.jsonl",encoding="utf-8"):
    line=line.strip()
    if not line: continue
    try: r=json.loads(line)
    except Exception: continue
    if "update" in r: last=r["update"]
    ev=r.get("eval_validation")
    if isinstance(ev,dict) and ev.get("mean_success_rate") is not None:
        out.append((last,float(ev["mean_success_rate"])))
print("    " + " ".join(f"u{u}={v:.4f}" for u,v in out))
PY
done

echo
echo "=== C. ent01_s42 的生效配置（关键字段）==="
if [ -f outputs/ent01_s42/resolved_config.yaml ]; then
  grep -nE "value_coef|epochs|minibatch_size|entropy_coef|clip_eps|target_kl|actor_lr|critic_lr|n_rollout_workers|rollout_batch_envs|gamma" \
    outputs/ent01_s42/resolved_config.yaml | sed 's/^/    /'
else
  echo "  缺 resolved_config.yaml"
fi

echo
echo "=== D. r4/r5 的 run 目录与时间（对照 d4bd79c = 2026-09-18）==="
for pat in 'r4_*' 'r5_*' 'impfix*' 'vcoef*' 'mini512*'; do
  for d in outputs/$pat; do
    [ -d "$d" ] || continue
    printf "  %-34s %s\n" "$(basename "$d")" "$(stat -c %y "$d/metrics.jsonl" 2>/dev/null | cut -c1-19)"
  done
done 2>/dev/null | sort -k2 | head -30

echo
echo "=== E. 现有 outputs 里所有 ent01 / r8 家族（确认基线只此一套）==="
ls -d outputs/ent01_s4* outputs/r8* 2>/dev/null | sed 's/^/  /'
