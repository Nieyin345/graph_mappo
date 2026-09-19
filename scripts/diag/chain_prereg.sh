#!/usr/bin/env bash
# 预注册实验的执行器：补 ent01 的种子 45、46，用来判定
# 「ent01 是否真的超过专家」。判据见 docs/训练诊断记录.md 的预注册节。
#
# 为什么是**独立脚本**而不是往 chain_safe_gain.sh 里加一波：
#   chain 正以 `bash /tmp/chain_safe_gain.sh` 运行，而 **bash 是按字节偏移
#   增量读取脚本文件的** —— 运行中改它，轻则该波跳过，重则读到半个命令
#   执行出错。所以链条一律不动，补测另起一个脚本。
#
# 安全性：本脚本等 **chain_safe_gain.sh 进程退出**（不是等"0 个 run"）才启动。
#
#   为什么不判"0 个 train run"：chain 的 wave() 是「等自己这一波跑完 → 立刻起
#   下一波」，所以"0 个 run"只是两次 wave 之间的**毫秒级空档**。拿它当条件有两
#   种坏结局：要么永远等不到（超时退出，白等 8h），要么真的抢在空档里启动
#   —— 那就变成第 4 个并发 run，正好落进"4 勉强"的临界区。
#   等 chain 进程退出则是**无歧义、无竞态**的：它一退，就是真的没有链了。
#
#   为什么不赌第 4 个 run：OOM 的受害者**是 RSS 最大的那个**，也就是已经跑了一
#   半的成熟 run —— 那正是链里最有价值的在飞数据。为了多跑 2 个种子而冒着
#   毁掉 vcoef1/ep2 的风险，收益为负。**不猜余量，只等空场。**
set -u
cd /opt/qkd/graph_mappo || exit 1

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
THREADS=4
UPDATES=30
BASE="rl_algorithm.yaml train_full_rl.yaml"
MARK=/tmp/wave_prereg.go

# 用**锚定**模式：`pgrep -f chain_safe_gain` 会连 ssh 包装进程
# （`bash -c rm -f ... nohup bash /tmp/chain_safe_gain.sh ...`）一起匹配上，
# 而那个包装进程的行里同样含这个串。实测锚定后只匹配真 chain 一个 PID。
CHAIN_RE='^bash /tmp/chain_safe_gain\.sh$'

echo "[$(date -Is)] 等 chain_safe_gain.sh 跑完（它还有 ep2/mini512/runt 三波，约 6h）…"
for _ in $(seq 1 1440); do        # 最多等 12h
  if ! pgrep -f "$CHAIN_RE" > /dev/null 2>&1; then
    echo "[$(date -Is)] chain 已退出"
    break
  fi
  sleep 30
done

if pgrep -f "$CHAIN_RE" > /dev/null 2>&1; then
  echo "[$(date -Is)] ⚠ 等 chain 超时（12h），它还在跑 —— **不启动**（宁可漏跑，不冒险 OOM）"
  exit 1
fi

# chain 刚退出，它最后一波的 run 可能还在收尾（trainer 进程还没退干净）
echo "[$(date -Is)] 等最后一波 run 收尾…"
for _ in $(seq 1 120); do
  n=$(pgrep -cf "train_graph_mappo.py" 2>/dev/null || echo 0)
  [ "$n" -eq 0 ] && break
  sleep 15
done
n=$(pgrep -cf "train_graph_mappo.py" 2>/dev/null || echo 0)
echo "[$(date -Is)] 剩余 train run: $n"

if [ -f "$MARK" ]; then
  echo "[$(date -Is)] 已有标记 $MARK，跳过"
  exit 0
fi

# 空场保护 + 再确认一次内存确实空出来了
a=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
echo "[$(date -Is)] MemAvailable ${a}G（空场）"
if [ "$a" -lt 60 ]; then
  echo "[$(date -Is)] ⚠ 空场但只有 ${a}G < 60G —— 可能有孤儿 worker 占着，先 reap"
  /opt/qkd/venv/bin/python /tmp/reap.py --apply 2>&1 | tail -5 || true
  a=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  echo "[$(date -Is)] reap 后 MemAvailable ${a}G"
fi

for seed in 45 46; do
  name="ent01_s${seed}"
  if [ -d "outputs/$name" ] && [ -f "outputs/$name/metrics.jsonl" ]; then
    echo "  !! outputs/$name 已存在，跳过（outputs 只增不删，不覆盖）"
    continue
  fi
  # 照抄 chain_safe_gain.sh 的 launch()：--configs 每项按 <ROOT>/configs/<name>
  # 解析，所以必须是**裸文件名**；写成 configs/xxx.yaml 会拼成
  # configs/configs/xxx.yaml。训练脚本只在启动时才报这个错，所以这里先挡。
  bad=0
  for c in $BASE train_ent01.yaml; do
    case "$c" in
      configs/*|/*) echo "  !! --configs 要裸文件名，收到 '$c'"; bad=1;;
    esac
    [ -f "configs/$c" ] || { echo "  !! 缺 configs/$c"; bad=1; }
  done
  [ "$bad" -eq 1 ] && continue
  setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs $BASE train_ent01.yaml \
      --checkpoint "$CKPT" \
      --seed "$seed" --num-updates "$UPDATES" --run-name "$name" \
      > "/tmp/${name}.log" 2>&1 < /dev/null &
  echo "  [$(date -Is)] 启动 ${name} (seed $seed)"
  sleep 12
done

# 启动后自检：确认真的在跑，**并且 resolved_config 真的落在预注册的配置上**。
# 只看"进程在"是不够的——链式的 --configs 覆盖错了照样能跑起来，
# 只是跑的不是我们要的实验。所以要回过头读 resolved_config.yaml 核对。
sleep 45
echo "  --- 启动后自检 ---"
for seed in 45 46; do
  name="ent01_s${seed}"
  if ! pgrep -f "run-name ${name}" > /dev/null 2>&1; then
    echo "  !! ${name}: 进程不在 —— 看 /tmp/${name}.log"
    tail -5 "/tmp/${name}.log" 2>/dev/null | sed 's/^/      /'
    continue
  fi
  rc="outputs/${name}/resolved_config.yaml"
  if [ ! -f "$rc" ]; then
    echo "  ?? ${name}: 进程在，但还没写出 $rc"
    continue
  fi
  ec=$(grep -A40 "^train:" "$rc" | grep "entropy_coef" | head -1 | awk '{print $2}')
  gs=$(grep -A3 "^seed:" "$rc" | grep "global_seed" | head -1 | awk '{print $2}')
  nu=$(grep -m1 "num_updates:" "$rc" | awk '{print $2}')
  if [ "$ec" = "0.01" ] && [ "$gs" = "$seed" ] && [ "$nu" = "$UPDATES" ]; then
    echo "  ok ${name}: entropy_coef=$ec global_seed=$gs num_updates=$nu"
  else
    echo "  !! ${name}: 配置与预注册不符！entropy_coef=$ec (期望 0.01) " \
         "global_seed=$gs (期望 $seed) num_updates=$nu (期望 $UPDATES)"
    echo "      —— **这个 run 的结果不能用**，停掉它查 --configs 链"
  fi
done

touch "$MARK"
echo "[$(date -Is)] 预注册波已启动（ent01_s45/s46）。跑完用："
echo "  /opt/qkd/venv/bin/python /tmp/prereg_ent01_nseeds.py   # 复核判据"
echo "  再算合并 n=5 的平台均值与 p"
