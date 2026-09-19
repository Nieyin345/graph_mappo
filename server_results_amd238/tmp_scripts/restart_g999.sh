#!/usr/bin/env bash
# 重启被看门狗误杀的 ent01_g999 臂。
#
# 事故经过（我自己造成的）：
#   看门狗阈值 --threshold 20 是在**错误前提**下设的 —— 当时我记的是"每 run 13 GB"，
#   实测每 run **23.3 GB**（父进程 14.5 GB + 8 个 worker × 1.1 GB）。
#   于是 3 个 ent01_s4x（69.9 GB）+ 2 个 g999（46.6 GB）= 116.5 GB，
#   在 125 GB 机器上必然顶到 20 GB 阈值，看门狗就按"杀最晚启动的"杀了
#   ent01_g999_s43（13:04:51）和 ent01_g999_s42（13:20:55）。
#   丢的是 **gamma×entropy 交互**这一臂，而它正是预先设计的归因对照
#   （ent01_g999 vs ent01 同种子 = gamma 净效应）。
#
# 内存纪律（这次算清楚再起）：
#   125 GB 机器 - 已用（3 个 ent01 ≈ 70 GB）= 约 50 GB 可用。
#   再加 2 个 g999（46.6 GB）只剩 3.7 GB —— 又是同一个坑。
#   **只加 1 个**（23.3 GB），留 ~27 GB 余量；等 ent01 跑完再补第二个。
#
# 注意：绝不使用 `pkill -f <模式>` —— 远程 bash -c 的命令行自身含该字符串，
# pkill 会把自己一起杀掉（实测 exit 255，脚本再也不会执行）。
set -u
cd /opt/qkd/graph_mappo

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
for c in configs/train_ent01.yaml configs/train_ent01_g999.yaml; do
    [ -f "$c" ] || { echo "缺配置文件 $c"; exit 1; }
done

avail_gb=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
n_train=$(pgrep -c -f train_graph_mappo.py || echo 0)
echo "启动前: MemAvailable=${avail_gb} GB, 存活训练=${n_train} 个"

# 每个 run 实测 23.3 GB；留 8 GB 安全余量
need=23.3
if awk -v a="$avail_gb" -v n="$need" 'BEGIN{exit !(a < n + 8)}'; then
    echo "余量不足（需 ${need} GB + 8 GB 安全余量），本次不启动。"
    exit 0
fi

seed="${1:-42}"
name="ent01_g999_s${seed}_r2"
if [ -d "outputs/$name" ]; then
    echo "outputs/$name 已存在，跳过（避免覆盖）"
    exit 0
fi

setsid nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
    --configs rl_algorithm.yaml train_full_rl.yaml \
               train_ent01.yaml train_ent01_g999.yaml \
    --checkpoint "$CKPT" \
    --seed "$seed" --num-updates 30 --run-name "$name" \
    > "/tmp/${name}.log" 2>&1 < /dev/null &

echo "  起 $name (pid $!)"
sleep 25
echo
echo "=== 存活训练 ==="
ps -eo etime,args | grep "[t]rain_graph_mappo" | grep -o "run-name [^ ]*" | sort
echo
avail2=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
echo "启动后: MemAvailable=${avail2} GB"
tail -3 "/tmp/${name}.log"
