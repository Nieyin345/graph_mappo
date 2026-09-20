#!/usr/bin/env bash
# 远端状态查询（在服务器上跑）。把嵌套引号全关在这一个文件里，
# 本地轮询器只做 `ssh qkd 'bash .../pm_status.sh'`，不再套引号。
#
# 输出（逐行）：
#   ARM=<名字> N=<update 行数> EV=<验证点数> END=<GOING|FIN|DEAD> AGE=<秒> LOG=<字节>
#   PROC=<真 python 训练进程数>   ← 用 comm 锚定，不用 pgrep -f
#   MEM=<MemAvailable GiB>
#   TRACE=<臂名>                  ← 有 traceback 才打
#
# ★★ 为什么要三态 END 而不是「N 够不够」：
#   N 是**已完成**的轮数；u 与 u 之间 N 不涨，那是正常在跑。
#   只看 N 会把「正在跑第 k 轮」误判成死了（过保守），
#   而「单条臂崩了但其它还在跑」时 PROC>0 ⟹ 又测不出（恒真）。
#
# ★★ 判据锚在**地面真值**上，不锚在日志文本上（我第一版照猜写了
#   `训练完成|Finished|done.`，实测日志里**根本没有**这些串）：
#     FIN  = `outputs/<臂>/checkpoint_final.pt` 存在
#            —— 它只在 main() 里 trainer.train() 返回后写一次，
#               是「跑完」的**唯一**权威标记（已在 v2_bottleneck_s42 上标定）
#     DEAD = 日志有 Traceback（崩了），或日志陈旧（冻死/被杀）
#     GOING= 日志还在涨
#   实测 round ≈ rollout 68s + update 78s ≈ 2.4 min，所以「陈旧」阈值
#   取 15 min（6 倍余量）—— 判据要**打印 AGE**，不能只打判决
#   （`gate-must-print-its-inputs`）。
#
# ★ AGE 抓的是**静默冻结**：本项目实测过 38 线程全 futex_wait、CPU 0 tick、
#   metric 不再写、近 2 小时才发现（`training-runs-can-deadlock-frozen`）。
#   那种情形 N 不涨、进程还在、也没 traceback ⟹ 只有新鲜度看得见。
#
# ★ 造反证（`LABEL_*` 不走主循环，直接考 classify —— **同一份实现**，
#   不另写第二份，见 `duplicate-implementation-drifts`）：
#     LABEL_GOING="<log>|<outdir>" LABEL_FIN=... LABEL_DEAD=... bash pm_status.sh
#   另附**正对照**：不存在的路径必须判 DEAD，不能返回空串（空串会被
#   本地轮询器当成「网络抖动」而继续等 —— `unavailable-must-not-look-like-a-value`）。
set -u
OUT=/opt/qkd/graph_mappo/outputs
ARMS="pm_decode_s42 pm_decode_s43 pm_decode_s44 pm_decode_s45 pm_decode_s46"
STALE=900          # 秒；round 实测 ≈ 146s，6 倍余量

# 取两处 mtime 里**较新**的那个：日志有 -u（无缓冲），metric 每轮也写，
# 取 max 可以消掉「某一侧缓冲/延迟」造成的一整类假警。
newest_mtime() {   # $1=log  $2=outdir
    local m=0 t
    for f in "$1" "$2/metrics.jsonl"; do
        [ -f "$f" ] || continue
        t=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        [ "$t" -gt "$m" ] && m=$t
    done
    echo "$m"
}

classify() {   # $1=log  $2=outdir  → 打印 GOING|FIN|DEAD
    local log="$1" out="$2"
    # 1. 跑完 = 地面真值（文件存在），不是日志文本
    if [ -f "$out/checkpoint_final.pt" ]; then echo "FIN"; return; fi
    # 2. 崩了
    if [ -f "$log" ] && grep -q "Traceback" "$log"; then echo "DEAD"; return; fi
    # 3. 日志还没写 ⟹ 起不来（空文件也是死，不是「还在跑」）
    if [ ! -s "$log" ]; then echo "DEAD"; return; fi
    # 4. 新鲜度：轮与轮之间 N 不涨，但文件一直在涨
    local mt now age
    mt=$(newest_mtime "$log" "$out"); now=$(date +%s); age=$((now - mt))
    if [ "$age" -le "$STALE" ]; then echo "GOING"; else echo "DEAD"; fi
}

age_of() {         # $1=log  $2=outdir  → 秒，打印用
    local mt; mt=$(newest_mtime "$1" "$2")
    [ "$mt" -eq 0 ] && { echo "-1"; return; }
    echo $(( $(date +%s) - mt ))
}

# ---- 造反证入口：给了 LABEL_* 就只考判据，不报状态 ----
if [ -n "${LABEL_GOING:-}${LABEL_FIN:-}${LABEL_DEAD:-}" ]; then
    rc=0
    for pair in "GOING:${LABEL_GOING:-}" "FIN:${LABEL_FIN:-}" "DEAD:${LABEL_DEAD:-}"; do
        want="${pair%%:*}"; spec="${pair#*:}"
        [ -z "$spec" ] && continue
        lg="${spec%%|*}"; od="${spec#*|}"
        got=$(classify "$lg" "$od")
        if [ "$got" = "$want" ]; then
            echo "LABEL_OK   want=$want got=$got  $spec"
        else
            echo "LABEL_FAIL want=$want got=$got  $spec"
            rc=2
        fi
    done
    # ★ 正对照 A：路径不存在 ⟹ 必须是 DEAD，不能是空串
    gone=$(classify "/nonexistent/no_such_log_$$.log" "/nonexistent/no_such_dir_$$")
    if [ "$gone" = "DEAD" ]; then echo "LABEL_OK   want=DEAD got=DEAD  (不存在的路径)";
    else echo "LABEL_FAIL want=DEAD got='$gone' (不存在的路径)"; rc=2; fi
    # ★ 正对照 B：**陈旧但无 traceback** ⟹ 必须判 DEAD（这正是冻结的样子）
    #   用真臂的日志复制一份、把 mtime 推回 1 小时，验证新鲜度这一支真的在工作。
    tmpd=$(mktemp -d); cp /tmp/pmlogs/pm_decode_s42.log "$tmpd/old.log" 2>/dev/null \
        && touch -d '1 hour ago' "$tmpd/old.log"
    if [ -f "$tmpd/old.log" ]; then
        stale=$(classify "$tmpd/old.log" "$tmpd")
        if [ "$stale" = "DEAD" ]; then echo "LABEL_OK   want=DEAD got=DEAD  (陈旧无 traceback)";
        else echo "LABEL_FAIL want=DEAD got=$stale (陈旧无 traceback)"; rc=2; fi
        # 且同一份日志在**新鲜**时必须判 GOING —— 否则第 4 支恒死
        touch "$tmpd/old.log"
        fresh=$(classify "$tmpd/old.log" "$tmpd")
        if [ "$fresh" = "GOING" ]; then echo "LABEL_OK   want=GOING got=GOING (同日志设为新鲜)";
        else echo "LABEL_FAIL want=GOING got=$fresh (同日志设为新鲜)"; rc=2; fi
    fi
    rm -rf "$tmpd"
    exit "$rc"
fi

for a in $ARMS; do
    d="$OUT/$a"; f="$d/metrics.jsonl"
    n=0; ev=0
    if [ -f "$f" ]; then
        n=$(grep -c '"update"' "$f")
        ev=$(grep -c '"eval_validation"' "$f")
    fi
    log="/tmp/pmlogs/$a.log"
    lb=0; [ -f "$log" ] && lb=$(wc -c < "$log")
    echo "ARM=$a N=$n EV=$ev END=$(classify "$log" "$d") AGE=$(age_of "$log" "$d") LOG=$lb"
done
echo "PROC=$(ps -eo comm,args | awk '$1 ~ /^python/ && /train_graph_mappo/' | wc -l)"
awk '/MemAvailable/ {printf "MEM=%.1f\n", $2/1048576}' /proc/meminfo
for a in $ARMS; do
    log="/tmp/pmlogs/$a.log"
    if [ -f "$log" ] && grep -q "Traceback" "$log"; then echo "TRACE=$a"; fi
done
