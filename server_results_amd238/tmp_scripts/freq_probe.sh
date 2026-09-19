#!/usr/bin/env bash
# 采样 CPU 频率与负载，验证 update_s 在进程内爬升是不是降频（功耗/温度）造成的。
# 用法：nohup bash .tmp/freq_probe.sh 120 > /tmp/freq_probe.log 2>&1 &
#   $1 = 采样次数（每次间隔 5 秒）
N=${1:-120}
echo "time avg_MHz min_MHz max_MHz load1"
for _ in $(seq 1 "$N"); do
    awk -F: '
        /cpu MHz/ {
            v = $2 + 0
            s += v; n++
            if (n == 1 || v < lo) lo = v
            if (n == 1 || v > hi) hi = v
        }
        END { printf "%s %.0f %.0f %.0f %s\n", strftime("%H:%M:%S"), s/n, lo, hi, "?" }
    ' /proc/cpuinfo | sed "s/ ?$/ $(cut -d' ' -f1 /proc/loadavg)/"
    sleep 5
done
