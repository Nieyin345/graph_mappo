#!/usr/bin/env bash
# 剩余时间速查：节点什么时候到期决定"该开长跑还是该收摊"。
echo "=== 服务器时间 ==="
date -Is
echo "UTC: $(date -u -Is)"
echo
echo "=== 已运行时长 ==="
uptime -p 2>/dev/null || uptime
echo "boot: $(uptime -s 2>/dev/null)"
echo
echo "=== 本机 SSH 侧的到期信息（如果 sshd 有 banner 或 /etc/motd）==="
cat /etc/motd 2>/dev/null | head -20
echo
echo "=== 节点自身主机名 / 是否 CloudLab 节点 ==="
hostname
cat /etc/hostname 2>/dev/null
echo
echo "=== 最近修改的输出（判断训练是否还在推进）==="
ls -lt --time-style=+%H:%M:%S /opt/qkd/graph_mappo/outputs/ent01_g999_s42_r2/ 2>/dev/null | head -5
echo
echo "=== 磁盘 ==="
df -h /opt 2>/dev/null | tail -2
