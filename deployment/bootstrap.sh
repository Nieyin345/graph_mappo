#!/usr/bin/env bash
# One command: a bare CloudLab/Emulab node -> ready to train.
#
#   bash deployment/bootstrap.sh qinglong@clnode101.clemson.cloudlab.us
#   bash deployment/bootstrap.sh <user@host> --skip-data    # image already has the dataset
#   bash deployment/bootstrap.sh <user@host> --no-smoke     # skip the timing self-check
#
# Runs from the LOCAL machine (the repo root). Idempotent: every step checks
# first and skips what is already in place, so a re-run after a partial failure
# costs only the missing steps.
#
# Why this exists on top of the three scripts it calls: each is usable alone,
# but a fresh node needs all of them in the right order, and the seams are where
# the time goes -- provisioning without the sync channel leaves no code to run;
# uploading data before the venv exists spends a 374 MB transfer on a node that
# cannot read it.
#
#   deployment/setup.sh   runs ON the node: python3-venv, venv at /opt/qkd,
#                             torch wheel matched to the hardware, dependencies
#   deployment/sync.sh    git push of the working-tree snapshot
#   deployment/upload_data.sh  dataset/ + BC checkpoint + rate_stats.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="${1:?usage: deployment/bootstrap.sh <user@host> [--skip-data] [--no-smoke]}"
shift || true
SKIP_DATA=0
RUN_SMOKE=1
for arg in "$@"; do
    case "$arg" in
        --skip-data) SKIP_DATA=1 ;;
        --no-smoke)  RUN_SMOKE=0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

REMOTE_DIR="${QKD_REMOTE_DIR:-/opt/qkd/graph_mappo}"
VENV_PY="/opt/qkd/venv/bin/python"

# deployment/sync.sh resolves its target through ~/.ssh/config by default, but a raw
# user@host works there too (both plain ssh and git's scp-like remote syntax
# accept it), so no ssh config entry is needed for a one-off node.
export QKD_HOST_ALIAS="$TARGET"

# 把目标记在本地，供之后单独运行 deployment/sync.sh / 探针脚本时回读 —— 否则
# 那次调用会回落到 ssh 别名的默认值 "qkd"，在这台工作站上并不存在。
# .tmp/ 已在 .gitignore 里，且同步通道只带 .tmp/*.py 和 .tmp/*.sh，所以这个
# 文件不会跑到节点上去。
mkdir -p .tmp
printf '%s\n' "$TARGET" > .tmp/qkd_host

# Same Windows/agent trap the other scripts document: the shell that runs this
# may be WSL (Windows drives at /mnt/c) or Git Bash (at /c), and in both cases
# the shell's own ssh -- WSL's or conda's MSYS one -- cannot reach the Windows
# ssh-agent service. Only the Windows OpenSSH binary can, so prefer it and let
# the -x test pick whichever path form this shell actually understands.
pick_ssh() {
    for cand in "C:/Windows/System32/OpenSSH/ssh.exe" \
                "/mnt/c/Windows/System32/OpenSSH/ssh.exe" \
                "/c/Windows/System32/OpenSSH/ssh.exe" \
                "$(command -v ssh 2>/dev/null || true)"; do
        if [ -x "$cand" ]; then
            echo "$cand"
            return 0
        fi
    done
    echo "no usable ssh found" >&2
    return 1
}
pick_scp() {
    case "$1" in
        *ssh.exe) echo "${1%ssh.exe}scp.exe" ;;
        *)        command -v scp ;;
    esac
}
SSH_BIN="$(pick_ssh)"
SCP_BIN="$(pick_scp "$SSH_BIN")"
[ -x "$SCP_BIN" ] || { echo "no usable scp found (looked next to $SSH_BIN)" >&2; exit 1; }

SSH() { "$SSH_BIN" -o BatchMode=yes "$@"; }
step() { echo; echo "======== $* ========"; }

# --- 1. preflight ----------------------------------------------------------
step "1/6  连通性与硬件"
# First contact needs `accept-new`: the platform hands out a NEW hostname every
# time the experiment restarts, so the key is never in known_hosts yet -- and
# every later step (git push, scp) would fail on it, not just this one. It still
# refuses a CHANGED key for a host already on record, which is the protection
# that actually matters. Once this succeeds, known_hosts is populated and the
# sub-scripts' plain `ssh`/`git push` work normally.
SSH -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "$TARGET" true 2>/dev/null || {
    cat >&2 <<'MSG'
连不上目标节点。检查：
  * ssh-agent 里有没有钥匙 —— 普通 PowerShell 里跑 ssh-add -l，应列出 ED25519 密钥；
  * 平台门户上这台节点是不是还在运行（换机器会换主机名）。
MSG
    exit 1
}
SSH "$TARGET" 'bash -s' <<'REMOTE'
echo "  host   $(hostname)"
echo "  cores  $(nproc) threads, $(lscpu | awk -F: '/^Socket\(s\)/{s=$2} /^Core\(s\) per socket/{c=$2} END{gsub(/ /,"",s); gsub(/ /,"",c); print s*c}') physical"
free -g | awk '/^Mem:/{print "  mem    " $2 " GB"}'
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  gpu    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "  gpu    none"
fi
grep -iE '^model name' /proc/cpuinfo | head -1 | sed 's/^/  cpu    /'
REMOTE

# --- 2. provision ----------------------------------------------------------
step "2/6  环境（venv + 依赖）"
if SSH "$TARGET" "[ -x $VENV_PY ] && $VENV_PY -c 'import torch, h5py, yaml, scipy, networkx' 2>/dev/null"; then
    echo "  /opt/qkd/venv 已可用，跳过。"
else
    echo "  传输并运行 deployment/setup.sh —— torch 轮子较大，约 3-6 分钟。"
    # scp 不会建中间目录，而全新节点上没有 ~/deployment/，直接 scp 会报
    # "No such file or directory"（2026-09-19 在 clnode316 上实测踩到）。
    # 脚本其余两处 scp 目标都自带 mkdir（upload_data.sh 有 sudo -n mkdir -p、
    # sync.sh 有 mkdir -p "$REMOTE_DIR"），只有这一处漏了。
    SSH "$TARGET" 'mkdir -p ~/deployment'
    "$SCP_BIN" -q deployment/setup.sh "$TARGET:~/deployment/setup.sh"
    SSH "$TARGET" 'bash ~/deployment/setup.sh'
fi

# --- 3. sync channel -------------------------------------------------------
step "3/6  代码同步通道"
# Probe for a usable HEAD, not just a .git directory -- a setup that died
# partway leaves .git present with no commit, and `--setup` is idempotent so
# running it again is the correct repair.
if SSH "$TARGET" "cd $REMOTE_DIR 2>/dev/null && git rev-parse --verify HEAD >/dev/null 2>&1"; then
    echo "  受管仓库已就绪，跳过 --setup。"
else
    bash deployment/sync.sh --setup
fi
bash deployment/sync.sh

# --- 4. data ---------------------------------------------------------------
step "4/6  数据与权重"
if [ "$SKIP_DATA" -eq 1 ]; then
    echo "  --skip-data：假定镜像里已有 dataset/ 与 BC 权重。"
else
    # --data-only, not the default: the default also ships a code tarball, and
    # extracting it over the remote tree clobbers the checkout step 3 just made
    # and leaves CRLF endings behind, which dirties the repo and gets the NEXT
    # push rejected. Code arrives through the git channel, which normalises to
    # LF; this step should only ever move data.
    bash deployment/upload_data.sh "$TARGET" --data-only
fi

# --- 5. verify -------------------------------------------------------------
# Every path below is load-bearing. The dangerous one is rate_stats.json: when
# it is missing RateNormalizer degrades to a constant p99 = 10.0 without saying
# anything, and the 11 rate columns of the edge features come out the wrong
# magnitude -- every number produced afterwards is incomparable to the record.
step "5/6  校验"
SSH "$TARGET" 'bash -s' <<REMOTE
cd "$REMOTE_DIR" || exit 1
missing=0
for f in \
    dataset/global/link_data.h5 \
    dataset/global/rate_stats.json \
    dataset/global/node_registry.csv \
    dataset/global/link_registry.csv \
    outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
do
    if [ -s "\$f" ]; then printf '  OK    %s\n' "\$f"
    else printf '  MISS  %s\n' "\$f"; missing=1; fi
done
if [ ! -s dataset/global/rate_stats.json ]; then
    cat >&2 <<'MSG'

  rate_stats.json 缺失 -> RateNormalizer 会静默退化成常量 p99 = 10.0，
  边特征里 11 列速率特征量级全错，结果与之前全部不可比。修：
      /opt/qkd/venv/bin/python scripts/estimate_rate_stats.py
MSG
fi
exit \$missing
REMOTE

# --- 6. smoke --------------------------------------------------------------
if [ "$RUN_SMOKE" -eq 1 ]; then
    step "6/6  自检（3 轮；看的是耗时，不是成功率）"
    SSH "$TARGET" "cd $REMOTE_DIR && OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
        $VENV_PY -u scripts/train/train_graph_mappo.py \
            --configs rl_algorithm.yaml train_diag_fast.yaml \
            --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
            --run-name bootstrap_smoke --num-updates 3 2>&1 | tail -5"
    cat <<'MSG'

  对照 docs/服务器连接与使用.md §7：
    update_s  应明显低于 45 s（amd276 的 48 线程上是 21 s）
    nb        应约等于 rollout_steps × episodes_per_update ÷ minibatch_size（diag 配置是 8）
  update_s 偏高通常是核少，或者忘了设 OMP_NUM_THREADS 导致进程互相抢核。
MSG
else
    step "6/6  自检已跳过（--no-smoke）"
fi

step "完成"
cat <<'NOTE'
节点已就绪。跑对比实验前先看两件事：

1) 那个疑似回归已经查清了 —— 是单次运行 + 单个训练种子的测量波动，不是代码问题。
   两版代码在 4 个训练种子上的配对平均差是 -0.0025 ± 0.015，符号还随种子翻转。
   过程见 docs/回归定位报告.md。

2) 但测量方法要改：这个协议分辨不了小于约 0.03 的差异（不配对时约 0.08）。
   测奖励/算法改动时至少跑 3 个训练种子、比分布不比单点：
       bash .tmp/run_seed_spread.sh          # 新代码，约 3 分钟/种子
   见 docs/测试规范.md §4 第 ⑦ 条。
NOTE
