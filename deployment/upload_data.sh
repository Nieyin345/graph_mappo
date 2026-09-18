#!/usr/bin/env bash
# Push the qkd_rl code (and optionally the data) to a CloudLab/Emulab node.
#
# Everything lives under /opt/qkd, NOT $HOME: Emulab strips the home directory
# out of disk images, so anything in ~ is lost when the node is imaged and
# would not survive into new nodes booted from that image.
#
# Usage (run from the repo root on the LOCAL machine):
#   bash deployment/upload_data.sh <user@host> [--full] [--code-only] [--data-only]
#
#   --code-only  ship the code tarball and nothing else. Use this once the node
#                was booted from a disk image that already contains the venv,
#                the dataset and the BC checkpoint -- it turns a ~7 minute
#                upload into a few seconds.
#   --data-only  ship the data and nothing else. Use this when the code got to
#                the node through deployment/sync.sh's git channel -- see the note
#                below.
#   --full       also ship outputs/trajs_pg_phased (4.3 GB), needed ONLY to
#                re-run behaviour-cloning. Skip it unless you are retraining
#                the BC warm start.
#
# WHY --data-only EXISTS: the code tarball is extracted straight over the
# remote working tree, so it clobbers whatever deployment/sync.sh checked out --
# and it arrives with CRLF line endings, which makes every file look modified
# to git. The push then gets silently rejected (receive.denyCurrentBranch=
# updateInstead wants a clean tree), and the node ends up running files whose
# line endings differ from the ones the snapshot recorded. deployment/sync.sh's own
# header documents this same trap from an earlier round. On any node that has
# the git channel, ship the code with deployment/sync.sh and call this script with
# --data-only.
#
# Requires: an ssh agent holding the key, and `ssh`/`scp` reachable as the
# Windows OpenSSH binaries when run under Git Bash on Windows.
set -euo pipefail

TARGET="${1:?usage: deployment/upload_data.sh <user@host> [--full] [--code-only] [--data-only]}"
shift || true
FULL=0
CODE_ONLY=0
DATA_ONLY=0
for arg in "$@"; do
    [ "${arg}" = "--full" ] && FULL=1
    [ "${arg}" = "--code-only" ] && CODE_ONLY=1
    [ "${arg}" = "--data-only" ] && DATA_ONLY=1
done
if [ "${CODE_ONLY}" = "1" ] && [ "${DATA_ONLY}" = "1" ]; then
    echo "--code-only 和 --data-only 不能同时用" >&2
    exit 2
fi

INSTALL_ROOT="/opt/qkd"
REMOTE_DIR="${INSTALL_ROOT}/graph_mappo"

# Running under WSL the Windows drives are at /mnt/c, under Git Bash at /c; the
# shell's own ssh in either case (WSL's, or conda's MSYS one) cannot talk to the
# Windows ssh-agent, so prefer the system OpenSSH binaries -- whichever path
# form this shell understands.
if [ -x "C:/Windows/System32/OpenSSH/ssh.exe" ]; then
    OPENSSH_DIR="C:/Windows/System32/OpenSSH"
elif [ -x "/mnt/c/Windows/System32/OpenSSH/ssh.exe" ]; then
    OPENSSH_DIR="/mnt/c/Windows/System32/OpenSSH"
elif [ -x "/c/Windows/System32/OpenSSH/ssh.exe" ]; then
    OPENSSH_DIR="/c/Windows/System32/OpenSSH"
else
    OPENSSH_DIR=""
fi
if [ -n "$OPENSSH_DIR" ]; then
    SSH="${OPENSSH_DIR}/ssh.exe"
    SCP="${OPENSSH_DIR}/scp.exe"
else
    SSH="ssh"
    SCP="scp"
fi

echo "== target: ${TARGET} (remote dir ${REMOTE_DIR}) =="

# --- 1. code ---------------------------------------------------------------
# Ship the working tree, not HEAD: the uncommitted changes are the fixes that
# make the run worth doing at all.
#
# --data-only skips this block: the tarball is extracted straight over the
# remote working tree, so it clobbers what deployment/sync.sh checked out, and it
# lands with CRLF endings that make every file look modified -- which gets the
# next push rejected. See the header for the full story.
if [ "${DATA_ONLY}" = "1" ]; then
    echo "== --data-only: 跳过代码打包（代码走 deployment/sync.sh 的 git 通道）=="
else
    echo "== packaging code =="
    CODE_TARBALL="$(mktemp -t qkd_code_XXXX).tar.gz"
    tar -czf "${CODE_TARBALL}" \
        --exclude=dataset --exclude=outputs --exclude=.git --exclude=.tmp \
        --exclude=__pycache__ --exclude='*.pyc' --exclude=.pytest_cache \
        --exclude=ref_work --exclude=weather \
        -C . .
    echo "   $(du -h "${CODE_TARBALL}" | cut -f1)"

    "${SCP}" -q "${CODE_TARBALL}" "${TARGET}:/tmp/qkd_code.tar.gz"
    "${SSH}" -o BatchMode=yes "${TARGET}" \
        "tar -xzf /tmp/qkd_code.tar.gz -C ${REMOTE_DIR} && rm /tmp/qkd_code.tar.gz"
    rm -f "${CODE_TARBALL}"
fi

# The install root is created either way -- the data steps below write into it,
# and on a fresh node this sudo mkdir is what makes /opt/qkd exist at all.
"${SSH}" -o BatchMode=yes "${TARGET}" \
    "sudo -n mkdir -p ${REMOTE_DIR}/dataset/global ${REMOTE_DIR}/outputs/supervised_pg_phased && sudo -n chown -R \$(id -u):\$(id -g) ${INSTALL_ROOT}"

# --- 2. data ---------------------------------------------------------------
if [ "${CODE_ONLY}" = "1" ]; then
    # The node was booted from an image that already carries the dataset, the
    # BC checkpoint and rate_stats.json. Verify that here rather than assuming
    # it: a silent fallback to the p99 = 10.0 default would make every run on
    # this node incomparable to the numbers already measured.
    echo "== --code-only: verifying the image already has data + warm start =="
    "${SSH}" -o BatchMode=yes "${TARGET}" "cd ${REMOTE_DIR} && \
        test -s dataset/global/link_data.h5 || { echo 'MISSING link_data.h5'; exit 1; }; \
        test -s dataset/global/rate_stats.json || { echo 'MISSING rate_stats.json'; exit 1; }; \
        test -s outputs/supervised_pg_phased/supervised_pg_phased_latest.pt || { echo 'MISSING BC checkpoint'; exit 1; }; \
        test -x ${INSTALL_ROOT}/venv/bin/python || { echo 'MISSING venv'; exit 1; }; \
        echo '  link_data.h5, rate_stats.json, BC checkpoint, venv: all present'"
    echo
    echo "UPLOAD_DONE (code only)"
    exit 0
fi

echo "== uploading registries + warm start =="
"${SCP}" -q \
    dataset/global/node_registry.csv \
    dataset/global/link_registry.csv \
    dataset/global/link_registry_skipped.csv \
    dataset/global/gs_positions.csv \
    dataset/global/weather_coverage.csv \
    "${TARGET}:${REMOTE_DIR}/dataset/global/"

"${SCP}" -q outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
    "${TARGET}:${REMOTE_DIR}/outputs/supervised_pg_phased/"

echo "== uploading link_data.h5 (374 MB, the slow part) =="
time "${SCP}" -q dataset/global/link_data.h5 "${TARGET}:${REMOTE_DIR}/dataset/global/"

if [ "${FULL}" = "1" ]; then
    echo "== uploading outputs/trajs_pg_phased (4.3 GB, BC retraining only) =="
    "${SSH}" -o BatchMode=yes "${TARGET}" "mkdir -p ${REMOTE_DIR}/outputs/trajs_pg_phased"
    time "${SCP}" -qr outputs/trajs_pg_phased/. "${TARGET}:${REMOTE_DIR}/outputs/trajs_pg_phased/"
fi

# --- 3. integrity ----------------------------------------------------------
echo "== verifying link_data.h5 (md5 must match local) =="
LOCAL_MD5="$(md5sum dataset/global/link_data.h5 | cut -d' ' -f1)"
REMOTE_MD5="$("${SSH}" -o BatchMode=yes "${TARGET}" "md5sum ${REMOTE_DIR}/dataset/global/link_data.h5 | cut -d' ' -f1")"
echo "   local  ${LOCAL_MD5}"
echo "   remote ${REMOTE_MD5}"
if [ "${LOCAL_MD5}" != "${REMOTE_MD5}" ]; then
    echo "!! MISMATCH -- re-upload before trusting any result" >&2
    exit 1
fi

# --- 4. rate_stats.json ----------------------------------------------------
# Without this file RateNormalizer silently falls back to a constant p99 = 10.0
# while the real global p99 is ~12,655 bps, so all 11 rate features come out at
# the wrong scale and the policies are not comparable to anything measured so far.
echo "== generating rate_stats.json on the node =="
# `set -o pipefail` on the remote side so a failure inside the python script is
# not masked by tail's exit status -- that is how a missing venv slipped
# through as "UPLOAD_DONE" with no rate_stats.json written.
"${SSH}" -o BatchMode=yes "${TARGET}" \
    "cd ${REMOTE_DIR} && set -o pipefail && ${INSTALL_ROOT}/venv/bin/python scripts/estimate_rate_stats.py 2>&1 | tail -3"

echo
echo "UPLOAD_DONE"
echo "Smoke test:"
echo "  ${SSH} ${TARGET} 'cd ${REMOTE_DIR} && ${INSTALL_ROOT}/venv/bin/python scripts/train/train_graph_mappo.py \\"
echo "      --configs rl_algorithm.yaml train_diag_fast.yaml \\"
echo "      --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \\"
echo "      --run-name diag_smoke --num-updates 3'"
