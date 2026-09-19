#!/usr/bin/env bash
# Ship the encoder/critic rewrite and its benchmark probes to the experiment
# node. Deliberately narrow: only the files that changed, so a run cannot
# accidentally clobber the node's data or venv.
set -eu

HOST="${QKD_HOST:-qinglong@amd276.utah.cloudlab.us}"
ROOT=/opt/qkd/graph_mappo
SSH=/c/Windows/System32/OpenSSH/ssh.exe
SCP=/c/Windows/System32/OpenSSH/scp.exe
[ -x "$SSH" ] || { SSH=ssh; SCP=scp; }

cd "$(dirname "$0")/.." || exit 1

echo "== verifying the remote copy was untouched HEAD =="
# Compared with CR stripped: the tarball upload landed the file with CRLF line
# endings, so a byte-for-byte md5 comparison flags a difference that is not one.
LOCAL_HEAD=$(git show HEAD:qkd_rl/rl/models/graph_mappo.py | tr -d '\r' | md5sum | cut -d' ' -f1)
REMOTE=$( "$SSH" -o BatchMode=yes "$HOST" "md5sum $ROOT/qkd_rl/rl/models/graph_mappo.py" | cut -d' ' -f1 )
echo "   local HEAD (LF) : $LOCAL_HEAD"
echo "   remote          : $REMOTE"
if [ "$LOCAL_HEAD" != "$REMOTE" ]; then
    echo "   -- byte hashes differ, retrying with CR stripped from the remote copy"
    REMOTE_LF=$( "$SSH" -o BatchMode=yes "$HOST" \
        "tr -d '\r' < $ROOT/qkd_rl/rl/models/graph_mappo.py | md5sum" | cut -d' ' -f1 )
    if [ "$LOCAL_HEAD" != "$REMOTE_LF" ]; then
        echo "   !! remote graph_mappo.py differs from HEAD beyond line endings."
        echo "   !! diff it by hand before overwriting."
        exit 1
    fi
    echo "   -> identical modulo CRLF; safe to overwrite"
fi

"$SSH" -o BatchMode=yes "$HOST" "mkdir -p $ROOT/.tmp"

echo "== uploading =="
"$SCP" -q qkd_rl/rl/models/graph_mappo.py "$HOST:$ROOT/qkd_rl/rl/models/graph_mappo.py"
"$SCP" -q .tmp/time_update.py .tmp/ref_forward.py .tmp/ab_encoder.sh "$HOST:$ROOT/.tmp/"
echo "   done"

echo "== verifying =="
"$SSH" -o BatchMode=yes "$HOST" "md5sum $ROOT/qkd_rl/rl/models/graph_mappo.py; grep -c _segment_sum $ROOT/qkd_rl/rl/models/graph_mappo.py"
md5sum qkd_rl/rl/models/graph_mappo.py
