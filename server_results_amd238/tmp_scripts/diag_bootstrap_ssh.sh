#!/usr/bin/env bash
# 诊断：server_bootstrap.sh 的 preflight 为什么连不上
for cand in "C:/Windows/System32/OpenSSH/ssh.exe" \
            "/c/Windows/System32/OpenSSH/ssh.exe" \
            "$(command -v ssh 2>/dev/null || true)"; do
    printf 'cand=[%s]  -n=%s  -x=%s\n' "$cand" \
        "$([ -n "$cand" ] && echo yes || echo no)" \
        "$([ -x "$cand" ] && echo yes || echo no)"
done

SSH_BIN=""
for cand in "C:/Windows/System32/OpenSSH/ssh.exe" \
            "/c/Windows/System32/OpenSSH/ssh.exe" \
            "$(command -v ssh 2>/dev/null || true)"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then SSH_BIN="$cand"; break; fi
done
printf 'SSH_BIN=[%s]\n' "$SSH_BIN"

echo "--- 用 BatchMode + accept-new 试连 ---"
"$SSH_BIN" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
    qinglong@clnode118.clemson.cloudlab.us true
echo "exit=$?"

echo "--- 只用 BatchMode（不带 accept-new） ---"
"$SSH_BIN" -o BatchMode=yes -o ConnectTimeout=20 \
    qinglong@clnode118.clemson.cloudlab.us true
echo "exit=$?"