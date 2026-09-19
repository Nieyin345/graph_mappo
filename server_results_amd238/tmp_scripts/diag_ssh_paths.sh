#!/usr/bin/env bash
# 找出这台机器上哪种路径写法能通过 bash 的 -x 测试，从而被 pick_ssh 选中
ssh_win="C:\\Windows\\System32\\OpenSSH\\ssh.exe"
for cand in \
    "C:/Windows/System32/OpenSSH/ssh.exe" \
    "/c/Windows/System32/OpenSSH/ssh.exe" \
    "/mnt/c/Windows/System32/OpenSSH/ssh.exe" \
    "$ssh_win" \
    "/cygdrive/c/Windows/System32/OpenSSH/ssh.exe" \
    "/c/windows/system32/openssh/ssh.exe"
do
    if [ -x "$cand" ]; then r=yes; else r=no; fi
    if [ -f "$cand" ]; then f=yes; else f=no; fi
    printf '  -x=%s  -f=%s  [%s]\n' "$r" "$f" "$cand"
done
echo "uname: $(uname -a)"
echo "MSYSTEM=${MSYSTEM:-<unset>}"
echo "command -v ssh: $(command -v ssh)"