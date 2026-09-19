#!/usr/bin/env bash
# 验证新测试真的能抓到 bug：临时还原 env.py，看测试是否失败，再恢复。
set -uo pipefail
cd /opt/qkd/graph_mappo || exit 1

cp qkd_rl/env/env.py /tmp/env_fixed.py

# 还原成旧的错误语义：时间上限被报成 terminated
/opt/qkd/venv/bin/python - <<'PY'
import pathlib
p = pathlib.Path("qkd_rl/env/env.py")
s = p.read_text(encoding="utf-8")
old = "return obs, reward_detail.total, False, truncated, self.metrics.last_info(reward_detail)"
new = "return obs, reward_detail.total, (not self.continuous) and self.steps >= episode_steps, truncated, self.metrics.last_info(reward_detail)"
assert old in s, "找不到要替换的行"
p.write_text(s.replace(old, new), encoding="utf-8")
print("已还原为旧语义（terminated = 时间上限）")
PY

echo "--- 用旧语义跑新测试（期望 FAILED）---"
/opt/qkd/venv/bin/python -m pytest tests/test_time_limit.py -q -p no:cacheprovider 2>&1 | tail -8

echo
echo "--- 恢复修复 ---"
cp /tmp/env_fixed.py qkd_rl/env/env.py
/opt/qkd/venv/bin/python -m pytest tests/test_time_limit.py -q -p no:cacheprovider 2>&1 | tail -4

echo
echo "--- 与服务器 git 快照比对（应无差异）---"
git status --porcelain qkd_rl/env/env.py || true
echo "=== done ==="
