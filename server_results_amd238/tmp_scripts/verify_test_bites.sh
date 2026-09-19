#!/usr/bin/env bash
# 确认新测试不是"永远通过"的摆设：**在副本里**注入结构差异，它必须失败。
#
# 为什么用副本：节点上 policy.py 与本地逐位一致（sha256 405e01cd…），且三个
# 延长臂正在跑。直接改节点文件虽然不影响已在内存里的进程，但万一没还原干净，
# 后续任何新起的 run 都会载入坏代码——这种"静默污染"代价远大于本次验证的收益。
# 所以在 /tmp 的副本里做，节点的代码一字不动。
set -u
SRC=/opt/qkd/graph_mappo
WORK=/tmp/tbcheck

rm -rf "$WORK"
mkdir -p "$WORK"

echo "=== 拷贝（排除 outputs/ 与 .git，只要代码 + 测试）==="
tar -C "$SRC" --exclude=outputs --exclude=.git --exclude=__pycache__ \
    --exclude='*.pt' -cf - . 2>/dev/null | tar -C "$WORK" -xf -
echo "  副本大小: $(du -sh "$WORK" | cut -f1)"

cd "$WORK" || exit 1
F=qkd_rl/rl/algos/policy.py
sha_before=$(sha256sum "$F" | cut -c1-16)

echo
echo "=== 1) 副本基线（应通过）==="
/opt/qkd/venv/bin/python -m pytest tests/test_matching_log_prob_paths.py -q 2>&1 | tail -2

echo
echo "=== 2) 在副本里注入结构差异（arrays 路径返回值 +1e-2）==="
# 用 ast 拿函数的真实行范围，别用正则猜——上一次正则没匹配上，
# 而脚本却把"注入失败"报成了"守卫无效"，得出完全相反的结论。
INJECT_OK=0
/opt/qkd/venv/bin/python - <<'PY' && INJECT_OK=1
import ast
import re
from pathlib import Path

p = Path("qkd_rl/rl/algos/policy.py")
src = p.read_text(encoding="utf-8")
tree = ast.parse(src)

target = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_matching_log_prob_entropy_arrays":
        target = node
        break
if target is None:
    print("  !! ast 里找不到 _matching_log_prob_entropy_arrays")
    raise SystemExit(1)

print(f"  ast 定位: L{target.lineno}..L{target.end_lineno}")
lines = src.splitlines(keepends=True)

# 在该函数体内最后一个 return 之前插入扰动
ret_line = None
for node in ast.walk(target):
    if isinstance(node, ast.Return) and node.lineno <= target.end_lineno:
        if ret_line is None or node.lineno > ret_line:
            ret_line = node.lineno
if ret_line is None:
    print("  !! 函数体内找不到 return")
    raise SystemExit(1)

# ret_line 是 1-based；取那一行的缩进
raw = lines[ret_line - 1]
indent = raw[:len(raw) - len(raw.lstrip())]
m = re.match(r"return\s+([A-Za-z_][A-Za-z0-9_]*)", raw.strip())
if not m:
    print(f"  !! return 行不是简单返回名字: {raw.strip()!r}")
    raise SystemExit(1)
var = m.group(1)
lines.insert(ret_line - 1, f"{indent}{var} = {var} + 1.0e-2  # INJECTED\n")
p.write_text("".join(lines), encoding="utf-8")
print(f"  已注入: 在 L{ret_line} 前对 {var} 加 1e-2")
PY
if [ "$INJECT_OK" -ne 1 ]; then
  echo "  ✗ 注入步骤失败——本次验证**没有结论**（不要读成'守卫无效'）"
  cd / ; rm -rf "$WORK" ; exit 2
fi

echo
echo "=== 3) 注入后（必须失败）==="
set +e
/opt/qkd/venv/bin/python -m pytest tests/test_matching_log_prob_paths.py -q 2>&1 | tail -8
rc=${PIPESTATUS[0]}
set -e

echo
echo "=== 4) 结论 ==="
if [ "$rc" -ne 0 ]; then
  echo "  ✓ 注入结构差异后测试失败 → 这个守卫真的会咬人，不是摆设"
else
  echo "  ✗ 注入成功但测试仍通过 → 守卫无效，必须修"
fi

cd /
rm -rf "$WORK"
echo
echo "=== 5) 节点代码未动（核对 sha256）==="
echo "  注入前副本: $sha_before"
echo "  节点原文件: $(sha256sum $SRC/$F | cut -c1-16)"
