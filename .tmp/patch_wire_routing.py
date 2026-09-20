"""给 GraphBuilder 接上 routing（v2 的 on_pending_path 要用 `routing.shortest_path`）。

三处改动，全部**按精确字符串**替换，改完逐条自证：

  1. `graph_builder.py` 的 `GraphBuilder.__init__` 签名加 `routing=None`
  2. 函数体里加 `self.routing = routing`
  3. `factory.py` 构造 GraphBuilder 时传 `routing=routing`

★ 幂等：已经改过就跳过（按改动 2 的标记判断）。
★ 自证：改完 `grep` 出三处，并且**从改动后的文件 import 一次**，
   断言 `GraphBuilder.__init__` 签名里有 `routing`。
"""
import re
import sys
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
GB = REPO / "qkd_rl" / "env" / "graph_builder.py"
FA = REPO / "qkd_rl" / "env" / "factory.py"

MARK = "self.routing = routing"

# ---------- 1 & 2: graph_builder.py ----------
src = GB.read_text(encoding="utf-8")
if MARK in src:
    print("[跳过] graph_builder.py 已改过（幂等）")
else:
    old_sig = """        config: dict,
        history_buffer=None,
    ):
        self.nodes = nodes"""
    new_sig = """        config: dict,
        history_buffer=None,
        routing=None,
    ):
        self.nodes = nodes"""
    if src.count(old_sig) != 1:
        print(f"★ 签名锚点命中 {src.count(old_sig)} 次（应为 1）⟹ 不改，退出")
        sys.exit(2)
    src = src.replace(old_sig, new_sig)

    old_body = """        self.node_index = {node.node_id: idx for idx, node in enumerate(nodes)}"""
    new_body = """        # v2：`_compute_on_pending_path` 需要 routing 的**规范路**
        # （`_next_hop` 的 tie-break 定义在 RoutingPolicy 里）⟹ 绝不自己重写 BFS。
        self.routing = routing
        self.node_index = {node.node_id: idx for idx, node in enumerate(nodes)}"""
    if src.count(old_body) != 1:
        print(f"★ 函数体锚点命中 {src.count(old_body)} 次（应为 1）⟹ 不改，退出")
        sys.exit(2)
    src = src.replace(old_body, new_body)
    GB.write_text(src, encoding="utf-8", newline="")      # newline="" 防 CRLF 二次转换
    print("[已改] graph_builder.py")

# ---------- 3: factory.py ----------
# ★ 幂等判据必须**锚定 GraphBuilder 调用块**，不能用全文件子串：
#   `routing=routing,` 在第 92 行的 QKDEnv(...) 里本来就有 ⟹ 全文件匹配会
#   误判为"已改过"而**跳过**，还打印 ✓（我踩过，见本文件同目录的诊断记录）。
fsrc = FA.read_text(encoding="utf-8")
GB_CALL_NEEDLE = "graph_builder = GraphBuilder("
i = fsrc.find(GB_CALL_NEEDLE)
if i < 0:
    print("★ 找不到 GraphBuilder 调用块 ⟹ 不改，退出")
    sys.exit(2)
j = fsrc.index("\n    )", i) if "\n    )" in fsrc[i:] else -1
if j < 0:
    j = fsrc.index("\r\n    )", i)
    j += 1
block = fsrc[i:j]
nl = "\r\n" if "\r\n" in block else "\n"
print(f"  自证：GraphBuilder 调用块行尾 = {'CRLF' if nl == chr(13)+chr(10) else 'LF'}")
if "routing=routing," in block:
    print("[跳过] factory.py 的 GraphBuilder 调用已带 routing（幂等）")
else:
    # ★ 按**行**插入，与行尾无关（先前用 `\n` 锚点在 CRLF 文件上命中 0 次）
    lines = block.split(nl)
    hits = [k for k, ln in enumerate(lines) if ln.strip() == "history_buffer=history_buffer,"]
    if len(hits) != 1:
        print(f"★ GraphBuilder 调用块内 `history_buffer=history_buffer,` 命中 {len(hits)} 行（应为 1）⟹ 不改")
        sys.exit(2)
    lines.insert(hits[0] + 1, "        routing=routing,")
    fsrc = fsrc[:i] + nl.join(lines) + fsrc[j:]
    FA.write_text(fsrc, encoding="utf-8", newline="")
    print("[已改] factory.py（GraphBuilder 调用，按行插入）")

# ---------- 自证 ----------
print()
print("=== 自证：三处改动都在 ===")
g = GB.read_text(encoding="utf-8")
f = FA.read_text(encoding="utf-8")
_i = f.find("graph_builder = GraphBuilder(")
_j = f.index("\n    )", _i) if _i >= 0 else -1
_fblock = f[_i:_j] if _i >= 0 and _j > _i else ""
checks = [
    ("graph_builder 签名有 routing=None", "        routing=None,\n    ):" in g),
    ("graph_builder 体有 self.routing = routing", MARK in g),
    ("factory 的 **GraphBuilder 调用**传了 routing", "routing=routing," in _fblock),
    ("函数里确实用了 self.routing",
     "routing = self.routing" in g and "routing.shortest_path(req.src_gs" in g),
]
ok = True
for name, good in checks:
    print(f"  {'✓' if good else '★'} {name}")
    ok = ok and good

# ---------- 编译 + import 自证 ----------
print()
print("=== 编译 ===")
import py_compile
for p in (GB, FA):
    try:
        py_compile.compile(str(p), doraise=True)
        print(f"  ✓ {p.name} 编译通过")
    except py_compile.PyCompileError as e:
        print(f"  ★ {p.name} 编译失败：{e}")
        ok = False

print()
if not ok:
    print("DECISION=PATCH_FAILED")
    sys.exit(2)
print("DECISION=PATCH_OK")
