"""造反证：证明 `test_on_pending_path_feature.py` **能红**，不只是能绿。

### 为什么必须做这一步

`tests/test_on_pending_path_feature.py` 全绿只说明"实现和测试**互相一致**"。
一个恒 0 的实现配一个只说"跑通了"的测试，也会全绿。

判据（本仓库的规矩，`gate-must-print-its-inputs` / `never-run-code-path-hides-bugs`）：
**把实现按已知坏法改坏，测试必须红；且要红在"该红的那条"上。**

三个已知坏法（都是这一列**特有**的静默失败模式）：

| 坏法 | 模拟的实现缺陷 | 必须红的测试 |
|---|---|---|
| `zeros` | 子图取错 / 查不到 node_index ⟹ **恒 0** | 第 3 条（该 1 却是 0）+ 第 5 条（取值不变） |
| `ones` | 判据写反 ⟹ **恒 1** | 第 2/4 条（该 0 却是 1）+ 第 5 条 |
| `all_edges` | 漏了"有存量"这个子图条件 ⟹ **有边就算 1** | 第 4 条（没有存量却给 1） |

`all_edges` 是最像"对的"那种坏：它把"几何最短路"误当"有存量子图上的路"，
肉眼读代码很难发现，只有"没有存量 ⟹ 0"这条能抓住。

用法（服务器上）：
    /opt/qkd/venv/bin/python .tmp/verify_on_path_test_can_fail.py
"""
from __future__ import annotations

import subprocess
import textwrap
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEST_FILE = "tests/test_on_pending_path_feature.py"


def _run_pytest() -> tuple[int, str]:
    """在**独立进程**里跑测试文件 —— 免得被本进程的 monkeypatch 污染。"""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", TEST_FILE, "-q", "--tb=line", "-p", "no:cacheprovider"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def _write_mutant(kind: str) -> str:
    """把一份"坏实现"写成 pytest 插件，注入到测试进程里。

    做法：生成一个插件文件，它在 import 时就把
    `GraphBuilder._compute_on_pending_path` 换掉。插件通过
    `PYTEST_PLUGINS` 环境变量加载，**不需要改仓库里任何文件**。

    ★ 缩进必须由 `textwrap.dedent` 统一处理：手写字面量很容易让续行多缩一层，
    生成出 `IndentationError` —— 而"插件炸了"和"测试抓住了"在 `rc != 0` 上
    **长得一模一样**（这一版的第一稿就踩了，判据已改成要求失败集非空）。
    """
    body = {
        "zeros": """
        return np.zeros(len(self._edge_list), dtype=np.float32)
        """,
        "ones": """
        return np.ones(len(self._edge_list), dtype=np.float32)
        """,
        # ★ 这条才是"最像对的"那种坏：把判据从「**有存量的边**子图上的路」
        #   退化成「**整张静态图**上的几何最短路」。代码读起来完全合理，
        #   只有"没有存量 ⟹ 必须 0"这一条能抓住它。
        "all_edges": """
        n = len(self._edge_list)
        out = np.zeros(n, dtype=np.float32)
        pending = requests.get_pending()
        if not pending:
            return out
        adj = {}
        for edge in self.edges:              # ← 故意的错：不看 positive
            si = self.node_index[edge.src]
            di = self.node_index[edge.dst]
            pi = self._edge_pos[edge.edge_id]
            adj.setdefault(si, []).append((di, pi))
            adj.setdefault(di, []).append((si, pi))
        for req in pending:
            src = self.node_index.get(req.src_gs)
            dst = self.node_index.get(req.dst_gs)
            if src is None or dst is None or src == dst:
                continue
            parent = {src: (-1, -1)}
            queue = deque([src])
            found = False
            while queue:
                cur = queue.popleft()
                if cur == dst:
                    found = True
                    break
                for nxt, pos in adj.get(cur, ()):
                    if nxt in parent:
                        continue
                    parent[nxt] = (cur, pos)
                    queue.append(nxt)
            if not found:
                continue
            cur = dst
            while parent[cur][0] != -1:
                prev, pos = parent[cur]
                out[pos] = 1.0
                cur = prev
        return out
        """,
    }[kind]
    inner = textwrap.indent(textwrap.dedent(body).strip("\n"), "    ")
    return (
        "import numpy as np\n"
        "from collections import deque\n"
        "from qkd_rl.env.graph_builder import GraphBuilder\n"
        "\n"
        "def _bad(self, requests):\n"
        f"{inner}\n"
        "\n"
        "GraphBuilder._compute_on_pending_path = _bad\n"
    )


def main() -> int:
    import os

    plug = ROOT / ".tmp" / "_mutant_plugin.py"

    print("=" * 72)
    print("造反证：测试必须能红，且红在'该红的那条'上")
    print("=" * 72)

    # ---- 0. 原版必须绿（否则后面"红"没有意义：本来就红） ----
    rc, out = _run_pytest()
    line = [l for l in out.splitlines() if "passed" in l or "failed" in l or "error" in l]
    print(f"\n[原版]        rc={rc}  {line[-1] if line else out.strip()[-120:]}")
    if rc != 0:
        print("  ⟹ 原版就不过，后面的造反证无意义。先修测试。")
        return 1

    verdicts = {}
    for kind, expect, must_fail in (
        ("zeros", "第 3/5 条", ("test_one_for_edges_on_a_pending_request_path",
                               "test_not_constant_across_a_real_run")),
        ("ones", "第 2/4/5 条", ("test_zero_when_no_pending_requests",
                                "test_zero_for_edges_without_stock")),
        ("all_edges", "第 4 条", ("test_zero_for_edges_without_stock",)),
    ):
        plug.write_text(_write_mutant(kind), encoding="utf-8")
        env = dict(os.environ)
        env["PYTEST_PLUGINS"] = "_mutant_plugin"
        env["PYTHONPATH"] = str(plug.parent) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", TEST_FILE, "-q", "--tb=line", "-p", "no:cacheprovider"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        out = proc.stdout + proc.stderr
        failed = sorted({l.split("::")[-1].split()[0].split(":")[0]
                         for l in out.splitlines() if "::" in l and "FAILED" in l})
        # ★ 判据不能只看 rc != 0 —— **收集期报错（import error 等）也给 rc != 0**，
        #   那是"测试根本没跑"，不是"测试抓住了"。必须要求失败集**非空**，
        #   且**包含**该坏法必须触发的那几条。
        missing = [m for m in must_fail if m not in failed]
        caught = bool(failed) and not missing
        verdicts[kind] = caught
        status = "红 ✓" if caught else ("★ 收集期就炸了（测试没跑，不算抓住）" if not failed else "★ 没红在该红的那条")
        print(f"\n[坏法 {kind:<10}] rc={proc.returncode}  {status}   期望红在：{expect}")
        for f in failed:
            print(f"    红：{f}")
        if missing:
            print(f"    ★ 缺失（该红没红）：{missing}")
        if not failed:
            tail = [l for l in out.splitlines() if l.strip()][-6:]
            for t in tail:
                print(f"    | {t[:160]}")

    # ---- 结论 ----
    ok = verdicts.get("zeros") and verdicts.get("ones") and verdicts.get("all_edges")
    print("\n" + "=" * 72)
    print("结论：三种已知坏法" + ("**全部**能让测试变红" if ok else "**没有**全部被抓到"))
    print("  → 测试是**红能力**的（不是只会绿的橡皮章）" if ok else
          "  → ★ 有坏法能静默通过 ⟹ 测试对这一类失效没有分辨力")
    print("=" * 72)
    try:
        plug.unlink()
    except OSError:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
