"""`idle_scorer` 逐位不动 —— 是「IDLE 候选从未合法」，还是「合法但不影响损失」？

### 为什么问

`graph_mappo.py:647` 只在 `idle_srcs.size` 非零时调用 `idle_scorer`。
`idle_srcs` 来自 `_build_plan_vectorized` 对**合法**候选的筛选（`:544-550`）。
所以两条可能：

  (A) IDLE 候选在训练时**从不合法** ⟹ 16,641 个参数是死重
  (B) 合法，但它的分数**不影响损失**（例如被 STOP 逻辑接管）

两者的处置不同：(A) 是纯浪费；(B) 说明有两条并行的"停"机制在打架。

### 怎么测（不需要训练，不需要跑 env）

只构造 `ActionSpace` + 物理图，数一数每个节点的合法候选里有没有 IDLE。
这比构造整个 env 便宜得多，也不写任何文件。

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/probe_idle_legal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/opt/qkd/graph_mappo")


def main() -> int:
    try:
        from qkd_rl.core.config import load_default_config
        from qkd_rl.env.action_resolver import NodeActionSpace
    except Exception as exc:                       # noqa: BLE001
        print(f"✗ import 失败: {exc}")
        return 1

    cfg = load_default_config()
    print(f"action_resolver.mode = "
          f"{cfg.get('env', {}).get('action_resolver', {}).get('mode')}")
    print(f"model.mode = {cfg.get('model', {}).get('mode')}")
    print()

    # 找 NodeActionSpace 的构造签名，按名字取参数，避免硬编码顺序
    import inspect
    sig = inspect.signature(NodeActionSpace.__init__)
    print(f"NodeActionSpace.__init__ 签名: {sig}")
    print()

    # IDLE 常量
    idle = getattr(NodeActionSpace, "IDLE", None)
    print(f"NodeActionSpace.IDLE = {idle!r}")
    print()

    # 枚举所有成员，看有没有能直接给出候选表的方法
    methods = [m for m in dir(NodeActionSpace) if not m.startswith("_")]
    print(f"公开方法: {methods}")
    print()
    print("⟹ 需要 env 才能构造 ActionSpace 的话，见下方说明。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
