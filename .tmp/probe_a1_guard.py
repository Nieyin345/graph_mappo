"""守卫对照：`_validate_options` 的新增检查真的会喊吗？（真·旧版 vs 修复版）

### 为什么单独测守卫

守卫是**新代码** —— 没有它时"放行"是必然的，所以单测修复版只能证明
"会喊"，不能证明"喊的是**该喊的那一处**"。这里把两份 `config.py` 同时
加载（`qkd_rl/core/config.py` **无相对导入**，可安全独立加载），在同一份
配置上对打。

判据是**双向**的：
- 非法输入：修复版必须报错，旧版必须放行（否则说明输入本来就被别处拦了）
- 合法输入：**两份都必须放行**（否则守卫会误伤正常配置 —— 过保守的门
  也是静默失效，见记忆 `over-conservative-gate-is-silent-too`）

用法（服务器上）：
    /opt/qkd/venv/bin/python -u /tmp/probe_a1_guard.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))


def load_validator(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ConfigValidator._validate_options


def build_cfg(extra: dict | None = None):
    from scripts.train import train_graph_mappo as tgm
    args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", seed=7, num_updates=1,
        run_name="probe_a1", device="cpu")
    cfg = tgm.build_config(args)
    if extra:
        for dotted, val in extra.items():
            node = cfg
            for p in dotted.split(".")[:-1]:
                node = node.setdefault(p, {})
            node[dotted.split(".")[-1]] = val
    return cfg


CASES = [
    # (标签, 补丁, 期望「修复版」是否报错)
    ("model.activation 写到顶层（非法位置）", {"model.activation": "gelu"}, True),
    ("model.dropout 写到顶层（非法位置）", {"model.dropout": 0.3}, True),
    ("encoder.activation=sigmoid（未实现）", {"model.encoder.activation": "sigmoid"}, True),
    ("encoder.dropout=1.5（越界）", {"model.encoder.dropout": 1.5}, True),
    ("encoder.dropout=-0.1（越界）", {"model.encoder.dropout": -0.1}, True),
    ("encoder.activation=gelu（合法）", {"model.encoder.activation": "gelu"}, False),
    ("encoder.dropout=0.2（合法）", {"model.encoder.dropout": 0.2}, False),
    ("原样不动（基线，必须放行）", {}, False),
]


def main() -> int:
    real_v = load_validator(ROOT / "qkd_rl" / "core" / "config.py", "cfg_real")
    fixed_v = load_validator(Path("/tmp/config_fixed.py"), "cfg_fixed")

    fails: list[str] = []
    print("=" * 78)
    print("守卫对照：真·旧版 vs 修复版（同一份配置、同一批输入）")
    print("=" * 78)
    print(f"{'用例':<40} {'旧版':<10} {'修复版':<10} 判定")
    print("-" * 78)
    for label, patch, should_raise in CASES:
        row = []
        for fn in (real_v, fixed_v):
            try:
                fn(build_cfg(patch))
                row.append("放行")
            except ValueError:
                row.append("报错")
        old_s, fix_s = row
        ok = (fix_s == "报错") == should_raise
        if should_raise and old_s == "报错":
            # 非法输入旧版也报错 ⟹ 那是别处拦的，本用例测不到新守卫
            ok = False
            note = "旧版也报错 ⟹ 测不到新守卫"
        else:
            note = ""
        print(f"{label:<40} {old_s:<10} {fix_s:<10} {'✓' if ok else '✗'} {note}")
        if not ok:
            fails.append(f"{label}: 旧版={old_s} 修复版={fix_s}（期望修复版"
                         f"{'报错' if should_raise else '放行'}）{note}")

    print("-" * 78)
    if fails:
        print(f"✗ {len(fails)} 条不通过：")
        for f in fails:
            print(f"    - {f}")
    else:
        print("✓ 守卫双向正确：非法必报错、合法必放行（不误伤正常配置）")
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
