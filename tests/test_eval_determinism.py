"""回归测试：Evaluator 驱动 RL 策略时必须走**确定性**动作。

为什么要钉这一条：`MAPPOPolicy.act(obs, deterministic=False)` 的默认值是
**采样**，而 `Evaluator._act` 调的是 `policy.act(obs)`——即默认。项目里
唯一把 RL 接进 Evaluator 的地方是 `scripts/baselines/run_baselines.py` 的
`_RLPolicyAdapter`，它显式传了 `deterministic=True`，所以**当前没有出错**。

但这是个靠"调用方记得传参"维持的口径：训练侧的 `evaluate_validation` 用
`act_batched(..., deterministic=True)`，专家本身恒确定性，只有 Evaluator
这条路径依赖调用方。2026-09-18 曾因此在探针上量出 4.5 个点的假差异
（采样 0.7830 vs 确定性 0.8247，见 docs/训练诊断记录.md）。
本测试把这个约定钉死：如果哪天有人把 `deterministic=True` 删掉，
这里会红。
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rl_adapter_passes_deterministic_true() -> None:
    """run_baselines.py 里的 RL 适配器必须显式请求确定性动作。"""
    src = (ROOT / "scripts" / "baselines" / "run_baselines.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)

    # 找 `class _RLPolicyAdapter` 里 act 方法对 self.policy.act(...) 的调用
    found = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "_RLPolicyAdapter":
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            fn = sub.func
            if not isinstance(fn, ast.Attribute) or fn.attr != "act":
                continue
            kw = {k.arg: k.value for k in sub.keywords if k.arg}
            found = kw.get("deterministic")
    assert found is not None, (
        "_RLPolicyAdapter.act 里没有给 self.policy.act 传 deterministic —— "
        "MAPPOPolicy.act 默认是**采样**，会让 Evaluator 出的数与其他口径不可比")
    assert isinstance(found, ast.Constant) and found.value is True, (
        "_RLPolicyAdapter 传的 deterministic 不是字面量 True")


def test_policy_act_default_is_sampling() -> None:
    """把"默认是采样"这件事本身记下来——它是上面那条测试存在的理由。"""
    src = (ROOT / "qkd_rl" / "rl" / "algos" / "policy.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "act":
            args = node.args
            names = [a.arg for a in args.args]
            if "deterministic" not in names:
                continue
            defaults = dict(zip(names[-len(args.defaults):], args.defaults))
            d = defaults.get("deterministic")
            assert isinstance(d, ast.Constant) and d.value is False, (
                "MAPPOPolicy.act 的 deterministic 默认值变了——"
                "若改成 True，请同时更新 docs/训练诊断记录.md 里的口径说明")
            return
    raise AssertionError("没找到 MAPPOPolicy.act")
