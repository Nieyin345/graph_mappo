"""--shuffle-seed / --lr-final（2026-10-03）：BC 数据目录模式的文件打乱与余弦退火。

- 缺省（两者 None）= 原行为：解析默认值为 None；
- shuffled_passes：每轮是同一文件集合的一个排列、可复现、各轮顺序不同；
- cosine_lr：端点 lr0 / lr1，单调不增；
- --arc-order 缺省 stored、--step-pool 缺省 1（原行为）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("sup_pg", ROOT / "scripts/train/supervised_train_pg_phased.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_defaults_are_off(monkeypatch) -> None:
    m = _mod()
    monkeypatch.setattr(sys, "argv", ["x"])
    a = m.parse_args()
    assert a.shuffle_seed is None and a.lr_final is None


def test_shuffled_passes_permutation_and_reproducible() -> None:
    m = _mod()
    files = [f"day_{i:03d}" for i in range(50)]
    out = m.shuffled_passes(files, 3, 7)
    assert len(out) == 150
    for p in range(3):
        assert sorted(out[p * 50:(p + 1) * 50]) == files
    assert out[:50] != files and out[:50] != out[50:100]
    assert out == m.shuffled_passes(files, 3, 7)


def test_cosine_lr_endpoints_monotone() -> None:
    m = _mod()
    n = 100
    lrs = [m.cosine_lr(1e-3, 1e-5, i, n) for i in range(n)]
    assert abs(lrs[0] - 1e-3) < 1e-15 and abs(lrs[-1] - 1e-5) < 1e-15
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert m.cosine_lr(1e-3, 1e-5, 0, 1) == 1e-3


def test_arc_order_and_step_pool_defaults(monkeypatch) -> None:
    m = _mod()
    monkeypatch.setattr(sys, "argv", ["x"])
    a = m.parse_args()
    assert a.arc_order == "stored" and a.step_pool == 1
