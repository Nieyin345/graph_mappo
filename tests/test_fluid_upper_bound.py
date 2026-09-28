"""流体 LP 上界的手算小算例（2026-09-27）。

上界必须 (1) 端口约束生效、(2) 截止期约束生效、(3) 分母含无路请求、
(4) 在无约束时恰好等于需求 —— 每条都有对应变异检查。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.milp.fluid_upper_bound import SCALE, solve_bound


class _Provider:
    """rates[t][link], avail[t][link]；get_window_blocks 返回 env 同构的 (rates, avail) 块。"""

    def __init__(self, rates: np.ndarray, avail: np.ndarray, ids: list[str]):
        self.rates, self.avail, self.ids = rates, avail, ids

    def edge_link_ids(self, eids):
        return np.array([self.ids.index(e) for e in eids])

    def get_window_blocks(self, t):
        return (self.rates[t][None, :], self.avail[t][None, :])


def _env(edge_list, rates, avail=None):
    edges = [SimpleNamespace(edge_id=f"E_{u}__{v}", src=u, dst=v) for u, v in edge_list]
    ids = [e.edge_id for e in edges]
    rates = np.asarray(rates, dtype=float)
    avail = np.ones_like(rates) if avail is None else np.asarray(avail, dtype=float)
    return SimpleNamespace(routing=SimpleNamespace(edges=edges),
                           rate_provider=_Provider(rates, avail, ids),
                           scenario=SimpleNamespace(slot_seconds=1.0))


def _req(s, d, amount, arr, dl):
    return SimpleNamespace(src_gs=s, dst_gs=d, amount=float(amount), arrival_t=arr, deadline_t=dl)


T = 10
R = SCALE  # 每槽每边生成 1 个 SCALE 单位


def test_unconstrained_line_serves_everything():
    env = _env([("A", "X"), ("X", "B")], np.full((T, 2), R))
    out = solve_bound(env, 0, T, [_req("A", "B", 3 * R, 0, 9)], bucket=1)
    assert out["lp_status"] == 0
    assert out["ub_success"] == pytest.approx(1.0, abs=1e-6)


def test_line_rate_limited_by_deadline():
    # 截止 t=4 ⟹ 每条边最多生成 5 槽 × R；需求 8R ⟹ 上界 5/8
    env = _env([("A", "X"), ("X", "B")], np.full((T, 2), R))
    out = solve_bound(env, 0, T, [_req("A", "B", 8 * R, 0, 4)], bucket=1)
    assert out["ub_success"] == pytest.approx(5 / 8, abs=1e-6)


def test_hub_port_constraint_binds():
    # 星形：中继 X 连 A,B,C,D；请求 A→B、C→D 各 100R。
    # 无端口约束：每请求 10 槽 × R = 10R，共 20R。
    # 端口：X 每槽至多 2 条边激活 ⟹ 总生成 ≤ 2R/槽；每单位流耗 2 把密钥 ⟹ 总服务 ≤ 10R。
    env = _env([("A", "X"), ("B", "X"), ("C", "X"), ("D", "X")], np.full((T, 4), R))
    reqs = [_req("A", "B", 100 * R, 0, 9), _req("C", "D", 100 * R, 0, 9)]
    out = solve_bound(env, 0, T, reqs, bucket=1)
    assert out["ub_served"] == pytest.approx(10 * R, rel=1e-6)


def test_edge_available_only_after_deadline_gives_zero():
    avail = np.zeros((T, 2))
    avail[:, 0] = 1
    avail[6:, 1] = 1  # X–B 只在 t≥6 可用，请求截止 t=3
    env = _env([("A", "X"), ("X", "B")], np.full((T, 2), R), avail)
    out = solve_bound(env, 0, T, [_req("A", "B", 2 * R, 0, 3)], bucket=1)
    assert out["ub_served"] == pytest.approx(0.0, abs=1e-6)


def test_unroutable_request_stays_in_denominator():
    env = _env([("A", "X"), ("X", "B")], np.full((T, 2), R))
    reqs = [_req("A", "B", 1 * R, 0, 9), _req("A", "Z", 3 * R, 0, 9)]  # Z 不在图里
    out = solve_bound(env, 0, T, reqs, bucket=1)
    assert out["arrived"] == pytest.approx(4 * R)
    assert out["ub_success"] == pytest.approx(0.25, abs=1e-6)


def test_bucket_rounding_only_loosens():
    # 分桶把截止期向上取整 ⟹ 上界只会变大（仍是上界）
    env = _env([("A", "X"), ("X", "B")], np.full((T, 2), R))
    r = [_req("A", "B", 8 * R, 0, 4)]
    exact = solve_bound(env, 0, T, r, bucket=1)["ub_served"]
    coarse = solve_bound(env, 0, T, r, bucket=4)["ub_served"]
    assert coarse >= exact - 1e-6


# ---- 整数激活版（改进项 5，2026-09-28）：integer_activation=True
# 激活 a[e,t] ∈ {0,1}。度 ≤2 的整数子图（路径/环）总能定向成每点 ≤1 发 ≤1 收，
# 所以整数版的端口约束是**精确的**；其余松弛不变 ⟹ 仍是严格上界，且 ≤ LP 上界。
# 时间限制内未证最优时用 MILP 的对偶界（仍是上界）。


def _star_one_slot():
    # 单时隙星形 X–A/B/C，速率 R；请求 X→A、X→B、X→C 各 2R/3，截止 t=0。
    # LP：每边 a=2/3（X 度 = 2）⟹ 三条边各生成 2R/3 ⟹ 全部满足，上界 1。
    # 整数：X 至多开 2 条边 ⟹ 至多服务 2 个请求 = 4R/3 ⟹ 上界 2/3。
    env = _env([("X", "A"), ("X", "B"), ("X", "C")], np.full((1, 3), R))
    reqs = [_req("X", v, 2 * R / 3, 0, 0) for v in ("A", "B", "C")]
    return env, reqs


def test_integer_closes_fractional_port_gap():
    env, reqs = _star_one_slot()
    lp = solve_bound(env, 0, 1, reqs, bucket=1)
    ip = solve_bound(env, 0, 1, reqs, bucket=1, integer_activation=True)
    assert lp["ub_success"] == pytest.approx(1.0, abs=1e-6)
    assert ip["lp_status"] == 0 and ip["integer"] is True
    assert ip["ub_success"] == pytest.approx(2 / 3, abs=1e-6)


def test_integer_equals_lp_when_no_fractional_advantage():
    # 星形 4 叶、10 槽：整数也能每槽开 2 条边轮换 ⟹ 与 LP 同为 10R
    env = _env([("A", "X"), ("B", "X"), ("C", "X"), ("D", "X")], np.full((T, 4), R))
    reqs = [_req("A", "B", 100 * R, 0, 9), _req("C", "D", 100 * R, 0, 9)]
    ip = solve_bound(env, 0, T, reqs, bucket=1, integer_activation=True)
    assert ip["ub_served"] == pytest.approx(10 * R, rel=1e-6)


def test_integer_bound_never_exceeds_lp():
    env = _env([("A", "X"), ("X", "B")], np.full((T, 2), R))
    r = [_req("A", "B", 8 * R, 0, 4), _req("A", "X", 3 * R, 0, 6)]
    lp = solve_bound(env, 0, T, r, bucket=1)["ub_served"]
    ip = solve_bound(env, 0, T, r, bucket=1, integer_activation=True)["ub_served"]
    assert ip <= lp + 1e-6


def test_default_is_lp_and_marks_integer_false():
    env, reqs = _star_one_slot()
    out = solve_bound(env, 0, 1, reqs, bucket=1)
    assert out["integer"] is False
