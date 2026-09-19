"""查清楚探针 G 为什么量出"所有失败请求都容量不足"（cap_opt 中位 = 0）。

中位 0.000 不可能是真的：那等于说这些路径在整个生命期一个 key 都生成不出来。
两个最可能的原因，一次查完：

  1. `rate_provider.is_available(edge, t)` 在我传的 t 上恒为 False
     （比如它只认"当前窗口"或绝对时刻，而我用的是 episode 内的相对步号）；
  2. `get_rate(edge, t)` 返回 0 —— 抓的是当天不可用的时段，或 H5 里这条边就是 0。

做法：跑一小段 episode（120 步），然后对**已被请求用过的最短路**逐跳打印
t=0..29 与 t=当前 的 rate/available，和 provider 的实际类型与内部时刻。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402

SEED = 7
STEPS = 120


def main() -> None:
    profile = _tp.load_validation_profile(ROOT / "configs" / "var2_diag.yaml")
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=STEPS,
        start_mode=profile["start_mode"])
    env = build_env_from_config(config)
    rp = env.rate_provider
    print(f"rate_provider 类型 = {type(rp).__module__}.{type(rp).__name__}")
    for attr in ("seed", "t", "current_t", "horizon", "_t", "day", "_day"):
        if hasattr(rp, attr):
            print(f"  rp.{attr} = {getattr(rp, attr)!r}")
    print(f"env.t（reset 前）= {getattr(env, 't', None)}")
    print(f"env.base_slot / start 类似字段 = "
          f"{ {k: getattr(env, k) for k in ('t', 'base_t', 'slot0', 'start_slot', 'day') if hasattr(env, k)} }")

    policy = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    obs = env.reset(seed=SEED, start_seed=profile.get("start_seed", 0) + SEED)
    print(f"env.t（reset 后）= {getattr(env, 't', None)}")

    # 记下这 120 步里出现过的请求路径（到期与否都算）
    seen_paths: list[tuple[str, str, list[str]]] = []
    seen: set[tuple[str, str]] = set()
    orig_expire = env.requests.expire

    def expire_patched(t: int):
        out = orig_expire(t)
        for req in out:
            p = env.routing.shortest_path(req.src_gs, req.dst_gs)
            if p and (req.src_gs, req.dst_gs) not in seen:
                seen.add((req.src_gs, req.dst_gs))
                seen_paths.append((req.src_gs, req.dst_gs, p))
        return out

    env.requests.expire = expire_patched
    for _ in range(STEPS):
        actions, scores = policy.act(obs)
        obs, _r, term, trunc, _info = env.step(actions, action_scores=scores)
        if term or trunc:
            break
    env.requests.expire = orig_expire

    print(f"\n步结束时 env.t = {getattr(env, 't', None)}；"
          f"抓到 {len(seen_paths)} 条不同的请求路径")
    for src, dst, path in seen_paths[:4]:
        print(f"\n  {src} -> {dst}  hops={len(path)}  edges={path}")
        for e in path:
            r0 = [rp.get_rate(e, t) for t in (0, 1, 5, 29)]
            a0 = [rp.is_available(e, t) for t in (0, 1, 5, 29)]
            print(f"    {e}")
            print(f"      get_rate t=0,1,5,29 -> {['%.3g' % v for v in r0]}")
            print(f"      is_available 同上    -> {a0}")
            print(f"      当前 t={env.t}: rate={rp.get_rate(e, env.t):.3g} "
                  f"avail={rp.is_available(e, env.t)}")
            w = rp.get_edge_window(e, env.t)
            print(f"      window@t: {w}")

    # 换个角度：任意随机边在 t=0 到底有没有速率
    edges = [e.edge_id for e in env.routing.edges][:8]
    print("\n任意 8 条边在 t=0 的 rate：")
    for e in edges:
        print(f"    {e:<40} rate={rp.get_rate(e, 0):.4g} "
              f"avail={rp.is_available(e, 0)}")


if __name__ == "__main__":
    main()