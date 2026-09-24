"""查验证 regime 下 `qkp.initial_level` 的实际取值。

## 为什么这一条决定前面所有结论

`probe_deadline_headroom` 在 dl=240 时报：专家 SR = 0.8175 > 全剧并集上界 0.8133。
**上界被突破**。若成立，唯一自洽的解释是：专家用了**从未出现在 `avail_by_t`
里的边** —— 而那只有一种可能：`initial_level > 0`（种子密钥直接铺在**所有**
物理边上，`key_ttl_steps=1e6` 使其永不过期）。

  · 若 il == 0 ⟹ 上界被突破是**不可能**的 ⟹ 我的探针还有 bug，必须继续查
  · 若 il > 0  ⟹ 上界**本来就不是上界**，且专家一直在用这条后门
    ⟹ 所有"天花板/空间"的结论都要带上这个前提

★ 同时打印：有多少条边**从未**出现在 avail_by_t，
  以及这些边在 il>0 时是否持有存量。

用法：
    python3 -u probe_initial_level_check.py --seed 100 --steps 240
"""
import importlib.util
import os
import sys
from pathlib import Path

_here = sys.path[0] if sys.path else ""
if _here and _here not in ("", "."):
    sys.path[:] = [p for p in sys.path if p != _here]

REPO = Path(os.environ.get("GM_REPO", "/opt/qkd/graph_mappo"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse                                                   # noqa: E402
from qkd_rl.env.factory import build_env_from_config              # noqa: E402


def _tp():
    spec = importlib.util.spec_from_file_location(
        "gm_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])

    print("=" * 92)
    print("验证 regime 下 qkp 的实际取值（读的是**运行时**配置，不是 yaml 字面量）")
    print("=" * 92)
    print(f"  cfg['qkp'] = {cfg.get('qkp')}")
    print(f"  cfg['requests']['deadline_steps'] = {cfg.get('requests', {}).get('deadline_steps')}")
    print()

    env = build_env_from_config(cfg)
    qkp = env.qkp
    il_cfg = float(env.config.get("qkp", {}).get("initial_level", 0.0))
    print(f"  环境自己读到的 initial_level = {il_cfg}")
    print(f"  key_ttl_steps = {env.config.get('qkp', {}).get('key_ttl_steps')}")
    n_edges = len(qkp.levels) if hasattr(qkp, "levels") else -1
    print(f"  qkp.levels 的边数 = {n_edges}")
    if hasattr(qkp, "capacities"):
        caps = list(qkp.capacities.values())
        print(f"  capacities: n={len(caps)}  min={min(caps) if caps else 0:.3g} "
              f"max={max(caps) if caps else 0:.3g}")

    obs = env.reset(seed=args.seed, start_seed=start0 + args.seed)
    print()
    print(f"  ★ reset 之后：qkp.positive 的边数 = {len(qkp.positive)}")
    print(f"    （il=0 时应为 0；>0 时应该等于全部物理边数）")
    lv = [v for v in qkp.levels.values() if v > 0]
    print(f"    有存量的边 = {len(lv)}  最小存量 = {min(lv) if lv else 0:.3g}")

    # 观察：跑一步，看 avail 与 positive 的关系
    avail = env._build_state().edge_windows.available0(sorted(qkp.levels))
    all_ids = sorted(qkp.levels)
    n_av = sum(1 for x in avail if bool(x))
    print(f"    t 处可用边 = {n_av} / {len(all_ids)}")
    pos_not_av = [e for e in all_ids if qkp.levels.get(e, 0) > 0 and not bool(
        avail[all_ids.index(e)])]
    print(f"    **有存量但当下不可用**的边 = {len(pos_not_av)}")
    if pos_not_av:
        print(f"      ⟹ 这些边**能**参与服务（服务路径不查可用性）")
        print(f"      ⟹ 且它们的存量来自 initial_level 而非生成 ⟹ "
              f"**不在任何 avail_by_t 里** ⟹ 并集上界失效")

    print()
    print("=" * 92)
    print("判读")
    print("=" * 92)
    if il_cfg > 0:
        print(f"  ⚠⚠ initial_level = {il_cfg} > 0 ⟹ **种子密钥铺在所有物理边上**")
        print("     ⟹ 「并集上界」**本来就不是上界**（专家绕过它用的是种子存量）")
        print("     ⟹ 所有天花板/空间结论都必须写明「il=0 前提」或改用含种子的口径")
    else:
        print("  ✓ initial_level = 0 ⟹ reset 后没有种子存量")
        print("     ⟹ 那么 dl=240 的「专家 0.8175 > 并集 0.8133」是**真矛盾**")
        print("     ⟹ 必须继续查我的探针（很可能是 avail 记录漏了 t）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
