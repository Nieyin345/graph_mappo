"""造反证：v2 的代码改动对**基线配置**是惰性的吗？

## 为什么必须测这个

`v1_onpath_s*` 与 `ent01_rerun_s*` 是 2026-09-20 01:2x 之前跑的，服务器上的
代码 HEAD 是本地工作区在 **02:16Z** 的快照；v2 的补丁（`graph_builder.py` 重写
`_compute_on_pending_path` + `factory.py` 传 `routing`）是**之后**打的。

⟹ 如果 v2 的改动**碰得到**基线路径，那么拿 `ent01_rerun_s*` 当 v2 的对照就是
**跨代码快照的伪 A/B**（`ab-pair-must-share-the-code-not-just-the-config`）。

## 静态论证（先写下来，再用实验去撞）

`_compute_on_pending_path` 全仓只有**一个**调用点（`graph_builder.py:496`），
且被 `if edge_cfg.get("include_on_pending_path", False):` 包着。基线链里这个开关
是 **false** ⟹ 函数对象**从不被调用**，函数体怎么改都无关。
签名加的是**带默认值的关键字参数**，`self.routing` 只在那个函数体里被读。

⟹ 静态上应当惰性。但静态论证正是"我读了代码所以我认为"——**要用实验撞**。

## 实验设计（正对照 + 反证，缺一不可）

把 `GraphBuilder._compute_on_pending_path` **替换成会抛异常的函数**：

  · 基线配置（开关 false）跑 20 步 ⟹ **必须不抛**（否则 v2 碰得到基线，对照作废）
  · v1 配置（开关 true）跑 20 步 ⟹ **必须抛**（正对照；不抛说明炸弹是哑的，
    上面那个"不抛"就什么都没证明）

★ 两个配置都走**真实启动器** `build_config()`（`test-harness-must-use-real-launch-path`：
在 build 之后 mutate 配置 ⟹ dims 不重算 ⟹ 假崩溃）。

用法：
    python3 -u probe_v2_inert.py
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

import argparse                                                      # noqa: E402


class Bomb(Exception):
    pass


_TRAIN_MOD = None


def _train_mod():
    """★ 只 exec 一次：`train_graph_mappo.py` 模块级有
    `torch.set_num_interop_threads(2)`，同进程第二次 exec 必抛
    （`untestable-module-level-side-effect`）。缓存住。"""
    global _TRAIN_MOD
    if _TRAIN_MOD is not None:
        return _TRAIN_MOD
    sys.argv = ["train_graph_mappo.py"]
    spec = importlib.util.spec_from_file_location(
        "gm_train", REPO / "scripts" / "train" / "train_graph_mappo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _TRAIN_MOD = mod
    return mod


def build(cfg_extra):
    mod = _train_mod()

    class A:
        pass

    a = A()
    a.mode = "random_episode"
    a.configs = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"] + cfg_extra
    a.seed = 42
    a.num_updates = 1
    a.run_name = "v2inert"
    a.device = "cpu"
    return mod.build_config(a)


def run_steps(cfg, steps):
    """跑几步；返回 (是否抛了 Bomb, 异常信息)。"""
    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.baselines.path_greedy import PathScoreGreedy
    from qkd_rl.baselines.serve_probe import ServeProbe

    v = cfg.get("validation", {}) or {}
    cfg["env"]["episode_start_mode"] = str(v.get("start_mode", "random_day"))
    cfg["env"]["episode_steps"] = int(v.get("episode_steps", 240))
    cfg["env"]["continuous"] = False
    w = v.get("window", {}) or {}
    cfg["env"]["activation_window_start_day"] = int(w.get("start_day", 330))
    cfg["env"]["activation_window_end_day"] = int(w.get("end_day", 365))
    cfg["env"]["activation_window_days"] = max(
        0, int(w.get("end_day", 365)) - int(w.get("start_day", 330)))
    ds = int(cfg["env"].get("day_steps", 1440))
    ep = int(cfg["env"]["episode_steps"])
    cfg["scenario"]["time_limit"]["days"] = int(w.get("end_day", 365)) + max(1, -(-ep // ds))

    env = build_env_from_config(cfg)
    env.reset(seed=100, start_seed=int(v.get("start_seed", 0) or 0) + 100)
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    obs = env._build_observation()
    for _ in range(steps):
        acts, scores = expert.act(obs)
        obs, _r, term, trunc, _info = env.step(acts, scores)
        if term or trunc:
            break
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    from qkd_rl.env.graph_builder import GraphBuilder

    real = GraphBuilder._compute_on_pending_path

    def bomb(self, requests):
        raise Bomb("_compute_on_pending_path 被调用了")

    results = {}
    for label, extra in (("baseline(开关 false)", []),
                         ("v1(开关 true)", ["train_v1_on_path.yaml"])):
        cfg = build(extra)
        flag = bool(cfg["features"]["edge"].get("include_on_pending_path", False))
        dim = cfg["features"]["dims"].get("edge_dim_resolved")
        GraphBuilder._compute_on_pending_path = bomb      # ★ 装炸弹
        try:
            run_steps(cfg, args.steps)
            outcome, msg = "不抛", ""
        except Bomb as e:
            outcome, msg = "抛了", str(e)
        except Exception as e:                            # noqa: BLE001
            outcome, msg = "抛了别的", f"{type(e).__name__}: {e}"
        finally:
            GraphBuilder._compute_on_pending_path = real  # 拆弹（防污染下一轮）
        results[label] = outcome
        print(f"  {label:<22} include_on_pending_path={str(flag):<5} "
              f"edge_dim={dim}  →  {outcome}  {msg}")

    print()
    print("=" * 88)
    base_ok = results["baseline(开关 false)"] == "不抛"
    ctrl_ok = results["v1(开关 true)"] != "不抛"
    print(f"反证：基线**不抛**   {'✓ 惰性成立' if base_ok else '★ 碰得到基线 ⟹ 对照作废'}")
    print(f"正对照：v1 **抛了**  {'✓ 炸弹是响的' if ctrl_ok else '★ 炸弹哑了 ⟹ 上面那句什么都没证明'}")
    print()
    if base_ok and ctrl_ok:
        print("DECISION=INERT   v2 对基线配置无影响 ⟹ ent01_rerun_s* 可作对照")
        return 0
    print("DECISION=NOT_INERT")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
