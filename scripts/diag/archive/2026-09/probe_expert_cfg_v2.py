"""专家在「profile 路径」与「训练器**真实**验证配置」下是不是同一个数？

## 这是个更正版 —— 前一版归因错了

前一版 `probe_expert_config_paths.py` 报「两条路径差 8 个点 ⟹ 专家锚必须重测」，
并把嫌疑指向 `include_on_pending_path`。那是**错的**：

  逐键 diff（`probe_config_diff.py`）显示 `seed`/`requests`/`rate` 三桶**全空**，
  而 `env` 桶有：
      env.activation_window_start_day   profile=330   前一版的"训练链"=0
      env.activation_window_end_day     profile=365   前一版的"训练链"=295
      env.episode_steps                 profile=240   前一版的"训练链"=1440

  前一版**没用训练器真实的验证配置**，而是自己手搓了几个 env 覆盖，**漏了窗口**
  ⟹ 那一跑其实在**训练窗口 0–295** 上采起日。8 个点是**两个 regime 的差**，
  与特征开关无关。（`confound-must-check-branch-polarity-not-diff` 的同族错：
  把一个 confound 归给了错误的变量。）

## 这一版怎么做

三份配置，逐种子跑同一个专家：

  A  profile 路径          = `test_protocol.build_validation_env_config`
  B  训练器**真实**验证配置 = 逐行复刻 `mappo_trainer._build_validation_envs`
                            （用真实训练链 + 真实 validation 段）
  C  **正对照**            = B，但把激活窗口强行换回训练窗口 0–295

判据（★ 三态，缺一不可）：
  ① A == B **逐位相同** ⟹ 专家锚与训练器内部数可直接比较
  ② C != B（必须有差）   ⟹ 探针**咬得住**窗口这个变量；否则①是空真
     （`unavailable-must-not-look-like-a-value`：证明"相同"之前，先证明
     被比的两样东西**本可以不同**）

用法：
    python3 -u probe_expert_cfg_v2.py --seeds 100-102
"""
import argparse
import copy
import importlib.util
import math
import sys
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cfg_profile():
    tp = load_mod("gm_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
    profile = tp.load_validation_profile(REPO / "configs" / "global.yaml")
    return tp.build_validation_env_config(profile, include_baselines=True)


def cfg_trainer_real():
    """逐行复刻 mappo_trainer._build_validation_envs 的配置构造。"""
    train = load_mod("gm_train", REPO / "scripts" / "train" / "train_graph_mappo.py")

    class A:
        pass

    a = A()
    a.mode = "random_episode"
    a.configs = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml",
                 "train_v2_bottleneck.yaml"]
    a.seed = 42
    a.num_updates = 1
    a.run_name = "expcfg"
    a.device = "cpu"
    base = train.build_config(a)

    v = base.get("validation", {}) or {}
    window = v.get("window", {}) or {}
    start_day = int(window.get("start_day", 0))
    end_day = int(window.get("end_day", 365))
    seeds = [int(s) for s in (v.get("request_seeds", []) or [])]
    episode_steps = int(v.get("episode_steps", 0) or 0)
    if episode_steps <= 0:
        episode_days = int(v.get("episode_days", 1) or 1)
        episode_steps = episode_days * int(base["env"].get("day_steps", 1440))
    day_steps = int(base["env"].get("day_steps", 1440))

    c = copy.deepcopy(base)
    c["env"]["episode_start_mode"] = str(v.get("start_mode", "random_day"))
    c["env"]["episode_steps"] = episode_steps
    c["env"]["continuous"] = False
    c["env"]["activation_window_start_day"] = start_day
    c["env"]["activation_window_end_day"] = end_day
    c["env"]["activation_window_days"] = max(0, end_day - start_day)
    c["scenario"]["time_limit"]["days"] = end_day + max(1, math.ceil(episode_steps / day_steps))
    c["seed"]["env_seed"] = seeds[0]
    return c, episode_steps


def run_expert(cfg, seeds, episode_steps):
    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.baselines.path_greedy import PathScoreGreedy
    from qkd_rl.baselines.serve_probe import ServeProbe

    out = {}
    for s in seeds:
        env = build_env_from_config(copy.deepcopy(cfg))
        env.reset(seed=s, start_seed=s)
        expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        obs = env._build_observation()
        n, done = 0, False
        while not done and n < episode_steps:
            acts, scores = expert.act(obs)
            obs, _r, term, trunc, _info = env.step(acts, scores)
            n += 1
            done = term or trunc
        m = env.metrics.episode_summary()
        arr = float(m.get("arrived_keys", 0.0))
        out[s] = float(m.get("served_keys", 0.0)) / arr if arr else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-102")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    A = cfg_profile()
    B, eps = cfg_trainer_real()
    C = copy.deepcopy(B)
    C["env"]["activation_window_start_day"] = 0
    C["env"]["activation_window_end_day"] = 295
    C["env"]["activation_window_days"] = 295
    C["scenario"]["time_limit"]["days"] = 295 + max(1, math.ceil(eps / 1440))

    print("=" * 96)
    print(f"三份配置的 env 段（episode_steps={eps}）")
    print("=" * 96)
    for nm, c in (("A profile", A), ("B 训练器真实", B), ("C 正对照(窗口回训练)", C)):
        e = c["env"]
        print(f"  {nm:<18} start_mode={e.get('episode_start_mode'):<12} "
              f"steps={e.get('episode_steps'):<6} "
              f"win={e.get('activation_window_start_day')}-{e.get('activation_window_end_day')}"
              f"  edge_dim={c['features']['dims'].get('edge_dim_resolved')}")

    print()
    print(f"跑专家，种子 {seeds}")
    ra = run_expert(A, seeds, eps)
    rb = run_expert(B, seeds, eps)
    rc = run_expert(C, seeds, eps)

    print()
    print("=" * 96)
    print(f"{'种子':>6}{'A profile':>14}{'B 训练器真实':>16}{'A-B':>12}"
          f"{'C 正对照':>14}{'C-B':>12}")
    print("-" * 96)
    nAB = nCB = 0
    for s in seeds:
        dAB = ra[s] - rb[s]
        dCB = rc[s] - rb[s]
        nAB += int(abs(dAB) > 1e-12)
        nCB += int(abs(dCB) > 1e-12)
        print(f"{s:>6}{ra[s]:>14.6f}{rb[s]:>16.6f}{dAB:>+12.2e}"
              f"{rc[s]:>14.6f}{dCB:>+12.2e}")
    print("-" * 96)
    print(f"  A vs B 不同的种子：{nAB}/{len(seeds)}")
    print(f"  C vs B 不同的种子：{nCB}/{len(seeds)}   ⟸ 正对照，必须 >0")
    print()
    ctrl_ok = nCB > 0
    print(f"① A == B 逐位相同     {'✓' if nAB == 0 else '★ 不同'}  "
          f"{'专家锚与训练器内部数可比' if nAB == 0 else '⟹ 专家锚要重测'}")
    print(f"② 正对照咬得住       {'✓' if ctrl_ok else '★ 咬不住 ⟹ ①是空真'}")
    print()
    if nAB == 0 and ctrl_ok:
        print("DECISION=SAME  两条路径给出同一个专家读数 ⟹ 专家锚可直接与 v2 的 Δ 比较")
        return 0
    print("DECISION=DIFFERENT")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
