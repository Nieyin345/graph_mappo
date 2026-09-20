"""规范路的「有货跳占比」分布 + 动作候选的正确口径。

两个 0.00% 要分开处理：

① 「规范路跳落在动作候选里 = 0.00%」—— **我比的错了**：
   `candidates_for_node()` 返回**邻接节点 id**（`action_space.py:80-82`），
   不是边 id。拿边 id 去比节点 id 必然是 0。要用 `action_to_edge` 映射。

② 「规范路每跳都有货 = 0.00%（0/3163）」—— 这个 0 **太整齐**，
   独立事件不该恰好一次都不发生 ⟹ 先当**方法错**查（读异常先怪方法）。
   这里打印**有货跳占比的分布** + `|qkp.positive|` 随步的变化。

   若分布显示有相当比例的路是 100% 有货，则 ② 是我的 bug；
   若分布的上界就够不到 1.0，那 0.00% 是真的、且就是**核心结构读数**。

用法：
    python3 -u probe_canon_stock.py --steps 240 --seeds 100-102
"""
import importlib.util
import os
import statistics as st
import sys
from pathlib import Path

_here = sys.path[0] if sys.path else ""
if _here and _here not in ("", "."):
    sys.path[:] = [p for p in sys.path if p != _here]

REPO = Path(os.environ.get("GM_REPO", "/opt/qkd/graph_mappo"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse                                                      # noqa: E402
from qkd_rl.env.factory import build_env_from_config                  # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy              # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe                   # noqa: E402


def _tp():
    spec = importlib.util.spec_from_file_location(
        "gm_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
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


def run_config(args):
    sys.argv = ["train_graph_mappo.py"]
    spec = importlib.util.spec_from_file_location(
        "gm_train", REPO / "scripts" / "train" / "train_graph_mappo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class A:
        pass

    a = A()
    a.mode = "random_episode"
    a.configs = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"]
    a.seed = 42
    a.num_updates = 1
    a.run_name = "canonprobe"
    a.device = "cpu"
    cfg = mod.build_config(a)
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
    return cfg, int(v.get("start_seed", 0) or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seeds", default="100-102")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    cfg, start0 = run_config(args)

    frac_hist = {}            # 桶（十分位）-> 计数
    all_stats = []            # 每条 (请求,步) 的 (有货跳数, 总跳数)
    pos_by_step = []
    act_hit = act_tot = 0
    act_edges = set()

    for s in seeds:
        env = build_env_from_config(cfg)
        env.reset(seed=s, start_seed=start0 + s)
        routing, qkp = env.routing, env.qkp
        gb = env.graph_builder
        # ★ 正确口径：把 (node, neighbour) 映射成无向边 id
        for node_id, cands in gb.candidates.items():
            for a in cands:
                e = gb.action_space.action_to_edge(node_id, a)
                if e is not None:
                    act_edges.add(e)
        expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        obs = env._build_observation()
        n = 0
        done = False
        while not done and n < args.steps:
            pos_by_step.append(len(qkp.positive))
            for req in list(env.requests.get_pending()):
                p = routing.shortest_path(req.src_gs, req.dst_gs)
                if not p:
                    continue
                k = sum(1 for e in p if e in qkp.positive)
                all_stats.append((k, len(p)))
                frac = k / len(p)
                b = min(int(frac * 10), 9)
                frac_hist[b] = frac_hist.get(b, 0) + 1
                for e in p:
                    act_tot += 1
                    act_hit += int(e in act_edges)
            acts, scores = expert.act(obs)
            obs, _r, term, trunc, _info = env.step(acts, scores)
            n += 1
            done = term or trunc

    print("=" * 96)
    print(f"规范路的「有货跳占比」分布   种子 {seeds[0]}–{seeds[-1]}   "
          f"样本 {len(all_stats)}")
    print("=" * 96)
    tot = sum(frac_hist.values())
    for b in range(10):
        c = frac_hist.get(b, 0)
        bar = "#" * int(60 * c / max(tot, 1))
        print(f"  [{b/10:.1f},{(b+1)/10:.1f})  {c:>6}  {c/max(tot,1):>7.2%}  {bar}")
    full = frac_hist.get(9, 0)     # 占比==1.0 落在 9 号桶
    print(f"  其中 100% 有货（桶 9 的全部）：{full} ({full/max(tot,1):.2%})")
    print()
    ks = [k for k, _ in all_stats]
    ls = [l for _, l in all_stats]
    print(f"  有货跳数  mean {st.mean(ks):.2f} / 总跳数 mean {st.mean(ls):.2f}"
          f"  ⟹ 平均有货占比 {st.mean(ks)/max(st.mean(ls),1e-9):.2%}")
    print()
    print("=" * 96)
    print("|qkp.positive| 随步变化（如果有货边一直很少，上面的 0 可能是真的）")
    print("=" * 96)
    for i in range(0, len(pos_by_step), max(1, len(pos_by_step) // 12)):
        print(f"  步 {i:>4}   |positive| = {pos_by_step[i]}")
    print(f"  末步 |positive| = {pos_by_step[-1] if pos_by_step else 0}"
          f"   max = {max(pos_by_step) if pos_by_step else 0}")
    print()
    print("=" * 96)
    print("动作候选（★ 已修正口径：node -> 无向边）")
    print("=" * 96)
    print(f"  候选无向边 {len(act_edges)} 条")
    print(f"  规范路跳总数 {act_tot}  落在候选里 {act_hit} ({act_hit/max(act_tot,1):.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
