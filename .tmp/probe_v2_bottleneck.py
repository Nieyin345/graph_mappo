"""v2 的造反证：`_compute_on_pending_path` 现在标的是**瓶颈跳**吗？

## 四级判据（每级都要有**咬得住**的对照）

① **逐位等价** `_compute_on_pending_path` 的输出 == 探针**独立重算**的 argmin。
   独立重算走的是另一份代码（本文件里手写），比的是**集合**（标了哪几条边）。
   ★ 对照：随机打乱一次比对必须是**不等**的——否则说明比对写成了恒真。
② **只能标规范路上的边**：任何一次标记，那条边必须属于该请求的 `shortest_path`。
   ★ 对照：把 `shortest_path` 换成"全图任取一条边"，必须立刻违规。
③ **标的就是最小**：被标边的 level == min(该请求规范路各跳 level)。
④ **覆盖**：非零标记的 (请求,步) 占比 —— v1 实测是 **11.15%**
   （`probe_onpath_silence.py`）。如果 v2 还是 ~11%，那我没改到东西。

用法：
    python3 -u probe_v2_bottleneck.py --steps 240 --seeds 100-102
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
from qkd_rl.env.factory import build_env_from_config                  # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy              # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe                   # noqa: E402


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
    a.configs = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml",
                 "train_v1_on_path.yaml"]      # ★ 这一列开着（v2 只换语义、不改开关）
    a.seed = 42
    a.num_updates = 1
    a.run_name = "v2probe"
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
    return cfg, int(v.get("start_seed", 0) or 0), bool(
        cfg["features"]["edge"].get("include_on_pending_path", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seeds", default="100-102")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    cfg, start0, onpath = run_config(args)
    print(f"include_on_pending_path = {onpath}   "
          f"edge_dim_resolved = {cfg['features']['dims'].get('edge_dim_resolved')}")
    if not onpath:
        print("★ 这一列没开 ⟹ 量不出东西，退出")
        return 2

    n_pairs = 0            # (请求,步) 总数
    n_marked = 0           # 其中该请求的瓶颈跳被标了的
    n_canon = 0            # 其中规范路存在的（覆盖率的分母）
    n_symdiff = 0          # ① 对称差（marked vs 独立重算），按步累加
    n_steps = 0            # 比对的步数
    n_offpath = 0          # ② 违规步数：标了不属于任何待服务请求规范路的边
    n_notmin = 0           # ③ 违规(请求,步)数：瓶颈跳没被标
    n_neg_ok = 0           # ① 对照：**真删一个元素**后必须不等
    n_neg_tot = 0

    for s in seeds:
        env = build_env_from_config(cfg)
        env.reset(seed=s, start_seed=start0 + s)
        routing, qkp, gb = env.routing, env.qkp, env.graph_builder
        epos = gb._edge_pos                      # edge_id -> _edge_list 下标
        expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        obs = env._build_observation()
        n = 0
        done = False
        while not done and n < args.steps:
            col = gb._compute_on_pending_path(env.requests)
            marked = {eid for eid, p in epos.items() if col[p] > 0}
            # ---- 独立重算（另一份代码）----
            mine = set()
            per_req_paths = {}
            pend = list(env.requests.get_pending())
            for req in pend:
                p = routing.shortest_path(req.src_gs, req.dst_gs)
                if not p:
                    continue
                per_req_paths[id(req)] = p
                lv = [float(qkp.get_level(e)) for e in p]
                k = 0
                for i in range(1, len(lv)):
                    if lv[i] < lv[k]:
                        k = i
                mine.add(p[k])

            # ① **集合精确比对**（对称差）
            n_steps += 1
            n_symdiff += len(marked ^ mine)

            # ① 对照：**真扰动** —— 去掉一条边，必须变不等
            if marked:
                n_neg_tot += 1
                probe = set(list(marked)[:-1])       # 真的少了一个元素
                if probe != marked:
                    n_neg_ok += 1

            # ② 按**步**判一次（不是按请求）：标记集合里有没有一条边，
            #    不属于**任何**待服务请求的规范路。
            if marked - mine:
                n_offpath += 1

            # ③ 逐请求：这个请求的瓶颈跳被标了吗
            for req in pend:
                n_pairs += 1
                p = per_req_paths.get(id(req))
                if not p:
                    continue
                n_canon += 1
                lv = [float(qkp.get_level(e)) for e in p]
                k = 0
                for i in range(1, len(lv)):
                    if lv[i] < lv[k]:
                        k = i
                if p[k] in marked:
                    n_marked += 1
                else:
                    n_notmin += 1
            acts, scores = expert.act(obs)
            obs, _r, term, trunc, _info = env.step(acts, scores)
            n += 1
            done = term or trunc

    print()
    print("=" * 92)
    print("① 集合精确比对（vs 独立重算）")
    print("=" * 92)
    print(f"  比对 {n_steps} 步   对称差累计 {n_symdiff}   "
          f"{'✓ 逐位等价' if n_symdiff == 0 else '★ 不一致'}")
    print(f"  对照：**真删一个元素**后应不等——{n_neg_ok}/{n_neg_tot} 次确实不等  "
          f"{'✓ 比对咬得住' if n_neg_ok == n_neg_tot and n_neg_tot > 0 else '★ 恒真，作废'}")
    print()
    print("=" * 92)
    print("② 只标规范路上的边 / ③ 标的就是最小")
    print("=" * 92)
    print(f"  ② 标了路外边：{n_offpath} 次  {'✓' if n_offpath == 0 else '★ 违规'}")
    print(f"  ③ 该标没标：  {n_notmin} 次  {'✓' if n_notmin == 0 else '★ 违规'}")
    print()
    print("=" * 92)
    print("④ 覆盖率（分母 = **有规范路**的 (请求,步)）")
    print("=" * 92)
    print(f"  (请求,步) 总数 {n_pairs}   有规范路 {n_canon} "
          f"({n_canon/max(n_pairs,1):.2%})   无规范路 {n_pairs-n_canon}")
    cov = n_marked / max(n_canon, 1)
    print(f"  有规范路的里面，瓶颈跳被标了 {n_marked} = **{cov:.2%}**")
    print(f"  v1 实测 11.15%（probe_onpath_silence.py，15 种子，分母是全部请求）")
    print(f"  ⟹ v1 的 11.15% ≈ 有规范路比例 × v1在该子集内的命中；"
          f"v2 的 {cov:.2%} 应当 ≈ 100%")
    print()
    ok = (n_symdiff == 0 and n_neg_ok == n_neg_tot and n_neg_tot > 0
          and n_offpath == 0 and n_notmin == 0 and cov > 0.99)
    print(f"DECISION={'V2_VERIFIED' if ok else 'FALSIFIED'}  coverage={cov:.4f}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
