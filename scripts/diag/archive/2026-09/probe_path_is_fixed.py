"""坐实一条断言：**服务的路径是环境写死的，策略完全够不着**。

## 断言（可否证）

`routing._usable_cached_path(req, qkp)` 走的是 `shortest_path(src, dst)` ——
纯拓扑最短路，由 `__init__` 里预计算的 `_next_hop` 决定，**与 qkp 状态、与 agent
激活了哪些边都无关**。所以：

  · `shortest_path(src, dst)` 在整局里**恒为同一条**（它是静态图的性质）
  · **只要**这条规范路的每一跳都还有正存量，服务就**一定**走它
    ⟹ 一条请求能不能被服务，等价于"这条**预定的**路有没有货"

⟹ 推论：`on_pending_path` 这一列无论怎么改，都只是把"这条路有没有货"重新编码一遍；
  **策略没有任何动作能把货放到一条它自己选的路上**——它只能影响被激活的那 46 条边，
  而其中落在规范路上的比例是个可测的数。

## 三个判据（都要打印输入）

① **恒定性**：对每个 (src,dst) 对，`shortest_path` 的返回值在整局里逐位不变。
   不变 ⟹ 静态；若变 ⟹ 我的断言错，停止解读。
   ★ 对照（必须咬得住）：**每次重算** `shortest_path`，逐位比对。这才会在
   "有人往 routing 里塞了状态依赖"时变红——只调一次是测不出恒定的。
② **充分性**：`_usable_cached_path` 非 None ⟺ 规范路每跳都有货。
   逐请求比对，任何一次不等都报出来。
③ **可触达性**：规范路的跳里，有多少在 agent 的**动作候选**（去重后的无向边）里。

用法：
    python3 -u probe_path_is_fixed.py --regime valid --steps 240 --seeds 100-114
"""
import importlib.util
import os
import statistics as st
import sys
from collections import deque
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
    """走真实启动器 build_config（避免 dims 不重算的假崩溃）。"""
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
    a.seed = args.seed
    a.num_updates = 1
    a.run_name = "pathprobe"
    a.device = "cpu"
    cfg = mod.build_config(a)
    if args.regime == "valid":
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
        cfg["scenario"]["time_limit"]["days"] = int(w.get("end_day", 365)) + max(
            1, -(-ep // ds))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["valid", "train"], default="valid")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds) if args.regime == "valid" else [args.seed]

    cfg = run_config(args)
    print(f"regime={args.regime}  episode_steps={cfg['env'].get('episode_steps')}  "
          f"time_limit.days={cfg['scenario']['time_limit']['days']}  "
          f"day_steps={cfg['env'].get('day_steps')}")

    # 恒定性：用一个集合记 (src,dst) -> path 的指纹
    seen = {}                 # (src,dst) -> tuple(path)
    n_vary = 0                # ★ 对照：每次重算后与首次不符的次数
    n_check = 0

    tot_req = 0               # 观测到的 (请求,步) 次
    tot_canon = 0             # 其中规范路非 None
    tot_pos_ok = 0            # 规范路每跳都有货
    mismatch_usable = 0       # ★ 判据②失败次数
    hop_hit = 0               # 规范路跳落在**动作候选**里的次数
    hop_all = 0               # 规范路跳总数

    for s in seeds:
        env = build_env_from_config(cfg)
        env.reset(seed=s, start_seed=int((cfg.get("validation", {}) or {}).get("start_seed", 0) or 0) + s
                  if args.regime == "valid" else s)
        routing = env.routing
        qkp = env.qkp
        # 动作候选（去重无向边）
        aspace = env.graph_builder.action_space
        act_edges = set()
        for node_id, cands in env.graph_builder.candidates.items():
            for a in cands:
                act_edges.add(a)
        expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        n = 0
        done = False
        obs = None
        while not done and n < args.steps:
            if obs is None:
                obs = env._build_observation()
            for req in list(env.requests.get_pending()):
                key = (req.src_gs, req.dst_gs)
                p1 = routing.shortest_path(req.src_gs, req.dst_gs)
                p2 = routing.shortest_path(req.src_gs, req.dst_gs)   # ★ 对照：重算
                n_check += 1
                if (tuple(p1) if p1 else None) != (tuple(p2) if p2 else None):
                    n_vary += 1
                if key not in seen:
                    seen[key] = tuple(p1) if p1 else None
                else:
                    if seen[key] != (tuple(p1) if p1 else None):
                        n_vary += 1
                tot_req += 1
                if p1 is None:
                    continue
                tot_canon += 1
                allpos = all(e in qkp.positive for e in p1)
                tot_pos_ok += int(allpos)
                usable = routing._usable_cached_path(req, qkp)
                if (usable is not None) != allpos:
                    mismatch_usable += 1
                for e in p1:
                    hop_all += 1
                    hop_hit += int(e in act_edges)
            acts, scores = expert.act(obs)
            obs, _r, term, trunc, _info = env.step(acts, scores)
            n += 1
            done = term or trunc

    print()
    print("=" * 96)
    print("判据① 恒定性：规范最短路在整局里是否恒定")
    print("=" * 96)
    print(f"  (src,dst) 对 {len(seen)} 个   比对 {n_check} 次   ★ 不符 {n_vary} 次")
    print(f"  {'✓ 静态（断言成立）' if n_vary == 0 else '✗ 会变 ⟹ 断言错，停止解读'}")
    print()
    print("=" * 96)
    print("判据② 充分性：_usable_cached_path 非 None ⟺ 规范路每跳都有货")
    print("=" * 96)
    print(f"  比对 {tot_canon} 次（规范路非 None 的请求）  ★ 不符 {mismatch_usable} 次")
    print(f"  {'✓ 等价' if mismatch_usable == 0 else '✗ 不等价 ⟹ 服务走的不是规范路'}")
    print()
    print("=" * 96)
    print("可触达性：规范路的跳有多少在 agent 的**动作候选**里")
    print("=" * 96)
    print(f"  规范路跳总数 {hop_all}   落在动作候选里 {hop_hit} "
          f"({hop_hit/max(hop_all,1):.2%})")
    print()
    print("=" * 96)
    print("结构读数")
    print("=" * 96)
    print(f"  (请求,步) 观测次数 {tot_req}")
    print(f"    规范路存在         {tot_canon} ({tot_canon/max(tot_req,1):.2%})")
    print(f"    规范路每跳都有货   {tot_pos_ok} ({tot_pos_ok/max(tot_req,1):.2%})  ⟸ 这些**自动**被服务")
    print(f"    规范路有跳没货     {tot_canon - tot_pos_ok} "
          f"({(tot_canon-tot_pos_ok)/max(tot_req,1):.2%})  ⟸ 这些服务量恒 0（v1 的列对它们全 0）")
    print(f"    无规范路（孤岛）   {tot_req - tot_canon} "
          f"({(tot_req-tot_canon)/max(tot_req,1):.2%})  ⟸ 谁都救不了")
    print()
    if n_vary == 0 and mismatch_usable == 0:
        print("DECISION=PATH_IS_FIXED  路径由环境写死、策略够不着")
    else:
        print("DECISION=ABORT  判据没过 ⟹ 不给结论")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
