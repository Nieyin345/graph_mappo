"""诊断：失败请求的解剖 —— 损失来自哪里？（2026-09-27）

动机：神谕预测实验表明决策时预测质量不是瓶颈（实验日志 §11.11）。要让注意力真正起作用，
得先知道 RL 丢掉的密钥量主要是哪一类失败造成的。

对每个到期未完成的请求（剩余量 r>0），在它到期那一刻（serve 之后）看它的**静态最短路**：
  - 各跳库存；库存 < r 的跳 = 缺货跳 Z
  - 对每个缺货跳，统计请求生命周期 [arrival_t, deadline_t) 内：
      vis  可见（在 obs.physical_edge_ids 中）的时隙数
      act  被激活的时隙数
      free 可见、未激活、且两端节点激活度都 <2（每节点 1 Tx + 1 Rx，端口没占满）的时隙数
按剩余密钥量加权归类（取最"严重"的缺货跳，优先级从上到下）：
  unroutable     生命周期内（逐步）可见边 ∪ 有库存边组成的图不连通 → 任何策略都救不了
  no_zero_hop    最短路各跳都有货（走了别的路或被别人抢光后仍有残量）
  never_visible  某缺货跳整个生命周期都不可见、也从未有货
  stock_competed 缺货跳期间不可见，但生命周期内有过存货 → 存货被其他请求用光（竞争）
  omitted_free   缺货跳可见过、从未激活、且存在端口空闲的时隙 → 纯"漏选"
  omitted_busy   缺货跳可见过、从未激活、可见时端口总被占 → 端口冲突
  activated_drained  缺货跳激活过，但库存被消耗完（竞争/量不够）
unroutable 再细分：并上到达前 W∈{30,60,120,240,480,回合起} 步内可见过的边后能否连通 → 可"提前囤货"救回的上限。
另记：缺货跳个数分布、路径上是否有部分跳已 ≥ r。
参照路径（2026-09-27 修正）：静态最短路不合适（静态图 1948 条边，最短路通常经一颗此刻不可见的卫星）。
改为在"生命周期内可见过的边 ∪ 到期时有库存的边"上，用 0-1 BFS 找**缺货跳（库存 < r）最少**的路径。

用法：
  python scripts/diag/failure_anatomy.py --run outputs/experiments/v8/load2x_b7_noattbias_s42 \
      --policy rl|expert --seeds 100-104 --out outputs/eval/anatomy_xxx.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.env.factory import build_env_from_config  # noqa: E402

EPS = 1.0e-9
PRESTOCK_WINDOWS = (30, 60, 120, 240, 480, 1 << 30)  # 1<<30 = 回合开始以来
CLASSES = ["unroutable", "no_zero_hop", "never_visible", "stock_competed", "omitted_free",
           "omitted_busy", "activated_drained"]


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def _load_eval_expert():
    spec = importlib.util.spec_from_file_location("eval_expert", ROOT / "scripts" / "eval" / "eval_expert.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def min_deficit_path(adj, src, dst, usable, cost):
    """0-1 BFS：只走 usable 中的边，最小化缺货跳数（同代价时跳数少者优先）。"""
    from collections import deque
    best = {src: (0, 0)}
    parent = {src: (None, None)}
    dq = deque([(0, 0, src)])
    while dq:
        c, h, u = dq.popleft()
        if best.get(u, (1 << 30, 0)) < (c, h):
            continue
        if u == dst:
            path = []
            while parent[u][0] is not None:
                path.append(parent[u][1])
                u = parent[u][0]
            return path[::-1]
        for v, e in adj.get(u, ()):
            if e not in usable:
                continue
            w = cost(e)
            cand = (c + w, h + 1)
            if cand < best.get(v, (1 << 30, 1 << 30)):
                best[v] = cand
                parent[v] = (u, e)
                (dq.appendleft if w == 0 else dq.append)((c + w, h + 1, v))
    return None


class Recorder:
    """挂在 env 上：逐步记录可见/激活集合，并在 expire() 时快照失败请求的路径库存。"""

    def __init__(self, env):
        self.env = env
        self.t_list: list[int] = []
        self.visible: list[set[str]] = []
        self.activated: list[set[str]] = []
        self.stocked: list[set[str]] = []  # 每步开始时有库存的边
        self.failed: list[dict] = []
        orig_expire = env.requests.expire

        def expire(t, _orig=orig_expire):
            out = _orig(t)
            for req in out:
                rem = max(0.0, req.amount - req.served_amount)
                if rem <= EPS:
                    continue
                levels = {e: float(lv) for e, lv in env.qkp.levels.items() if lv > EPS}
                self.failed.append({"req": req, "t": int(t), "rem": rem, "levels": levels})
            return out

        env.requests.expire = expire

    def before_step(self, obs):
        self.t_list.append(int(self.env.t))
        self.visible.append(set(obs.physical_edge_ids or []))
        self.stocked.append(set(self.env.qkp.positive))

    def after_step(self):
        self.activated.append(set(self.env.last_activated_edges or []))

    def classify(self) -> list[dict]:
        edges = self.env.qkp.edges
        idx_of = {t: i for i, t in enumerate(self.t_list)}
        degree: list[Counter] = []
        for act in self.activated:
            c: Counter = Counter()
            for e in act:
                c[edges[e].src] += 1
                c[edges[e].dst] += 1
            degree.append(c)
        rows = []
        adj = self.env.routing.adj
        for f in self.failed:
            req, levels, rem = f["req"], f["levels"], f["rem"]
            steps = [idx_of[t] for t in range(int(req.arrival_t), int(req.deadline_t)) if t in idx_of]
            usable = set(levels)
            for i in steps:
                usable |= self.visible[i]
                usable |= self.stocked[i]
            path = min_deficit_path(adj, req.src_gs, req.dst_gs, usable,
                                    lambda e: 0 if levels.get(e, 0.0) >= rem else 1)
            row = {"rem": rem, "amount": float(req.amount), "n_hops": None, "n_zero": None,
                   "others_ready": None}
            if path is None:
                row["cls"] = "unroutable"
                # 能否靠"提前囤货"救回：并上到达前 W 步内可见过的边后是否连通
                i0 = steps[0] if steps else len(self.t_list)
                row["prestock_w"] = None
                extra = set(usable)
                prev = i0
                for w in PRESTOCK_WINDOWS:
                    lo = max(0, i0 - w)
                    for i in range(lo, prev):
                        extra |= self.visible[i]
                    prev = lo
                    if min_deficit_path(adj, req.src_gs, req.dst_gs, extra, lambda e: 0) is not None:
                        row["prestock_w"] = w
                        break
                rows.append(row)
                continue
            lv = [levels.get(e, 0.0) for e in path]
            row["n_hops"] = len(path)
            zero = [e for e, x in zip(path, lv) if x < rem]
            row["n_zero"] = len(zero)
            row["others_ready"] = bool(zero) and len(zero) < len(path)
            if not zero:
                row["cls"] = "no_zero_hop"
                rows.append(row)
                continue
            worst = None
            rank = {c: i for i, c in enumerate(CLASSES)}
            hop_stats = []
            for e in zero:
                vis = sum(1 for i in steps if e in self.visible[i])
                act = sum(1 for i in steps if e in self.activated[i])
                src, dst = edges[e].src, edges[e].dst
                free = sum(1 for i in steps if e in self.visible[i] and e not in self.activated[i]
                           and degree[i][src] < 2 and degree[i][dst] < 2)
                if vis == 0:
                    c = "stock_competed" if any(e in self.stocked[i] for i in steps) else "never_visible"
                elif act == 0:
                    c = "omitted_free" if free > 0 else "omitted_busy"
                else:
                    c = "activated_drained"
                hop_stats.append({"vis": vis, "act": act, "free": free, "cls": c,
                                  "type": edges[e].link_type.value})
                if worst is None or rank[c] < rank[worst]:
                    worst = c
            row["cls"] = worst
            row["zero_hops"] = hop_stats
            rows.append(row)
        return rows


def run_rl(run: Path, seeds: list[int], threads: int):
    from qkd_rl.core.config import load_config
    from qkd_rl.evaluation.test_protocol import run_validation_env_config
    from qkd_rl.model_zoo import build_model
    from qkd_rl.rl.algos.checkpoint import load_checkpoint
    from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
    from qkd_rl.rl.algos.policy import DETERMINISTIC_RESOLVER_MODES, MAPPOPolicy

    torch.set_num_threads(threads)
    config = load_config([run / "experiment_config.yaml"])
    vcfg = config.get("validation", {}) or {}
    env_cfg = run_validation_env_config(config, vcfg)
    val_seeds = [int(s) for s in (vcfg.get("request_seeds") or [])]
    env_cfg["seed"]["env_seed"] = val_seeds[0] if val_seeds else seeds[0]
    start_base = int(vcfg.get("start_seed", 0) or 0)
    env = build_env_from_config(env_cfg)
    model = build_model(env.action_resolver.action_space, config)
    model.load_state_dict(load_checkpoint(run / "checkpoint_final.pt", "cpu").model_state)
    model.eval()
    policy = MAPPOPolicy(model, "cpu")
    trainer = MAPPOTrainer(env, policy, config, run, device="cpu")
    needs_scores, resolver_mode = trainer._needs_edge_scores, trainer.resolver_mode
    for seed in seeds:
        obs = env.reset(seed=seed, start_seed=start_base + seed)
        rec = Recorder(env)
        done = False
        while not done:
            rec.before_step(obs)
            with torch.no_grad():
                out = policy.act_batched([obs], deterministic=True, build_scores=needs_scores,
                                         use_edge_arrays=not needs_scores)[0]
            obs, _r, te, tr, _i = env.step(
                out.actions, out.action_scores, edge_scores=out.edge_scores,
                expected_matched_edges=(None if resolver_mode in DETERMINISTIC_RESOLVER_MODES
                                        else list(out.matched_edges or [])))
            rec.after_step()
            done = te or tr
        del env.requests.expire  # 恢复类方法（reset 不重建 requests 时防止叠加包装）
        yield seed, env, rec


def run_expert(run: Path, seeds: list[int], overlays: list[str]):
    ee = _load_eval_expert()
    config, _s, start_base = ee.build_expert_task(str(run), "", "", 0, overlays)
    for seed in seeds:
        env = build_env_from_config(config)
        obs = env.reset(seed=seed, start_seed=start_base + seed)
        expert = ee.build_policy("expert", env, obs, config)
        rec = Recorder(env)
        done = False
        while not done:
            rec.before_step(obs)
            actions, scores = expert.act(obs)
            obs, _r, te, tr, _i = env.step(actions, scores)
            rec.after_step()
            done = te or tr
        yield seed, env, rec


def summarize(rows: list[dict], arrived: float) -> dict:
    by = defaultdict(float)
    cnt = Counter()
    for r in rows:
        by[r["cls"]] += r["rem"]
        cnt[r["cls"]] += 1
    lost = sum(by.values())
    nz = Counter(r["n_zero"] for r in rows if r["n_zero"])
    nz_keys = defaultdict(float)
    for r in rows:
        if r["n_zero"]:
            nz_keys[min(r["n_zero"], 3)] += r["rem"]
    ready = sum(r["rem"] for r in rows if r.get("others_ready"))
    hop_types = defaultdict(float)
    for r in rows:
        for h in r.get("zero_hops", []):
            hop_types[f'{h["type"]}:{h["cls"]}'] += r["rem"] / max(1, len(r["zero_hops"]))
    # 缺货跳的时隙利用：act/vis（可见时隙里开了多少）、free/vis（可见、没开、端口空闲）、
    # busy/vis（可见、没开、端口被占）。按"该跳分到的丢失量"加权。
    hop_use = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # cls -> [w, w*act/vis, w*free/vis, w*busy/vis]
    for r in rows:
        hs = r.get("zero_hops", [])
        for h in hs:
            if h["vis"] <= 0:
                continue
            w = r["rem"] / len(hs)
            acc = hop_use[h["cls"]]
            acc[0] += w
            acc[1] += w * h["act"] / h["vis"]
            acc[2] += w * h["free"] / h["vis"]
            acc[3] += w * (h["vis"] - h["act"] - h["free"]) / h["vis"]
    hop_use_out = {c: {"weight_share": a[0] / lost if lost else 0.0, "act_over_vis": a[1] / a[0],
                       "free_over_vis": a[2] / a[0], "busy_over_vis": a[3] / a[0]}
                   for c, a in hop_use.items() if a[0] > 0}
    pre = defaultdict(float)
    for r in rows:
        if r["cls"] == "unroutable":
            w = r.get("prestock_w")
            pre["never" if w is None else ("episode" if w >= (1 << 30) else str(w))] += r["rem"]
    return {
        "unroutable_prestock_share_of_arrived": {k: v / arrived for k, v in pre.items()} if arrived else {},
        "hop_slot_use": hop_use_out,
        "arrived": arrived, "lost_keys": lost, "lost_share_of_arrived": lost / arrived if arrived else 0.0,
        "class_share_of_lost": {c: by[c] / lost if lost else 0.0 for c in CLASSES},
        "class_share_of_arrived": {c: by[c] / arrived if arrived else 0.0 for c in CLASSES},
        "class_count": {c: cnt[c] for c in CLASSES},
        "n_zero_share_of_lost": {str(k): v / lost if lost else 0.0 for k, v in sorted(nz_keys.items())},
        "others_ready_share_of_lost": ready / lost if lost else 0.0,
        "zero_hop_type_x_cls_share": {k: v / lost for k, v in sorted(hop_types.items())} if lost else {},
        "n_zero_count": dict(nz),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--policy", choices=["rl", "expert"], default="rl")
    ap.add_argument("--seeds", default="100-104")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"{out_path} 已存在（outputs 只增不改）")
    run, seeds = Path(args.run), parse_seeds(args.seeds)
    gen = run_rl(run, seeds, args.threads) if args.policy == "rl" else run_expert(run, seeds, args.overlay)
    all_rows, per_seed, arrived_tot = [], {}, 0.0
    for seed, env, rec in gen:
        s = env.metrics.episode_summary()
        rows = rec.classify()
        arrived = float(s.get("arrived_keys", 0.0))
        sm = summarize(rows, arrived)
        sm["success_rate"] = float(s.get("success_rate", 0.0))
        # 自检：分类到的丢失量 ≈ 1 − SR（SR = served/arrived；episode 末仍在途的请求不计入丢失）
        sm["check_lost_vs_1mSR"] = [sm["lost_share_of_arrived"], 1.0 - sm["success_rate"]]
        per_seed[seed] = sm
        all_rows.extend(rows)
        arrived_tot += arrived
        print(f"seed {seed}: SR {sm['success_rate']:.4f} lost/arr {sm['lost_share_of_arrived']:.4f} "
              + " ".join(f"{c}={sm['class_share_of_lost'][c]:.2f}" for c in CLASSES), flush=True)
    total = summarize(all_rows, arrived_tot)
    print("TOTAL share of lost: " + " ".join(f"{c}={total['class_share_of_lost'][c]:.3f}" for c in CLASSES))
    print("  unroutable 按提前囤货窗口（占到达量）: " + " ".join(
        f"{k}={v:.4f}" for k, v in sorted(total["unroutable_prestock_share_of_arrived"].items())))
    for c, u in total["hop_slot_use"].items():
        print(f"  hops {c}: w {u['weight_share']:.3f} act/vis {u['act_over_vis']:.2f} "
              f"free/vis {u['free_over_vis']:.2f} busy/vis {u['busy_over_vis']:.2f}")
    print(f"TOTAL others_ready {total['others_ready_share_of_lost']:.3f}  n_zero {total['n_zero_share_of_lost']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"run": str(run), "policy": args.policy, "seeds": seeds,
                                    "overlay": args.overlay, "total": total, "per_seed": per_seed},
                                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
