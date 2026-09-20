"""诊断：专家预存得比 RL 多吗？（`storage_reward` 那条建议的前置量）

### 为什么上一版探针是废的

第一版量的是 `routing.shortest_path(src,dst)` 的 `min(hop_levels)` —— 实测
**100% 为 0**。因为 `shortest_path` 是**纯拓扑最短路**，它**不是服务用的路**：
`partial_consume_for_request` 走的是 `_usable_cached_path`（优先）→
`_find_positive_path`（回退，**正存量子图**）。拓扑最短路上的边大多没存量。

★ 教训：量"瓶颈跳"必须挂在**真正决定服务量的那个函数**上
（`routing.py:237` 的 `min(hop_levels)`），而不是名字看起来对的函数。
[[variable-name-is-not-the-config-key]] 的同族：**函数名不是调用点。**

### 正确的挂点

`routing.partial_consume_for_request(req, qkp, t)`（`routing.py:212`）：
  - path = `_usable_cached_path` or `_find_positive_path`
  - `serve_now = min(hop_levels + [remaining])`

本探针**包一层**（monkeypatch 实例方法）记录每次调用的：
  · `hop_min`  = min(hop_levels) —— 瓶颈跳存量（**服务量的上界**）
  · `served`   = serve_now 实际服务量
  · `got_path` = 是否找到路（**False = 彻底够不着**，与"路在但存量 0"不同）

`got_path=False` 必须**单独计数**，不能记成 hop_min=0
（[[unavailable-must-not-look-like-a-value]]）。

regime：两侧都用 `load_validation_profile` + `build_validation_env_config`。
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402


def build_env():
    prof = _tp.load_validation_profile(str(REPO / "configs" / "global.yaml"))
    cfg = _tp.build_validation_env_config(prof)
    return build_env_from_config(cfg)


class Recorder:
    """包住 partial_consume_for_request，记录每次调用的瓶颈跳存量。"""

    def __init__(self, routing):
        self.routing = routing
        self.hop_min = []      # 有路时：min(hop_levels)
        self.served = []       # 有路时：serve_now
        self.no_path = 0       # 无路：够不着
        self.zero_hop = 0      # 有路但 min(hop_levels)==0
        self._orig = routing.partial_consume_for_request
        routing.partial_consume_for_request = self._wrapper

    def _wrapper(self, request, qkp, t, attributed=False):
        path = self.routing._usable_cached_path(request, qkp)
        if path is None:
            comp = getattr(self.routing, "_pos_component", None)
            if comp is not None and comp.get(request.src_gs, -1) != comp.get(request.dst_gs, -1):
                self.no_path += 1
                return (0.0, {}) if attributed else 0.0
            path = self.routing._find_positive_path(request, qkp)
        if path is None:
            self.no_path += 1
            return (0.0, {}) if attributed else 0.0
        levels = [qkp.get_level(e) for e in path]
        hm = min(levels) if levels else 0.0
        rem = max(0.0, request.amount - request.served_amount)
        sn = min(levels + [rem]) if levels else 0.0
        self.hop_min.append(hm)
        self.served.append(sn)
        if hm <= 1.0e-9:
            self.zero_hop += 1
        # 真的执行原函数（保持副作用）
        return self._orig(request, qkp, t, attributed=attributed)

    def detach(self):
        self.routing.partial_consume_for_request = self._orig


def run_expert(seed, steps=240):
    env = build_env()
    rec = Recorder(env.routing)
    obs = env.reset(seed=seed, start_seed=seed)
    exp = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                          principles=False, router=ServeProbe(env))
    done = False
    k = 0
    while not done and k < steps:
        acts, scores = exp.act(obs)
        obs, _r, term, trunc, _i = env.step(acts, scores)
        done = term or trunc
        k += 1
    rec.detach()
    return rec, env


def run_rl(ckpt, seed, steps=240):
    import argparse
    import importlib.util as iu
    import torch
    from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
    from qkd_rl.rl.algos.policy import MAPPOPolicy

    args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"],
        checkpoint=None, seed=seed, num_updates=30, run_name="_probe_prestock",
        device="cpu", mode="random_episode")
    # 直接读该臂的 resolved_config，避免再跑 build_config（会与 torch 线程冲突）
    import yaml
    cfgpath = REPO / "outputs" / ckpt.split("/")[1] / "resolved_config.yaml"
    config = yaml.safe_load(cfgpath.read_text(encoding="utf-8"))

    env = build_env()
    rec = Recorder(env.routing)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, "cpu")
    ck = torch.load(REPO / ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck.get("model", ck))
    model.load_state_dict(sd, strict=False)
    model.eval()

    obs = env.reset(seed=seed, start_seed=seed)
    done = False
    k = 0
    with torch.no_grad():
        while not done and k < steps:
            out = policy.act_batched([obs], deterministic=True, build_scores=True)
            o = out[0]
            obs, _r, term, trunc, _i = env.step(
                o.actions, o.action_scores,
                edge_scores=getattr(o, "edge_scores", None),
                expected_matched_edges=list(o.matched_edges or []))
            done = term or trunc
            k += 1
    rec.detach()
    return rec, env


def summarize(name, rec):
    n = len(rec.hop_min)
    print(f"\n  {name}")
    print(f"    有路调用  {n}")
    print(f"    无路调用  {rec.no_path}  （彻底够不着）")
    if n:
        z = rec.zero_hop / n
        print(f"    瓶颈跳=0  {rec.zero_hop} ({z:.1%})  ← 有路但被最弱跳饿死")
        nz = [x for x in rec.hop_min if x > 0]
        if nz:
            print(f"    非零瓶颈跳 n={len(nz)}  median={statistics.median(nz):,.0f} "
                  f"mean={statistics.mean(nz):,.0f}")
    return n, rec.no_path, rec.zero_hop


def main():
    seeds = [100, 101, 102, 103, 104]
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        "outputs/ent01_rerun_s42/checkpoint_final.pt"

    print("=" * 92)
    print("诊断：瓶颈跳存量（真正的服务路径）—— 专家 vs RL")
    print("=" * 92)
    print(f"  挂点：routing.partial_consume_for_request（routing.py:212）")
    print(f"  RL checkpoint: {ckpt}")
    print(f"  验证 regime，种子 {seeds}，240 步")

    E_np = E_n = E_z = 0
    R_np = R_n = R_z = 0
    E_nonzero, R_nonzero = [], []
    for s in seeds:
        erec, eenv = run_expert(s)
        rrec, renv = run_rl(ckpt, s)
        a, np_, z = summarize(f"seed {s} 专家", erec)
        b, np2, z2 = summarize(f"seed {s} RL", rrec)
        E_n += a; E_np += np_; E_z += z
        R_n += b; R_np += np2; R_z += z2
        E_nonzero += [x for x in erec.hop_min if x > 0]
        R_nonzero += [x for x in rrec.hop_min if x > 0]

    print("\n" + "=" * 92)
    print("汇总")
    print("=" * 92)
    print(f"  {'':<10}{'有路':>10}{'无路':>10}{'瓶颈=0':>10}{'零占比':>10}")
    print(f"  {'专家':<10}{E_n:>10}{E_np:>10}{E_z:>10}{E_z/max(1,E_n):>10.1%}")
    print(f"  {'RL':<10}{R_n:>10}{R_np:>10}{R_z:>10}{R_z/max(1,R_n):>10.1%}")

    if E_nonzero and R_nonzero:
        print(f"\n  非零瓶颈跳存量（预存的实际水平）")
        print(f"    {'':<10}{'n':>8}{'median':>16}{'mean':>16}")
        print(f"    {'专家':<10}{len(E_nonzero):>8}"
              f"{statistics.median(E_nonzero):>16,.0f}{statistics.mean(E_nonzero):>16,.0f}")
        print(f"    {'RL':<10}{len(R_nonzero):>8}"
              f"{statistics.median(R_nonzero):>16,.0f}{statistics.mean(R_nonzero):>16,.0f}")
        ratio = statistics.median(E_nonzero) / max(1.0, statistics.median(R_nonzero))
        print(f"\n  ★ 专家/RL 中位数比 = {ratio:.3f}")
        print("    比 > 1.2 ⟹ 专家预存更多，storage weight 有靶子")
        print("    比 ≈ 1.0 ⟹ 相当，调 weight 是朝空处推")
        print("    比 < 0.8 ⟹ RL 已更多")

    json.dump({"expert_nonzero": E_nonzero, "rl_nonzero": R_nonzero,
               "expert_n": E_n, "rl_n": R_n,
               "expert_nopath": E_np, "rl_nopath": R_np,
               "expert_zero": E_z, "rl_zero": R_z},
              open("/tmp/probe_prestock.json", "w"))


if __name__ == "__main__":
    main()
