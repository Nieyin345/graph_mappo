"""深挖：为什么 scratch_s42（从零训）能超专家？

## 已知差异（上一步实测）

| 量 | 专家 | scratch_s42 | ent01_rerun_s42（BC） |
|---|---|---|---|
| 有路调用/5局 | 2467 | **3086** | 3096 |
| 无路调用/5局 | 9399 | 8327 | 9026 |
| 非零瓶颈跳中位 | 14,502 | 14,064 | 12,431 |

★ 注意：scratch_s42 与 **BC 臂的 3096 几乎相同**，两者都远高于专家的 2467。
⟹ "有路调用多"**不是 scratch 独有的**，BC 臂也有。所以它不是超专家的原因。

## 本探针要分辨的

超专家必须有一个**专家不做**的行为。候选：

  A. **服务时机**：更早/更晚地把请求服务掉（换成整局完成数）
  B. **密钥预存分布**：在**未来会被用到的边**上存更多（而不是全部边平均）
  C. **端口分配效率**：同样激活边数下服务更多
  D. **纯粹是种子的运气**：与 BC 臂无差别

做法：把 scratch_s42 与 **ent01_rerun_s42（同种子、有 BC、不过专家）**
并排，逐量比。**同种子** ⟹ 请求流完全相同 ⟹ 差异只可能来自策略。

★ 若 scratch_s42 与 ent01_rerun_s42 在所有这些量上**都无法区分**，
那 0.7246 vs 0.6982 就是**单种子噪声**（本项目单种子分辨率 ~0.035），
不该再做梦。
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


def episode_stats(env, acts_fn, seed, steps=240):
    """跑一局，收集 per-step 与 per-episode 的量。"""
    obs = env.reset(seed=seed, start_seed=seed)
    per_step = []
    done = False
    k = 0
    while not done and k < steps:
        acts, scores = acts_fn(obs)
        obs, _r, term, trunc, _i = env.step(acts, scores)
        # 本步的全局量
        st = env.qkp
        try:
            lv = list(st.levels.values()) if hasattr(st, "levels") else []
        except Exception:
            lv = []
        per_step.append({
            "n_active": len(getattr(env, "last_matched_arcs", []) or []),
            "pos_edges": len(st.positive) if hasattr(st, "positive") else 0,
            "total_stock": sum(lv) if lv else 0.0,
            "pending": len(env.requests.pending),
        })
        done = term or trunc
        k += 1
    m = env.metrics.episode_summary()
    return m, per_step


def run_expert(env, seed):
    exp = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                          principles=False, router=ServeProbe(env))
    return episode_stats(env, exp.act, seed)


def make_rl_actor(env, ckpt, seed):
    import torch
    import yaml
    from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
    from qkd_rl.rl.algos.policy import MAPPOPolicy

    cfgpath = REPO / ckpt.split("/")[0] / ckpt.split("/")[1] / "resolved_config.yaml"
    config = yaml.safe_load(cfgpath.read_text(encoding="utf-8"))
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, "cpu")
    ck = torch.load(REPO / ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck.get("model", ck))
    model.load_state_dict(sd, strict=False)
    model.eval()

    def act_fn(obs):
        with torch.no_grad():
            out = policy.act_batched([obs], deterministic=True, build_scores=True)
            o = out[0]
        return o.actions, o.action_scores
    return act_fn


def main():
    seeds = [100, 101, 102, 103, 104]
    arms = {"scratch_s42": "outputs/scratch_s42/checkpoint_final.pt",
            "ent01_rerun_s42": "outputs/ent01_rerun_s42/checkpoint_final.pt"}

    print("=" * 92)
    print("深挖：scratch_s42（从零，0.7246）vs ent01_rerun_s42（有BC，0.6982）")
    print("=" * 92)
    print(f"  ⚠ 两者**同训练种子 42** ⟹ 请求流相同 ⟹ 差异只来自策略")
    print(f"  ⚠ 对照：专家 = 0.6979\n")

    rows = {}
    for name, ck in arms.items():
        env = build_env()
        act = make_rl_actor(env, ck, 42)
        srs, stock, actv, pend = [], [], [], []
        for s in seeds:
            m, ps = episode_stats(env, act, s)
            srs.append(m["success_rate"])
            stock.append(statistics.mean(p["total_stock"] for p in ps))
            actv.append(statistics.mean(p["n_active"] for p in ps))
            pend.append(statistics.mean(p["pending"] for p in ps))
        rows[name] = {"sr": srs, "stock": stock, "actv": actv, "pend": pend}

    # 专家
    env = build_env()
    esr, estock, eactv, epend = [], [], [], []
    for s in seeds:
        m, ps = run_expert(env, s)
        esr.append(m["success_rate"])
        estock.append(statistics.mean(p["total_stock"] for p in ps))
        eactv.append(statistics.mean(p["n_active"] for p in ps))
        epend.append(statistics.mean(p["pending"] for p in ps))

    print(f"  {'策略':<18}{'SR 均值':>10}{'SR 逐种子':>36}")
    print(f"  {'专家':<18}{statistics.mean(esr):>10.4f}   {[round(x,4) for x in esr]}")
    for name, r in rows.items():
        print(f"  {name:<18}{statistics.mean(r['sr']):>10.4f}   "
              f"{[round(x,4) for x in r['sr']]}")

    print(f"\n  {'策略':<18}{'全局存量均值':>16}{'激活边/步':>12}{'pending/步':>12}")
    print(f"  {'专家':<18}{statistics.mean(estock):>16,.0f}"
          f"{statistics.mean(eactv):>12.1f}{statistics.mean(epend):>12.1f}")
    for name, r in rows.items():
        print(f"  {name:<18}{statistics.mean(r['stock']):>16,.0f}"
              f"{statistics.mean(r['actv']):>12.1f}{statistics.mean(r['pend']):>12.1f}")

    # 逐步配对（同种子）
    print(f"\n  同种子配对（对本探针里的专家复算）：")
    for name, r in rows.items():
        d = [a - b for a, b in zip(r["sr"], esr)]
        n = len(d)
        m = statistics.mean(d)
        sd = statistics.stdev(d)
        se = sd / n ** 0.5
        print(f"    {name:<18} Δ={m:+.4f}  t={m/se:+.2f}  (同向 {sum(1 for x in d if x>0)}/{n})")

    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in rows.items()}
              | {"expert": {"sr": esr, "stock": estock}},
              open("/tmp/probe_deep_s42.json", "w"))


if __name__ == "__main__":
    main()
