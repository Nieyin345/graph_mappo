"""坐实：scratch_s42 超专家的机理 = 「少占边、把货集中到会被用的边上」

## 已经测到的三条独立证据（同种子 42，验证 regime，种子 100-104）

| 量 | 专家 | scratch_s42 | ent01_rerun_s42（BC） |
|---|---|---|---|
| 验证 SR | 0.6540 | **0.6855** | 0.6628 |
| 对专家配对 Δ | — | **+0.0315 (t=+2.44)** | +0.0088 (t=+1.19) |
| 全局存量均值 | 959,336,377 | **493,152,759** | 600,167,022 |
| key_efficiency（u30） | **246.97** | **105.1** | 135.6 |
| 激活边/步（训练侧） | 55.2 | **50.8** | 58.4 |

★ 三量同向：**更少的边、更少的密钥、更高的成功率**。

## 这一步要排除的替代解释

候选：**scratch 只是"生成得少"**（少生成自然 key_efficiency 低）。
但服务量**没降**（SR 更高），所以不是"少干活"，是"干得更准"。
本探针直接量**每步生成量**与**服务量**，把两者分开。
"""
import importlib.util
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


def make_rl_actor(env, ckpt):
    import torch
    import yaml
    from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
    from qkd_rl.rl.algos.policy import MAPPOPolicy

    parts = ckpt.split("/")
    config = yaml.safe_load(
        (REPO / parts[0] / parts[1] / "resolved_config.yaml").read_text(encoding="utf-8"))
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


def episode(env, act_fn, seed, steps=240):
    obs = env.reset(seed=seed, start_seed=seed)
    gen, srv, act = 0.0, 0.0, []
    done, k = False, 0
    prev_served = 0.0
    while not done and k < steps:
        acts, scores = act_fn(obs)
        obs, _r, term, trunc, _i = env.step(acts, scores)
        gen += float(getattr(env.qkp, "generated_this_step", 0.0) or 0.0) \
            if hasattr(env.qkp, "generated_this_step") else 0.0
        act.append(len(getattr(env, "last_matched_arcs", []) or []))
        done = term or trunc
        k += 1
    m = env.metrics.episode_summary()
    return m, statistics.mean(act) if act else 0.0


def main():
    seeds = [100, 101, 102, 103, 104]
    print("=" * 92)
    print("坐实：生成量 vs 服务量（分开量，排除「只是生成得少」）")
    print("=" * 92)

    rows = {}
    for name, ck in (("scratch_s42", "outputs/scratch_s42/checkpoint_final.pt"),
                     ("ent01_rerun_s42", "outputs/ent01_rerun_s42/checkpoint_final.pt")):
        env = build_env()
        act = make_rl_actor(env, ck)
        G, S, A, SR = [], [], [], []
        for s in seeds:
            m, a = episode(env, act, s)
            G.append(m["generated_keys"])
            S.append(m["served_keys"])
            A.append(a)
            SR.append(m["success_rate"])
        rows[name] = (statistics.mean(G), statistics.mean(S),
                      statistics.mean(A), statistics.mean(SR))

    env = build_env()
    exp = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                          principles=False, router=ServeProbe(env))
    G, S, A, SR = [], [], [], []
    for s in seeds:
        m, a = episode(env, exp.act, s)
        G.append(m["generated_keys"])
        S.append(m["served_keys"])
        A.append(a)
        SR.append(m["success_rate"])
    rows["专家"] = (statistics.mean(G), statistics.mean(S),
                    statistics.mean(A), statistics.mean(SR))

    print(f"\n  {'策略':<18}{'生成':>16}{'服务':>16}{'生成/服务':>11}"
          f"{'激活边':>9}{'SR':>8}")
    print("  " + "-" * 78)
    for name, (g, s, a, sr) in rows.items():
        print(f"  {name:<18}{g:>16,.0f}{s:>16,.0f}{g/max(1,s):>11.1f}"
              f"{a:>9.1f}{sr:>8.4f}")

    print("\n  判读：")
    gg = rows["scratch_s42"]
    ee = rows["专家"]
    print(f"    生成量 scratch/专家 = {gg[0]/ee[0]:.3f}   （<1 ⟹ 更少生成）")
    print(f"    服务量 scratch/专家 = {gg[1]/ee[1]:.3f}   （>1 ⟹ 服务更多）")
    print(f"    ⟹ 若服务量不降而生成降，则是「更准」不是「少干」")


if __name__ == "__main__":
    main()
