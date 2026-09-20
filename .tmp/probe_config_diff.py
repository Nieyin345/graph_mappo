"""两条配置路径**逐键** diff —— 找出专家成功率差 8 个点的真因。

## 背景

`probe_expert_config_paths.py` 实测：同一个专家、同一批种子，在
  · `test_protocol.build_validation_env_config`（profile 路径，专家锚用的）
  · 训练链配置（训练器内部验证用的）
下成功率差最多 **8 个百分点**，且符号两边都有。

★ 那个探针只比了 `features.edge` 段，并且把结论写成"⟹ 专家锚必须在训练链
配置下重测"。**归因未验证**：差异可能来自任何一处配置，不一定是那个特征开关。
（"大且变号"的模式更像**请求流不同**，而不是系统性环境差异。）

## 这个探针做什么

把两条配置**完全摊平**成 `a.b.c = value`，逐键比对，打印全部差异。
再按"会不会改变请求流 / 会不会改变动力学"分类，好定位真因。

用法：
    python3 -u probe_config_diff.py
"""
import importlib.util
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


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (list, tuple)):
        out[prefix[:-1]] = repr(d)
    else:
        out[prefix[:-1]] = d
    return out


def main():
    tp = load_mod("gm_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
    profile = tp.load_validation_profile(REPO / "configs" / "global.yaml")
    cfg_a = tp.build_validation_env_config(profile, include_baselines=True)

    train = load_mod("gm_train", REPO / "scripts" / "train" / "train_graph_mappo.py")

    class A:
        pass

    a = A()
    a.mode = "random_episode"
    a.configs = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml",
                 "train_v2_bottleneck.yaml"]
    a.seed = 42
    a.num_updates = 1
    a.run_name = "cfgdiff"
    a.device = "cpu"
    cfg_b = train.build_config(a)

    fa, fb = flatten(cfg_a), flatten(cfg_b)
    keys = sorted(set(fa) | set(fb))

    print("=" * 100)
    print("两条配置路径的**全部**差异")
    print("=" * 100)
    ndiff = 0
    buckets = {"seed": [], "requests": [], "rate": [], "env": [], "features": [],
               "reward": [], "other": []}
    for k in keys:
        va, vb = fa.get(k, "<缺>"), fb.get(k, "<缺>")
        if va == vb:
            continue
        ndiff += 1
        top = k.split(".")[0]
        if top in buckets:
            buckets[top].append((k, va, vb))
        else:
            buckets["other"].append((k, va, vb))

    for name, rows in buckets.items():
        if not rows:
            continue
        print()
        print(f"--- {name}（{len(rows)} 项）---")
        for k, va, vb in rows:
            print(f"  {k:<48} profile={str(va):<24} 训练链={str(vb)}")

    print()
    print("=" * 100)
    print(f"共 {ndiff} 项差异")
    print("=" * 100)
    if any(k.startswith("seed.") for k, _, _ in buckets["seed"]):
        print("★ seed 段有差异 ⟹ **请求流不同**，这足以单独造成大而变号的逐种子差异")
        print("  ⟹ 「专家两条路径不同」的真因很可能是播种，不是 features")
    if not ndiff:
        print("★ 两条配置完全一样 ⟹ 上面那个探针的差异只能是别的来源")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
