"""★ 端到端「指标是不是活的」造反证梯子 —— 回答「底层代码到底对不对」。

### 为什么是梯子

`success_rate = served_keys / arrived_keys`。直接改 `served` 再看它变不变是
**循环论证** —— 同一个量，必然变，什么也没证明。

要证的是「**环境的物理事实**变了，指标会不会跟着动」。每一级的预期
**事前写死**，跑完对账；任何一级不符预期就报红。

| 级 | 改哪里 | 事前预期 | 证什么 |
|---|---|---|---|
| T0 | 无 | 参照 | — |
| T1 | **匹配器**：强制所有 `_resolve_*` 返回空 | **served 大跌** | 决策真的驱动密钥服务 |
| T2 | **请求流**：到达 `amount` ×10 | **arrived 约 ×10、sr 约 ÷10** | 需求真的进分母 |
| T3 | **QKP/服务**：`serve_result.served_keys` 读数 ÷10（**在 metrics 里改，不动 reward**） | **sr 约 ×10** | 供方真的进分子 |
| T4 | **特异性**：只改 `conflict_count`（下游没人读） | **其余量逐位不动**，而 `conflict_count` **必须**动 | 探针不是噪音 |
| T5 | 交叉验证 | 逐步 `info` 之和 == 整局 `summary` | 训练侧聚合与评测侧报告是同一个数 |
| T6 | **指标**：`success_rate` 读数 ×0.5 | **精确减半** | 报告器确实读这个量 |

★ T4 是**假阳性探测器**：若改一个下游没人读的量指标也动，T1/T2/T3 的「动」
   就不值钱。T4 同时是**正对照**（`conflict_count` 必须动），否则
   「测不出」与「没跑」无法区分。
★ T5 是造反证：训练日志聚合的是逐步 `info`，评测报告读的是整局
   `summary` —— 两处必须一致，否则 A/B 与论文引的不是同一个数。

### ★★ 一臂 = 一个**子进程**（这是 v1 失败的直接教训）

v1 在同一进程里改磁盘上的 `.py` 再跑，六级读数**逐位完全相同**（连 T6 的
0.5 注入都没变）—— 因为 Python 把模块缓存在 `sys.modules` 里，
**改文件不会让已在内存的模块对象更新**。六个数一模一样是「补丁一次都没
生效」的指纹，不是「代码坏了」。

子进程带来两条硬保证：① 不可能有陈旧模块；② 补丁与运行环境彻底隔离，
一个臂的改动**结构上**流不到下一个臂。

### 方法：补丁打在一份**克隆**上，且必然还原

`/tmp/gm_probe/` 是 `/opt/qkd/graph_mappo` 的副本。peer 正在 /opt 下跑训练
⟹ 全程不碰 /opt。`apply_patch.patched()` 用 `finally` 还原源码。

用法（服务器上）：
    /opt/qkd/venv/bin/python -u /tmp/probe_metric_alive_ladder.py [--seeds 100-102]

子进程模式（父进程自己调，不用手敲）：
    /opt/qkd/venv/bin/python -u /tmp/probe_metric_alive_ladder.py --arm T1
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

CLONE = Path("/tmp/gm_probe")
sys.path.insert(0, str(CLONE))
sys.path.insert(0, "/tmp")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RES = CLONE / "qkd_rl" / "env" / "action_resolver.py"
ENV = CLONE / "qkd_rl" / "env" / "env.py"
MET = CLONE / "qkd_rl" / "env" / "metrics.py"

# 注入代码需要在模块层有 `import os` / `import dataclasses`。
# ★ 必须插在 `from __future__ import ...` **之后** —— future 语句只允许被
#   docstring/注释/空行/其他 future 前置，插它前面是 SyntaxError，
#   而那个 SyntaxError 会伪装成「环境坏了」。
A_FUTURE = "from __future__ import annotations"
I_FUTURE = "from __future__ import annotations\nimport dataclasses, os  # PROBE-INJECT"

# ---------------- 六个臂的补丁（每个臂只打一处） ----------------
PATCHES: dict[str, tuple[Path, str, str]] = {
    # T1 决策侧：把四个 `_resolve_*` 全换成返回空，并强制走 mutual_choice 分支。
    #    实例属性遮蔽类方法 ⟹ 无论外面配的 mode 是什么，匹配结果都是空集。
    "T1": (RES,
           "        self.mode = config[\"mode\"]",
           "        self.mode = config[\"mode\"]\n"
           "        self._resolve_priority_matching = lambda *a, **k: []\n"
           "        self._resolve_mutual_choice = lambda *a, **k: []\n"
           "        self._resolve_greedy_rate_matching = lambda *a, **k: []\n"
           "        self._resolve_max_weight_matching = lambda *a, **k: []\n"
           "        self.mode = 'mutual_choice'"),
    # T2 需求侧：到达量 ×10。`dataclasses.replace` 造新对象，不原地改，
    #    避免被下游按 id 缓存（request_history 会记 arrivals）。
    "T2": (ENV,
           "        arrivals = self.request_generator.generate(self.t)",
           "        arrivals = self.request_generator.generate(self.t)\n"
           "        arrivals = [dataclasses.replace(_a, amount=_a.amount * 10)\n"
           "                    for _a in arrivals]"),
    # T3 供方侧：在 metrics.update 里把读数缩小 10 倍。★ 这个位置在
    #    `reward.compute`（env.py:214）**之后**、`last_info`（env.py:257）**之前**
    #    ⟹ 奖励看到原值、指标看到缩水值，所以它精确地只测「指标有没有接线」。
    "T3": (MET,
           "        self.served_keys += serve_result.served_keys",
           "        serve_result.served_keys = serve_result.served_keys * 0.1\n"
           "        self.served_keys += serve_result.served_keys"),
    # T4 特异性 + 正对照。★ 锚点必须打在**累加器**上而不是 `last` 字典上：
    #    `metrics.py` 里 `conflict_count` 有**两列同名** ——
    #    `self.last["conflict_count"]`（当步快照，进 `info`）与
    #    `self.conflict_count`（整局累加，进 `episode_summary`）。
    #    v2 打的是前一列、读的是后一列 ⟹ 正对照报警「补丁没生效」。
    #    正对照在这里救了场：若没有它，「其余五量逐位不动」会被误读成
    #    「特异性通过」，而不是「压根没改到东西」。
    "T4": (MET,
           "        self.conflict_count += resolved_action.conflict_count",
           "        self.conflict_count += resolved_action.conflict_count * 777 + 3"),
    # T6 指标注入：只改报告出来的比率，原始计数不动。
    "T6": (MET,
           '            "success_rate": self.served_keys / self.arrived_keys'
           ' if self.arrived_keys > 0 else 0.0,',
           '            "success_rate": (self.served_keys / self.arrived_keys'
           ' if self.arrived_keys > 0 else 0.0) * 0.5,'),
}


# ==========================================================================
#                            子进程：跑一个臂
# ==========================================================================
def parse_seeds(spec: str) -> list[int]:
    """`"100-102"` → `[100,101,102]`；`"100"` → `[100]`；`"7,9"` 也收。"""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    if not out:
        raise SystemExit(f"--seeds 解析出空集：{spec!r}")
    return out


def run_one_arm(tag: str) -> dict:
    """在**本进程**里（已被父进程打上 tag 对应的补丁）跑一臂。"""
    spec = importlib.util.spec_from_file_location(
        "_tp", CLONE / "qkd_rl" / "evaluation" / "test_protocol.py")
    _tp = importlib.util.module_from_spec(spec)
    sys.modules["_tp"] = _tp
    spec.loader.exec_module(_tp)

    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.baselines.path_greedy import PathScoreGreedy
    from qkd_rl.baselines.serve_probe import ServeProbe

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="T0")
    ap.add_argument("--seeds", default="100-102")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--config", default="configs/train_full_rl.yaml")
    a, _ = ap.parse_known_args()
    seeds = parse_seeds(a.seeds)

    profile = _tp.load_validation_profile(CLONE / a.config)
    cfg = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=a.steps,
        start_mode=profile["start_mode"])

    out = {"tag": tag, "served": 0.0, "arrived": 0.0, "failed": 0.0,
           "gen": 0.0, "conflict": 0.0, "n_steps": 0,
           "info_served_sum": 0.0, "info_arrived_sum": 0.0,
           "sr_src": [], "sr_rec": [],
           # ★ 见证：这臂里匹配器到底有没有被真正改写、真的执行了
           "witness_mode": None, "witness_matched_len": [], "witness_disarmed": None}
    for seed in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=seed,
                        start_seed=int(profile.get("start_seed", 0)) + seed)
        out["witness_mode"] = getattr(env.action_resolver, "mode", None)
        out["witness_disarmed"] = callable(
            getattr(env.action_resolver, "_resolve_priority_matching", None)) and (
            getattr(env.action_resolver, "_resolve_priority_matching").__name__
            == "<lambda>")
        expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2),
                                 phased=True, principles=False,
                                 router=ServeProbe(env))
        n = 0
        for _ in range(a.steps):
            actions, scores = expert.act(obs)
            obs, _r, term, trunc, info = env.step(actions, scores)
            n += 1
            out["info_served_sum"] += float(info.get("served_keys", 0.0))
            out["info_arrived_sum"] += float(info.get("arrived_keys", 0.0))
            if term or trunc:
                break
        # `env.last_matched_arcs` 每步都被赋成当步的匹配结果（env.py:238）
        # ⟹ 无需任何补丁就能拿到「匹配器到底匹没匹」的见证。
        out["witness_matched_len"].append(len(env.last_matched_arcs or []))
        s = env.metrics.episode_summary()
        served = float(s.get("served_keys", 0.0))
        arrived = float(s.get("arrived_keys", 0.0))
        out["served"] += served
        out["arrived"] += arrived
        out["failed"] += float(s.get("failed_keys", 0.0))
        out["gen"] += float(s.get("generated_keys", 0.0))
        out["conflict"] += float(s.get("conflict_count", 0.0))
        out["n_steps"] += n
        out["sr_src"].append(float(s.get("success_rate", float("nan"))))
        out["sr_rec"].append(served / arrived if arrived > 0 else float("nan"))
    return out


# ==========================================================================
#                       父进程：打补丁 → 起子进程 → 对账
# ==========================================================================
def spawn(tag: str, seeds: str, steps: int, patch_mod) -> dict:
    """打上 tag 的补丁，起一个**新解释器**跑那一臂，还原补丁，回收结果。"""
    if tag == "T0":
        print(f"\n[T0] 不 patch（基线）")
        mods = []
    else:
        path, anchor, injected = PATCHES[tag]
        print(f"\n[{tag}] patch {path.name}")
        mods = [patch_mod.patched(path, A_FUTURE, I_FUTURE, label=f"{tag}-pre"),
                patch_mod.patched(path, anchor, injected, label=tag)]
    import contextlib
    with contextlib.ExitStack() as st:
        oks = [st.enter_context(m) for m in mods]
        if not all(oks):
            print(f"    ✗ 锚点不唯一（{oks}）⟹ 拒绝跑，本级作废")
            return {}
        cmd = [sys.executable, "-u", __file__, "--arm", tag,
               "--seeds", seeds, "--steps", str(steps)]
        p = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")]
    if not line:
        print(f"    ✗ 子进程没有吐结果（rc={p.returncode}）")
        print("      stderr 末尾：")
        for l in (p.stderr or "").splitlines()[-12:]:
            print(f"        {l}")
        return {}
    return json.loads(line[-1][len("@@RESULT@@"):])


def sr(r: dict) -> float:
    return r["served"] / r["arrived"] if r["arrived"] else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None,
                    help="内部用：只跑这一个臂并打印 JSON（被父进程调起）")
    ap.add_argument("--seeds", default="100-102")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", default="/tmp/probe_metric_alive_ladder.json")
    a = ap.parse_args()

    # ---------------- 子进程模式 ----------------
    if a.arm:
        r = run_one_arm(a.arm)
        print("@@RESULT@@" + json.dumps(r))
        return 0

    # ---------------- 父进程模式 ----------------
    patch = importlib.import_module("apply_patch")
    fails: list[str] = []
    R: dict[str, dict] = {}
    print("=" * 84)
    print(f"★ 指标活性造反证梯子 —— {a.seeds} 号种子 × {a.steps} 步（验证 regime）")
    print(f"  克隆根 = {CLONE}（全程不碰 /opt/qkd/graph_mappo）")
    print(f"  模式 = **一臂一子进程**（v1 同进程失败：模块被 sys.modules 缓存）")
    print("=" * 84)

    for tag in ("T0", "T1", "T2", "T3", "T4", "T6"):
        r = spawn(tag, a.seeds, a.steps, patch)
        R[tag] = r
        if r:
            print(f"    served={r['served']:>12,.0f}  arrived={r['arrived']:>12,.0f}"
                  f"  success_rate={sr(r):.4f}  conflict={r['conflict']:,.0f}")
            print(f"    见证: resolver.mode={r['witness_mode']!r}  "
                  f"已被改写={r['witness_disarmed']}  "
                  f"每种子匹配弧数={r['witness_matched_len']}")
            if r["n_steps"] != r["n_steps"]:
                fails.append(f"{tag}: 步数异常")
    if not R.get("T0"):
        print("✗ 基线没跑出来，后续对账无意义")
        return 1
    base = sr(R["T0"])

    # ---------------- T1 ----------------
    r1 = R.get("T1")
    if r1:
        print(f"    事前预期：served 大跌。实测 Δsr = {sr(r1)-base:+.4f}")
        if not r1["witness_disarmed"] or r1["witness_mode"] != "mutual_choice":
            fails.append(f"T1 见证失败：resolver 没被改写"
                         f"（mode={r1['witness_mode']}, disarmed={r1['witness_disarmed']}）"
                         f"⟹ 本级没测到东西")
        elif max(r1["witness_matched_len"] or [1]) > 0:
            fails.append(f"T1 见证矛盾：已强制返回空，`last_matched_arcs` 却非空"
                         f"（{r1['witness_matched_len']}）")
        elif r1["arrived"] != R["T0"]["arrived"]:
            fails.append(f"T1 只动匹配器，`arrived` 却变了（{R['T0']['arrived']:.0f} →"
                         f" {r1['arrived']:.0f}）⟹ 需求流被牵连")
        elif r1["served"] >= R["T0"]["served"] * 0.9:
            fails.append(f"T1 匹配器返回空集，served 却只从 {R['T0']['served']:.0f}"
                         f" 变到 {r1['served']:.0f} ⟹ **决策到密钥服务这条链是断的**")
        else:
            print(f"    ✓ 见证通过（匹配弧数 {R['T0']['witness_matched_len'][0]} → 0），"
                  f"served {R['T0']['served']:.0f} → {r1['served']:.0f}"
                  f"（{100*(1-r1['served']/R['T0']['served']):.1f}% ↓）⟹ 环境是活的")

    # ---------------- T2 ----------------
    r2 = R.get("T2")
    if r2:
        print(f"    事前预期：arrived ≈ ×10、sr ≈ ÷10（{base:.4f} → ≈{base/10:.4f}）")
        ratio = r2["arrived"] / R["T0"]["arrived"] if R["T0"]["arrived"] else float("nan")
        print(f"    实测 arrived ×{ratio:.2f}，sr={sr(r2):.4f}")
        if ratio < 5:
            fails.append(f"T2 到达量 ×10，`arrived` 只涨 {ratio:.2f}× ⟹ 需求没进分母")
        elif sr(r2) > base * 0.5:
            fails.append(f"T2 arrived 涨 {ratio:.2f}× 而 sr 只到 {sr(r2):.4f} ⟹ "
                         f"指标对需求不敏感")
        else:
            print("    ✓ 需求放大精确传导到分母")

    # ---------------- T3 ----------------
    r3 = R.get("T3")
    if r3:
        s_ratio = r3["served"] / R["T0"]["served"] if R["T0"]["served"] else float("nan")
        print(f"    事前预期（★ v2 写反了，此处更正）：`serve_result.served_keys` 现在"
              f"也是 `summary['served_keys']` 的构造项 ⟹ served 与 sr **同向** ÷10")
        print(f"    实测 served ×{s_ratio:.3f}  sr ×{sr(r3)/base:.3f}")
        if r3["arrived"] != R["T0"]["arrived"]:
            fails.append("T3 只动 served 通路，arrived 却变了 ⟹ 归因不干净")
        elif not (0.05 < s_ratio < 0.2):
            fails.append(f"T3 把 served 缩到 1/10，served 却变成 {s_ratio:.3f}×"
                         f" ⟹ 这条通路没被切断")
        elif not (0.05 < sr(r3) / base < 0.2):
            fails.append(f"T3 served ÷10 但 sr 变成 {sr(r3)/base:.3f}×"
                         f" ⟹ **success_rate 的分子不是这条 served 通路**")
        else:
            print("    ✓ served 通路精确传动：served 与 sr 同步 ÷10")
            print("      ⟹ 证的是「`success_rate` 的分子就是 serve_result.served_keys」")

    # ---------------- T4 特异性 + 正对照 ----------------
    r4 = R.get("T4")
    if r4:
        keys = ("served", "arrived", "failed", "gen", "n_steps")
        same = {k: (R["T0"][k] == r4[k]) for k in keys}
        print(f"    事前预期：除 conflict_count 外逐位不动。逐位相同={same}")
        if r4["conflict"] == R["T0"]["conflict"]:
            fails.append("T4 正对照失败：改了 conflict_count 它自己却没动 ⟹ "
                         "补丁没生效，本级的「不动」不作数")
        elif not all(same.values()):
            bad = [k for k, v in same.items() if not v]
            fails.append(f"T4 改一个下游没人读的量，这些却动了：{bad} ⟹ "
                         f"探针在报噪音，T1/T2/T3 的「动」不作数")
        else:
            print(f"    ✓ conflict {R['T0']['conflict']:,.0f} → {r4['conflict']:,.0f}"
                  f"（补丁确实生效），其余五量逐位不动 ⟹ 探针不是噪音")
            print("      ⟹ `conflict_count` 只在 info 里，**指标与奖励都不读它**")

    # ---------------- T5 交叉验证 ----------------
    print("\n[T5] 交叉验证：逐步 `info` 之和 vs 整局 `summary`，逐臂")
    for tag in ("T0", "T1", "T2", "T3", "T4"):
        # ★ T6 被排除：它**故意**只改报告比率 ⟹ 报告器与重算必然不同，
        #   那不是缺陷而是它的目的。把它拉进来做交叉验证是判据用错地方
        #   （v2 就是这么误报的：偏差 2.78e-01 == 0.5×0.5478）。
        r = R.get(tag)
        if not r:
            continue
        d_s = abs(r["info_served_sum"] - r["served"])
        d_a = abs(r["info_arrived_sum"] - r["arrived"])
        d_r = max((abs(x - y) for x, y in zip(r["sr_src"], r["sr_rec"])), default=float("nan"))
        print(f"    {tag:<3} |Σinfo−summary|: served {d_s:.4e}  arrived {d_a:.4e}"
              f"   |报告器−重算| {d_r:.2e}")
        if d_s > 1e-6 or d_a > 1e-6:
            fails.append(f"{tag}: 逐步 info 之和 ≠ 整局 summary"
                         f"（served 差 {d_s:.4e}）⟹ 训练日志聚合的与评测报告读的"
                         f"**不是同一个数**")
        if d_r > 1e-9:
            fails.append(f"{tag}: 报告器与重算不一致（{d_r:.2e}）")

    # ---------------- T6 指标注入 ----------------
    r6 = R.get("T6")
    if r6:
        exp = [v * 0.5 for v in R["T0"]["sr_src"]]
        d = max((abs(x - y) for x, y in zip(r6["sr_src"], exp)), default=float("nan"))
        print(f"\n[T6] 事前预期：报告器**精确**减半。最大偏差 {d:.2e}")
        if d > 1e-12:
            fails.append(f"T6 指标注入没有精确传导（最大偏差 {d:.2e}）")
        else:
            print("    ✓ 同一个量，注一半报一半 ⟹ 报告器读的确实是这个量")
        if r6["served"] != R["T0"]["served"] or r6["arrived"] != R["T0"]["arrived"]:
            fails.append("T6 只改报告比率，原始计数却变了 ⟹ 补丁越界")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 84)
    Path(a.out).write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {a.out}")
    desc = {"T0": "（基线，不 patch）", "T1": "匹配器：强制返回空集",
            "T2": "请求流：到达量 ×10", "T3": "QKP：served 读数 ÷10",
            "T4": "冲突计数 ×777（下游无人读）", "T6": "指标：success_rate ×0.5"}
    print(f"{'级':<4} {'改哪里':<30} {'served':>13} {'arrived':>13}"
          f" {'success_rate':>13} {'conflict':>8}")
    print("-" * 84)
    for tag in ("T0", "T1", "T2", "T3", "T4", "T6"):
        r = R.get(tag)
        if r:
            print(f"{tag:<4} {desc[tag]:<30} {r['served']:>13,.0f}"
                  f" {r['arrived']:>13,.0f} {sr(r):>13.4f} {r['conflict']:>8,.0f}")
    print("-" * 84)
    if fails:
        print(f"✗ {len(fails)} 条不通过：")
        for f in fails:
            print(f"    - {f}")
    else:
        print("✓ 梯子全部通过：决策→需求→供方→指标→报告器这条链是活的，"
              "且探针不是噪音、不是循环论证")
    print("=" * 84)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
