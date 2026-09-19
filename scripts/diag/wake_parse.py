#!/usr/bin/env python
"""服务器端状态探针：被 wake_poll.sh 每 2 分钟调一次，输出**只在有新信息时**改变。

为什么写它（而不是直接 tail 日志）：
  - tail -f 一断就静默，静默看起来和"还在跑"一样；
  - 先前的唤醒链把"有没有新验证点"和"内存够不够"混在一起，报不准。

本脚本的输出会被 md5，只有变化才会唤醒主代理。所以：
  **稳定态不要抖**（比如"还有几个 run"要排序后拼接，不能带时间戳）。

另外做一件自动事：**内存安全余量足够时补起 ent01_g999_s43**。
  g999 第二臂被看门狗误杀过（见 docs/训练诊断记录.md），需要补回来才能
  做三种子的 gamma 净效应配对。判据用**内存**（每 run 23.3 GB + 0.21/轮），
  不是 CPU。补过后写状态文件，不会重复补。

  **可用 /tmp/qkd_g999_respawn.json 里的 "hold": "原因" 关掉自动补起**：
  补起会占用一个 run 的内存名额，而内存是这台机器唯一的瓶颈，所以当有
  **更有价值的实验**要跑时（例如 ent01_s42_u25_base 这个起点配对对照），
  必须显式让它让位，否则两个 run 会互相抢内存、把对照臂 OOM 掉。

用法：python /tmp/wake_parse.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
STATE = Path("/tmp/qkd_g999_respawn.json")

# 一个 run 的最低成本（实测 23.3 GB），再加上每轮的增长余量。
RUN_BASE_GB = 23.3
RUN_GROWTH_GB = 0.21   # 多点拟合均值（+0.085~+0.298），不是心算的 0.43
SAFETY_GB = 10.0
# 本文件自动补起的 run 只跑 30 轮（见下面 cmd 里的 --num-updates）。
# 内存判据要推演到这个轮数，不能只看"现在够不够"——内存是增长的。
TARGET_UPDATES = 30
# 单 run 跑到 TARGET_UPDATES 的峰值占用。
RUN_FULL_GB = RUN_BASE_GB + RUN_GROWTH_GB * TARGET_UPDATES   # ≈ 29.6 GB
# 全部跑到头时，机器**绝对**要留的余量。
#
# 来历：本项目记录说「125 GB 机器上并发 3 稳、**4 勉强（108 GB，余 17 GB）**、
# 5 一定 OOM」。那个 108 GB 是 4 个 run 跑到头的物理占用，余 17 GB 是它的余量。
# 所以 17 不是拍脑袋的数，是**记录里那条"勉强"线的余量本身**。
# 低于它就等于把机器推到"勉强"或更差，而机器上还有探针、pytest、ssh 要吃内存。
MIN_HEADROOM_GB = 17.0


def lines(p: Path):
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def align(d: Path):
    """返回 (末轮, [(轮, 验证均值), ...])。验证是独立一行，靠自己前面最近的 update 定位。"""
    last_u, vals = None, []
    for r in lines(d / "metrics.jsonl"):
        if "update" in r:
            last_u = r["update"]
        ev = r.get("eval_validation")
        if isinstance(ev, dict) and ev.get("mean_success_rate") is not None:
            vals.append((last_u, float(ev["mean_success_rate"])))
    return last_u, vals


def series(d: Path):
    """返回 {轮: (均值, 逐种子 list)} —— 配对比较要逐种子值，align() 只给均值。

    为什么要逐种子：本项目的规矩是**配对才是可比口径**（docs/测试规范.md §4）。
    两臂各自对同一批验证种子求均值再相减，与"逐种子相减再求均值"不是一回事；
    后者能把种子间的难度差抵掉，前者不能。所以配对比较必须拿到 per_seed_success。
    """
    out = {}
    last_u = None
    for r in lines(d / "metrics.jsonl"):
        if "update" in r:
            last_u = r["update"]
        ev = r.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success"):
            try:
                out[last_u] = (float(ev["mean_success_rate"]),
                               [float(x) for x in ev["per_seed_success"]])
            except (TypeError, ValueError):
                pass
    return out


def meminfo():
    mi = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                k, _, v = line.partition(":")
                mi[k.strip()] = int(v.split()[0]) / 1048576.0
    except OSError:
        pass
    return mi


def trainers():
    """{run_name: (pid, pid_count)}，只认父进程。"""
    out = {}
    res = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True)
    for line in res.stdout.splitlines()[1:]:
        if "train_graph_mappo.py" not in line or "grep" in line:
            continue
        pid_s, _, args = line.strip().partition(" ")
        m = re.search(r"--run-name\s+(\S+)", args)
        if m:
            out[m.group(1)] = pid_s
    return out


def maybe_respawn(avail: float, n_train: int, names: list[str], update: int,
                  total: float = 0.0):
    """内存够就补 ent01_g999_s43。**两个视角都要过。**

    视角 A（增量）：MemAvailable 还够不够填未来的增长？
        在跑的各补足增长 + 新 run 从 0 到顶 + 安全垫
    视角 B（峰值）：全部跑到头时，机器还剩多少**绝对**余量？
        现在的占用 + 未来的增长，机器总量减去它 ≥ MIN_HEADROOM_GB

    为什么需要两个：视角 A 单独不够。2026-09-18 17:42，旧判据（只看新 run 自己）
    在余量 62.8G 时放行了第 4 个 run；我第一版"修正"改成视角 A，**变异测试证明
    它照样放行**（62.8 >= 55.98）。真正该拒绝的理由在视角 B：全部跑到头要
    108.6 GB，机器 125.4，余 16.8 < 17 —— 正好落在记录里那条"4 勉强"线上。

    视角 A 单独也不够的另一面：它是个**相对**判据，机器上别的东西（探针、
    pytest、ssh）吃掉的量它看不见；视角 B 是绝对的，看得见。

    `total` 缺省为 0 时**拒绝**，不是跳过视角 B。理由见函数体里的注释。
    """
    if any(n.startswith("ent01_g999_s43") for n in names):
        return "g999_s43 已在跑"
    st = {}
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text())
        except (OSError, json.JSONDecodeError):
            st = {}
    if st.get("hold"):
        return f"g999_s43 被 hold 住了：{st['hold']}"
    if st.get("done"):
        return f"g999_s43 已补过（{st.get('at')}）"

    # **没有 total 就拒绝，不能"跳过视角 B 继续"。**
    # 初稿写的是 `if total > 0:` ——缺 total 时静默降级成只看视角 A，而视角 A
    # 正是被证明会放行 17:42 那一次的那个判据。也就是说：一个**静默丢掉一半
    # 保护**的守卫。本项目已经有过一次同类的教训（阈值设错的看门狗精确杀掉了
    # 唯一想保的对照臂）——**建在错误数量级上的自动保护就是破坏措施，比没有更糟**。
    # 调用方（main）一定传 MemTotal，所以这条分支正常走不到；走到就说明调用方
    # 改了，宁可拒绝并报错，也不假装保护还在。
    if total <= 0:
        return ("g999_s43 未补：未拿到 MemTotal，无法做绝对余量判断"
                "（两个视角缺一个 → 按拒绝处理）")

    growth_running = n_train * RUN_GROWTH_GB * max(0, TARGET_UPDATES - update)
    need = growth_running + RUN_FULL_GB + SAFETY_GB
    if avail < need:
        return (f"g999_s43 未补：余量 {avail:.1f}G < 需 {need:.1f}G"
                f"（在跑 {n_train} 个还要涨 {growth_running:.1f}G"
                f" + 新 run 到顶 {RUN_FULL_GB:.1f}G + 垫 {SAFETY_GB:.0f}G）")

    peak = (total - avail) + growth_running + RUN_FULL_GB
    head = total - peak
    if head < MIN_HEADROOM_GB:
        return (f"g999_s43 未补：跑到头占 {peak:.1f}G / 机器 {total:.1f}G，"
                f"余 {head:.1f}G < 底线 {MIN_HEADROOM_GB:.0f}G"
                f"（记录：4 并发 108G 只是「勉强」）")

    cfg = ["configs/train_ent01.yaml", "configs/train_ent01_g999.yaml"]
    for c in cfg:
        if not (Path("/opt/qkd/graph_mappo") / c).exists():
            return f"g999_s43 未补：缺 {c}"
    name = "ent01_g999_s43_r2"
    if (OUT / name).exists():
        st["done"] = True
        st["at"] = time.strftime("%Y-%m-%d %H:%M")
        st["note"] = "outputs 已存在，未覆盖"
        STATE.write_text(json.dumps(st))
        return "g999_s43 未补：outputs 已存在"

    cmd = (
        "cd /opt/qkd/graph_mappo && setsid nohup env OMP_NUM_THREADS=4 "
        "MKL_NUM_THREADS=4 /opt/qkd/venv/bin/python -u "
        "scripts/train/train_graph_mappo.py --configs rl_algorithm.yaml "
        "train_full_rl.yaml train_ent01.yaml train_ent01_g999.yaml "
        "--checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt "
        f"--seed 43 --num-updates 30 --run-name {name} "
        f"> /tmp/{name}.log 2>&1 < /dev/null &"
    )
    subprocess.run(["bash", "-lc", cmd], check=False)
    st["done"] = True
    st["at"] = time.strftime("%Y-%m-%d %H:%M")
    st["avail_at_launch"] = round(avail, 1)
    STATE.write_text(json.dumps(st))
    return f"g999_s43 已补起（当时余量 {avail:.1f}G，需 {need:.1f}G）"


def finished_cleanly(run: str) -> bool:
    """这个 run 是**正常跑完**退出的吗？

    仅凭"run 数变少"分不清"跑完退出"与"被杀"。正常收尾有两个硬标志：
      1. `checkpoint_final.pt` 存在（`train()` 的收尾只在这里写）；
      2. 日志里有 `Final: UpdateStats` 那一行。
    两个都查，任一缺失就按"可疑"处理——宁可疑报，不可漏报。

    为什么要这个：本地唤醒链把"run 数变少"一律报成 `ALERT 疑似被杀`，
    于是每次正常收尾都误报一次事故。**误报的代价是真实的**——它训练使用者
    去忽视 ALERT，而 ALERT 是整条链里唯一的事故信号。
    """
    d = OUT / run
    if (d / "checkpoint_final.pt").exists():
        return True
    log = Path("/tmp") / f"{run}.log"
    try:
        if log.exists():
            tail = log.read_text(encoding="utf-8", errors="replace")[-8000:]
            if "Final: UpdateStats" in tail:
                return True
    except OSError:
        pass
    return False


def main():
    runs = trainers()
    names = sorted(runs)
    mi = meminfo()
    avail = mi.get("MemAvailable", -1.0)

    max_u = 0
    for n in names:
        u, _ = align(OUT / n)
        max_u = max(max_u, u or 0)
    respawn = maybe_respawn(avail, len(names), names, max_u,
                            mi.get("MemTotal", 0.0))

    print(f"MEM {avail:.1f}G swap {mi.get('SwapFree', -1):.1f}G "
          f"committed {mi.get('Committed_AS', -1):.0f}G runs {len(names)}")
    print(f"RESPAWN {respawn}")
    print("RUNNING " + " ".join(names) if names else "RUNNING (无)")

    # 已经收尾的 run：本地链据此区分"正常跑完"与"被杀"，
    # 否则每次正常结束都会误报一次 ALERT（已验证过一次）。
    done = sorted(
        p.name for p in OUT.iterdir()
        if p.is_dir() and p.name not in names and finished_cleanly(p.name)
    ) if OUT.exists() else []
    # 只报最近改动的那些，避免把几百个历史 run 全打出来（输出要稳定，
    # 否则指纹每次都变，又会每轮唤醒）。
    if done:
        recent = sorted(
            done, key=lambda n: (OUT / n / "metrics.jsonl").stat().st_mtime
            if (OUT / n / "metrics.jsonl").exists() else 0,
            reverse=True,
        )[:8]
        print("FINISHED " + " ".join(sorted(recent)))

    # 参照臂：所有实验都用同一个 BC 起点的专家/基线数字
    refs = {
        "expert_val240": 0.6980,
        "expert_val100_114": 0.6979,   # 同 regime（种子 100-114），配对比较的唯一口径
        "bc_val240": 0.6483,
        "r8base_u5": 0.6537, "r8base_u10": 0.6755, "r8base_u15": 0.6893,
        "r7base_u20": 0.6159, "r7fixent_u20": 0.7113,
        "ent01_u5": 0.6861,
    }
    for n in names:
        u, vals = align(OUT / n)
        if not vals:
            print(f"VAL {n} u={u} (无验证点)")
            continue
        s = " ".join(f"u{a}={b:.4f}" for a, b in vals)
        # 延长臂（u30→u50）：这一步唯一要回答的是"曲线还在不在上行、
        # 平台在哪"，所以直接与同 regime 的专家比，不必再绕 r8base。
        if "_u30to50" in n:
            cmp_s = "  vs_expert: " + " ".join(
                f"u{a} {b - refs['expert_val100_114']:+.4f}" for a, b in vals)
        else:
            cmp_s = ""
        # 与 r8base 同轮的对照（配对，才是可比口径）
        if n.startswith("ent01_s4"):
            mp = {"u5": refs["r8base_u5"], "u10": refs["r8base_u10"],
                  "u15": refs["r8base_u15"]}
            parts = []
            for a, b in vals:
                k = f"u{a}"
                if k in mp:
                    parts.append(f"{k} {b - mp[k]:+.4f}")
            if parts:
                cmp_s += "  vs_r8base: " + " ".join(parts)
        print(f"VAL {n} u={u} {s}{cmp_s}")

    # 旋钮臂 vs ent01 基线：**配对**差（同训练种子 → 同验证种子集）。
    # 这是本轮实验唯一要回答的问题，所以直接打出来，不必等我去手工配对：
    # 唤醒时看到的就是判决所需的那一行。
    #
    # ⚠ 2026-09-19 修：原来是 `d = pb[i] - pa[i]`（基线 − 臂），却标注成
    # 「臂 − 基线」，**符号与标注相反**。后果是 VERDICT 判读整体反向：
    # 实测 `PAIR vcoef1_s44 u=20 +0.0284 (臂 0.6889 − 基线 0.7173)` ——
    # 0.6889 − 0.7173 = **−0.0284**，打印成 +0.0284；于是 `m>=+0.035`
    # 这个「有效」判据实际命中的是**比基线差**的臂。已用算术复核。
    # 现在改成与标注、与阈值语义一致的 `臂 − 基线`。
    # ⚠ 每个臂的**基线是哪个，必须显式写**——不能一律拿 ent01 当基线。
    #    `vcoef1` 跑在 entropy_coef=0.001 上（resolved_config 实测），它要测的
    #    value_coef 0.5→1.0 只能和**同样 ent=0.001 的 r8_base** 比。
    #    拿它去比 ent01 等于把 entropy_coef 那一项也算进「旋钮效应」，读数偏大。
    #    （旧版就是这么比的：把 vcoef1 报成 +0.0717「有效」，而实际上它比自己的
    #      基线还差 0.02。）
    #    `ep2e1`/`mini512e1` 是本轮重做的、叠在 ent01 之上，所以基线是 ent01。
    #    计划中但已作废的 `ep2`/`mini512`/`runt` 保留映射，便于对照旧读数。
    ARM_BASE = {
        "vcoef1":    "r8_base",
        "ep2":       "ent01",      # 作废（跑在 ent=0.001 上）
        "mini512":   "ent01",      # 作废
        "runt":      "ent01",      # 作废（且注释与代码不符）
        "ep2e1":     "ent01",      # 本轮重做
        "mini512e1": "ent01",      # 本轮重做
    }
    for arm, base in ARM_BASE.items():
        tot = []
        for seed in ("42", "43", "44"):
            a = f"{arm}_s{seed}"
            b = f"{base}_s{seed}"
            sa, sb = series(OUT / a), series(OUT / b)
            if not sa or not sb:
                continue
            common = sorted(set(sa) & set(sb))
            if not common:
                continue
            u = common[-1]
            pa, pb = sa[u][1], sb[u][1]
            if len(pa) != len(pb) or not pa:
                continue
            d = [pa[i] - pb[i] for i in range(len(pa))]     # 臂 − 基线
            md = sum(d) / len(d)
            tot.append(md)
            print(f"PAIR {arm}_s{seed} u={u} {md:+.4f} "
                  f"(臂 {sa[u][0]:.4f} − 基线[{base}] {sb[u][0]:.4f})")
        if tot:
            m = sum(tot) / len(tot)
            tag = ("有效 >=+0.035" if m >= 0.035 else
                   "有害 <=-0.035" if m <= -0.035 else
                   "测不出（<0.035）")
            print(f"VERDICT {arm} (基线 {base}) 三种子均值 Δ={m:+.4f} → {tag}")


if __name__ == "__main__":
    main()
