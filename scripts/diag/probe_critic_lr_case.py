"""critic_lr 值不值得测？—— 先算分辨率，再决定起不起臂。

## 已知（docs/训练诊断记录.md:12403-12406）
· `critic_lr` **不是空旋钮**（裁剪只削掉约 1/3 有效步长）
· 但**信号被压扁**：71% 的 update 上有效步长只有名义值的 ~63%
· ⟹ 「需要的种子数比平时更多」

## 本脚本做三件事
① 从已跑臂的 `mean_abs_advantage` 等字段，估计 critic 漂移的可影响幅度
② 用 BC 族实测 SD（0.0137）算：若效应量是 X，需要 n 多少
③ 给出**是否值得起臂**的判断
"""
import json, math, statistics
from pathlib import Path
OUT = Path("/opt/qkd/graph_mappo/outputs")
SD_FAMILY = 0.0137      # BC 族实测（n=15）

def series(run, key):
    out = {}
    p = OUT/run/"metrics.jsonl"
    if not p.exists(): return out
    for line in p.read_text(encoding="utf-8",errors="replace").splitlines():
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: continue
        if "update" in o:
            out[o["update"]] = o.get(key)
    return out

print("="*88)
print("critic_lr 的可测性评估")
print("="*88)

# ① 先看 critic 学习率实际值与其他超参的量级
import yaml
c = yaml.safe_load(open(OUT/"ent01_rerun_s42"/"resolved_config.yaml", encoding="utf-8"))
ppo = c["train"]["ppo"]; opt = c["train"]["optimizer"]
print(f"\n  ① 当前超参")
print(f"     actor_lr={opt['actor_lr']}  critic_lr={opt['critic_lr']}"
      f"  (比 {opt['critic_lr']/opt['actor_lr']:.2f}×)")
print(f"     max_grad_norm={ppo['max_grad_norm']}  value_coef={ppo['value_coef']}")
print(f"     clip_eps={ppo['clip_eps']}  target_kl={ppo['target_kl']}")

# ② 实测 critic 的"可影响幅度"：看 value_return_corr 与 |advantage|
print(f"\n  ② 实测 critic 健康度（ent01_rerun_s42）")
for k in ("mean_abs_advantage","critic_loss","value_return_corr","V_std"):
    s = series("ent01_rerun_s42", k)
    if s and any(v is not None for v in s.values()):
        vs=[v for v in s.values() if v is not None]
        print(f"     {k}: u1={vs[0]:.4f}  末={vs[-1]:.4f}  均值={statistics.mean(vs):.4f}")

# ③ 分辨率算术
print(f"\n  ③ 分辨率算术（BC 族 SD = {SD_FAMILY}）")
print(f"     {'想测出的 Δ':<14}{'所需 n':>8}{'成本(小时)':>12}")
for d in (0.005, 0.010, 0.015, 0.020, 0.030):
    # n ≈ (t*SD/d)^2，t 随 df 变，用迭代
    n = 2
    for _ in range(50):
        tcrit = {2:4.303,4:2.776,9:2.262,14:2.145,19:2.093,29:2.045}.get(n-1, 2.0)
        n_new = math.ceil((tcrit*SD_FAMILY/d)**2)
        if n_new == n: break
        n = max(2, n_new)
    print(f"     {d:<14.3f}{n:>8}{n*1.2:>12.1f}")

print(f"\n  ④ 判断")
print(f"     若 critic_lr 的效应量在 1~2 点（与今晚其他增益类旋钮同量级）")
print(f"     ⟹ 需 n≈10~40 ⟹ 成本 12~48 小时")
print(f"     ⚠ 而今晚 4 条增益类旋钮**全部**在 ±1 点内 ⟹ 先验上 critic_lr 大概率也是")
