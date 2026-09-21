"""筛「活旋钮」：非维度 + **确实被当前配置消费**。

## 为什么需要这一步

今晚两次踩到「键存在但走不到」：
  · `max_candidates_per_node` / `max_candidate_edges` / `tie_break`
    只在 `_resolve_max_weight_matching` 里用（`action_resolver.py:234/247`），
    当前是 `mutual_choice` ⟹ 死
  · `score_merge` 只在 `use_edge_scores=False` 分支用（`:181`），
    而 `score_source` 缺省 `"edge"` ⟹ 永远走不到 ⟹ 死

## 本脚本做法

给出**两段信息**，让人工判断谁活：
  A. 每个候选键的**取值**（从 resolved_config）
  B. **grep 声称的读取点**（文件:行号）

真正的死活判断要读控制流 —— 本脚本只把材料摆齐，
并**显式标注已知的守卫条件**（如 `if use_edge_scores`）。
"""
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
OUT = REPO / "outputs"

# 已知的「守卫条件」：键在什么分支下才有消费者
GUARDS = {
    "action_resolver.max_candidates_per_node":
        "只在 _resolve_max_weight_matching（:273）→ 当前 mode=mutual_choice ⟹ **死**",
    "action_resolver.max_candidate_edges":
        "同上（:274）⟹ **死**",
    "action_resolver.tie_break":
        "只被 _sort_candidates（:199/:216）调用 → 那两个都在 max_weight/priority 分支 ⟹ 需核 mutual_choice 路径",
    "action_resolver.score_merge":
        "只在 use_edge_scores=False 分支（:181）→ score_source 缺省 'edge' ⟹ **死**",
    "action_resolver.score_source":
        "缺省 'edge' ⟹ use_edge_scores=True ⟹ score_merge 分支走不到",
}

# 有意排除：明显与瓶颈无关或不可动
EXCLUDE = re.compile(
    r"^(project|runtime|seed|validation|env\.name|env\.day_steps|"
    r"rate_provider\.(h5|time|normalization)|qkp\.(type|overflow)|"
    r"reward\.|requests\.(type|source|pair_|hourly|steps_per_hour)|"
    r"model\.(name|distribution)|model\.encoder\.gnn_type)")


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def main():
    import yaml
    seen = defaultdict(set)
    for cfgp in OUT.glob("*/resolved_config.yaml"):
        try:
            c = yaml.safe_load(cfgp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(c, dict):
            continue
        for k, v in flatten(c).items():
            try:
                seen[k].add(json.dumps(v, sort_keys=True))
            except TypeError:
                pass

    DIM = re.compile(r"features\.|dims\.|_dim|history|hidden_dim|hidden_dims|num_layers")
    cands = [(k, list(v)[0]) for k, v in seen.items()
             if len(v) == 1 and not DIM.search(k) and not EXCLUDE.search(k)]
    cands.sort()

    print("=" * 96)
    print(f"候选「非维度 + 从未变过」旋钮：{len(cands)} 个（已排除维度链与无关项）")
    print("=" * 96)
    for k, v in cands:
        vs = v if len(v) < 22 else v[:21] + "…"
        guard = GUARDS.get(k, "")
        mark = "★死" if "**死**" in guard else ("?" if guard else "  ")
        print(f"\n  {mark} {k}")
        print(f"      当前值: {vs}")
        if guard:
            print(f"      守卫: {guard}")
        # grep 读取点
        r = subprocess.run(
            f"cd {REPO} && grep -rn '{k.split('.')[-1]}' qkd_rl/ --include=*.py",
            shell=True, capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.strip()][:3]
        for l in lines:
            print(f"      {l[:100]}")


if __name__ == "__main__":
    main()
