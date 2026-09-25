"""Build the experiment ledger and per-model result docs from raw artifacts.

Two outputs, both derived, both regenerable:

  1. ``outputs/model_ledger.jsonl`` -- ONE central ledger, one line per record.
     Never hand-edited; always rewritten from the artifacts below it.
  2. ``qkd_rl/model_zoo/<model>/RESULTS.md`` -- one result doc PER MODEL,
     generated from the ledger. This is the durable, committed, queryable
     artifact: "what has model vN scored, under which environment".

Raw sources are opened READ-ONLY and never modified. The frozen reports depend
on those bytes staying identical, so this script only ever *adds* files.

Why backfill at all: the frozen batches already carry self-describing eval JSON
(a ``policy`` field with the run path plus a complete ``protocol`` block), so
every field in 实验产物规范.md §4 is recoverable without guessing. Where a field
is NOT recoverable it is written as ``null`` rather than invented -- see the
honesty rules in that spec. Faking a protocol is the failure mode the spec
warns about: it does not error, it just lets someone draw a wrong comparison.

Usage:
    python scripts/diag/backfill_manifest.py --repo-root .            # report
    python scripts/diag/backfill_manifest.py --repo-root . --apply    # write
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

SCHEMA = "qkd-eval-v2"

#: Central ledger (gitignored, next to the other runtime outputs).
LEDGER_REL = "outputs/model_ledger.jsonl"

#: Per-model doc, generated from the ledger.
MODEL_DOC_REL = "qkd_rl/model_zoo/{model}/RESULTS.md"

SEED_RE = re.compile(r"s(\d+)(?=[^0-9]|$)")

#: batch dir -> (batch_id, node). The node is not in any artifact, so it is
#: stated here explicitly (from the server_results_<node>/ path component).
BATCHES: dict[str, tuple[str, str]] = {
    "server_results_node342/frozen_20260924": ("20260924-frozen-eval", "clnode342"),
    "server_results_node342/iso_attr_20260924": ("20260924-iso-attribution", "clnode342"),
    "server_results_node342/retrain_v3_8ep_20260925": ("20260925-retrain-v3-8ep", "clnode342"),
    "server_results_node342/diagnostics_v3_20260925": ("20260925-v3-diagnostics", "clnode342"),
    "server_results/pairhist_v4": ("v4-lstm-from-dminj30", "unknown-amd238-era"),
    "server_results/pairhist_v3": ("v3-from-dminj30", "unknown-amd238-era"),
}

#: Batches whose model cannot be read from a resolved config (none was archived)
#: and is instead stated explicitly. Kept separate from inference on purpose:
#: an inferred model that is wrong is worse than a null one.
BATCH_MODEL_HINT: dict[str, str] = {
    "v4-lstm-from-dminj30": "v4",
    "v3-from-dminj30": "v3",
}


# ---------------------------------------------------------------- loading

def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_yaml(path: Path) -> dict:
    """Load a resolved config. PyYAML only -- these files are YAML, not JSON."""
    try:
        import yaml
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _env_fingerprint(cfg: dict) -> dict:
    """The environment a number is valid under.

    Two readings are comparable only if this matches. Recorded per record rather
    than once per batch because the same batch can span configs, and because the
    whole point of the spec is that a number without its environment is unusable
    six months later.
    """
    env = cfg.get("env") or {}
    sc = cfg.get("scenario") or {}
    rq = cfg.get("requests") or {}
    rt = cfg.get("routing") or {}
    rp = cfg.get("rate_provider") or {}
    return {
        "env_name": env.get("name"),
        "episode_steps": env.get("episode_steps"),
        "episode_start_mode": env.get("episode_start_mode"),
        "day_steps": env.get("day_steps"),
        "activation_window": [
            env.get("activation_window_start_day"),
            env.get("activation_window_end_day"),
        ],
        "topology": {
            "num_gs": sc.get("num_gs"),
            "num_hap": sc.get("num_hap"),
            "num_sat": sc.get("num_sat"),
            "allowed_link_types": sc.get("allowed_link_types"),
        },
        "requests": {
            "type": rq.get("type"),
            "arrival_rate": rq.get("arrival_rate"),
            "amount_mean": rq.get("amount_mean"),
            "deadline_steps": rq.get("deadline_steps"),
            "pair_hotness": rq.get("pair_hotness"),
            "pair_zipf_s": rq.get("pair_zipf_s"),
            "pair_seed": rq.get("pair_seed"),
        },
        "routing": {
            "mode": rt.get("mode"),
            "path_selection": rt.get("path_selection"),
            "serve_order": rt.get("serve_order"),
        },
        "rate_provider": rp.get("provider"),
    }


def _env_fp_id(fp: dict) -> str:
    """Short stable id for an environment fingerprint, for compact tables."""
    blob = json.dumps(fp, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _seed_of(cfg_seed, manifest_seed) -> int | None:
    """A single train seed from either source.

    In the frozen batches ``seed`` is a dict {global_seed, env_seed}; in the
    newer runs manifest.json has a plain int. Prefer the explicit int.
    """
    if isinstance(manifest_seed, int):
        return manifest_seed
    if isinstance(cfg_seed, int):
        return cfg_seed
    if isinstance(cfg_seed, dict):
        for key in ("global_seed", "env_seed", "seed"):
            if isinstance(cfg_seed.get(key), int):
                return cfg_seed[key]
    return None


def _train_seed(policy: str) -> int | None:
    m = SEED_RE.search(Path(policy).name)
    return int(m.group(1)) if m else None


def _model_of(policy: str, arm_dir: str) -> str | None:
    name = f"{arm_dir} {policy}".lower()
    for tag in ("v4", "v3", "v2"):
        if re.search(rf"(?<![a-z0-9]){tag}(?![0-9])", name):
            return tag
    return None


def _is_isolated(arm_dir: str, policy: str) -> bool:
    return "iso" in arm_dir.lower() or "iso" in policy.lower()


# ---------------------------------------------------------------- eval records

def _record_from_eval_json(
    path: Path, batch_id: str, batch_dir: str, node: str,
    env_fp: dict | None, legacy: bool = False,
) -> dict | None:
    """One ledger record from a frozen_eval output JSON.

    Returns None when the file is not a frozen-protocol eval (no protocol block
    and no per-seed success), so the caller can treat it as legacy instead of
    writing a malformed record.
    """
    data = _load_json(path)
    if data is None:
        return None
    policy = str(data.get("policy", ""))
    protocol = data.get("protocol") or {}
    per_seed = data.get("per_seed_success")
    if not protocol.get("request_seeds") or not per_seed:
        return None

    arm_dir = path.stem
    arm = re.sub(r"^rl_outputs_experiments_", "", path.stem)
    arm = re.sub(r"_[AB]$", "", arm).replace("_", "-")

    is_expert = policy == "expert" or path.name.startswith("expert")
    model = None if is_expert else _model_of(policy, arm_dir)

    rec = {
        "schema": SCHEMA,
        "batch_id": batch_id,
        "directory": batch_dir,
        "arm": "expert" if is_expert else arm,
        "kind": "expert" if is_expert else "rl",
        "window": data.get("window") or protocol.get("window"),
        "run": {
            "model": "expert" if is_expert else model,
            "train_seed": None if is_expert else _train_seed(policy),
            "checkpoint": policy if not is_expert else None,
            "isolation": None if is_expert else _is_isolated(arm_dir, policy),
        },
        "protocol": {
            "start_day": protocol.get("start_day"),
            "end_day": protocol.get("end_day"),
            "request_seeds": protocol.get("request_seeds"),
            "episode_steps": protocol.get("episode_steps"),
            "start_mode": protocol.get("start_mode"),
            "deterministic": True,
        },
        "result": {
            "mean_success_rate": data.get("mean_success_rate"),
            "mean_key_efficiency": data.get("mean_key_efficiency"),
            "mean_served_keys": data.get("mean_served_keys"),
            "per_seed_success": per_seed,
            "per_seed_key_efficiency": data.get("per_seed_key_efficiency"),
        },
        "provenance": {
            # Honest nulls: absent from the archived artifacts, never invented.
            "node": node,
            "git_commit": None,
            "dirty": None,
            "evaluated_at_utc": data.get("evaluated_at"),
            "source_json": path.as_posix(),
        },
    }
    if env_fp is not None:
        # Copy and recompute the id from a CLEAN dict. Assigning the shared
        # batch-level dict and then mutating it in place made each record hash a
        # dict that already carried the PREVIOUS record's fingerprint_id, so one
        # identical environment produced a cascade of different ids -- exactly
        # the false "these are incomparable" signal this field exists to prevent.
        fp = {k: v for k, v in env_fp.items() if k != "fingerprint_id"}
        fp["fingerprint_id"] = _env_fp_id(fp)
        rec["environment"] = fp
    if legacy:
        rec["legacy"] = True
    return rec


def _eval_fingerprint(rec: dict) -> tuple:
    """Identity of the *measurement*, not of the file.

    Same (run, window, seeds, evaluated_at) == the SAME evaluation copied to two
    directories, not two independent runs. Measured here: the 20 records under
    iso_attr_20260924/frozen_eval_ppo_stage/ duplicate
    frozen_20260924/frozen_eval/ exactly, identical ``evaluated_at``
    (2026-09-24T15:26:38Z) and identical values -- files copied when the
    attribution run re-staged its inputs. Unlabelled they read as n=2.
    """
    return (
        rec["run"].get("checkpoint"),
        rec["window"],
        tuple(rec["protocol"].get("request_seeds") or ()),
        rec["provenance"].get("evaluated_at_utc"),
    )


# ---------------------------------------------------------------- train records

TRAIN_SUMMARY_KEYS = (
    "mean_success_rate", "critic_loss", "kl", "clip_frac",
    "n_minibatches", "mean_ratio", "actor_grad_norm",
)


def _training_records(bdir: Path, batch_id: str, batch_dir: str, node: str) -> list[dict]:
    """One record per training run, summarising its metrics.jsonl.

    Deduped across the whole batch by (run name, first-update metrics): the
    attribution batch archived two copies of the same runs (experiments_v3/
    and experiments_v3/v3/), which would otherwise appear as 2x the evidence.
    """
    out: list[dict] = []
    seen_run: set[tuple] = set()
    for mf in sorted(bdir.rglob("metrics.jsonl")):
        rows = []
        for line in mf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # The final line is NOT an update: the trainer appends one
            # {"eval_validation": {...}} summary block. Treating it as the last
            # update made every last_update field null.
            if "eval_validation" in obj:
                continue
            rows.append(obj)
        if not rows:
            continue
        run_dir = mf.parent
        arm = re.sub(r"[^a-z0-9-]+", "-", run_dir.name.lower()).strip("-")
        first, last = rows[0], rows[-1]

        dedup_key = (arm, tuple(sorted(
            (k, first.get(k)) for k in TRAIN_SUMMARY_KEYS
        )))
        if dedup_key in seen_run:
            continue
        seen_run.add(dedup_key)

        # resolved_config.yaml sits in the run dir. It is YAML: parsing it as
        # JSON silently yielded {} and blanked every protocol field.
        # Newer runs drop resolved_config.yaml in favour of experiment_config.yaml
        # plus a manifest.json; fall back rather than losing the whole record.
        cfg = _load_yaml(run_dir / "resolved_config.yaml")
        if not cfg:
            cfg = _load_yaml(run_dir / "experiment_config.yaml")
        manifest = _load_json(run_dir / "manifest.json") or {}
        val = cfg.get("validation") or {}
        exp = cfg.get("experiment") or {}
        ppo = ((cfg.get("train") or {}).get("ppo")) or {}

        def g(d, *keys):
            cur = d
            for k in keys:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(k)
            return cur

        model = exp.get("model") or manifest.get("model") or BATCH_MODEL_HINT.get(batch_id)
        ppo_moved = any(
            (r.get("kl") or 0.0) > 0 or (r.get("clip_frac") or 0.0) > 0 for r in rows
        )
        rec = {
            "schema": SCHEMA,
            "batch_id": batch_id,
            "directory": batch_dir,
            "arm": arm,
            "kind": "train",
            "window": None,
            "run": {
                "model": model,
                # manifest.json carries a real train seed; the config's `seed`
                # is a dict {global_seed, env_seed} in the frozen batches.
                "train_seed": _seed_of(cfg.get("seed"), manifest.get("seed")),
                "checkpoint": None,
                "isolation": None,
                "ppo_updates": len(rows),
                "config": {
                    "episodes_per_update": g(cfg, "train", "episodes_per_update"),
                    "minibatch_size": ppo.get("minibatch_size"),
                    "epochs": ppo.get("epochs"),
                    "actor_lr": g(cfg, "train", "optimizer", "actor_lr"),
                    "entropy_coef": ppo.get("entropy_coef"),
                },
            },
            "protocol": {
                "start_day": g(val, "window", "start_day"),
                "end_day": g(val, "window", "end_day"),
                "request_seeds": val.get("seeds") or val.get("request_seeds"),
                "episode_steps": val.get("episode_steps"),
                "start_mode": val.get("start_mode"),
                "deterministic": None,
            },
            "result": {
                "first_update": {k: first.get(k) for k in TRAIN_SUMMARY_KEYS},
                "last_update": {k: last.get(k) for k in TRAIN_SUMMARY_KEYS},
                "ppo_moved": ppo_moved,
            },
            "provenance": {
                "node": node,
                # manifest.json (newer runs) has the real code fingerprint; the
                # older batches have neither a commit nor a manifest -> null.
                "git_commit": None,
                "dirty": None,
                "config_sha256": manifest.get("config_sha256"),
                "model_sources_sha256": manifest.get("model_sources_sha256"),
                "started_at_utc": manifest.get("started_at"),
                "finished_at_utc": manifest.get("finished_at"),
                "evaluated_at_utc": None,
                "source_json": mf.as_posix(),
            },
        }
        if cfg:
            fp = _env_fingerprint(cfg)
            fp["fingerprint_id"] = _env_fp_id(fp)
            rec["environment"] = fp
        out.append(rec)
    return out


# ---------------------------------------------------------------- validation

def _validate(rec: dict, label: str) -> list[str]:
    if rec.get("kind") == "train":
        return []
    problems: list[str] = []
    seeds = rec["protocol"]["request_seeds"] or []
    for key in ("per_seed_success", "per_seed_key_efficiency"):
        arr = rec["result"].get(key)
        if arr is not None and len(arr) != len(seeds):
            problems.append(f"{label}: {key} len {len(arr)} != seeds {len(seeds)}")
    return problems


# ---------------------------------------------------------------- model docs

def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)


def _env_row(fp: dict) -> list[str]:
    topo = fp.get("topology") or {}
    rq = fp.get("requests") or {}
    rt = fp.get("routing") or {}
    aw = fp.get("activation_window") or [None, None]
    return [
        fp.get("fingerprint_id", "—"),
        f"{aw[0]}–{aw[1]}",
        _fmt(fp.get("episode_steps")),
        _fmt(fp.get("episode_start_mode")),
        f"{topo.get('num_gs')}/{topo.get('num_hap')}/{topo.get('num_sat')}",
        f"{rq.get('type')} rate={rq.get('arrival_rate')} amt={rq.get('amount_mean')} dl={rq.get('deadline_steps')}",
        f"{rq.get('pair_hotness')} s={rq.get('pair_zipf_s')}",
        f"{rt.get('mode')}/{rt.get('path_selection')}",
        _fmt(fp.get("rate_provider")),
    ]


ENV_HEADER = ["env_fp", "训练窗(激活)", "回合步", "起始模式", "拓扑 gs/hap/sat",
              "请求", "配对热度", "路由", "速率源"]


def _write_model_doc(root: Path, model: str, records: list[dict]) -> Path | None:
    ev = [r for r in records if r["kind"] in ("rl", "expert")]
    tr = [r for r in records if r["kind"] == "train"]
    if not ev and not tr:
        return None

    lines: list[str] = []
    lines.append(f"# {model} 实验结果台账")
    lines.append("")
    lines.append("> **本文件由脚本生成，不要手改表格。** 生成命令：")
    lines.append(">")
    lines.append("> ```bash")
    lines.append("> python scripts/diag/backfill_manifest.py --repo-root . --apply")
    lines.append("> ```")
    lines.append(">")
    lines.append("> 数据源：`outputs/model_ledger.jsonl`（中心台账，由原始产物重建）。")
    lines.append("> 手写内容写在文末「设计要点」「备注」段，重新生成时原样保留。")
    lines.append("")

    # --- environment table
    fps: dict[str, dict] = {}
    for r in ev + tr:
        fp = r.get("environment")
        if fp:
            fps[fp.get("fingerprint_id")] = fp
    lines.append("## 测试环境")
    lines.append("")
    if fps:
        lines.append("**`env_fp` 不同 = 数字不可直接配对比较。**")
        lines.append("")
        lines.append("| " + " | ".join(ENV_HEADER) + " |")
        lines.append("|" + "---|" * len(ENV_HEADER))
        for fid, fp in sorted(fps.items()):
            lines.append("| " + " | ".join(_env_row(fp)) + " |")
    else:
        lines.append("**无配置存档，环境不可恢复**（该批次的 `resolved_config.yaml` 未回传）。")
        lines.append("引用这些数字前必须补记环境。")
    lines.append("")

    # --- results by window
    for window in ("A", "B"):
        wr = [r for r in ev if r["window"] == window]
        if not wr:
            continue
        lines.append(f"## 验证结果 · 窗口 {window}")
        lines.append("")
        s0 = wr[0]["protocol"]
        lines.append(f"协议：第 {s0['start_day']}–{s0['end_day']} 天，"
                     f"{len(s0['request_seeds'] or [])} 个请求种子 "
                     f"`{_fmt(s0['request_seeds'])}`，每集 {s0['episode_steps']} 步，"
                     f"起始模式 `{s0['start_mode']}`。")
        lines.append("")
        lines.append("| 批次 | arm | 模型 | 训练种子 | 成功率 | 密钥比 | 隔离 | env_fp | 重复 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(wr, key=lambda x: (x["batch_id"], x["arm"])):
            dup = "是" if "duplicate_of" in r else ""
            iso = r["run"].get("isolation")
            iso_s = "—" if iso is None else ("是" if iso else "否")
            efp = (r.get("environment") or {}).get("fingerprint_id", "—")
            lines.append(
                f"| {r['batch_id']} | {r['arm']} | {r['run'].get('model') or '—'} | "
                f"{_fmt(r['run'].get('train_seed'))} | "
                f"**{_fmt(r['result'].get('mean_success_rate'))}** | "
                f"{_fmt(r['result'].get('mean_key_efficiency'))} | {iso_s} | {efp} | {dup} |"
            )
        lines.append("")
        lines.append("> 「重复 = 是」表示该条与更早一条是同一次测量（同 `evaluated_at`、同数值）"
                     "被复制到两个目录，**不构成独立证据**。")
        lines.append("")

    # --- training records
    if tr:
        lines.append("## 训练记录")
        lines.append("")
        lines.append("`PPO 更新` 列为「否」= 全程 `kl=0` 且 `clip_frac=0`，"
                     "即 PPO 从未真正更新（预算配置错误），该次训练不构成架构结论。")
        lines.append("")
        lines.append("| 批次 | arm | 模型 | 轮数 | 是否多 minibatch | PPO 更新 | 每轮集数 | minibatch |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(tr, key=lambda x: (x["batch_id"], x["arm"])):
            c = r["run"]["config"]
            nb = (r["result"]["last_update"] or {}).get("n_minibatches")
            lines.append(
                f"| {r['batch_id']} | {r['arm']} | {r['run'].get('model') or '—'} | "
                f"{r['run']['ppo_updates']} | {'是' if (nb or 0) > 1 else '否'} | "
                f"{'是' if r['result']['ppo_moved'] else '**否**'} | "
                f"{_fmt(c.get('episodes_per_update'))} | {_fmt(c.get('minibatch_size'))} |"
            )
        lines.append("")

    # --- hand-written sections, preserved across regeneration.
    # Everything from the first hand-written heading onward is kept verbatim, so
    # the design notes and caveats written here survive every rebuild. Only the
    # generated tables above them are rewritten.
    tail = ""
    doc = root / MODEL_DOC_REL.format(model=model)
    if doc.exists():
        old = doc.read_text(encoding="utf-8")
        for marker in ("## 设计要点", "## 备注"):
            if marker in old:
                tail = old[old.index(marker):]
                break
    lines.append(tail if tail else _EMPTY_TAIL)
    lines.append("")

    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")
    return doc


_EMPTY_TAIL = """## 设计要点

（本段由人工维护，脚本重新生成时保留。写该模型专属的设计、归属与已知限制。）

## 备注

（本段由人工维护，脚本重新生成时保留。）"""


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--apply", action="store_true", help="write ledger + model docs")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()

    all_records: list[dict] = []
    problems: list[str] = []
    seen_fp: dict[tuple, str] = {}

    for rel, (batch_id, node) in BATCHES.items():
        bdir = root / rel
        if not bdir.is_dir():
            print(f"[skip] {rel} (missing)")
            continue

        # One env fingerprint per batch: every run in a batch resolved the same
        # config in these archives (verified), so read it once from any run.
        cfg_path = next(iter(sorted(bdir.rglob("resolved_config.yaml"))), None)
        env_fp = _env_fingerprint(_load_yaml(cfg_path)) if cfg_path else None

        records: list[dict] = []
        seen_arm: set[tuple] = set()
        n_dup = 0
        for jf in sorted(bdir.rglob("*.json")):
            rec = _record_from_eval_json(jf, batch_id, rel, node, env_fp)
            if rec is None:
                continue
            key = (rec["arm"], rec["window"])
            if key in seen_arm:
                continue
            seen_arm.add(key)
            problems += _validate(rec, jf.name)
            fp = _eval_fingerprint(rec)
            if fp in seen_fp:
                rec["duplicate_of"] = seen_fp[fp]
                n_dup += 1
            else:
                seen_fp[fp] = f"{batch_id}:{rec['arm']}:{rec['window']}"
            records.append(rec)

        train = _training_records(bdir, batch_id, rel, node)
        records += train
        n_eval = len(records) - len(train)
        print(f"[{batch_id}] dir={rel}  {n_eval} eval + {len(train)} train "
              f"({n_dup} dup)")
        all_records += records

    # Model docs are grouped by model across ALL batches -- that is the point of
    # "每个模型文件夹下有自己的记录". The ledger stays flat; only the views split.
    by_model: dict[str, list[dict]] = {}
    unmatched = 0
    for r in all_records:
        m = r["run"].get("model")
        if r["kind"] == "expert":
            for mm in ("v2", "v3"):          # expert is the shared anchor
                by_model.setdefault(mm, []).append(r)
            continue
        if m in ("v2", "v3", "v4"):
            by_model.setdefault(m, []).append(r)
        else:
            unmatched += 1

    n_eval_all = sum(1 for r in all_records if r["kind"] in ("rl", "expert"))
    print(f"\nledger: {len(all_records)} records "
          f"({n_eval_all} eval, {len(all_records) - n_eval_all} train)")
    for m, rs in sorted(by_model.items()):
        ne = sum(1 for r in rs if r["kind"] in ("rl", "expert"))
        nt = sum(1 for r in rs if r["kind"] == "train")
        print(f"  model {m}: {ne} eval + {nt} train")
    if unmatched:
        print(f"  UNMATCHED model (no docs written): {unmatched}")

    if problems:
        print("\nALIGNMENT PROBLEMS:")
        for p in problems:
            print("  " + p)
    else:
        print("alignment: OK (per_seed_* length == request_seeds length)")

    if not args.apply:
        print("\n(dry run -- pass --apply to write ledger and model docs)")
        return

    ledger = root / LEDGER_REL
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n-> wrote {ledger} ({len(all_records)} lines)")

    for model in ("v2", "v3", "v4"):
        doc = _write_model_doc(root, model, by_model.get(model, []))
        if doc:
            print(f"-> wrote {doc.relative_to(root).as_posix()} "
                  f"({len(by_model[model])} records)")


if __name__ == "__main__":
    main()
