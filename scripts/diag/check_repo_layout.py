"""Fail fast when scratch/config/document layout regresses."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    errors: list[str] = []

    tracked_tmp = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", ".tmp"],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout.splitlines()
    live_tracked_tmp = [rel for rel in tracked_tmp if (ROOT / rel).exists()]
    if live_tracked_tmp:
        errors.append("tracked scratch files: " + ", ".join(live_tracked_tmp[:10]))

    allowed_root_md = {"README.md", "CLAUDE.md"}
    extra_root_md = sorted(p.name for p in ROOT.glob("*.md") if p.name not in allowed_root_md)
    if extra_root_md:
        errors.append("root research/docs should live under docs/: " + ", ".join(extra_root_md))

    basenames: dict[str, list[str]] = defaultdict(list)
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        rel = path.relative_to(ROOT / "configs").as_posix()
        basenames[path.name].append(rel)
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid YAML {rel}: {exc}")
            continue
        if value is not None and not isinstance(value, dict):
            errors.append(f"config {rel} must contain a top-level mapping")
    for name, paths in sorted(basenames.items()):
        if len(paths) > 1:
            errors.append(f"duplicate config basename {name}: {paths}")

    if errors:
        print("REPO_LAYOUT_FAIL")
        for error in errors:
            print(" -", error)
        return 1
    print(f"REPO_LAYOUT_OK configs={sum(len(v) for v in basenames.values())} tracked_tmp_live=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
