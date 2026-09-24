from pathlib import Path

import pytest

from qkd_rl.core.config import resolve_config_path


def test_resolve_config_path_active_and_archive(tmp_path: Path) -> None:
    (tmp_path / "active.yaml").write_text("x: 1\n", encoding="utf-8")
    archive = tmp_path / "archive" / "legacy"
    archive.mkdir(parents=True)
    (archive / "old.yaml").write_text("x: 0\n", encoding="utf-8")

    assert resolve_config_path(tmp_path, "active.yaml") == (tmp_path / "active.yaml").resolve()
    assert resolve_config_path(tmp_path, "archive/legacy/old.yaml") == (archive / "old.yaml").resolve()

    with pytest.raises(FileNotFoundError, match=r"archived at 'archive[/\\]legacy[/\\]old.yaml'"):
        resolve_config_path(tmp_path, "old.yaml")


def test_resolve_config_path_rejects_escape_and_absolute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        resolve_config_path(tmp_path, "../outside.yaml")
    with pytest.raises(ValueError, match="must be relative"):
        resolve_config_path(tmp_path, tmp_path / "active.yaml")
