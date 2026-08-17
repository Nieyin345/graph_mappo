"""Training process manager for the QKD-RL UI.

Manages subprocess training, monitors metrics.jsonl in real-time,
and provides status updates to the Streamlit app.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


class TrainProcess:
    """Manages a single training subprocess with real-time monitoring."""

    def __init__(self, name: str, output_dir: Path):
        self.name = name
        self.output_dir = output_dir
        self._process: subprocess.Popen | None = None
        self._lock = Lock()
        self._metrics_cache: list[dict] = []
        self._eval_cache: list[dict] = []
        self._console_lines: list[str] = []
        self._status = "idle"  # idle | running | stopping | done | failed
        self._monitor_thread: Thread | None = None
        self._error: str | None = None
        self._metrics_path = output_dir / "metrics.jsonl"
        self._last_metrics_size = 0

    @property
    def status(self) -> str:
        return self._status

    @property
    def metrics(self) -> list[dict]:
        with self._lock:
            return list(self._metrics_cache)

    @property
    def eval_records(self) -> list[dict]:
        with self._lock:
            return list(self._eval_cache)

    @property
    def console(self) -> str:
        with self._lock:
            return "\n".join(self._console_lines[-200:])

    @property
    def error(self) -> str | None:
        return self._error

    def start(self, command: list[str], env: dict | None = None) -> None:
        """Start training in a subprocess."""
        with self._lock:
            if self._process is not None:
                return
            self._status = "running"
            self._console_lines = []
            self._metrics_cache = []
            self._eval_cache = []
            self._error = None
            self._last_metrics_size = 0

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Merge env
        proc_env = os.environ.copy()
        proc_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        if env:
            proc_env.update(env)

        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=proc_env,
            text=True,
            bufsize=1,
        )
        # Save PID so the UI can re-attach after restart
        self._save_pid()

        self._monitor_thread = Thread(target=self._monitor, daemon=True)
        self._monitor_thread.start()

    def _save_pid(self) -> None:
        """Write the PID to a file so the UI can re-attach after restart."""
        if self._process:
            pid_path = self.output_dir / ".train_pid"
            try:
                pid_path.write_text(str(self._process.pid), encoding="utf-8")
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the training process."""
        with self._lock:
            if self._process is None:
                return
            self._status = "stopping"

        # Try graceful shutdown first
        if self._process and self._process.poll() is None:
            if sys.platform == "win32":
                self._process.terminate()
            else:
                self._process.send_signal(signal.SIGTERM)

            # Wait up to 5 seconds
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if self._process:
                    self._process.kill()
                    self._process.wait()

        with self._lock:
            self._status = "idle"
            self._process = None

    def _monitor(self) -> None:
        """Background thread: read stdout + metrics.jsonl."""
        try:
            # Read stdout line by line
            if self._process and self._process.stdout:
                for line in iter(self._process.stdout.readline, ""):
                    with self._lock:
                        self._console_lines.append(line.rstrip())
                    # Check metrics file periodically
                    self._poll_metrics()

            self._process.stdout.close()
            exit_code = self._process.wait() if self._process else 0

            with self._lock:
                if self._status == "stopping":
                    self._status = "idle"
                elif exit_code == 0:
                    self._status = "done"
                else:
                    self._status = "failed"
                    self._error = f"Process exited with code {exit_code}"

        except Exception as e:
            with self._lock:
                self._status = "failed"
                self._error = str(e)

    def _poll_metrics(self) -> None:
        """Read new metrics from metrics.jsonl."""
        try:
            if not self._metrics_path.is_file():
                return
            current_size = self._metrics_path.stat().st_size
            if current_size <= self._last_metrics_size:
                return

            with self._metrics_path.open("r", encoding="utf-8") as f:
                f.seek(self._last_metrics_size)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if "update" in record:
                            self._metrics_cache.append(record)
                        elif "eval" in record or "eval_validation" in record:
                            self._eval_cache.append(record)
                    except json.JSONDecodeError:
                        pass
                self._last_metrics_size = f.tell()
        except Exception:
            pass

    def poll(self) -> None:
        """Call from Streamlit to refresh metrics."""
        self._poll_metrics()
        # Check if process is still alive
        if self._process and self._process.poll() is not None and self._status == "running":
            # Process finished without being caught by monitor thread
            self._monitor()


# Global singleton
_training_registry: dict[str, TrainProcess] = {}
_registry_lock = Lock()


def get_train_process(name: str) -> TrainProcess | None:
    with _registry_lock:
        return _training_registry.get(name)


def register_train_process(name: str, proc: TrainProcess) -> None:
    with _registry_lock:
        _training_registry[name] = proc


def unregister_train_process(name: str) -> None:
    with _registry_lock:
        _training_registry.pop(name, None)


def generate_command(
    profile_name: str,
    run_name: str,
    overrides: dict | None = None,
    checkpoint: str | None = None,
    config_files: list[str] | None = None,
) -> list[str]:
    """Generate the command list for subprocess."""
    cmd = [
        sys.executable,
        "scripts/train_graph_mappo.py",
        f"--mode={profile_name}",
        f"--run-name={run_name}",
    ]
    if checkpoint:
        cmd.append(f"--checkpoint={checkpoint}")
    if config_files:
        cmd.append("--configs")
        cmd.extend(config_files)
    return cmd


def generate_override_yaml(overrides: dict) -> Path | None:
    """Write user overrides to a temp yaml file and return path."""
    if not overrides:
        return None
    path = ROOT / "outputs" / ".ui_override.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(overrides, f, sort_keys=False, allow_unicode=True)
    return path