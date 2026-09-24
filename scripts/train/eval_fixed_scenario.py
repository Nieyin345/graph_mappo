"""Compatibility wrapper; canonical implementation lives in scripts/eval/."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "eval" / "eval_fixed_scenario.py"
    runpy.run_path(str(target), run_name="__main__")
