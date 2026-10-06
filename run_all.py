"""
Run the whole pipeline in order.

Usage (from the repository root):
    python run_all.py

Runs every script in src/ whose name starts with a number (01_, 02_, ...),
in numeric order, and stops with a clear message if any step fails.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

scripts = sorted(p for p in SRC.glob("[0-9][0-9]_*.py"))
if not scripts:
    sys.exit("No numbered scripts found in src/ (expected names like 01_preprocess.py).")

print(f"Running {len(scripts)} step(s) from {SRC}\n")
start_all = time.time()

for i, script in enumerate(scripts, 1):
    print("=" * 70)
    print(f"[{i}/{len(scripts)}] {script.name}")
    print("=" * 70)
    start = time.time()
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nStopped: {script.name} failed (exit code {result.returncode}). "
                 "Fix the error above and run again.")
    print(f"-> {script.name} finished in {time.time() - start:.1f}s\n")

print(f"All steps completed in {time.time() - start_all:.1f}s.")
print("Outputs: models/ (trained models), reports/figures/ (charts), "
      "reports/model_metrics.csv (metrics).")
