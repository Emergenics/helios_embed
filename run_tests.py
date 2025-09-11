# --- START OF FILE run_tests.py (CORRECTED) ---
#!/usr/bin/env python3
# HELIOS_EMBED/run_tests.py — PR CI test dispatcher (CPU-first, fast)
# Behavior:
# - Runs only short tests by default.
# - Auto-skips GPU/CUDA paths when torch.cuda.is_available() is False.
# - Optional --run-long-tests flag for scheduled/non-blocking workflows.

from pathlib import Path
import argparse
import subprocess
import sys

def _run_py(script: str) -> int:
    cmd = [sys.executable, script]
    try:
        print(f"==> Running {script}")
        return subprocess.call(cmd)
    except Exception as e:
        print(f"[ERROR] {script} failed to launch: {e}", flush=True)
        return 1

def _has_gpu() -> bool:
    try:
        import torch
        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        return False

def main() -> int:
    parser = argparse.ArgumentParser(description="Helios.Embed test runner (CPU-fast)")
    parser.add_argument("--run-long-tests", action="store_true",
                        help="Include long-running scalability benchmarks.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    tests_dir = project_root  # adjust if your scripts live elsewhere

    short = [
        "benchmark_edge_cases.py",
        "benchmark_hardening.py",
        "benchmark_nystrom.py",
        "benchmark_streaming.py",
        "test_correctness.py",
        "test_sanitizers.py",
        "benchmark_micro.py"
    ]
    long = ["benchmark_scalability.py"]

    gpu_present = _has_gpu()
    if not gpu_present:
        print("GPU not detected: CUDA-dependent paths will be skipped.")

    selected = list(short)
    if args.run_long_tests:
        print("Including long-running tests (scalability).")
        selected.extend(long)

    rc = 0
    for name in selected:
        script = tests_dir / name
        if not script.exists():
            print(f"[WARN] Skipping missing test: {name}")
            continue
        # If script name implies GPU-only logic and no GPU is present, skip.
        if ("gpu" in name.lower() or "cuda" in name.lower()) and not gpu_present:
            print(f"[SKIP] {name} (no GPU available)")
            continue
        rc |= _run_py(str(script))

    return rc

if __name__ == "__main__":
    sys.exit(main())
# --- END OF FILE run_tests.py (CORRECTED) ---




