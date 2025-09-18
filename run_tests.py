# --- START OF FILE HELIOS_EMBED/run_tests.py (FINAL v1.4 - Robust PYTHONPATH) ---
import sys
import os
from pathlib import Path
import subprocess
import argparse

# --- THIS IS THE CRITICAL FIX ---
# Define paths relative to this script's location.
project_root = Path(__file__).parent.resolve()
src_path = project_root / 'src'
tests_path = project_root / 'tests'

def run_test(script_path):
    """Runs a single test script in a subprocess with a corrected PYTHONPATH."""
    print("="*80)
    print(f"🚀 EXECUTING TEST: {script_path.name}")
    print("="*80)
    
    # Construct the correct PYTHONPATH: it needs both the project root AND the src dir.
    env = os.environ.copy()
    current_python_path = env.get('PYTHONPATH', '')
    # Prepend our paths to ensure they are found first.
    env['PYTHONPATH'] = f"{str(project_root)}{os.pathsep}{str(src_path)}{os.pathsep}{current_python_path}"

    try:
        # We run the script from the project root directory to ensure all relative paths are correct.
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True, capture_output=True, text=True,
            cwd=project_root, # Set the working directory
            env=env # Pass the corrected environment
        )
        print(result.stdout)
        if result.stderr:
             print("--- STDERR ---")
             print(result.stderr)
        print(f"\n--- ✅ TEST SUCCEEDED: {script_path.name} ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n--- ❌ TEST FAILED WITH EXIT CODE {e.returncode}: {script_path.name} ---")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        print("-----------------------------------------------------")
        return False
# --- END OF CRITICAL FIX ---


if __name__ == "__main__":
    # --- (The rest of the script is largely unchanged) ---
    parser = argparse.ArgumentParser(description="Helios.Embed Test and Benchmark Runner")
    parser.add_argument(
        '--run-long-tests',
        action='store_true',
        help="Include long-running benchmarks like the full scalability suite."
    )
    args = parser.parse_args()

    short_tests = [
        "benchmark_edge_cases.py",
        "benchmark_hardening.py",
        "test_correctness.py",
        "benchmark_nystrom.py",
        "benchmark_streaming.py",
        "benchmark_micro.py",
        "benchmark_production_readiness.py"
    ]
    
    long_tests = ["benchmark_scalability.py"]

    tests_to_run_names = short_tests
    if args.run_long_tests:
        tests_to_run_names.extend(long_tests)
    
    tests_to_run = [tests_path / name for name in tests_to_run_names]
    
    print(f"Found {len(tests_to_run)} test scripts to execute.")
    
    print("\n--- 🛠️  Performing a clean build... ---")
    build_process = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        capture_output=True, text=True, cwd=project_root
    )
    if build_process.returncode != 0:
        print("--- ❌ CRITICAL BUILD FAILURE ---")
        print(build_process.stdout)
        print(build_process.stderr)
        sys.exit(1)
    print("--- ✅ Build successful. ---")
    
    failures = []
    for script_path in tests_to_run:
        if not script_path.exists():
            print(f"[WARN] Skipping missing test: {script_path.name}")
            continue
        if not run_test(script_path):
            failures.append(script_path.name)
            
    print("\n" + "="*80)
    if not failures:
        print("✅✅✅ MONUMENTAL SUCCESS: The entire Helios.Embed test suite passed.")
    else:
        print(f"❌ FAILURE: The following {len(failures)} benchmarks failed: {', '.join(failures)}")
        sys.exit(1)
    print("="*80)
# --- END OF FILE HELIOS_EMBED/run_tests.py (FINAL v1.4 - Robust PYTHONPATH) ---