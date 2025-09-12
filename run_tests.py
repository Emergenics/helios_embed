# --- START OF FILE run_tests.py (MINIMAL UPDATE) ---
import subprocess
import sys
from pathlib import Path
import os
import argparse  # ✅ minimal addition

def run_test(script_path):
    print("="*80)
    print(f"🚀 EXECUTING BENCHMARK: {script_path.name}")
    print("="*80)
    
    env = os.environ.copy()
    project_root = Path(__file__).parent.resolve()
    src_path = project_root / 'src'
    # Programmatically add src/ to the PYTHONPATH for the subprocess
    env["PYTHONPATH"] = f"{str(src_path)}{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    result = subprocess.run(
        [sys.executable, str(script_path)], 
        capture_output=True, text=True, env=env, cwd=project_root
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"--- ❌ BENCHMARK FAILED WITH EXCEPTION: {script_path.name} ---")
        print(result.stderr)
        print("-" * 50)
        return False
        
    print(f"--- ✅ BENCHMARK SUCCEEDED: {script_path.name} ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helios.Embed test/benchmark runner")
    parser.add_argument(
        "--run-long-tests",
        action="store_true",
        help="Include long-running tests (benchmark_scalability.py, benchmark_streaming.py)."
    )
    args = parser.parse_args()

    tests_dir = Path(__file__).parent.resolve() / "tests"
    
    # Short tests (default set for PR CI / fast runs)
    benchmark_scripts = [
        "benchmark_edge_cases.py",
        "benchmark_hardening.py",
        "benchmark_nystrom.py",
        "test_correctness.py",
        "test_sanitizers.py",
        "benchmark_micro.py"
    ]

    # Long-running tests (opt-in)
    long_tests = [
        "benchmark_scalability.py",
        "benchmark_streaming.py"
    ]

    if args.run_long_tests:
        print("--- 🔬 Including long-running tests (scalability, streaming) ---")
        benchmark_scripts.extend(long_tests)

    absolute_script_paths = [tests_dir / script for script in benchmark_scripts]
    
    print(f"Found {len(absolute_script_paths)} benchmark scripts to execute.")
    
    print("\n--- 🛠️  Performing a clean build... ---")
    build_process = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        capture_output=True, text=True, cwd=Path(__file__).parent.resolve()
    )
    if build_process.returncode != 0:
        print("--- ❌ CRITICAL BUILD FAILURE ---")
        print(build_process.stdout)
        print(build_process.stderr)
        sys.exit(1)
    print("--- ✅ Build successful. ---")
    
    failures = []
    for script_path in absolute_script_paths:
        if not run_test(script_path):
            failures.append(script_path.name)
            
    print("\n" + "="*80)
    if not failures:
        print("✅✅✅ MONUMENTAL SUCCESS: The entire Helios.Embed test suite passed.")
    else:
        print(f"❌ FAILURE: The following {len(failures)} benchmarks failed: {', '.join(failures)}")
    print("="*80)

# --- END OF FILE run_tests.py (MINIMAL UPDATE) ---
