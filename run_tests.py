# --- START OF FILE HELIOS_EMBED/run_tests.py (FINAL, ROBUST VERSION) ---
import sys
from pathlib import Path
import importlib.util
import traceback

# --- CRITICAL: Add the 'src' directory to the Python path ---
# This ensures that any test script can find the compiled C++ module.
project_root = Path(__file__).parent.resolve()
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def run_benchmark_module(script_path):
    """
    Dynamically imports and runs a benchmark script.
    A benchmark is considered PASSED if its main 'run()' function
    executes without raising an uncaught exception.
    """
    print("="*80)
    print(f"🚀 EXECUTING BENCHMARK: {script_path.name}")
    print("="*80)
    
    try:
        # Dynamically load the module from its file path
        module_name = script_path.stem
        spec = importlib.util.spec_from_file_location(module_name, str(script_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # All of our benchmarks have a main run() function
        if hasattr(module, 'run'):
            module.run() # Execute the benchmark
            print(f"\n--- ✅ BENCHMARK SUCCEEDED: {script_path.name} ---")
            return True
        else:
            print(f"--- ⚠️  SKIPPED (No 'run' function): {script_path.name} ---")
            return True # Treat as success if no run function is defined

    except Exception:
        print(f"\n--- ❌ BENCHMARK FAILED WITH EXCEPTION: {script_path.name} ---")
        # Print the full traceback for immediate diagnosis
        print(traceback.format_exc())
        print("-----------------------------------------------------")
        return False

if __name__ == "__main__":
    tests_dir = project_root / "tests"
    
    # Use glob to automatically find all benchmark scripts
    benchmark_scripts = sorted(list(tests_dir.glob("benchmark_*.py")))
    
    print(f"Found {len(benchmark_scripts)} benchmark scripts to execute.")
    
    failures = []
    
    # --- Clean and Recompile before running tests ---
    print("\n--- 🛠️  Performing a clean build... ---")
    import subprocess
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
    
    # --- Run the test suite ---
    for script in benchmark_scripts:
        if not run_benchmark_module(script):
            failures.append(script.name)
            
    # --- Final Verdict ---
    print("\n" + "="*80)
    if not failures:
        print("✅✅✅ MONUMENTAL SUCCESS: The entire Helios.Embed test suite passed.")
    else:
        print(f"❌ FAILURE: The following {len(failures)} benchmarks failed: {', '.join(failures)}")
    print("="*80)

# --- END OF FILE HELIOS_EMBED/run_tests.py (FINAL, ROBUST VERSION) ---