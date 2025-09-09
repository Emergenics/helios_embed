# --- START OF FILE HELIOS_EMBED/run_tests.py (Final, Corrected Version v3) ---
import subprocess
import sys
import os
from pathlib import Path

# --- CRITICAL FIX: Add the 'src' directory to the Python path ---
project_root = Path(__file__).parent.resolve()
src_path = project_root / 'src'
# Prepend our local source directory to the system path
sys.path.insert(0, str(src_path))
# ----------------------------------------------------------------

def run_test(script_path):
    print("="*80)
    print(f"🚀 EXECUTING TEST: {script_path.name}")
    print("="*80)
    
    # We now run the script in the same process, which inherits our path fix.
    # This is simpler and more robust than managing subprocess environments.
    try:
        # Dynamically import and run the script's main function
        spec = __import__("importlib.util").util.spec_from_file_location(
            script_path.stem, str(script_path)
        )
        module = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # We assume each benchmark script has a 'run()' function
        if hasattr(module, 'run'):
            module.run()
            print(f"✅ TEST PASSED: {script_path.name}")
            return True
        else:
            print(f"--- ⚠️  SKIPPED: No 'run()' function found in {script_path.name} ---")
            return True # Treat as a pass for now, as it's a structural issue

    except Exception as e:
        print(f"\n--- ❌ TEST FAILED: {script_path.name} ---")
        import traceback
        print(traceback.format_exc())
        print("--------------------")
        return False

if __name__ == "__main__":
    tests_dir = project_root / "tests"
    scripts_to_run = sorted(list(tests_dir.glob("benchmark_*.py")))
    
    print(f"Found {len(scripts_to_run)} tests to execute.")
    
    failures = []
    for script_path in scripts_to_run:
        # We need to ensure each benchmark has a 'run' function for this to work
        # Let's also fix the benchmarks themselves.
        pass # The logic below will be implemented after fixing the files.

    print("\nExecuting tests with a simplified, robust runner...")
    # A simpler runner that just calls the python executable, now that the path issue is understood.
    # We must fix the benchmark files themselves to be runnable modules.
    
    # For now, let's fix the subprocess runner to be correct.
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(src_path)}{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    failures = []
    for script_path in scripts_to_run:
        print("="*80)
        print(f"🚀 EXECUTING TEST: {script_path.name}")
        print("="*80)
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, env=env
        )
        print(result.stdout)
        if result.returncode != 0 or "❌" in result.stdout or "Error:" in result.stdout:
            print(f"\n--- ❌ TEST FAILED: {script_path.name} ---")
            if result.stderr:
                print("--- Stderr ---")
                print(result.stderr)
            print("--------------------")
            failures.append(script_path.name)

    print("\n" + "="*80)
    if not failures:
        print("✅✅✅ MONUMENTAL SUCCESS: All tests passed in the new repository structure!")
    else:
        print(f"❌ FAILURE: The following {len(failures)} tests failed: {', '.join(failures)}")
    print("="*80)
# --- END OF FILE HELIOS_EMBED/run_tests.py (Final, Corrected Version v3) ---