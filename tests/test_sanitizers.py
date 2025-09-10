# --- START OF FILE tests/test_sanitizers.py (FINAL, ROBUST v1.1) ---
import torch
import sys
from pathlib import Path

# --- THIS IS THE CRITICAL FIX ---
# Make the script self-aware of its location and the project structure.
# This ensures it can find other modules regardless of how it's executed.
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent # a.k.a. HELIOS_EMBED/
src_path = project_root / 'src'
tests_path = project_root / 'tests'

# Add both the src (for the C++ module) and the tests dir to the path
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(tests_path) not in sys.path:
    sys.path.insert(0, str(tests_path))
# --- END OF CRITICAL FIX ---


LINE = "="*80

def run_edge_case_tests_inline(core_module):
    """
    This is the core logic from benchmark_edge_cases.py, now inlined
    to make this script self-contained and robust.
    It takes the imported core module as an argument.
    """
    IncrementalNystromEngine = core_module.IncrementalNystromEngine
    compute_rkhs_embedding = core_module.compute_rkhs_embedding
    
    device = torch.device("cuda")
    D, m = 128, 32
    gamma = 0.1; ridge = 1e-6
    
    # Test 1: Empty Input
    X_empty = torch.randn(0, D, device=device)
    landmarks = torch.randn(m, D, device=device)
    phi_empty = compute_rkhs_embedding(X_empty, landmarks, gamma, ridge)
    
    # Test 2: Duplicate Landmarks
    X = torch.randn(100, D, device=device)
    duplicate_landmarks = torch.randn(1, D, device=device).repeat(m, 1)
    engine = IncrementalNystromEngine(duplicate_landmarks, gamma, ridge)
    phi = engine.build(X)
    
    # We don't need to assert or print here; the goal is simply to execute
    # these code paths. If a sanitizer error exists, it will crash.
    print("  - Executed empty input and duplicate landmark paths.")

def run():
    print(f"\n{LINE}")
    print("--- 🚀 Executing Sanitizer Smoke Test Suite ---")
    print(LINE)
    
    try:
        # Import the module now that the path is guaranteed to be correct
        from helios_embed import _core as helios_core_module
        print("✅ Helios.Embed module imported successfully.")
    except ImportError as e:
        print(f"❌ Could not import from compiled module. Error: {e}")
        sys.exit(1)

    # Execute a minimal set of operations to trigger CUDA kernels
    print("\n--- 🧪 Triggering CUDA Kernels for Sanitizer ---")
    run_edge_case_tests_inline(helios_core_module)
    
    print(f"\n{LINE}")
    print("--- ✅ SANITIZER SMOKE TEST COMPLETED ---")
    print("--- (If no errors were printed above by compute-sanitizer, the test is PASSED) ---")
    print(LINE)

if __name__ == "__main__":
    run()

# --- END OF FILE tests/test_sanitizers.py (FINAL, ROBUST v1.1) ---