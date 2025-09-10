# --- START OF FILE tests/benchmark_hardening.py ---
import torch
import sys
from pathlib import Path
import numpy as np

# Add src to path to allow direct import for testing
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

LINE = "="*80
PASS = "✅ TEST PASSED"
FAIL = "❌ TEST FAILED"

def run():
    print(f"\n{LINE}")
    print("--- Helios.Embed: Definitive Hardening & Robustness Suite ---")
    print(LINE)

    try:
        from helios_embed._core import IncrementalNystromEngine, compute_rkhs_embedding
    except ImportError as e:
        print(f"❌ Could not import from compiled module. Error: {e}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping hardening tests.")
        return

    device = torch.device("cuda")
    D = 128
    m = 64
    N = 1024
    
    test_results = {}

    print("\n--- 🧪 TESTING: Non-Contiguous Memory Input ---")
    try:
        X_non_contig = torch.randn(D, N, device=device).T
        assert not X_non_contig.is_contiguous()
        landmarks = torch.randn(m, D, device=device)
        # Our C++ code now forces .contiguous(), so this should not crash.
        compute_rkhs_embedding(X_non_contig, landmarks, 0.1, 1e-6)
        print("  - Handled non-contiguous input correctly without error.")
        test_results["Non-Contiguous"] = True
    except Exception as e:
        print(f"  - FAILED with unexpected exception: {e}")
        test_results["Non-Contiguous"] = False
    print(PASS if test_results["Non-Contiguous"] else FAIL)
    
    print("\n--- 🧪 TESTING: NaN Input ---")
    try:
        X_nan = torch.randn(N, D, device=device)
        X_nan[0, 0] = float('nan')
        landmarks = torch.randn(m, D, device=device)
        compute_rkhs_embedding(X_nan, landmarks, 0.1, 1e-6)
        print("  - FAILED: Did not throw an error for NaN input.")
        test_results["NaN Input"] = False
    except RuntimeError as e:
        if "must not contain nan or inf" in str(e).lower():
            print("  - Correctly threw error for NaN input.")
            test_results["NaN Input"] = True
        else:
            print(f"  - FAILED with wrong error type: {e}")
            test_results["NaN Input"] = False
    print(PASS if test_results["NaN Input"] else FAIL)
    
    print("\n--- 🧪 TESTING: Inf Input ---")
    try:
        X_inf = torch.randn(N, D, device=device)
        X_inf[0, 0] = float('inf')
        landmarks = torch.randn(m, D, device=device)
        compute_rkhs_embedding(X_inf, landmarks, 0.1, 1e-6)
        print("  - FAILED: Did not throw an error for Inf input.")
        test_results["Inf Input"] = False
    except RuntimeError as e:
        if "must not contain nan or inf" in str(e).lower():
            print("  - Correctly threw error for Inf input.")
            test_results["Inf Input"] = True
        else:
            print(f"  - FAILED with wrong error type: {e}")
            test_results["Inf Input"] = False
    print(PASS if test_results["Inf Input"] else FAIL)

    print("\n--- 🧪 TESTING: Mismatched D Dimension ---")
    try:
        X = torch.randn(N, D, device=device)
        landmarks_wrong_d = torch.randn(m, D + 1, device=device)
        compute_rkhs_embedding(X, landmarks_wrong_d, 0.1, 1e-6)
        print("  - FAILED: Did not throw an error for D dimension mismatch.")
        test_results["D Mismatch"] = False
    except RuntimeError as e:
        if "feature dimensions of input x and landmarks lm must match" in str(e).lower():
            print("  - Correctly threw error for D dimension mismatch.")
            test_results["D Mismatch"] = True
        else:
            print(f"  - FAILED with wrong error type: {e}")
            test_results["D Mismatch"] = False
    print(PASS if test_results["D Mismatch"] else FAIL)

    # --- Summary ---
    print(f"\n{LINE}")
    num_passed = sum(test_results.values())
    num_total = len(test_results)
    print(f"--- 📊 SUITE SUMMARY: {num_passed} / {num_total} HARDENING TESTS PASSED ---")
    print(LINE)
    
    if num_passed == num_total:
        print("\n  ✅ MONUMENTAL SUCCESS: Helios.Embed is security-hardened and robust.")
    else:
        print("\n  ❌ ATTENTION: One or more hardening tests failed. Review logs.")
        sys.exit(1)

if __name__ == "__main__":
    run()
# --- END OF FILE tests/benchmark_hardening.py ---