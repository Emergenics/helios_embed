# --- START OF FILE benchmarks/benchmark_edge_cases.py (FINAL, NO-PYTEST) ---
import torch
import sys
import numpy as np

LINE = "="*80
print(f"\n{LINE}")
print("--- Helios.Embed: Definitive Edge-Case & Robustness Suite ---")
print(LINE)

try:
    from helios_embed._core import IncrementalNystromEngine, compute_rkhs_embedding
except ImportError as e:
    print(f"❌ Helios.Embed module not compiled. Run 'python setup.py build_ext --inplace'. Error: {e}"); sys.exit()

def run_test(test_name, test_function):
    print(f"\n--- 🧪 TESTING: {test_name} ---")
    try:
        result = test_function()
        if result:
            print(f"  ✅ TEST PASSED")
            return True
        else:
            print(f"  ❌ TEST FAILED: The expected error was not triggered.")
            return False
    except Exception as e:
        print(f"  ❌ TEST FAILED with unexpected exception: {type(e).__name__}: {e}"); return False

def test_empty_input():
    device = torch.device("cuda"); D, m = 128, 32; landmarks = torch.randn(m, D, device=device)
    gamma = 0.1; ridge = 1e-6
    X_empty = torch.randn(0, D, device=device)
    phi_empty = compute_rkhs_embedding(X_empty, landmarks, gamma, ridge)
    assert phi_empty.shape == (0, m)
    engine = IncrementalNystromEngine(landmarks, gamma, ridge); X_initial = torch.randn(10, D, device=device)
    phi_initial = engine.build(X_initial); phi_updated = engine.update(X_empty, phi_initial)
    assert torch.allclose(phi_updated, phi_initial)
    return True

def test_zero_landmarks():
    device = torch.device("cuda"); D = 128; X = torch.randn(100, D, device=device); gamma = 0.1; ridge = 1e-6
    try:
        IncrementalNystromEngine(torch.randn(0, D, device=device), gamma, ridge)
    except RuntimeError as e:
        if "must have at least one landmark" in str(e).lower():
            print("  - Constructor correctly threw error for zero landmarks.")
        else:
            print(f"  - Constructor threw wrong error: {e}"); return False
    else: return False
    try:
        zero_landmarks = torch.randn(0, D, device=device).contiguous()
        compute_rkhs_embedding(X, torch.randn(0, D, device=device), gamma, ridge)
    except RuntimeError as e:
        if "must have at least one landmark" in str(e).lower():
            print("  - Stateless function correctly threw error for zero landmarks.")
        else:
            print(f"  - Stateless function threw wrong error: {e}"); return False
    else: return False
    return True

def test_duplicate_landmarks():
    device = torch.device("cuda"); D, m = 128, 32; X = torch.randn(100, D, device=device); gamma = 0.1; ridge = 1e-6
    duplicate_landmarks = torch.randn(1, D, device=device).repeat(m, 1)
    engine = IncrementalNystromEngine(duplicate_landmarks, gamma, ridge); phi = engine.build(X)
    assert phi.shape == (100, m) and torch.isfinite(phi).all()
    return True

def test_nan_inf_input():
    device = torch.device("cuda"); D, m = 128, 32; landmarks = torch.randn(m, D, device=device); gamma = 0.1; ridge = 1e-6
    X_nan = torch.randn(10, D, device=device); X_nan[5, 5] = float('nan')
    X_inf = torch.randn(10, D, device=device); X_inf[3, 3] = float('inf')
    try:
        compute_rkhs_embedding(X_nan, landmarks, gamma, ridge)
    except RuntimeError as e:
        if "must not contain nan or inf" in str(e).lower():
            print("  - Correctly caught NaN input.")
        else:
            print(f"  - Threw wrong error for NaN: {e}"); return False
    else: return False
    try:
        compute_rkhs_embedding(X_inf, landmarks, gamma, ridge)
    except RuntimeError as e:
        if "must not contain nan or inf" in str(e).lower():
            print("  - Correctly caught Inf input.")
        else:
            print(f"  - Threw wrong error for Inf: {e}"); return False
    else: return False
    return True

def test_mismatched_dims():
    device = torch.device("cuda"); D1, D2, m = 128, 64, 32; N1, N2 = 100, 50; gamma = 0.1; ridge = 1e-6
    landmarks_d1 = torch.randn(m, D1, device=device); X_d2 = torch.randn(N1, D2, device=device)
    try:
        compute_rkhs_embedding(X_d2, landmarks_d1, gamma, ridge)
    except RuntimeError as e:
        if "feature dimensions of input x and landmarks lm must match" in str(e).lower():
            print("  - Correctly caught D dimension mismatch.")
        else:
            print(f"  - Threw wrong error for D mismatch: {e}"); return False
    else: return False
    engine = IncrementalNystromEngine(landmarks_d1, gamma, ridge)
    phi_old_wrong_m = torch.randn(N1, m-1, device=device)
    X_new = torch.randn(N2, D1, device=device)
    try:
        engine.update(X_new, phi_old_wrong_m)
    except RuntimeError as e:
        if "feature dimension m of phi_old" in str(e).lower():
            print("  - Correctly caught m dimension mismatch in update.")
        else:
            print(f"  - Threw wrong error for m mismatch: {e}"); return False
    else: return False
    return True

def run():
    if not torch.cuda.is_available(): print("CUDA not available."); return
    all_tests = {
        "Empty Input": test_empty_input, "Zero Landmarks": test_zero_landmarks,
        "Duplicate Landmarks": test_duplicate_landmarks, "NaN and Inf Input": test_nan_inf_input,
        "Mismatched Dimensions": test_mismatched_dims,
    }
    passed_count = sum(run_test(name, func) for name, func in all_tests.items())
    print(f"\n{LINE}\n--- 📊 SUITE SUMMARY: {passed_count} / {len(all_tests)} TESTS PASSED ---\n{LINE}")
    if passed_count == len(all_tests): print("\n  ✅ MONUMENTAL SUCCESS: Helios.Embed is robust, stable, and handles all tested edge cases correctly.")
    else: print("\n  ❌ ATTENTION: One or more edge cases failed.")

if __name__ == "__main__":
    run()
# --- END OF FILE benchmarks/benchmark_edge_cases.py (NO-PYTEST) ---