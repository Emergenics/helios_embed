# --- START OF FILE tests/test_correctness.py ---
import torch
import sys
from pathlib import Path

# Add src to path to allow direct import for testing
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# The reference implementation is our "Oracle"
from tests.helios_ref import nystrom_embed_ref
# The compiled C++/CUDA module is our "Device Under Test"
from helios_embed._core import compute_rkhs_embedding

# --- Configuration ---
LINE = "="*80
TOL = {'float32': 1e-7, 'float64': 1e-15} # OUR NEW, STRICT ACCURACY STANDARD
SEED = 1234
torch.manual_seed(SEED)

def run_single_test(name, func):
    """A simple test runner utility."""
    print(f"\n--- 🧪 TESTING: {name} ---")
    try:
        func()
        print(f"  ✅ TEST PASSED")
        return True
    except Exception as e:
        print(f"  ❌ TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Test Functions ---

def test_basic_correctness():
    """Unit Test: Verifies basic shape, dtype, and numerical correctness against the CPU reference."""
    device = torch.device("cuda")
    # Test a standard configuration
    N, L, D = 64, 16, 8
    gamma, ridge = 0.7, 1e-6
    
    X = torch.randn(N, D, device=device, dtype=torch.float32)
    landmarks = torch.randn(L, D, device=device, dtype=torch.float32)

    # Device Under Test (CUDA)
    out_cuda = compute_rkhs_embedding(X, landmarks, gamma, ridge)
    
    # Oracle (CPU Reference)
    ref_cpu = nystrom_embed_ref(X, landmarks, gamma, ridge)

    # Verification
    assert out_cuda.shape == ref_cpu.shape, f"Shape mismatch: CUDA={out_cuda.shape}, Ref={ref_cpu.shape}"
    assert out_cuda.dtype == torch.float32
    
    rel_mse = torch.mean((out_cuda.cpu().double() - ref_cpu)**2) / (torch.mean(ref_cpu**2) + 1e-20)
    assert rel_mse.item() <= TOL['float32'], f"Numerical mismatch: Rel MSE={rel_mse.item()} > {TOL['float32']}"
    print("  - Shape, dtype, and numerical accuracy are correct.")

def test_batching_equivalence():
    """Property Test: Verifies that processing a large batch is equivalent to processing smaller chunks."""
    device = torch.device("cuda")
    N, L, D = 128, 32, 16
    gamma, ridge = 1.2, 1e-5
    
    X = torch.randn(N, D, device=device)
    Lm = torch.randn(L, D, device=device)

    # Process as one large batch
    out_big_batch = compute_rkhs_embedding(X, Lm, gamma, ridge)

    # Process as two smaller chunks and concatenate
    X1, X2 = X[:N//2], X[N//2:]
    out1 = compute_rkhs_embedding(X1, Lm, gamma, ridge)
    out2 = compute_rkhs_embedding(X2, Lm, gamma, ridge)
    out_small_batches = torch.cat([out1, out2], dim=0)

    assert torch.allclose(out_big_batch, out_small_batches, rtol=0, atol=TOL['float32'])
    print("  - Concatenated result from small batches matches the full batch result.")

def test_near_singular_stability():
    """Property Test: Verifies that the ridge parameter correctly stabilizes near-singular matrices."""
    device = torch.device("cuda")
    D, L = 4, 8
    # Create landmarks with nearly duplicate rows to make K_LL near-singular
    base = torch.randn(L // 2, D, device=device)
    landmarks = torch.cat([base, base + 1e-7 * torch.randn_like(base)], dim=0)
    X = torch.randn(10, D, device=device)
    gamma = 1.0

    # With a small ridge, the computation should succeed and be stable.
    out_cuda = compute_rkhs_embedding(X, landmarks, gamma, 1e-4)
    ref_cpu = nystrom_embed_ref(X, landmarks, gamma, 1e-4)
    
    rel_mse = torch.mean((out_cuda.cpu().double() - ref_cpu)**2) / (torch.mean(ref_cpu**2) + 1e-20)
    assert rel_mse.item() <= TOL['float32'], f"Numerical mismatch on near-singular matrix: Rel MSE={rel_mse.item()}"
    print("  - Computation is stable and correct for near-singular landmark matrix when ridge > 0.")


def run():
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run correctness tests.")
        return

    tests_to_run = {
        "Basic Correctness": test_basic_correctness,
        "Batching Equivalence": test_batching_equivalence,
        "Near-Singular Stability": test_near_singular_stability,
    }

    passed_count = sum(run_single_test(name, func) for name, func in tests_to_run.items())
    
    print(f"\n{LINE}\n--- 📊 SUITE SUMMARY: {passed_count} / {len(tests_to_run)} CORRECTNESS TESTS PASSED ---\n{LINE}")
    if passed_count == len(tests_to_run):
        print("\n  ✅ MONUMENTAL SUCCESS: Helios.Embed correctness and stability are validated.")
    else:
        print("\n  ❌ ATTENTION: One or more correctness tests failed. Review logs.")
        sys.exit(1)


if __name__ == "__main__":
    run()

# --- END OF FILE tests/test_correctness.py ---