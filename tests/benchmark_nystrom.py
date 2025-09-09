# --- START OF FILE benchmarks/benchmark_nystrom.py (FINAL, STANDARDIZED) ---
import torch
import time
import sys
import pandas as pd

try:
    from helios_embed._core import IncrementalNystromEngine, compute_rkhs_embedding
except ImportError as e:
    print(f"❌ Helios.Embed module not compiled. Run 'python setup.py build_ext --inplace'. Error: {e}")
    sys.exit()

def get_nystrom_features_pytorch(X, landmarks, gamma, ridge):
    K_nm = torch.exp(-gamma * torch.cdist(X, landmarks).pow(2))
    K_mm = torch.exp(-gamma * torch.cdist(landmarks, landmarks).pow(2))
    K_mm_reg = K_mm + torch.eye(landmarks.shape[0], device=X.device, dtype=X.dtype) * ridge
    try:
        L = torch.linalg.cholesky(K_mm_reg)
        features = torch.linalg.solve_triangular(L, K_nm.t(), upper=False).t()
    except torch.linalg.LinAlgError:
        K_mm_inv_sqrt = torch.linalg.pinv(K_mm_reg).sqrt()
        features = K_nm @ K_mm_inv_sqrt
    return features.contiguous()

def run_nystrom_benchmark(N, D, m, device):
    print(f"\n--- Final Validation: Helios.Embed (Nystrom Engine) v1.0.0 ---")
    print(f"Parameters: N={N}, D={D}, m={m}")

    torch.manual_seed(42)
    X = torch.randn(N, D, device=device, dtype=torch.float32)
    landmarks = X[torch.randperm(N, device=device)[:m]]
    gamma = 0.1; ridge = 1e-6

    features_pytorch = get_nystrom_features_pytorch(X, landmarks, gamma, ridge) # Warmup
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(10): features_pytorch = get_nystrom_features_pytorch(X, landmarks, gamma, ridge)
    torch.cuda.synchronize()
    time_pytorch = (time.perf_counter() - start_time) / 10

    features_helios = compute_rkhs_embedding(X, landmarks, gamma, ridge) # Warmup
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(10): features_helios = compute_rkhs_embedding(X, landmarks, gamma, ridge)
    torch.cuda.synchronize()
    time_helios = (time.perf_counter() - start_time) / 10

    rel_mse = torch.mean((features_pytorch - features_helios)**2) / (torch.mean(features_pytorch**2) + 1e-20)

    print(f"\n  PyTorch Baseline Time: {time_pytorch * 1000:.4f} ms")
    print(f"  Final Helios C++ Time: {time_helios * 1000:.4f} ms")
    print(f"  Speedup vs PyTorch:    {time_pytorch / time_helios:.2f}x")
    print(f"\n  Relative MSE:            {rel_mse.item():.6e}")
    
    if rel_mse.item() > 1e-14:
        print("  ❌ ACCURACY TEST FAILED (Not bit-perfect)")
    else:
        print("  ✅ ACCURACY TEST PASSED (Bit-perfect)")

# --- THIS IS THE CRITICAL FIX ---
# We wrap the main execution block in a `run()` function for our master runner.
def run():
    if not torch.cuda.is_available(): 
        print("CUDA not available, exiting benchmark.")
        sys.exit()
    device = torch.device("cuda")
    run_nystrom_benchmark(4096, 128, 256, device)

if __name__ == "__main__":
    run()
# --- END OF FILE benchmarks/benchmark_nystrom.py (FINAL, STANDARDIZED) ---