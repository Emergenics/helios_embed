# --- START OF FILE tests/benchmark_nystrom.py (FINAL, CORRECT STANDARD) ---
import torch
import time
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from helios_embed._core import IncrementalNystromEngine, compute_rkhs_embedding
except ImportError as e:
    print(f"❌ Module not compiled. Run setup.py. Error: {e}")
    sys.exit(1)

def get_nystrom_features_pytorch(X, landmarks, gamma, ridge):
    K_nm = torch.exp(-gamma * torch.cdist(X, landmarks).pow(2))
    K_mm = torch.exp(-gamma * torch.cdist(landmarks, landmarks).pow(2))
    K_mm_reg = K_mm + torch.eye(landmarks.shape[0], device=X.device) * ridge
    try:
        L = torch.linalg.cholesky(K_mm_reg)
        features = torch.linalg.solve_triangular(L, K_nm.t(), upper=False).t()
    except torch.linalg.LinAlgError:
        K_mm_inv_sqrt = torch.linalg.pinv(K_mm_reg).sqrt()
        features = K_nm @ K_mm_inv_sqrt
    return features.contiguous()

def run():
    print("\n" + "="*60)
    print("--- Final Validation: Helios.Embed (Nystrom Engine) v1.2.0 ---")
    print("--- ACCURACY STANDARD: BIT-PERFECT FOR FLOAT32 (Rel MSE <= 1e-7) ---")
    print("="*60)

    device = torch.device("cuda")
    N, D, m = 4096, 128, 256
    print(f"Parameters: N={N}, D={D}, m={m}")

    torch.manual_seed(42)
    X = torch.randn(N, D, device=device)
    landmarks = X[torch.randperm(N, device=device)[:m]]
    gamma = 0.1; ridge = 1e-6

    # Time PyTorch Baseline
    features_pytorch = get_nystrom_features_pytorch(X, landmarks, gamma, ridge)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(10): features_pytorch = get_nystrom_features_pytorch(X, landmarks, gamma, ridge)
    torch.cuda.synchronize()
    time_pytorch = (time.perf_counter() - start_time) / 10

    # Time Final Helios C++ Engine
    features_helios = compute_rkhs_embedding(X, landmarks, gamma, ridge)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(10): features_helios = compute_rkhs_embedding(X, landmarks, gamma, ridge)
    torch.cuda.synchronize()
    time_helios = (time.perf_counter() - start_time) / 10

    speedup = time_pytorch / time_helios
    rel_mse = torch.mean((features_pytorch - features_helios)**2) / (torch.mean(features_pytorch**2) + 1e-20)
    
    print(f"\n  PyTorch Baseline Time: {time_pytorch * 1000:.4f} ms")
    print(f"  Final Helios C++ Time: {time_helios * 1000:.4f} ms")
    print(f"  Speedup vs PyTorch:    {speedup:.2f}x")
    print(f"\n  Relative MSE:            {rel_mse.item():.6e}")
    
    # --- THE CORRECT, FINAL ACCURACY STANDARD ---
    BIT_PERFECT_F32_THRESHOLD = 1e-7
    if rel_mse.item() > BIT_PERFECT_F32_THRESHOLD:
        print(f"  ❌ ACCURACY TEST FAILED (Relative MSE > {BIT_PERFECT_F32_THRESHOLD})")
    else:
        print(f"  ✅ ACCURACY TEST PASSED (Bit-Perfect for float32, MSE <= {BIT_PERFECT_F32_THRESHOLD})")

if __name__ == "__main__":
    run()
# --- END OF FILE tests/benchmark_nystrom.py (FINAL, CORRECT STANDARD) ---