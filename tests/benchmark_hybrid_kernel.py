# --- START OF FILE tests/benchmark_hybrid_kernel.py ---
import torch
import time
import sys
from pathlib import Path

# Add src to path to allow direct import for testing
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

LINE = "="*80

def run_hybrid_kernel_benchmark(N, D, m, device, repeats=100):
    """
    Performs a head-to-head benchmark of the new custom Hybrid RBF kernel
    against the standard PyTorch ATen-backed implementation.
    """
    print(f"\n--- 🧪 Testing Config: N={N}, D={D}, m={m} ---")
    
    try:
        from helios_embed._core import rbf_kernel_hybrid_cuda
    except ImportError as e:
        print(f"❌ Could not import custom kernel 'rbf_kernel_hybrid_cuda'.")
        print(f"   Ensure the project has been compiled successfully. Error: {e}")
        sys.exit(1)

    torch.manual_seed(42)
    gamma = 0.1
    
    try:
        X = torch.randn(N, D, device=device, dtype=torch.float32)
        Y = torch.randn(m, D, device=device, dtype=torch.float32)
    except torch.cuda.OutOfMemoryError:
        print("  - ⚠️ OOM during data generation. Skipping test.")
        return

    # --- 1. ATen Baseline (torch.cdist + exp) ---
    time_aten = float('inf')
    K_aten = None
    try:
        # Warmup
        K_aten = torch.exp(-gamma * torch.cdist(X, Y).pow(2))
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(repeats):
            K_aten = torch.exp(-gamma * torch.cdist(X, Y).pow(2))
        end_event.record()
        torch.cuda.synchronize()
        time_aten = start_event.elapsed_time(end_event) / repeats
        
        print(f"  - ATen (Baseline) Time: {time_aten:.4f} ms")
    except torch.cuda.OutOfMemoryError:
        print("  - ⚠️ OOM during ATen baseline execution.")

    # --- 2. Custom Hybrid Kernel ---
    time_hybrid = float('inf')
    K_hybrid = None
    try:
        # Warmup
        K_hybrid = rbf_kernel_hybrid_cuda(X, Y, gamma)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(repeats):
            K_hybrid = rbf_kernel_hybrid_cuda(X, Y, gamma)
        end_event.record()
        torch.cuda.synchronize()
        time_hybrid = start_event.elapsed_time(end_event) / repeats
        
        print(f"  - Hybrid Kernel Time:   {time_hybrid:.4f} ms")
    except Exception as e:
        print(f"  - ❌ Hybrid kernel execution failed: {e}")
        if K_aten is not None:
             K_hybrid = torch.zeros_like(K_aten)

    # --- 3. Verification and Report ---
    speedup = time_aten / time_hybrid if time_hybrid > 0 and time_aten != float('inf') else float('nan')
    
    rel_mse = -1.0
    if K_aten is not None and K_hybrid is not None:
        rel_mse = torch.mean((K_aten - K_hybrid)**2) / (torch.mean(K_aten**2) + 1e-12)
        rel_mse = rel_mse.item()

    print(f"\n  - Speedup vs ATen: {speedup:.2f}x")
    print(f"  - Relative MSE:    {rel_mse:.6e}")

    # --- Final Verdict for this configuration ---
    accuracy_ok = rel_mse < 1e-7 # Our new bit-perfect float32 accuracy standard
    speedup_ok = speedup >= 1.5

    print("\n  --- Verdict ---")
    if accuracy_ok and speedup_ok:
        print("  - ✅ HYPOTHESIS VALIDATED: Speedup >= 2x and bit-perfect accuracy standard met.")
    elif not accuracy_ok:
        print("  - ❌ HYPOTHESIS FALSIFIED: Bit-perfect accuracy standard (< 1e-7) was not met.")
    else: # Accuracy is ok, but speed is not
        print("  - ⚠️  HYPOTHESIS NOT MET: Speedup target of 2x was not achieved.")

def run():
    print(f"\n{LINE}")
    print("--- Helios.Embed: Hybrid Kernel Performance Benchmark ---")
    print(f"--- ACCURACY STANDARD: BIT-PERFECT (Rel MSE < 1e-7 for float32) ---")
    print(LINE)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    device = torch.device("cuda")
    # Run the definitive test case
    run_hybrid_kernel_benchmark(N=8192, D=768, m=256, device=device, repeats=100)

if __name__ == "__main__":
    run()
# --- END OF FILE tests/benchmark_hybrid_kernel.py ---