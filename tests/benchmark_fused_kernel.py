# --- START OF FILE tests/benchmark_fused_kernel.py ---
import torch
import time
import sys
from pathlib import Path
import pandas as pd

# Add src to path to allow direct import for testing
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

LINE = "="*80

def run_fused_kernel_benchmark(N, D, m, device, repeats=100):
    print(f"\n--- 🧪 Testing Config: N={N}, D={D}, m={m} ---")
    
    try:
        from helios_embed._core import rbf_kernel_fused_cuda
    except ImportError as e:
        print(f"❌ Could not import from compiled module. Error: {e}")
        return None

    torch.manual_seed(42)
    gamma = 0.1
    
    try:
        X = torch.randn(N, D, device=device, dtype=torch.float32)
        Y = torch.randn(m, D, device=device, dtype=torch.float32)
    except torch.cuda.OutOfMemoryError:
        print("  - ⚠️ OOM during data generation. Skipping test.")
        return None
        
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
        print("  - ⚠️ OOM during ATen baseline. Marking as failed.")

    # --- 2. Custom Fused Kernel ---
    time_fused = float('inf')
    K_fused = None
    try:
        # Warmup
        K_fused = rbf_kernel_fused_cuda(X, Y, gamma)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(repeats):
            K_fused = rbf_kernel_fused_cuda(X, Y, gamma)
        end_event.record()
        torch.cuda.synchronize()
        time_fused = start_event.elapsed_time(end_event) / repeats
        
        print(f"  - Fused Kernel Time:    {time_fused:.4f} ms")
    except Exception as e:
        print(f"  - ❌ Fused kernel failed: {e}")
        K_fused = torch.zeros_like(X[:,:m]) if K_aten is None else torch.zeros_like(K_aten)

    # --- 3. Accuracy and Speedup ---
    speedup = time_aten / time_fused if time_fused > 0 and time_aten != float('inf') else float('nan')
    
    rel_mse = -1.0
    if K_aten is not None and K_fused is not None:
        rel_mse = torch.mean((K_aten - K_fused)**2) / (torch.mean(K_aten**2) + 1e-12)
        rel_mse = rel_mse.item()

    print(f"\n  - Speedup vs ATen: {speedup:.2f}x")
    print(f"  - Relative MSE:    {rel_mse:.6e}")

    # --- Final Verdict for this configuration ---
    accuracy_ok = rel_mse < 1e-5 # Looser tolerance for fused kernel FP32 math
    speedup_ok = speedup >= 5.0

    if accuracy_ok and speedup_ok:
        print("  - ✅ HYPOTHESIS VALIDATED: Speedup >= 5x and accuracy standard met.")
    elif not accuracy_ok:
        print("  - ❌ HYPOTHESIS FALSIFIED: Accuracy standard not met.")
    else: # Accuracy is ok, but speed is not
        print("  - ⚠️  HYPOTHESIS NOT MET: Speedup target of 5x not achieved.")
    
    return { "speedup": speedup, "rel_mse": rel_mse, "passed": (accuracy_ok and speedup_ok) }

def run():
    print(f"\n{LINE}")
    print("--- Helios.Embed: Fused Kernel Performance Benchmark (Phase 2A) ---")
    print(LINE)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    device = torch.device("cuda")
    # Run the definitive test case
    run_fused_kernel_benchmark(N=8192, D=768, m=256, device=device, repeats=100)

if __name__ == "__main__":
    run()
# --- END OF FILE tests/benchmark_fused_kernel.py ---