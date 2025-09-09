# --- START OF FILE benchmarks/benchmark_production_readiness.py ---
import torch
import time
import sys
import pandas as pd
from itertools import product
import gc

LINE = "="*80

try:
    from helios_embed._core import IncrementalNystromEngine, compute_rkhs_embedding
except ImportError as e:
    print(f"❌ Helios.Embed module not compiled. Run 'python setup.py build_ext --inplace'. Error: {e}")
    sys.exit()

# --- Profiler Function (for the 'build' method) ---
def profile_build_operation(N, D, m, device):
    print(f"\n{LINE}\n--- 🚀 PROFILING: Stateless 'build' Operation (N={N}, D={D}, m={m}) ---\n{LINE}")
    torch.manual_seed(42)
    gamma = 0.1; ridge = 1e-6
    try:
        X = torch.randn(N, D, device=device)
        landmarks = X[torch.randperm(N, device=device)[:m]]
    except torch.cuda.OutOfMemoryError:
        print("  - ⚠️ OOM during data generation. Skipping profile."); return
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(5): # A few iterations for profiling
            compute_rkhs_embedding(X, landmarks, gamma, ridge)
            
    print("\n--- 📊 PyTorch Profiler Summary for 'build()' (Top 10 CUDA Kernels) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# --- Streaming Benchmark Function (for the 'update' method) ---
def benchmark_update_operation(N_initial, N_update, num_updates, D, m, device):
    print(f"\n{LINE}\n--- 🚀 BENCHMARKING: Stateful 'update' Operation ---\n(N_initial={N_initial}, N_update={N_update}, num_updates={num_updates})\n{LINE}")
    # ... (This is the same successful logic from our validated benchmark_streaming.py) ...
    torch.manual_seed(42)
    try:
        X_initial = torch.randn(N_initial, D, device=device)
        update_batches = [torch.randn(N_update, D, device=device) for _ in range(num_updates)]
        landmarks = X_initial[torch.randperm(N_initial, device=device)[:m]]
    except torch.cuda.OutOfMemoryError:
        print("  - ⚠️ OOM during data generation. Skipping."); return None
    gamma = 0.1; ridge = 1e-6; time_baseline = 0; time_incremental = 0;
    try:
        current_X_base = X_initial.clone()
        torch.cuda.synchronize(); start_time_base = time.perf_counter()
        for i in range(num_updates):
            current_X_base = torch.cat([current_X_base, update_batches[i]], 0)
            _ = compute_rkhs_embedding(current_X_base, landmarks, gamma, ridge)
        torch.cuda.synchronize(); time_baseline = time.perf_counter() - start_time_base
    except torch.cuda.OutOfMemoryError: time_baseline = float('inf')
    try:
        engine = IncrementalNystromEngine(landmarks, gamma, ridge)
        torch.cuda.synchronize(); start_time_itmd = time.perf_counter()
        Phi_itmd = engine.build(X_initial)
        for i in range(num_updates): Phi_itmd = engine.update(update_batches[i], Phi_itmd)
        torch.cuda.synchronize(); time_incremental = time.perf_counter() - start_time_itmd
    except torch.cuda.OutOfMemoryError: time_incremental = float('inf')
    speedup = time_baseline / time_incremental if time_incremental > 0 else float('inf')
    print(f"  - Total Baseline Time: {time_baseline * 1000:.2f} ms")
    print(f"  - Total ITMD Time:     {time_incremental * 1000:.2f} ms")
    print(f"  - Streaming Speedup:   {speedup:.2f}x  {'✅' if speedup > 1.0 else '❌'}")
    return speedup

# --- Main Runner ---
def run():
    if not torch.cuda.is_available(): print("CUDA not available."); return
    device = torch.device("cuda")
    
    # Part 1: Profile the 'build' operation
    profile_build_operation(N=8192, D=768, m=256, device=device)
    
    # Part 2: Benchmark the 'update' operation (a single, representative case)
    benchmark_update_operation(N_initial=8192, N_update=64, num_updates=200, D=128, m=256, device=device)

if __name__ == "__main__":
    run()
# --- END OF FILE benchmarks/benchmark_production_readiness.py ---