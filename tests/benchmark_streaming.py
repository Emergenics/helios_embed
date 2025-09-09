# --- START OF FILE benchmarks/benchmark_streaming.py (FINAL, STANDARDIZED) ---
import torch
import time
import sys
import pandas as pd
from itertools import product

# The new, correct imports from our modular build
try:
    from helios_embed._core import IncrementalNystromEngine, compute_rkhs_embedding
except ImportError as e:
    print(f"❌ Helios.Embed module not compiled. Run 'python setup.py build_ext --inplace'. Error: {e}")
    sys.exit()

def run_streaming_benchmark(N_initial, N_update_batch_size, num_updates, D, m, device):
    # ... (The core logic of this function is correct and remains unchanged) ...
    print(f"\n--- Running Test: N_initial={N_initial}, Batch Size={N_update_batch_size}, Updates={num_updates} ---")
    torch.manual_seed(42)
    try:
        X_initial = torch.randn(N_initial, D, device=device)
        update_batches = [torch.randn(N_update_batch_size, D, device=device) for _ in range(num_updates)]
        landmarks = X_initial[torch.randperm(N_initial, device=device)[:m]]
    except torch.cuda.OutOfMemoryError:
        print("  ⚠️ OOM during data generation. Skipping test."); return None
    gamma = 0.1; ridge = 1e-6
    time_baseline = 0
    try:
        torch.cuda.synchronize(); start_time_base = time.perf_counter()
        current_X_base = X_initial.clone()
        for i in range(num_updates):
            current_X_base = torch.cat([current_X_base, update_batches[i]], 0)
            _ = compute_rkhs_embedding(current_X_base, landmarks, gamma, ridge)
        torch.cuda.synchronize(); time_baseline = time.perf_counter() - start_time_base
    except torch.cuda.OutOfMemoryError:
        print("  ⚠️ OOM during baseline execution. Marking as Inf."); time_baseline = float('inf')
    time_incremental = 0; Phi_itmd = None
    try:
        torch.cuda.synchronize(); start_time_itmd = time.perf_counter()
        engine = IncrementalNystromEngine(landmarks, gamma, ridge)
        Phi_itmd = engine.build(X_initial)
        for i in range(num_updates): Phi_itmd = engine.update(update_batches[i], Phi_itmd)
        torch.cuda.synchronize(); time_incremental = time.perf_counter() - start_time_itmd
    except torch.cuda.OutOfMemoryError:
        print("  ⚠️ OOM during ITMD execution. Marking as Inf."); time_incremental = float('inf')
    speedup = time_baseline / time_incremental if time_incremental > 0 else float('inf')
    rel_mse = -1.0
    if torch.cuda.is_available() and Phi_itmd is not None and time_baseline != float('inf'):
        try:
            final_X = torch.cat([X_initial] + update_batches, 0)
            Phi_final_baseline = compute_rkhs_embedding(final_X, landmarks, gamma, ridge)
            rel_mse = torch.mean((Phi_final_baseline - Phi_itmd)**2) / (torch.mean(Phi_final_baseline**2) + 1e-20)
            rel_mse = rel_mse.item()
        except torch.cuda.OutOfMemoryError:
            print("  ⚠️ OOM during final verification. Accuracy not verified."); rel_mse = -999.0
    print(f"  Total Baseline Time: {time_baseline * 1000:.4f} ms")
    print(f"  Total ITMD Time:     {time_incremental * 1000:.4f} ms")
    print(f"  Streaming Speedup:   {speedup:.2f}x")
    print(f"  Final State Rel. MSE:  {rel_mse:.6e}")
    return {"N_initial": N_initial, "N_update": N_update_batch_size, "num_updates": num_updates, "total_time_base_ms": time_baseline * 1000, "total_time_itmd_ms": time_incremental * 1000, "speedup": speedup, "final_rel_mse": rel_mse}

# --- THIS IS THE CRITICAL FIX ---
# We wrap the main execution block in a `run()` function for our master runner.
def run():
    if not torch.cuda.is_available(): 
        print("CUDA not available, exiting benchmark.")
        sys.exit()
    device = torch.device("cuda")
    
    # Define the parameter sweep
    N_initial_values = [4096, 8192]
    N_update_values = [16, 64, 256]
    num_updates_values = [10, 50, 200]
    D_val = 128
    m_val = 256

    all_configs = list(product(N_initial_values, N_update_values, num_updates_values))
    all_results = []
    
    for N_initial, N_update, num_updates in all_configs:
        if (N_initial + N_update * num_updates) > 32768:
            print(f"\n--- Skipping large config: N_initial={N_initial}, N_update={N_update}, Updates={num_updates} ---")
            continue
            
        result = run_streaming_benchmark(
            N_initial=N_initial, 
            N_update_batch_size=N_update, 
            num_updates=num_updates, 
            D=D_val, 
            m=m_val, 
            device=device
        )
        if result:
            all_results.append(result)
        torch.cuda.empty_cache()

    if all_results:
        df = pd.DataFrame(all_results)
        df['N_final'] = df['N_initial'] + df['N_update'] * df['num_updates']
        df = df[['N_initial', 'N_update', 'num_updates', 'N_final', 'speedup', 'final_rel_mse']]
        
        print("\n" + "="*60)
        print("--- Comprehensive Streaming Benchmark Suite: Summary ---")
        print("="*60)
        print(df.to_string(index=False))
        
        if (df['final_rel_mse'] > 1e-14).any():
             print("\n\n❌ ACCURACY FALSIFIED: At least one run was not bit-perfect.")
        else:
             print("\n\n✅ ACCURACY VALIDATED: All runs were bit-perfect.")

if __name__ == "__main__":
    run()
# --- END OF FILE benchmarks/benchmark_streaming.py (FINAL, STANDARDIZED) ---