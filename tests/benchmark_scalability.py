# --- START OF FILE tests/benchmark_scalability.py (Corrected Paths) ---
import torch
import time
import sys
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import gc
import os
from pathlib import Path # Import Path

LINE = "="*80
print(f"\n{LINE}")
print("--- Helios.Embed: Definitive Scalability & Performance Benchmark ---")
print(LINE)

try:
    from helios_embed._core import IncrementalNystromEngine, compute_rkhs_embedding
except ImportError as e:
    print(f"❌ Helios.Embed module not compiled. Run 'pip install -e .'. Error: {e}")
    sys.exit()

# --- (The rest of the script is unchanged until the Main Runner) ---
def get_gpu_memory_usage(device):
    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / (1024**2) # in MB
    return 0

def run_scalability_test(N, D, m, device, repeats=5):
    print(f"\n--- 🧪 Testing Config: N={N}, D={D}, m={m} ---")
    torch.manual_seed(42)
    gamma = 0.1; ridge = 1e-6
    try:
        X = torch.randn(N, D, device=device, dtype=torch.float32)
        if N >= m: landmarks = X[torch.randperm(N, device=device)[:m]]
        else: landmarks = torch.randn(m, D, device=device, dtype=torch.float32)
    except torch.cuda.OutOfMemoryError:
        print("  - ⚠️ OOM during data generation. Skipping test.")
        gc.collect(); torch.cuda.empty_cache()
        return None
    time_build = -1.0; mem_build = -1.0
    try:
        _ = compute_rkhs_embedding(X, landmarks, gamma, ridge)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(device); mem_before = get_gpu_memory_usage(device)
        start_event.record()
        for _ in range(repeats): features = compute_rkhs_embedding(X, landmarks, gamma, ridge)
        end_event.record()
        torch.cuda.synchronize()
        time_build = start_event.elapsed_time(end_event) / repeats
        mem_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        mem_build = mem_peak - mem_before
        print(f"  - Stateless Build Time: {time_build:.4f} ms")
        print(f"  - Stateless Build Peak Memory: {mem_build:.2f} MB")
    except torch.cuda.OutOfMemoryError:
        print("  - ⚠️ OOM during stateless build. Marking as failed.")
        time_build = float('inf'); mem_build = float('inf')
    del X, landmarks, features
    gc.collect(); torch.cuda.empty_cache()
    return {"N": N, "D": D, "m": m, "build_time_ms": time_build, "build_mem_mb": mem_build}

# --- Main Runner ---
def run():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping scalability tests."); return

    device = torch.device("cuda")
    
    N_values = [1024, 4096, 8192, 16384, 32768]
    D_values = [128, 384, 768]
    m_values = [64, 128, 256, 512]
    all_configs = list(product(N_values, D_values, m_values))
    all_results = []

    for N, D, m in all_configs:
        if (N * D * 4 > 8e9) or (m * D * 4 > 8e9) or (N * m * 4 > 8e9):
            print(f"\n--- ⏭️ Skipping large config (heuristic): N={N}, D={D}, m={m} ---")
            continue
        result = run_scalability_test(N, D, m, device, repeats=10)
        if result: all_results.append(result)

    if not all_results:
        print("\nNo benchmark results collected."); return
        
    df = pd.DataFrame(all_results)
    
    print(f"\n{LINE}\n--- 📊 Helios.Embed: Scalability Benchmark Summary ---\n{LINE}")
    print(df.to_string())
    
    # --- THIS IS THE CRITICAL FIX ---
    # Create an output directory relative to the *project root* for robustness.
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'benchmark_outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Define corrected output paths
    csv_path = output_dir / "scalability_results.csv"
    plot_path = output_dir / "scalability_plots.png"
    # --- END OF CRITICAL FIX ---

    df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Results saved to '{csv_path}'")

    print("\n--- 📈 Generating Performance Plots ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Helios.Embed: Performance & Scalability Envelope', fontsize=16)
    
    ax = axes[0, 0]
    fixed_D, fixed_m = 768, 256
    subset1 = df[(df['D'] == fixed_D) & (df['m'] == fixed_m)]
    if not subset1.empty:
        ax.plot(subset1['N'], subset1['build_time_ms'], 'o-'); ax.set_title(f'Time vs. N (D={fixed_D}, m={fixed_m})')
        ax.set_xlabel('Number of Vectors (N)'); ax.set_ylabel('Build Time (ms)'); ax.grid(True, alpha=0.3); ax.set_xscale('log'); ax.set_yscale('log')

    ax = axes[0, 1]
    fixed_N, fixed_m = 8192, 256
    subset2 = df[(df['N'] == fixed_N) & (df['m'] == fixed_m)]
    if not subset2.empty:
        ax.plot(subset2['D'], subset2['build_time_ms'], 'o-'); ax.set_title(f'Time vs. D (N={fixed_N}, m={fixed_m})')
        ax.set_xlabel('Feature Dimension (D)'); ax.set_ylabel('Build Time (ms)'); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    fixed_N, fixed_D = 8192, 768
    subset3 = df[(df['N'] == fixed_N) & (df['D'] == fixed_D)]
    if not subset3.empty:
        ax.plot(subset3['m'], subset3['build_time_ms'], 'o-'); ax.set_title(f'Time vs. m (N={fixed_N}, D={fixed_D})')
        ax.set_xlabel('Number of Landmarks (m)'); ax.set_ylabel('Build Time (ms)'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if not subset1.empty:
        ax.plot(subset1['N'], subset1['build_mem_mb'], 'o-'); ax.set_title(f'Peak Memory vs. N (D={fixed_D}, m={fixed_m})')
        ax.set_xlabel('Number of Vectors (N)'); ax.set_ylabel('Peak Additional Memory (MB)'); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_path)
    print(f"  ✅ Plots saved to '{plot_path}'")
    # plt.show() # Commented out for non-interactive runner

if __name__ == "__main__":
    run()

# --- END OF FILE tests/benchmark_scalability.py (Corrected Paths) ---