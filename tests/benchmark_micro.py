# --- START OF FILE tests/benchmark_micro.py (FINAL v1.2 with CSV Output) ---
import torch
import sys
from pathlib import Path
import pandas as pd
import argparse # Import argparse to handle command-line arguments

# Standard setup for path and import
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from helios_embed._core import compute_rkhs_embedding
except ImportError as e:
    print(f"❌ Module not compiled. Run setup.py. Error: {e}"); sys.exit(1)

LINE = "="*80

# This function is now the main execution logic.
def run(csv_output_path=None):
    print(f"\n{LINE}")
    print("--- 🚀 Helios.Embed: Definitive Microbenchmark & Profiling Suite ---")
    print(LINE)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping microbenchmark."); return

    device = torch.device("cuda")
    
    configs = [
        {'N': 4096, 'D': 128, 'm': 256},
        {'N': 8192, 'D': 768, 'm': 256},
        {'N': 16384, 'D': 384, 'm': 512},
    ]
    
    all_results = []

    for config in configs:
        N, D, m = config['N'], config['D'], config['m']
        print(f"\n--- 🔬 Profiling Config: N={N}, D={D}, m={m} ---")
        
        torch.manual_seed(42)
        X = torch.randn(N, D, device=device)
        # For a realistic scenario, landmarks are a subset of the data
        landmarks = X[torch.randperm(N, device=device)[:m]]
        gamma = 0.1; ridge = 1e-6

        # Use torch.cuda.Event for manual timing, which is the gold standard for CUDA.
        
        # Warmup
        _ = compute_rkhs_embedding(X, landmarks, gamma, ridge)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        repeats = 10
        start_event.record()
        for _ in range(repeats):
            # The record_function adds a label to Nsight/PyTorch profiler timelines
            with torch.autograd.profiler.record_function("Helios.Embed::compute_rkhs_embedding_manual_NVTX"):
                compute_rkhs_embedding(X, landmarks, gamma, ridge)
        end_event.record()
        
        torch.cuda.synchronize()
        
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / repeats

        bytes_moved_approx_gb = (2 * N * m + 2 * m * m) * 4 / (1024**3)

        print(f"  - Total GPU Kernel Time (avg, via Events): {avg_time_ms:.4f} ms")
        print(f"  - Approx. Bytes Moved (internal): {bytes_moved_approx_gb:.3f} GB")
        
        all_results.append({
            'N': N, 'D': D, 'm': m,
            'kernel_time_ms': avg_time_ms,
            'bytes_moved_gb': bytes_moved_approx_gb,
        })
    
    df = pd.DataFrame(all_results)
    
    # --- THIS IS THE MINIMAL LOGIC ADDITION FOR CI ---
    if csv_output_path:
        # The performance comparator script will provide this path.
        # We need to reshape the data into the simple "name,metric,value" format.
        perf_data = []
        for _, row in df.iterrows():
            config_name = f"config_{int(row['N'])}_{int(row['D'])}_{int(row['m'])}"
            perf_data.append({'name': config_name, 'metric': 'kernel_time_ms', 'value': row['kernel_time_ms']})
        
        perf_df = pd.DataFrame(perf_data)
        perf_df.to_csv(csv_output_path, index=False)
        print(f"--- ✅ Performance data for CI saved to: {csv_output_path} ---")
    # --- END OF ADDITION ---
    
    else: # Default behavior when run manually from the command line
        output_dir = project_root / "benchmark_outputs"
        output_dir.mkdir(exist_ok=True)
        baseline_path = output_dir / "performance_baseline_v1.csv"
        df.to_csv(baseline_path, index=False, float_format='%.6f')
        
        print(f"\n{LINE}")
        print(f"--- ✅ PERFORMANCE BASELINE ESTABLISHED ---")
        print(f"---    Results saved to: {baseline_path}    ---")
        print(LINE)
        print(df.to_string())

if __name__ == "__main__":
    # --- THIS IS THE MINIMAL LOGIC ADDITION FOR CI ---
    parser = argparse.ArgumentParser(description="Helios.Embed Microbenchmark Runner")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional path to save performance results in CI format.")
    args = parser.parse_args()
    # --- END OF ADDITION ---
    
    run(csv_output_path=args.csv_out)

# --- END OF FILE tests/benchmark_micro.py (FINAL v1.2 with CSV Output) ---