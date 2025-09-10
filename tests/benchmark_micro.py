# --- START OF FILE tests/benchmark_micro.py (FINAL, PROFILER-AWARE v1.1) ---
import torch
import sys
from pathlib import Path
import pandas as pd
import subprocess

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

def run():
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
        landmarks = X[torch.randperm(N, device=device)[:m]]
        gamma = 0.1; ridge = 1e-6

        # --- THIS IS THE CRITICAL FIX: Use torch.cuda.Event for manual timing ---
        # The high-level profiler is unreliable for custom C++ extensions.
        # Manual, event-based timing is the gold standard for CUDA.
        
        # Warmup
        _ = compute_rkhs_embedding(X, landmarks, gamma, ridge)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        repeats = 10
        start_event.record()
        for _ in range(repeats):
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
    run()

# --- END OF FILE tests/benchmark_micro.py (FINAL, PROFILER-AWARE v1.1) ---