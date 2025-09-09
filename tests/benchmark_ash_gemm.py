# --- START OF FILE tests/benchmark_ash_gemm.py (Hardened Version) ---
import torch
import numpy as np
import time
import sys
from pathlib import Path
import math

# Add src to path to allow direct import for testing
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

LINE = "="*80

# --- THIS IS THE CRITICAL FIX: Fail immediately if the import is broken ---
try:
    from helios_embed._core import gemm_ash_algebra_cuda
except ImportError as e:
    print(f"\n--- ❌ CRITICAL FAILURE: Could not import 'gemm_ash_algebra_cuda'. ---")
    print(f"   The C++ extension is not compiled correctly. Error: {e}")
    print("   Aborting test.")
    sys.exit(1) # Exit with a failure code
# --- END OF CRITICAL FIX ---


def create_bipolar_matrix_cpu(rows, cols, seed=42):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(rows, cols)).astype(np.int8)

def pack_bipolar_to_uint64(matrix_cpu):
    binary_matrix = ((matrix_cpu + 1) // 2).astype(np.uint8)
    rows, cols = binary_matrix.shape
    padded_cols = math.ceil(cols / 64) * 64
    if cols != padded_cols:
        padded_matrix = np.zeros((rows, padded_cols), dtype=np.uint8)
        padded_matrix[:, :cols] = binary_matrix
    else:
        padded_matrix = binary_matrix
    packed = np.packbits(padded_matrix, axis=1, bitorder='little')
    return packed.view(np.uint64)

def run_ash_gemm_benchmark(N, repeats=10):
    print(f"\n--- 🧪 Testing Ash GEMM: N={N} ---")
    
    device = torch.device("cuda")
    
    A_cpu = create_bipolar_matrix_cpu(N, N, seed=1)
    B_cpu = create_bipolar_matrix_cpu(N, N, seed=2)
    A_packed_cpu = pack_bipolar_to_uint64(A_cpu)
    B_packed_T_cpu = pack_bipolar_to_uint64(B_cpu.T.copy())

    A_packed_gpu = torch.from_numpy(A_packed_cpu).to(device, dtype=torch.int64)
    B_packed_T_gpu = torch.from_numpy(B_packed_T_cpu).to(device, dtype=torch.int64)
    
    A_float_gpu = torch.from_numpy(A_cpu.astype(np.float32)).to(device)
    B_float_gpu = torch.from_numpy(B_cpu.astype(np.float32)).to(device)
    
    # Warmup
    C_baseline_gpu = torch.matmul(A_float_gpu, B_float_gpu)
    C_ash_gpu = gemm_ash_algebra_cuda(A_packed_gpu, B_packed_T_gpu, N)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Time Baseline
    start_event.record()
    for _ in range(repeats):
        C_baseline_gpu = torch.matmul(A_float_gpu, B_float_gpu)
    end_event.record()
    torch.cuda.synchronize()
    time_baseline = start_event.elapsed_time(end_event) / repeats

    # Time Ash GEMM
    start_event.record()
    for _ in range(repeats):
        C_ash_gpu = gemm_ash_algebra_cuda(A_packed_gpu, B_packed_T_gpu, N)
    end_event.record()
    torch.cuda.synchronize()
    time_ash = start_event.elapsed_time(end_event) / repeats
    
    C_baseline_int = C_baseline_gpu.to(torch.int32)
    abs_error = torch.abs(C_baseline_int - C_ash_gpu).sum().item()
    speedup = time_baseline / time_ash

    print(f"  - PyTorch/cuBLAS Time: {time_baseline:.4f} ms")
    print(f"  - Ash GEMM Time:        {time_ash:.4f} ms")
    print(f"  - Speedup vs cuBLAS:    {speedup:.2f}x")
    print(f"  - Absolute Error:       {abs_error}")
    
    if abs_error == 0 and speedup >= 1.5:
         print("  - ✅ HYPOTHESIS VALIDATED: Ash GEMM is bit-perfect and significantly faster.")
    elif abs_error != 0:
         print("  - ❌ HYPOTHESIS FALSIFIED: Accuracy failure.")
         sys.exit(1)
    else:
         print("  - ⚠️ HYPOTHESIS NOT MET: Speedup target of 1.5x not achieved.")


def run():
    print(f"\n{LINE}")
    print("--- Ash GEMM vs cuBLAS Performance Benchmark ---")
    print(LINE)
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    run_ash_gemm_benchmark(N=4096, repeats=10)

if __name__ == "__main__":
    run()
# --- END OF FILE tests/benchmark_ash_gemm.py (Hardened Version) ---