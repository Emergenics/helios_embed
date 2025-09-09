# --- START OF FILE tests/benchmark_ash_gemm_DIAGNOSTIC.py ---
import torch
import numpy as np
import time
import sys
from pathlib import Path
import math
import os

LINE = "="*80
print(f"\n{LINE}")
print("--- Helios.Embed: Ash GEMM DIAGNOSTIC SUITE ---")
print(LINE)

# --- Step 1: Environment and Path Validation ---
print("\n--- 🔬 DIAGNOSTIC STEP 1: Environment Validation ---")
try:
    project_root = Path(__file__).parent.parent.resolve()
    src_path = project_root / 'src'
    core_so_path = next((src_path / 'helios_embed').glob('_core*.so'))
    
    print(f"  - Project Root: {project_root}")
    print(f"  - Src Path: {src_path}")
    print(f"  - Compiled Module Path: {core_so_path}")
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    print("  - 'src' directory is in Python path.")
    print("  ✅ Environment looks correct.")
except StopIteration:
    print(f"  - ❌ CRITICAL FAILURE: Compiled module ('_core*.so') not found in 'src/helios_embed/'.")
    print("  - REASON: The 'python setup.py build_ext --inplace' command likely failed or was not run.")
    sys.exit(1)

# --- Step 2: Module Import and Function Inspection ---
print("\n--- 🔬 DIAGNOSTIC STEP 2: Module Import & Function Inspection ---")
try:
    import helios_embed._core as helios_core
    print("  - ✅ Module 'helios_embed._core' imported successfully.")
    
    # Now, let's inspect the module to see what functions it actually contains.
    available_functions = dir(helios_core)
    print("  - Functions available in module:")
    for func in available_functions:
        if not func.startswith('__'):
            print(f"    - {func}")
            
    # Explicitly check for our target function
    if 'gemm_ash_algebra_cuda' in available_functions:
        print("  - ✅ 'gemm_ash_algebra_cuda' function is present in the compiled module.")
        gemm_ash_algebra_cuda = helios_core.gemm_ash_algebra_cuda
    else:
        print("  - ❌ CRITICAL FAILURE: 'gemm_ash_algebra_cuda' is MISSING from the compiled module.")
        print("  - REASON: This is a C++/Pybind linkage error. The function was either not included in the setup.py sources or not bound in the pybind.cpp file.")
        sys.exit(1)
        
except ImportError as e:
    print(f"  - ❌ CRITICAL FAILURE: Could not import 'helios_embed._core'.")
    print(f"     Error: {e}")
    sys.exit(1)

# If we've reached this point, the module is imported and the function is available.
# Now we can proceed to the actual benchmark logic.

# --- Helper Functions (Validated) ---
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

# --- Step 3: Execution and Verification ---
def run_ash_gemm_benchmark(N, repeats=10):
    print(f"\n--- 🔬 DIAGNOSTIC STEP 3: Execution & Verification (N={N}) ---")
    
    device = torch.device("cuda")
    
    print("  - Preparing CPU data...")
    A_cpu = create_bipolar_matrix_cpu(N, N, seed=1)
    B_cpu = create_bipolar_matrix_cpu(N, N, seed=2)
    
    print("  - Packing bipolar data to uint64...")
    A_packed_cpu = pack_bipolar_to_uint64(A_cpu)
    B_packed_T_cpu = pack_bipolar_to_uint64(B_cpu.T.copy())

    print("  - Moving data to GPU...")
    A_packed_gpu = torch.from_numpy(A_packed_cpu).to(device, dtype=torch.int64)
    B_packed_T_gpu = torch.from_numpy(B_packed_T_cpu).to(device, dtype=torch.int64)
    A_float_gpu = torch.from_numpy(A_cpu.astype(np.float32)).to(device)
    B_float_gpu = torch.from_numpy(B_cpu.astype(np.float32)).to(device)
    
    print("  - Running PyTorch/cuBLAS baseline for ground truth...")
    try:
        C_baseline_gpu = torch.matmul(A_float_gpu, B_float_gpu)
        torch.cuda.synchronize()
        C_baseline_int = C_baseline_gpu.to(torch.int32)
        print("  - ✅ Baseline computation successful.")
    except Exception as e:
        print(f"  - ❌ CRITICAL FAILURE: Baseline computation failed. Error: {e}")
        sys.exit(1)
        
    print("  - Running Ash GEMM CUDA kernel...")
    try:
        C_ash_gpu = gemm_ash_algebra_cuda(A_packed_gpu, B_packed_T_gpu, N)
        torch.cuda.synchronize()
        print("  - ✅ Ash GEMM kernel executed without crashing.")
    except Exception as e:
        print(f"  - ❌ CRITICAL FAILURE: Ash GEMM kernel execution failed. Error: {e}")
        sys.exit(1)
        
    print("  - Verifying accuracy...")
    abs_error = torch.abs(C_baseline_int - C_ash_gpu).sum().item()
    
    if abs_error == 0:
        print(f"  - ✅ ACCURACY VALIDATED: Absolute Error is {abs_error}.")
    else:
        print(f"  - ❌ CRITICAL FAILURE: Accuracy test failed. Absolute Error is {abs_error}.")
        sys.exit(1)
        
    print("\n--- ✅ DIAGNOSTIC COMPLETE: All steps passed. ---")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Aborting diagnostic.")
        sys.exit(1)
    
    run_ash_gemm_benchmark(N=1024, repeats=10) # Use a smaller size for a quick diagnostic
# --- END OF FILE tests/benchmark_ash_gemm_DIAGNOSTIC.py ---