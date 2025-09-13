# --- START OF FILE src/helios_embed/__init__.py (FINAL v1.2 - Build-Aware) ---
import torch
import warnings

# Step 1: Import the core C++ module. This contains the 'get_build_info' function.
from . import _core

# Step 2: Immediately get the build information.
build_info = _core.get_build_info()
IS_CUDA_BUILD = build_info.get("cuda_version", "cpu") != "cpu"

# Step 3: The Runtime Guard (uses the build_info we just fetched)
def _runtime_guard():
    runtime_torch_ver = torch.__version__.split('+')[0]
    def _major_minor(v_str): return ".".join(v_str.split('.')[:2])
    if _major_minor(runtime_torch_ver) != _major_minor(build_info["torch_version"]):
        raise ImportError(f"Torch version mismatch: built with {build_info['torch_version']}, running with {runtime_torch_ver}.")
    
    runtime_cuda_ver  = getattr(torch.version, "cuda", None) or "cpu"
    if IS_CUDA_BUILD and runtime_cuda_ver == "cpu":
        raise ImportError(f"CUDA mismatch: This is a CUDA-enabled build but runtime is CPU-only.")
    if not IS_CUDA_BUILD and runtime_cuda_ver != "cpu":
        warnings.warn("You are running a CPU-only build of Helios.Embed with a CUDA-enabled PyTorch.", RuntimeWarning)

_runtime_guard()

# Step 4: Conditionally expose the public API
if IS_CUDA_BUILD:
    # If this is a CUDA build, we expect the core functions to exist.
    from ._core import compute_rkhs_embedding, IncrementalNystromEngine
else:
    # If this is a CPU-only build, we define Python stubs that will raise a helpful error.
    def _cuda_not_available(*args, **kwargs):
        raise NotImplementedError(
            "This function is not available in the CPU-only build of Helios.Embed. "
            "Please install the CUDA-enabled version to use this functionality."
        )
    compute_rkhs_embedding = _cuda_not_available
    IncrementalNystromEngine = _cuda_not_available

# Step 5: Always expose the build info function
get_build_info = _core.get_build_info

# Clean up namespace
del torch, warnings, _runtime_guard, build_info, IS_CUDA_BUILD
# --- END OF FILE src/helios_embed/__init__.py (FINAL v1.2 - Build-Aware) ---