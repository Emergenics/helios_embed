# --- START OF FILE src/helios_embed/__init__.py (FINAL v1.1 - Simplified & Correct) ---
# This file makes the 'helios_embed' directory a Python package
# and exposes the core C++ functions at the top level.

import torch
import warnings

# Directly import the compiled C++ core module.
# If this fails, it's a critical installation error, and it SHOULD raise immediately.
from . import _core

# --- The Runtime Guard (unchanged) ---
def _runtime_guard():
    build_info = _core.get_build_info()
    runtime_torch_ver = torch.__version__.split('+')[0]
    runtime_cuda_ver  = getattr(torch.version, "cuda", None) or "cpu"
    def _major_minor(v_str): return ".".join(v_str.split('.')[:2])
    if _major_minor(runtime_torch_ver) != _major_minor(build_info["torch_version"]):
        raise ImportError(f"Torch version mismatch: built with {build_info['torch_version']}, running with {runtime_torch_ver}.")
    build_cuda = build_info["cuda_version"]
    if build_cuda != "cpu" and runtime_cuda_ver == "cpu":
        raise ImportError(f"CUDA mismatch: This is a CUDA-enabled build ({build_cuda}) but runtime is CPU-only.")
    if build_cuda == "cpu" and runtime_cuda_ver != "cpu":
        warnings.warn("You are running a CPU-only build of Helios.Embed with a CUDA-enabled PyTorch.", RuntimeWarning)

_runtime_guard()

# --- Expose the public API ---
# This makes the functions available directly, e.g., `helios_embed.compute_rkhs_embedding`
from ._core import compute_rkhs_embedding, IncrementalNystromEngine

# Clean up the package namespace
del torch, warnings, _runtime_guard
# --- END OF FILE src/helios_embed/__init__.py (FINAL v1.1 - Simplified & Correct) ---