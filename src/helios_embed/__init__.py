# --- START OF FILE src/helios_embed/__init__.py (FINAL v1.0.0 with Guard) ---
import torch
import warnings

# --- Step 1: Attempt to import the compiled C++ extension ---
try:
    from . import _core
except ImportError:
    # This is a critical error, often due to a failed compilation.
    raise ImportError(
        "Failed to import the Helios.Embed C++/CUDA core module.\n"
        "Please ensure the module has been compiled successfully by running:\n"
        "  pip install -e ."
    )

# --- Step 2: The Runtime Guard ---
def _runtime_guard():
    """
    Performs fast, import-time checks to ensure the runtime environment
    is compatible with the build-time environment of the C++ module.
    This prevents cryptic CUDA errors and crashes down the line.
    """
    try:
        build_info = _core.get_build_info()
        runtime_torch_ver = torch.__version__.split('+')[0]
        runtime_cuda_ver  = getattr(torch.version, "cuda", None) or "cpu"

        # Helper to compare major.minor versions
        def _major_minor(v_str):
            return ".".join(v_str.split('.')[:2])

        # Check 1: PyTorch Version Mismatch
        if _major_minor(runtime_torch_ver) != _major_minor(build_info["torch_version"]):
            raise ImportError(
                f"[Helios.Embed] Environment Mismatch: The module was built against PyTorch version "
                f"{build_info['torch_version']}, but you are running with PyTorch {runtime_torch_ver}.\n"
                f"Please install a version of Helios.Embed that matches your PyTorch installation."
            )

        # Check 2: CUDA Version Mismatch
        build_cuda = build_info["cuda_version"]
        if build_cuda == "cpu":
            if runtime_cuda_ver != "cpu":
                warnings.warn(
                    "[Helios.Embed] You have installed the CPU-only version of Helios.Embed, "
                    "but you have a CUDA-enabled version of PyTorch. The engine will run on the CPU.",
                    RuntimeWarning
                )
        else: # Module was built for CUDA
            if runtime_cuda_ver == "cpu":
                raise ImportError(
                    f"[Helios.Embed] Environment Mismatch: You have installed a CUDA-enabled version of "
                    f"Helios.Embed (for CUDA {build_cuda}), but your PyTorch installation is CPU-only.\n"
                    f"Please install the CPU version of Helios.Embed or a CUDA-enabled version of PyTorch."
                )
            if _major_minor(runtime_cuda_ver) != _major_minor(build_cuda):
                 raise ImportError(
                    f"[Helios.Embed] Environment Mismatch: The module was built for CUDA {build_cuda}, "
                    f"but your PyTorch is running with CUDA {runtime_cuda_ver}.\n"
                    f"Please install the version of Helios.Embed that matches your PyTorch's CUDA version."
                )
        
        # Check 3: C++ ABI Mismatch (this is a critical, subtle check)
        build_abi = build_info["cxx11_abi"]
        runtime_abi = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
        if build_abi != runtime_abi:
            raise ImportError(
                f"[Helios.Embed] C++ ABI Mismatch: The module was built with _GLIBCXX_USE_CXX11_ABI={build_abi}, "
                f"but your PyTorch was built with _GLIBCXX_USE_CXX11_ABI={runtime_abi}.\n"
                f"This is a critical incompatibility. Please rebuild Helios.Embed from source in your environment."
            )

    except Exception as e:
        # If the guard itself fails, re-raise the error with a helpful message.
        raise ImportError(f"Helios.Embed failed an import-time compatibility check. Error: {e}")

# --- Step 3: Execute the guard and expose the public API ---
_runtime_guard()

# Expose the public API at the top level of the package
from ._core import compute_rkhs_embedding, IncrementalNystromEngine

# Clean up namespace
del torch, warnings, _runtime_guard
# --- END OF FILE src/helios_embed/__init__.py (FINAL v1.0.0 with Guard) ---