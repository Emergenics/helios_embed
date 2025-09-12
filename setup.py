# --- START OF FILE HELIOS_EMBED/setup.py (FINAL v3.1 - Dynamic Sources) ---
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch
import os
import shutil
import sys

# --- Configuration ---
MODULE_NAME = "helios_embed"
VERSION = "1.0.0"

# --- Build-time CUDA Detection (Robust Method) ---
def _has_nvcc():
    """Checks for the presence of the nvcc compiler in the build environment."""
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        return True
    return shutil.which("nvcc") is not None

force_cuda = os.environ.get("FORCE_CUDA", "0") == "1"
torch_cuda_tag_present = getattr(torch.version, "cuda", None) is not None
nvcc_available = _has_nvcc()
can_build_cuda = force_cuda or (torch_cuda_tag_present and nvcc_available)

# --- Dynamic C++11 ABI Flag from PyTorch ---
def get_torch_cxx11_abi():
    """Returns 1 or 0 based on the PyTorch build's ABI."""
    try:
        return int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except ImportError:
        # This can happen in very stripped-down environments. Default to 0.
        return 0

# --- Source Files & Extension Configuration ---
project_root = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(project_root, 'src', MODULE_NAME)

common_sources = [
    'helios_embed_pybind.cpp',
]

if can_build_cuda:
    print("--- Building for CUDA ---")
    ExtensionType = CUDAExtension
    extension_sources = common_sources + [
        'incremental_nystrom_engine.cu',
        'nystrom_engine.cu',
    ]
    cuda_version_str = getattr(torch.version, "cuda")
    final_version = f"{VERSION}+cu{cuda_version_str.replace('.', '')}"
    
    define_macros = [
        ("HELIOS_BUILD_TORCH_VERSION", f'"{torch.__version__.split("+")[0]}"'),
        ("HELIOS_BUILD_CUDA_VERSION", f'"{cuda_version_str}"'),
        ("HELIOS_BUILD_CXX11_ABI", f'"{get_torch_cxx11_abi()}"'),
    ]
    extra_nvcc_args = ['-O3', '--expt-relaxed-constexpr', '-std=c++17']

else:
    print("--- Building for CPU-ONLY with Stubs ---")
    ExtensionType = CppExtension
    extension_sources = common_sources + [
        'incremental_nystrom_engine_cpu.cpp',
        'nystrom_engine_cpu.cpp',
    ]
    final_version = f"{VERSION}+cpu"
    
    define_macros = [
        ("HELIOS_BUILD_TORCH_VERSION", f'"{torch.__version__.split("+")[0]}"'),
        ("HELIOS_BUILD_CUDA_VERSION", '"cpu"'),
        ("HELIOS_BUILD_CXX11_ABI", f'"{get_torch_cxx11_abi()}"'),
        ("HELIOS_CPU_BUILD", "1"), # Macro to guard CUDA-only code
    ]
    extra_nvcc_args = []

absolute_source_paths = [os.path.join(source_dir, f) for f in extension_sources]

print(f"--- Building {MODULE_NAME} version {final_version} (CUDA Build: {can_build_cuda}) ---")

# --- Setup Configuration ---
setup(
    name=MODULE_NAME,
    version=final_version,
    author='IRBSurfer & Ashley Kelly',
    description='Helios.Embed: A Production-Ready, High-Performance Nystrom Feature Engine',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    
    ext_modules=[
        ExtensionType(
            name=f'{MODULE_NAME}._core', 
            sources=absolute_source_paths,
            define_macros=define_macros,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', f'-D_GLIBCXX_USE_CXX11_ABI={get_torch_cxx11_abi()}'],
                'nvcc': extra_nvcc_args
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=["torch>=2.1.2,<2.2"], 
)
# --- END OF FILE HELIOS_EMBED/setup.py (FINAL v3.1 - Dynamic Sources) ---