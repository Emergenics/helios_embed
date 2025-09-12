# --- START OF FILE HELIOS_EMBED/setup.py (FINAL v2.7.0 - Robust CUDA Detection) ---
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch
import os
import sys
import shutil

# --- Configuration ---
MODULE_NAME = "helios_embed"
VERSION = "1.0.0"

# --- THIS IS THE CRITICAL FIX: Robust, Build-Time CUDA Detection ---
def has_nvcc():
    """Checks for the presence of the nvcc compiler in the environment."""
    # This check is purely based on the build environment, not runtime torch.
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        return True
    return shutil.which("nvcc") is not None

# The decision to build a CUDA extension depends on two things:
# 1. Is the installed PyTorch package *aware* of CUDA?
# 2. Is the nvcc compiler *actually available* in this build environment?
has_torch_cuda_support = getattr(torch.version, "cuda", None) is not None
can_build_cuda = has_torch_cuda_support and has_nvcc()

ExtensionType = CUDAExtension if can_build_cuda else CppExtension
# --- END OF CRITICAL FIX ---


# --- Dynamic C++11 ABI Flag from PyTorch ---
def get_torch_cxx11_abi():
    return int(torch._C._GLIBCXX_USE_CXX11_ABI)

# --- Source Files ---
project_root = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(project_root, 'src', MODULE_NAME)
source_files = [
    'helios_embed_pybind.cpp',
    'incremental_nystrom_engine.cu',
    'nystrom_engine.cu'
]
absolute_source_paths = [os.path.join(source_dir, f) for f in source_files]


# --- Versioning with PEP 440 Local Version Identifier (+cuXXX / +cpu) ---
cuda_version_str = getattr(torch.version, "cuda", None)
if can_build_cuda and cuda_version_str:
    cuda_tag = "+cu" + cuda_version_str.replace('.', '')
    final_version = VERSION + cuda_tag
else:
    cuda_tag = "+cpu"
    final_version = VERSION + cuda_tag

print(f"--- Building {MODULE_NAME} version {final_version} (CUDA Build: {can_build_cuda}) ---")


# --- Build-time Metadata Injection ---
def get_build_macros():
    torch_ver = torch.__version__.split('+')[0]
    # Use the version string from the build, not a runtime check
    cuda_ver_str = cuda_version_str if can_build_cuda else "cpu"
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    return [
        ("HELIOS_BUILD_TORCH_VERSION", f'"{torch_ver}"'),
        ("HELIOS_BUILD_CUDA_VERSION", f'"{cuda_ver_str}"'),
        ("HELIOS_BUILD_ARCH_LIST", f'"{arch_list}"'),
        ("HELIOS_BUILD_CXX11_ABI", f'"{get_torch_cxx11_abi()}"'),
    ]

# --- Setup Configuration ---
setup(
    name=MODULE_NAME,
    version=final_version,
    author='IRBSurfer & Ashley Kelly',
    description='Helios.Embed: The Final, Validated Nystrom Feature Engine',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    
    ext_modules=[
        ExtensionType(
            name=f'{MODULE_NAME}._core', 
            sources=absolute_source_paths,
            define_macros=get_build_macros(),
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', f'-D_GLIBCXX_USE_CXX11_ABI={get_torch_cxx11_abi()}'],
                'nvcc': ['-O3', '--expt-relaxed-constexpr', '-std=c++17'] if can_build_cuda else []
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=["torch>=2.1.2,<2.2"], 
)
# --- END OF FILE HELIOS_EMBED/setup.py (FINAL v2.8.0 - Robust CUDA Detection) ---