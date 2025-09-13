# --- START OF FILE HELIOS_EMBED/setup.py (FINAL v2.6.0 - Correct Source List) ---
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch
import os
import shutil

# --- Configuration ---
MODULE_NAME = "helios_embed"
VERSION = "1.0.0"

# --- Build-time CUDA Detection (Robust Method) ---
def _has_nvcc():
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        return True
    return shutil.which("nvcc") is not None

force_cuda = os.environ.get("FORCE_CUDA", "0") == "1"
torch_cuda_tag_present = getattr(torch.version, "cuda", None) is not None
nvcc_available = _has_nvcc()
can_build_cuda = force_cuda or (torch_cuda_tag_present and nvcc_available)

ExtensionType = CUDAExtension if can_build_cuda else CppExtension
print(f"[setup.py] Diagnosis: can_build_cuda={can_build_cuda} | torch.version.cuda_present={torch_cuda_tag_present} | nvcc_found={nvcc_available}")

# --- Dynamic C++11 ABI Flag from PyTorch ---
def get_torch_cxx11_abi():
    return int(torch._C._GLIBCXX_USE_CXX11_ABI)

# --- Source Files ---
project_root = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(project_root, 'src', MODULE_NAME)

# --- THIS IS THE CORRECTED SOURCE LIST ---
extension_sources = [
    'helios_embed_pybind.cpp',
    'incremental_nystrom_engine.cu', # We are looking for .cu files
    'nystrom_engine.cu',             # We are looking for .cu files
]
absolute_source_paths = [os.path.join(source_dir, f) for f in extension_sources]

# --- Build Configuration ---
cuda_version_str = getattr(torch.version, "cuda", None)
if can_build_cuda and cuda_version_str:
    final_version = f"{VERSION}+cu{cuda_version_str.replace('.', '')}"
else:
    final_version = f"{VERSION}+cpu"

print(f"--- Building {MODULE_NAME} version {final_version} ---")

def get_build_macros():
    torch_ver = torch.__version__.split('+')[0]
    cuda_ver = getattr(torch.version, "cuda", None) or "cpu"
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    return [
        ("HELIOS_BUILD_TORCH_VERSION", f'"{torch_ver}"'),
        ("HELIOS_BUILD_CUDA_VERSION", f'"{cuda_ver}"'),
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
# --- END OF FILE HELIOS_EMBED/setup.py (FINAL v2.6.0 - Correct Source List) ---