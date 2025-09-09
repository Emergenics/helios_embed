# --- START OF FILE HELIOS_EMBED/setup.py (Version 1.2.0 - Hybrid Kernel) ---
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

project_root = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(project_root, 'src', 'helios_embed')

# --- DEFINITIVE SOURCE LIST FOR THIS PHASE ---
# REMOVED: rbf_kernel.cu, ash_gemm.cu (They are not part of the hybrid kernel test)
# ADDED: hybrid_kernel.cu
source_file_names = [
    'helios_embed_pybind.cpp',
    'incremental_nystrom_engine.cu',
    'nystrom_engine.cu',
    'hybrid_kernel.cu' # <-- The new hybrid kernel
]
absolute_source_paths = [os.path.join(source_dir, f) for f in source_file_names]

# Sanity check
for p in absolute_source_paths:
    if not os.path.exists(p):
        raise FileNotFoundError(f"CRITICAL BUILD FAILURE: Source file not found: {p}")

setup(
    name='helios_embed',
    version='1.2.0', # Version bump for hybrid kernel
    author='IRBSurfer & Ashley Kelly',
    description='Helios.Embed: Nystrom Engine with Hybrid Kernel Accelerator',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    ext_modules=[
        CUDAExtension(
            name='helios_embed._core', 
            sources=absolute_source_paths,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': ['-O3', '--expt-relaxed-constexpr', '-std=c++17', '-gencode', 'arch=compute_75,code=sm_75']
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
# --- END OF FILE HELIOS_EMBED/setup.py (Version 1.2.0 - Hybrid Kernel) ---