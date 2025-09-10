# --- START OF FILE HELIOS_EMBED/setup.py (FINAL v2.2.0) ---
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

project_root = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(project_root, 'src', 'helios_embed')

source_files = [
    'helios_embed_pybind.cpp',
    'incremental_nystrom_engine.cu',
    'nystrom_engine.cu'
]
absolute_source_paths = [os.path.join(source_dir, f) for f in source_files]

# A quick sanity check to ensure the files exist before attempting to build.
for p in absolute_source_paths:
    if not os.path.exists(p):
        raise FileNotFoundError(f"CRITICAL BUILD FAILURE: Source file not found: {p}")

setup(
    name='helios_embed',
    version='2.2.0',
    author='Michael Young & Ashley Kelly',
    description='Helios.Embed: The Nystrom Feature Engine',
    
    # This tells setuptools to look for packages in the 'src' directory
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
# --- END OF FILE HELIOS_EMBED/setup.py (FINAL v2.2.0) ---