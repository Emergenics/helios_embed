# --- START OF FILE HELIOS_EMBED/setup.py (Final Corrected Version v1.2.0) ---
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

project_root = os.path.dirname(os.path.abspath(__file__))
# The source directory is now explicitly defined
source_dir = os.path.join(project_root, 'src', 'helios_embed')

# We list the source files relative to the project root for clarity
source_files = [
    'src/helios_embed/helios_embed_pybind.cpp',
    'src/helios_embed/incremental_nystrom_engine.cu',
    'src/helios_embed/nystrom_engine.cu'
]

setup(
    name='helios_embed',
    version='1.2.0', # Version bump for this critical fix
    author='IRBSurfer & Ashley Kelly',
    description='Helios.Embed: A Production-Ready, High-Performance Nyström Feature Engine for PyTorch',
    
    # These two lines are the critical fix.
    # They tell setuptools that our packages live inside the 'src' directory.
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    
    ext_modules=[
        CUDAExtension(
            # The name MUST match the package structure: helios_embed._core
            name='helios_embed._core', 
            sources=source_files,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': ['-O3', '--expt-relaxed-constexpr', '-std=c++17', '-gencode', 'arch=compute_75,code=sm_75']
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
# --- END OF FILE HELIOS_EMBED/setup.py (Final Corrected Version v1.2.0) ---