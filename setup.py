# --- START OF FILE HELIOS_EMBED/setup.py (FINAL v2.3.0 with Sanitizers) ---
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

# --- SANITIZER SUPPORT ---
# Read environment variables to enable sanitizers
use_asan = os.environ.get('HELIOS_ASAN', '0') == '1'
use_ubsan = os.environ.get('HELIOS_UBSAN', '0') == '1'

extra_cxx_args = ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0']
extra_nvcc_args = ['-O3', '--expt-relaxed-constexpr', '-std=c++17', '-gencode', 'arch=compute_75,code=sm_75']

if use_asan:
    print("--- 🛠️  Building with AddressSanitizer (ASan) enabled ---")
    extra_cxx_args.extend(['-fsanitize=address', '-fno-omit-frame-pointer'])

if use_ubsan:
    print("--- 🛠️  Building with UndefinedBehaviorSanitizer (UBSan) enabled ---")
    extra_cxx_args.extend(['-fsanitize=undefined'])

setup(
    name='helios_embed',
    version='2.3.0', # Version bump for sanitizer support
    author='IRBSurfer & Ashley Kelly',
    description='Helios.Embed: The Final, Validated Nystrom Feature Engine',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    ext_modules=[
        CUDAExtension(
            name='helios_embed._core', 
            sources=absolute_source_paths,
            extra_compile_args={
                'cxx': extra_cxx_args,
                'nvcc': extra_nvcc_args
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
# --- END OF FILE HELIOS_EMBED/setup.py (FINAL v2.3.0 with Sanitizers) ---