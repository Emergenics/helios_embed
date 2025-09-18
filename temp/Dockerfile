# --- START OF FILE HELIOS_EMBED/Dockerfile (FINAL v1.1 - Production Grade) ---
# Helios.Embed manylinux CUDA 11.8 builder (single-Python cp310)

# --- Stage 1: CUDA toolchain provider (CentOS 7 base matches manylinux2014 glibc era) ---
FROM nvidia/cuda:11.8.0-devel-centos7 AS cuda118

# --- Stage 2: manylinux builder with Python toolchains and auditwheel policy ---
FROM quay.io/pypa/manylinux2014_x86_64 AS builder

# copy CUDA toolkit into the manylinux image
COPY --from=cuda118 /usr/local/cuda /usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Use manylinux Python interpreter for correct wheel tags
ENV PYBIN=/opt/python/cp310-cp310/bin
ENV PATH=${PYBIN}:${PATH}
RUN ${PYBIN}/python -m pip install --upgrade pip setuptools wheel build auditwheel pybind11 ninja

# Pin Torch to the CUDA variant you target (cu118 here)
RUN ${PYBIN}/python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
       torch==2.1.2+cu118 --extra-index-url https://pypi.org/simple

# Build-time knobs
ENV TORCH_CUDA_ARCH_LIST=7.5
ENV MAX_JOBS=8

# Build on container start, against the bind-mounted /io
WORKDIR /io
ENTRYPOINT ["/bin/bash","-lc","${PYBIN}/python -m build --wheel --no-isolation && ${PYBIN}/python -m pip install dist/*.whl && ${PYBIN}/python -c \"import torch, helios_embed; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('build', helios_embed._core.get_build_info())\" && auditwheel show dist/*.whl"]
# --- END OF FILE HELIOS_EMBED/Dockerfile (FINAL v1.1 - Production Grade) ---