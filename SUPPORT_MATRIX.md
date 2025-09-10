# --- START OF FILE HELIOS_EMBED/SUPPORT_MATRIX.md ---
# Helios.Embed v1.0.0 - Official Support Matrix

This document specifies the exact versions, platforms, and hardware architectures on which `Helios.Embed` v1.0.0 is officially tested and guaranteed to be performant, stable, and bit-perfectly accurate.

---

## 1. Environment & Runtimes

| Component | Supported Version(s) | Notes |
| :--- | :--- | :--- |
| **Operating System**| `manylinux_2_28_x86_64` | Wheels are built for broad Linux compatibility. Windows/macOS are out-of-scope for v1.0. |
| **Python** | `3.10` | The primary development and testing environment. |
| **PyTorch** | `2.1.2+cu118` | Must be the CUDA 11.8 variant. Incompatible PyTorch builds will fail. |
| **CUDA Toolkit** | `11.8` | The host system's CUDA Toolkit must match the PyTorch build. |
| **C++ Compiler** | `GCC 9.4.0+` | `std=c++17` is required. |
| **Pybind11** | `2.10.0+` | Required for C++ bindings. |

---

## 2. Supported GPU Architectures

The compiled binaries (`.whl`) are built with support for the following NVIDIA GPU Compute Capabilities (SM Architectures).

| Architecture Name | Compute Capability (SM) | Target GPUs (Examples) |
| :--- | :--- | :--- |
| **Turing** | `sm_75` | NVIDIA T4, RTX 20-series |
| **Ampere** | `sm_80`, `sm_86` | NVIDIA A100, RTX 30-series |
| **Ada Lovelace** | `sm_90` | RTX 40-series |

**Note:** While the engine may run on older architectures via PTX JIT compilation, official performance guarantees and support are limited to the architectures listed above.

---

## 3. Out-of-Scope for v1.0.0

The following are explicitly **not supported** in the v1.0.0 release:

*   **Operating Systems:** Windows, macOS.
*   **Hardware:** AMD GPUs, Intel GPUs, Apple Silicon (Metal).
*   **Mixed-Precision:** `float16` and `bfloat16` data types are not supported. All inputs must be `float32`.
*   **CPU Execution:** The C++ extension is compiled for CUDA only. There is no CPU fallback path.
*   **Gradient Computation:** This module (`Helios.Embed`) does not support `torch.autograd`. It is for inference and forward-pass operations only.

---
# --- END OF FILE HELIOS_EMBED/SUPPORT_MATRIX.md ---