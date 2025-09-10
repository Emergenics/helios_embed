# --- START OF FILE docs/design.md ---
# Helios.Embed v1.0.0 - Design & Architecture

This document outlines the core design principles, architectural decisions, and implementation rationale for the `Helios.Embed` (Nyström Feature Engine) module.

---

## 1. Core Philosophy: Principled Performance

The design of `Helios.Embed` is governed by a single philosophy: to provide a high-performance, production-ready implementation of a mathematically principled feature embedding. Every architectural choice is a balance between:

*   **Correctness:** The engine must be bit-perfectly accurate relative to a high-precision reference.
*   **Performance:** The engine must be significantly faster than naive implementations and scale predictably.
*   **Robustness:** The engine must be hardened against invalid inputs and fail gracefully with clear errors.

This philosophy is inspired by the **Wavelet Operator Theory (WOT)** framework, which reframes machine learning as "operator estimation in a Hilbert space." `Helios.Embed` serves as the foundational **"portal"** into this space, providing the high-quality feature map `Φ` upon which all higher-level reasoning and acceleration operators depend.

## 2. Architectural Components

The module is composed of two primary, user-facing interfaces built upon a unified C++/CUDA backend.

### 2.1. The Stateless Engine (`compute_rkhs_embedding`)

*   **Purpose:** To provide a simple, functional, and highly optimized entry point for one-shot feature embedding. It has no memory of past computations.
*   **Design Choice:** The implementation is a single C++ function (`compute_rkhs_embedding`) exposed to Python via Pybind11. This minimizes overhead and provides a clean, stateless API.

### 2.2. The Stateful Engine (`IncrementalNystromEngine`)

*   **Purpose:** To provide a highly efficient path for streaming workloads where data arrives incrementally.
*   **Design Choice:** Implemented as a C++ class that holds state. The key design decision is the **pre-computation and caching of the `K_mm_inv_sqrt` matrix** in the constructor. This is the most expensive part of the Nyström calculation. By caching it, the subsequent `update()` calls only need to compute the `K_nm` matrix for the *new* data, which is a much cheaper operation. This design directly implements the **ITMD (Input-Transform-Measure-Difference)** principle of amortizing expensive initial computations over many cheap incremental updates.

## 3. Core Algorithm & Numerical Stability

### 3.1. Nyström Approximation vs. RFF

*   **Decision:** The engine exclusively uses the **Nyström method** for kernel approximation.
*   **Rationale:** In a rigorous "Science Shredder" process, the Nyström method was empirically proven to be more accurate and stable for our target workloads compared to the Random Fourier Features (RFF) method. The falsification of RFF and the validation of Nyström was a critical, data-driven architectural decision.

### 3.2. Matrix Inverse Square Root: `Eigh` over `Cholesky`

*   **Decision:** The numerically sensitive computation of `(K_LL + λI)^(-1/2)` is performed using **eigen-decomposition (`linalg.eigh`)**.
*   **Rationale:** Early prototypes revealed that the `linalg.cholesky` method, while faster, was numerically unstable for the ill-conditioned (near-singular) `K_LL` matrices that can arise in practice. The `eigh` path is more computationally expensive but provides superior numerical stability and robustness, which is non-negotiable under our "Accuracy Above All" principle. This choice was validated after a series of correctness test failures, proving its necessity.

## 4. Security and Robustness: "Hardening by Design"

*   **Principle:** All inputs from the Python world are considered untrusted.
*   **Implementation:**
    *   **Input Validation:** Every public C++ function begins with a call to a comprehensive `validate_inputs` function. This function uses a series of `TORCH_CHECK` macros to enforce strict constraints on tensor device, dtype, shape, and data integrity (`NaN`/`Inf`). This ensures the engine fails fast with clear, informative errors before any computation begins.
    *   **Memory Contiguity:** All input tensors are immediately converted to a `.contiguous()` memory layout upon entering the C++ boundary. This prevents CUDA errors and simplifies kernel logic by guaranteeing a predictable memory stride, a lesson learned from our hardening benchmarks.
    *   **Error Mapping:** All C++ exceptions thrown by `TORCH_CHECK` are automatically and cleanly mapped to Python `RuntimeError` exceptions by the Pybind11/PyTorch C++ extension framework, providing a seamless and predictable error-handling experience for the Python user.

# --- END OF FILE docs/design.md ---