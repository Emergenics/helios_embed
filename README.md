# Helios.Embed: A Production-Ready Nyström Feature Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.1.2+cu118-orange.svg)](https://pytorch.org/)
[![CUDA Version](https://img.shields.io/badge/cuda-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**`Helios.Embed` is a high-performance, production-ready C++/CUDA engine for creating Nyström feature embeddings in PyTorch.** It serves as the foundational "portal" for lifting data into a high-dimensional Hilbert space, a critical first step for advanced, operator-based AI as described in the **Wavelet Operator Theory (WOT)** framework.

This module is designed for performance, stability, and correctness, providing a bit-perfectly accurate implementation that is significantly faster than naive approaches for streaming workloads.

---

## 🏛️ Core Philosophy

The design of `Helios.Embed` is governed by three non-negotiable principles:

1.  **Accuracy Above All:** The engine's output is guaranteed to be bit-perfectly accurate against a high-precision CPU reference, meeting a strict `Relative MSE <= 1e-7` standard for `float32`.
2.  **Principled Performance:** Speedups are achieved through principled architectural design (caching, incremental updates) and low-level optimization, not by sacrificing accuracy.
3.  **Production-Ready Robustness:** The engine is security-hardened with comprehensive input validation and error handling to ensure stability in real-world applications.

## ✨ Key Features

*   **High-Performance C++/CUDA Backend:** Core logic is written in C++ and CUDA to eliminate Python overhead and maximize GPU utilization.
*   **Stateless and Stateful APIs:**
    *   `compute_rkhs_embedding()`: A simple, one-shot function for batch processing.
    *   `IncrementalNystromEngine`: A stateful class that provides massive speedups for streaming data by caching expensive computations.
*   **Bit-Perfect Accuracy:** Validated to be numerically identical to a `float64` reference implementation.
*   **Security Hardened:** Robust input validation prevents crashes from common data issues like non-contiguous memory, `NaN`/`Inf` values, or mismatched dimensions.
*   **Professional, Modular Design:** Built with a clean `src/` layout and a dedicated build system for easy integration and maintenance.

## 🚀 Getting Started

This guide will walk you through compiling and using `Helios.Embed`.

### 1. Prerequisites

Ensure your environment meets the specifications in our official [Support Matrix](docs/SUPPORT_MATRIX.md). Key requirements are:
*   **OS:** Linux
*   **Python:** 3.10
*   **CUDA Toolkit:** 11.8
*   **PyTorch:** 2.1.2 (the `+cu118` variant)

### 2. Compilation

Clone the repository and run the following command from the project root. This will compile the C++/CUDA extension and install the package in editable mode.

```bash
# From the root of the HELIOS_EMBED directory
python setup.py build_ext --inplace
```

### 3. Usage Example

The following example demonstrates both the stateless and stateful APIs. You can find this code in `examples/getting_started.py`.

```python
import torch
from helios_embed._core import compute_rkhs_embedding, IncrementalNystromEngine

# Ensure you are on a CUDA device
device = torch.device("cuda")

# Sample data
N_initial, N_update, D, m = 1024, 128, 384, 128
X_initial = torch.randn(N_initial, D, device=device)
X_update = torch.randn(N_update, D, device=device)
landmarks = X_initial[torch.randperm(N_initial, device=device)[:m]]
gamma = 0.1
ridge = 1e-6

# --- Example 1: Stateless one-shot computation ---
print("Running stateless example...")
features_stateless = compute_rkhs_embedding(X_initial, landmarks, gamma, ridge)
print(f"Stateless output shape: {features_stateless.shape}")

# --- Example 2: Stateful streaming computation ---
print("\nRunning stateful streaming example...")
# Initialize the engine (caches the expensive part)
streaming_engine = IncrementalNystromEngine(landmarks, gamma, ridge)
# Process the initial batch
features_old = streaming_engine.build(X_initial)
# Process a new, incoming batch of data (this is the fast part)
features_new = streaming_engine.update(X_update, features_old)
print(f"Final streaming output shape: {features_new.shape}")

# Verify correctness against a full re-computation
X_combined = torch.cat([X_initial, X_update], dim=0)
features_ground_truth = compute_rkhs_embedding(X_combined, landmarks, gamma, ridge)
rel_mse = torch.mean((features_ground_truth - features_new)**2) / torch.mean(features_ground_truth**2)
print(f"Relative MSE between streaming and ground truth: {rel_mse.item():.2e}")
assert rel_mse.item() <= 1e-7, "Accuracy test failed!"
print("✅ Success! The streaming engine is bit-perfectly accurate.")
```

### 4. Developer Notes
# Dependencies required for running the Helios.Embed test and benchmark suite.

# Core runtime dependency (for clarity)
torch>=2.1.2,<2.2

# Test suite dependencies
numpy
pandas
matplotlib

# Optional, but recommended for development
pybind11
ninja
# --- END OF FILE HELIOS_EMBED/requirements-test.txt ---
```

#### **Action 2: Update the `README.md` with Development Setup Instructions**

We need to add a "Developer Setup" section to our `README.md` to instruct contributors on how to correctly set up their environment for testing.

**File to Modify:** `HELIOS_EMBED/README.md`

*   **Action:** Add the following new section to the end of your `README.md` file.

```markdown
# --- ADD THIS SECTION TO README.md ---

## 🛠️ For Developers: Setting Up a Test Environment

To run the full test and benchmark suite, you need to install the test-time dependencies in addition to the core package.

**1. Create a clean environment:**
```bash
python3.10 -m venv .venv-dev
source .venv-dev/bin/activate
```

**2. Install PyTorch:**
```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

**3. Install Test Dependencies:**
```bash
pip install -r requirements-test.txt
```

**4. Build the Extension:**
```bash
python setup.py build_ext --inplace
```

**5. Run the Full Test Suite:**
```bash
python run_tests.py
```


## 📚 Full Documentation

For more detailed information, please see our full documentation website, which includes:

*   **[API Reference](docs/api_contract.md):** Formal contract for all public functions and classes.
*   **[Design & Architecture](docs/design.md):** The "why" behind our engineering choices.
*   **[Performance & Scalability](docs/perf_baseline.md):** Definitive benchmark results and scaling analysis.
*   **[Numerical Accuracy](docs/numerical_tolerances.md):** Our strict standards for bit-perfect correctness.
*   **[Concurrency Model](docs/threading_model.md):** Guarantees for using the engine in multi-threaded environments.

---








