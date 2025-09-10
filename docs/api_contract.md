
# Helios.Embed v1.0.0 - API Contract

This document provides a formal and stable contract for the public-facing API of the `Helios.Embed` module. All functions and classes documented here are considered part of the stable API and will adhere to Semantic Versioning (SemVer) principles.

---

## 1. Python Module: `helios_embed._core`

The core C++/CUDA functionalities are exposed through a compiled Python module accessible at `helios_embed._core`.

### 1.1. Stateless Function: `compute_rkhs_embedding`

This is the primary, stateless function for performing the Nyström feature embedding.

**Signature:**
```python
compute_rkhs_embedding(
    X: torch.Tensor,
    landmarks: torch.Tensor,
    gamma: float,
    ridge: float
) -> torch.Tensor
```

**Parameters:**

| Name | Type | Shape | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `X` | `torch.Tensor` | `[N, D]` | `float32`, `cuda`, `contiguous` | The input data tensor containing `N` vectors of dimension `D`. |
| `landmarks` | `torch.Tensor` | `[m, D]` | `float32`, `cuda`, `contiguous` | The `m` landmark vectors of dimension `D` that form the basis of the RKHS. |
| `gamma` | `float` | Scalar | `> 0` | The bandwidth parameter for the Radial Basis Function (RBF) kernel. |
| `ridge` | `float` | Scalar | `>= 0` | A small regularization term to ensure the numerical stability of the kernel matrix inversion. |

**Returns:**

*   **`torch.Tensor`**: A new tensor of shape `[N, m]` and dtype `float32` on the same CUDA device, representing the Nyström feature embedding `Φ(X)`.

**Error Conditions (Raises `RuntimeError`):**
*   If any input tensor is not a `float32` CUDA tensor.
*   If `X` or `landmarks` are not 2-dimensional.
*   If the feature dimension `D` does not match between `X` and `landmarks`.
*   If `landmarks` has zero rows (`m=0`).
*   If `gamma <= 0` or `ridge < 0`.
*   If any input tensor contains non-finite values (`NaN` or `Inf`).

---

### 1.2. Stateful Class: `IncrementalNystromEngine`

This class provides an efficient, stateful interface for streaming workloads where new data is added incrementally.

**Signature:**
```python
class IncrementalNystromEngine:
    def __init__(self, landmarks: torch.Tensor, gamma: float, ridge: float):
        ...
    
    def build(self, X: torch.Tensor) -> torch.Tensor:
        ...
        
    def update(self, X_new: torch.Tensor, Phi_old: torch.Tensor) -> torch.Tensor:
        ...
```

#### `__init__(self, landmarks, gamma, ridge)`
*   **Description:** Constructor. Initializes the engine and pre-computes the expensive `(K_LL + λI)^(-1/2)` matrix, which is then cached for all subsequent operations.
*   **Parameters:** Same as the `landmarks`, `gamma`, and `ridge` parameters for the stateless function.
*   **Error Conditions:** Throws `RuntimeError` under the same conditions as the `landmarks` validation in the stateless function.

#### `build(self, X)`
*   **Description:** Computes the initial, full feature matrix for a starting dataset `X`. This method internally calls the stateless `compute_rkhs_embedding` function.
*   **Parameters:**
    *   `X`: The initial data tensor of shape `[N_initial, D]`.
*   **Returns:** A new tensor `Φ` of shape `[N_initial, m]`.

#### `update(self, X_new, Phi_old)`
*   **Description:** Efficiently computes the Nyström features for a new batch of data (`X_new`) and concatenates them to a previously computed feature matrix (`Phi_old`). This avoids recomputing features for the old data.
*   **Parameters:**
    *   `X_new`: The new data tensor of shape `[N_update, D]`.
    *   `Phi_old`: The existing feature matrix of shape `[N_old, m]`.
*   **Returns:** A new, combined feature matrix of shape `[N_old + N_update, m]`.
*   **Error Conditions:** Throws `RuntimeError` if the feature dimension `m` of `Phi_old` does not match the number of landmarks the engine was initialized with.
