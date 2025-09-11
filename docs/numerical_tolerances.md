<!-- MathJax configuration for MkDocs -->

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']]
  }
};
</script>

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

<!-- Make all headers yellow -->

<style>
  h1, h2, h3, h4, h5, h6 {
    color: yellow;
  }
</style>

# Helios.Embed v1.0.0 - Numerical Tolerances & Accuracy Standards

This document specifies the official numerical accuracy standards for the `Helios.Embed` module. All test suites and validation benchmarks must adhere to these tolerances.

---

## 1. Guiding Principle: "Accuracy Above All"

The foremost design principle of the Helios Engine is **"Accuracy Above All."** Performance optimizations are only considered valid if they do not compromise the numerical correctness of the underlying mathematical operations. The engine's output must be verifiably equivalent to a high-precision, trusted reference implementation.

## 2. The Reference Oracle

The official source of truth for numerical correctness is the high-precision CPU reference implementation located in `tests/helios_ref.py`. This reference implementation performs all calculations in **`float64` (double precision)** and uses numerically stable algorithms (`torch.linalg.eigh`) to provide a "golden" result.

## 3. The Official Accuracy Metric

Accuracy is measured using the **Relative Mean Squared Error (Relative MSE)** between the output of our `float32` CUDA engine and the `float64` CPU reference.

The formula is:
```
rel_mse = mean((output_cuda.double() - output_ref)**2) / (mean(output_ref**2) + epsilon)
```
where `epsilon` is a small constant (e.g., `1e-20`) to prevent division by zero.

## 4. Formal Tolerance Standards

The following tolerances are enforced by our automated test suite (`tests/test_correctness.py`). Any result outside these bounds constitutes a **FAILURE** and a breaking change.

| Data Type | Tolerance Standard (Max `Relative MSE`) | Definition |
| :--- | :--- | :--- |
| **`float32`** | `1e-7` | **Bit-Perfect for `float32`**. This is the standard for our primary CUDA engine. It accounts for minor differences in operation order and hardware-specific floating-point behavior between CPU and GPU, while being orders of magnitude stricter than typical machine learning tolerances. |
| **`float64`** | `1e-15` | **Bit-Perfect for `float64`**. This is the standard for our CPU reference and would be the standard for any future `float64` CUDA kernels. It is at the threshold of machine epsilon for double-precision numbers. |

**Rationale:**
By enforcing these extremely strict tolerances, we guarantee that our engine is not an approximation, but a high-performance, numerically identical implementation of the underlying mathematical theory.
