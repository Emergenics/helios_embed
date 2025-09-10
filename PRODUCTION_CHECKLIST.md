# Governance & Release Hygiene

* [ ] **SemVer policy** documented; public Python/C++ symbols mapped to versioning rules (API vs ABI).
* [ ] **Release checklist** with sign-offs (owner, security, perf, docs).
* [ ] **CHANGELOG** (keep breaking changes explicit and migration steps concrete).
* [ ] **CODEOWNERS** + PR review requirements; protected main branch.
* [ ] **Licensing**: repo license file, per-file headers, third-party notices.
* [ ] **Support matrix**: OS (manylinux/macos/win), CUDA versions, PyTorch versions, GPU arch (sm\_70+), Python versions.

# Supply Chain & Build Integrity

* [ ] **Reproducible builds**: pinned toolchains (CMake & compilers), Dockerfile for wheels.
* [ ] **SBOM** generated (SPDX/CycloneDX) and attached to releases.
* [ ] **Artifact signing** (e.g., cosign/sigstore) for wheels/tarballs.
* [ ] **Dependency pinning**: exact versions for build-time deps; hash-pinned downloads.
* [ ] **No vendored binaries**; if vendored code, track upstream commit hashes.
* [ ] **CI isolation**: minimal privileges, no secrets in PRs from forks.

# Security (Product & Process)

* [ ] **Threat model** (STRIDE) for inputs (tensors) and environment (GPU/driver).
* [ ] **Inputs are untrusted**: enforce dtype/device/layout/shape/range before kernels.
* [ ] **Secrets policy**: prevent secrets in logs; scanning pre-commit & CI (e.g., gitleaks).
* [ ] **Sandbox guidance** for users (container images with fixed CUDA/cuDNN).
* [ ] **Safe defaults** (no telemetry by default; opt-in with explicit env var).
* [ ] **Vulnerability handling policy** (security.md + contact channel).
* [ ] **CVE scanning** for base images and deps in CI.
* [ ] **DoS surfaces** documented (e.g., worst-case O(N·L) paths) + limits.

# API Contracts (Python & C++)

* [ ] **Stable signatures** documented with types, units, constraints.
* [ ] **Docstrings** for all bindings (✅ you added); `help()` shows parameters & errors.
* [ ] **Error mapping**: C++ exceptions → precise Python exceptions (ValueError, RuntimeError).
* [ ] **Determinism switch**: env or flag; document ops that may be nondeterministic.
* [ ] **Backward-compat deprecations**: warnings include removal version & alternative.

# Input Validation (front-doored, fast-fail)

* [ ] **Tensor checks**: device (cpu/cuda), dtype (float32/64), layout (contiguous), shape (N×D vs L×D), gradient requirements if applicable.
* [ ] **Parameter checks**: `gamma > 0`, `ridge ≥ 0`, bounds for batch sizes.
* [ ] **Memory size checks** for allocations (guard against overflow: size\_t math).
* [ ] **Clear messages**: include actual values in thrown errors.
* [ ] **Optional autocast** handling: normalize dtype internally (e.g., promote half→float).

# Error Handling & Robustness

* [ ] **Every CUDA call checked** (`cudaGetLastError`, `AT_CUDA_CHECK` or equivalent).
* [ ] **Guard devices/streams**: use `at::cuda::CUDAGuard` and `at::cuda::CUDAStreamGuard`.
* [ ] **No implicit syncs**: avoid `cudaDeviceSynchronize()` in hot paths; document sync points.
* [ ] **Graceful fallback**: clear CPU fallback path or explicit failure with guidance.
* [ ] **Resource safety**: RAII for temporaries; `noexcept` where possible; avoid raw `new/delete`.
* [ ] **Timeout/iteration caps** in iterative updates; detect & abort diverging states.

# Concurrency & Parallelism

* [ ] **Thread-safety policy** documented: reentrancy of functions, shared state handling.
* [ ] **Streams**: allow caller-provided stream or respect current stream; no hidden streams unless documented.
* [ ] **Host parallelism**: OpenMP/TBB usage consistent with PyTorch’s threadpool (avoid oversubscription).
* [ ] **Locking**: minimal critical sections; avoid device-wide locks.

# Memory Safety & Performance Hygiene

* [ ] **Address/UB sanitizers** in CPU builds (ASan/UBSan) for tests.
* [ ] **Compute Sanitizer** (`compute-sanitizer` / `cuda-memcheck`) runs in CI on small cases.
* [ ] **Valgrind (CPU path)** smoke tests where relevant.
* [ ] **Alignment**: aligned loads/stores in kernels; consider `__ldg`/LDG on older archs.
* [ ] **Coalescing**: structure of arrays or appropriate tiling to ensure coalesced memory.
* [ ] **Shared memory** use bounded & checked; `extern __shared__` sizes validated.
* [ ] **Bank conflicts** analysis done (Nsight Compute metrics).
* [ ] **Occupancy**: kernel launch params validated; `__launch_bounds__` if justified.
* [ ] **Register pressure** checked; no unexpected spills on target SMs.
* [ ] **Mixed precision**: numerics validated (loss of significance) + accumulation strategy (FP32 accum for FP16).

# Numerical Correctness & Stability

* [ ] **Reference impl** (slow/CPU or high-precision) for cross-checking.
* [ ] **Property-based tests** (e.g., positive-definiteness with ridge, invariances).
* [ ] **Conditioning**: ridge behavior tested on near-singular Gram matrices.
* [ ] **Batching effects**: identical results for different batch partitions within tolerance.
* [ ] **Deterministic seeds** and tolerance bands documented.
* [ ] **NaN/Inf handling**: detect early; propagate with clear error or sanitize.

# Testing Matrix

* [ ] **Unit tests**: kernel small cases, edge cases (N=0, L=1, extreme gamma/ridge).
* [ ] **Integration tests**: full pipeline from Python to CUDA and back.
* [ ] **Golden tests**: saved expected outputs for canonical inputs per CUDA/PyTorch version.
* [ ] **Fuzz tests**: libFuzzer/pyfuzzer to mutate shapes/values; crash & OOM resistant.
* [ ] **Stress tests**: high-N/L boundary with memory pressure.
* [ ] **Compatibility**: test grid across Python, PyTorch, CUDA, GPU arch.
* [ ] **Performance guards**: regression thresholds (fail build if >X% slower on metrics).

# CI/CD

* [ ] **Matrix builds** (Linux/macOS/Windows, multiple CUDA/PyTorch).
* [ ] **Unit & perf tests** gated on CI; perf run allowed on smaller representative GPU.
* [ ] **Sanitizer jobs** (ASan/UBSan CPU; Compute Sanitizer CUDA).
* [ ] **Wheel builds**: manylinux + macOS + Windows; tagged by CUDA version.
* [ ] **Example scripts** run in CI (import, run, smoke verification).
* [ ] **Cache** compilers & third-party deps with checksums to ensure integrity.

# Observability (Prod & Dev)

* [ ] **Configurable logging**: upstream to Python logger; levels (ERROR/WARN/INFO).
* [ ] **Metrics hooks** (optional): op latency, kernel time, bytes moved, occupancy.
* [ ] **Tracing**: NVTX ranges around kernels; integrates with Nsight Systems.
* [ ] **Error codes**: include CUDA error strings; surface to Python cleanly.
* [ ] **Minimal PII**: ensure no data content lands in logs/metrics by default.

# Documentation & Examples

* [ ] **API reference** auto-generated (Doxygen → Breathe → Sphinx or pdoc for Python).
* [ ] **Design doc**: architecture, invariants, kernel math, memory layout, threading model.
* [ ] **Ops guide**: install, compatible drivers, troubleshooting (common CUDA errors).
* [ ] **Security doc**: threat model, data handling, hardening tips.
* [ ] **Performance guide**: recommended batch sizes, streams, mixed precision caveats.
* [ ] **End-to-end example notebooks** (CPU & CUDA, small reproducible datasets).
* [ ] **Versioned docs** matching releases.

# Packaging & Distribution

* [ ] **PyTorch extension** setup using `pytorch/torch.utils.cpp_extension` or PEP 517 build backend.
* [ ] **Fatbins**: include PTX + SASS for target SMs or rely on PTX JIT policy (documented tradeoff).
* [ ] **Wheel tags** reflect CUDA runtime (`+cu12x`) vs CPU; avoid ambiguous “none” tags.
* [ ] **Import-time checks**: helpful errors if wrong CUDA/PyTorch versions.
* [ ] **Lazy initialization** to reduce import cost.

# Backward/Forward Compatibility

* [ ] **ABI isolation**: avoid exposing STL types in public C++ headers if distributing binaries.
* [ ] **Serialization stability** if any state is saved; include versioned metadata.
* [ ] **Feature flags** for experimental behavior toggled off by default.

# Operational Resilience

* [ ] **Graceful OOM handling**: catch, free temporaries, retry smaller batch suggestion.
* [ ] **Cancellation**: respect Python/ATen cancellation if applicable.
* [ ] **Time limits**: watchdogs for unusually long kernels (documented).
* [ ] **Fallback strategy** documented (CPU path, smaller tiles, lower rank).

# CUDA-Specific Hardening

* [ ] **Device capability checks** at runtime (tensor cores, shared mem per block).
* [ ] **Pinned memory** for H2D/D2H where appropriate; async copies with streams.
* [ ] **No host-side UB**: kernel launches guarded by bounds; index math uses 64-bit where needed.
* [ ] **Bank-/warp-safe** reductions; avoid atomics unless necessary; if atomics, define order assumptions.
* [ ] **Kernel parameter validation**: `gridDim/blockDim` computed safely; prevent integer overflow.

# Tooling (recommended commands)

* [ ] **clang-format** enforced via pre-commit.
* [ ] **clang-tidy** checks: modernize/readability/performance/google-\* profiles with a suppressions file.
* [ ] **include-what-you-use** to trim headers.
* [ ] **Sanitizers**:

  * CPU: `-fsanitize=address,undefined -fno-omit-frame-pointer` for tests.
  * CUDA: `compute-sanitizer --tool memcheck|racecheck|initcheck` in CI.
* [ ] **Profilers**: Nsight Systems (timeline), Nsight Compute (kernel metrics) with saved baselines.

# Data Policy & Privacy

* [ ] **No data retention** by default; temporary buffers zeroed only if policy requires.
* [ ] **User consent** for any metrics/telemetry; disable by default.
* [ ] **Dataset examples** are synthetic or approved for redistribution.

---

## Minimal “Definition of Done” (ship gate)

1. All unit/integration tests pass across support matrix.
2. Compute Sanitizer clean on smoke tests.
3. No perf regressions >X% vs last release on standard benchmarks.
4. Wheel build artifacts signed + SBOM attached.
5. Docs published and versioned; CHANGELOG updated.
6. Security scan clean; dependency pins updated; base images patched.
7. Support matrix verified on real hardware (at least one GPU per major SM group).

---

## Templates you can drop in

### C++/CUDA error macro (maps cleanly to Python)

```cpp
#define HELIOS_CUDA_CHECK(stmt)                               \
  do {                                                        \
    cudaError_t err__ = (stmt);                               \
    if (err__ != cudaSuccess) {                               \
      throw std::runtime_error(std::string("CUDA error: ") +  \
        cudaGetErrorString(err__) + " at " + __FILE__ + ":" + \
        std::to_string(__LINE__));                            \
    }                                                         \
  } while (0)
```

### Tensor validation (PyTorch C++ API)

```cpp
static inline void check_tensor(const at::Tensor& t, c10::DeviceType dev,
                                at::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  TORCH_CHECK(t.device().type() == dev, name, " must be on ", dev);
  TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype);
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == 2, name, " must be rank-2 (N x D)");
}
```

### Stream/device guards (inside binding-exposed functions)

```cpp
at::cuda::CUDAGuard device_guard(X.device());
at::cuda::CUDAStreamGuard stream_guard(at::cuda::getCurrentCUDAStream());
```

---