// --- START OF FILE src/helios_embed/common.h ---
#pragma once

// This macro will be defined by setup.py during a CPU-only build.
// It allows us to use preprocessor directives to include/exclude code paths.
#ifdef HELIOS_CPU_BUILD
// For CPU builds, many CUDA-specific types are not available.
// We provide stub definitions or use CPU-compatible PyTorch headers.
#include <torch/extension.h>
namespace helios_api = torch;

// This macro is a device-agnostic way to throw a runtime error when
// a function that explicitly requires CUDA is called in a CPU-only build.
#define HELIOS_CUDA_ONLY_FUNC(...)                                             \
  TORCH_CHECK(false, "Helios.Embed was compiled in CPU-only mode. This "       \
                     "function requires a CUDA device.");                      \
  return torch::Tensor();

#else
// For CUDA builds, we include the full CUDA-specific ATen library.
#include <ATen/ATen.h>
namespace helios_api = at;

// In a CUDA build, this macro does nothing, allowing the function to execute.
#define HELIOS_CUDA_ONLY_FUNC(...)

#endif
// --- END OF FILE src/helios_embed/common.h ---