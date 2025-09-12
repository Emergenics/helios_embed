// --- START OF FILE src/helios_embed/nystrom_engine_cpu.cpp (Formatted) ---
#include "nystrom_engine.h"

#include <torch/extension.h>

#include <stdexcept>

// This is a CPU-only stub implementation.
// It allows the project to be compiled and imported in a CPU environment,
// but will raise an error if the CUDA-dependent functions are actually called.

void validate_inputs_stateless(const at::Tensor &X, const at::Tensor &Lm) {
  // This validation can still run on CPU tensors.
  TORCH_CHECK(X.dim() == 2 && Lm.dim() == 2,
              "Input tensors must be 2-dimensional.");
  // Add other non-device-specific checks here if desired...
}

torch::Tensor compute_rkhs_embedding(const torch::Tensor &X,
                                     const torch::Tensor &landmarks,
                                     float gamma, float ridge) {
  TORCH_CHECK(false, "Helios.Embed was compiled in CPU-only mode. The "
                     "'compute_rkhs_embedding' "
                     "function requires CUDA and is not available.");
  // Return an empty tensor to satisfy the function signature.
  return torch::Tensor();
}
// --- END OF FILE src/helios_embed/nystrom_engine_cpu.cpp (Formatted) ---