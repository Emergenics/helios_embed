// --- START OF FILE src/helios_embed/nystrom_engine_cpu.cpp ---
#include "nystrom_engine.h"
#include <torch/extension.h>

// This is a CPU-only stub implementation. It allows the project to be compiled
// and imported in a CPU environment, but will raise an error if the functions
// are actually called, as they require CUDA.

void validate_inputs_stateless(const at::Tensor &X, const at::Tensor &Lm) {
  // Basic checks can still run on CPU tensors.
  TORCH_CHECK(X.dim() == 2 && Lm.dim() == 2,
              "Input tensors must be 2-dimensional.");
  // Add other non-device-specific checks if desired...
}

torch::Tensor compute_rkhs_embedding(const torch::Tensor &X,
                                     const torch::Tensor &landmarks,
                                     float gamma, float ridge) {
  TORCH_CHECK(
      false,
      "Helios.Embed was compiled in CPU-only mode. The "
      "'compute_rkhs_embedding' function requires CUDA and is not available.");
  return torch::Tensor();
}
// --- END OF FILE src/helios_embed/nystrom_engine_cpu.cpp ---