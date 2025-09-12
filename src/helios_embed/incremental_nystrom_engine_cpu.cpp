// --- START OF FILE src/helios_embed/incremental_nystrom_engine_cpu.cpp
// (Formatted) ---
#include "incremental_nystrom_engine.h"

#include <torch/extension.h>

#include <stdexcept>

// CPU-only stub implementations for the stateful engine class.

IncrementalNystromEngine::IncrementalNystromEngine(torch::Tensor landmarks,
                                                   float gamma, float ridge) {
  TORCH_CHECK(false, "Helios.Embed was compiled in CPU-only mode. The "
                     "'IncrementalNystromEngine' "
                     "class requires CUDA and is not available.");
}

// These methods must exist to satisfy the class definition, but they will never
// be reached because the constructor throws.
torch::Tensor IncrementalNystromEngine::build(const torch::Tensor &X) {
  return torch::Tensor();
}

torch::Tensor
IncrementalNystromEngine::update(const torch::Tensor &X_new_in,
                                 const torch::Tensor &Phi_old_in) {
  return torch::Tensor();
}
// --- END OF FILE src/helios_embed/incremental_nystrom_engine_cpu.cpp
// (Formatted) ---