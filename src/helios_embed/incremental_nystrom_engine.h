// --- START OF FILE src/helios_embed/incremental_nystrom_engine.h (FINAL,
// STABLE v1.1.0) ---
#pragma once
#include "nystrom_engine.h" // For the stateless function declaration
#include <torch/extension.h>

class IncrementalNystromEngine {
public:
  IncrementalNystromEngine(torch::Tensor landmarks, float gamma, float ridge);
  torch::Tensor build(const torch::Tensor &X);
  torch::Tensor update(const torch::Tensor &X_new,
                       const torch::Tensor &Phi_old);

private:
  torch::Tensor landmarks_;
  float gamma_;
  float ridge_;
  torch::Tensor K_mm_inv_sqrt_; // Cached component
  void initialize_engine();
};
// --- END OF FILE src/helios_embed/incremental_nystrom_engine.h (FINAL, STABLE
// v1.1.0) ---