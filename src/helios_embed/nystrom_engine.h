// --- START OF FILE src/helios_embed/nystrom_engine.h (FINAL, STABLE v3.1.0) ---
#pragma once
#include <torch/extension.h>

void validate_inputs_stateless(const at::Tensor& X, const at::Tensor& Lm);

torch::Tensor compute_rkhs_embedding(
    const torch::Tensor& X,
    const torch::Tensor& landmarks,
    float gamma,
    float ridge);
// --- END OF FILE src/helios_embed/nystrom_engine.h (FINAL, STABLE v3.1.0) ---