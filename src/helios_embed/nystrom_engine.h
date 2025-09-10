// --- START OF FILE src/helios_embed/nystrom_engine.h (Version 2.1.0 - FINAL) ---
#pragma once
#include <torch/extension.h>

// This is our single, robust validation function.
void validate_inputs_stateless(const at::Tensor& X, const at::Tensor& Lm);

// This is the single, production-ready entry point.
torch::Tensor compute_rkhs_embedding_nystrom(
    const torch::Tensor& X,
    const torch::Tensor& landmarks,
    float gamma,
    float ridge);
// --- END OF FILE src/helios_embed/nystrom_engine.h (Version 2.1.0 - FINAL) ---