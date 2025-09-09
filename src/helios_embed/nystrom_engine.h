// --- START OF FILE nystrom_engine/nystrom_engine.h (FINAL, Stateless) ---
#pragma once
#include <torch/extension.h>

// This is our new, more robust validation function.
void validate_inputs_stateless(const at::Tensor& X, const at::Tensor& Lm);

// Forward declaration for the stateless Nystrom feature embedding function.
torch::Tensor compute_rkhs_embedding_nystrom(
    const torch::Tensor& X,
    const torch::Tensor& landmarks,
    float gamma,
    float ridge);

// Forward declaration for the new stateless reasoning operator function.
torch::Tensor apply_linear_operator_cuda(
    const torch::Tensor& operator_matrix,
    const torch::Tensor& input_features);
// --- END OF FILE nystrom_engine/nystrom_engine.h (FINAL, Stateless) ---