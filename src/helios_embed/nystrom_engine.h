// --- START OF FILE src/helios_embed/nystrom_engine.h (Version 1.2.0 - Final) ---
#pragma once
#include <torch/extension.h>

// Forward declaration from our new hybrid kernel file
torch::Tensor rbf_kernel_hybrid_cuda(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    float gamma);

// This is our robust validation function, now used by both stateless and stateful engines.
void validate_inputs_stateless(const at::Tensor& X, const at::Tensor& Lm);

/**
 * @brief Computes the Nystrom feature embedding Φ(X) for a given dataset X.
 * This is the high-performance, stateless entry point.
 * It now internally uses the hybrid kernel for a ~1.7x speedup.
 *
 * @param X The input data tensor of shape [N, D].
 * @param landmarks The landmark tensor of shape [m, D].
 * @param gamma The RBF kernel's bandwidth parameter.
 * @param ridge The regularization strength for the linear solve.
 * @return A torch::Tensor of shape [N, m] containing the Nystrom features.
 */
torch::Tensor compute_rkhs_embedding_nystrom(
    const torch::Tensor& X,
    const torch::Tensor& landmarks,
    float gamma,
    float ridge);

// Note: The apply_linear_operator_cuda is a concept for Helios.Reason and is not implemented here.
// --- END OF FILE src/helios_embed/nystrom_engine.h (Version 1.2.0 - Final) ---