// --- START OF FILE src/helios_embed/nystrom_engine.h (FINAL, STABLE v1.1.0) ---
#pragma once
#include <torch/extension.h>

/**
 * @brief Validates input tensors for the stateless Nystrom feature embedding function.
 * Throws informative C++ exceptions (mapped to Python exceptions) on failure.
 * This is the primary security gate for the stateless API.
 */
void validate_inputs_stateless(const at::Tensor& X, const at::Tensor& Lm);

/**
 * @brief Computes the Nystrom feature embedding (Φ) for a given dataset X.
 * This is the core, stateless, production-ready function of Helios.Embed.
 *
 * @param X The input data tensor of shape [N, D].
 * @param landmarks The landmark tensor of shape [m, D].
 * @param gamma The RBF kernel's bandwidth parameter. Must be > 0.
 * @param ridge The regularization parameter for the kernel matrix. Must be >= 0.
 * @return A torch::Tensor of shape [N, m] containing the Nystrom features.
 */
torch::Tensor compute_rkhs_embedding(
    const torch::Tensor& X,
    const torch::Tensor& landmarks,
    float gamma,
    float ridge);
// --- END OF FILE src/helios_embed/nystrom_engine.h (FINAL, STABLE v1.1.0) ---