// --- START OF FILE src/helios_embed/rbf_kernel.h ---
#pragma once
#include <torch/extension.h>

/**
 * @brief Computes the RBF kernel matrix K(X, Y) = exp(-gamma * ||x_i - y_j||^2)
 * using a custom, high-performance, fused CUDA kernel with shared memory tiling.
 * This function computes ONLY the K_nm matrix, not the full Nystrom feature set.
 *
 * @param X The first input tensor of shape [N, D].
 * @param Y The second input tensor of shape [m, D].
 * @param gamma The RBF kernel's bandwidth parameter.
 * @return A torch::Tensor of shape [N, m] containing the kernel matrix.
 */
torch::Tensor rbf_kernel_fused_cuda(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    float gamma);
// --- END OF FILE src/helios_embed/rbf_kernel.h ---