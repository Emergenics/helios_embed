// --- START OF FILE src/helios_embed/hybrid_kernel.h ---
#pragma once
#include <torch/extension.h>

/**
 * @brief Computes the RBF kernel matrix K(X, Y) using a hybrid approach.
 * It leverages the hyper-optimized cuBLAS for the core GEMM (X @ Y.T) and
 * then uses a custom fused kernel for the remaining element-wise operations.
 *
 * @param X The first input tensor of shape [N, D].
 * @param Y The second input tensor of shape [m, D].
 * @param gamma The RBF kernel's bandwidth parameter.
 * @return A torch::Tensor of shape [N, m] containing the kernel matrix.
 */
torch::Tensor rbf_kernel_hybrid_cuda(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    float gamma);
// --- END OF FILE src/helios_embed/hybrid_kernel.h ---