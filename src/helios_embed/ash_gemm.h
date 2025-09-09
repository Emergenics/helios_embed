// --- START OF FILE src/helios_embed/ash_gemm.h ---
#pragma once
#include <torch/extension.h>

/**
 * @brief Performs a General Matrix Multiplication (GEMM) for bipolar (±1) matrices
 * using the "Difference-as-Compute" paradigm (XNOR+POPCOUNT).
 *
 * @param A_packed The first input matrix A, with rows bit-packed into uint64_t. Shape: [M, K_packed].
 * @param B_packed_T The second input matrix B, transposed and with rows bit-packed. Shape: [N, K_packed].
 * @param K The original inner dimension of the matrices.
 * @return A torch::Tensor of shape [M, N] with the integer result of the GEMM.
 */
torch::Tensor gemm_ash_algebra_cuda(
    const torch::Tensor& A_packed,
    const torch::Tensor& B_packed_T,
    int K);
// --- END OF FILE src/helios_embed/ash_gemm.h ---