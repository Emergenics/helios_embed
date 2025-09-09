// --- START OF FILE src/helios_embed/hybrid_kernel.cu ---
#include "hybrid_kernel.h"
#include <ATen/ATen.h>
#include <cuda_runtime.h>

// This kernel is much simpler. It's a fast, element-wise "finishing" kernel.
// It takes the result of the big GEMM and the vector norms and does the final calculation.
__global__ void rbf_finish_kernel(
    const float* __restrict__ XY_T,      // The result of X @ Y.T, shape [N, m]
    const float* __restrict__ X_norms_sq, // Squared norms of X, shape [N]
    const float* __restrict__ Y_norms_sq, // Squared norms of Y, shape [m]
    float* __restrict__ K_out,           // The output kernel matrix, shape [N, m]
    int N, int m, float neg_gamma)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // global row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // global col

    if (i < N && j < m) {
        // ||x-y||^2 = ||x||^2 + ||y||^2 - 2*<x,y>
        float dist_sq = X_norms_sq[i] + Y_norms_sq[j] - 2.0f * XY_T[i * m + j];
        K_out[i * m + j] = expf(neg_gamma * dist_sq);
    }
}

torch::Tensor rbf_kernel_hybrid_cuda(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    float gamma)
{
    TORCH_CHECK(X.is_cuda() && Y.is_cuda(), "Inputs must be CUDA tensors.");
    TORCH_CHECK(X.dim() == 2 && Y.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(X.size(1) == Y.size(1), "Inner dimensions (D) must match.");

    const int N = X.size(0);
    const int m = Y.size(0);

    // Step 1: Let the hyper-optimized cuBLAS do the heavy lifting (X @ Y.T)
    auto XY_T = at::matmul(X, Y.t());

    // Step 2: Compute the squared norms of the vectors.
    // This is a fast reduction operation.
    auto X_norms_sq = at::sum(X * X, 1);
    auto Y_norms_sq = at::sum(Y * Y, 1);

    // Step 3: Call our simple, fused kernel to do the final element-wise work.
    auto K_out = torch::empty({N, m}, X.options());

    dim3 threads(16, 16);
    dim3 blocks((m + 15) / 16, (N + 15) / 16);

    rbf_finish_kernel<<<blocks, threads>>>(
        XY_T.contiguous().data_ptr<float>(),
        X_norms_sq.contiguous().data_ptr<float>(),
        Y_norms_sq.contiguous().data_ptr<float>(),
        K_out.data_ptr<float>(),
        N, m, -gamma);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Hybrid RBF Kernel Launch Error: ") + cudaGetErrorString(err));
    }

    return K_out;
}
// --- END OF FILE src/helios_embed/hybrid_kernel.cu ---