// --- START OF FILE src/helios_embed/hybrid_kernel.cu (Security Hardened v1.2.1) ---
#include "hybrid_kernel.h"
#include <ATen/ATen.h>
#include <cuda_runtime.h>

// This kernel is now hardened to use TensorAccessor for safe, bounds-checked access.
__global__ void rbf_finish_kernel_safe(
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> XY_T_acc,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> X_norms_sq_acc,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> Y_norms_sq_acc,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> K_out_acc,
    int N, int m, float neg_gamma)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // global row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // global col

    if (i < N && j < m) {
        float dist_sq = X_norms_sq_acc[i] + Y_norms_sq_acc[j] - 2.0f * XY_T_acc[i][j];
        K_out_acc[i][j] = expf(neg_gamma * dist_sq);
    }
}

torch::Tensor rbf_kernel_hybrid_cuda(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    float gamma)
{
    // --- SECURITY: Enforce contiguous memory layout on all inputs ---
    auto X_cont = X.contiguous();
    auto Y_cont = Y.contiguous();

    TORCH_CHECK(X_cont.is_cuda() && Y_cont.is_cuda(), "Inputs must be CUDA tensors.");
    TORCH_CHECK(X_cont.dim() == 2 && Y_cont.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(X_cont.size(1) == Y_cont.size(1), "Inner dimensions (D) must match.");

    const int N = X_cont.size(0);
    const int m = Y_cont.size(0);

    auto XY_T = at::matmul(X_cont, Y_cont.t());
    auto X_norms_sq = at::sum(X_cont * X_cont, 1);
    auto Y_norms_sq = at::sum(Y_cont * Y_cont, 1);

    auto K_out = torch::empty({N, m}, X_cont.options());

    dim3 threads(16, 16);
    dim3 blocks((m + 15) / 16, (N + 15) / 16);

    // --- SECURITY: Pass safe TensorAccessors to the kernel ---
    rbf_finish_kernel_safe<<<blocks, threads>>>(
        K_out.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        X_norms_sq.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
        Y_norms_sq.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
        XY_T.packed_accessor32<float, 2, at::RestrictPtrTraits>(), // Note: order changed for clarity
        N, m, -gamma);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Hybrid RBF Kernel Launch Error: ") + cudaGetErrorString(err));
    }

    return K_out;
}
// --- END OF FILE src/helios_embed/hybrid_kernel.cu (Security Hardened v1.2.1) ---