// --- START OF FILE src/helios_embed/nystrom_engine.cu (Version 2.1.0 - FINAL) ---
#include "nystrom_engine.h"
#include <ATen/ATen.h>

// Forward declaration for our internal, corrected hybrid kernel.
torch::Tensor rbf_kernel_hybrid_cuda_internal(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    float gamma);


void validate_inputs_stateless(const at::Tensor& X, const at::Tensor& Lm) {
    TORCH_CHECK(X.is_cuda() && Lm.is_cuda(), "Input tensors must reside on a CUDA device.");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && Lm.dtype() == torch::kFloat32, "Input tensors must be of type float32.");
    TORCH_CHECK(X.dim() == 2 && Lm.dim() == 2, "Input tensors must be 2-dimensional.");
    if (Lm.size(0) <= 0) {
        throw std::runtime_error("Landmarks tensor must have at least one landmark (m > 0).");
    }
    TORCH_CHECK(X.size(1) == Lm.size(1), "Feature dimensions of input X and landmarks Lm must match.");
    TORCH_CHECK(torch::isfinite(X).all().item<bool>(), "Input tensor X must not contain NaN or Inf values.");
    TORCH_CHECK(torch::isfinite(Lm).all().item<bool>(), "Landmarks tensor must not contain NaN or Inf values.");
}

torch::Tensor compute_rkhs_embedding_nystrom(
    const torch::Tensor& X_in,
    const torch::Tensor& landmarks_in,
    float gamma,
    float ridge)
{
    torch::NoGradGuard no_grad;
    auto X = X_in.contiguous();
    auto Lm = landmarks_in.contiguous();
    
    if (X.size(0) == 0) {
        return torch::empty({0, Lm.size(0)}, X.options());
    }
    validate_inputs_stateless(X, Lm);

    auto K_nm = rbf_kernel_hybrid_cuda_internal(X, Lm, gamma);
    auto K_mm = rbf_kernel_hybrid_cuda_internal(Lm, Lm, gamma);
    
    auto K_mm_reg = K_mm + torch::eye(Lm.size(0), X.options()) * ridge;

    auto eigh_result = at::linalg_eigh(K_mm_reg, "L");
    auto eigenvalues = std::get<0>(eigh_result);
    auto eigenvectors = std::get<1>(eigh_result);

    eigenvalues = at::clamp_min(eigenvalues, 1e-8);
    auto S_inv_sqrt = at::diag_embed(at::rsqrt(eigenvalues));

    auto K_mm_inv_sqrt = eigenvectors.mm(S_inv_sqrt).mm(eigenvectors.t());
    return at::mm(K_nm, K_mm_inv_sqrt).contiguous();
}

// --- Internal Hybrid Kernel Implementation ---
__global__ void rbf_finish_kernel_final(
    const at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> XY_T_acc,
    const at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> X_norms_sq_acc,
    const at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> Y_norms_sq_acc,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> K_out_acc,
    int N, int m, float neg_gamma)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < m) {
        float dist_sq = X_norms_sq_acc[i] + Y_norms_sq_acc[j] - 2.0f * XY_T_acc[i][j];
        K_out_acc[i][j] = expf(neg_gamma * dist_sq);
    }
}

torch::Tensor rbf_kernel_hybrid_cuda_internal(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    float gamma)
{
    const int N = X.size(0);
    const int m = Y.size(0);

    auto XY_T = at::matmul(X, Y.t());
    auto X_norms_sq = at::sum(X * X, 1);
    auto Y_norms_sq = at::sum(Y * Y, 1);

    auto K_out = torch::empty({N, m}, X.options());

    dim3 threads(16, 16);
    dim3 blocks((m + 15) / 16, (N + 15) / 16);

    // --- DEFINITIVE FIX for ACCURACY ---
    // The arguments are now passed in the correct order, matching the kernel definition.
    rbf_finish_kernel_final<<<blocks, threads>>>(
        XY_T.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        X_norms_sq.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
        Y_norms_sq.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
        K_out.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        N, m, -gamma);
    // --- END OF FIX ---
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Hybrid RBF Kernel Launch Error: ") + cudaGetErrorString(err));
    }

    return K_out;
}
// --- END OF FILE src/helios_embed/nystrom_engine.cu (Version 2.1.0 - FINAL) ---