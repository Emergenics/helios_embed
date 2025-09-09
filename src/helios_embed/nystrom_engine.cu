// --- START OF FILE nystrom_engine/nystrom_engine.cu ---
#include "nystrom_engine.h"
#include <ATen/ATen.h>

// This file contains the stateless, ATen-backed Nystrom feature embedding.
// It is our validated, bit-perfect foundation.
void validate_inputs_stateless(const at::Tensor& X, const at::Tensor& Lm) {
    TORCH_CHECK(X.is_cuda() && Lm.is_cuda(), "Input tensors must reside on a CUDA device.");
    TORCH_CHECK(Lm.is_cuda(), "Landmarks tensor Lm must reside on a CUDA device.");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && Lm.dtype() == torch::kFloat32, "Input tensors must be of type float32.");
    TORCH_CHECK(Lm.dtype() == torch::kFloat32, "Landmarks tensor Lm must be of type float32.");
    TORCH_CHECK(X.dim() == 2 && Lm.dim() == 2, "Input tensors must be 2-dimensional.");
    TORCH_CHECK(Lm.dim() == 2, "Landmarks tensor Lm must be 2-dimensional ([m, D]).");
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
    float initial_ridge)
{
    torch::NoGradGuard no_grad;
    TORCH_CHECK(landmarks_in.is_contiguous(), "Landmarks tensor must be contiguous.");
    if (X_in.size(0) == 0) {
        return torch::empty({0, landmarks_in.size(0)}, X_in.options());
    }
    validate_inputs_stateless(X_in, landmarks_in);
    auto X = X_in.contiguous();
    auto Lm = landmarks_in.contiguous();
    const auto m = Lm.size(0);
    auto K_nm = at::exp(-gamma * at::cdist(X, Lm).pow(2));
    auto K_mm = at::exp(-gamma * at::cdist(Lm, Lm).pow(2));
    float current_ridge = initial_ridge;
    for (int i = 0; i < 3; ++i) {
        auto K_mm_reg = K_mm + torch::eye(m, X.options()) * current_ridge;
        try {
            auto L_factor = at::linalg_cholesky(K_mm_reg, false);
            auto Z = at::linalg_solve_triangular(L_factor, K_nm.t(), false);
            return Z.t().contiguous();
        } catch (const c10::Error& e) {
            current_ridge *= 10.0f;
        }
    }
    auto K_mm_reg_final = K_mm + torch::eye(m, X.options()) * current_ridge;
    auto K_mm_inv_sqrt = at::linalg_pinv(K_mm_reg_final).sqrt();
    return at::mm(K_nm, K_mm_inv_sqrt).contiguous();
}
// --- END OF FILE ---