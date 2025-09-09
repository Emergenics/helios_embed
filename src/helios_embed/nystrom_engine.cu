// --- START OF FILE src/helios_embed/nystrom_engine.cu (Version 1.2.0 - Final) ---
#include "nystrom_engine.h"
#include <ATen/ATen.h>

// The validation function is unchanged and remains robust.
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

// --- THIS IS THE CRITICAL UPGRADE ---
// The core `build` function is now permanently upgraded to use the faster hybrid kernel.
torch::Tensor compute_rkhs_embedding_nystrom(
    const torch::Tensor& X_in,
    const torch::Tensor& landmarks_in,
    float gamma,
    float initial_ridge)
{
    torch::NoGradGuard no_grad;

    // Handle empty input edge case
    if (X_in.size(0) == 0) {
        return torch::empty({0, landmarks_in.size(0)}, X_in.options());
    }
    
    // Use our single, robust validation function
    validate_inputs_stateless(X_in, landmarks_in);

    auto X = X_in.contiguous();
    auto Lm = landmarks_in.contiguous();
    const auto m = Lm.size(0);

    // Step 1: Compute K_nm and K_mm using our new, faster hybrid kernel
    auto K_nm = rbf_kernel_hybrid_cuda(X, Lm, gamma);
    auto K_mm = rbf_kernel_hybrid_cuda(Lm, Lm, gamma);

    // Step 2: Perform the linear algebra (Cholesky solve or pinv fallback)
    // This logic remains the same, but it now operates on the faster-computed kernel matrices.
    float current_ridge = initial_ridge;
    for (int i = 0; i < 3; ++i) { // Adaptive ridge retry loop
        auto K_mm_reg = K_mm + torch::eye(m, X.options()) * current_ridge;
        try {
            auto L_factor = at::linalg_cholesky(K_mm_reg, false);
            auto Z = at::linalg_solve_triangular(L_factor, K_nm.t(), false);
            return Z.t().contiguous();
        } catch (const c10::Error& e) {
            // If Cholesky fails, increase ridge and retry. This is our robust stability path.
            current_ridge *= 10.0f;
        }
    }

    // Fallback to pseudo-inverse if Cholesky repeatedly fails (rare, for ill-conditioned matrices)
    auto K_mm_reg_final = K_mm + torch::eye(m, X.options()) * current_ridge;
    auto K_mm_inv_sqrt = at::linalg_pinv(K_mm_reg_final).sqrt();
    return at::mm(K_nm, K_mm_inv_sqrt).contiguous();
}
// --- END OF FILE src/helios_embed/nystrom_engine.cu (Version 1.2.0 - Final) ---