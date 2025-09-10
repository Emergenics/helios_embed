// --- START OF FILE src/helios_embed/incremental_nystrom_engine.cu (FINAL, STABLE v1.1.0) ---
#include "incremental_nystrom_engine.h"
#include <ATen/ATen.h>

IncrementalNystromEngine::IncrementalNystromEngine(
    torch::Tensor landmarks,
    float gamma,
    float ridge)
{
    // --- SECURITY: Immediate validation in constructor ---
    TORCH_CHECK(landmarks.is_cuda(), "Landmarks must be a CUDA tensor.");
    TORCH_CHECK(landmarks.dtype() == torch::kFloat32, "Landmarks must be a float32 tensor.");
    TORCH_CHECK(landmarks.dim() == 2, "Landmarks must be 2D.");
    TORCH_CHECK(landmarks.size(0) > 0, "Landmarks tensor must have at least one landmark (m > 0).");
    TORCH_CHECK(torch::isfinite(landmarks).all().item<bool>(), "Landmarks must not contain NaN or Inf.");
    
    this->landmarks_ = landmarks.contiguous();
    this->gamma_ = gamma;
    this->ridge_ = ridge;

    initialize_engine();
}

void IncrementalNystromEngine::initialize_engine() {
    torch::NoGradGuard no_grad;
    // This logic is self-contained and already validated as numerically stable.
    auto K_mm = at::exp(-gamma_ * at::cdist(landmarks_, landmarks_).pow(2));
    auto K_mm_reg = K_mm + torch::eye(landmarks_.size(0), landmarks_.options()) * ridge_;
    
    auto eigh_result = at::linalg_eigh(K_mm_reg, "L");
    auto eigenvalues = std::get<0>(eigh_result);
    auto eigenvectors = std::get<1>(eigh_result);
    eigenvalues = at::clamp_min(eigenvalues, 1e-8);
    auto S_inv_sqrt = at::diag_embed(at::rsqrt(eigenvalues));
    this->K_mm_inv_sqrt_ = eigenvectors.mm(S_inv_sqrt).mm(eigenvectors.t());
}

torch::Tensor IncrementalNystromEngine::build(const torch::Tensor& X) {
    // The `build` method correctly re-uses the now-hardened stateless function.
    return compute_rkhs_embedding(X, landmarks_, gamma_, ridge_);
}

torch::Tensor IncrementalNystromEngine::update(const torch::Tensor& X_new_in, const torch::Tensor& Phi_old_in) {
    torch::NoGradGuard no_grad;
    auto X_new = X_new_in.contiguous();
    auto Phi_old = Phi_old_in.contiguous();
    
    // --- Gracefully handle empty update edge case ---
    if (X_new.size(0) == 0) {
        return Phi_old.clone();
    }

    // --- SECURITY: Validation gate for the update operation ---
    validate_inputs_stateless(X_new, landmarks_); // Re-use shared validator
    TORCH_CHECK(Phi_old.is_cuda() && Phi_old.dtype() == torch::kFloat32, "Phi_old must be a float32 CUDA tensor.");
    if (Phi_old.size(0) > 0) {
        TORCH_CHECK(Phi_old.dim() == 2, "Phi_old must be 2D.");
        TORCH_CHECK(Phi_old.size(1) == landmarks_.size(0), "Feature dimension m of Phi_old must match number of landmarks.");
    }
    
    // The correct, lean, high-performance logic.
    auto K_nm_new = at::exp(-gamma_ * at::cdist(X_new, landmarks_).pow(2));
    auto Phi_new_rows = at::mm(K_nm_new, this->K_mm_inv_sqrt_);

    return at::cat({Phi_old, Phi_new_rows}, 0).contiguous();
}
// --- END OF FILE src/helios_embed/incremental_nystrom_engine.cu (FINAL, STABLE v1.1.0) ---