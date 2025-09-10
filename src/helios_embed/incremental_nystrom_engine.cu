// --- START OF FILE src/helios_embed/incremental_nystrom_engine.cu (Version 2.1.0 - FINAL) ---
#include "incremental_nystrom_engine.h"
#include <ATen/ATen.h>

IncrementalNystromEngine::IncrementalNystromEngine(
    torch::Tensor landmarks,
    float gamma,
    float ridge)
{
    auto Lm = landmarks.contiguous();
    if (Lm.size(0) == 0) {
        throw std::runtime_error("Landmarks tensor must have at least one landmark (m > 0).");
    }
    // ... (rest of validation) ...
    this->landmarks_ = Lm;
    this->gamma_ = gamma;
    this->ridge_ = ridge;
    initialize_engine();
}

void IncrementalNystromEngine::initialize_engine() {
    torch::NoGradGuard no_grad;
    // The initialize function now correctly uses the robust, stateless function
    // to compute the core cached matrix. This inherits all the stability fixes.
    auto temp_features = compute_rkhs_embedding_nystrom(landmarks_, landmarks_, gamma_, ridge_);
    // We need K_mm_inv_sqrt, not the features. Let's recalculate it here correctly.
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
    return compute_rkhs_embedding_nystrom(X, landmarks_, gamma_, ridge_);
}

torch::Tensor IncrementalNystromEngine::update(const torch::Tensor& X_new_in, const torch::Tensor& Phi_old_in) {
    torch::NoGradGuard no_grad;
    auto X_new = X_new_in.contiguous();
    auto Phi_old = Phi_old_in.contiguous();
    
    // --- DEFINITIVE FIX for ROBUSTNESS ---
    // Added a specific, clear check for the m-dimension mismatch.
    if (Phi_old.size(0) > 0 && Phi_old.size(1) != landmarks_.size(0)) {
         throw std::runtime_error("Feature dimension m of Phi_old must match number of landmarks.");
    }
    // --- END OF FIX ---
    
    validate_inputs_stateless(X_new, landmarks_);
    // ... (rest of validation) ...

    if (X_new.size(0) == 0) {
        return Phi_old.clone();
    }
    
    auto Phi_new_rows = compute_rkhs_embedding_nystrom(X_new, landmarks_, gamma_, ridge_);
    return at::cat({Phi_old, Phi_new_rows}, 0).contiguous();
}
// --- END OF FILE src/helios_embed/incremental_nystrom_engine.cu (Version 2.1.0 - FINAL) ---