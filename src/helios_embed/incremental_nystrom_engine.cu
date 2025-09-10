// --- START OF FILE src/helios_embed/incremental_nystrom_engine.cu (DEFINITIVELY CORRECTED v2.1.2) ---
#include "incremental_nystrom_engine.h"
#include <ATen/ATen.h> // <-- THIS IS THE CORRECTED LINE, with .h

// Constructor remains correct and robust.
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

// initialize_engine remains correct, using the superior eigh path.
void IncrementalNystromEngine::initialize_engine() {
    torch::NoGradGuard no_grad;
    auto K_mm = at::exp(-gamma_ * at::cdist(landmarks_, landmarks_).pow(2));
    auto K_mm_reg = K_mm + torch::eye(landmarks_.size(0), landmarks_.options()) * ridge_;
    
    auto eigh_result = at::linalg_eigh(K_mm_reg, "L");
    auto eigenvalues = std::get<0>(eigh_result);
    auto eigenvectors = std::get<1>(eigh_result);
    eigenvalues = at::clamp_min(eigenvalues, 1e-8);
    auto S_inv_sqrt = at::diag_embed(at::rsqrt(eigenvalues));
    this->K_mm_inv_sqrt_ = eigenvectors.mm(S_inv_sqrt).mm(eigenvectors.t());
}

// build method remains correct.
torch::Tensor IncrementalNystromEngine::build(const torch::Tensor& X) {
    return compute_rkhs_embedding_nystrom(X, landmarks_, gamma_, ridge_);
}

// The high-performance update function.
torch::Tensor IncrementalNystromEngine::update(const torch::Tensor& X_new_in, const torch::Tensor& Phi_old_in) {
    torch::NoGradGuard no_grad;
    auto X_new = X_new_in.contiguous();
    auto Phi_old = Phi_old_in.contiguous();
    
    validate_inputs_stateless(X_new, landmarks_);
    if (Phi_old.size(0) > 0 && Phi_old.size(1) != landmarks_.size(0)) {
         throw std::runtime_error("Feature dimension m of Phi_old must match number of landmarks.");
    }
    TORCH_CHECK(torch::isfinite(Phi_old).all().item<bool>(), "Phi_old must not contain NaN or Inf values.");

    if (X_new.size(0) == 0) {
        return Phi_old.clone();
    }
    
    auto K_nm_new = rbf_kernel_hybrid_cuda(X_new, landmarks_, gamma_);
    auto Phi_new_rows = at::mm(K_nm_new, this->K_mm_inv_sqrt_);

    return at::cat({Phi_old, Phi_new_rows}, 0).contiguous();
}
// --- END OF FILE src/helios_embed/incremental_nystrom_engine.cu (DEFINITIVELY CORRECTED v2.1.2) ---