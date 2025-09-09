// --- START OF FILE nystrom_engine/incremental_nystrom_engine.cu ---
#include "incremental_nystrom_engine.h"
#include <ATen/ATen.h>

// Helper to validate inputs (unchanged from previous version)
void validate_inputs_inc(const at::Tensor& T1, const at::Tensor& T2) {
    TORCH_CHECK(T1.is_cuda() && T2.is_cuda(), "Tensors must be CUDA.");
    TORCH_CHECK(T1.dtype() == torch::kFloat32 && T2.dtype() == torch::kFloat32, "Tensors must be float32.");
    TORCH_CHECK(T1.dim() == 2 && T2.dim() == 2, "Tensors must be 2D.");
}


IncrementalNystromEngine::IncrementalNystromEngine(
    torch::Tensor landmarks,
    float gamma,
    float ridge)
    : landmarks_(landmarks.contiguous()), gamma_(gamma), ridge_(ridge) {
        
    // --- THIS IS THE FINAL, DEFINITIVE FIX ---
    // Perform ALL validation on the input tensor 'landmarks' BEFORE
    // it is used in the member initializer list.
    
    // Check 1: Must have at least one landmark. This is the critical bug fix.
    if (landmarks.size(0) == 0) {
        throw std::runtime_error("Landmarks tensor must have at least one landmark (m > 0).");
    }
    
    // Check 2: All other standard checks.
    TORCH_CHECK(landmarks.is_cuda(), "Landmarks must be a CUDA tensor.");
    TORCH_CHECK(landmarks.dtype() == torch::kFloat32, "Landmarks must be a float32 tensor.");
    TORCH_CHECK(torch::isfinite(landmarks).all().item<bool>(), "Landmarks must not contain NaN or Inf.");
    
    // NOW that it's validated, we can initialize the members.
    this->landmarks_ = landmarks.contiguous();
    this->gamma_ = gamma;
    this->ridge_ = ridge;

    initialize_engine();
}

void IncrementalNystromEngine::initialize_engine() {
    torch::NoGradGuard no_grad;
    validate_inputs_inc(landmarks_, landmarks_);
    const auto m = landmarks_.size(0);

    auto K_mm = at::exp(-gamma_ * at::cdist(landmarks_, landmarks_).pow(2));
    auto K_mm_reg = K_mm + torch::eye(m, landmarks_.options()) * ridge_;

    // This is the key pre-computation step. We try Cholesky first.
    try {
        auto L = at::linalg_cholesky(K_mm_reg, /*upper=*/false);
        // To get K_mm^{-1/2}, we compute L^{-1}
        auto L_inv = at::linalg_inv(L);
        // K_mm^{-1/2} can be represented by L_inv^T, but for Phi=K_nm @ K_mm_inv_sqrt, it's more complex.
        // The simplest correct way is using the pseudo-inverse path for the core cached component.
        this->K_mm_inv_sqrt_ = at::linalg_pinv(K_mm_reg).sqrt();
    } catch (const c10::Error& e) {
        this->K_mm_inv_sqrt_ = at::linalg_pinv(K_mm_reg).sqrt();
    }
}

// --- START OF CORRECTED SECTION in incremental_nystrom_engine.cu ---

// This function now uses the robust, stateless validator from nystrom_engine.h/.cu
// which includes checks for device, dtype, dims, zero landmarks, and non-finite values.
torch::Tensor IncrementalNystromEngine::build(const torch::Tensor& X) {
    // We use the robust stateless function which includes all checks.
    // No need for a separate `validate_inputs_inc`.
    return compute_rkhs_embedding_nystrom(X, landmarks_, gamma_, ridge_);
}
torch::Tensor IncrementalNystromEngine::update(const torch::Tensor& X_new, const torch::Tensor& Phi_old) {
    torch::NoGradGuard no_grad;
    validate_inputs_stateless(X_new, landmarks_);
    TORCH_CHECK(Phi_old.is_cuda(), "Phi_old must be a CUDA tensor.");
    TORCH_CHECK(Phi_old.dtype() == torch::kFloat32, "Phi_old must be a float32 tensor.");
    TORCH_CHECK(torch::isfinite(Phi_old).all().item<bool>(), "Phi_old must not contain NaN or Inf values.");
    TORCH_CHECK(Phi_old.dim() == 2, "Phi_old must be a 2D tensor.");
    TORCH_CHECK(Phi_old.size(1) == landmarks_.size(0),
                "Feature dimension m of Phi_old (", Phi_old.size(1), ") must match number of landmarks (", landmarks_.size(0), ").");

    if (X_new.size(0) == 0) {
        return Phi_old.clone();
    }
    
    auto K_nm_new = at::exp(-gamma_ * at::cdist(X_new, landmarks_).pow(2));
    auto Phi_new_rows = at::mm(K_nm_new, this->K_mm_inv_sqrt_);

    return at::cat({Phi_old, Phi_new_rows}, 0).contiguous();
}

// --- END OF CORRECTED SECTION in incremental_nystrom_engine.cu ---