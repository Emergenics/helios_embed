// --- START OF FILE src/helios_embed/incremental_nystrom_engine.cpp (FINAL v3.3
// - DEFINITIVELY CORRECTED) ---
#include "incremental_nystrom_engine.h"
#include "common.h" // Includes torch headers and our HELIOS_CPU_BUILD macro

// --- CPU-ONLY BUILD PATH ---
// This code is ONLY compiled when the HELIOS_CPU_BUILD macro is defined by
// setup.py
#ifdef HELIOS_CPU_BUILD

IncrementalNystromEngine::IncrementalNystromEngine(torch::Tensor landmarks,
                                                   float gamma, float ridge) {
  TORCH_CHECK(
      false,
      "Helios.Embed was compiled in CPU-only mode. The "
      "'IncrementalNystromEngine' class requires CUDA and is not available.");
}

// These methods must exist to satisfy the class definition for the compiler,
// but they will never be reached because the constructor always throws in a CPU
// build.
torch::Tensor IncrementalNystromEngine::build(const torch::Tensor &X) {
  return torch::Tensor();
}

torch::Tensor
IncrementalNystromEngine::update(const torch::Tensor &X_new_in,
                                 const torch::Tensor &Phi_old_in) {
  return torch::Tensor();
}

void IncrementalNystromEngine::initialize_engine() {
  // Empty stub for CPU build
}

// --- CUDA BUILD PATH ---
// This code is ONLY compiled when HELIOS_CPU_BUILD is NOT defined.
#else

IncrementalNystromEngine::IncrementalNystromEngine(torch::Tensor landmarks,
                                                   float gamma, float ridge) {
  // --- SECURITY: Immediate validation in constructor ---
  TORCH_CHECK(landmarks.is_cuda(), "Landmarks must be a CUDA tensor.");
  TORCH_CHECK(landmarks.dtype() == torch::kFloat32,
              "Landmarks must be a float32 tensor.");
  TORCH_CHECK(landmarks.dim() == 2, "Landmarks must be 2D.");
  TORCH_CHECK(landmarks.size(0) > 0,
              "Landmarks tensor must have at least one landmark (m > 0).");
  TORCH_CHECK(torch::isfinite(landmarks).all().item<bool>(),
              "Landmarks must not contain NaN or Inf.");

  this->landmarks_ = landmarks.contiguous();
  this->gamma_ = gamma;
  this->ridge_ = ridge;

  initialize_engine();
}

void IncrementalNystromEngine::initialize_engine() {
  torch::NoGradGuard no_grad;
  auto K_mm = at::exp(-gamma_ * at::cdist(landmarks_, landmarks_).pow(2));
  auto K_mm_reg =
      K_mm + torch::eye(landmarks_.size(0), landmarks_.options()) * ridge_;

  auto eigh_result = at::linalg_eigh(K_mm_reg, "L");
  auto eigenvalues = std::get<0>(eigh_result);
  auto eigenvectors = std::get<1>(eigh_result);
  eigenvalues = at::clamp_min(eigenvalues, 1e-8);
  auto S_inv_sqrt = at::diag_embed(at::rsqrt(eigenvalues));
  this->K_mm_inv_sqrt_ = eigenvectors.mm(S_inv_sqrt).mm(eigenvectors.t());
}

torch::Tensor IncrementalNystromEngine::build(const torch::Tensor &X) {
  return compute_rkhs_embedding(X, landmarks_, gamma_, ridge_);
}

torch::Tensor
IncrementalNystromEngine::update(const torch::Tensor &X_new_in,
                                 const torch::Tensor &Phi_old_in) {
  torch::NoGradGuard no_grad;
  auto X_new = X_new_in.contiguous();
  auto Phi_old = Phi_old_in.contiguous();

  if (X_new.size(0) == 0) {
    return Phi_old.clone();
  }

  validate_inputs_stateless(X_new, landmarks_);
  TORCH_CHECK(Phi_old.is_cuda() && Phi_old.dtype() == torch::kFloat32,
              "Phi_old must be a float32 CUDA tensor.");
  if (Phi_old.size(0) > 0) {
    TORCH_CHECK(Phi_old.dim() == 2, "Phi_old must be 2D.");
    TORCH_CHECK(
        Phi_old.size(1) == landmarks_.size(0),
        "Feature dimension m of Phi_old must match number of landmarks.");
  }

  auto K_nm_new = at::exp(-gamma_ * at::cdist(X_new, landmarks_).pow(2));
  auto Phi_new_rows = at::mm(K_nm_new, this->K_mm_inv_sqrt_);

  return at::cat({Phi_old, Phi_new_rows}, 0).contiguous();
}

#endif // HELIOS_CPU_BUILD
// --- END OF FILE src/helios_embed/incremental_nystrom_engine.cpp (FINAL v3.3 -
// DEFINITIVELY CORRECTED) ---