// --- START OF FILE src/helios_embed/nystrom_engine.cu (CORRECTED BACK TO
// v1.1.0) ---
#include "nystrom_engine.h"
#include <ATen/ATen.h>

// This is the single, definitive validation function for the stateless API.
void validate_inputs_stateless(const at::Tensor &X, const at::Tensor &Lm) {
  TORCH_CHECK(X.is_cuda() && Lm.is_cuda(),
              "Input tensors must reside on a CUDA device.");
  TORCH_CHECK(X.dtype() == torch::kFloat32 && Lm.dtype() == torch::kFloat32,
              "Input tensors must be of type float32.");
  TORCH_CHECK(X.dim() == 2 && Lm.dim() == 2,
              "Input tensors must be 2-dimensional.");
  TORCH_CHECK(Lm.size(0) > 0,
              "Landmarks tensor must have at least one landmark (m > 0).");
  TORCH_CHECK(X.size(1) == Lm.size(1),
              "Feature dimensions of input X and landmarks Lm must match.");
  TORCH_CHECK(torch::isfinite(X).all().item<bool>(),
              "Input tensor X must not contain NaN or Inf values.");
  TORCH_CHECK(torch::isfinite(Lm).all().item<bool>(),
              "Landmarks tensor must not contain NaN or Inf values.");
}

torch::Tensor compute_rkhs_embedding(const torch::Tensor &X_in,
                                     const torch::Tensor &landmarks_in,
                                     float gamma, float ridge) {
  torch::NoGradGuard no_grad;

  auto X = X_in.contiguous();
  auto Lm = landmarks_in.contiguous();

  if (X.size(0) == 0) {
    return torch::empty({0, Lm.size(0)}, X.options());
  }

  validate_inputs_stateless(X, Lm);
  TORCH_CHECK(gamma > 0, "gamma must be positive.");
  TORCH_CHECK(ridge >= 0, "ridge must be non-negative.");

  const auto m = Lm.size(0);

  auto K_nm = at::exp(-gamma * at::cdist(X, Lm).pow(2));
  auto K_mm = at::exp(-gamma * at::cdist(Lm, Lm).pow(2));

  auto K_mm_reg = K_mm + torch::eye(m, X.options()) * ridge;

  auto eigh_result = at::linalg_eigh(K_mm_reg, "L");
  auto eigenvalues = std::get<0>(eigh_result);
  auto eigenvectors = std::get<1>(eigh_result);

  eigenvalues = at::clamp_min(eigenvalues, 1e-8);
  auto S_inv_sqrt = at::diag_embed(at::rsqrt(eigenvalues));

  auto K_mm_inv_sqrt = eigenvectors.mm(S_inv_sqrt).mm(eigenvectors.t());
  return at::mm(K_nm, K_mm_inv_sqrt).contiguous();
}
// --- END OF FILE src/helios_embed/nystrom_engine.cu (CORRECTED BACK TO v1.1.0)
// ---