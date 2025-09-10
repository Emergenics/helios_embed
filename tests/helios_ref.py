# --- START OF FILE tests/helios_ref.py (CORRECTED v1.1) ---
import torch

def _pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    x2 = (X**2).sum(dim=1, keepdim=True)
    y2 = (Y**2).sum(dim=1, keepdim=True).T
    XY = X @ Y.T
    D2 = x2 + y2 - 2.0 * XY
    return torch.clamp(D2, min=0.0)

def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, gamma: float) -> torch.Tensor:
    assert X.device.type == "cpu" and Y.device.type == "cpu"
    assert X.dtype == torch.float64 and Y.dtype == torch.float64
    D2 = _pairwise_sq_dists(X, Y)
    return torch.exp(-gamma * D2)

def nystrom_embed_ref(
    X: torch.Tensor,
    landmarks: torch.Tensor,
    gamma: float,
    ridge: float,
) -> torch.Tensor:
    """
    High-precision CPU reference implementation of the Nystrom embedding.
    --- CORRECTED: Uses the EIGH (eigen-decomposition) path for the inverse square root,
    --- exactly mirroring the C++/CUDA implementation for a bit-perfect comparison.
    """
    if gamma <= 0.0: raise ValueError(f"gamma must be > 0, got {gamma}")
    if ridge < 0.0: raise ValueError(f"ridge must be >= 0, got {ridge}")
    if X.dim() != 2 or landmarks.dim() != 2: raise ValueError("X and landmarks must be 2D tensors.")
    if landmarks.size(0) > 0 and X.size(1) != landmarks.size(1): 
        raise ValueError("X and landmarks must have the same feature dimension D.")

    X64 = X.to(dtype=torch.float64, device="cpu", non_blocking=True)
    L64 = landmarks.to(dtype=torch.float64, device="cpu", non_blocking=True)
    
    if X64.size(0) == 0 or L64.size(0) == 0:
        return torch.empty((X64.size(0), L64.size(0)), dtype=torch.float64)

    K_XL = rbf_kernel(X64, L64, gamma)
    K_LL = rbf_kernel(L64, L64, gamma)
    K_LL_reg = K_LL + ridge * torch.eye(K_LL.size(0), dtype=torch.float64)
    
    # --- THIS IS THE CRITICAL FIX ---
    # Use the same numerically stable eigen-decomposition method as the CUDA code.
    eigenvalues, eigenvectors = torch.linalg.eigh(K_LL_reg, UPLO='L')
    
    # Clamp eigenvalues for stability, preventing division by zero from round-off.
    eigenvalues = torch.clamp(eigenvalues, min=1e-12) # Use a slightly safer floor for float64
    
    # Compute K_mm^{-1/2} = V @ diag(1/sqrt(S)) @ V.T
    S_inv_sqrt = torch.diag(torch.rsqrt(eigenvalues))
    K_mm_inv_sqrt = eigenvectors @ S_inv_sqrt @ eigenvectors.T
    # --- END OF CRITICAL FIX ---

    return (K_XL @ K_mm_inv_sqrt).contiguous()
# --- END OF FILE tests/helios_ref.py (CORRECTED v1.1) ---