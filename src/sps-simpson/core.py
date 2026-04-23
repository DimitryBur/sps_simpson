"""Core implementation of SPS-Simpson algorithm."""

import time
import torch
import numpy as np

__all__ = ["sps_simpson_svd", "randomized_svd_fast", "get_error"]


def sps_simpson_svd(A, eps=1e-4, speed_preset='balanced'):
    """
    SPS-Simpson: Fast SVD approximation using Simpson index.
    
    Parameters
    ----------
    A : torch.Tensor
        Input matrix of shape (m, n)
    eps : float, optional
        Target accuracy (default: 1e-4)
    speed_preset : str, optional
        One of 'accurate', 'balanced', 'fast', 'ultra'
    
    Returns
    -------
    U : torch.Tensor
        Left singular vectors
    S : torch.Tensor
        Singular values
    V : torch.Tensor
        Right singular vectors
    elapsed : float
        Computation time
    """
    start_time = time.time()
    m, n = A.shape
    
    # Settings
    if speed_preset == 'accurate':
        rank_max, p_min, use_fp16 = 500, 0.5, False
    elif speed_preset == 'balanced':
        rank_max, p_min, use_fp16 = 300, 0.6, False
    elif speed_preset == 'fast':
        rank_max, p_min, use_fp16 = 150, 0.7, False
    else:  # ultra
        rank_max, p_min, use_fp16 = 80, 0.8, True
    
    if use_fp16 and A.dtype != torch.float16:
        A = A.half()
    
    # Estimate k_eff using Simpson index
    row_norms = torch.norm(A.float(), dim=1)**2
    sqrt_row_norms = torch.sqrt(row_norms + 1e-8)
    k_eff = int(2.5 * (torch.sum(sqrt_row_norms)**2 / torch.sum(row_norms)))
    k_eff = min(k_eff, m, n, rank_max)
    
    # Adaptive row selection
    p = max(p_min, 1 - (k_eff / m))
    threshold = torch.quantile(row_norms.float(), p)
    mask = row_norms >= threshold
    A_heavy = A[mask, :]
    actual_k = A_heavy.shape[0]
    
    # SVD on submatrix
    U_h, S_h, V_h = torch.linalg.svd(A_heavy.float(), full_matrices=False)
    
    k_target = min(k_eff, len(S_h))
    V_h = V_h[:k_target, :]
    S_h = S_h[:k_target]
    
    # Shift mu
    if speed_preset == 'ultra':
        mu = torch.sqrt(torch.mean(S_h**2) / actual_k) * np.sqrt(eps)
    else:
        G_small_diag = torch.sum(A_heavy.float()**2, dim=1)
        mu = torch.sqrt(torch.mean(G_small_diag) / actual_k) * np.sqrt(eps)
    
    S_rec = torch.sqrt(torch.clamp(S_h**2 - mu**2, min=0))
    
    # Reconstruct U
    V_full = V_h.t().to(A.dtype)
    B = A @ V_full
    S_inv = torch.where(S_rec > 1e-8, 1.0 / S_rec, torch.zeros_like(S_rec))
    U_full = B * S_inv.unsqueeze(0).to(A.dtype)
    
    if use_fp16:
        U_full = U_full.float()
        S_rec = S_rec.float()
        V_full = V_full.float()
    
    return U_full, S_rec, V_full, time.time() - start_time


def randomized_svd_fast(A, rank=100):
    """Fast randomized SVD using torch.svd_lowrank."""
    start = time.time()
    U, S, V = torch.svd_lowrank(A.float(), q=rank)
    return U, S, V, time.time() - start


def get_error(A, U, S, V):
    """Compute relative approximation error."""
    A = A.float()
    U = U.float()
    S = S.float()
    V = V.float()
    
    k = min(S.shape[0], U.shape[1], V.shape[1])
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = V[:, :k]
    A_rec = U_k @ torch.diag(S_k) @ V_k.t()
    return torch.norm(A - A_rec).item() / torch.norm(A).item()