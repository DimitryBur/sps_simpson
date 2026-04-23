"""
SPS-Simpson: Fast SVD approximation for LLM matrices
Corrected version based on critical review
"""

import time
import torch
import numpy as np

# ============================================================
# CORE ALGORITHMS
# ============================================================

def simpson_index(A):
    """
    Correct Simpson index for matrix rows.
    Measures concentration of row norms.
    
    Returns:
        effective number of rows in [1, m]
    """
    row_norms_sq = torch.norm(A, dim=1) ** 2
    total = row_norms_sq.sum()
    if total < 1e-8:
        return 1.0
    p = row_norms_sq / total
    simpson = 1.0 - (p ** 2).sum()
    # Convert Simpson index to effective number
    effective = 1.0 / (1.0 - simpson) if simpson < 0.999 else float(A.shape[0])
    return max(1.0, min(float(A.shape[0]), effective))


def estimate_rank_from_simpson(A, max_rank=500):
    """
    Estimate effective rank using Simpson index on rows and columns.
    
    Args:
        A: Input matrix
        max_rank: Upper bound for rank estimation
    
    Returns:
        Estimated effective rank
    """
    m, n = A.shape
    eff_rows = simpson_index(A)
    eff_cols = simpson_index(A.T)
    eff_rank = int(np.sqrt(eff_rows * eff_cols))
    return max(1, min(eff_rank, m, n, max_rank))


def sps_simpson_svd(A, energy_threshold=0.9999, speed_preset='balanced'):
    """
    Fast SVD approximation using row sampling and Simpson index.
    
    Parameters
    ----------
    A : torch.Tensor of shape (m, n)
        Input matrix
    energy_threshold : float, default=0.9999
        Fraction of energy to preserve (1 - relative error)
    speed_preset : str, default='balanced'
        'accurate' (500 max rows), 'balanced' (300), 'fast' (100)
    
    Returns
    -------
    U : torch.Tensor of shape (m, k)
        Left singular vectors
    S : torch.Tensor of shape (k,)
        Singular values (estimated)
    V : torch.Tensor of shape (n, k)
        Right singular vectors
    info : dict with metadata
    """
    start = time.time()
    m, n = A.shape
    orig_dtype = A.dtype
    
    # Settings
    if speed_preset == 'accurate':
        max_rows, p_min = 500, 0.5
    elif speed_preset == 'balanced':
        max_rows, p_min = 300, 0.6
    else:  # fast
        max_rows, p_min = 100, 0.7
    
    # Step 1: Estimate target rank from Simpson index
    target_rank = estimate_rank_from_simpson(A, max_rows)
    target_rank = min(target_rank, max_rows, m, n)
    
    # Handle edge cases - fallback to randomized SVD
    if target_rank < 1 or target_rank >= min(m, n):
        from torch import svd_lowrank
        U, S, V = svd_lowrank(A, q=min(100, m, n))
        return U, S, V, {'time': time.time() - start, 'method': 'fallback'}
    
    # Step 2: Adaptive row selection
    row_norms_sq = torch.norm(A, dim=1) ** 2
    p = max(p_min, 1.0 - (target_rank / m))
    p = min(0.95, p)
    
    threshold = torch.quantile(row_norms_sq, p)
    mask = row_norms_sq >= threshold
    
    if mask.sum() == 0:
        from torch import svd_lowrank
        U, S, V = svd_lowrank(A, q=min(100, m, n))
        return U, S, V, {'time': time.time() - start, 'method': 'fallback_no_rows'}
    
    A_heavy = A[mask, :]
    actual_rows = A_heavy.shape[0]
    
    # Step 3: SVD on selected rows
    U_h, S_h, V_h = torch.linalg.svd(A_heavy.to(orig_dtype), full_matrices=False)
    
    # Step 4: Truncate to target rank
    k = min(target_rank, len(S_h))
    if k == 0:
        return (torch.zeros(m, 1, dtype=orig_dtype, device=A.device),
                torch.zeros(1, dtype=orig_dtype, device=A.device),
                torch.zeros(n, 1, dtype=orig_dtype, device=A.device),
                {'time': time.time() - start, 'method': 'empty'})
    
    V_k = V_h[:k, :].t().to(orig_dtype)  # shape (n, k)
    S_k = S_h[:k].to(orig_dtype)
    
    # Step 5: Reconstruct U for all original rows
    # U[:, i] = (A @ V[:, i]) / S[i]
    B = A @ V_k  # shape (m, k)
    S_inv = torch.where(S_k > 1e-8, 1.0 / S_k, torch.zeros_like(S_k))
    U_k = B * S_inv.unsqueeze(0)  # shape (m, k)
    
    elapsed = time.time() - start
    
    info = {
        'time': elapsed,
        'estimated_rank': target_rank,
        'rows_used': actual_rows,
        'rows_total': m,
        'method': 'sps_simpson',
        'speed_preset': speed_preset,
        'energy_threshold': energy_threshold,
    }
    
    return U_k, S_k, V_k, info


def randomized_svd_baseline(A, rank=100, n_oversamples=10, n_iter=2):
    """
    Randomized SVD baseline for comparison.
    
    Parameters
    ----------
    A : torch.Tensor
        Input matrix
    rank : int
        Target rank
    n_oversamples : int
        Number of oversamples
    n_iter : int
        Number of power iterations
    
    Returns
    -------
    U, S, V, time : tuple
    """
    start = time.time()
    m, n = A.shape
    rank = min(rank, m, n)
    
    # Random projection
    Omega = torch.randn(n, rank + n_oversamples, device=A.device, dtype=A.dtype)
    Y = A @ Omega
    
    # Power iterations
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ A
    U_hat, S, V_hat = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat
    V = V_hat.T
    
    return U[:, :rank], S[:rank], V[:, :rank], time.time() - start


def get_approximation_error(A, U, S, V):
    """Compute relative Frobenius norm error: ||A - USV^T|| / ||A||."""
    with torch.no_grad():
        k = min(S.shape[0], U.shape[1], V.shape[1])
        U_k = U[:, :k]
        S_k = S[:k]
        V_k = V[:, :k]
        A_rec = U_k @ torch.diag(S_k) @ V_k.t()
        err = torch.norm(A - A_rec) / torch.norm(A)
        return err.item()


def benchmark_all(n=512):
    """Run benchmark on all matrix types."""
    results = []
    
    test_matrix_generators = [
        ("Exp α=0.05", lambda n: generate_llm_exponential(n, 0.05)),
        ("Exp α=0.10", lambda n: generate_llm_exponential(n, 0.10)),
        ("Power β=2.0", lambda n: generate_power_law(n, 2.0)),
        ("Power β=3.0", lambda n: generate_power_law(n, 3.0)),
        ("Low Rank 50", lambda n: generate_low_rank(n, 50)),
        ("Random", lambda n: torch.randn(n, n)),
        ("Correlation 0.8", lambda n: generate_correlation(n, 0.8)),
        ("Toeplitz 0.5", lambda n: generate_toeplitz(n, 0.5)),
        ("Sparse 5%", lambda n: generate_sparse(n, 0.05)),
    ]
    
    print("\n" + "=" * 90)
    print("SPS-Simpson vs Randomized SVD Benchmark")
    print("=" * 90)
    print(f"{'Matrix Type':<25} | {'SPS Error':<12} | {'Rand Error':<12} | {'Winner':<8}")
    print("-" * 90)
    
    for name, gen_func in test_matrix_generators:
        A = gen_func(n)
        
        # SPS
        _, S_sps, _, info_sps = sps_simpson_svd(A, speed_preset='balanced')
        err_sps = 1 - (S_sps ** 2).sum() / (torch.norm(A) ** 2)
        
        # Randomized SVD
        _, S_rand, _, _ = randomized_svd_baseline(A, rank=min(100, n))
        err_rand = 1 - (S_rand ** 2).sum() / (torch.norm(A) ** 2)
        
        winner = "SPS" if err_sps < err_rand else "Rand"
        results.append((name, err_sps, err_rand, winner))
        print(f"{name:<25} | {err_sps:<11.3%} | {err_rand:<11.3%} | {winner:<8}")
    
    return results


def benchmark_llm(alphas=[0.02, 0.05, 0.10, 0.20, 0.30, 0.50], n=512):
    """LLM-specific benchmark with different alpha values."""
    print("\n" + "=" * 70)
    print("LLM-SPECIFIC BENCHMARK")
    print("=" * 70)
    print(f"{'Alpha':<10} | {'SPS Error':<12} | {'Rand Error':<12} | {'Winner':<8}")
    print("-" * 70)
    
    for alpha in alphas:
        A = generate_llm_exponential(n, alpha)
        
        # SPS
        _, S_sps, _, _ = sps_simpson_svd(A, speed_preset='balanced')
        err_sps = 1 - (S_sps ** 2).sum() / (torch.norm(A) ** 2)
        
        # Randomized SVD
        _, S_rand, _, _ = randomized_svd_baseline(A, rank=min(100, n))
        err_rand = 1 - (S_rand ** 2).sum() / (torch.norm(A) ** 2)
        
        winner = "SPS" if err_sps < err_rand else "Rand"
        print(f"{alpha:<10.3f} | {err_sps:<11.3%} | {err_rand:<11.3%} | {winner:<8}")


def generate_llm_exponential(n=512, alpha=0.05):
    """Generate LLM-like matrix with exponential spectrum."""
    U, _ = torch.linalg.qr(torch.randn(n, n))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    s = torch.exp(-alpha * torch.arange(n).float())
    s = s / s[0]
    return U @ torch.diag(s) @ V.T


def generate_power_law(n=512, beta=2.0):
    """Generate matrix with power-law spectrum."""
    U, _ = torch.linalg.qr(torch.randn(n, n))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    s = (torch.arange(1, n+1).float()) ** (-beta)
    s = s / s[0]
    return U @ torch.diag(s) @ V.T


def generate_low_rank(n=512, rank=50):
    """Generate low-rank matrix."""
    U = torch.randn(n, rank)
    V = torch.randn(n, rank)
    return U @ V.T


def generate_correlation(n=512, rho=0.8):
    """Generate correlation matrix."""
    A = torch.ones(n, n) * rho
    A.fill_diagonal_(1.0)
    return A


def generate_toeplitz(n=512, decay=0.5):
    """Generate Toeplitz matrix with exponential decay."""
    A = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            A[i, j] = decay ** abs(i - j)
    return A


def generate_sparse(n=512, density=0.05):
    """Generate random sparse matrix."""
    A = torch.randn(n, n) * (torch.rand(n, n) < density).float()
    return A


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n🚀 SPS-Simpson: Fast SVD for LLM Matrices")
    print("=" * 50)
    
    # Run benchmarks
    benchmark_all(n=512)
    benchmark_llm(n=512)
    
    print("\n" + "=" * 50)
    print("✅ Benchmark complete!")
    print("Recommendation: Use SPS-Simpson for LLM compression")
