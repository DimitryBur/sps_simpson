```text
# SPS-Simpson: Fast SVD for LLM Matrices

**Fast approximation of singular values, effective rank, and spectrum for large matrices with exponential decay (LLM weights, recommendation systems, graphs).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

---

## 🚀 Quick Install

```bash
pip install git+https://github.com/DimitryBur/sps-simpson.git
```

---

📌 Key Features

Feature Description
Fast SVD 0.15s on 4096×4096 matrix (193× faster than full SVD)
High accuracy 0.03% error on LLM-like matrices
Automatic rank selection No need to specify target rank
Multiple speed presets accurate / balanced / fast
Memory efficient Works with 10KB sample instead of full matrix

---

🎯 When to Use

✅ Use SPS-Simpson 
LLM weights (LLaMA, Mistral, GPT) Random matrices
Recommendation system matrices Matrices with flat spectrum
Graphs with exponential decay ❌ Don't Use When you need full accuracy (<0.001%)
PCA on large datasets When speed is NOT a concern

---

📊 Performance on 4096×4096 LLM Matrix

Method Time (s) Error Speedup
Full SVD 29.00 0% 1×
SPS-Simpson (balanced) 0.15 0.03% 193×
SPS-Simpson (accurate) 0.69 0.011% 42×
SPS-Simpson (fast) 0.13 0.29% 223×
Randomized SVD 0.42 0.70% 69×

---

💻 Usage Examples

Example 1: Quick Rank Estimation

```python
import torch
from sps_simpson import sps_simpson_svd

# Generate LLM-like matrix
n = 4096
U, _ = torch.linalg.qr(torch.randn(n, n))
V, _ = torch.linalg.qr(torch.randn(n, n))
s = torch.exp(-0.05 * torch.arange(n).float())
A = U @ torch.diag(s) @ V.T

# Compute approximate SVD
U, S, V, elapsed = sps_simpson_svd(A, speed_preset='balanced')

print(f"Rank: {len(S)}")
print(f"Time: {elapsed:.4f}s")
print(f"Top-5 singular values: {S[:5]}")
```

Example 2: Compression Ratio for LLM Layer

```python
def analyze_layer(weight_matrix):
    U, S, V, _ = sps_simpson_svd(weight_matrix, speed_preset='fast')
    rank = len(S)
    original_size = weight_matrix.shape[0] * weight_matrix.shape[1]
    compressed_size = rank * (weight_matrix.shape[0] + weight_matrix.shape[1])
    compression = 1 - compressed_size / original_size
    print(f"Optimal rank: {rank}")
    print(f"Compression: {compression:.1%}")
    return rank

# Analyze transformer layer
ffn_weight = torch.randn(4096, 11008)
analyze_layer(ffn_weight)
```

Example 3: Effective Rank with Error Control

```python
from sps_simpson import get_error

U, S, V, _ = sps_simpson_svd(A, speed_preset='accurate')
error = get_error(A, U, S, V)

if error < 0.001:  # 0.1% threshold
    print(f"Rank {len(S)} is sufficient (error: {error:.3%})")
```

Example 4: Comparison with Alternatives

```python
from sps_simpson import randomized_svd_fast

# SPS-Simpson
U1, S1, V1, t1 = sps_simpson_svd(A, speed_preset='balanced')
err1 = get_error(A, U1, S1, V1)

# Randomized SVD
U2, S2, V2, t2 = randomized_svd_fast(A, rank=100)
err2 = get_error(A, U2, S2, V2)

print(f"SPS: {t1:.3f}s, err={err1:.3%}")
print(f"Rand: {t2:.3f}s, err={err2:.3%}")
```

Example 5: Custom Spectrum Matrix

```python
def generate_power_law_matrix(n=4096, beta=2.0):
    """Generate matrix with power-law spectrum (recommender systems)"""
    U, _ = torch.linalg.qr(torch.randn(n, n))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    s = (torch.arange(1, n+1).float()) ** (-beta)
    s = s / s[0]
    return U @ torch.diag(s) @ V.T

A_power = generate_power_law_matrix(4096, beta=2.0)
U, S, V, t = sps_simpson_svd(A_power, speed_preset='balanced')
print(f"Power law rank: {len(S)}, time: {t:.3f}s")
```

---

⚙️ Speed Presets


Preset accurate Rank 500 Time (4096) 0.69s Error 0.011% Use Case Production compression

Preset balanced Rank 300 Time (4096) 0.31s Error 0.032% Default Use Case best trade-off

Preset fast Rank 150 Time (4096) 0.13s Error 0.29% Use Case Prototyping, exploration

```python
# Switch presets easily
U, S, V, t = sps_simpson_svd(A, speed_preset='fast')   # 0.13s
U, S, V, t = sps_simpson_svd(A, speed_preset='accurate') # 0.69s
```

---

🧪 Generate Test Matrix

```python
def generate_llm_matrix(n=4096, alpha=0.05):
    """
    Generate matrix with exponential spectrum (LLM-like)
    alpha: decay rate (0.05 = slow, 0.30 = fast)
    """
    U, _ = torch.linalg.qr(torch.randn(n, n))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    s = torch.exp(-alpha * torch.arange(n).float())
    return U @ torch.diag(s) @ V.T

# Test with different decay rates
for alpha in [0.05, 0.10, 0.20]:
    A = generate_llm_matrix(4096, alpha)
    U, S, V, t = sps_simpson_svd(A, speed_preset='balanced')
    print(f"α={alpha}: rank={len(S)}, time={t:.3f}s")
```

---

📈 Real-World Applications

Domain Use Case Benefit
LLM Compression Analyze layer compressibility 50-80% parameter reduction
Recommender Systems Select latent factors Optimal SVD rank
Graph Analysis Spectrum of adjacency matrix Understanding graph complexity
PCA on Big Data Estimate intrinsic dimension Hours → seconds

---

🔧 Requirements

· Python 3.8+
· PyTorch 1.9+
· NumPy

---

📚 API Reference

sps_simpson_svd(A, eps=1e-4, speed_preset='balanced')

Parameter Type Description
A torch.Tensor Input matrix (m, n)
eps float Target accuracy (default: 1e-4)
speed_preset str 'accurate', 'balanced', 'fast'

Returns: (U, S, V, elapsed_time)

get_error(A, U, S, V)

Compute relative Frobenius norm error: ||A - U·diag(S)·Vᵀ|| / ||A||

randomized_svd_fast(A, rank=100)

Fast randomized SVD baseline for comparison.

---

⚠️ Limitations

· Designed for exponential/power-law spectra (LLM, recommenders, graphs)
· Not suitable for random matrices (error >50%)
· Use full SVD or randomized SVD for flat spectra

---

📄 Citation

```bibtex
@software{sps_simpson_2025,
  author = {DimitryBur},
  title = {SPS-Simpson: Fast SVD for LLM Matrices},
  url = {https://github.com/DimitryBur/sps-simpson},
  year = {2025}
}
```

---

📝 License

MIT License — free for commercial and academic use.

---

🤝 Contributing

Issues and pull requests welcome!

---

⭐ Star the Repository

If this library helps your work, please star the repository on GitHub.

```

