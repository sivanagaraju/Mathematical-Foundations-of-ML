"""
Simulation 1: Matrix Representation & Linear Subspace Projections (Q, K, V)
Course: Mathematical Foundations of Machine Learning
Lecture 48: Attention Part 1

Description:
This script simulates the foundational token projection mechanism in attention
architectures. Given an input sequence matrix X in R^{T x D}, it applies three
parameterized linear transformations W_Q, W_K, W_V to generate Query, Key,
and Value representations. It numerically proves row-wise token independence
and asserts geometric invariance under permutation equivariance.

All assertions are verified with torch.allclose. Clean exit code 0.
"""

import torch
import torch.nn as nn


def test_linear_projections():
    print("=" * 70)
    print("STEP 1: Initializing Sequence Tensor & Projection Operators")
    print("=" * 70)

    # Set deterministic seed for reproducibility
    torch.manual_seed(42)

    T = 4       # Sequence length (4 tokens)
    D = 8       # Input embedding dimension
    D_k = 6     # Projected Query/Key/Value subspace dimension

    # Sequence matrix X: shape (T, D)
    # Each row X[i, :] is the D-dimensional vector of token i
    X = torch.randn(T, D, dtype=torch.float64)

    # Learnable projection matrices W_Q, W_K, W_V in R^{D x D_k}
    W_Q = torch.randn(D, D_k, dtype=torch.float64, requires_grad=True)
    W_K = torch.randn(D, D_k, dtype=torch.float64, requires_grad=True)
    W_V = torch.randn(D, D_k, dtype=torch.float64, requires_grad=True)

    print(f"Input Matrix X shape:      {X.shape} (T={T}, D={D})")
    print(f"Weight Matrix W_Q shape:   {W_Q.shape} (D={D}, D_k={D_k})")
    print(f"Weight Matrix W_K shape:   {W_K.shape} (D={D}, D_k={D_k})")
    print(f"Weight Matrix W_V shape:   {W_V.shape} (D={D}, D_k={D_k})")

    print("\n" + "=" * 70)
    print("STEP 2: Computing Projections via Matrix Multiplication")
    print("=" * 70)

    # Matrix multiplication: Q = X W_Q, K = X W_K, V = X W_V
    Q = torch.matmul(X, W_Q)
    K = torch.matmul(X, W_K)
    V = torch.matmul(X, W_V)

    assert Q.shape == (T, D_k), f"Q shape mismatch: expected {(T, D_k)}, got {Q.shape}"
    assert K.shape == (T, D_k), f"K shape mismatch: expected {(T, D_k)}, got {K.shape}"
    assert V.shape == (T, D_k), f"V shape mismatch: expected {(T, D_k)}, got {V.shape}"
    print(f"Computed Q shape: {Q.shape}")
    print(f"Computed K shape: {K.shape}")
    print(f"Computed V shape: {V.shape}")

    print("\n" + "=" * 70)
    print("STEP 3: Proving Row-Wise Token Independence")
    print("=" * 70)

    # Prove that the i-th row of Q is strictly the projection of the i-th token:
    # Q[i, :] == X[i, :] @ W_Q
    for i in range(T):
        token_i = X[i:i+1, :]  # Shape (1, D)
        proj_i = torch.matmul(token_i, W_Q)  # Shape (1, D_k)
        assert torch.allclose(Q[i:i+1, :], proj_i, atol=1e-12), (
            f"Row {i} token projection mismatch!"
        )
    print("Confirmed: Every row of Q is an isolated subspace projection of that token.")

    print("\n" + "=" * 70)
    print("STEP 4: Verifying Permutation Equivariance")
    print("=" * 70)

    # Create an arbitrary permutation of token positions
    perm = torch.tensor([2, 0, 3, 1])
    X_permuted = X[perm, :]

    # Project the permuted matrix
    Q_from_permuted = torch.matmul(X_permuted, W_Q)
    # Permute the original projected rows
    Q_permuted_directly = Q[perm, :]

    # Invariance check: (Pi @ X) @ W_Q == Pi @ (X @ W_Q)
    assert torch.allclose(Q_from_permuted, Q_permuted_directly, atol=1e-12), (
        "Permutation equivariance violated!"
    )
    print("Confirmed: Linear projections commute with token permutation: Pi(X) W == Pi(X W).")

    print("\n[SUCCESS] Simulation 01: All assertions passed cleanly.")


if __name__ == "__main__":
    test_linear_projections()
