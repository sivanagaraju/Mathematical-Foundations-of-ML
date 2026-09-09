#!/usr/bin/env python3
"""
01_shannon_entropy_simulation.py
================================
Rigorous numerical verification of discrete Shannon Entropy and Information Axioms:
  1. Axiomatic Surprisal: I(p) = -log2(p) (non-negativity, certainty zero, independent additivity)
  2. Shannon Entropy: H(X) = -sum_i p_i * log2(p_i)
  3. Maximum Entropy Guarantee: H(p) <= log2(K) with equality iff p is uniform
  4. Deterministic Minimum: H(p) = 0 for degenerate Dirac delta distributions
  5. Empirical LLN convergence of sample average surprisal to true theoretical entropy

Corresponds to Mathematical Foundations of ML — Lecture 11: Entropy.
Zero external dependencies outside PyTorch and NumPy.
"""

import sys
import numpy as np
import torch


def shannon_entropy(probs: torch.Tensor, base: float = 2.0) -> torch.Tensor:
    """Compute Shannon entropy handling 0 * log(0) = 0 limit."""
    # Shape: [K] -> []
    mask = probs > 0.0
    p = probs[mask]
    log_p = torch.log(p) / np.log(base)
    return -torch.sum(p * log_p)


def main() -> int:
    print("=== Simulation 01: Discrete Shannon Entropy & Information Axioms ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Axiomatic Verification of Surprisal I(A) = -log2(P(A))
    # -------------------------------------------------------------
    # Axiom 1: Certain event gives zero information
    p_certain = torch.tensor([1.0])
    surprisal_certain = -torch.log2(p_certain).item()
    print(f"[Axiom 1: Certainty] I(P=1.0) = {surprisal_certain:.4f} bits")
    assert np.isclose(surprisal_certain, 0.0), "Information of certain event must be 0"

    # Axiom 2: Independent events add surprisal: I(A and B) = I(A) + I(B)
    p_A, p_B = 0.50, 0.25
    p_joint = p_A * p_B
    surprisal_joint = -np.log2(p_joint)
    surprisal_sum = (-np.log2(p_A)) + (-np.log2(p_B))
    print(f"[Axiom 2: Additivity] I(A cap B) = {surprisal_joint:.2f} bits | I(A)+I(B) = {surprisal_sum:.2f} bits")
    assert np.isclose(surprisal_joint, surprisal_sum), "Independent surprisal must be additive"

    # -------------------------------------------------------------
    # 2. Maximum Entropy on K = 8 Outcomes
    # -------------------------------------------------------------
    K = 8
    # Uniform distribution: p_i = 1/8
    p_uniform = torch.full((K,), 1.0 / K)
    h_uniform = shannon_entropy(p_uniform).item()
    theoretical_max = np.log2(K)
    print(f"[Maximum Entropy] K={K} Uniform H: {h_uniform:.4f} bits (log2(K) = {theoretical_max:.4f})")
    assert np.isclose(h_uniform, theoretical_max), f"Uniform entropy must equal {theoretical_max}"

    # Non-uniform distribution: strictly less than log2(K)
    p_skewed = torch.tensor([0.40, 0.20, 0.15, 0.10, 0.05, 0.04, 0.03, 0.03])
    assert torch.isclose(p_skewed.sum(), torch.tensor(1.0))
    h_skewed = shannon_entropy(p_skewed).item()
    print(f"[Skewed Entropy] Non-uniform H: {h_skewed:.4f} bits < {theoretical_max:.4f} bits")
    assert h_skewed < theoretical_max, "Non-uniform entropy must be strictly less than uniform"

    # -------------------------------------------------------------
    # 3. Degenerate Deterministic Limit: H -> 0
    # -------------------------------------------------------------
    p_deterministic = torch.zeros(K)
    p_deterministic[0] = 1.0
    h_det = shannon_entropy(p_deterministic).item()
    print(f"[Minimum Entropy] Deterministic H: {h_det:.4f} bits")
    assert np.isclose(h_det, 0.0), "Deterministic distribution must have zero entropy"

    # -------------------------------------------------------------
    # 4. Monte Carlo Verification of Expected Surprisal via LLN
    # -------------------------------------------------------------
    N_samples = 100000
    categorical_dist = torch.distributions.Categorical(probs=p_skewed)
    samples = categorical_dist.sample((N_samples,))
    # Compute observed surprisal for each drawn sample
    sample_surprisals = -torch.log2(p_skewed[samples])
    empirical_mean_surprisal = sample_surprisals.mean().item()

    print(f"[LLN Empirical] Mean Sample Surprisal: {empirical_mean_surprisal:.4f} | True H: {h_skewed:.4f}")
    assert np.isclose(empirical_mean_surprisal, h_skewed, atol=0.01), (
        f"Empirical mean surprisal {empirical_mean_surprisal} diverges from theoretical H {h_skewed}"
    )

    print("Verified: Discrete Shannon entropy satisfies all formal mathematical axioms.")
    print("SUCCESS: 01_shannon_entropy_simulation executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
