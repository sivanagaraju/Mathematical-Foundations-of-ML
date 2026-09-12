#!/usr/bin/env python3
"""
File: 01_numerical_verification.py
Package: 20-Lec09-VAEs-Part1
Lecture Mapping: Topic 1 (00:00 - 04:23), Topic 2 (04:23 - 06:17), & Topic 8 (22:02 - 32:56) in NOTES.md
Mathematical Theorems & Concepts Verified:
    1. The Exact ELBO Identity & Jensen's Inequality: log p(x) == ELBO + KL(q(z|x) || p(z|x))
    2. Analytical Closed-Form Diagonal Gaussian KL Divergence vs PyTorch Distribution KL
    3. The Probability Integral Transform (PIT) & Inverse CDF Sampling Equivalence
Prerequisites & Foundations:
    - PREREQUISITES.md#p3-jensens-inequality
    - PREREQUISITES.md#p4-kl-divergence
    - PREREQUISITES.md#p5-elbo-decomposition
    - MathsTerms/06-Deep-Architectures-and-Generative-Models/07-ELBO_and_Variational_Inference.md
    - MathsTerms/05-Information-Theory-and-Divergences/02-KL_Divergence.md
Hardware Requirements:
    - Pure CPU execution (device = torch.device('cpu'))
    - Deterministic execution with fixed seeds (seed = 42)
    - Dependencies: python >= 3.9, numpy >= 1.22, torch >= 2.0
Execution:
    python examples/01_numerical_verification.py
Expected Result:
    All assertion checks pass; prints verification summary; exits with code 0.
"""

from __future__ import annotations
import sys
import torch
import numpy as np


def verify_elbo_identity() -> None:
    """Theorem 1: Prove the exact ELBO decomposition on a discrete latent model."""
    print("=" * 70)
    print("Verifying Theorem 1: Evidence Lower Bound (ELBO) Decomposition Identity")
    print("=" * 70)

    # Consider a discrete observation x_0 with 4 latent states z in {1, 2, 3, 4}
    # Unnormalized joint probabilities p(x_0, z)
    p_joint = torch.tensor([0.08, 0.24, 0.40, 0.08], dtype=torch.float64)
    # True marginal evidence: p(x_0) = sum_z p(x_0, z) = 0.80
    p_evidence = torch.sum(p_joint)
    true_log_evidence = torch.log(p_evidence)

    # True Bayes posterior: p(z | x_0) = p(x_0, z) / p(x_0)
    p_true_posterior = p_joint / p_evidence

    # Arbitrary variational distribution q(z | x_0) (proposing a sub-optimal posterior)
    q_variational = torch.tensor([0.15, 0.35, 0.30, 0.20], dtype=torch.float64)

    # Path A: Direct manual ELBO computation via Jensen's expectation
    # ELBO = E_q [ log( p(x, z) / q(z) ) ]
    manual_elbo = torch.sum(q_variational * torch.log(p_joint / q_variational))

    # Path B: Two-term ELBO breakdown: E_q[log p(x|z)] - KL(q(z) || p(z))
    # Prior p(z) = [0.25, 0.25, 0.25, 0.25]
    p_prior = torch.full_like(q_variational, 0.25)
    p_conditional_x_given_z = p_joint / p_prior  # p(x,z) / p(z)
    reconstruction_term = torch.sum(q_variational * torch.log(p_conditional_x_given_z))
    kl_to_prior = torch.sum(q_variational * torch.log(q_variational / p_prior))
    two_term_elbo = reconstruction_term - kl_to_prior

    # KL gap between variational posterior and true posterior
    kl_gap = torch.sum(q_variational * torch.log(q_variational / p_true_posterior))

    # Assertions
    diff_elbo_forms = torch.abs(manual_elbo - two_term_elbo).item()
    diff_decomposition = torch.abs(true_log_evidence - (manual_elbo + kl_gap)).item()

    assert torch.allclose(manual_elbo, two_term_elbo, atol=1e-6), (
        f"ELBO representations do not match! diff={diff_elbo_forms:.6e}"
    )
    assert torch.allclose(true_log_evidence, manual_elbo + kl_gap, atol=1e-6), (
        f"ELBO decomposition violated! true_log={true_log_evidence:.6f}, "
        f"ELBO+KL={manual_elbo + kl_gap:.6f}, diff={diff_decomposition:.6e}"
    )
    assert manual_elbo.item() <= true_log_evidence.item(), (
        f"ELBO ({manual_elbo.item():.4f}) must be a lower bound on log evidence ({true_log_evidence.item():.4f})!"
    )
    assert kl_gap.item() >= 0.0, f"KL divergence gap must be non-negative! got {kl_gap.item()}"

    print(f"  True Log-Evidence log p(x):     {true_log_evidence.item():.6f}")
    print(f"  Manual ELBO (Jensen bound):     {manual_elbo.item():.6f}")
    print(f"  Two-Term ELBO (Recon - KL):     {two_term_elbo.item():.6f}")
    print(f"  Variational Gap KL(q || p_post): {kl_gap.item():.6f}")
    print(f"  [PASS] Bound Verification:       ELBO <= log p(x) ({manual_elbo.item():.4f} <= {true_log_evidence.item():.4f})")
    print(f"  [PASS] Decomposition Equality:   diff = {diff_decomposition:.2e} < 1e-6\n")


def verify_gaussian_kl_divergence() -> None:
    """Theorem 2: Prove analytical diagonal Gaussian KL divergence matches PyTorch distributions."""
    print("=" * 70)
    print("Verifying Theorem 2: Closed-Form Gaussian KL Divergence Equivalence")
    print("=" * 70)

    # Batch of 4 latent vectors in K = 3 dimensions
    batch_size, k_dim = 4, 3
    mu = torch.tensor([
        [0.5, -1.2, 0.0],
        [1.8,  0.4, -0.7],
        [-0.9, -0.2, 1.1],
        [0.1,  2.0, -1.5]
    ], dtype=torch.float32)

    logvar = torch.tensor([
        [-0.5,  0.2, -1.0],
        [ 0.8, -0.4,  0.1],
        [-1.2,  0.6, -0.3],
        [ 0.0, -0.8,  0.5]
    ], dtype=torch.float32)

    sigma = torch.exp(0.5 * logvar)

    # Path A: Manual first-principles closed-form formula
    # KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_manual_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_manual_total = torch.sum(kl_manual_per_sample)

    # Path B: PyTorch native distributions implementation
    q_dist = torch.distributions.Normal(loc=mu, scale=sigma)
    p_dist = torch.distributions.Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(sigma))
    kl_torch_per_sample = torch.distributions.kl_divergence(q_dist, p_dist).sum(dim=1)
    kl_torch_total = torch.sum(kl_torch_per_sample)

    # Assertions
    max_diff_sample = torch.max(torch.abs(kl_manual_per_sample - kl_torch_per_sample)).item()
    diff_total = torch.abs(kl_manual_total - kl_torch_total).item()

    assert torch.allclose(kl_manual_per_sample, kl_torch_per_sample, atol=1e-5), (
        f"Per-sample KL mismatch! max_diff={max_diff_sample:.6e}"
    )
    assert torch.allclose(kl_manual_total, kl_torch_total, atol=1e-5), (
        f"Total KL mismatch! diff={diff_total:.6e}"
    )
    assert (kl_manual_per_sample >= 0.0).all(), "KL divergence must be non-negative everywhere!"

    print(f"  Latent Batch Shape:              {mu.shape} (B={batch_size}, K={k_dim})")
    print(f"  Manual Closed-Form KL (Total):   {kl_manual_total.item():.6f}")
    print(f"  PyTorch Native KL (Total):       {kl_torch_total.item():.6f}")
    print(f"  Per-Sample KL Values:            {kl_manual_per_sample.numpy()}")
    print(f"  [PASS] Sample Max Discrepancy:   {max_diff_sample:.2e} < 1e-5")
    print(f"  [PASS] Total Sum Equivalence:    {diff_total:.2e} < 1e-5\n")


def verify_inverse_cdf_sampling() -> None:
    """Theorem 3: Prove Probability Integral Transform & Inverse CDF Sampling Equivalence."""
    print("=" * 70)
    print("Verifying Theorem 3: Probability Integral Transform & Inverse CDF Sampling")
    print("=" * 70)

    # Target: Exponential distribution with rate lambda = 1.5
    # PDF: f(x) = lambda * exp(-lambda * x) for x >= 0
    # CDF: F(x) = 1 - exp(-lambda * x)
    # Inverse CDF: F^{-1}(u) = - (1 / lambda) * log(1 - u)
    rate = 1.5
    n_samples = 150_000

    # Path A: Draw uniform variates u ~ Uniform(0, 1) and map through Inverse CDF
    u = np.random.uniform(0.0, 1.0, size=n_samples)
    x_inverse_cdf = - (1.0 / rate) * np.log(1.0 - u)

    # Path B: Direct standard library exponential sampling
    x_scipy = np.random.exponential(scale=1.0 / rate, size=n_samples)

    # 1. Theoretical moments for Exponential(rate)
    theoretical_mean = 1.0 / rate        # 0.6667
    theoretical_var = 1.0 / (rate**2)     # 0.4444

    # 2. Empirical moments
    mean_inv = np.mean(x_inverse_cdf)
    var_inv = np.var(x_inverse_cdf)
    mean_ref = np.mean(x_scipy)
    var_ref = np.var(x_scipy)

    # Assertions
    assert np.isclose(mean_inv, theoretical_mean, atol=1e-2), (
        f"Inverse CDF mean mismatch: {mean_inv:.4f} vs {theoretical_mean:.4f}"
    )
    assert np.isclose(var_inv, theoretical_var, atol=1.5e-2), (
        f"Inverse CDF variance mismatch: {var_inv:.4f} vs {theoretical_var:.4f}"
    )
    assert np.isclose(mean_inv, mean_ref, atol=1.5e-2), (
        f"Inverse CDF vs Native mean discrepancy: {abs(mean_inv - mean_ref):.4f}"
    )

    # 3. Verify the Probability Integral Transform directly: F(X) ~ Uniform(0, 1)
    # Feed the generated x back into its CDF
    u_reconstructed = 1.0 - np.exp(-rate * x_inverse_cdf)
    mean_u = np.mean(u_reconstructed)  # Uniform(0, 1) mean should be 0.5
    var_u = np.var(u_reconstructed)    # Uniform(0, 1) variance should be 1/12 ≈ 0.08333

    assert np.isclose(mean_u, 0.5, atol=5e-3), f"PIT mean failed: {mean_u:.4f}"
    assert np.isclose(var_u, 1.0 / 12.0, atol=5e-3), f"PIT variance failed: {var_u:.4f}"

    print(f"  Target Distribution:             Exponential(lambda={rate})")
    print(f"  Theoretical Mean / Variance:     {theoretical_mean:.4f} / {theoretical_var:.4f}")
    print(f"  Inverse CDF Mean / Variance:     {mean_inv:.4f} / {var_inv:.4f}")
    print(f"  Reference Sampler Mean / Var:    {mean_ref:.4f} / {var_ref:.4f}")
    print(f"  PIT Reconstructed U Mean / Var:  {mean_u:.4f} / {var_u:.4f} (Expected 0.5000 / 0.0833)")
    print(f"  [PASS] Inverse CDF Mean Match:   abs_err = {abs(mean_inv - theoretical_mean):.2e} < 1e-2")
    print(f"  [PASS] Probability Integral PIT: abs_err = {abs(mean_u - 0.5):.2e} < 5e-3\n")


def main() -> None:
    print("=" * 70)
    print("RUNNING VAE NUMERICAL VERIFICATION SUITE (CPU DETERMINISTIC)")
    print("=" * 70 + "\n")

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        verify_elbo_identity()
        verify_gaussian_kl_divergence()
        verify_inverse_cdf_sampling()
    except AssertionError as err:
        print(f"\n[FATAL ERROR] Numerical Verification FAILED: {err}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("SUCCESS: ALL 3 VAE MATHEMATICAL THEOREMS VERIFIED CLEANLY (EXIT CODE 0)")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
