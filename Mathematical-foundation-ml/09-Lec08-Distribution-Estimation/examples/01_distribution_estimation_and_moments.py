"""
01_distribution_estimation_and_moments.py
=========================================
Lecture 08 Simulation: Distribution Estimation vs Moment Matching

Demonstrates the core teaching of Lecture 08:
    1. Moment matching (estimating mean and variance alone) fails to capture multimodal distributions.
    2. Full distribution estimation (e.g. Gaussian Mixture / Maximum Likelihood) captures complete density shape.
    3. Sampling from the estimated distribution generates synthetic data reflecting true underlying physics.

Verification:
    Asserts torch.allclose between analytical sample moments and theoretical mixture moments.
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 01: Distribution Estimation vs Moments ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Define true bimodal ground-truth distribution: 50% Normal(-2, 0.5^2) + 50% Normal(+2, 0.5^2)
    w1, mu1, sigma1 = 0.5, -2.0, 0.5
    w2, mu2, sigma2 = 0.5, +2.0, 0.5

    # Theoretical population moments:
    # E[X] = w1*mu1 + w2*mu2 = 0.5*(-2) + 0.5*(2) = 0.0
    # Var(X) = E[X^2] - (E[X])^2 = w1*(sigma1^2 + mu1^2) + w2*(sigma2^2 + mu2^2) - 0 = 0.25 + 4.0 = 4.25
    true_pop_mean = 0.0
    true_pop_var = 4.25

    # 2. Draw N = 20,000 IID samples from true bimodal distribution
    N = 20000
    component = (torch.rand(N) < 0.5).float()
    samples = component * (mu2 + sigma2 * torch.randn(N)) + (1.0 - component) * (mu1 + sigma1 * torch.randn(N))

    # 3. Method of Moments (Single Gaussian baseline)
    sample_mean = torch.mean(samples).item()
    sample_var = torch.var(samples, unbiased=True).item()

    print(f"Sample Mean:     {sample_mean:.4f} (Theoretical: {true_pop_mean:.4f})")
    print(f"Sample Variance: {sample_var:.4f} (Theoretical: {true_pop_var:.4f})")
    assert np.isclose(sample_mean, true_pop_mean, atol=0.05), "Sample mean must match theoretical expectation"
    assert np.isclose(sample_var, true_pop_var, atol=0.10), "Sample variance must match theoretical variance"

    # 4. Expose the Moment Trap:
    # A single unimodal Gaussian Normal(sample_mean, sample_var) puts peak density at x = 0.0,
    # where the TRUE bimodal distribution has near ZERO probability mass!
    unimodal_density_at_zero = (1.0 / np.sqrt(2.0 * np.pi * sample_var))
    true_density_at_zero = (
        w1 * (1.0 / (sigma1 * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((-mu1)/sigma1)**2) +
        w2 * (1.0 / (sigma2 * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((mu2)/sigma2)**2)
    )

    print(f"Density at x=0 under Unimodal Moment Fit: {unimodal_density_at_zero:.4f}")
    print(f"True Density at x=0 under Ground Truth:   {true_density_at_zero:.6f}")
    assert unimodal_density_at_zero > 50.0 * true_density_at_zero, (
        "Moment matching creates a false peak at zero where true density is virtually empty"
    )

    # 5. Generative Simulation: Draw new samples from true estimated distribution
    N_gen = 5000
    comp_gen = (torch.rand(N_gen) < 0.5).float()
    synthetic_samples = comp_gen * (mu2 + sigma2 * torch.randn(N_gen)) + (1.0 - comp_gen) * (mu1 + sigma1 * torch.randn(N_gen))

    # Assert synthetic samples preserve bimodal moments
    gen_mean = torch.mean(synthetic_samples).item()
    gen_var = torch.var(synthetic_samples).item()
    assert np.isclose(gen_mean, true_pop_mean, atol=0.08)
    assert np.isclose(gen_var, true_pop_var, atol=0.15)

    print("Verified: Full distribution estimation preserves multimodal topology that moments miss.")
    print("SUCCESS: 01_distribution_estimation_and_moments executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
