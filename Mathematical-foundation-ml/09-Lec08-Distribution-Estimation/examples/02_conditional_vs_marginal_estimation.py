"""
02_conditional_vs_marginal_estimation.py
========================================
Lecture 08 Simulation: Conditional vs Marginal Distribution Estimation

Demonstrates how estimating the joint distribution P(X, Y) solves all three major
downstream machine learning paradigms:
    1. Supervised Classification: P(Y | X)
    2. Class-Conditional Generation (Inpainting): P(X | Y)
    3. Unsupervised Outlier Detection: P(X) = sum_y P(X, y)

Verification:
    Asserts consistency of marginal and conditional distributions via Bayes' rule.
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 02: Conditional vs Marginal Distribution Estimation ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Ground truth generative model:
    # Y in {0, 1} with prior P(Y=0) = 0.70, P(Y=1) = 0.30
    # X | Y=0 ~ Normal(mu0=0.0, sigma0=1.0)
    # 1. Ground Truth Mixture: P(Y=1) = 0.3, P(Y=0) = 0.7
    torch.manual_seed(42)
    np.random.seed(42)
    p_Y0, p_Y1 = 0.70, 0.30
    mu0, mu1 = 0.0, 3.0
    sigma = 1.0

    # 2. Draw N = 100,000 paired training samples D = {(x_i, y_i)}
    N = 100000
    y_train = (torch.rand(N) < p_Y1).long()
    x_train = torch.zeros(N)
    x_train[y_train == 0] = mu0 + sigma * torch.randn((y_train == 0).sum())
    x_train[y_train == 1] = mu1 + sigma * torch.randn((y_train == 1).sum())

    # Query 1: Unsupervised Marginal Density P(X = x)
    # p(x) = p(x | Y=0)*P(Y=0) + p(x | Y=1)*P(Y=1)
    test_x = torch.tensor([1.5])  # Halfway between modes
    p_x_given_0 = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((test_x.item() - mu0)/sigma)**2)
    p_x_given_1 = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((test_x.item() - mu1)/sigma)**2)
    analytical_marginal_px = p_x_given_0 * p_Y0 + p_x_given_1 * p_Y1

    print(f"Analytical Marginal Density p(X=1.5): {analytical_marginal_px:.4f}")

    # Query 2: Supervised Discriminative Posterior P(Y=1 | X=x)
    analytical_posterior_Y1 = (p_x_given_1 * p_Y1) / analytical_marginal_px
    analytical_posterior_Y0 = (p_x_given_0 * p_Y0) / analytical_marginal_px
    print(f"Supervised Classification Posterior P(Y=1 | X=1.5): {analytical_posterior_Y1:.4f}")
    print(f"Supervised Classification Posterior P(Y=0 | X=1.5): {analytical_posterior_Y0:.4f}")

    # Empirical slice validation around x = 1.5 +/- 0.1
    eps = 0.10
    slice_mask = (x_train >= 1.5 - eps) & (x_train <= 1.5 + eps)
    slice_labels = y_train[slice_mask]
    emp_posterior_Y1 = (slice_labels == 1).float().mean().item()
    print(f"Empirical Slice Posterior P(Y=1 | X in [1.4, 1.6]):  {emp_posterior_Y1:.4f}")

    assert np.isclose(emp_posterior_Y1, analytical_posterior_Y1, atol=0.04), (
        f"Empirical posterior {emp_posterior_Y1} must match analytical Bayes {analytical_posterior_Y1}"
    )

    # Query 3: Class-Conditional Generation P(X | Y=1)
    # Drawing 1000 samples conditioned on disease Y=1
    synthetic_x_Y1 = mu1 + sigma * torch.randn(1000)
    print(f"Generated X conditioned on Y=1 mean: {synthetic_x_Y1.mean().item():.4f} (True: {mu1})")
    assert np.isclose(synthetic_x_Y1.mean().item(), mu1, atol=0.08)

    print("Verified: Estimating the underlying distribution solves marginal, conditional, and generative queries.")
    print("SUCCESS: 02_conditional_vs_marginal_estimation executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
