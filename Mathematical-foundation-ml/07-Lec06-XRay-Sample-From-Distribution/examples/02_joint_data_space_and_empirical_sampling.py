"""
02_joint_data_space_and_empirical_sampling.py
=============================================
Lecture 06 Simulation: Joint Data Space & Empirical Sampling

Simulates drawing a supervised training dataset D = {(x_i, y_i)}_{i=1}^N from joint
probability measure P(X, Y) on R^d x {0, 1}.

Mathematical Invariant:
    By the Law of Large Numbers, the empirical mean vector (1/N) sum_i x_i
    converges almost surely to the population marginal expectation:
    E[X] = P(Y=0) * E[X | Y=0] + P(Y=1) * E[X | Y=1].
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 02: Joint Data Space & Empirical Sampling ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Setup joint population parameters
    d = 16
    p_Y0 = 0.70
    p_Y1 = 0.30

    mu_0 = torch.ones(d) * 0.3
    mu_1 = torch.ones(d) * 0.8
    sigma = 0.20

    # Theoretical population expectation: E[X] = p_0 * mu_0 + p_1 * mu_1
    expected_pop_mean = p_Y0 * mu_0 + p_Y1 * mu_1

    # 2. Draw synthetic dataset D of size N = 20,000
    N = 20000
    # Sample binary labels Y_i ~ Bernoulli(p_Y1)
    labels = (torch.rand(N) < p_Y1).long()  # 0 or 1

    # Sample feature vectors X_i conditioned on Y_i
    features = torch.zeros(N, d, dtype=torch.float32)
    mask_0 = (labels == 0)
    mask_1 = (labels == 1)

    n_0 = int(mask_0.sum().item())
    n_1 = int(mask_1.sum().item())

    features[mask_0] = mu_0 + sigma * torch.randn(n_0, d)
    features[mask_1] = mu_1 + sigma * torch.randn(n_1, d)

    print(f"Generated dataset D with N = {N} paired samples.")
    print(f"Class 0 (Healthy) count:   {n_0} ({n_0/N*100:.1f}%)")
    print(f"Class 1 (Pneumonia) count: {n_1} ({n_1/N*100:.1f}%)")

    # 3. Compute empirical statistics
    emp_mean = torch.mean(features, dim=0)
    emp_mean_0 = torch.mean(features[mask_0], dim=0)
    emp_mean_1 = torch.mean(features[mask_1], dim=0)

    # Verify Law of Large Numbers on class means
    assert torch.allclose(emp_mean_0, mu_0, atol=0.02), "Class 0 mean must converge"
    assert torch.allclose(emp_mean_1, mu_1, atol=0.02), "Class 1 mean must converge"

    # Verify Law of Total Expectation on total dataset mean
    mean_diff = torch.norm(emp_mean - expected_pop_mean).item()
    print(f"Population E[X] (first 3): {expected_pop_mean[:3].tolist()}")
    print(f"Empirical mean  (first 3): {emp_mean[:3].tolist()}")
    print(f"L2 Norm Discrepancy:       {mean_diff:.5f}")

    assert torch.allclose(emp_mean, expected_pop_mean, atol=0.02), (
        f"Empirical mean must converge to theoretical E[X], diff={mean_diff}"
    )

    print("SUCCESS: 02_joint_data_space_and_empirical_sampling executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
