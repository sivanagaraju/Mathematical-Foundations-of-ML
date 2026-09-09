#!/usr/bin/env python3
"""
02_empirical_vs_true_distribution_distance.py
============================================
Mathematical simulation demonstrating the fundamental challenge of ML:
  Empirical Risk Minimization (ERM) on finite sample D_N vs Population Risk.
Evaluates:
  1. Population Risk: R(theta) = E_{x ~ p}[ loss(x; theta) ]
  2. Empirical Risk: R_hat_N(theta) = 1/N sum_{i=1}^N loss(x_i; theta)
  3. Generalization Error gap: |R_hat_N(theta) - R(theta)| ~ O(1 / sqrt(N))

Corresponds to Mathematical Foundations of ML — Lecture 10: Challenges of ML.
Zero external dependencies outside PyTorch and NumPy.
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 02: Empirical Risk vs Population Distance ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Ground Truth Population p: Standard Normal N(0, 1)
    # -------------------------------------------------------------
    # Loss function: Quadratic error to candidate parameter theta
    # loss(x; theta) = (x - theta)^2
    # Population Risk: R(theta) = E[(X - theta)^2] = E[X^2] - 2*theta*E[X] + theta^2
    # Since X ~ N(0, 1), E[X] = 0, E[X^2] = 1, so:
    # R(theta) = 1.0 + theta^2
    theta_test = 1.50
    theoretical_population_risk = 1.0 + theta_test**2  # 1.0 + 2.25 = 3.25
    print(f"[Population Risk] True analytical R(theta={theta_test}) = {theoretical_population_risk:.4f}")

    # -------------------------------------------------------------
    # 2. Empirical Risk Convergence as N -> Infinity
    # -------------------------------------------------------------
    sample_sizes = [20, 100, 1000, 10000, 100000]
    gaps = []

    print(f"{'N Samples':>10} | {'Empirical Risk':>15} | {'Risk Gap |R_hat - R|':>22}")
    print("-" * 55)

    for n in sample_sizes:
        samples = torch.randn(n)
        empirical_risk = torch.mean((samples - theta_test) ** 2).item()
        gap = abs(empirical_risk - theoretical_population_risk)
        gaps.append(gap)
        print(f"{n:>10} | {empirical_risk:>15.5f} | {gap:>22.5f}")

    # -------------------------------------------------------------
    # 3. Assert Law of Large Numbers (LLN) Convergence Rate
    # -------------------------------------------------------------
    # For large N = 100,000, empirical risk must match population risk within 0.02
    assert gaps[-1] < 0.02, f"Empirical risk at N=100k ({gaps[-1]}) exceeded tolerance 0.02"

    # Monotonic trend check: risk gap at N=100,000 should be significantly smaller than at N=20
    assert gaps[-1] < gaps[0], "Large sample empirical risk must have tighter gap than small sample"

    # -------------------------------------------------------------
    # 4. Empirical Argmin vs True Argmin
    # -------------------------------------------------------------
    # Argmin of Population Risk: theta* = argmin (1 + theta^2) = 0.0
    # Argmin of Empirical Risk: theta_hat_N = sample mean x_bar
    small_samples = torch.randn(25)
    large_samples = torch.randn(50000)

    theta_hat_small = small_samples.mean().item()
    theta_hat_large = large_samples.mean().item()

    print(f"[Argmin Estimation] Small N=25: theta_hat = {theta_hat_small:.4f} (Error: {abs(theta_hat_small):.4f})")
    print(f"[Argmin Estimation] Large N=50k: theta_hat = {theta_hat_large:.4f} (Error: {abs(theta_hat_large):.4f})")

    assert abs(theta_hat_large) < 0.015, f"Large sample argmin error ({abs(theta_hat_large)}) must be < 0.015"

    print("Verified: Empirical risk strictly converges to population risk under LLN.")
    print("SUCCESS: 02_empirical_vs_true_distribution_distance executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
