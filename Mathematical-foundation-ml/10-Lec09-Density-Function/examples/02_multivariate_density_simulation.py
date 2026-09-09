#!/usr/bin/env python3
"""
02_multivariate_density_simulation.py
====================================
Rigorous numerical verification of multivariate probability density functions (PDFs):
1. 2D Joint Density integration: int int f_{X_1, X_2}(x_1, x_2) dx_1 dx_2 = 1.0
2. Marginalization via integration: f_{X_1}(x_1) = int f_{X_1, X_2}(x_1, x_2) dx_2
3. Rectangular Volume Probability: P(A) = int_{a_1}^{b_1} int_{a_2}^{b_2} f(x_1, x_2) dx_2 dx_1
4. Monte Carlo vs Numerical Riemann 2D integration comparison

Corresponds to Mathematical Foundations of ML — Lecture 09: Density Function.
Zero external dependencies outside PyTorch and NumPy.
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 02: Multivariate Continuous Density Functions ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Bivariate Gaussian Definition
    # -------------------------------------------------------------
    # Mean vector mu = [0, 0]
    # Covariance Sigma = [[1.0, 0.6], [0.6, 1.0]]
    # Correlation rho = 0.6
    rho = 0.6
    det_Sigma = 1.0 - rho**2
    inv_Sigma = np.array([[1.0 / det_Sigma, -rho / det_Sigma],
                          [-rho / det_Sigma, 1.0 / det_Sigma]])
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_Sigma))

    def joint_pdf(x1, x2):
        # Quadratic form: x^T Sigma^-1 x
        q = inv_Sigma[0, 0] * x1**2 + 2 * inv_Sigma[0, 1] * x1 * x2 + inv_Sigma[1, 1] * x2**2
        return norm_const * np.exp(-0.5 * q)

    # -------------------------------------------------------------
    # 2. 2D Grid Integration over [-4, 4] x [-4, 4]
    # -------------------------------------------------------------
    N_grid = 401
    coords = np.linspace(-4.0, 4.0, N_grid)
    dx = coords[1] - coords[0]
    X1, X2 = np.meshgrid(coords, coords)
    Z = joint_pdf(X1, X2)

    total_2d_integral = np.sum(Z) * (dx ** 2)
    print(f"[Total Volume] 2D Riemann integral of joint PDF over [-4, 4]^2 = {total_2d_integral:.6f}")
    assert np.isclose(total_2d_integral, 1.0, atol=1e-3), (
        f"Joint PDF volume integral must equal 1.0, got {total_2d_integral}"
    )

    # -------------------------------------------------------------
    # 3. Marginal Density Extraction via Numerical Slicing
    # -------------------------------------------------------------
    # Integrating out X2 for a fixed test point x1 = 0.5
    # Theoretical marginal: X1 ~ Normal(0, 1) -> f_X1(0.5) = (1/sqrt(2pi)) * exp(-0.5 * 0.5^2)
    test_x1 = 0.5
    theoretical_marginal = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * test_x1**2)

    # Numerical marginalization: sum_x2 f(test_x1, x2) * dx
    marginal_slice = joint_pdf(test_x1, coords)
    numerical_marginal = np.sum(marginal_slice) * dx

    print(f"[Marginal Extraction] at x1={test_x1}: Analytical: {theoretical_marginal:.6f} | Numerical: {numerical_marginal:.6f}")
    assert np.isclose(numerical_marginal, theoretical_marginal, atol=1e-3), (
        f"Marginal integral {numerical_marginal} diverges from theoretical Gaussian {theoretical_marginal}"
    )

    # -------------------------------------------------------------
    # 4. Rectangular Event Probability: P(0.0 <= X1 <= 1.0, 0.0 <= X2 <= 1.0)
    # -------------------------------------------------------------
    sub_mask = (X1 >= 0.0) & (X1 <= 1.0) & (X2 >= 0.0) & (X2 <= 1.0)
    prob_rect_numerical = np.sum(Z[sub_mask]) * (dx ** 2)

    # Monte Carlo estimation using PyTorch Multivariate Normal
    mean_torch = torch.tensor([0.0, 0.0])
    cov_torch = torch.tensor([[1.0, rho], [rho, 1.0]])
    L = torch.linalg.cholesky(cov_torch)
    std_samples = torch.randn(100000, 2)
    joint_samples = std_samples @ L.T

    mc_rect_mask = (joint_samples[:, 0] >= 0.0) & (joint_samples[:, 0] <= 1.0) & \
                   (joint_samples[:, 1] >= 0.0) & (joint_samples[:, 1] <= 1.0)
    prob_rect_mc = mc_rect_mask.float().mean().item()

    print(f"[Rectangular Probability] 2D Numerical Integral: {prob_rect_numerical:.4f} | Monte Carlo: {prob_rect_mc:.4f}")
    assert np.isclose(prob_rect_mc, prob_rect_numerical, atol=0.01), (
        f"Monte Carlo rectangle prob {prob_rect_mc} diverges from numerical {prob_rect_numerical}"
    )

    print("Verified: Multivariate densities integrate to 1 and correctly marginalize to 1D continuous densities.")
    print("SUCCESS: 02_multivariate_density_simulation executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
