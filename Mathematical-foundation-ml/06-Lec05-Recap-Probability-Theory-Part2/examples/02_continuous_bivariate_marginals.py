"""
02_continuous_bivariate_marginals.py
====================================
Lecture 05 Simulation: Continuous Bivariate Conditioning & Marginalization

Simulates a 2D continuous Gaussian random vector [X, Y]^T and computes
empirical conditional distributions P(Y | X in [x - eps, x + eps]),
proving convergence to the exact analytical Gaussian conditioning formulas.

Analytical Invariant:
    For bivariate normal [X, Y]^T with correlation rho:
    Y | X = x ~ Normal(mu_Y + rho * (sigma_Y / sigma_X) * (x - mu_X), sigma_Y^2 * (1 - rho^2))
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 02: Continuous Bivariate Conditioning ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Define bivariate normal parameters
    mu_X, mu_Y = 1.0, 2.0
    sigma_X, sigma_Y = 1.5, 2.0
    rho = 0.7

    cov_matrix = torch.tensor([
        [sigma_X**2, rho * sigma_X * sigma_Y],
        [rho * sigma_X * sigma_Y, sigma_Y**2]
    ], dtype=torch.float32)

    # 2. Sample N = 200,000 points from bivariate normal
    N = 200000
    L = torch.linalg.cholesky(cov_matrix)
    Z = torch.randn(N, 2, dtype=torch.float32)
    samples = Z @ L.T + torch.tensor([mu_X, mu_Y], dtype=torch.float32)

    X_samples = samples[:, 0]
    Y_samples = samples[:, 1]

    # 3. Check empirical marginals
    print(f"Empirical Mean X: {torch.mean(X_samples):.4f} (True: {mu_X:.4f})")
    print(f"Empirical Mean Y: {torch.mean(Y_samples):.4f} (True: {mu_Y:.4f})")
    assert np.isclose(torch.mean(X_samples).item(), mu_X, atol=0.02)
    assert np.isclose(torch.mean(Y_samples).item(), mu_Y, atol=0.02)

    # 4. Condition on slice around x = 2.5: X in [2.45, 2.55]
    x_condition = 2.5
    eps = 0.05
    slice_mask = (X_samples >= x_condition - eps) & (X_samples <= x_condition + eps)
    Y_slice = Y_samples[slice_mask]
    n_slice = Y_slice.shape[0]

    print(f"Conditioning slice X in [{x_condition - eps:.2f}, {x_condition + eps:.2f}]: {n_slice} samples.")
    assert n_slice > 1000, f"Insufficient sample size in slice: {n_slice}"

    # Empirical conditional moments
    emp_cond_mean = torch.mean(Y_slice).item()
    emp_cond_var = torch.var(Y_slice, unbiased=True).item()

    # Analytical conditional moments:
    # mu_{Y|X} = mu_Y + rho * (sigma_Y / sigma_X) * (x - mu_X)
    # var_{Y|X} = sigma_Y^2 * (1 - rho^2)
    analytical_cond_mean = mu_Y + rho * (sigma_Y / sigma_X) * (x_condition - mu_X)
    analytical_cond_var = (sigma_Y**2) * (1.0 - rho**2)

    print(f"Empirical Conditional Mean:  {emp_cond_mean:.4f}")
    print(f"Analytical Conditional Mean: {analytical_cond_mean:.4f}")
    print(f"Empirical Conditional Var:   {emp_cond_var:.4f}")
    print(f"Analytical Conditional Var:  {analytical_cond_var:.4f}")

    assert np.isclose(emp_cond_mean, analytical_cond_mean, atol=0.05), (
        f"Conditional mean error: {emp_cond_mean} vs {analytical_cond_mean}"
    )
    assert np.isclose(emp_cond_var, analytical_cond_var, atol=0.15), (
        f"Conditional variance error: {emp_cond_var} vs {analytical_cond_var}"
    )

    print("SUCCESS: 02_continuous_bivariate_marginals executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
