#!/usr/bin/env python3
"""
01_three_step_ml_recipe.py
==========================
Mathematical simulation of the canonical Three-Step Machine Learning Recipe:
  Step 1: Model Family Selection {p_theta | theta in Theta} (Gaussian model)
  Step 2: Distance / Divergence Formulation (Negative Log-Likelihood / KL surrogate)
  Step 3: Optimization via Gradient Descent (theta* = argmin_theta L(theta))

Corresponds to Mathematical Foundations of ML — Lecture 10: Challenges of ML.
Zero external dependencies outside PyTorch and NumPy.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def main() -> int:
    print("=== Simulation 01: Canonical Three-Step ML Recipe ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # Ground Truth Data Generation: p_true ~ Normal(mu_true, sigma_true^2)
    # -------------------------------------------------------------
    mu_true = 3.50
    sigma_true = 1.25
    N = 10000
    x_train = mu_true + sigma_true * torch.randn(N)

    # -------------------------------------------------------------
    # Step 1: Choose Model Family p_theta
    # -------------------------------------------------------------
    # Parametric continuous density: p_theta(x) = Normal(mu_theta, sigma_theta^2)
    # Parameters theta = (mu, log_sigma) ensuring sigma > 0
    mu_param = nn.Parameter(torch.tensor(0.0))
    log_sigma_param = nn.Parameter(torch.tensor(0.0))  # Initial sigma = 1.0

    print(f"Step 1 (Model Family): Normal(mu, sigma^2) initialized at mu={mu_param.item():.2f}, sigma={torch.exp(log_sigma_param).item():.2f}")

    # -------------------------------------------------------------
    # Step 2: Choose Distance / Divergence Surrogate
    # -------------------------------------------------------------
    # Kullback-Leibler divergence D_KL(p_true || p_theta) up to constant is
    # equivalent to Negative Log-Likelihood:
    # L(theta) = - 1/N sum_{i=1}^N log p_theta(x_i)
    # log p_theta(x) = -0.5 * log(2*pi) - log_sigma - 0.5 * ((x - mu)/sigma)^2
    def compute_nll(x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        nll = 0.5 * np.log(2.0 * np.pi) + log_sigma + 0.5 * torch.mean(((x - mu) / sigma) ** 2)
        return nll

    # -------------------------------------------------------------
    # Step 3: Optimize via Gradient Descent
    # -------------------------------------------------------------
    optimizer = optim.Adam([mu_param, log_sigma_param], lr=0.05)
    epochs = 400

    print("Step 3 (Optimization): Minimizing NLL via Adam...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_nll(x_train, mu_param, log_sigma_param)
        loss.backward()
        optimizer.step()

    fitted_mu = mu_param.item()
    fitted_sigma = torch.exp(log_sigma_param).item()

    print(f"[Results] True:   mu={mu_true:.4f}, sigma={sigma_true:.4f}")
    print(f"[Results] Fitted: mu={fitted_mu:.4f}, sigma={fitted_sigma:.4f}")

    # Verify mathematical convergence
    assert np.isclose(fitted_mu, mu_true, atol=0.05), (
        f"Fitted mean {fitted_mu} did not converge to true mean {mu_true}"
    )
    assert np.isclose(fitted_sigma, sigma_true, atol=0.05), (
        f"Fitted sigma {fitted_sigma} did not converge to true sigma {sigma_true}"
    )

    print("Verified: Three-step recipe successfully recovers true distribution parameters.")
    print("SUCCESS: 01_three_step_ml_recipe executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
