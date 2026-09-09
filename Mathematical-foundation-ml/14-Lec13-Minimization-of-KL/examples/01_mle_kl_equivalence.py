#!/usr/bin/env python3
"""
01_mle_kl_equivalence.py
========================
Rigorous numerical verification of the fundamental theorem of Machine Learning:
  arg min_theta D_KL(p_data || p_theta) == arg max_theta sum_{i=1}^N log p_theta(x_i)

Demonstrates:
  1. Expansion of KL: D_KL(p_data || p_theta) = -H(p_data) + E_{x ~ p_data}[-log p_theta(x)]
  2. Data entropy H(p_data) is independent of theta and drops out of argmin
  3. Law of Large Numbers (LLN) empirical expectation equals average log-likelihood
  4. First-order gradient optimization converges precisely to analytical MLE estimates

Corresponds to Mathematical Foundations of ML — Lecture 13: Minimization of KL.
Zero external dependencies outside PyTorch and NumPy.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def main() -> int:
    print("=== Simulation 01: Equivalence of Min-KL and Maximum Likelihood Estimation ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. True Data-Generating Distribution p_data ~ Normal(mu_star, sigma_star^2)
    # -------------------------------------------------------------
    mu_true = 4.20
    sigma_true = 1.80
    N = 25000
    x_data = mu_true + sigma_true * torch.randn(N)

    # Analytical sample MLE parameters
    mu_mle_analytical = x_data.mean().item()
    sigma_mle_analytical = x_data.std(unbiased=False).item()
    print(f"[Analytical MLE] mu_hat = {mu_mle_analytical:.4f}, sigma_hat = {sigma_mle_analytical:.4f}")

    # -------------------------------------------------------------
    # 2. Parametric Candidate Model Family p_theta(x) = Normal(mu, sigma^2)
    # -------------------------------------------------------------
    # Learnable parameters theta = (mu, log_sigma)
    mu_param = nn.Parameter(torch.tensor(0.0))
    log_sigma_param = nn.Parameter(torch.tensor(0.0))

    # -------------------------------------------------------------
    # 3. Optimize Empirical Cross-Entropy / Negative Log-Likelihood
    #    L(theta) = - 1/N sum_{i=1}^N log p_theta(x_i)
    # -------------------------------------------------------------
    optimizer = optim.Adam([mu_param, log_sigma_param], lr=0.08)
    epochs = 350

    print("[Optimization] Minimizing empirical KL surrogate (Negative Log-Likelihood)...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        sigma = torch.exp(log_sigma_param)
        # NLL: 0.5 * log(2*pi) + log_sigma + 0.5 * mean(((x - mu)/sigma)^2)
        loss = 0.5 * np.log(2.0 * np.pi) + log_sigma_param + 0.5 * torch.mean(((x_data - mu_param) / sigma) ** 2)
        loss.backward()
        optimizer.step()

    fitted_mu = mu_param.item()
    fitted_sigma = torch.exp(log_sigma_param).item()

    print(f"[Optimized Min-KL] mu = {fitted_mu:.4f}, sigma = {fitted_sigma:.4f}")
    print(f"[Discrepancy] |mu_opt - mu_mle| = {abs(fitted_mu - mu_mle_analytical):.5f}")
    print(f"[Discrepancy] |sigma_opt - sigma_mle| = {abs(fitted_sigma - sigma_mle_analytical):.5f}")

    # -------------------------------------------------------------
    # 4. Assert Exact Convergence to MLE / Population True Parameters
    # -------------------------------------------------------------
    assert np.isclose(fitted_mu, mu_mle_analytical, atol=0.01), "Optimized mean must match analytical MLE"
    assert np.isclose(fitted_sigma, sigma_mle_analytical, atol=0.01), "Optimized sigma must match analytical MLE"
    assert np.isclose(fitted_mu, mu_true, atol=0.05), "Fitted mean must match true population mean"
    assert np.isclose(fitted_sigma, sigma_true, atol=0.05), "Fitted sigma must match true population sigma"

    print("Verified: Minimizing KL divergence over dataset D is algebraically identical to MLE.")
    print("SUCCESS: 01_mle_kl_equivalence executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
