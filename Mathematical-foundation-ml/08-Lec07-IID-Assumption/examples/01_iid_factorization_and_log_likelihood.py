"""
01_iid_factorization_and_log_likelihood.py
==========================================
Lecture 07 Simulation: IID Factorization & Additive Log-Likelihood

Demonstrates the mathematical consequence of the Independent and Identically
Distributed (I.I.D.) assumption:
    1. Joint density factors into product of marginals: p(x_1, ..., x_N) = prod_{i=1}^N p(x_i)
    2. Log-likelihood decomposes into a simple sum: log p(x_1, ..., x_N) = sum_{i=1}^N log p(x_i)
    3. Contrasts with non-IID autoregressive sequence where naive sum misestimates joint likelihood.

Mathematical Invariant:
    Under IID: log p(X_{1:N}) == sum_{i=1}^N log p(X_i)
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 01: IID Factorization & Additive Log-Likelihood ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Define univariate Gaussian distribution P_data ~ Normal(mu, sigma)
    mu = 2.5
    sigma = 1.2
    dist = torch.distributions.Normal(loc=mu, scale=sigma)

    # 2. Draw N = 1,000 independent samples X_i ~ P_data
    N = 1000
    samples_iid = dist.sample((N,))

    # 3. Compute individual log probabilities log p(x_i)
    log_probs_individual = dist.log_prob(samples_iid)

    # 4. Joint log probability under IID assumption: sum of individual logs
    joint_log_prob_iid = torch.sum(log_probs_individual).item()

    # Manual verification of sum decomposition
    manual_sum = 0.0
    for i in range(N):
        x = samples_iid[i].item()
        # Analytical Gaussian log-density: -0.5 * log(2*pi*sigma^2) - 0.5 * ((x-mu)/sigma)^2
        manual_val = -0.5 * np.log(2.0 * np.pi * (sigma**2)) - 0.5 * (((x - mu) / sigma)**2)
        manual_sum += manual_val

    print(f"PyTorch Vectorized Sum Log-Likelihood: {joint_log_prob_iid:.4f}")
    print(f"Manual Analytical Sum Log-Likelihood:   {manual_sum:.4f}")
    assert np.isclose(joint_log_prob_iid, manual_sum, atol=1e-3), (
        "IID log-likelihood must identically equal the sum of analytical individual log-densities"
    )

    # 5. Contrast with NON-IID Markov Process: X_t = alpha * X_{t-1} + eps_t
    alpha = 0.85
    noise_sigma = 0.5
    samples_non_iid = torch.zeros(N)
    samples_non_iid[0] = torch.randn(1) * 1.0
    for t in range(1, N):
        samples_non_iid[t] = alpha * samples_non_iid[t-1] + noise_sigma * torch.randn(1)

    # True joint log-likelihood of AR(1) chain: log p(x_1) + sum_{t=2}^N log p(x_t | x_{t-1})
    ar_dist = torch.distributions.Normal(loc=0.0, scale=noise_sigma)
    true_ar_joint_log = torch.distributions.Normal(0.0, 1.0).log_prob(samples_non_iid[0]).item()
    for t in range(1, N):
        cond_mean = alpha * samples_non_iid[t-1]
        true_ar_joint_log += torch.distributions.Normal(cond_mean, noise_sigma).log_prob(samples_non_iid[t]).item()

    # Naive naive IID evaluation assuming independence:
    naive_iid_eval = torch.sum(torch.distributions.Normal(0.0, 1.0).log_prob(samples_non_iid)).item()

    print(f"True AR(1) Non-IID Joint Log-Likelihood:  {true_ar_joint_log:.2f}")
    print(f"Naive IID Miscalculated Log-Likelihood:   {naive_iid_eval:.2f}")
    assert abs(true_ar_joint_log - naive_iid_eval) > 100.0, (
        "Non-IID sequence must diverge significantly from naive IID factorization"
    )

    print("SUCCESS: 01_iid_factorization_and_log_likelihood executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
