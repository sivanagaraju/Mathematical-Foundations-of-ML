#!/usr/bin/env python3
"""
02_gaussian_kl_analytical.py
============================
Mathematical simulation verifying Continuous Gaussian KL Divergence:
  1. Closed-Form Analytical Formula:
     D_KL(N(mu1, sig1^2) || N(mu2, sig2^2)) = ln(sig2/sig1) + (sig1^2 + (mu1 - mu2)^2)/(2*sig2^2) - 1/2
  2. High-precision numerical Riemann integration of continuous integral
  3. PyTorch's native torch.distributions.kl.kl_divergence verification
  4. Multi-parameter test suite ensuring exact three-way mathematical equivalence

Corresponds to Mathematical Foundations of ML — Lecture 12: KL-Divergence.
Zero external dependencies outside PyTorch and NumPy.
"""

import sys
import numpy as np
import torch
from torch.distributions import Normal, kl_divergence


def analytical_gaussian_kl(mu1: float, sig1: float, mu2: float, sig2: float) -> float:
    """Analytical KL divergence in nats between two 1D Gaussians."""
    term1 = np.log(sig2 / sig1)
    term2 = (sig1**2 + (mu1 - mu2)**2) / (2.0 * sig2**2)
    return float(term1 + term2 - 0.5)


def numerical_gaussian_kl(mu1: float, sig1: float, mu2: float, sig2: float, span: float = 7.0, points: int = 40001) -> float:
    """Numerical Riemann integration of continuous KL divergence."""
    # Integrate over support where P is significant
    x_min = mu1 - span * sig1
    x_max = mu1 + span * sig1
    x = np.linspace(x_min, x_max, points)
    dx = x[1] - x[0]

    p = (1.0 / (sig1 * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu1) / sig1)**2)
    q = (1.0 / (sig2 * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu2) / sig2)**2)

    # p(x) * ln(p(x) / q(x))
    integrand = p * np.log(p / q)
    return float(np.sum(integrand) * dx)


def main() -> int:
    print("=== Simulation 02: Continuous Gaussian KL Divergence Equivalence ===")
    torch.manual_seed(42)
    np.random.seed(42)

    test_cases = [
        {"mu1": 0.0, "sig1": 1.0, "mu2": 1.0, "sig2": 1.0, "desc": "Mean shift only"},
        {"mu1": 0.0, "sig1": 0.5, "mu2": 0.0, "sig2": 1.5, "desc": "Variance shift only"},
        {"mu1": 1.5, "sig1": 0.8, "mu2": -0.5, "sig2": 1.2, "desc": "Joint mean & variance shift"},
        {"mu1": 2.0, "sig1": 1.0, "mu2": 2.0, "sig2": 1.0, "desc": "Identical distributions (D_KL=0)"},
    ]

    print(f"{'Description':<32} | {'Analytical':>12} | {'Numerical':>12} | {'PyTorch':>12}")
    print("-" * 75)

    for case in test_cases:
        m1, s1 = case["mu1"], case["sig1"]
        m2, s2 = case["mu2"], case["sig2"]
        desc = case["desc"]

        kl_ana = analytical_gaussian_kl(m1, s1, m2, s2)
        kl_num = numerical_gaussian_kl(m1, s1, m2, s2)

        # PyTorch native KL
        dist1 = Normal(torch.tensor(m1), torch.tensor(s1))
        dist2 = Normal(torch.tensor(m2), torch.tensor(s2))
        kl_torch = kl_divergence(dist1, dist2).item()

        print(f"{desc:<32} | {kl_ana:>12.6f} | {kl_num:>12.6f} | {kl_torch:>12.6f}")

        # Assert three-way equivalence
        assert np.isclose(kl_ana, kl_torch, atol=1e-6), "Analytical KL must match PyTorch native KL"
        assert np.isclose(kl_num, kl_ana, atol=1e-3), (
            f"Numerical integration {kl_num} diverged from analytical {kl_ana}"
        )

    print("Verified: Continuous Gaussian KL divergence analytical formula matches numerical and PyTorch native implementations.")
    print("SUCCESS: 02_gaussian_kl_analytical executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
