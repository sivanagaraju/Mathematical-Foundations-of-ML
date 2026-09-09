#!/usr/bin/env python3
"""
02_differential_entropy_simulation.py
=====================================
Rigorous numerical verification of Continuous Differential Entropy:
  1. Analytical Gaussian Entropy: h(X) = 0.5 * ln(2 * pi * e * sigma^2)
  2. Contrastive Demonstration: Differential entropy can be NEGATIVE (unlike discrete H(X) >= 0)
  3. Critical Variance Threshold: sigma* = 1 / sqrt(2 * pi * e) ~= 0.24197
  4. Numerical Riemann Integration vs Closed-Form Formula across varied scales

Corresponds to Mathematical Foundations of ML — Lecture 11: Entropy.
Zero external dependencies outside PyTorch and NumPy.
"""

import sys
import numpy as np
import torch


def gaussian_analytical_entropy(sigma: float) -> float:
    """Return theoretical differential entropy in nats for 1D Gaussian."""
    return 0.5 * np.log(2.0 * np.pi * np.e * (sigma ** 2))


def gaussian_numerical_entropy(sigma: float, mu: float = 0.0, span_sigmas: float = 6.0, points: int = 20001) -> float:
    """Compute differential entropy via high-density Riemann numerical integration."""
    x_min = mu - span_sigmas * sigma
    x_max = mu + span_sigmas * sigma
    x = np.linspace(x_min, x_max, points)
    dx = x[1] - x[0]

    # Gaussian PDF: f(x)
    norm_const = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    f_x = norm_const * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Differential entropy integrand: - f(x) * ln(f(x))
    log_f_x = np.log(f_x + 1e-300)
    integrand = -f_x * log_f_x

    return float(np.sum(integrand) * dx)


def main() -> int:
    print("=== Simulation 02: Continuous Differential Entropy & Negative Entropy ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Critical Zero-Entropy Threshold
    # -------------------------------------------------------------
    sigma_crit = 1.0 / np.sqrt(2.0 * np.pi * np.e)
    h_crit = gaussian_analytical_entropy(sigma_crit)
    print(f"[Critical Variance] sigma* = {sigma_crit:.5f} -> h(X) = {h_crit:.6f} nats")
    assert np.isclose(h_crit, 0.0, atol=1e-5), "Differential entropy at critical sigma must be 0"

    # -------------------------------------------------------------
    # 2. Demonstration of Negative Differential Entropy
    # -------------------------------------------------------------
    # Sharp Gaussian: sigma = 0.10 < sigma* (0.24197)
    sigma_sharp = 0.10
    h_sharp_analytical = gaussian_analytical_entropy(sigma_sharp)
    h_sharp_numerical = gaussian_numerical_entropy(sigma_sharp)

    print(f"[Negative Entropy] sigma={sigma_sharp}: Analytical={h_sharp_analytical:.4f} | Numerical={h_sharp_numerical:.4f}")
    assert h_sharp_analytical < 0.0, "Sharp Gaussian differential entropy must be strictly negative"
    assert np.isclose(h_sharp_numerical, h_sharp_analytical, atol=1e-3), (
        f"Numerical entropy {h_sharp_numerical} diverges from analytical {h_sharp_analytical}"
    )

    # -------------------------------------------------------------
    # 3. Multi-Scale Convergence Suite across Scales
    # -------------------------------------------------------------
    test_sigmas = [0.05, 0.20, 0.50, 1.00, 2.50]

    print(f"{'Sigma':>8} | {'Analytical (nats)':>18} | {'Numerical (nats)':>18} | {'Absolute Error':>15}")
    print("-" * 65)

    for sig in test_sigmas:
        h_ana = gaussian_analytical_entropy(sig)
        h_num = gaussian_numerical_entropy(sig)
        err = abs(h_ana - h_num)
        print(f"{sig:>8.2f} | {h_ana:>18.5f} | {h_num:>18.5f} | {err:>15.6f}")
        assert err < 1e-3, f"Numerical integration error {err} at sigma={sig} exceeded 1e-3"

    print("Verified: Differential entropy accurately captures continuous information and admits negative values.")
    print("SUCCESS: 02_differential_entropy_simulation executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
