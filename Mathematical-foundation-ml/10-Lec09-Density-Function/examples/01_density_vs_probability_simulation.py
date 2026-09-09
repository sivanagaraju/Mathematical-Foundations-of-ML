#!/usr/bin/env python3
"""
01_density_vs_probability_simulation.py
======================================
Rigorous simulation demonstrating the mathematical distinction between:
1. Probability Density Function (PDF) height f_X(x) (which can exceed 1)
2. Event Probability P(X in [a, b]) (which is strictly bounded in [0, 1])
3. Singleton Event Probability P(X = c) = 0 for continuous random variables

Corresponds to Mathematical Foundations of ML — Lecture 09: Density Function.
Zero external dependencies outside PyTorch and NumPy.
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 01: Density vs Event Probability ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Counterexample to 'Probability <= 1': PDF height exceeding 1
    # -------------------------------------------------------------
    # Uniform distribution on [0, 0.2]
    # Length of support: L = 0.2 - 0.0 = 0.2
    # Density height f(x) = 1 / 0.2 = 5.0 for all x in [0, 0.2]
    a, b = 0.0, 0.2
    expected_density_height = 1.0 / (b - a)
    assert expected_density_height == 5.0, "Uniform[0, 0.2] density must equal 5.0"
    print(f"[Axiom Check] Uniform[0, {b}] density height f(x) = {expected_density_height:.2f} > 1.0")

    # High-precision numerical Riemann integration over support
    grid = np.linspace(a, b, 10001)
    dx = grid[1] - grid[0]
    densities = np.full_like(grid, expected_density_height)
    # NumPy 2.0 renamed trapz to trapezoid; provide fallback
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    total_prob_mass = trap_fn(densities, grid) if trap_fn is not None else np.sum(densities[:-1]) * dx
    print(f"[Total Mass] Integral of f(x) dx over [0, 0.2] = {total_prob_mass:.6f}")
    assert np.isclose(total_prob_mass, 1.0, atol=1e-5), "Integral of PDF over entire support must equal 1.0"

    # -------------------------------------------------------------
    # 2. Probability of an Interval [0.05, 0.15]
    # -------------------------------------------------------------
    # Analytical: P(0.05 <= X <= 0.15) = (0.15 - 0.05) * 5.0 = 0.10 * 5.0 = 0.50
    sub_a, sub_b = 0.05, 0.15
    analytical_prob = (sub_b - sub_a) * expected_density_height
    assert np.isclose(analytical_prob, 0.50, atol=1e-7)

    # Monte Carlo verification with N = 100,000 samples
    N = 100000
    samples = torch.empty(N).uniform_(a, b)
    empirical_prob = ((samples >= sub_a) & (samples <= sub_b)).float().mean().item()
    print(f"[Interval Probability] Analytical: {analytical_prob:.4f} | Monte Carlo: {empirical_prob:.4f}")
    assert np.isclose(empirical_prob, analytical_prob, atol=0.01), (
        f"Empirical probability {empirical_prob} diverges from analytical {analytical_prob}"
    )

    # -------------------------------------------------------------
    # 3. Singleton Probability P(X = c) = 0 for continuous variable
    # -------------------------------------------------------------
    # P(c - eps <= X <= c + eps) = 2 * eps * 5.0 -> 0 as eps -> 0
    c = 0.10
    eps_values = [0.05, 0.01, 0.001, 0.0001, 0.0]
    for eps in eps_values:
        prob_eps = min(b, c + eps) - max(a, c - eps)
        mass = prob_eps * expected_density_height if eps > 0 else 0.0
        print(f"  P({c} - {eps:.4f} <= X <= {c} + {eps:.4f}) = {mass:.6f}")

    # Exactly 0 points match exact float equality in continuous draws
    exact_matches = (samples == c).sum().item()
    assert exact_matches == 0, f"Expected 0 exact hits for singleton in continuous distribution, got {exact_matches}"

    # -------------------------------------------------------------
    # 4. Gaussian PDF with Small Variance (Dirac Spike Limit)
    # -------------------------------------------------------------
    # Normal(0, sigma^2) with sigma = 0.05
    # f(0) = 1 / (sigma * sqrt(2*pi)) = 1 / (0.05 * 2.5066) = 7.9788
    sigma = 0.05
    gaussian_peak = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    print(f"[Gaussian Peak] N(0, 0.05^2) peak density f(0) = {gaussian_peak:.4f} >> 1.0")
    assert gaussian_peak > 7.9, "Sharp Gaussian peak must exceed 7.9"

    print("Verified: PDF heights can be arbitrarily large; probabilities are integrals bounded in [0, 1].")
    print("SUCCESS: 01_density_vs_probability_simulation executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
