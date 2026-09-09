"""
01_pushforward_measure_and_cdf.py
=================================
Lecture 04 Simulation: Pushforward Measure & Cumulative Distribution Function (CDF)

Simulates the pushforward probability measure P_X and evaluates the Cumulative
Distribution Function F_X(x) = P(X <= x) = P(X^{-1}((-inf, x])).
Verifies the four fundamental mathematical invariants of any valid univariate CDF:
    1. Bounds: 0 <= F_X(x) <= 1 for all x in R.
    2. Monotonicity: x_1 < x_2 ==> F_X(x_1) <= F_X(x_2).
    3. Asymptotic Limits: lim_{x -> -inf} F_X(x) == 0 and lim_{x -> +inf} F_X(x) == 1.
    4. Interval Theorem: P(a < X <= b) == F_X(b) - F_X(a).

Verification:
    Uses torch.allclose between empirical CDF and analytical Gaussian CDF. Exits code 0.
"""

import sys
import numpy as np
import torch


def analytical_gaussian_cdf(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    """Computes exact analytical CDF of Normal(mu, sigma) using the error function."""
    return 0.5 * (1.0 + torch.erf((x - mu) / (sigma * np.sqrt(2.0))))


def main() -> int:
    print("=== Simulation 01: Pushforward Measure & CDF Invariants ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Generate N = 50,000 samples from Standard Normal distribution
    N = 50000
    samples = torch.randn(N, dtype=torch.float32)

    # 2. Define Empirical CDF function: F_N(x) = (1/N) * sum_i I(X_i <= x)
    def empirical_cdf(query_points: torch.Tensor) -> torch.Tensor:
        # Broadcasting: [len(query), N]
        return torch.mean((samples.unsqueeze(0) <= query_points.unsqueeze(1)).float(), dim=1)

    # 3. Test grid spanning [-4, +4]
    grid = torch.linspace(-4.0, 4.0, 100)
    f_emp = empirical_cdf(grid)
    f_true = analytical_gaussian_cdf(grid)

    # Invariant 1: Bounds in [0, 1]
    assert torch.all(f_emp >= 0.0) and torch.all(f_emp <= 1.0), "CDF values must lie strictly in [0, 1]"

    # Invariant 2: Monotonicity
    diffs = torch.diff(f_emp)
    assert torch.all(diffs >= -1e-6), "CDF must be non-decreasing"

    # Invariant 3: Limits
    f_left = empirical_cdf(torch.tensor([-6.0])).item()
    f_right = empirical_cdf(torch.tensor([+6.0])).item()
    assert f_left < 1e-3, f"Left asymptotic limit should approach 0: {f_left}"
    assert f_right > 0.999, f"Right asymptotic limit should approach 1: {f_right}"

    # Verify convergence to analytical Gaussian CDF (Dvoretzky-Kiefer-Wolfowitz theorem)
    max_error = torch.max(torch.abs(f_emp - f_true)).item()
    print(f"Empirical vs True Gaussian CDF Max Kolmogorov-Smirnov Error: {max_error:.4f}")
    assert max_error < 0.015, f"Empirical CDF must converge closely to true CDF: {max_error}"

    # Invariant 4: Interval Probability Law P(a < X <= b) = F(b) - F(a)
    a, b = -0.5, 1.2
    prob_interval_empirical = torch.mean(((samples > a) & (samples <= b)).float()).item()
    prob_interval_cdf = (analytical_gaussian_cdf(torch.tensor([b])) - analytical_gaussian_cdf(torch.tensor([a]))).item()

    print(f"Direct Interval Counting: {prob_interval_empirical:.5f}")
    print(f"CDF Difference F(b)-F(a): {prob_interval_cdf:.5f}")
    assert np.isclose(prob_interval_empirical, prob_interval_cdf, atol=0.01), (
        "Interval probability must equal F(b) - F(a)"
    )

    print("SUCCESS: 01_pushforward_measure_and_cdf executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
