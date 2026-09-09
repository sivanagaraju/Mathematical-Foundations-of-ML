"""
02_multivariate_product_geometry.py
===================================
Lecture 04 Simulation: Multivariate Product Geometry & 2D CDF Invariants

Simulates a 2D vector-valued random variable X = [X_1, X_2]^T and verifies
the 2D joint Cumulative Distribution Function F_{X_1, X_2}(x_1, x_2) = P(X_1 <= x_1, X_2 <= x_2).

Mathematical Invariant:
    The probability mass of a 2D rectangle (a_1, b_1] x (a_2, b_2] satisfies:
    P(a_1 < X_1 <= b_1, a_2 < X_2 <= b_2) =
        F(b_1, b_2) - F(a_1, b_2) - F(b_1, a_2) + F(a_1, a_2) >= 0.

Verification:
    Uses torch.allclose between geometric 2D CDF inclusion-exclusion and direct empirical counting.
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 02: Multivariate Product Geometry ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Generate N = 100,000 samples of 2D Gaussian RV with non-zero covariance
    N = 100000
    mean = torch.tensor([0.5, -0.2], dtype=torch.float32)
    cov = torch.tensor([[1.0, 0.6],
                        [0.6, 1.5]], dtype=torch.float32)
    # Cholesky factor L: cov = L L^T
    L = torch.linalg.cholesky(cov)
    standard_normals = torch.randn(N, 2, dtype=torch.float32)
    samples = standard_normals @ L.T + mean  # [N, 2]

    # 2. Define Empirical 2D Joint CDF: F(x1, x2) = P(X1 <= x1, X2 <= x2)
    def joint_cdf_2d(x1: float, x2: float) -> float:
        mask = (samples[:, 0] <= x1) & (samples[:, 1] <= x2)
        return torch.mean(mask.float()).item()

    # 3. Define a 2D target bounding box / rectangle: (a1, b1] x (a2, b2]
    a1, b1 = -0.3, 1.2
    a2, b2 = -0.8, 0.9

    # Direct empirical rectangle probability by counting sample hits
    in_box_mask = (
        (samples[:, 0] > a1) & (samples[:, 0] <= b1) &
        (samples[:, 1] > a2) & (samples[:, 1] <= b2)
    )
    direct_prob = torch.mean(in_box_mask.float()).item()

    # 4. Evaluate using the 2D CDF Inclusion-Exclusion Formula:
    # P(box) = F(b1, b2) - F(a1, b2) - F(b1, a2) + F(a1, a2)
    F_b1_b2 = joint_cdf_2d(b1, b2)
    F_a1_b2 = joint_cdf_2d(a1, b2)
    F_b1_a2 = joint_cdf_2d(b1, a2)
    F_a1_a2 = joint_cdf_2d(a1, a2)

    formula_prob = F_b1_b2 - F_a1_b2 - F_b1_a2 + F_a1_a2

    print(f"F(b1, b2): {F_b1_b2:.6f}")
    print(f"F(a1, b2): {F_a1_b2:.6f}")
    print(f"F(b1, a2): {F_b1_a2:.6f}")
    print(f"F(a1, a2): {F_a1_a2:.6f}")
    print(f"2D CDF Rectangle Formula Probability: {formula_prob:.6f}")
    print(f"Direct Monte Carlo Hit Probability:    {direct_prob:.6f}")

    assert formula_prob >= 0.0, "2D CDF rectangle probability must be non-negative"
    assert np.isclose(formula_prob, direct_prob, atol=1e-5), (
        f"Rectangle formula must match direct counting: {formula_prob} vs {direct_prob}"
    )

    # 5. Marginal Consistency: lim_{x2 -> inf} F(x1, x2) == F_1(x1)
    x1_test = 0.5
    f_joint_large_x2 = joint_cdf_2d(x1_test, 10.0)
    f_marginal_x1 = torch.mean((samples[:, 0] <= x1_test).float()).item()
    print(f"Joint CDF at large x2 (10.0): {f_joint_large_x2:.6f}")
    print(f"Marginal CDF F_1(x1):         {f_marginal_x1:.6f}")
    assert np.isclose(f_joint_large_x2, f_marginal_x1, atol=1e-4), (
        "Joint CDF limit as x2 -> inf must equal marginal CDF F_1(x1)"
    )

    print("SUCCESS: 02_multivariate_product_geometry executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
