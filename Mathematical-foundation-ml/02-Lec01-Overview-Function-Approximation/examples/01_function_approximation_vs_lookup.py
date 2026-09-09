"""
01_function_approximation_vs_lookup.py
======================================
Lecture 01 Simulation: Function Approximation vs. Table-Lookup Trap

Demonstrates why continuous function approximation generalizes across unseen test queries,
whereas exact memorization (table lookup) fails in high-dimensional or continuous domains.

Mathematical Guarantee:
    Polynomial / linear basis approximation achieves bounded generalization error on unseen
    test points sampled from the continuous domain [0, 2pi], whereas zero-order lookup tables
    suffer O(h) interpolation gap where h is grid spacing.

Verification:
    Asserts torch.allclose between analytical polynomial fit and pytorch autograd gradient descent.
"""

import sys
import torch
import numpy as np


def main() -> int:
    print("=== Simulation 01: Function Approximation vs Table-Lookup ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Define true underlying physical function: f(x) = sin(x) + 0.1 * x
    def true_f(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + 0.1 * x

    # 2. Generate discrete training dataset D = {(x_i, y_i)} with N = 25 points
    x_train = torch.linspace(0.0, 2.0 * torch.pi, 25)
    y_train = true_f(x_train)

    # 3. Table Lookup Baseline: Nearest-neighbor table lookup
    # For a test point halfway between training samples, error equals step spacing gap
    x_test = torch.tensor([0.5 * (x_train[0] + x_train[1])])
    y_test_true = true_f(x_test)

    # Nearest neighbor index in lookup table
    nearest_idx = torch.argmin(torch.abs(x_train - x_test))
    y_lookup_pred = y_train[nearest_idx]
    lookup_error = torch.abs(y_lookup_pred - y_test_true).item()

    # 4. Parametric Function Approximation: Degree-3 Polynomial Basis
    # Phi(x) = [1, x, x^2, x^3]
    def poly_features(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.ones_like(x), x, x**2, x**3], dim=-1)

    X_poly = poly_features(x_train)  # [N, 4]

    # Analytical Ordinary Least Squares: w = (X^T X)^{-1} X^T y
    w_ols = torch.linalg.lstsq(X_poly, y_train).solution

    # 5. Gradient Descent Function Approximation via PyTorch Autograd with L-BFGS
    w_gd = torch.zeros(4, requires_grad=True)
    optimizer = torch.optim.LBFGS([w_gd], lr=1.0, max_iter=100)

    def closure():
        optimizer.zero_grad()
        pred = X_poly @ w_gd
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Verify that gradient descent converges close to analytical OLS weights
    diff = torch.norm(w_ols - w_gd.detach()).item()
    print(f"OLS vs L-BFGS weight divergence: {diff:.6f}")
    assert np.allclose(w_ols.numpy(), w_gd.detach().numpy(), atol=1e-3), (
        f"L-BFGS weights must converge to analytical OLS solution, diff={diff}"
    )

    # Test generalization at unseen midpoint
    X_test_poly = poly_features(x_test)
    y_approx_pred = (X_test_poly @ w_ols).squeeze()
    approx_error = torch.abs(y_approx_pred - y_test_true).item()

    print(f"Unseen test point x: {x_test.item():.4f}")
    print(f"True value y:        {y_test_true.item():.4f}")
    print(f"Table lookup pred:   {y_lookup_pred.item():.4f} (Error: {lookup_error:.4f})")
    print(f"Approximator pred:   {y_approx_pred.item():.4f} (Error: {approx_error:.4f})")

    assert approx_error < lookup_error, (
        f"Continuous function approximator must outperform table lookup on unseen test point: "
        f"{approx_error} vs {lookup_error}"
    )

    print("SUCCESS: 01_function_approximation_vs_lookup executed cleanly with verified generalization.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
