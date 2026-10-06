"""
Example 1: Optimizer Dynamics and Pathologies on Ill-Conditioned Landscapes
Lecture 53: SGD, RMSprop, Adam: Optimizers

This script demonstrates two core mathematical phenomena:
1. Learning Rate Pathologies on Ill-Conditioned Ravines:
   Demonstrating the theoretical stability limit alpha < 2 / lambda_max.
   When alpha exceeds this threshold, Vanilla SGD oscillates violently and diverges,
   while Adam normalizes coordinate curvature and remains stable.
2. Comparative Trajectories on Non-Convex Rosenbrock Banana Valley:
   Benchmarking Vanilla SGD, SGD with Momentum, RMSprop, and Adam on
   f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2.
"""

import torch
import torch.nn as nn

def evaluate_stability_limit():
    """
    Demonstrates the theoretical stability limit for gradient descent:
    For quadratic f(x, y) = 0.1 * x^2 + 50 * y^2, the Hessian is diag([0.2, 100]).
    The maximum eigenvalue is lambda_max = 100.
    The theoretical threshold is alpha_max = 2 / lambda_max = 2 / 100 = 0.02.
    """
    print("-" * 70)
    print("PART 1: Gradient Descent Stability Limit on Ill-Conditioned Ravine")
    print("Function: f(x, y) = 0.1 * x^2 + 50 * y^2  (lambda_max = 100, alpha_crit = 0.02)")
    print("-" * 70)

    # 1. Vanilla SGD with alpha = 0.022 > 0.02 (Diverges on steep axis y)
    theta_sgd = torch.tensor([5.0, 1.0], requires_grad=True)  # Shape: [2]
    opt_sgd = torch.optim.SGD([theta_sgd], lr=0.022)

    initial_loss = 0.1 * (theta_sgd[0] ** 2) + 50.0 * (theta_sgd[1] ** 2)
    for _ in range(25):
        opt_sgd.zero_grad()
        loss = 0.1 * (theta_sgd[0] ** 2) + 50.0 * (theta_sgd[1] ** 2)
        loss.backward()
        opt_sgd.step()

    final_sgd_loss = (0.1 * (theta_sgd[0] ** 2) + 50.0 * (theta_sgd[1] ** 2)).item()
    print(f"Vanilla SGD (alpha=0.022 > 0.02) -> Final y: {theta_sgd[1].item():.4f}, Loss: {final_sgd_loss:.2e}")
    # Assert SGD diverged along high-curvature axis y
    assert final_sgd_loss > 1000.0, "Vanilla SGD should diverge when alpha > 2 / lambda_max"
    assert abs(theta_sgd[1].item()) > 10.0, "Coordinate y should oscillate out of control"

    # 2. Adam with alpha = 0.05 on the same ill-conditioned ravine (Normalizes curvature)
    theta_adam = torch.tensor([5.0, 1.0], requires_grad=True)  # Shape: [2]
    opt_adam = torch.optim.Adam([theta_adam], lr=0.05)

    for _ in range(150):
        opt_adam.zero_grad()
        loss = 0.1 * (theta_adam[0] ** 2) + 50.0 * (theta_adam[1] ** 2)
        loss.backward()
        opt_adam.step()

    final_adam_loss = (0.1 * (theta_adam[0] ** 2) + 50.0 * (theta_adam[1] ** 2)).item()
    print(f"Adam        (alpha=0.05)         -> Final y: {theta_adam[1].item():.4f}, Loss: {final_adam_loss:.6f}")
    assert final_adam_loss < 0.10, "Adam should stably converge despite anisotropic curvature"
    assert final_adam_loss < final_sgd_loss / 1e5, "Adam loss should be orders of magnitude lower than diverged SGD"
    print("[OK] Part 1 Verified: Adam avoids catastrophic ravine oscillations.")

def evaluate_rosenbrock_trajectories():
    """
    Benchmarks SGD, Momentum, RMSprop, and Adam on Rosenbrock Banana Valley:
    f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
    Global minimum is at (1.0, 1.0) with loss 0.0.
    """
    print("\n" + "-" * 70)
    print("PART 2: Optimizer Trajectories on Rosenbrock Banana Valley")
    print("Function: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2")
    print("Starting Point: (-1.0, 1.0), Optimum: (1.0, 1.0)")
    print("-" * 70)

    configs = [
        ("SGD", lambda p: torch.optim.SGD(p, lr=0.001)),
        ("Momentum", lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9)),
        ("RMSprop", lambda p: torch.optim.RMSprop(p, lr=0.01)),
        ("Adam", lambda p: torch.optim.Adam(p, lr=0.05))
    ]

    results = {}
    for name, opt_fn in configs:
        th = torch.tensor([-1.0, 1.0], requires_grad=True)  # Shape: [2]
        opt = opt_fn([th])
        for _ in range(500):
            opt.zero_grad()
            loss = (1.0 - th[0]) ** 2 + 100.0 * (th[1] - th[0] ** 2) ** 2
            loss.backward()
            opt.step()
        results[name] = (th.detach(), loss.item())
        print(f"[{name:8s}] Final coords: ({th[0].item():.4f}, {th[1].item():.4f}), Loss: {loss.item():.6f}")

    sgd_pos, sgd_loss = results["SGD"]
    mom_pos, mom_loss = results["Momentum"]
    rms_pos, rms_loss = results["RMSprop"]
    adam_pos, adam_loss = results["Adam"]

    # Structural assertions
    assert sgd_loss > 1.0, "Vanilla SGD gets stuck in the flat curved valley"
    assert mom_loss < sgd_loss, "Momentum navigates curved valleys faster than Vanilla SGD"
    assert adam_loss < 0.01, f"Adam should reach the neighborhood of (1, 1), got loss {adam_loss}"
    assert torch.allclose(adam_pos, torch.tensor([1.0, 1.0]), atol=0.05), "Adam reaches within 0.05 of optimum"

    print("[OK] Part 2 Verified: Momentum and Adam successfully traverse non-convex valley floor.")

def main():
    print("=" * 70)
    print("LECTURE 53 SCRIPT 1: OPTIMIZER DYNAMICS & STABILITY VERIFICATION")
    print("=" * 70)
    evaluate_stability_limit()
    evaluate_rosenbrock_trajectories()
    print("\n" + "=" * 70)
    print("[ALL CHECKS PASSED] Optimizer trajectories and stability limits verified.")
    print("=" * 70)

if __name__ == "__main__":
    main()
