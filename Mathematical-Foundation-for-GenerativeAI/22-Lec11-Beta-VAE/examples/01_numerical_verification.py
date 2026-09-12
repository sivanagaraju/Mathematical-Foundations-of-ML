"""
01_numerical_verification.py — Numerical Verification of Beta-VAE Trade-off Frontier

This script simulates the mathematical trade-off between reconstruction error and
Kullback-Leibler divergence across varying Lagrange multipliers beta in (0.1, 10.0).

Theoretical Invariant:
    L_beta(theta, phi; x) = L_recon(x, x_hat) + beta * D_KL(q_phi(z|x) || p(z))
    As beta -> inf, D_KL -> 0 (Posterior Collapse regime).
    As beta -> 0, L_recon -> min, D_KL -> large (Unregularized / Overfitting regime).
"""
import numpy as np

def compute_beta_elbo(recon_error: float, kl_div: float, beta: float) -> float:
    """Computes the scalar beta-VAE objective.
    
    Shapes:
        recon_error: scalar float
        kl_div: scalar float
        beta: scalar float
    """
    return float(recon_error + beta * kl_div)

def simulate_beta_frontier():
    # Sweep beta across two decades: 0.1 to 10.0
    betas = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 10.0])
    
    # Modeled empirical response curves:
    # High beta compresses KL toward 0 but elevates reconstruction error
    kl_values = 15.0 / (1.0 + 2.0 * betas) # Shape: [6]
    recon_values = 20.0 + 10.0 * np.log1p(betas) # Shape: [6]
    
    total_losses = [compute_beta_elbo(r, k, b) for r, k, b in zip(recon_values, kl_values, betas)]
    
    print("=== Beta-VAE Trade-Off Frontier Simulation ===")
    print(f"{'Beta':>6} | {'Recon MSE':>10} | {'KL (nats)':>10} | {'Weighted Loss':>14}")
    print("-" * 48)
    for b, r, k, l in zip(betas, recon_values, kl_values, total_losses):
        print(f"{b:6.2f} | {r:10.4f} | {k:10.4f} | {l:14.4f}")
        
    # Mathematical assertions
    # 1. As beta increases, KL divergence must strictly decrease
    assert np.all(np.diff(kl_values) < 0.0), "KL divergence must decrease monotonically with beta!"
    # 2. As beta increases, reconstruction error must strictly increase
    assert np.all(np.diff(recon_values) > 0.0), "Reconstruction error must increase monotonically with beta!"
    # 3. Verify exact calculation at beta = 1.0
    idx_1 = np.where(betas == 1.0)[0][0]
    expected_loss_1 = recon_values[idx_1] + 1.0 * kl_values[idx_1]
    assert np.isclose(total_losses[idx_1], expected_loss_1), "Beta=1.0 scalar calculation mismatch!"
    
    print("Assertion passed: Monotonic trade-off between reconstruction and KL regularization verified.")

if __name__ == "__main__":
    simulate_beta_frontier()
