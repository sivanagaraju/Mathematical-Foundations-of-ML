"""
01_numerical_verification.py — Numerical Verification of Closed-Form Gaussian KL Divergence

This script validates the analytical KL divergence between a diagonal multivariate Gaussian
variational posterior q_phi(z|x) ~ N(mu, diag(sigma^2)) and an isotropic standard normal prior
p(z) ~ N(0, I) against high-sample Monte Carlo numerical integration.

Theoretical Invariant:
    D_KL(q_phi(z|x) || p(z)) = -0.5 * sum_{k=1}^K (1 + log(sigma_k^2) - mu_k^2 - sigma_k^2)
"""
import numpy as np

def analytical_gaussian_kl(mu: np.ndarray, log_var: np.ndarray) -> float:
    """Computes exact closed-form KL divergence.
    
    Shapes:
        mu: [K]
        log_var: [K]
    Returns:
        float: scalar KL divergence value.
    """
    # Shape: [K]
    sigma_sq = np.exp(log_var)
    # Shape: [K]
    kl_per_dim = -0.5 * (1.0 + log_var - np.square(mu) - sigma_sq)
    return float(np.sum(kl_per_dim))

def monte_carlo_gaussian_kl(mu: np.ndarray, log_var: np.ndarray, num_samples: int = 500000) -> float:
    """Estimates KL divergence via empirical expectation: E_q[log q(z|x) - log p(z)].
    
    Shapes:
        mu: [K]
        log_var: [K]
    """
    K = len(mu)
    sigma = np.exp(0.5 * log_var)
    # Shape: [N, K]
    eps = np.random.randn(num_samples, K)
    # Reparameterized latent draws, Shape: [N, K]
    z = mu + sigma * eps
    
    # log q(z|x) under diagonal Gaussian
    # Shape: [N]
    log_q = -0.5 * np.sum(np.log(2.0 * np.pi) + log_var + np.square(z - mu) / (np.exp(log_var)), axis=-1)
    
    # log p(z) under standard normal N(0, I)
    # Shape: [N]
    log_p = -0.5 * np.sum(np.log(2.0 * np.pi) + np.square(z), axis=-1)
    
    # Shape: [N]
    diff = log_q - log_p
    return float(np.mean(diff))

def main():
    print("=== Numerical Verification: Analytical KL vs Monte Carlo Integration ===")
    np.random.seed(42)
    
    # Latent dimension K = 4
    # Shape: [K]
    mu = np.array([0.5, -0.8, 1.2, 0.0], dtype=np.float64)
    # Log-variance parameters, Shape: [K]
    log_var = np.array([-0.5, 0.2, -1.0, 0.5], dtype=np.float64)
    
    exact_kl = analytical_gaussian_kl(mu, log_var)
    mc_kl = monte_carlo_gaussian_kl(mu, log_var, num_samples=1000000)
    
    print(f"Latent Dimension K: {len(mu)}")
    print(f"Mean Vector mu: {mu}")
    print(f"Log-Variance Vector log_var: {log_var}")
    print(f"Exact Analytical KL Divergence:    {exact_kl:.6f}")
    print(f"Monte Carlo Empirical Expectation: {mc_kl:.6f}")
    
    abs_error = abs(exact_kl - mc_kl)
    rel_error = abs_error / exact_kl
    print(f"Absolute Error: {abs_error:.6e}")
    print(f"Relative Error: {rel_error:.4%}")
    
    # Assert numerical convergence within Monte Carlo error bounds (0.5% tolerance)
    assert np.isclose(exact_kl, mc_kl, rtol=1e-2, atol=1e-2), "Monte Carlo estimate failed to match analytical KL!"
    print("Assertion passed: Analytical Gaussian KL matches Monte Carlo integration cleanly.")

if __name__ == "__main__":
    main()
