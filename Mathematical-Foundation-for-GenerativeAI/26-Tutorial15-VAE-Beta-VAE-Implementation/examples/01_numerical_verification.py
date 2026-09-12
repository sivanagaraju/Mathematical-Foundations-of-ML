"""
Numerical Verification of Analytical Gaussian KL Divergence and Beta Scaling.
Demonstrates exact closed-form relative entropy across varying beta values.
"""
# Shape: [B, K] for latent parameter tensors
import numpy as np

def run_verification():
    np.random.seed(42)
    B, K = 4, 8
    
    # Shape: [B, K]
    mu = np.random.randn(B, K) * 0.5
    logvar = np.random.randn(B, K) * 0.2
    
    # 1. Closed-form KL calculation: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    var = np.exp(logvar)
    kl_per_dim = -0.5 * (1.0 + logvar - mu**2 - var)
    assert np.all(kl_per_dim >= 0.0), "Relative entropy per dimension must be non-negative"
    
    # Shape: [B]
    kl_per_sample = np.sum(kl_per_dim, axis=1)
    avg_kl = np.mean(kl_per_sample)
    
    # 2. Beta scaling check
    recon_loss = 120.0
    betas = [0.1, 1.0, 4.0, 10.0]
    total_losses = [recon_loss + b * avg_kl for b in betas]
    
    # Total loss must strictly increase with beta
    diffs = np.diff(total_losses)
    assert np.all(diffs > 0), "Total loss must increase monotonically with beta"
    
    # 3. Test exact zero KL when mu=0, logvar=0
    zero_kl = -0.5 * np.sum(1.0 + 0.0 - 0.0 - np.exp(0.0))
    assert np.isclose(zero_kl, 0.0), "Zero divergence for identical distributions"
    
    print(f"Numerical Verification Passed: Avg KL={avg_kl:.4f} nats.")
    for b, l in zip(betas, total_losses):
        print(f"  beta={b:.1f} -> total_loss={l:.2f}")

if __name__ == "__main__":
    run_verification()
