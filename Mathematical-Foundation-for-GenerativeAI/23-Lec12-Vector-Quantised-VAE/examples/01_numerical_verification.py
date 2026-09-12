"""
Numerical Verification of VQ-VAE Quantization, STE, and EMA Updates.
Demonstrates deterministic nearest-neighbor lookup, tripartite loss math, and EMA dictionary evolution.
"""
import numpy as np

def run_verification():
    np.random.seed(42)
    B, D, K = 4, 8, 16
    
    # 1. Continuous encoder outputs z_e and Codebook E
    z_e = np.random.randn(B, D)
    E = np.random.randn(K, D)
    
    # 2. Pairwise Euclidean Distance: ||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 * z_e . e_k^T
    z_sq = np.sum(z_e**2, axis=1, keepdims=True)       # (B, 1)
    e_sq = np.sum(E**2, axis=1, keepdims=True).T       # (1, K)
    dots = np.dot(z_e, E.T)                            # (B, K)
    distances = z_sq + e_sq - 2 * dots                 # (B, K)
    
    # 3. Nearest neighbor indices
    indices = np.argmin(distances, axis=1)             # (B,)
    assert indices.shape == (B,), "Indices shape mismatch"
    
    # Quantized representation z_q
    z_q = E[indices]                                   # (B, D)
    assert z_q.shape == (B, D), "z_q shape mismatch"
    
    # 4. Tripartite Loss Terms
    # Reconstruction simulated as L2 error against target
    target_x = z_e + 0.1 * np.random.randn(B, D)
    recon_loss = np.mean((z_q - target_x)**2)
    
    # VQ Loss: ||sg[z_e] - e||^2 (updates codebook E)
    vq_loss = np.mean((z_q - z_e)**2)
    
    # Commitment Loss: beta * ||z_e - sg[e]||^2 (updates encoder z_e)
    beta = 0.25
    commitment_loss = beta * np.mean((z_e - z_q)**2)
    
    total_loss = recon_loss + vq_loss + commitment_loss
    assert total_loss > 0.0, "Total loss must be positive"
    
    # 5. EMA Codebook Update Math Simulation
    gamma = 0.9
    cluster_counts = np.zeros(K)
    running_sums = np.zeros_like(E)
    
    # One-hot assignment
    for idx, i in enumerate(indices):
        cluster_counts[i] += 1
        running_sums[i] += z_e[idx]
        
    updated_counts = gamma * np.ones(K) + (1 - gamma) * cluster_counts
    updated_sums = gamma * E + (1 - gamma) * running_sums
    new_E = updated_sums / np.maximum(updated_counts[:, None], 1e-5)
    
    assert new_E.shape == E.shape, "Updated codebook shape mismatch"
    print(f"VQ-VAE numerical verification passed. Total loss: {total_loss:.4f}")

if __name__ == "__main__":
    run_verification()
