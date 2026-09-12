"""
Numerical Verification of Vectorized Distance Expansion and Codebook Perplexity.
Demonstrates expanded Euclidean distance calculation and Shannon entropy perplexity.
"""
# Shape: [N, D] for input features, [K, D] for codebook prototypes
import numpy as np

def run_verification():
    np.random.seed(42)
    N, D, K = 8, 4, 16
    
    # Shape: [N, D] and [K, D]
    z = np.random.randn(N, D)
    E = np.random.randn(K, D)
    
    # 1. Vectorized Distance Expansion: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z * e^T
    z_sq = np.sum(z**2, axis=1, keepdims=True) # (N, 1)
    e_sq = np.sum(E**2, axis=1, keepdims=True).T # (1, K)
    dots = np.dot(z, E.T) # (N, K)
    distances = z_sq + e_sq - 2 * dots # (N, K)
    
    # Verify against direct pairwise distance
    direct_dists = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            direct_dists[i, j] = np.sum((z[i] - E[j])**2)
            
    assert np.allclose(distances, direct_dists), "Expanded distance calculation mismatch"
    
    # Nearest neighbor indices
    indices = np.argmin(distances, axis=1) # Shape: [N]
    assert indices.shape == (N,)
    
    # 2. Perplexity Calculation
    counts = np.bincount(indices, minlength=K)
    probs = counts / N
    # Entropy: -sum(p * log(p))
    active_probs = probs[probs > 0]
    entropy = -np.sum(active_probs * np.log(active_probs))
    perplexity = np.exp(entropy)
    
    assert 1.0 <= perplexity <= K, "Perplexity must be bounded in [1, K]"
    
    print(f"Numerical Verification Passed: Perplexity={perplexity:.2f} out of {K} codes.")
    print(f"  Nearest indices: {indices.tolist()}")

if __name__ == "__main__":
    run_verification()
