#!/usr/bin/env python3
"""
Example 01: MLP vs Local Receptive Field vs CNN Parameter Efficiency and Toeplitz Equivalence.

Demonstrates:
1. Exact parameter scaling comparison across input dimension d.
2. Embedding a convolutional kernel into a sparse banded Toeplitz matrix.
3. Mathematical equivalence of discrete convolution and Toeplitz matrix-vector multiplication.
4. Numerical proof of translation equivariance: f(T_v x) == T_v f(x).
"""

import numpy as np

def run_parameter_comparison():
    print("=" * 70)
    print("1. PARAMETER COUNT SCALING COMPARISON")
    print("=" * 70)
    
    resolutions = [32, 64, 128, 256, 512]  # 2D square image dimensions (H=W)
    k_size = 5  # 5x5 kernel (25 weights)
    
    print(f"{'Image Size (HxW)':<20} | {'Input Dim (d)':<15} | {'Dense MLP Params':<18} | {'Local RF (No Share)':<20} | {'CNN (Shared)':<12}")
    print("-" * 95)
    
    for res in resolutions:
        d = res * res
        # Assume hidden layer has matching spatial output dimension M = d
        mlp_params = d * d
        local_rf_params = d * (k_size * k_size)
        cnn_params = (k_size * k_size) + 1  # 25 weights + 1 bias
        
        mlp_mb = (mlp_params * 4) / (1024 ** 2)
        local_mb = (local_rf_params * 4) / (1024 ** 2)
        
        mlp_str = f"{mlp_params:,} ({mlp_mb:.1f} MB)" if mlp_mb < 1024 else f"{mlp_params:,} ({mlp_mb/1024:.1f} GB)"
        local_str = f"{local_rf_params:,} ({local_mb:.1f} MB)"
        cnn_str = f"{cnn_params} weights"
        
        print(f"{f'{res}x{res}':<20} | {d:<15,} | {mlp_str:<18} | {local_str:<20} | {cnn_str:<12}")
        
    print("\nTakeaway: At 256x256 resolution, an MLP requires 16 GB for a single layer,")
    print("whereas a CNN requires only 26 numbers (0.0001 MB) regardless of image resolution!\n")

def run_toeplitz_equivalence():
    print("=" * 70)
    print("2. DISCRETE CONVOLUTION AS A TOEPLITZ MATRIX MULTIPLICATION")
    print("=" * 70)
    
    np.random.seed(42)
    d = 7
    k = 3
    s = 1
    M = (d - k) // s + 1  # M = 5
    
    # 1D Input signal and kernel
    x = np.array([2.5, -1.0, 3.2, 0.5, -2.0, 1.8, 0.4])  # Shape: [7]
    w = np.array([0.5, -1.5, 2.0])                         # Shape: [3]
    b = 0.75                                               # Scalar bias
    
    # Method A: Direct sliding cross-correlation loop
    z_conv = np.zeros(M)
    for j in range(M):
        patch = x[j * s : j * s + k]
        z_conv[j] = np.dot(w, patch) + b
        
    # Method B: Banded Toeplitz matrix construction
    W_toeplitz = np.zeros((M, d))
    for j in range(M):
        W_toeplitz[j, j * s : j * s + k] = w
        
    b_vec = np.full(M, b)
    z_toeplitz = W_toeplitz @ x + b_vec
    
    print("Constructed Sparse Banded Toeplitz Matrix W (Shape: [5, 7]):")
    print(np.array2string(W_toeplitz, precision=2, suppress_small=True))
    print(f"\nConvolution Output:   {np.round(z_conv, 4)}")
    print(f"Toeplitz Matrix @ x:  {np.round(z_toeplitz, 4)}")
    
    assert np.allclose(z_conv, z_toeplitz), "Convolution and Toeplitz multiplication must be identical!"
    print("\n[PASS] Mathematical Equivalence Verified: z_conv == W_toeplitz @ x + b\n")

def run_translation_equivariance():
    print("=" * 70)
    print("3. NUMERICAL PROOF OF TRANSLATION EQUIVARIANCE")
    print("=" * 70)
    
    # Signal with a sharp local bump
    x = np.array([0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    w = np.array([1.0, -1.0])  # Difference / edge detector
    
    # Standard valid convolution: length = 8 - 2 + 1 = 7
    z = np.zeros(len(x) - len(w) + 1)
    for j in range(len(z)):
        z[j] = np.dot(w, x[j:j+len(w)])
        
    # Translate input right by 2 positions
    shift = 2
    x_shifted = np.roll(x, shift)
    x_shifted[:shift] = 0.0  # Zero out wrapped boundary
    
    z_shifted = np.zeros(len(x_shifted) - len(w) + 1)
    for j in range(len(z_shifted)):
        z_shifted[j] = np.dot(w, x_shifted[j:j+len(w)])
        
    print(f"Original Input x:         {x}")
    print(f"Convolved Feature Map z:  {z}")
    print(f"Shifted Input T_2(x):     {x_shifted}")
    print(f"Convolved Shifted Map z': {z_shifted}")
    
    # The output slice from shift onwards matches the original unshifted output
    assert np.allclose(z[:len(z)-shift], z_shifted[shift:]), "Equivariance check failed!"
    print("\n[PASS] Translation Equivariance Verified: f(T_v x) == T_v f(x)")
    print("=" * 70)

if __name__ == "__main__":
    run_parameter_comparison()
    run_toeplitz_equivalence()
    run_translation_equivariance()
    print("All simulations completed successfully (exit code 0).")
