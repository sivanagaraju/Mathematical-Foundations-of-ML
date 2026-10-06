"""
Verification Script 1: Exact BPTT Jacobian Chain & Vanishing Gradient Numerical Proof.

This script implements:
1. Analytical Jacobian computation: J_{k+1, k} = diag(sigma'(z_{k+1})) * W_hh.
2. Cumulative temporal chain rule: d h_T / d h_t = prod_{k=t}^{T-1} J_{k+1, k}.
3. Exact numerical parity against PyTorch Autograd.
4. Empirical verification of the sub-multiplicative exponential bound (gamma * lambda_max)^{T-t}.
"""

import numpy as np
import torch

def verify_bptt_jacobian_chain():
    print("=== Step 1: Analytical vs PyTorch BPTT Jacobian Parity ===")
    torch.manual_seed(42)
    np.random.seed(42)
    
    T = 6
    m = 3
    d = 2
    
    # Initialize weights
    W_hh_np = np.random.randn(m, m) * 0.6
    W_xh_np = np.random.randn(m, d) * 0.5
    b_h_np = np.zeros(m)
    
    # Check spectral norm of W_hh
    singular_values = np.linalg.svd(W_hh_np, compute_uv=False)
    lambda_max = singular_values[0]
    print(f"Recurrent matrix spectral norm ||W_hh||_2 = {lambda_max:.4f}")
    
    # Input sequence
    X_np = np.random.randn(T, d)
    
    # PyTorch implementation
    W_hh_pt = torch.tensor(W_hh_np, dtype=torch.float64, requires_grad=True)
    W_xh_pt = torch.tensor(W_xh_np, dtype=torch.float64, requires_grad=True)
    b_h_pt = torch.tensor(b_h_np, dtype=torch.float64, requires_grad=True)
    
    h_states_pt = []
    z_states_pt = []
    
    h_curr = torch.zeros(m, dtype=torch.float64)
    for t in range(T):
        x_t = torch.tensor(X_np[t], dtype=torch.float64)
        z_t = W_hh_pt @ h_curr + W_xh_pt @ x_t + b_h_pt
        h_curr = torch.tanh(z_t)
        h_states_pt.append(h_curr)
        z_states_pt.append(z_t)
    
    # Suppose loss is sum of terminal hidden state: L = sum(h_T)
    loss = h_states_pt[-1].sum()
    
    # We want to verify analytical dh_T / dh_t for t = 0, 1, ..., T-2
    # In PyTorch, dL / dh_t can be retrieved by creating a hook or retaining grads
    # Let's rerun forward pass retaining grad on intermediate h
    h_intermediates = []
    h_curr = torch.zeros(m, dtype=torch.float64)
    for t in range(T):
        x_t = torch.tensor(X_np[t], dtype=torch.float64)
        z_t = W_hh_pt @ h_curr + W_xh_pt @ x_t + b_h_pt
        h_curr = torch.tanh(z_t)
        h_curr.retain_grad()
        h_intermediates.append(h_curr)
    
    loss = h_intermediates[-1].sum()
    loss.backward()
    
    # Analytical Jacobian chain
    # Cache forward activations in numpy
    h_np = [np.zeros(m)]
    z_np = []
    for t in range(T):
        z_t = W_hh_np @ h_np[-1] + W_xh_np @ X_np[t] + b_h_np
        z_np.append(z_t)
        h_np.append(np.tanh(z_t))
    
    # Compute local Jacobians J_{k+1, k} = diag(1 - tanh^2(z_{k+1})) * W_hh
    # For k in 0, ..., T-2 (where states are h_1 to h_T)
    # Note: h_intermediates[t] corresponds to h_{t+1} (1-indexed)
    print("\nVerifying backpropagated error sensitivities dL/dh_t:")
    dL_dh_terminal = np.ones(m) # d(sum(h_T)) / dh_T = [1, 1, 1]
    
    # Backpropagate analytically
    dL_dh_analytical = [None] * T
    dL_dh_analytical[-1] = dL_dh_terminal
    
    for t in range(T - 2, -1, -1):
        # State at t+1 is produced from state at t:
        # z_{t+1} = W_hh @ h_t + W_xh @ x_{t+1} + b
        z_next = z_np[t + 1]
        sigma_prime = 1.0 - np.tanh(z_next)**2
        J_next_curr = np.diag(sigma_prime) @ W_hh_np
        # dL / dh_t = (dL / dh_{t+1}) @ J_{t+1, t}
        dL_dh_analytical[t] = dL_dh_analytical[t + 1] @ J_next_curr
        
        # Compare with PyTorch
        pt_grad = h_intermediates[t].grad.numpy()
        np.testing.assert_allclose(dL_dh_analytical[t], pt_grad, rtol=1e-5, atol=1e-7)
        print(f"  Step t={t}: analytical norm={np.linalg.norm(dL_dh_analytical[t]):.6e} matches PyTorch ({np.linalg.norm(pt_grad):.6e})")
    
    print("=== Step 2: Exponential Vanishing Bound Verification ===")
    # Theoretical bound check: ||dL/dh_t||_2 <= ||dL/dh_T||_2 * (gamma * lambda_max)^{T-1 - t}
    gamma = 1.0 # for tanh
    bound_decay_rate = gamma * lambda_max
    
    for t in range(T - 1):
        steps_back = (T - 1) - t
        theoretical_bound = np.linalg.norm(dL_dh_terminal) * (bound_decay_rate ** steps_back)
        actual_norm = np.linalg.norm(dL_dh_analytical[t])
        assert actual_norm <= theoretical_bound + 1e-6
        print(f"  Gap {steps_back} steps: actual norm={actual_norm:.6e} <= bound={theoretical_bound:.6e}")
        
    print("\n[SUCCESS] Script 1: BPTT Jacobian Chain & Vanishing Bound Verified Cleanly.")

if __name__ == "__main__":
    verify_bptt_jacobian_chain()
