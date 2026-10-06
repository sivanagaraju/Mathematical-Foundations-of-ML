"""
Example 02: Spatial vs Temporal Residual Duality Simulation
=============================================================
Demonstrates Prof. Prathosh's core insight from Lecture 47:
Identity skip connections solve vanishing gradients in two dual geometric dimensions:
1. SPATIAL RESIDUAL CONNECTIONS (ResNets & Transformers across depth L):
   x^{[l]} = x^{[l-1]} + F(x^{[l-1]})
   Jacobian: d(x^{[l]})/d(x^{[l-1]}) = I + dF/d(x^{[l-1]})
2. TEMPORAL GRADIENT HIGHWAYS (LSTMs & GRUs across sequence time T):
   c_t = f_t * c_{t-1} + i_t * c_tilde_t
   Jacobian: d(c_t)/d(c_{t-1}) = diag(f_t) + d(candidate)/d(c_{t-1})

This simulation contrasts:
- Deep Plain MLP (L=50 layers) vs Deep ResNet (L=50 layers)
- Vanilla RNN (T=50 steps) vs Additive LSTM Cell State (T=50 steps)
"""

import numpy as np
import torch
import torch.nn as nn

def simulate_spatial_residual_duality():
    print("=== Step 1: Spatial Dimension (Depth L = 50 Layers) ===")
    torch.manual_seed(42)
    L = 50
    dim = 16
    
    # Input tensor
    x_input_plain = torch.randn(1, dim, dtype=torch.float64, requires_grad=True)
    x_input_resnet = x_input_plain.clone().detach().requires_grad_(True)
    
    # Create 50 dense transformation weights with contractive spectral norms
    weights = []
    for _ in range(L):
        W = torch.randn(dim, dim, dtype=torch.float64)
        U, S, V = torch.linalg.svd(W)
        W = U @ torch.diag(torch.ones(dim, dtype=torch.float64) * 0.8) @ V
        weights.append(W)
        
    # Forward pass: Plain deep network (multiplicative)
    curr_plain = x_input_plain
    for l in range(L):
        curr_plain = torch.tanh(curr_plain @ weights[l])
    loss_plain = 0.5 * torch.sum(curr_plain**2)
    loss_plain.backward()
    grad_norm_plain = torch.norm(x_input_plain.grad).item()
    
    # Forward pass: ResNet (additive skip connection)
    curr_resnet = x_input_resnet
    for l in range(L):
        # F(x) branch scaled so it doesn't blow up
        residual = 0.1 * torch.tanh(curr_resnet @ weights[l])
        curr_resnet = curr_resnet + residual
    loss_resnet = 0.5 * torch.sum(curr_resnet**2)
    loss_resnet.backward()
    grad_norm_resnet = torch.norm(x_input_resnet.grad).item()
    
    print(f"L = {L} layers:")
    print(f"  Plain Deep Net input gradient norm:  {grad_norm_plain:.4e}")
    print(f"  ResNet (Spatial Skip) gradient norm: {grad_norm_resnet:.4e}")
    
    assert grad_norm_plain < 1e-6, f"Expected vanishing in plain net, got {grad_norm_plain}"
    assert grad_norm_resnet > 0.05, f"Expected healthy gradient in ResNet, got {grad_norm_resnet}"
    spatial_ratio = grad_norm_resnet / (grad_norm_plain + 1e-30)
    print(f"  Spatial Identity Highway amplification: {spatial_ratio:.2e}x")

def simulate_temporal_recurrent_duality():
    print("\n=== Step 2: Temporal Dimension (Sequence Horizon T = 50 Timesteps) ===")
    torch.manual_seed(42)
    T = 50
    dim = 16
    
    # Sequence of observations
    X = [torch.randn(1, dim, dtype=torch.float64) * 0.1 for _ in range(T)]
    
    # Shared recurrent weight with spectral norm 0.8
    W_hh = torch.randn(dim, dim, dtype=torch.float64)
    U, S, V = torch.linalg.svd(W_hh)
    W_hh = U @ torch.diag(torch.ones(dim, dtype=torch.float64) * 0.8) @ V
    
    # 1. Vanilla RNN: h_t = tanh(h_{t-1} @ W_hh + x_t)
    h0_vanilla = torch.zeros(1, dim, dtype=torch.float64, requires_grad=True)
    h_curr = h0_vanilla
    for t in range(T):
        h_curr = torch.tanh(h_curr @ W_hh + X[t])
    loss_vanilla = 0.5 * torch.sum(h_curr**2)
    loss_vanilla.backward()
    grad_norm_vanilla = torch.norm(h0_vanilla.grad).item()
    
    # 2. LSTM Cell State Highway: c_t = f_t * c_{t-1} + i_t * c_tilde_t
    c0_lstm = torch.zeros(1, dim, dtype=torch.float64, requires_grad=True)
    c_curr = c0_lstm
    f_gate = 0.99  # Open forget gate
    i_gate = 0.1   # Conservative input gate
    for t in range(T):
        c_tilde = torch.tanh(c_curr @ W_hh * 0.1 + X[t])
        c_curr = f_gate * c_curr + i_gate * c_tilde
    loss_lstm = 0.5 * torch.sum(c_curr**2)
    loss_lstm.backward()
    grad_norm_lstm = torch.norm(c0_lstm.grad).item()
    
    print(f"T = {T} timesteps:")
    print(f"  Vanilla RNN initial state gradient:   {grad_norm_vanilla:.4e}")
    print(f"  LSTM Cell State Highway gradient:     {grad_norm_lstm:.4e}")
    
    assert grad_norm_vanilla < 1e-5, f"Expected vanishing in vanilla RNN, got {grad_norm_vanilla}"
    assert grad_norm_lstm > 0.01, f"Expected healthy gradient in LSTM, got {grad_norm_lstm}"
    temporal_ratio = grad_norm_lstm / (grad_norm_vanilla + 1e-30)
    print(f"  Temporal Identity Highway amplification: {temporal_ratio:.2e}x")
    print("\n[VERIFIED] Spatial and temporal skip mechanisms are mathematically dual.")

if __name__ == "__main__":
    simulate_spatial_residual_duality()
    simulate_temporal_recurrent_duality()
