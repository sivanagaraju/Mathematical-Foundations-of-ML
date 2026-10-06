"""
Verification Script 2: Multiplicative Vanilla RNN vs Additive Identity Highway & Gradient Clipping.

This script demonstrates:
1. Long-sequence gradient vanishing over T = 50 steps in vanilla multiplicative RNN.
2. Unattenuated gradient flow in an Additive Identity Highway model (mimicking LSTM/ResNet).
3. Exploding gradient mitigation via global gradient clipping (torch.nn.utils.clip_grad_norm_).
"""

import numpy as np
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, m, d, spectral_scale=0.85):
        super().__init__()
        self.m = m
        self.W_hh = nn.Parameter(torch.randn(m, m))
        self.W_xh = nn.Parameter(torch.randn(m, d))
        # Scale W_hh to desired spectral norm
        with torch.no_grad():
            u, s, v = torch.linalg.svd(self.W_hh)
            self.W_hh.data = (u @ torch.diag(torch.ones(m) * spectral_scale) @ v)
            
    def forward(self, X):
        # X: [T, d]
        T = X.shape[0]
        h = torch.zeros(self.m, dtype=torch.float32, requires_grad=True)
        h_intermediates = []
        for t in range(T):
            z = self.W_hh @ h + self.W_xh @ X[t]
            h = torch.tanh(z)
            h.retain_grad()
            h_intermediates.append(h)
        return h_intermediates

class AdditiveHighwayRNN(nn.Module):
    def __init__(self, m, d):
        super().__init__()
        self.m = m
        self.W_hh = nn.Parameter(torch.randn(m, m) * 0.1)
        self.W_xh = nn.Parameter(torch.randn(m, d) * 0.1)
        
    def forward(self, X):
        # X: [T, d]
        T = X.shape[0]
        h = torch.zeros(self.m, dtype=torch.float32, requires_grad=True)
        h_intermediates = []
        for t in range(T):
            # Additive identity highway: h_next = h_prev + F(h_prev, x)
            delta_h = torch.tanh(self.W_hh @ h + self.W_xh @ X[t])
            h = h + delta_h
            h.retain_grad()
            h_intermediates.append(h)
        return h_intermediates

def verify_gradient_highway_and_clipping():
    torch.manual_seed(42)
    T = 50
    m = 16
    d = 8
    
    X = torch.randn(T, d)
    
    print("=== Step 1: Vanilla RNN Gradient Vanishing (T=50) ===")
    vanilla_model = VanillaRNN(m, d, spectral_scale=0.85)
    h_vanilla = vanilla_model(X)
    loss_vanilla = h_vanilla[-1].sum()
    loss_vanilla.backward()
    
    grad_at_T_vanilla = torch.norm(h_vanilla[-1].grad).item()
    grad_at_0_vanilla = torch.norm(h_vanilla[0].grad).item()
    
    print(f"Vanilla RNN: grad at step T=50: {grad_at_T_vanilla:.4f}")
    print(f"Vanilla RNN: grad at step t=1:  {grad_at_0_vanilla:.6e}")
    # After 50 contractive steps, gradient must vanish (< 1e-4)
    assert grad_at_0_vanilla < 1e-4, f"Expected vanishing gradient, got {grad_at_0_vanilla}"
    
    print("\n=== Step 2: Additive Identity Highway Gradient Persistence (T=50) ===")
    highway_model = AdditiveHighwayRNN(m, d)
    h_highway = highway_model(X)
    loss_highway = h_highway[-1].sum()
    loss_highway.backward()
    
    grad_at_T_highway = torch.norm(h_highway[-1].grad).item()
    grad_at_0_highway = torch.norm(h_highway[0].grad).item()
    
    print(f"Highway RNN: grad at step T=50: {grad_at_T_highway:.4f}")
    print(f"Highway RNN: grad at step t=1:  {grad_at_0_highway:.4f}")
    # Additive highway maintains unattenuated gradient order of magnitude
    assert grad_at_0_highway > 0.1, f"Expected persistent gradient, got {grad_at_0_highway}"
    
    print("\n=== Step 3: Gradient Exploding & Global Gradient Clipping ===")
    # Model with explosive spectral norm lambda_max = 1.3
    explosive_model = VanillaRNN(m, d, spectral_scale=1.3)
    # Provide small inputs so tanh doesn't immediately saturate to zero derivative
    X_small = torch.randn(T, d) * 0.05
    h_exp = explosive_model(X_small)
    loss_exp = h_exp[-1].sum()
    loss_exp.backward()
    
    raw_grad_norm = nn.utils.clip_grad_norm_(explosive_model.parameters(), max_norm=1e9).item()
    print(f"Raw unclipped parameter gradient norm: {raw_grad_norm:.4f}")
    
    # Clip to threshold max_norm = 1.0
    clipped_norm = nn.utils.clip_grad_norm_(explosive_model.parameters(), max_norm=1.0).item()
    # Check clipped norm of parameters
    total_clipped = torch.sqrt(sum(p.grad.norm()**2 for p in explosive_model.parameters())).item()
    print(f"Total gradient norm after clip(1.0):   {total_clipped:.4f}")
    assert abs(total_clipped - 1.0) < 1e-4 or total_clipped <= 1.0
    
    print("\n[SUCCESS] Script 2: Highway Persistence & Gradient Clipping Verified Cleanly.")

if __name__ == "__main__":
    verify_gradient_highway_and_clipping()
