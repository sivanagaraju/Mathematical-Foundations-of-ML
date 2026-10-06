"""
Mathematical and Numerical Proof of Variance Scaling in Dot-Product Attention.

This script empirically proves why dividing by sqrt(D_k) is load-bearing in modern
transformer architectures:
1. For independent zero-mean unit-variance components q_i, k_i ~ N(0, 1):
       Var(q^T k) = sum_{i=1}^{D_k} Var(q_i k_i) = D_k
2. Normalization by 1/sqrt(D_k) scales the variance back to 1.0:
       Var( (q^T k) / sqrt(D_k) ) = (1 / D_k) * D_k = 1.0
3. In the unscaled regime for large D_k, dot product logits explode, pushing the
   softmax function into regions of near-zero gradients (vanishing gradient trap).
"""

import math
import torch
import torch.nn.functional as F

def verify_dot_product_variance():
    torch.manual_seed(42)
    print("=" * 70)
    print("Verification 1: Empirical Dot-Product Variance vs Dimension D_k")
    print("=" * 70)
    
    dimensions = [16, 64, 256, 1024, 4096]
    n_samples = 20000
    
    print(f"{'D_k':>6} | {'Empirical Var(q^T k)':>22} | {'Expected D_k':>12} | {'Var(scaled)':>14} | {'Target':>8}")
    print("-" * 72)
    
    for d in dimensions:
        # Sample independent vectors from standard normal distribution
        # Shape: [N, D_k]
        q = torch.randn(n_samples, d)  # Shape: [20000, D_k]
        k = torch.randn(n_samples, d)  # Shape: [20000, D_k]
        
        # Raw dot products: dot = sum(q * k, dim=-1)
        raw_dots = (q * k).sum(dim=-1)  # Shape: [N]
        scaled_dots = raw_dots / math.sqrt(d)  # Shape: [N]
        
        emp_var_raw = raw_dots.var().item()
        emp_var_scaled = scaled_dots.var().item()
        
        print(f"{d:>6} | {emp_var_raw:>22.2f} | {d:>12} | {emp_var_scaled:>14.4f} | {'1.0000':>8}")
        
        # Assert variance scales linearly with D_k (within 5% Monte Carlo tolerance)
        rel_err_raw = abs(emp_var_raw - d) / d
        assert rel_err_raw < 0.05, f"Expected Var(raw) ~ {d}, got {emp_var_raw:.2f}"
        
        # Assert scaled variance remains approximately 1.0 (within 5% tolerance)
        assert abs(emp_var_scaled - 1.0) < 0.05, f"Expected Var(scaled) ~ 1.0, got {emp_var_scaled:.4f}"
        
    print("\n[Assertion 1 PASS] Variance scaling strictly tracks D_k and normalizes to 1.0.")

def verify_gradient_saturation():
    print("\n" + "=" * 70)
    print("Verification 2: Softmax Gradient Saturation and Vanishing Gradients")
    print("=" * 70)
    
    d_k = 512
    n_tokens = 64
    
    # 1. Scaled Attention Setup
    q_scaled = torch.randn(1, n_tokens, d_k, requires_grad=True)  # Shape: [1, 64, 512]
    k_scaled = torch.randn(1, n_tokens, d_k, requires_grad=True)  # Shape: [1, 64, 512]
    
    # Scaled logits: (Q K^T) / sqrt(D_k)
    logits_scaled = torch.matmul(q_scaled, k_scaled.transpose(-1, -2)) / math.sqrt(d_k)  # Shape: [1, 64, 64]
    probs_scaled = F.softmax(logits_scaled, dim=-1)  # Shape: [1, 64, 64]
    
    # Target loss on an arbitrary element to measure gradient flow
    target_loss_scaled = probs_scaled[0, 0, 5]
    target_loss_scaled.backward()
    grad_target_scaled = q_scaled.grad.norm().item()
    
    # 2. Unscaled Attention Setup (large logits causing saturation)
    q_unscaled = q_scaled.detach().clone().requires_grad_(True)  # Shape: [1, 64, 512]
    k_unscaled = k_scaled.detach().clone().requires_grad_(True)  # Shape: [1, 64, 512]
    
    logits_unscaled = torch.matmul(q_unscaled, k_unscaled.transpose(-1, -2))  # Shape: [1, 64, 64]
    probs_unscaled = F.softmax(logits_unscaled, dim=-1)  # Shape: [1, 64, 64]
    
    target_loss_unscaled = probs_unscaled[0, 0, 5]
    target_loss_unscaled.backward()
    grad_target_unscaled = q_unscaled.grad.norm().item()
    
    # Softmax entropy: H(p) = -sum(p * log(p))
    entropy_scaled = -(probs_scaled * torch.log(probs_scaled + 1e-12)).sum(dim=-1).mean().item()
    entropy_unscaled = -(probs_unscaled * torch.log(probs_unscaled + 1e-12)).sum(dim=-1).mean().item()
    
    print(f"Results for sequence length T = {n_tokens}, feature dimension D_k = {d_k}:")
    print(f"  Unscaled Logits Variance:   {logits_unscaled.var().item():.2f}")
    print(f"  Scaled Logits Variance:     {logits_scaled.var().item():.2f}")
    print(f"  Unscaled Attention Entropy: {entropy_unscaled:.4f} nats (close to 0 = collapsed peak)")
    print(f"  Scaled Attention Entropy:   {entropy_scaled:.4f} nats (rich distributed attention)")
    print(f"  Unscaled Gradient Norm:     {grad_target_unscaled:.4e}")
    print(f"  Scaled Gradient Norm:       {grad_target_scaled:.4e}")
    
    # Assert that unscaled gradients vanish significantly compared to scaled gradients
    assert grad_target_unscaled < grad_target_scaled * 0.1 or grad_target_unscaled < 1e-4, \
        "Unscaled gradients should be severely attenuated due to softmax saturation"
    print(f"\n[Assertion 2 PASS] Gradient vanishing demonstrated: unscaled gradient is heavily attenuated.")

if __name__ == "__main__":
    verify_dot_product_variance()
    verify_gradient_saturation()
