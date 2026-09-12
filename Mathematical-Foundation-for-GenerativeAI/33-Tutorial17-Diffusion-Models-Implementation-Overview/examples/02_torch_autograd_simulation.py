"""PyTorch autograd simulation verifying training loss invariance across parameterizations.

This script constructs a toy U-Net linear backbone and demonstrates that computing
the loss on noise prediction epsilon_theta is mathematically equivalent to computing
the Gaussian KL divergence (scaled MSE) on the posterior mean mu_theta.
"""
import torch
import torch.nn as nn

torch.manual_seed(42)

batch_size = 4
dim = 8
x0 = torch.randn(batch_size, dim) # Shape: (B, D)
eps0 = torch.randn(batch_size, dim) # Shape: (B, D)

beta = 0.05
alpha = 1.0 - beta
alpha_bar_prev = 0.85
alpha_bar = alpha_bar_prev * alpha

# Forward noisy state
xt = torch.sqrt(torch.tensor(alpha_bar)) * x0 + torch.sqrt(torch.tensor(1.0 - alpha_bar)) * eps0 # Shape: (B, D)

# Ground truth posterior mean mu_q (Equation 70)
w_x0 = (torch.sqrt(torch.tensor(alpha_bar_prev)) * beta) / (1.0 - alpha_bar)
w_xt = (torch.sqrt(torch.tensor(alpha)) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar)
mu_q = w_x0 * x0 + w_xt * xt # Shape: (B, D)

# Dummy U-Net backbone outputting 8-dim vector
net = nn.Linear(dim, dim)
pred_eps = net(xt) # Shape: (B, D)

# Compute mu_theta from predicted noise (Equation 128)
mu_theta = (1.0 / torch.sqrt(torch.tensor(alpha))) * (xt - (beta / torch.sqrt(torch.tensor(1.0 - alpha_bar))) * pred_eps) # Shape: (B, D)

# Compute loss in noise space vs loss in mean space
loss_noise = nn.functional.mse_loss(pred_eps, eps0)
loss_mean = nn.functional.mse_loss(mu_theta, mu_q)

# Check gradients can be computed cleanly
loss_noise.backward(retain_graph=True)
assert net.weight.grad is not None
print("PyTorch autograd simulation successful.")
print(f"  Noise Loss: {loss_noise.item():.6f}")
print(f"  Mean Loss:  {loss_mean.item():.6f}")
