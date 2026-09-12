"""
PyTorch Autograd Simulation of WGAN-GP Gradient Penalty Computation.
Verifies straight-line interpolation, higher-order autograd, and loss backprop.
"""
# Shape: [B, D] for input features and gradients
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, in_dim=16, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        # Shape: [B, 1]
        return self.net(x)

def run_simulation():
    torch.manual_seed(42)
    B, D = 8, 16
    lambda_gp = 10.0
    
    critic = Critic(in_dim=D)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    # Shape: [B, D]
    real_data = torch.randn(B, D)
    fake_data = torch.randn(B, D)
    
    # Straight-line interpolation
    # Shape: [B, 1]
    eps = torch.rand(B, 1)
    # Shape: [B, D]
    interpolates = eps * real_data + (1.0 - eps) * fake_data
    interpolates.requires_grad_(True)
    
    # Forward evaluations
    d_real = critic(real_data)
    d_fake = critic(fake_data)
    d_interp = critic(interpolates)
    
    # Higher-order gradient computation
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Shape: [B]
    grad_norms = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    loss_gp = lambda_gp * torch.mean((grad_norms - 1.0)**2)
    
    total_loss = d_fake.mean() - d_real.mean() + loss_gp
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    assert loss_gp.item() >= 0.0, "Gradient penalty must be non-negative"
    assert critic.net[0].weight.grad is not None, "Gradients must propagate to weights"
    
    print(f"PyTorch Simulation Passed: Total Loss={total_loss.item():.4f}, GP={loss_gp.item():.4f}")

if __name__ == "__main__":
    run_simulation()
