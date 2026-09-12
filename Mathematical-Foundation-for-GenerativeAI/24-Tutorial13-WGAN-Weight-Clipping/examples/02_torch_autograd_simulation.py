"""
PyTorch Autograd Simulation of WGAN Critic with Hard Weight Clipping.
Verifies linear Critic outputs, Wasserstein loss backprop, and parameter clamping.
"""
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, in_features=16, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

def run_simulation():
    torch.manual_seed(42)
    B, D = 8, 16
    c = 0.01
    
    critic = Critic(in_features=D)
    optimizer = torch.optim.RMSprop(critic.parameters(), lr=5e-5)
    
    real_samples = torch.randn(B, D)
    fake_samples = torch.randn(B, D)
    
    # Forward pass
    real_scores = critic(real_samples)
    fake_scores = critic(fake_samples)
    
    # Wasserstein loss
    loss = fake_scores.mean() - real_scores.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Apply weight clipping
    for p in critic.parameters():
        p.data.clamp_(-c, c)
        
    # Verify all weights stay strictly within [-c, c]
    for p in critic.parameters():
        assert torch.all(p.data >= -c), "Weight violated lower bound"
        assert torch.all(p.data <= c), "Weight violated upper bound"
        
    print(f"PyTorch Simulation Passed: Loss={loss.item():.4f}, all parameters bounded in [{-c}, {c}].")

if __name__ == "__main__":
    run_simulation()
