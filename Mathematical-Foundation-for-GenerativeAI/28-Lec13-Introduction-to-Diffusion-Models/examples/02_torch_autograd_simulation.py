import torch
import torch.nn as nn

class MiniDenoiser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, dim)
        )
        self.net = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, dim)
        )
        
    def forward(self, x, t):
        # Shape: x is (B, D), t is (B, 1)
        t_embed = self.time_mlp(t) # Shape: (B, D)
        inp = torch.cat([x, t_embed], dim=-1) # Shape: (B, 2*D)
        return self.net(inp) # Shape: (B, D)

def test_autograd_diffusion_step():
    torch.manual_seed(42)
    B, D = 8, 4
    model = MiniDenoiser(D)
    
    x0 = torch.randn(B, D) # Shape: (B, D)
    t = torch.randint(0, 100, (B, 1)).float() / 100.0 # Shape: (B, 1)
    
    # Inject synthetic noise
    noise = torch.randn_like(x0) # Shape: (B, D)
    alpha_bar = torch.tensor([0.7])
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise # Shape: (B, D)
    
    # Model predicts noise
    pred_noise = model(xt, t) # Shape: (B, D)
    loss = nn.functional.mse_loss(pred_noise, noise)
    
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0, "Gradients must propagate through denoiser network"
    print(f"Autograd simulation passed. Loss: {loss.item():.4f}, Grad norm: {grad_norm:.4f}")

if __name__ == "__main__":
    test_autograd_diffusion_step()
