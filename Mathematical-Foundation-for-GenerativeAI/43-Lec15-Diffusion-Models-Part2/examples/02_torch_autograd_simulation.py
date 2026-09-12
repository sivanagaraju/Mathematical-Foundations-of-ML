import torch
import torch.nn as nn

class ToyUNetDenoiser(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32),
            nn.SiLU(),
            nn.Linear(32, d)
        )
    def forward(self, xt, t):
        # Shape: xt is (B, D)
        return self.net(xt) # Shape: (B, D)

def test_autograd_l_simple():
    torch.manual_seed(42)
    B, D = 16, 4
    model = ToyUNetDenoiser(D)
    
    x0 = torch.randn(B, D) # Shape: (B, D)
    eps = torch.randn_like(x0) # Shape: (B, D)
    alpha_bar = torch.tensor([0.65])
    
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * eps # Shape: (B, D)
    t = torch.randint(0, 1000, (B,))
    
    pred_eps = model(xt, t) # Shape: (B, D)
    loss = nn.functional.mse_loss(pred_eps, eps)
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0, "Gradients must propagate through model"
    print(f"Autograd L_simple passed. Loss: {loss.item():.4f}, Grad norm: {grad_norm:.4f}")

if __name__ == "__main__":
    test_autograd_l_simple()
