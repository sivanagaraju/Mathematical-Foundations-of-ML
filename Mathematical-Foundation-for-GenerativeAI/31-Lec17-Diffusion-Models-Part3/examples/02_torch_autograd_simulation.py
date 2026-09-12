"""Autograd simulation of multi-head time conditioning."""
import torch
import torch.nn as nn

class TimeModulatedLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 2))
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, x, t_emb):
        # Shape: x is (B, D), t_emb is (B, D)
        scale_shift = self.time_mlp(t_emb) # Shape: (B, 2*D)
        scale, shift = scale_shift.chunk(2, dim=-1) # Shapes: (B, D), (B, D)
        h = self.linear(x) # Shape: (B, D)
        return (1.0 + scale) * h + shift # Shape: (B, D)

def test_autograd_time_conditioning():
    torch.manual_seed(42)
    B, D = 8, 16
    layer = TimeModulatedLayer(D)
    
    x = torch.randn(B, D) # Shape: (B, D)
    t_emb = torch.randn(B, D) # Shape: (B, D)
    
    out = layer(x, t_emb) # Shape: (B, D)
    loss = out.sum()
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in layer.parameters() if p.grad is not None)
    assert grad_norm > 0.0, "Gradients must propagate through time conditioning layer"
    print(f"Autograd time conditioning PASSED. Grad norm: {grad_norm:.4f}")

if __name__ == "__main__":
    test_autograd_time_conditioning()
