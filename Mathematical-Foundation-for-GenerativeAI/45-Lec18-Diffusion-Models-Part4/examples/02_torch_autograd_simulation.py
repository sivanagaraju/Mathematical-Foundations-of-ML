"""Autograd simulation of Latent Diffusion cross-attention conditioning."""
import torch
import torch.nn as nn

class MiniLatentCrossAttention(nn.Module):
    def __init__(self, d_latent, d_context):
        super().__init__()
        self.to_q = nn.Linear(d_latent, d_latent)
        self.to_k = nn.Linear(d_context, d_latent)
        self.to_v = nn.Linear(d_context, d_latent)
        self.out = nn.Linear(d_latent, d_latent)
        
    def forward(self, z, context):
        # Shape: z is (B, D_l), context is (B, D_c)
        q = self.to_q(z) # Shape: (B, D_l)
        k = self.to_k(context) # Shape: (B, D_l)
        v = self.to_v(context) # Shape: (B, D_l)
        
        # Dot-product attention scalar per sample
        scores = torch.sum(q * k, dim=-1, keepdim=True) / (q.shape[-1]**0.5) # Shape: (B, 1)
        attn = torch.sigmoid(scores) # Shape: (B, 1)
        h = attn * v # Shape: (B, D_l)
        return self.out(h) # Shape: (B, D_l)

def test_autograd_latent_attention():
    torch.manual_seed(42)
    B, D_l, D_c = 4, 16, 32
    layer = MiniLatentCrossAttention(D_l, D_c)
    
    z = torch.randn(B, D_l) # Shape: (B, D_l)
    context = torch.randn(B, D_c) # Shape: (B, D_c)
    
    out = layer(z, context) # Shape: (B, D_l)
    loss = out.sum()
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in layer.parameters() if p.grad is not None)
    assert grad_norm > 0.0, "Gradients must propagate through cross-attention layer"
    print(f"Autograd Latent Attention PASSED. Grad norm: {grad_norm:.4f}")

if __name__ == "__main__":
    test_autograd_latent_attention()
