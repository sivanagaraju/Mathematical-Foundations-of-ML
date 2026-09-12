"""
PyTorch Autograd Simulation of Modular VectorQuantizer and VQ-VAE.
Verifies straight-through estimator gradient flow and tripartite loss backward pass.
"""
# Shape: [B, C, H, W] for image batches, [B, D, H', W'] for latent grids
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=8, embedding_dim=4, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, inputs):
        # Shape: inputs is [B, D, H, W] -> permute to [B, H, W, D]
        B, D, H, W = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, D)
        
        # Distances: [N, K]
        dists = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        indices = torch.argmin(dists, dim=1)
        
        # Quantize
        quantized = self.embedding(indices).view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        
        # Tripartite losses
        loss_vq = F.mse_loss(quantized, inputs.detach())
        loss_commit = self.beta * F.mse_loss(inputs, quantized.detach())
        
        # STE trick
        quantized_ste = inputs + (quantized - inputs).detach()
        return quantized_ste, loss_vq + loss_commit, indices

def run_simulation():
    torch.manual_seed(42)
    B, C, H, W = 2, 4, 4, 4
    K = 8
    
    vq = MiniVectorQuantizer(num_embeddings=K, embedding_dim=C)
    inputs = torch.randn(B, C, H, W, requires_grad=True)
    
    quantized, vq_loss, indices = vq(inputs)
    target = torch.randn(B, C, H, W)
    recon_loss = F.mse_loss(quantized, target)
    total_loss = recon_loss + vq_loss
    
    total_loss.backward()
    
    assert inputs.grad is not None, "Encoder gradient failed (STE trick broken)"
    assert not torch.allclose(inputs.grad, torch.zeros_like(inputs.grad)), "Encoder grad is zero"
    assert vq.embedding.weight.grad is not None, "Codebook gradient failed"
    
    print(f"PyTorch Simulation Passed: Loss={total_loss.item():.4f}, Recon={recon_loss.item():.4f}, VQ_Loss={vq_loss.item():.4f}")

if __name__ == "__main__":
    run_simulation()
