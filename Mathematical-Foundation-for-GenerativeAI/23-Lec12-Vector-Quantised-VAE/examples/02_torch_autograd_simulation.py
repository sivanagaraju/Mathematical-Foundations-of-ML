"""
PyTorch Autograd Simulation of VQ-VAE Vector Quantizer with Straight-Through Estimator.
Verifies gradient flow through continuous encoder while detaching discrete dictionary lookups.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=16, embedding_dim=8, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, inputs):
        # inputs shape: (B, C, H, W) -> permute to (B, H, W, C)
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_permuted.view(-1, self.embedding_dim)
        
        # Distances: (N, K)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs_permuted.shape)
        
        # Losses
        loss_vq = F.mse_loss(quantized, inputs_permuted.detach())
        loss_commit = F.mse_loss(inputs_permuted, quantized.detach())
        loss = loss_vq + self.commitment_cost * loss_commit
        
        # Straight-Through Estimator trick
        quantized = inputs_permuted + (quantized - inputs_permuted).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, loss, encoding_indices

def run_simulation():
    torch.manual_seed(42)
    B, C, H, W = 2, 8, 4, 4
    encoder_output = torch.randn(B, C, H, W, requires_grad=True)
    vq = VectorQuantizer(num_embeddings=16, embedding_dim=C, commitment_cost=0.25)
    
    quantized, vq_loss, indices = vq(encoder_output)
    
    # Dummy decoder: 1x1 conv to target
    target = torch.randn(B, C, H, W)
    recon_loss = F.mse_loss(quantized, target)
    total_loss = recon_loss + vq_loss
    
    total_loss.backward()
    
    assert encoder_output.grad is not None, "Encoder gradient must not be None (STE failed)"
    assert not torch.allclose(encoder_output.grad, torch.zeros_like(encoder_output.grad)), "Encoder grad is zero"
    assert vq.embedding.weight.grad is not None, "Codebook embedding grad must not be None"
    
    print(f"VQ-VAE PyTorch autograd simulation passed. Total loss: {total_loss.item():.4f}")
    print(f"Encoder grad norm: {encoder_output.grad.norm().item():.4f}")

if __name__ == "__main__":
    run_simulation()
