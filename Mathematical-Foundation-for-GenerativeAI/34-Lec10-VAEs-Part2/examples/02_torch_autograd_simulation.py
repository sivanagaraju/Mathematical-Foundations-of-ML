"""
02_torch_autograd_simulation.py — PyTorch Autograd Graph & Gradient Flow in VAE

This script implements an end-to-end Variational Autoencoder forward and backward pass,
verifying:
1. Shape transformations from input data [B, D] -> latents [B, K] -> reconstructions [B, D].
2. Pathwise gradient propagation through the reparameterization layer z = mu + sigma * epsilon.
3. Decoupled backpropagation into encoder parameters phi and decoder parameters theta.
4. Absence of NaN gradients or graph disconnections.
"""
import torch
import torch.nn as nn
import torch.optim as optim

class ToyEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor):
        # Shape: [B, in_dim] -> [B, hidden_dim]
        h = torch.relu(self.fc(x))
        # Mean vector, Shape: [B, latent_dim]
        mu = self.fc_mu(h)
        # Log-variance vector, Shape: [B, latent_dim]
        logvar = self.fc_logvar(h)
        return mu, logvar

class ToyDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, z: torch.Tensor):
        # Shape: [B, latent_dim] -> [B, hidden_dim]
        h = torch.relu(self.fc1(z))
        # Reconstructed mean x_hat, Shape: [B, out_dim]
        x_hat = self.fc2(h)
        return x_hat

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Standard deviation: sigma = exp(0.5 * logvar), Shape: [B, K]
    std = torch.exp(0.5 * logvar)
    # Auxiliary standard normal noise, Shape: [B, K]
    eps = torch.randn_like(std)
    # Latent realization, Shape: [B, K]
    z = mu + eps * std
    return z

def compute_elbo_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    # Gaussian log-likelihood equivalence: Mean Squared Error reconstruction loss
    # Shape: [B, D] -> scalar sum over batch
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    
    # Analytical Gaussian relative entropy (KL divergence against N(0, I))
    # Shape: [B, K] -> scalar sum over batch
    kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    
    # Total negative ELBO to minimize
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss

def main():
    print("=== PyTorch VAE Autograd Gradient Simulation ===")
    torch.manual_seed(42)
    
    batch_size = 16
    in_dim = 64
    hidden_dim = 32
    latent_dim = 8
    
    # Instantiate networks
    encoder = ToyEncoder(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = ToyDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=in_dim)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    
    # Generate synthetic input batch, Shape: [B, D]
    x_input = torch.randn(batch_size, in_dim)
    
    # Forward pass
    optimizer.zero_grad()
    mu, logvar = encoder(x_input)
    assert mu.shape == (batch_size, latent_dim), f"Expected shape {(batch_size, latent_dim)}, got {mu.shape}"
    assert logvar.shape == (batch_size, latent_dim), f"Expected shape {(batch_size, latent_dim)}, got {logvar.shape}"
    
    z = reparameterize(mu, logvar)
    assert z.shape == (batch_size, latent_dim), f"Expected latent shape {(batch_size, latent_dim)}, got {z.shape}"
    
    x_recon = decoder(z)
    assert x_recon.shape == (batch_size, in_dim), f"Expected output shape {(batch_size, in_dim)}, got {x_recon.shape}"
    
    loss, recon_loss, kl_loss = compute_elbo_loss(x_input, x_recon, mu, logvar)
    print(f"Batch Size: {batch_size}, InDim: {in_dim}, LatentDim: {latent_dim}")
    print(f"Total -ELBO Loss: {loss.item():.4f} (Recon MSE: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f})")
    
    # Backward pass
    loss.backward()
    
    # Verify gradients exist, are non-zero, and contain no NaNs
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"Gradient missing for encoder parameter {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in encoder parameter {name}"
        assert param.grad.abs().sum() > 0.0, f"Zero gradient in encoder parameter {name}"
        
    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"Gradient missing for decoder parameter {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in decoder parameter {name}"
        assert param.grad.abs().sum() > 0.0, f"Zero gradient in decoder parameter {name}"
        
    optimizer.step()
    print("Optimization step completed successfully.")
    print("Assertion passed: All gradients populated differentiably through reparameterization without NaNs.")

if __name__ == "__main__":
    main()
