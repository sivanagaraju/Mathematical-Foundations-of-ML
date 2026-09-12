"""
PyTorch Autograd Simulation of Convolutional Beta-VAE.
Verifies reparameterization gradient flow, analytical KL calculation, and backward pass.
"""
# Shape: [B, C, H, W] for image batches, [B, K] for latent variables
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1), # (B, 8, 8, 8)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(8 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(8 * 8 * 8, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 8 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 1, 16, 16)
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, -15.0, 10.0))
        eps = torch.randn_like(std)
        return mu + std * eps
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(self.decoder_input(z).view(-1, 8, 8, 8))
        return recon, mu, logvar

def run_simulation():
    torch.manual_seed(42)
    B, C, H, W = 4, 1, 16, 16
    K = 4
    beta = 2.0
    
    model = MiniConvVAE(in_channels=C, latent_dim=K)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Shape: [B, C, H, W]
    images = torch.rand(B, C, H, W)
    recon, mu, logvar = model(images)
    
    loss_recon = F.binary_cross_entropy(recon, images, reduction='sum') / B
    loss_kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / B
    total_loss = loss_recon + beta * loss_kl
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    assert total_loss.item() > 0.0, "Loss must be positive"
    assert model.fc_mu.weight.grad is not None, "Encoder gradients must not be None"
    assert model.decoder[0].weight.grad is not None, "Decoder gradients must not be None"
    
    print(f"PyTorch Simulation Passed: Loss={total_loss.item():.4f}, Recon={loss_recon.item():.4f}, KL={loss_kl.item():.4f}")

if __name__ == "__main__":
    run_simulation()
