"""Autograd simulation for Gaussian KL divergence loss term."""
import torch
import torch.nn as nn

def test_autograd_elbo_term():
    torch.manual_seed(42)
    B, D = 16, 4
    
    # Model predicting reverse mean mu_theta
    model = nn.Sequential(
        nn.Linear(D, 32),
        nn.SiLU(),
        nn.Linear(32, D)
    )
    
    x0 = torch.randn(B, D) # Shape: (B, D)
    xt = torch.randn(B, D) # Shape: (B, D)
    
    # Target analytical posterior mean \tilde{\mu}_t
    mu_tilde = 0.6 * x0 + 0.4 * xt # Shape: (B, D)
    sigma_sq = 0.05
    
    # Predict mean
    pred_mu = model(xt) # Shape: (B, D)
    
    # Gaussian KL term: 1/(2*sigma^2) * ||mu_tilde - pred_mu||^2
    kl_loss = 0.5 * torch.sum((mu_tilde - pred_mu)**2, dim=-1) / sigma_sq # Shape: (B,)
    total_loss = kl_loss.mean()
    
    total_loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0, "Gradients must propagate through model"
    print(f"Autograd ELBO term simulation passed. Loss: {total_loss.item():.4f}, Grad norm: {grad_norm:.4f}")

if __name__ == "__main__":
    test_autograd_elbo_term()
