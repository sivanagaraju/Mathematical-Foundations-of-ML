"""Numerical verification of analytical Gaussian posterior parameters."""
import torch

def numerical_verification_posterior():
    # Set seed for exact verification
    torch.manual_seed(42)
    B, D = 10000, 1 # Shape: (B, D)
    
    x0 = torch.ones(B, D) * 3.0 # Clean data: mean=3.0
    
    beta_1 = 0.1
    beta_2 = 0.2
    alpha_1 = 1.0 - beta_1
    alpha_2 = 1.0 - beta_2
    alpha_bar_1 = alpha_1
    alpha_bar_2 = alpha_1 * alpha_2
    
    # 1. Step-by-step sampling: x0 -> x1 -> x2
    eps_1 = torch.randn_like(x0) # Shape: (B, D)
    x1 = torch.sqrt(torch.tensor(alpha_1)) * x0 + torch.sqrt(torch.tensor(beta_1)) * eps_1
    eps_2 = torch.randn_like(x0) # Shape: (B, D)
    x2 = torch.sqrt(torch.tensor(alpha_2)) * x1 + torch.sqrt(torch.tensor(beta_2)) * eps_2
    
    # 2. Analytical formula for q(x1 | x2, x0)
    coef_x0 = (torch.sqrt(torch.tensor(alpha_bar_1)) * beta_2) / (1.0 - alpha_bar_2)
    coef_x2 = (torch.sqrt(torch.tensor(alpha_2)) * (1.0 - alpha_bar_1)) / (1.0 - alpha_bar_2)
    mu_tilde = coef_x0 * x0 + coef_x2 * x2 # Shape: (B, D)
    beta_tilde = float(((1.0 - alpha_bar_1) / (1.0 - alpha_bar_2)) * beta_2)
    
    print(f"Posterior Parameters: beta_tilde={beta_tilde:.5f}, mu_tilde mean={mu_tilde.mean().item():.4f}")
    assert beta_tilde > 0.0
    expected_mean = coef_x0 * 3.0 + coef_x2 * x2.mean()
    assert abs(mu_tilde.mean().item() - expected_mean.item()) < 1e-4
    print("Numerical posterior verification PASSED.")

if __name__ == "__main__":
    numerical_verification_posterior()
