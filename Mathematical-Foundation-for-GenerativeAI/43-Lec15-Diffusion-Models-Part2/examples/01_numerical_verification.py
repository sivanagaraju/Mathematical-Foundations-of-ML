import torch

def numerical_verification_noise_param():
    # Set seed for exact reproduction
    torch.manual_seed(42)
    B, D = 1000, 1 # Shape: (B, D)
    
    alpha_t = 0.9
    beta_t = 0.1
    alpha_bar_prev = 0.8
    alpha_bar_t = alpha_t * alpha_bar_prev
    
    x0 = torch.randn(B, D) # Shape: (B, D)
    eps = torch.randn(B, D) # Shape: (B, D)
    xt = torch.sqrt(torch.tensor(alpha_bar_t)) * x0 + torch.sqrt(torch.tensor(1.0 - alpha_bar_t)) * eps
    
    # 1. True posterior mean computed with x0
    c1 = (torch.sqrt(torch.tensor(alpha_bar_prev)) * beta_t) / (1.0 - alpha_bar_t)
    c2 = (torch.sqrt(torch.tensor(alpha_t)) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
    mu_true = c1 * x0 + c2 * xt
    
    # 2. Posterior mean computed with eps
    mu_reparam = (1.0 / torch.sqrt(torch.tensor(alpha_t))) * (xt - (beta_t / torch.sqrt(torch.tensor(1.0 - alpha_bar_t))) * eps)
    
    max_diff = torch.max(torch.abs(mu_true - mu_reparam)).item()
    print(f"Max difference between representations: {max_diff:.8f}")
    assert max_diff < 1e-6
    print("Numerical noise parameterization PASSED.")

if __name__ == "__main__":
    numerical_verification_noise_param()
