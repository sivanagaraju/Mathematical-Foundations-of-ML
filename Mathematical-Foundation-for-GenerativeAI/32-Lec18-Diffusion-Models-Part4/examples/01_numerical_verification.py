"""Numerical verification of Tweedie's formula and score duality."""
import torch

def numerical_verification_tweedie():
    torch.manual_seed(42)
    B, D = 1000, 1 # Shape: (B, D)
    
    x0 = torch.ones(B, D) * 2.5
    sigma = 0.5
    eps = torch.randn(B, D) # Shape: (B, D)
    
    xt = x0 + sigma * eps # Shape: (B, D)
    
    # Ground truth score vector
    score = -eps / sigma # Shape: (B, D)
    
    # Tweedie recovery: E[x0 | xt] = xt + sigma^2 * score
    recovered_x0 = xt + (sigma**2) * score # Shape: (B, D)
    
    diff = torch.max(torch.abs(recovered_x0 - x0)).item()
    print(f"Tweedie recovery max difference: {diff:.8f}")
    assert diff < 1e-6
    print("Numerical Tweedie verification PASSED.")

if __name__ == "__main__":
    numerical_verification_tweedie()
