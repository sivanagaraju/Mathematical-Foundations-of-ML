"""Numerical verification of Classifier-Free Guidance extrapolation."""
import torch

def numerical_verification_cfg():
    torch.manual_seed(42)
    B, D = 100, 4 # Shape: (B, D)
    
    eps_uncond = torch.randn(B, D) # Shape: (B, D)
    steering_direction = torch.ones(B, D) * 0.5
    eps_cond = eps_uncond + steering_direction # Shape: (B, D)
    
    s = 4.0 # Guidance scale
    
    guided_1 = eps_uncond + s * (eps_cond - eps_uncond) # Shape: (B, D)
    guided_2 = (1.0 - s) * eps_uncond + s * eps_cond # Shape: (B, D)
    
    diff = torch.max(torch.abs(guided_1 - guided_2)).item()
    print(f"CFG numerical equivalence max difference: {diff:.8f}")
    assert diff < 1e-6
    
    expected_boost = steering_direction * s
    actual_boost = guided_1 - eps_uncond
    assert torch.allclose(actual_boost, expected_boost)
    print("Numerical CFG verification PASSED.")

if __name__ == "__main__":
    numerical_verification_cfg()
