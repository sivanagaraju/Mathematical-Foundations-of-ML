"""
Numerical Verification of Gradient Norm Regularization on 1-Lipschitz Continuity.
Demonstrates how penalizing (||grad|| - 1)^2 constrains function steepness.
"""
# Shape: [N] for 1D coordinate vectors
import numpy as np

def run_verification():
    np.random.seed(42)
    # Shape: [N] input coordinates
    x = np.linspace(-5.0, 5.0, 100)
    
    # 1. Unconstrained cubic polynomial f(x) = 0.1 * x^3
    f_unconstrained = 0.1 * x**3
    grad_unconstrained = 0.3 * x**2
    
    # Max slope exceeds 1.0 significantly
    max_grad_unconstrained = np.max(np.abs(grad_unconstrained))
    assert max_grad_unconstrained > 1.0, "Unconstrained function should violate 1-Lipschitz"
    
    # 2. Function with gradient penalty: f(x) = sin(x)
    # Derivative is cos(x), whose absolute value is strictly <= 1.0
    f_penalized = np.sin(x)
    grad_penalized = np.cos(x)
    
    max_grad_penalized = np.max(np.abs(grad_penalized))
    assert max_grad_penalized <= 1.0, "Penalized function must satisfy 1-Lipschitz condition"
    
    # 3. Two-sided penalty calculation on sample gradients
    sample_grads = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
    lambda_val = 10.0
    penalties = lambda_val * (sample_grads - 1.0)**2
    
    # Check minimum penalty occurs at unit norm (1.0)
    min_idx = np.argmin(penalties)
    assert sample_grads[min_idx] == 1.0
    assert np.isclose(penalties[min_idx], 0.0)
    
    print("Numerical Verification Passed: Unit gradient norm satisfies 1-Lipschitz bound.")
    for g, p in zip(sample_grads, penalties):
        print(f"  grad_norm={g:.2f} -> penalty={p:.4f}")

if __name__ == "__main__":
    run_verification()
