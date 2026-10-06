"""
Simulation 02: Universal Approximation Theorem (UAT) Verification

This script provides numerical verification of Cybenko (1989) & Hornik (1991):
1. Target: A non-linear continuous function f(x) = sin(2*pi*x) + 0.5*cos(4*pi*x)
   defined on the compact domain X = [0, 1].
2. Hypothesis Class: A single-hidden-layer Multi-Layer Perceptron (MLP) with
   N=128 hidden units and continuous sigmoidal squashing non-linearities:
   h(x) = W_2 * sigma(W_1 * x + b_1) + b_2
3. Assertion: Verifies that the supremum error sup_{x in [0, 1]} |f(x) - h(x)|
   is strictly bounded below epsilon = 0.20 across dense evaluation points.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def target_function(x: torch.Tensor) -> torch.Tensor:
    """Continuous non-linear target function on compact interval [0, 1]."""
    # Shape: [B, 1]
    return torch.sin(2.0 * np.pi * x) + 0.5 * torch.cos(4.0 * np.pi * x)

class UniversalApproximator(nn.Module):
    """Single-hidden-layer feedforward network satisfying UAT conditions."""
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # W_1: [hidden_dim, 1], b_1: [hidden_dim]
        self.fc1 = nn.Linear(1, hidden_dim)
        # Continuous sigmoidal squashing activation
        self.activation = nn.Sigmoid()
        # W_2: [1, hidden_dim], b_2: [1]
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 1]
        z1 = self.fc1(x)         # Shape: [B, hidden_dim]
        a1 = self.activation(z1) # Shape: [B, hidden_dim]
        out = self.fc2(a1)       # Shape: [B, 1]
        return out

def verify_universal_approximation():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Generate training grid on compact set [0, 1]
    # Shape: [300, 1]
    x_train = torch.linspace(0.0, 1.0, 300).unsqueeze(1)
    y_train = target_function(x_train) # Shape: [300, 1]
    
    # 2. Instantiate single-hidden-layer network (N = 128 hidden units)
    model = UniversalApproximator(hidden_dim=128)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 3. Fit network via gradient descent
    print("Training single-hidden-layer MLP to verify UAT bound...")
    for epoch in range(2500):
        optimizer.zero_grad()
        preds = model(x_train) # Shape: [300, 1]
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
        if loss.item() < 1e-4:
            break
            
    # 4. Dense evaluation grid for supremum norm bound verification
    # Shape: [500, 1]
    x_test = torch.linspace(0.0, 1.0, 500).unsqueeze(1)
    y_test = target_function(x_test) # Shape: [500, 1]
    
    model.eval()
    with torch.no_grad():
        test_preds = model(x_test) # Shape: [500, 1]
        
    # Calculate supremum norm: sup_{x in [0, 1]} |f(x) - h(x)|
    absolute_errors = torch.abs(test_preds - y_test) # Shape: [500, 1]
    supremum_error = torch.max(absolute_errors).item()
    mean_squared_error = torch.mean((test_preds - y_test) ** 2).item()
    
    print(f"[RESULTS] UAT Convergence achieved:")
    print(f"          Max Absolute Error (Supremum Norm): {supremum_error:.4f}")
    print(f"          Mean Squared Error (L2 Norm):       {mean_squared_error:.6f}")
    
    # Mathematical assertion: epsilon threshold = 0.20
    epsilon = 0.20
    assert supremum_error < epsilon, (
        f"UAT bound violated! Expected sup error < {epsilon}, got {supremum_error:.4f}"
    )
    
    # Numerical validation with torch.allclose
    assert torch.allclose(test_preds, y_test, atol=epsilon), (
        "Numerical divergence: predictions do not match target within epsilon!"
    )
    print(f"[PASS] Successfully confirmed: sup_{{x in [0, 1]}} |f(x) - h(x)| < {epsilon}")

if __name__ == "__main__":
    verify_universal_approximation()
    print("[SUCCESS] Universal Approximation simulation completed cleanly with code 0.")
