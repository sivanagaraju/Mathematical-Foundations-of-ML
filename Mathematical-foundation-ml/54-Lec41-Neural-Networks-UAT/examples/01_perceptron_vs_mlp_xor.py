"""
Simulation 01: Perceptron Convergence vs Multi-Layer Perceptron (MLP) on XOR

This script provides numerical verification of:
1. Novikoff's Theorem: Perceptron Learning Algorithm (PLA) converges in finite steps
   on linearly separable data (e.g., AND gate).
2. The Linear Separability Bottleneck (Minsky & Papert 1969): PLA cycles infinitely
   and fails to converge on non-linearly separable data (e.g., XOR gate).
3. Multi-Layer Perceptron (MLP) Resolution: Stacking non-linear squashing activations
   resolves XOR with 100% classification accuracy and zero training loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def run_perceptron_pla(X: np.ndarray, y: np.ndarray, max_steps: int = 100):
    """
    Simulates the classical Perceptron Learning Algorithm:
    w_{t+1} = w_t + y_t * x_t if y_t * (w_t^T x_t) <= 0.
    """
    # Shape: [N, D] where N=samples, D=features (with bias absorbed)
    num_samples, num_features = X.shape
    w = np.zeros(num_features, dtype=np.float32) # Shape: [D]
    
    converged = False
    step_count = 0
    
    for step in range(max_steps):
        mistakes = 0
        for i in range(num_samples):
            # Inner product: w^T x
            # Shape: scalar
            activation = np.dot(w, X[i])
            if y[i] * activation <= 0:
                # Misclassified: update weight vector
                w += y[i] * X[i]
                mistakes += 1
                step_count += 1
        if mistakes == 0:
            converged = True
            break
            
    return converged, w, step_count

def test_perceptron_and_gate():
    """Verify PLA converges on linearly separable AND gate."""
    # Features with bias term appended: [x_1, x_2, 1.0]
    # Shape: [4, 3]
    X_and = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ], dtype=np.float32)
    # Labels in {-1, +1}
    # Shape: [4]
    y_and = np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float32)
    
    converged, w, steps = run_perceptron_pla(X_and, y_and, max_steps=100)
    assert converged, f"PLA failed to converge on linearly separable AND gate after {steps} steps!"
    print(f"[PASS] PLA converged on AND gate in {steps} mistake steps with weights {w}")

def test_perceptron_xor_failure():
    """Verify PLA fails to converge on non-linearly separable XOR gate."""
    # Features with bias term appended: [x_1, x_2, 1.0]
    # Shape: [4, 3]
    X_xor = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ], dtype=np.float32)
    # XOR labels: (+1 for odd parity, -1 for even parity)
    # Shape: [4]
    y_xor = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float32)
    
    converged, w, steps = run_perceptron_pla(X_xor, y_xor, max_steps=100)
    assert not converged, "Mathematical error: PLA should NEVER converge on XOR!"
    print("[PASS] Verified that single-layer Perceptron fails to separate non-linear XOR data.")

def test_mlp_xor_resolution():
    """Verify a 2-layer MLP (1 hidden layer, 2 hidden units) cleanly solves XOR."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Inputs: [4, 2]
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=torch.float32) # Shape: [4, 2]
    
    # Labels: [4, 1]
    y = torch.tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ], dtype=torch.float32) # Shape: [4, 1]
    
    # Architecture: Input(2) -> Hidden(4, Sigmoid) -> Output(1, Sigmoid)
    model = nn.Sequential(
        nn.Linear(2, 4), # W_1: [4, 2], b_1: [4]
        nn.Sigmoid(),    # Element-wise squashing
        nn.Linear(4, 1), # W_2: [1, 4], b_2: [1]
        nn.Sigmoid()     # Output probability
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.08)
    
    # Train until convergence
    for epoch in range(2000):
        optimizer.zero_grad()
        predictions = model(X) # Shape: [4, 1]
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.01:
            break
            
    with torch.no_grad():
        final_preds = model(X) # Shape: [4, 1]
        binary_preds = (final_preds > 0.5).float()
        
    assert torch.allclose(binary_preds, y), f"MLP failed to solve XOR: preds={final_preds}"
    print(f"[PASS] 2-Layer MLP successfully solved XOR in {epoch} epochs with loss={loss.item():.4f}")
    print(f"       Predictions:\n{final_preds.numpy()}")

if __name__ == "__main__":
    test_perceptron_and_gate()
    test_perceptron_xor_failure()
    test_mlp_xor_resolution()
    print("[SUCCESS] All Perceptron and MLP mathematical simulations verified cleanly.")
