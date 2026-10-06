"""
01_scalar_vs_vectorized_backprop.py
===================================
Demonstration of exact equivalence between:
1. Pure Scalar Error Backpropagation (Lec 42 chalkboard derivation)
2. Vectorized Matrix Backpropagation (NumPy GEMM)
3. PyTorch Reverse-Mode Automatic Differentiation (Autograd)

Mathematical formulation verified:
- Pre-activation: z_j^[l] = sum_k w_{jk}^[l] a_k^[l-1] + b_j^[l]
- Post-activation: a_j^[l] = sigma(z_j^[l])
- Output base error: delta_j^[L] = (a_j^[L] - y_j) * sigma'(z_j^[L])
- Hidden layer recurrence: delta_j^[l] = (sum_i delta_i^[l+1] w_{ij}^[l+1]) * sigma'(z_j^[l])
- Weight gradient: dR / dw_{jk}^[l] = delta_j^[l] * a_k^[l-1]
- Bias gradient: dR / db_j^[l] = delta_j^[l]
"""

import numpy as np
import torch
import torch.nn as nn

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Architecture dimensions: d=3 inputs, N1=4 hidden, N2=2 output
    d, N1, N2 = 3, 4, 2

    # Fixed synthetic sample
    # Shape: [d] = [3]
    x_np = np.array([0.5, -1.2, 0.8], dtype=np.float64)
    # Shape: [N2] = [2]
    y_np = np.array([1.0, 0.0], dtype=np.float64)

    # Initial weights and biases
    # W1: [N1, d] = [4, 3], b1: [N1] = [4]
    W1_np = np.random.randn(N1, d) * 0.5
    b1_np = np.random.randn(N1) * 0.1

    # W2: [N2, N1] = [2, 4], b2: [N2] = [2]
    W2_np = np.random.randn(N2, N1) * 0.5
    b2_np = np.random.randn(N2) * 0.1

    # -------------------------------------------------------------
    # 1. PURE SCALAR IMPLEMENTATION (Lec 42 Chalkboard Derivation)
    # -------------------------------------------------------------
    # Forward Pass: Layer 1
    z1_scalar = np.zeros(N1)
    a1_scalar = np.zeros(N1)
    for j in range(N1):
        affine_sum = b1_np[j]
        for k in range(d):
            affine_sum += W1_np[j, k] * x_np[k]
        z1_scalar[j] = affine_sum
        a1_scalar[j] = sigmoid(affine_sum)

    # Forward Pass: Layer 2 (Output)
    z2_scalar = np.zeros(N2)
    a2_scalar = np.zeros(N2)
    for j in range(N2):
        affine_sum = b2_np[j]
        for k in range(N1):
            affine_sum += W2_np[j, k] * a1_scalar[k]
        z2_scalar[j] = affine_sum
        a2_scalar[j] = sigmoid(affine_sum)

    # Scalar Loss: Half Mean Squared Error
    loss_scalar = 0.5 * np.sum((a2_scalar - y_np) ** 2)

    # Backward Pass: Output Layer Base Error delta^[2]
    delta2_scalar = np.zeros(N2)
    for j in range(N2):
        dL_da2 = a2_scalar[j] - y_np[j]
        da2_dz2 = sigmoid_prime(z2_scalar[j])
        delta2_scalar[j] = dL_da2 * da2_dz2

    # Backward Pass: Hidden Layer Error delta^[1] (Recursive Formula)
    delta1_scalar = np.zeros(N1)
    for j in range(N1):
        downstream_error_sum = 0.0
        for i in range(N2):
            downstream_error_sum += delta2_scalar[i] * W2_np[i, j]
        delta1_scalar[j] = downstream_error_sum * sigmoid_prime(z1_scalar[j])

    # Parameter Gradients (Scalar)
    grad_W2_scalar = np.zeros((N2, N1))
    grad_b2_scalar = np.zeros(N2)
    for j in range(N2):
        grad_b2_scalar[j] = delta2_scalar[j]
        for k in range(N1):
            grad_W2_scalar[j, k] = delta2_scalar[j] * a1_scalar[k]

    grad_W1_scalar = np.zeros((N1, d))
    grad_b1_scalar = np.zeros(N1)
    for j in range(N1):
        grad_b1_scalar[j] = delta1_scalar[j]
        for k in range(d):
            grad_W1_scalar[j, k] = delta1_scalar[j] * x_np[k]

    # -------------------------------------------------------------
    # 2. VECTORIZED MATRIX IMPLEMENTATION (NumPy GEMM)
    # -------------------------------------------------------------
    # Forward Pass
    z1_vec = W1_np @ x_np + b1_np
    a1_vec = sigmoid(z1_vec)
    z2_vec = W2_np @ a1_vec + b2_np
    a2_vec = sigmoid(z2_vec)

    # Backward Pass
    delta2_vec = (a2_vec - y_np) * sigmoid_prime(z2_vec)
    delta1_vec = (W2_np.T @ delta2_vec) * sigmoid_prime(z1_vec)

    # Outer product gradients
    grad_W2_vec = np.outer(delta2_vec, a1_vec)
    grad_b2_vec = delta2_vec
    grad_W1_vec = np.outer(delta1_vec, x_np)
    grad_b1_vec = delta1_vec

    # -------------------------------------------------------------
    # 3. PYTORCH AUTOGRAD IMPLEMENTATION
    # -------------------------------------------------------------
    x_pt = torch.tensor(x_np, dtype=torch.float64)
    y_pt = torch.tensor(y_np, dtype=torch.float64)
    W1_pt = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
    b1_pt = torch.tensor(b1_np, dtype=torch.float64, requires_grad=True)
    W2_pt = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
    b2_pt = torch.tensor(b2_np, dtype=torch.float64, requires_grad=True)

    z1_pt = torch.matmul(W1_pt, x_pt) + b1_pt
    a1_pt = torch.sigmoid(z1_pt)
    z2_pt = torch.matmul(W2_pt, a1_pt) + b2_pt
    a2_pt = torch.sigmoid(z2_pt)

    loss_pt = 0.5 * torch.sum((a2_pt - y_pt) ** 2)
    loss_pt.backward()

    # -------------------------------------------------------------
    # 4. NUMERICAL VERIFICATION & ASSERTIONS
    # -------------------------------------------------------------
    # Verify forward activations match
    assert np.allclose(a1_scalar, a1_vec, atol=1e-12)
    assert np.allclose(a2_scalar, a2_vec, atol=1e-12)
    assert np.allclose(a1_vec, a1_pt.detach().numpy(), atol=1e-12)
    assert np.allclose(a2_vec, a2_pt.detach().numpy(), atol=1e-12)

    # Verify error sensitivities delta match
    assert np.allclose(delta1_scalar, delta1_vec, atol=1e-12)
    assert np.allclose(delta2_scalar, delta2_vec, atol=1e-12)

    # Verify weight gradients match
    assert np.allclose(grad_W1_scalar, grad_W1_vec, atol=1e-12)
    assert np.allclose(grad_W2_scalar, grad_W2_vec, atol=1e-12)
    assert np.allclose(grad_W1_vec, W1_pt.grad.numpy(), atol=1e-12)
    assert np.allclose(grad_W2_vec, W2_pt.grad.numpy(), atol=1e-12)

    # Verify bias gradients match
    assert np.allclose(grad_b1_scalar, grad_b1_vec, atol=1e-12)
    assert np.allclose(grad_b2_scalar, grad_b2_vec, atol=1e-12)
    assert np.allclose(grad_b1_vec, b1_pt.grad.numpy(), atol=1e-12)
    assert np.allclose(grad_b2_vec, b2_pt.grad.numpy(), atol=1e-12)

    print("[VERIFICATION SUCCESSFUL]")
    print(f"Scalar vs Vectorized vs PyTorch Autograd: 100% exact numerical match.")
    print(f"Loss: {loss_scalar:.8f}")
    print(f"Norm of W1 grad: {np.linalg.norm(grad_W1_vec):.6f}")
    print(f"Norm of W2 grad: {np.linalg.norm(grad_W2_vec):.6f}")

if __name__ == "__main__":
    main()
