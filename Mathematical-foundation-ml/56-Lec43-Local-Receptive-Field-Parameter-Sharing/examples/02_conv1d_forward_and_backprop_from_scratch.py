#!/usr/bin/env python3
"""
Example 02: Conv1D Forward Pass and Exact Error Backpropagation from Scratch.

Demonstrates:
1. Pure NumPy implementation of 1D convolution with local receptive field and weight sharing.
2. Exact multivariable calculus chain rule derivation for shared weights:
   dL/dw_r = sum_{j} delta_j * x_{j*s + r}
3. Verification against PyTorch autograd engine (`torch.nn.Conv1d`).
4. End-to-end ERM optimization loop on a 1D synthetic edge detection task.
"""

import numpy as np
import torch
import torch.nn as nn

class ScratchConv1D:
    def __init__(self, kernel_size, stride=1):
        self.k = kernel_size
        self.s = stride
        # Initialize kernel and bias
        self.w = np.random.randn(self.k) * 0.5  # Shape: [k]
        self.b = 0.0                            # Scalar bias
        
        # Cache for backward pass
        self.x_cached = None
        self.z_cached = None
        
    def forward(self, x):
        """
        Forward cross-correlation: z_j = sum_{r=0}^{k-1} w_r * x_{j*s + r} + b
        x: 1D array of shape [d]
        returns z: 1D array of shape [M], where M = (d - k)//s + 1
        """
        self.x_cached = x.copy()
        d = len(x)
        M = (d - self.k) // self.s + 1
        z = np.zeros(M)
        
        for j in range(M):
            patch = x[j * self.s : j * self.s + self.k]
            z[j] = np.dot(self.w, patch) + self.b
            
        self.z_cached = z
        return z

    def backward(self, delta):
        """
        Backward adjoint sensitivity propagation.
        delta: upstream gradient dL/dz of shape [M]
        
        Weight gradient: dL/dw_r = sum_{j=0}^{M-1} delta_j * x_{j*s + r}
        Bias gradient:   dL/db   = sum_{j=0}^{M-1} delta_j
        Input gradient:  dL/dx_i = sum_{j, r: j*s + r = i} delta_j * w_r
        """
        M = len(delta)
        d = len(self.x_cached)
        
        grad_w = np.zeros(self.k)
        grad_b = np.sum(delta)
        grad_x = np.zeros(d)
        
        # 1. Accumulate gradients for shared weights across all spatial patches
        for j in range(M):
            patch = self.x_cached[j * self.s : j * self.s + self.k]
            grad_w += delta[j] * patch
            
        # 2. Propagate error sensitivities back to input coordinates (transposed convolution)
        for j in range(M):
            for r in range(self.k):
                grad_x[j * self.s + r] += delta[j] * self.w[r]
                
        return grad_w, grad_b, grad_x

def verify_against_pytorch():
    print("=" * 70)
    print("VERIFYING NUMPY BACKPROP DERIVATION AGAINST PYTORCH AUTOGRAD")
    print("=" * 70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    d = 10
    k = 3
    s = 1
    M = (d - k) // s + 1  # 8
    
    # Random input and initial weights
    x_np = np.random.randn(d)
    w_init = np.random.randn(k)
    b_init = 0.5
    
    # Setup Scratch Model
    model_scratch = ScratchConv1D(kernel_size=k, stride=s)
    model_scratch.w = w_init.copy()
    model_scratch.b = b_init
    
    # Setup PyTorch Model (Conv1d: in_channels=1, out_channels=1)
    model_torch = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, stride=s, bias=True).to(torch.float64)
    with torch.no_grad():
        model_torch.weight.copy_(torch.from_numpy(w_init).view(1, 1, k))
        model_torch.bias.copy_(torch.tensor([b_init], dtype=torch.float64))
        
    # PyTorch Forward
    x_torch = torch.from_numpy(x_np).view(1, 1, d).requires_grad_(True)
    z_torch = model_torch(x_torch)
    
    # NumPy Forward
    z_scratch = model_scratch.forward(x_np)
    
    assert np.allclose(z_scratch, z_torch.detach().numpy().flatten()), "Forward pass mismatch!"
    print("[PASS] Forward activation outputs match PyTorch Conv1d exactly.")
    
    # Dummy upstream loss: L = 0.5 * sum((z - target)^2)
    target_np = np.ones(M)
    delta_np = z_scratch - target_np  # dL/dz
    
    # NumPy Backward
    grad_w_np, grad_b_np, grad_x_np = model_scratch.backward(delta_np)
    
    # PyTorch Backward
    target_torch = torch.from_numpy(target_np).view(1, 1, M)
    loss_torch = 0.5 * torch.sum((z_torch - target_torch) ** 2)
    loss_torch.backward()
    
    grad_w_torch = model_torch.weight.grad.numpy().flatten()
    grad_b_torch = model_torch.bias.grad.numpy().item()
    grad_x_torch = x_torch.grad.numpy().flatten()
    
    print(f"NumPy Weight Gradient:   {np.round(grad_w_np, 5)}")
    print(f"PyTorch Weight Gradient: {np.round(grad_w_torch, 5)}")
    print(f"NumPy Bias Gradient:     {grad_b_np:.5f} | PyTorch: {grad_b_torch:.5f}")
    
    assert np.allclose(grad_w_np, grad_w_torch), "Weight gradient mismatch!"
    assert np.isclose(grad_b_np, grad_b_torch), "Bias gradient mismatch!"
    assert np.allclose(grad_x_np, grad_x_torch), "Input gradient mismatch!"
    print("[PASS] All gradients (weight, bias, input) match PyTorch autograd to machine precision!\n")

def run_synthetic_erm_training():
    print("=" * 70)
    print("RUNNING CONV1D ERM TRAINING LOOP ON SYNTHETIC EDGE DETECTION TASK")
    print("=" * 70)
    
    np.random.seed(1337)
    # Synthetic problem: 1D signal with step edge. Target output is 1 where positive edge occurs, 0 elsewhere.
    d = 16
    k = 3
    s = 1
    M = (d - k) // s + 1  # 14
    
    model = ScratchConv1D(kernel_size=k, stride=s)
    model.w = np.random.randn(k) * 0.1
    model.b = 0.0
    lr = 0.1
    
    # Training dataset: 50 random step-edge signals
    X_train = []
    Y_train = []
    for _ in range(50):
        sig = np.zeros(d)
        edge_pos = np.random.randint(2, d - 4)
        sig[edge_pos:] = 1.0  # Step edge
        sig += np.random.normal(0, 0.02, size=d)
        
        target = np.zeros(M)
        if edge_pos - 1 < M:
            target[edge_pos - 1] = 1.0  # Edge marker
            
        X_train.append(sig)
        Y_train.append(target)
        
    initial_loss = 0.0
    for x, y in zip(X_train, Y_train):
        z = model.forward(x)
        initial_loss += np.mean((z - y) ** 2)
    initial_loss /= len(X_train)
    
    # Train for 80 epochs
    for epoch in range(80):
        total_loss = 0.0
        grad_w_acc = np.zeros(k)
        grad_b_acc = 0.0
        
        for x, y in zip(X_train, Y_train):
            z = model.forward(x)
            loss = np.mean((z - y) ** 2)
            total_loss += loss
            
            # d(MSE)/dz = (2/M) * (z - y)
            delta = (2.0 / M) * (z - y)
            gw, gb, _ = model.backward(delta)
            grad_w_acc += gw
            grad_b_acc += gb
            
        # SGD update
        model.w -= lr * (grad_w_acc / len(X_train))
        model.b -= lr * (grad_b_acc / len(X_train))
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(X_train)
            print(f"Epoch {epoch+1:02d}/80 | MSE Loss: {avg_loss:.5f} | Learned Kernel: {np.round(model.w, 4)}")
            
    final_loss = total_loss / len(X_train)
    assert final_loss < initial_loss * 0.7, "Training loss should decrease substantially!"
    print(f"\n[PASS] ERM Convergence Verified: Loss reduced from {initial_loss:.4f} to {final_loss:.4f}")
    print(f"Final Learned Edge Filter: {np.round(model.w, 3)} (Notice difference operator structure!)")
    print("=" * 70)

if __name__ == "__main__":
    verify_against_pytorch()
    run_synthetic_erm_training()
    print("All tests passed cleanly with exit code 0.")
