"""
Simulation Script 01: Vanilla RNN Cell & Sequence Forward Pass from Scratch
Course: Mathematical Foundations of Machine Learning (NPTEL / IISc)
Lecture 45: Recurrent Neural Networks (RNNs)

Demonstrates:
1. Pure NumPy vector-matrix formulation of recurrent hidden state transitions.
2. Parameter sharing across temporal sequence steps.
3. Numerical parity verification against torch.nn.RNN.
"""

import numpy as np
import torch
import torch.nn as nn


def rnn_forward_numpy(X, h0, W_hh, W_xh, b_h, W_hy, b_y):
    """
    Computes forward pass of a Vanilla Elman RNN across sequence length T.
    
    Args:
        X: Input tensor of shape [B, T, d]
        h0: Initial hidden state of shape [B, m]
        W_hh: Hidden-to-hidden transition matrix of shape [m, m]
        W_xh: Input-to-hidden matrix of shape [m, d]
        b_h: Hidden bias vector of shape [m]
        W_hy: Readout projection matrix of shape [d_out, m]
        b_y: Readout bias vector of shape [d_out]
        
    Returns:
        H_all: Hidden state trajectory of shape [B, T, m]
        Y_all: Output predictions of shape [B, T, d_out]
    """
    B, T, d = X.shape
    m = W_hh.shape[0]
    d_out = W_hy.shape[0]
    
    H_all = np.zeros((B, T, m))
    Y_all = np.zeros((B, T, d_out))
    
    h_prev = h0.copy()  # Shape: [B, m]
    
    for t in range(T):
        xt = X[:, t, :]  # Shape: [B, d]
        
        # Dual-stream linear fusion: z_t = h_{t-1} W_hh^T + x_t W_xh^T + b_h
        # Shape: [B, m]
        zt = h_prev @ W_hh.T + xt @ W_xh.T + b_h
        
        # Pointwise non-linear state update
        ht = np.tanh(zt)  # Shape: [B, m]
        
        # Readout projection: y_t = h_t W_hy^T + b_y
        yt = ht @ W_hy.T + b_y  # Shape: [B, d_out]
        
        H_all[:, t, :] = ht
        Y_all[:, t, :] = yt
        h_prev = ht
        
    return H_all, Y_all


def test_numpy_pytorch_parity():
    np.random.seed(42)
    torch.manual_seed(42)
    
    B, T, d, m, d_out = 3, 5, 4, 8, 2
    
    # Initialize random inputs
    X_np = np.random.randn(B, T, d).astype(np.float32)
    h0_np = np.random.randn(B, m).astype(np.float32)
    
    # Initialize weights
    W_hh_np = np.random.randn(m, m).astype(np.float32) * 0.1
    W_xh_np = np.random.randn(m, d).astype(np.float32) * 0.1
    b_h_np = np.random.randn(m).astype(np.float32) * 0.05
    W_hy_np = np.random.randn(d_out, m).astype(np.float32) * 0.1
    b_y_np = np.random.randn(d_out).astype(np.float32) * 0.05
    
    # 1. Forward pass via Pure NumPy implementation
    H_np, Y_np = rnn_forward_numpy(X_np, h0_np, W_hh_np, W_xh_np, b_h_np, W_hy_np, b_y_np)
    
    # 2. Forward pass via PyTorch nn.RNN
    rnn_pt = nn.RNN(input_size=d, hidden_size=m, num_layers=1, bias=True, batch_first=True, nonlinearity='tanh')
    
    # Copy weights into PyTorch module
    with torch.no_grad():
        rnn_pt.weight_ih_l0.copy_(torch.from_numpy(W_xh_np))
        rnn_pt.weight_hh_l0.copy_(torch.from_numpy(W_hh_np))
        # PyTorch has two biases: b_ih and b_hh; sum equals total bias
        rnn_pt.bias_ih_l0.copy_(torch.from_numpy(b_h_np))
        rnn_pt.bias_hh_l0.zero_()
        
    X_pt = torch.from_numpy(X_np)
    h0_pt = torch.from_numpy(h0_np).unsqueeze(0)  # Shape: [1, B, m]
    
    H_pt, hn_pt = rnn_pt(X_pt, h0_pt)
    
    # Check hidden state trajectory parity
    np.testing.assert_allclose(H_np, H_pt.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(H_np[:, -1, :], hn_pt.squeeze(0).detach().numpy(), rtol=1e-5, atol=1e-5)
    
    # Check readout predictions
    fc_pt = nn.Linear(m, d_out, bias=True)
    with torch.no_grad():
        fc_pt.weight.copy_(torch.from_numpy(W_hy_np))
        fc_pt.bias.copy_(torch.from_numpy(b_y_np))
    Y_pt = fc_pt(H_pt)
    np.testing.assert_allclose(Y_np, Y_pt.detach().numpy(), rtol=1e-5, atol=1e-5)
    
    print("[PASS] Vanilla RNN NumPy vs PyTorch numerical parity strictly verified!")
    print(f"       Batch Size: {B}, Sequence Length: {T}, Input Dim: {d}, Hidden Dim: {m}")
    print(f"       Max absolute difference in H: {np.max(np.abs(H_np - H_pt.detach().numpy())):.2e}")


if __name__ == "__main__":
    test_numpy_pytorch_parity()
