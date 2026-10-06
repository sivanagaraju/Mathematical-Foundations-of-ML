#!/usr/bin/env python3
"""
Example 01: Multi-Channel 2D Convolution and Backpropagation from Scratch

This script verifies the core mathematical equivalence taught by Prof. Prathosh in Lecture 44:
1. 3D Filter Bank Operation: Each filter of size [C_in, k_h, k_w] performs Frobenius inner
   products across all input channels simultaneously, summing (marginalizing) out the channel dimension.
2. Stacking L filters produces an output activation tensor of shape [B, C_out, H_out, W_out].
3. Backpropagation under Parameter Sharing: Gradients with respect to shared kernel weights
   accumulate upstream error sensitivities across all spatial positions and batch instances.
4. Numerical Parity: Exact bitwise-level gradient verification against PyTorch nn.Conv2d.

Author: Mathematical Foundations of ML Course Team
"""

import numpy as np
import torch
import torch.nn as nn


def conv2d_forward_scratch(x, w, b, stride=1, padding=0):
    """
    Pure NumPy implementation of multi-channel 2D cross-correlation (deep learning convolution).

    Parameters:
    -----------
    x : np.ndarray, shape [B, C_in, H, W]
        Input activation tensor.
    w : np.ndarray, shape [C_out, C_in, k_h, k_w]
        Convolutional filter bank.
    b : np.ndarray, shape [C_out]
        Per-filter scalar bias.
    stride : int
        Spatial stride step.
    padding : int
        Zero-padding added to perimeter of H and W.

    Returns:
    --------
    out : np.ndarray, shape [B, C_out, H_out, W_out]
        Output feature maps after channel marginalization.
    """
    B, C_in, H, W = x.shape
    C_out, _, k_h, k_w = w.shape

    # Apply symmetric zero-padding
    if padding > 0:
        x_pad = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0.0
        )
    else:
        x_pad = x

    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
    H_out = (H_pad - k_h) // stride + 1
    W_out = (W_pad - k_w) // stride + 1

    out = np.zeros((B, C_out, H_out, W_out), dtype=np.float64)

    # Compute sliding inner product across channels
    for n in range(B):
        for c_out in range(C_out):
            for i in range(H_out):
                i_start = i * stride
                i_end = i_start + k_h
                for j in range(W_out):
                    j_start = j * stride
                    j_end = j_start + k_w
                    # Patch across ALL input channels: shape [C_in, k_h, k_w]
                    patch = x_pad[n, :, i_start:i_end, j_start:j_end]
                    # Frobenius inner product + channel marginalization sum
                    out[n, c_out, i, j] = np.sum(patch * w[c_out]) + b[c_out]

    return out, x_pad


def conv2d_backward_scratch(dout, x_pad, w, stride=1, padding=0, orig_h=None, orig_w=None):
    """
    Reverse-mode adjoint sensitivity backpropagation through 2D multi-channel convolution.

    Parameters:
    -----------
    dout : np.ndarray, shape [B, C_out, H_out, W_out]
        Upstream loss sensitivities (dL / dZ).
    x_pad : np.ndarray, shape [B, C_in, H_pad, W_pad]
        Padded input tensor from forward cache.
    w : np.ndarray, shape [C_out, C_in, k_h, k_w]
        Convolutional filter bank.

    Returns:
    --------
    dx : np.ndarray, shape [B, C_in, H, W]
        Input gradient (dL / dX).
    dw : np.ndarray, shape [C_out, C_in, k_h, k_w]
        Filter gradient (dL / dW).
    db : np.ndarray, shape [C_out]
        Bias gradient (dL / dB).
    """
    B, C_out, H_out, W_out = dout.shape
    C_out, C_in, k_h, k_w = w.shape

    dx_pad = np.zeros_like(x_pad, dtype=np.float64)
    dw = np.zeros_like(w, dtype=np.float64)
    db = np.zeros_like(dout[0, :, 0, 0], dtype=np.float64)

    # Bias gradient sums across batch and spatial positions
    db = np.sum(dout, axis=(0, 2, 3))

    for n in range(B):
        for c_out in range(C_out):
            for i in range(H_out):
                i_start = i * stride
                i_end = i_start + k_h
                for j in range(W_out):
                    j_start = j * stride
                    j_end = j_start + k_w
                    delta = dout[n, c_out, i, j]

                    # Weight gradient accumulation across space: dL/dw_r = sum delta * x
                    dw[c_out] += delta * x_pad[n, :, i_start:i_end, j_start:j_end]

                    # Input gradient accumulation: dL/dx = sum delta * w
                    dx_pad[n, :, i_start:i_end, j_start:j_end] += delta * w[c_out]

    # Crop zero-padding to recover original input dimensions
    if padding > 0:
        dx = dx_pad[:, :, padding:padding + orig_h, padding:padding + orig_w]
    else:
        dx = dx_pad

    return dx, dw, db


def run_parity_test():
    print("=" * 70)
    print("VERIFICATION: Multi-Channel 2D Convolution Parity with PyTorch")
    print("=" * 70)

    # Fix random seed for exact reproducibility
    np.random.seed(44)
    torch.manual_seed(44)

    # Define hyperparameter configuration
    B = 2         # Batch size
    C_in = 3      # RGB input channels (e.g. Bayer array reconstructed)
    H, W = 6, 6   # Spatial dimensions
    C_out = 4     # Number of distinct feature filters
    k_h, k_w = 3, 3 # Filter dimensions
    stride = 1
    padding = 1

    # Generate synthetic input and parameters (float64 for numerical precision)
    x_np = np.random.randn(B, C_in, H, W).astype(np.float64)
    w_np = np.random.randn(C_out, C_in, k_h, k_w).astype(np.float64)
    b_np = np.random.randn(C_out).astype(np.float64)

    # 1. Scratch NumPy Forward Pass
    out_np, x_pad = conv2d_forward_scratch(x_np, w_np, b_np, stride=stride, padding=padding)

    # 2. PyTorch Equivalent Module
    conv_pt = nn.Conv2d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=(k_h, k_w),
        stride=stride,
        padding=padding,
        bias=True
    ).to(torch.float64)

    # Load identical weights and biases into PyTorch
    with torch.no_grad():
        conv_pt.weight.copy_(torch.from_numpy(w_np))
        conv_pt.bias.copy_(torch.from_numpy(b_np))

    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    out_pt = conv_pt(x_pt)

    # Verify Forward Parity
    forward_diff = np.max(np.abs(out_np - out_pt.detach().numpy()))
    print(f"Forward Output Shape: {out_np.shape}")
    print(f"Forward Max Absolute Discrepancy: {forward_diff:.2e}")
    np.testing.assert_allclose(out_np, out_pt.detach().numpy(), rtol=1e-7, atol=1e-7)
    print("[OK] Forward pass numerically identical to PyTorch.")

    # 3. Backward Pass Parity Test
    # Upstream sensitivity gradient dL/dOut
    dout_np = np.random.randn(*out_np.shape).astype(np.float64)
    dout_pt = torch.from_numpy(dout_np)

    # PyTorch Autograd Backward
    out_pt.backward(dout_pt)

    # Scratch NumPy Backward
    dx_np, dw_np, db_np = conv2d_backward_scratch(
        dout_np, x_pad, w_np, stride=stride, padding=padding, orig_h=H, orig_w=W
    )

    # Verify Backward Gradients
    dx_diff = np.max(np.abs(dx_np - x_pt.grad.numpy()))
    dw_diff = np.max(np.abs(dw_np - conv_pt.weight.grad.numpy()))
    db_diff = np.max(np.abs(db_np - conv_pt.bias.grad.numpy()))

    print(f"Input Gradient (dL/dX) Max Discrepancy:  {dx_diff:.2e}")
    print(f"Weight Gradient (dL/dW) Max Discrepancy: {dw_diff:.2e}")
    print(f"Bias Gradient (dL/dB) Max Discrepancy:   {db_diff:.2e}")

    np.testing.assert_allclose(dx_np, x_pt.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(dw_np, conv_pt.weight.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(db_np, conv_pt.bias.grad.numpy(), rtol=1e-7, atol=1e-7)
    print("[OK] Backward gradients numerically identical to PyTorch autograd.")
    print("[OK] All assertions PASSED with exit code 0.")


if __name__ == "__main__":
    run_parity_test()
