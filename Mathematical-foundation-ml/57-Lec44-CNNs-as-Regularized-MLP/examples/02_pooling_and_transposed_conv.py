#!/usr/bin/env python3
"""
Example 02: Spatial Pooling Dynamics & Transposed Convolution from Scratch

This script verifies the spatial decimation and expansion operations analyzed by
Prof. Prathosh in Lecture 44:
1. Average Pooling: Formal proof that average pooling is a fixed, non-learnable convolution
   with constant kernel weights 1 / (k_p^2), acting as a low-pass spatial filter.
2. Max Pooling: Exact forward pass with argmax cache, verifying non-differentiable
   subgradient backpropagation routing dL/dOut exclusively to the coordinate achieving max.
3. Transposed Convolution (Fractionally Strided Convolution): Pure NumPy implementation
   of spatial upsampling as used in U-Net decoder pathways, verifying numerical parity
   with PyTorch nn.ConvTranspose2d.

Author: Mathematical Foundations of ML Course Team
"""

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# 1. AVERAGE POOLING FROM SCRATCH
# =====================================================================
def avg_pool2d_forward_scratch(x, kernel_size=2, stride=2):
    """
    Computes spatial average pooling over input tensor x.
    Shape x: [B, C, H, W] -> Output: [B, C, H_out, W_out]
    """
    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    out = np.zeros((B, C, H_out, W_out), dtype=np.float64)

    for n in range(B):
        for c in range(C):
            for i in range(H_out):
                i_start = i * stride
                i_end = i_start + kernel_size
                for j in range(W_out):
                    j_start = j * stride
                    j_end = j_start + kernel_size
                    patch = x[n, c, i_start:i_end, j_start:j_end]
                    out[n, c, i, j] = np.mean(patch)

    cache = (x.shape, kernel_size, stride)
    return out, cache


def avg_pool2d_backward_scratch(dout, cache):
    """
    Backpropagates upstream sensitivities through average pooling:
    dL/dx_patch = dout / (k * k).
    """
    orig_shape, kernel_size, stride = cache
    B, C, H, W = orig_shape
    _, _, H_out, W_out = dout.shape

    dx = np.zeros(orig_shape, dtype=np.float64)
    scale = 1.0 / (kernel_size * kernel_size)

    for n in range(B):
        for c in range(C):
            for i in range(H_out):
                i_start = i * stride
                i_end = i_start + kernel_size
                for j in range(W_out):
                    j_start = j * stride
                    j_end = j_start + kernel_size
                    delta = dout[n, c, i, j]
                    dx[n, c, i_start:i_end, j_start:j_end] += delta * scale

    return dx


# =====================================================================
# 2. MAX POOLING FROM SCRATCH (SUBGRADIENT ROUTING)
# =====================================================================
def max_pool2d_forward_scratch(x, kernel_size=2, stride=2):
    """
    Computes spatial max pooling and preserves argmax coordinates in binary mask cache.
    Shape x: [B, C, H, W] -> Output: [B, C, H_out, W_out]
    """
    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    out = np.zeros((B, C, H_out, W_out), dtype=np.float64)
    mask = np.zeros_like(x, dtype=np.float64)

    for n in range(B):
        for c in range(C):
            for i in range(H_out):
                i_start = i * stride
                i_end = i_start + kernel_size
                for j in range(W_out):
                    j_start = j * stride
                    j_end = j_start + kernel_size
                    patch = x[n, c, i_start:i_end, j_start:j_end]
                    max_val = np.max(patch)
                    out[n, c, i, j] = max_val

                    # Argmax routing: set mask to 1 only at coordinate of maximum
                    # In case of ties, take the first occurrence
                    flat_idx = np.argmax(patch)
                    unravel_idx = np.unravel_index(flat_idx, patch.shape)
                    mask[n, c, i_start + unravel_idx[0], j_start + unravel_idx[1]] = 1.0

    cache = (mask, x.shape, kernel_size, stride)
    return out, cache


def max_pool2d_backward_scratch(dout, cache):
    """
    Subgradient backpropagation: upstream gradient delta is routed exclusively
    to the argmax coordinate recorded in the forward mask cache.
    """
    mask, orig_shape, kernel_size, stride = cache
    B, C, H, W = orig_shape
    _, _, H_out, W_out = dout.shape

    dx = np.zeros(orig_shape, dtype=np.float64)

    for n in range(B):
        for c in range(C):
            for i in range(H_out):
                i_start = i * stride
                i_end = i_start + kernel_size
                for j in range(W_out):
                    j_start = j * stride
                    j_end = j_start + kernel_size
                    delta = dout[n, c, i, j]
                    # Route gradient using forward argmax mask
                    dx[n, c, i_start:i_end, j_start:j_end] += (
                        delta * mask[n, c, i_start:i_end, j_start:j_end]
                    )

    return dx


# =====================================================================
# 3. 2D TRANSPOSED CONVOLUTION (UPSAMPLING) FROM SCRATCH
# =====================================================================
def conv_transpose2d_forward_scratch(x, w, b, stride=2, padding=0):
    """
    Computes 2D transposed convolution (spatial expansion for U-Net decoders).

    Parameters:
    -----------
    x : np.ndarray, shape [B, C_in, H, W]
    w : np.ndarray, shape [C_in, C_out, k_h, k_w] (PyTorch convention for ConvTranspose2d)
    b : np.ndarray, shape [C_out]
    stride : int
    padding : int

    Returns:
    --------
    out : np.ndarray, shape [B, C_out, H_out, W_out]
    """
    B, C_in, H, W = x.shape
    _, C_out, k_h, k_w = w.shape

    H_out = (H - 1) * stride - 2 * padding + k_h
    W_out = (W - 1) * stride - 2 * padding + k_w

    # Expand accumulator tensor
    H_pad = H_out + 2 * padding
    W_pad = W_out + 2 * padding
    out_pad = np.zeros((B, C_out, H_pad, W_pad), dtype=np.float64)

    for n in range(B):
        for c_in in range(C_in):
            for c_out in range(C_out):
                for i in range(H):
                    i_start = i * stride
                    i_end = i_start + k_h
                    for j in range(W):
                        j_start = j * stride
                        j_end = j_start + k_w
                        # Add scaled kernel to output patch
                        out_pad[n, c_out, i_start:i_end, j_start:j_end] += (
                            x[n, c_in, i, j] * w[c_in, c_out]
                        )

    # Crop padding if padding > 0
    if padding > 0:
        out = out_pad[:, :, padding:padding + H_out, padding:padding + W_out]
    else:
        out = out_pad

    # Add per-channel bias
    for c_out in range(C_out):
        out[:, c_out] += b[c_out]

    return out


def run_all_verifications():
    print("=" * 70)
    print("VERIFICATION: Pooling & Transposed Convolution Parity with PyTorch")
    print("=" * 70)

    np.random.seed(44)
    torch.manual_seed(44)

    # Test 1: Average Pooling Parity
    print("\n--- [1] AVERAGE POOLING TEST ---")
    x_np = np.random.randn(2, 3, 6, 6).astype(np.float64)
    x_pt = torch.from_numpy(x_np).requires_grad_(True)

    avg_out_np, avg_cache = avg_pool2d_forward_scratch(x_np, kernel_size=2, stride=2)
    avg_pt = nn.AvgPool2d(kernel_size=2, stride=2)
    avg_out_pt = avg_pt(x_pt)

    np.testing.assert_allclose(avg_out_np, avg_out_pt.detach().numpy(), rtol=1e-7, atol=1e-7)
    print(f"AvgPool Output Shape: {avg_out_np.shape}")
    print("[OK] AvgPool Forward Pass matches PyTorch.")

    dout = np.random.randn(*avg_out_np.shape).astype(np.float64)
    avg_out_pt.backward(torch.from_numpy(dout))
    dx_avg_np = avg_pool2d_backward_scratch(dout, avg_cache)

    np.testing.assert_allclose(dx_avg_np, x_pt.grad.numpy(), rtol=1e-7, atol=1e-7)
    print("[OK] AvgPool Backward Pass matches PyTorch autograd.")

    # Test 2: Max Pooling Parity (Subgradient Routing)
    print("\n--- [2] MAX POOLING TEST ---")
    x_pt_max = torch.from_numpy(x_np).requires_grad_(True)
    max_out_np, max_cache = max_pool2d_forward_scratch(x_np, kernel_size=2, stride=2)
    max_pt = nn.MaxPool2d(kernel_size=2, stride=2)
    max_out_pt = max_pt(x_pt_max)

    np.testing.assert_allclose(max_out_np, max_out_pt.detach().numpy(), rtol=1e-7, atol=1e-7)
    print(f"MaxPool Output Shape: {max_out_np.shape}")
    print("[OK] MaxPool Forward Pass matches PyTorch.")

    dout_max = np.random.randn(*max_out_np.shape).astype(np.float64)
    max_out_pt.backward(torch.from_numpy(dout_max))
    dx_max_np = max_pool2d_backward_scratch(dout_max, max_cache)

    np.testing.assert_allclose(dx_max_np, x_pt_max.grad.numpy(), rtol=1e-7, atol=1e-7)
    print("[OK] MaxPool Subgradient Backward Pass matches PyTorch autograd.")

    # Test 3: Transposed Convolution Parity (Spatial Upsampling)
    print("\n--- [3] TRANSPOSED CONVOLUTION (UPSAMPLING) TEST ---")
    B, C_in, H_in, W_in = 2, 4, 3, 3
    C_out = 3
    k_h, k_w = 3, 3
    stride = 2
    padding = 1

    x_up_np = np.random.randn(B, C_in, H_in, W_in).astype(np.float64)
    w_up_np = np.random.randn(C_in, C_out, k_h, k_w).astype(np.float64)
    b_up_np = np.random.randn(C_out).astype(np.float64)

    up_out_np = conv_transpose2d_forward_scratch(
        x_up_np, w_up_np, b_up_np, stride=stride, padding=padding
    )

    conv_tr_pt = nn.ConvTranspose2d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=(k_h, k_w),
        stride=stride,
        padding=padding,
        bias=True
    ).to(torch.float64)

    with torch.no_grad():
        conv_tr_pt.weight.copy_(torch.from_numpy(w_up_np))
        conv_tr_pt.bias.copy_(torch.from_numpy(b_up_np))

    up_out_pt = conv_tr_pt(torch.from_numpy(x_up_np))

    np.testing.assert_allclose(up_out_np, up_out_pt.detach().numpy(), rtol=1e-7, atol=1e-7)
    print(f"Transposed Conv Input Shape:  {x_up_np.shape}")
    print(f"Transposed Conv Output Shape: {up_out_np.shape} (Expanded via Stride {stride})")
    print("[OK] Transposed Conv2d Forward Pass matches PyTorch.")

    print("\n[OK] All pooling and upsampling verifications PASSED with exit code 0.")


if __name__ == "__main__":
    run_all_verifications()
