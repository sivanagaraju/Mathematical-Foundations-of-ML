#!/usr/bin/env python3
"""
Tutorial 11 : PyTorch - Tensors and Data Loaders
Simulation Script 01: Tensor Memory Layout, Zero-Copy Views, and Linear Algebra Verification

This standalone script verifies the core computational and linear algebra mechanics of
PyTorch tensors:
1. Memory buffer allocation and zero-copy sharing between NumPy ndarrays and PyTorch tensors.
2. Coordinate strides, memory offsets, and contiguous memory layouts.
3. Strict parity between PyTorch GEMM operators (@, torch.matmul, torch.mm) and NumPy matmul.
4. Hadamard coordinate-wise multiplication and its role in sliding-window 2D convolution.
5. Axis concatenation geometry across batch (dim=0) and feature (dim=1) dimensions.
6. Scalar reduction extraction and memory detachment via .item().

All assertions execute cleanly with exit code 0.
"""

import sys
import numpy as np
import torch


def verify_memory_and_views():
    print("[1/5] Verifying Tensor Memory Layout and Zero-Copy Sharing...")
    
    # 1. Native Python list to PyTorch Tensor (copies memory)
    py_list = [1.0, 2.0, 3.0, 4.0]
    t_from_list = torch.tensor(py_list, dtype=torch.float32)
    # Shape: [4]
    assert t_from_list.shape == torch.Size([4])
    assert t_from_list.dtype == torch.float32
    assert t_from_list.device.type == "cpu"
    
    # 2. NumPy ndarray to PyTorch Tensor via torch.from_numpy (zero-copy view)
    np_arr = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    # Shape: [2, 3]
    t_from_np = torch.from_numpy(np_arr)
    
    # Verify shared underlying memory buffer
    assert t_from_np.data_ptr() == np_arr.ctypes.data, "Pointers must match for zero-copy view!"
    
    # In-place mutation on NumPy reflects in PyTorch
    np_arr[0, 1] = 999.0
    assert t_from_np[0, 1].item() == 999.0, "Mutation in NumPy must reflect in PyTorch tensor!"
    
    # In-place mutation on PyTorch reflects in NumPy
    t_from_np[1, 2] = -42.0
    assert np_arr[1, 2] == -42.0, "Mutation in PyTorch must reflect in NumPy array!"
    
    # Verify stride layout
    # Shape [2, 3], row-major strides should be (3, 1) in elements
    assert t_from_np.stride() == (3, 1), f"Expected stride (3, 1), got {t_from_np.stride()}"
    assert t_from_np.is_contiguous(), "Tensor must be contiguous in C-order!"
    
    print("      [OK] Zero-copy memory sharing and stride geometry verified.")


def verify_linear_algebra_gemm():
    print("[2/5] Verifying Matrix Multiplication (GEMM) Interfaces...")
    
    # Dimensions: A in R^{3 x 4}, B in R^{4 x 5}
    m, k, n = 3, 4, 5
    torch.manual_seed(42)
    np.random.seed(42)
    
    A_np = np.random.randn(m, k).astype(np.float32)
    B_np = np.random.randn(k, n).astype(np.float32)
    
    A_th = torch.from_numpy(A_np)  # Shape: [3, 4]
    B_th = torch.from_numpy(B_np)  # Shape: [4, 5]
    
    # 1. NumPy reference GEMM
    C_np_ref = np.matmul(A_np, B_np)  # Shape: [3, 5]
    
    # 2. PyTorch @ infix operator
    C_th_infix = A_th @ B_th  # Shape: [3, 5]
    
    # 3. PyTorch functional matmul
    C_th_matmul = torch.matmul(A_th, B_th)  # Shape: [3, 5]
    
    # 4. PyTorch low-level mm with pre-allocated buffer
    C_th_out = torch.empty((m, n), dtype=torch.float32)  # Shape: [3, 5]
    torch.mm(A_th, B_th, out=C_th_out)
    
    # Assert exact numerical parity
    np.testing.assert_allclose(C_th_infix.numpy(), C_np_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(C_th_matmul.numpy(), C_np_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(C_th_out.numpy(), C_np_ref, rtol=1e-5, atol=1e-6)
    
    print("      [OK] Infix @, torch.matmul, and torch.mm(out=...) all match NumPy GEMM.")


def verify_hadamard_and_convolution():
    print("[3/5] Verifying Hadamard Element-Wise Products & Convolution Decomposition...")
    
    # 1. Coordinate-wise Hadamard product
    # Shapes: [2, 3] and [2, 3]
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    W = torch.tensor([[0.5, 1.0, 2.0], [0.1, 0.2, 0.3]], dtype=torch.float32)
    
    H_prod = X * W  # Shape: [2, 3]
    H_func = torch.mul(X, W)  # Shape: [2, 3]
    assert torch.equal(H_prod, H_func), "Infix * and torch.mul must be identical!"
    
    expected_H = torch.tensor([[0.5, 2.0, 6.0], [0.4, 1.0, 1.8]], dtype=torch.float32)
    assert torch.allclose(H_prod, expected_H, atol=1e-6)
    
    # 2. Decomposing 2D Convolution into Hadamard Product + Scalar Reduction Sum
    # Receptive field patch P in R^{3 x 3}, filter kernel K in R^{3 x 3}
    P = torch.tensor([
        [1.0, 2.0, 1.0],
        [0.0, 1.0, 0.0],
        [2.0, 1.0, 3.0]
    ], dtype=torch.float32)  # Shape: [3, 3]
    
    K = torch.tensor([
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0]
    ], dtype=torch.float32)  # Sobel vertical edge filter, Shape: [3, 3]
    
    # Convolution step: Y = sum_{u, v} K_{uv} * P_{uv}
    conv_manual = (P * K).sum().item()
    
    # Verification via PyTorch official conv2d
    P_batched = P.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]
    K_batched = K.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]
    conv_official = torch.nn.functional.conv2d(P_batched, K_batched).item()
    
    assert abs(conv_manual - conv_official) < 1e-6, f"Manual {conv_manual} != Official {conv_official}"
    print(f"      [OK] 2D Convolution decomposed into Hadamard + Sum: Output = {conv_manual:.4f}")


def verify_concatenation_geometry():
    print("[4/5] Verifying Tensor Concatenation Geometry (dim=0 vs dim=1)...")
    
    # Input tensor Z in R^{3 x 4}
    Z = torch.ones((3, 4), dtype=torch.float32)
    
    # Concatenation along dim=0 (Batch stacking)
    cat_dim0 = torch.cat([Z, Z], dim=0)  # Shape: [6, 4]
    assert cat_dim0.shape == (6, 4), f"Expected (6, 4), got {cat_dim0.shape}"
    
    # Concatenation along dim=1 (Feature stacking)
    cat_dim1 = torch.cat([Z, Z, Z], dim=1)  # Shape: [3, 12]
    assert cat_dim1.shape == (3, 12), f"Expected (3, 12), got {cat_dim1.shape}"
    
    print("      [OK] Concatenation along dim=0 (batch: [6, 4]) and dim=1 (feature: [3, 12]) verified.")


def verify_reductions_and_item():
    print("[5/5] Verifying Reductions and .item() Host Memory Extraction...")
    
    # Tensor in R^{4 x 4}
    T = torch.full((4, 4), 2.5, dtype=torch.float32)
    
    # Sum reduction
    T_sum = T.sum()  # Shape: []
    assert T_sum.ndim == 0, "Scalar tensor must have 0 dimensions!"
    assert isinstance(T_sum, torch.Tensor), "T_sum is still a torch.Tensor!"
    
    # Host extraction via .item()
    val = T_sum.item()
    assert isinstance(val, float), "val must be a native Python float!"
    assert abs(val - 40.0) < 1e-6, f"Expected 40.0, got {val}"
    
    print(f"      [OK] Reduction to scalar tensor and native Python float via .item() verified ({val}).")


if __name__ == "__main__":
    print("=" * 70)
    print("TUTORIAL 11 : PYTORCH TENSORS & MEMORY SIMULATION SUITE")
    print("=" * 70)
    verify_memory_and_views()
    verify_linear_algebra_gemm()
    verify_hadamard_and_convolution()
    verify_concatenation_geometry()
    verify_reductions_and_item()
    print("=" * 70)
    print("ALL NUMERICAL & MEMORY ASSERTIONS PASSED (EXIT CODE 0)")
    print("=" * 70)
    sys.exit(0)
