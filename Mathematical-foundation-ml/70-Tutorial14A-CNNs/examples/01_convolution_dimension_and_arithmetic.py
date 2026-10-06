"""
Tutorial 14A: Example 1 - 2D Convolution Dimension Arithmetic & CS231n Hand Calculation
======================================================================================
This script implements:
1. Spatial output dimension arithmetic formula:
      H_out = floor((H_in - F + 2*P) / S) + 1
      W_out = floor((W_in - F + 2*P) / S) + 1
      D_out = K  (number of filters)
2. Numerical verification of the CS231n animated visualizer from lecture:
      Input volume: (5 x 5 x 3), Padding P=1, Stride S=2, Filter size F=3
      Output spatial dimensions: (3 x 3 x 2) for K=2 filters
3. Exact hand-calculation inner product matching the lecture:
      Receptive field at top-left corner across 3 channels produces:
      Channel 0 dot product = -3
      Channel 1 dot product = -1
      Channel 2 dot product = -2
      Bias = 0
      Total output scalar = -3 + (-1) + (-2) + 0 = -6.
4. Equivalence check comparing manual pure-Python convolution against torch.nn.functional.conv2d.
"""

import math
import torch
import torch.nn.functional as F

def compute_conv_output_dim(h_in: int, w_in: int, f: int, p: int, s: int, k: int):
    """
    Computes output spatial dimensions and channel depth for a 2D convolution.
    
    Formula:
        H_out = floor((H_in - F + 2*P) / S) + 1
        W_out = floor((W_in - F + 2*P) / S) + 1
        D_out = K
    """
    h_out = math.floor((h_in - f + 2 * p) / s) + 1
    w_out = math.floor((w_in - f + 2 * p) / s) + 1
    d_out = k
    return h_out, w_out, d_out


def run_dimension_formula_checks():
    print("=== Step 1: Spatial Output Dimension Formula Checks ===")
    
    # CS231n lecture configuration: 5x5x3, F=3, P=1, S=2, K=2
    h_out, w_out, d_out = compute_conv_output_dim(h_in=5, w_in=5, f=3, p=1, s=2, k=2)
    print(f"CS231n visualizer: 5x5x3 with F=3, P=1, S=2, K=2 -> Output: ({h_out}, {w_out}, {d_out})")
    assert (h_out, w_out, d_out) == (3, 3, 2), f"Expected (3, 3, 2), got ({h_out}, {w_out}, {d_out})"

    # LeNet-5 Conv1 (Padded): 28x28x1, F=5, P=2, S=1, K=6
    h_lenet1, w_lenet1, d_lenet1 = compute_conv_output_dim(h_in=28, w_in=28, f=5, p=2, s=1, k=6)
    print(f"LeNet-5 Conv1 (P=2): 28x28x1 -> Output: ({h_lenet1}, {w_lenet1}, {d_lenet1})")
    assert (h_lenet1, w_lenet1, d_lenet1) == (28, 28, 6)

    # LeNet-5 Conv2 (No pad): 14x14x6, F=5, P=0, S=1, K=16
    h_lenet2, w_lenet2, d_lenet2 = compute_conv_output_dim(h_in=14, w_in=14, f=5, p=0, s=1, k=16)
    print(f"LeNet-5 Conv2 (P=0): 14x14x6 -> Output: ({h_lenet2}, {w_lenet2}, {d_lenet2})")
    assert (h_lenet2, w_lenet2, d_lenet2) == (10, 10, 16)

    # Standard ImageNet / AlexNet Conv1: 224x224x3, F=11, P=2, S=4, K=96 (or 64)
    h_alex, w_alex, d_alex = compute_conv_output_dim(h_in=224, w_in=224, f=11, p=2, s=4, k=96)
    print(f"AlexNet Conv1: 224x224x3 with F=11, P=2, S=4, K=96 -> Output: ({h_alex}, {w_alex}, {d_alex})")
    assert (h_alex, w_alex, d_alex) == (55, 55, 96)

    print("[PASS] Dimension formula checks passed successfully.")


def run_cs231n_numerical_hand_calculation():
    print("\n=== Step 2: CS231n Hand Calculation Exact Verification ===")
    
    # In the lecture, at timestamp [09:55 - 12:19], the instructor evaluates
    # the 3x3 receptive field at top-left for Filter 1 (W1):
    #
    # Padded Input 3x3 patch for Channel 0 (including zero padding):
    #   [ 0,  0,  0 ]
    #   [ 0,  2,  2 ]
    #   [ 0,  1,  0 ]
    # Kernel W1 Channel 0:
    #   [ 0, -1,  0 ]
    #   [ 1, -1, -1 ]
    #   [ 1,  1, -1 ]
    # Multiplication for Channel 0:
    #   Row 0: 0*0 + 0*(-1) + 0*0 = 0
    #   Row 1: 0*1 + 2*(-1) + 2*(-1) = -2 - 2 = -4
    #   Row 2: 0*1 + 1*1 + 0*(-1) = +1
    #   Sum = -4 + 1 = -3.
    #
    # Padded Input 3x3 patch for Channel 1:
    #   [ 0,  0,  0 ]
    #   [ 0,  2,  2 ]
    #   [ 0,  0,  1 ]
    # Kernel W1 Channel 1:
    #   [ 1,  0,  1 ]
    #   [ 0,  1, -1 ]
    #   [ 0, -1, -1 ]
    # Multiplication for Channel 1:
    #   Row 0: 0
    #   Row 1: 0*0 + 2*1 + 2*(-1) = 2 - 2 = 0
    #   Row 2: 0*0 + 0*(-1) + 1*(-1) = -1
    #   Sum = -1.
    #
    # Padded Input 3x3 patch for Channel 2:
    #   [ 0,  0,  0 ]
    #   [ 0,  0,  2 ]
    #   [ 0,  0,  0 ]
    # Kernel W1 Channel 2:
    #   [ 0,  1,  0 ]
    #   [ 0,  0, -1 ]
    #   [ 1,  0,  1 ]
    # Multiplication for Channel 2:
    #   Row 0: 0
    #   Row 1: 0*0 + 0*0 + 2*(-1) = -2
    #   Row 2: 0
    #   Sum = -2.
    #
    # Sum across all channels:
    #   Ch0 + Ch1 + Ch2 = (-3) + (-1) + (-2) = -6
    # Bias b1 = 0
    # Total output = -6 + 0 = -6.

    ch0_in = torch.tensor([[0.0, 0.0, 0.0],
                           [0.0, 2.0, 2.0],
                           [0.0, 1.0, 0.0]])
    ch0_w  = torch.tensor([[0.0, -1.0, 0.0],
                           [1.0, -1.0, -1.0],
                           [1.0,  1.0, -1.0]])

    ch1_in = torch.tensor([[0.0, 0.0, 0.0],
                           [0.0, 2.0, 2.0],
                           [0.0, 0.0, 1.0]])
    ch1_w  = torch.tensor([[1.0,  0.0,  1.0],
                           [0.0,  1.0, -1.0],
                           [0.0, -1.0, -1.0]])

    ch2_in = torch.tensor([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 2.0],
                           [0.0, 0.0, 0.0]])
    ch2_w  = torch.tensor([[0.0,  1.0,  0.0],
                           [0.0,  0.0, -1.0],
                           [1.0,  0.0,  1.0]])

    bias = 0.0

    prod_ch0 = torch.sum(ch0_in * ch0_w).item()
    prod_ch1 = torch.sum(ch1_in * ch1_w).item()
    prod_ch2 = torch.sum(ch2_in * ch2_w).item()
    total_scalar = prod_ch0 + prod_ch1 + prod_ch2 + bias

    print(f"Channel 0 inner product: {prod_ch0} (Expected: -3.0)")
    print(f"Channel 1 inner product: {prod_ch1} (Expected: -1.0)")
    print(f"Channel 2 inner product: {prod_ch2} (Expected: -2.0)")
    print(f"Bias term: {bias}")
    print(f"Total convolution output at (0, 0): {total_scalar} (Expected: -6.0)")

    assert prod_ch0 == -3.0, f"Channel 0 mismatch: {prod_ch0}"
    assert prod_ch1 == -1.0, f"Channel 1 mismatch: {prod_ch1}"
    assert prod_ch2 == -2.0, f"Channel 2 mismatch: {prod_ch2}"
    assert total_scalar == -6.0, f"Total scalar mismatch: {total_scalar}"

    print("[PASS] Lecture CS231n manual calculation verified.")


def run_pytorch_equivalence():
    print("\n=== Step 3: PyTorch nn.functional.conv2d Equivalence ===")
    
    # Build complete 5x5x3 input matching CS231n demo slice
    # Shape: [batch=1, channels=3, height=5, width=5]
    torch.manual_seed(42)
    x = torch.zeros(1, 3, 5, 5)
    # Populate the top-left 2x2 unpadded region that corresponds to our 3x3 padded patch:
    # In padded coordinate [1:3, 1:3], unpadded is [0:2, 0:2]
    x[0, 0, 0:2, 0:2] = torch.tensor([[2.0, 2.0],
                                      [1.0, 0.0]])
    x[0, 1, 0:2, 0:2] = torch.tensor([[2.0, 2.0],
                                      [0.0, 1.0]])
    x[0, 2, 0:2, 0:2] = torch.tensor([[0.0, 2.0],
                                      [0.0, 0.0]])

    # Build 1 filter of size 3x3x3:
    # Shape: [out_channels=1, in_channels=3, kernel_h=3, kernel_w=3]
    w = torch.zeros(1, 3, 3, 3)
    w[0, 0] = torch.tensor([[0.0, -1.0, 0.0],
                            [1.0, -1.0, -1.0],
                            [1.0,  1.0, -1.0]])
    w[0, 1] = torch.tensor([[1.0,  0.0,  1.0],
                            [0.0,  1.0, -1.0],
                            [0.0, -1.0, -1.0]])
    w[0, 2] = torch.tensor([[0.0,  1.0,  0.0],
                            [0.0,  0.0, -1.0],
                            [1.0,  0.0,  1.0]])
    b = torch.tensor([0.0])

    # Run PyTorch conv2d with padding=1, stride=2
    out = F.conv2d(x, w, bias=b, stride=2, padding=1)
    
    # Shape of out: [1, 1, 3, 3]
    print(f"Input shape:  {list(x.shape)}")
    print(f"Weight shape: {list(w.shape)}")
    print(f"Output shape: {list(out.shape)}")
    print(f"PyTorch computed output[0, 0, 0, 0] = {out[0, 0, 0, 0].item()}")

    assert out.shape == (1, 1, 3, 3), f"Expected shape (1, 1, 3, 3), got {out.shape}"
    assert torch.isclose(out[0, 0, 0, 0], torch.tensor(-6.0)), f"PyTorch output {out[0, 0, 0, 0]} != -6.0"

    print("[PASS] PyTorch conv2d output exactly equals hand calculation (-6.0).")


if __name__ == "__main__":
    run_dimension_formula_checks()
    run_cs231n_numerical_hand_calculation()
    run_pytorch_equivalence()
    print("\nAll 2D convolution arithmetic checks passed successfully!")
