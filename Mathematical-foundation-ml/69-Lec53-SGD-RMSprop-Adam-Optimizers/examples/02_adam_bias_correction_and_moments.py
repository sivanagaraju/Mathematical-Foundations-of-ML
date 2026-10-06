"""
Example 2: Adam First and Second Moment Dynamics with Bias Correction
Lecture 53: SGD, RMSprop, Adam: Optimizers

This script provides an exact mathematical and computational verification of:
1. From-scratch Adam implementation following Kingma & Ba (2014) and Lecture 53 formulas:
   - First raw moment:   m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
   - Second raw moment:  v_t = beta_2 * v_{t-1} + (1 - beta_2) * (g_t ** 2)
   - Bias correction 1:  m_hat_t = m_t / (1 - beta_1 ** t)
   - Bias correction 2:  v_hat_t = v_t / (1 - beta_2 ** t)
   - Parameter update:   theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + eps)
2. Exact numerical parity with PyTorch's native torch.optim.Adam.
3. Analysis of the bias-correction distortion factor at step t=1:
   Showing why without bias correction, initial steps suffer a ~3.16x magnitude distortion.
"""

import math
import torch
import torch.nn as nn

def verify_adam_from_scratch():
    """
    Verifies step-by-step from-scratch Adam against PyTorch's native C++ implementation.
    """
    print("-" * 70)
    print("PART 1: From-Scratch Adam vs PyTorch torch.optim.Adam Numerical Parity")
    print("-" * 70)

    # Initialize identical weight vectors
    initial_w = torch.tensor([2.5, -1.8, 0.75, -0.4], dtype=torch.float32)  # Shape: [4]
    w_manual = initial_w.clone()                                            # Shape: [4]
    w_torch = initial_w.clone().requires_grad_(True)                        # Shape: [4]

    # Hyperparameters
    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    num_steps = 20

    opt_torch = torch.optim.Adam([w_torch], lr=alpha, betas=(beta1, beta2), eps=eps)

    # Initialize moment accumulators to zero
    m = torch.zeros_like(w_manual)  # Shape: [4]
    v = torch.zeros_like(w_manual)  # Shape: [4]

    print(f"Step | Max Absolute Diff | Manual Norm  | PyTorch Norm")
    print("-" * 55)

    for t in range(1, num_steps + 1):
        # Synthetic non-stationary gradient simulating minibatch updates
        g = torch.tensor([
            0.5 * math.sin(t),
            -0.2 * math.cos(t),
            0.1 * (t ** 0.5),
            -0.05 * t
        ], dtype=torch.float32)  # Shape: [4]

        # 1. Update biased 1st moment estimate
        m = beta1 * m + (1.0 - beta1) * g  # Shape: [4]
        # 2. Update biased 2nd raw moment estimate
        v = beta2 * v + (1.0 - beta2) * (g ** 2)  # Shape: [4]
        # 3. Compute bias-corrected 1st moment estimate
        m_hat = m / (1.0 - (beta1 ** t))  # Shape: [4]
        # 4. Compute bias-corrected 2nd raw moment estimate
        v_hat = v / (1.0 - (beta2 ** t))  # Shape: [4]
        # 5. Coordinate-wise update
        w_manual = w_manual - alpha * (m_hat / (torch.sqrt(v_hat) + eps))  # Shape: [4]

        # PyTorch Adam step
        opt_torch.zero_grad()
        w_torch.grad = g.clone()
        opt_torch.step()

        # Compute numerical difference
        diff = torch.max(torch.abs(w_manual - w_torch)).item()
        print(f"{t:4d} | {diff:17.10e} | {torch.norm(w_manual).item():12.6f} | {torch.norm(w_torch).item():12.6f}")

        # Strict parity assertion
        assert torch.allclose(w_manual, w_torch, atol=1e-7), f"Parity mismatch at step {t}: max diff = {diff}"

    print("[OK] Part 1 Verified: Exact numerical equivalence with torch.optim.Adam confirmed.")

def verify_bias_correction_dynamics():
    """
    Analyzes the mathematical role of bias correction at t=1.
    Without bias correction:
        m_1 = (1 - beta1) * g_1
        v_1 = (1 - beta2) * g_1^2
        Step = alpha * m_1 / sqrt(v_1) = alpha * [(1 - beta1) / sqrt(1 - beta2)] * sign(g_1)
        For beta1=0.9, beta2=0.999:
        Ratio = 0.1 / sqrt(0.001) = 0.1 / 0.03162277 = sqrt(10) ~ 3.162277
    With bias correction:
        m_hat_1 = m_1 / (1 - beta1) = g_1
        v_hat_1 = v_1 / (1 - beta2) = g_1^2
        Step = alpha * m_hat_1 / sqrt(v_hat_1) = alpha * sign(g_1) (exact unit step!)
    """
    print("\n" + "-" * 70)
    print("PART 2: Analytical & Numerical Proof of Bias Correction Mechanics")
    print("-" * 70)

    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.001
    eps = 1e-8

    g1 = torch.tensor([4.0, -2.5, 0.1, -100.0])  # Shape: [4]

    # Without bias correction
    m1_uncorrected = (1.0 - beta1) * g1
    v1_uncorrected = (1.0 - beta2) * (g1 ** 2)
    step_uncorrected = alpha * m1_uncorrected / (torch.sqrt(v1_uncorrected) + eps)

    # With bias correction
    m1_corrected = m1_uncorrected / (1.0 - beta1 ** 1)
    v1_corrected = v1_uncorrected / (1.0 - beta2 ** 1)
    step_corrected = alpha * m1_corrected / (torch.sqrt(v1_corrected) + eps)

    theoretical_distortion = (1.0 - beta1) / math.sqrt(1.0 - beta2)
    print(f"Hyperparameters: beta1 = {beta1}, beta2 = {beta2}")
    print(f"Theoretical uncorrected scale factor: (1 - beta1) / sqrt(1 - beta2) = {theoretical_distortion:.6f}")
    print(f"Observed step ratio (uncorrected / corrected):")
    for i in range(len(g1)):
        ratio = (step_uncorrected[i] / step_corrected[i]).item()
        print(f"  Coord {i} (g={g1[i].item():6.1f}): uncorrected={step_uncorrected[i].item():.6f}, corrected={step_corrected[i].item():.6f}, ratio={ratio:.6f}")
        assert math.isclose(ratio, theoretical_distortion, rel_tol=1e-4), "Ratio must match theoretical factor"

    # Verify that corrected step magnitude is exactly alpha * sign(g)
    for i in range(len(g1)):
        expected_step = alpha * math.copysign(1.0, g1[i].item())
        assert math.isclose(step_corrected[i].item(), expected_step, rel_tol=1e-4)

    print(f"[OK] Part 2 Verified: Bias correction eliminates the {theoretical_distortion:.4f}x distortion factor at step 1.")

def main():
    print("=" * 70)
    print("LECTURE 53 SCRIPT 2: ADAM MOMENTUM & BIAS CORRECTION MECHANICS")
    print("=" * 70)
    verify_adam_from_scratch()
    verify_bias_correction_dynamics()
    print("\n" + "=" * 70)
    print("[ALL CHECKS PASSED] Adam mechanics, formulas, and bias correction verified.")
    print("=" * 70)

if __name__ == "__main__":
    main()
