"""
02_permutation_equivariance_and_rope.py
=======================================
Verification of Permutation Equivariance in Transformers and modern
Rotary Position Embedding (RoPE) relative dot product invariance.

Verifies:
1. Pure self-attention is strictly permutation equivariant: Attn(Pi @ X) = Pi @ Attn(X).
2. Addition of Positional Encoding (PE) successfully breaks permutation equivariance.
3. Rotary Position Embedding (RoPE) rotates 2D subspace coordinates.
4. RoPE inner product <R_m q, R_n k> depends strictly on relative offset (m - n).
"""

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Standard scaled dot-product attention: softmax(Q @ K.T / sqrt(d_k)) @ V
    """
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ V


def apply_rope_2d(x: torch.Tensor, pos: int, theta: float) -> torch.Tensor:
    """
    Applies 2D RoPE rotation matrix to a 2D vector [x0, x1] at position `pos`.
    R_pos = [[cos(pos*theta), -sin(pos*theta)],
             [sin(pos*theta),  cos(pos*theta)]]
    """
    angle = pos * theta
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x0, x1 = x[0].item(), x[1].item()
    return torch.tensor([
        x0 * cos_a - x1 * sin_a,
        x0 * sin_a + x1 * cos_a
    ], dtype=x.dtype, device=x.device)


def verify_permutation_and_rope():
    print("=" * 70)
    print("TEST 1: Permutation Equivariance of Pure Self-Attention")
    print("=" * 70)
    torch.manual_seed(1337)
    T, D = 4, 16
    X = torch.randn(T, D) # Shape: [T, D]

    # Random projection matrices
    W_q = torch.randn(D, D) # Shape: [D, D]
    W_k = torch.randn(D, D) # Shape: [D, D]
    W_v = torch.randn(D, D) # Shape: [D, D]

    def forward_unpositioned(tokens):
        # Shape: tokens is [T, D]
        Q = tokens @ W_q # Shape: [T, D]
        K = tokens @ W_k # Shape: [T, D]
        V = tokens @ W_v # Shape: [T, D]
        return scaled_dot_product_attention(Q, K, V) # Shape: [T, D]

    Z_original = forward_unpositioned(X) # Shape: [T, D]

    # Permutation matrix swapping tokens (0 -> 3, 1 -> 0, 2 -> 1, 3 -> 2)
    perm_order = torch.tensor([3, 0, 1, 2])
    Pi = torch.eye(T)[perm_order] # Shape: [T, T]
    X_perm = Pi @ X # Shape: [T, D]

    Z_perm = forward_unpositioned(X_perm) # Shape: [T, D]
    Z_expected = Pi @ Z_original # Shape: [T, D]

    max_equivariance_err = (Z_perm - Z_expected).abs().max().item()
    print(f"Permuted Input Shape: {X_perm.shape}")
    print(f"Max Equivariance Discrepancy: {max_equivariance_err:.2e}")
    assert max_equivariance_err < 1e-5, f"Equivariance failed: {max_equivariance_err}"
    print("[OK] Unpositioned attention is strictly permutation equivariant.\n")

    print("=" * 70)
    print("TEST 2: Breaking Permutation Equivariance with Positional Encoding")
    print("=" * 70)
    # Generate sinusoidal PE
    pe = torch.zeros(T, D) # Shape: [T, D]
    position = torch.arange(0, T, dtype=torch.float).unsqueeze(1) # Shape: [T, 1]
    div_term = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D)) # Shape: [D / 2]
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # Add PE to original and permuted sequences
    # Note: Tokens are permuted, but positions remain fixed slots 0..T-1!
    X_pos_orig = X + pe # Shape: [T, D]
    X_pos_perm = (Pi @ X) + pe # Shape: [T, D]

    Z_pos_orig = forward_unpositioned(X_pos_orig) # Shape: [T, D]
    Z_pos_perm = forward_unpositioned(X_pos_perm) # Shape: [T, D]
    Z_pos_expected_if_equivariant = Pi @ Z_pos_orig # Shape: [T, D]

    order_sensitivity_diff = (Z_pos_perm - Z_pos_expected_if_equivariant).abs().max().item()
    print(f"Order Sensitivity Divergence: {order_sensitivity_diff:.4f}")
    assert order_sensitivity_diff > 0.05, "Positional encoding failed to break permutation equivariance!"
    print("[OK] Positional encoding successfully sensitized model to sequence ordering.\n")

    print("=" * 70)
    print("TEST 3: Rotary Position Embedding (RoPE) Relative Invariance")
    print("=" * 70)
    # RoPE property: <R_m q, R_n k> depends strictly on relative offset (m - n)
    theta = 0.05
    q_vec = torch.randn(2)
    k_vec = torch.randn(2)

    # Test at base positions m=10, n=4 (relative offset = 6)
    m1, n1 = 10, 4
    q_rot_1 = apply_rope_2d(q_vec, pos=m1, theta=theta)
    k_rot_1 = apply_rope_2d(k_vec, pos=n1, theta=theta)
    dot_product_1 = torch.dot(q_rot_1, k_rot_1).item()

    # Shift both positions by arbitrary offset s = 35 (m=45, n=39; relative offset still 6)
    shift_s = 35
    m2, n2 = m1 + shift_s, n1 + shift_s
    q_rot_2 = apply_rope_2d(q_vec, pos=m2, theta=theta)
    k_rot_2 = apply_rope_2d(k_vec, pos=n2, theta=theta)
    dot_product_2 = torch.dot(q_rot_2, k_rot_2).item()

    rope_diff = abs(dot_product_1 - dot_product_2)
    print(f"Base Inner Product (m={m1}, n={n1}, diff={m1-n1}):   {dot_product_1:.6f}")
    print(f"Shifted Inner Product (m={m2}, n={n2}, diff={m2-n2}): {dot_product_2:.6f}")
    print(f"Discrepancy: {rope_diff:.2e}")

    assert rope_diff < 1e-6, f"RoPE relative shift invariance failed: diff = {rope_diff}"
    print("[OK] RoPE relative distance invariance mathematically verified.\n")

    print("ALL ROPE AND PERMUTATION SUITES PASSED CLEANLY (EXIT CODE 0).")


if __name__ == "__main__":
    verify_permutation_and_rope()
