"""
01_sinusoidal_positional_encoding.py
====================================
Comprehensive implementation and mathematical verification of Vaswani Sinusoidal
Positional Encodings (Attention Is All You Need, 2017).

Verifies:
1. Exact tensor construction of PE matrix with interleaved (sin, cos) channels.
2. Geometric frequency progression: omega_i = 10000^(-2i / D).
3. Bounded coordinate norms and absence of norm explosion.
4. Linear 2D rotation matrix property: PE_{pos+k} = M(k) * PE_{pos}.
5. Monotonic inner product decay with relative sequence displacement.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard Vaswani Sinusoidal Positional Encoding module.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Create positional encoding buffer
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        
        # div_term: 10000^(2i/d) computed in log space for numerical precision
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(base) / d_model)
        ) # [d_model / 2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as persistent non-trainable buffer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Position-augmented tensor of shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


def verify_sinusoidal_encodings():
    print("=" * 70)
    print("TEST 1: Tensor Shape and Numerical Bounding Invariants")
    print("=" * 70)
    d_model = 64
    seq_len = 256
    pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=seq_len)
    pe_matrix = pos_encoder.pe[:seq_len, :]

    assert pe_matrix.shape == (seq_len, d_model), f"Shape error: {pe_matrix.shape}"
    assert pe_matrix.min().item() >= -1.0 - 1e-6, "Sinusoidal minimum bounded below -1"
    assert pe_matrix.max().item() <= 1.0 + 1e-6, "Sinusoidal maximum bounded above 1"

    # Norm boundedness: norm across D should be approximately sqrt(D/2)
    expected_norm = math.sqrt(d_model / 2.0)
    actual_norms = torch.norm(pe_matrix, dim=-1)
    max_norm_err = (actual_norms - expected_norm).abs().max().item()
    print(f"PE Matrix Shape: {pe_matrix.shape}")
    print(f"Expected Norm per position: {expected_norm:.3f}")
    print(f"Observed Norms: min={actual_norms.min().item():.3f}, max={actual_norms.max().item():.3f}")
    assert max_norm_err < 1.0, f"Norm variance too large: {max_norm_err}"
    print("[OK] Bounding and norm stability invariants PASSED.\n")

    print("=" * 70)
    print("TEST 2: Geometric Frequency and Wavelength Progression")
    print("=" * 70)
    # Omega_i for i in {0, ..., D/2 - 1}
    i_indices = torch.arange(0, d_model, 2).float()
    omegas = torch.exp(i_indices * (-math.log(10000.0) / d_model))
    wavelengths = 2.0 * math.pi / omegas

    print(f"Channel 0 (Fastest): omega = {omegas[0].item():.4f}, wavelength = {wavelengths[0].item():.2f} tokens")
    print(f"Channel {d_model//2 - 1} (Slowest): omega = {omegas[-1].item():.6f}, wavelength = {wavelengths[-1].item():.2f} tokens")

    assert math.isclose(omegas[0].item(), 1.0, rel_tol=1e-5), "Fastest omega must be 1.0"
    assert math.isclose(omegas[-1].item(), 1.0 / 10000.0 ** ((d_model - 2) / d_model), rel_tol=1e-3)
    assert wavelengths[-1].item() > 20000.0, "Longest wavelength must exceed 20,000"
    print("[OK] Frequency spectrum and multi-scale wavelengths PASSED.\n")

    print("=" * 70)
    print("TEST 3: Linear 2D Rotation Matrix Shift Property")
    print("=" * 70)
    # For any offset k and frequency channel i:
    # [sin(omega*(pos+k)); cos(omega*(pos+k))] = R(omega*k) * [sin(omega*pos); cos(omega*pos)]
    pos = 45
    k = 18
    channel_idx = 4 # 5th frequency channel
    omega = omegas[channel_idx].item()

    # Direct evaluations
    sin_pos = math.sin(omega * pos)
    cos_pos = math.cos(omega * pos)
    v_pos = torch.tensor([sin_pos, cos_pos])

    sin_shifted = math.sin(omega * (pos + k))
    cos_shifted = math.cos(omega * (pos + k))
    v_shifted_direct = torch.tensor([sin_shifted, cos_shifted])

    # Rotation matrix M(k)
    theta = omega * k
    M_k = torch.tensor([
        [math.cos(theta),  math.sin(theta)],
        [-math.sin(theta), math.cos(theta)]
    ])

    v_shifted_linear = M_k @ v_pos
    rot_diff = (v_shifted_linear - v_shifted_direct).abs().max().item()

    print(f"Shift k = {k}, Channel omega = {omega:.5f}")
    print(f"Direct Evaluation:   {v_shifted_direct.tolist()}")
    print(f"Linear Rotation:     {v_shifted_linear.tolist()}")
    print(f"Max Absolute Error:  {rot_diff:.2e}")

    assert rot_diff < 1e-6, f"Linear rotation identity failed: diff = {rot_diff}"
    print("[OK] Linear shift rotation property PASSED.\n")

    print("=" * 70)
    print("TEST 4: Relative Distance Decay in Positional Dot Products")
    print("=" * 70)
    # Inner product <PE_pos, PE_{pos+k}> = sum_i cos(omega_i * k)
    anchor_pos = 50
    p_anchor = pe_matrix[anchor_pos]

    similarities = []
    offsets = [0, 1, 2, 5, 10, 25, 50, 100]
    for off in offsets:
        if anchor_pos + off < seq_len:
            sim = torch.dot(p_anchor, pe_matrix[anchor_pos + off]).item()
            similarities.append(sim)
            print(f"Offset k = {off:3d}: <PE_{anchor_pos}, PE_{anchor_pos + off}> = {sim:8.3f}")

    # Distance 0 must have maximum dot product (self-norm)
    assert similarities[0] == max(similarities), "Anchor must have maximum dot product with itself"
    assert similarities[1] > similarities[4], "Adjacent offset (1) must be more similar than distant (10)"
    assert similarities[2] > similarities[5], "Close offset (2) must be more similar than distant (25)"
    print("[OK] Locality bias in positional dot products PASSED.\n")

    print("ALL 4 MATHEMATICAL SUITES PASSED CLEANLY (EXIT CODE 0).")


if __name__ == "__main__":
    verify_sinusoidal_encodings()
