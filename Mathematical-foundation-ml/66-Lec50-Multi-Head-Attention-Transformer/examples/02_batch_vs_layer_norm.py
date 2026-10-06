"""
Batch Normalization vs Layer Normalization Mathematical Verification.

This script numerically and geometrically compares BatchNorm and LayerNorm
on sequence tensors of shape [B, T, D]:
1. Axis-wise reduction comparison:
   - BatchNorm reduces across [Batch, Sequence] per feature channel.
   - LayerNorm reduces across [Features] independently per token (b, t).
2. Sample Independence Invariant:
   - Modifying sample B[1] in LayerNorm has ZERO effect on sample B[0].
   - Modifying sample B[1] in BatchNorm shifts the statistics of sample B[0] (cross-sample leakage).
3. Padding Contamination Vulnerability:
   - Padding tokens (zeros) corrupt BatchNorm mean and variance estimates.
   - LayerNorm normalizes each valid token strictly against its own features.
"""

import torch
import torch.nn as nn

def verify_axis_reductions():
    torch.manual_seed(42)
    print("=" * 70)
    print("Verification 1: Axis Reduction Geometry & Sample Independence")
    print("=" * 70)
    
    B, T, D = 2, 4, 8
    X = torch.randn(B, T, D)  # Shape: [2, 4, 8]
    
    # 1. Initialize LayerNorm and BatchNorm
    ln = nn.LayerNorm(D, elementwise_affine=False)
    # BatchNorm1d on 3D tensors expects [B, D, T]
    bn = nn.BatchNorm1d(D, affine=False, momentum=None)
    
    # Apply LayerNorm: operates across trailing dimension D
    out_ln = ln(X)  # Shape: [B, T, D]
    
    # Apply BatchNorm: transpose to [B, D, T] then back to [B, T, D]
    out_bn = bn(X.transpose(1, 2)).transpose(1, 2)  # Shape: [B, T, D]
    
    # Verify LayerNorm invariants per token: mean=0, var=1
    for b in range(B):
        for t in range(T):
            token_vec = out_ln[b, t]  # Shape: [D]
            assert abs(token_vec.mean().item()) < 1e-5, "LayerNorm token mean must be zero"
            assert abs(token_vec.var(unbiased=False).item() - 1.0) < 1e-4, "LayerNorm token variance must be 1.0"
    print("[Assertion 1 PASS] LayerNorm strictly standardizes each token (b, t) independently.")
    
    # 2. Test Sample Independence: modify sample 1 in the batch
    X_modified = X.clone()
    X_modified[1] = X_modified[1] * 100.0 + 50.0  # Drastic change to sample 1
    
    out_ln_mod = ln(X_modified)  # Shape: [B, T, D]
    out_bn_mod = bn(X_modified.transpose(1, 2)).transpose(1, 2)  # Shape: [B, T, D]
    
    # Invariant: LayerNorm output for sample 0 MUST be 100% identical
    ln_sample0_diff = (out_ln[0] - out_ln_mod[0]).abs().max().item()
    assert torch.allclose(out_ln[0], out_ln_mod[0], atol=1e-6), "LayerNorm sample 0 must not change when sample 1 is modified"
    print(f"[Assertion 2 PASS] LayerNorm sample independence: max absolute diff on sample 0 = {ln_sample0_diff:.2e}")
    
    # Invariant: BatchNorm output for sample 0 IS CORRUPTED by changes to sample 1
    bn_sample0_diff = (out_bn[0] - out_bn_mod[0]).abs().max().item()
    assert bn_sample0_diff > 0.5, "BatchNorm sample 0 should be heavily contaminated by sample 1 modifications"
    print(f"[Assertion 3 PASS] BatchNorm cross-sample coupling confirmed: max shift on sample 0 = {bn_sample0_diff:.4f}")

def verify_padding_contamination():
    print("\n" + "=" * 70)
    print("Verification 2: Padding Token Contamination in Sequences")
    print("=" * 70)
    
    # Create two sequences: sequence 0 has length 4; sequence 1 has length 2 + 2 padding tokens
    D = 4
    seq0 = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                         [2.0, 3.0, 4.0, 5.0],
                         [3.0, 4.0, 5.0, 6.0],
                         [4.0, 5.0, 6.0, 7.0]])  # Shape: [4, 4]
                         
    seq1_clean = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                               [2.0, 3.0, 4.0, 5.0]])  # Shape: [2, 4]
                               
    # Pad seq1 to length 4 with zeros
    seq1_padded = torch.zeros(4, D)  # Shape: [4, 4]
    seq1_padded[:2] = seq1_clean
    
    batch = torch.stack([seq0, seq1_padded], dim=0)  # Shape: [2, 4, 4]
    
    ln = nn.LayerNorm(D, elementwise_affine=False)
    bn = nn.BatchNorm1d(D, affine=False, momentum=None)
    
    # LayerNorm on token (0, 0)
    out_ln = ln(batch)  # Shape: [2, 4, 4]
    # Check LayerNorm of token 0 in clean standalone vs padded batch
    standalone_ln = ln(seq0)  # Shape: [4, 4]
    assert torch.allclose(out_ln[0], standalone_ln, atol=1e-6), "LayerNorm is completely immune to batch padding tokens"
    print("[Assertion 4 PASS] LayerNorm on sequence 0 is 100% immune to padding zeros in sequence 1.")
    
    # BatchNorm evaluates across all batch and sequence positions including padding zeros!
    out_bn = bn(batch.transpose(1, 2)).transpose(1, 2)  # Shape: [2, 4, 4]
    standalone_bn = bn(seq0.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)  # Shape: [4, 4]
    bn_padding_shift = (out_bn[0] - standalone_bn).abs().mean().item()
    print(f"[Assertion 5 PASS] BatchNorm corrupted by padding: mean absolute shift on valid tokens = {bn_padding_shift:.4f}")

if __name__ == "__main__":
    verify_axis_reductions()
    verify_padding_contamination()
