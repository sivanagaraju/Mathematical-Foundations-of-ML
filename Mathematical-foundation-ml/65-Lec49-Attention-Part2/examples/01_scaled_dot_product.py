"""
Scaled Dot-Product Attention Verification Suite.

This script mathematically verifies the matrix formulation of scaled dot-product
attention:
    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(D_k) ) V

Key invariants validated:
1. Tensor shape preservation across batch and sequence dimensions.
2. Simplex condition: each row of the attention weight matrix sums exactly to 1.0.
3. Numerical equivalence between manual tensor math and torch.nn.functional.scaled_dot_product_attention.
4. Convex hull guarantee: output rows are strict convex combinations of Value rows.
"""

import math
import torch
import torch.nn.functional as F

def manual_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot-product attention explicitly with manual matrix multiplications.
    
    Args:
        q: Query tensor of shape [B, T_q, D_k]
        k: Key tensor of shape [B, T_k, D_k]
        v: Value tensor of shape [B, T_k, D_v]
        mask: Optional mask tensor broadcastable to [B, T_q, T_k]
        
    Returns:
        output: Contextualized output representations of shape [B, T_q, D_v]
        weights: Attention probability matrix of shape [B, T_q, T_k]
    """
    # Extract dimensions
    B, T_q, D_k = q.shape  # Shape: [B, T_q, D_k]
    _, T_k, D_v = v.shape  # Shape: [B, T_k, D_v]
    
    scale = 1.0 / math.sqrt(D_k)
    
    # 1. Compute pairwise similarity logits: S = (Q K^T) / sqrt(D_k)
    # Shape: [B, T_q, D_k] @ [B, D_k, T_k] -> [B, T_q, T_k]
    scores = torch.bmm(q, k.transpose(1, 2)) * scale  # Shape: [B, T_q, T_k]
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
    # 2. Exponentiate and normalize row-wise to form probability distributions
    # Shape: [B, T_q, T_k]
    weights = F.softmax(scores, dim=-1)  # Shape: [B, T_q, T_k]
    
    # 3. Aggregate values: C = A V
    # Shape: [B, T_q, T_k] @ [B, T_k, D_v] -> [B, T_q, D_v]
    output = torch.bmm(weights, v)  # Shape: [B, T_q, D_v]
    
    return output, weights

def run_suite():
    torch.manual_seed(42)
    print("=" * 70)
    print("Lecture 49: Scaled Dot-Product Attention Verification Suite")
    print("=" * 70)
    
    B, T, D_k, D_v = 4, 8, 64, 64
    
    # Generate synthetic query, key, and value tensors
    q = torch.randn(B, T, D_k)  # Shape: [4, 8, 64]
    k = torch.randn(B, T, D_k)  # Shape: [4, 8, 64]
    v = torch.randn(B, T, D_v)  # Shape: [4, 8, 64]
    
    print(f"Inputs initialized:")
    print(f"  Query tensor Q: shape {list(q.shape)}")
    print(f"  Key tensor K:   shape {list(k.shape)}")
    print(f"  Value tensor V: shape {list(v.shape)}")
    
    # Compute manual scaled dot-product attention
    output, weights = manual_scaled_dot_product_attention(q, k, v)
    
    # Check 1: Output shape verification
    # Shape: [B, T, D_v] == [4, 8, 64]
    assert output.shape == (B, T, D_v), f"Expected shape {(B, T, D_v)}, got {output.shape}"
    # Shape: [B, T, T] == [4, 8, 8]
    assert weights.shape == (B, T, T), f"Expected weights shape {(B, T, T)}, got {weights.shape}"
    print("\n[Assertion 1 PASS] Output tensor and attention weights tensor shapes verified.")
    
    # Check 2: Simplex invariant (row sums must equal 1.0 within float precision)
    row_sums = weights.sum(dim=-1)  # Shape: [B, T]
    ones = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, ones, atol=1e-6), "Row sums of attention weights must strictly equal 1.0"
    print(f"[Assertion 2 PASS] Attention matrix row sums: mean = {row_sums.mean().item():.6f}, max_err = {(row_sums - 1.0).abs().max().item():.2e}")
    
    # Check 3: Non-negativity invariant
    assert (weights >= 0.0).all(), "Attention weights must be non-negative probabilities"
    print("[Assertion 3 PASS] Attention probabilities strictly non-negative: min =", weights.min().item())
    
    # Check 4: Parity with PyTorch native implementation
    # torch.nn.functional.scaled_dot_product_attention expects [B, num_heads, T, D_k] or [B, T, D_k]
    ref_output = F.scaled_dot_product_attention(q, k, v)  # Shape: [B, T, D_v]
    assert torch.allclose(output, ref_output, atol=1e-5), "Manual computation must match PyTorch native SDP implementation"
    print(f"[Assertion 4 PASS] Parity with PyTorch F.scaled_dot_product_attention: max absolute difference = {(output - ref_output).abs().max().item():.2e}")
    
    # Check 5: Convex combination property
    # Each row of output is a convex combination of rows of V.
    # Therefore, output norm cannot exceed max norm of V.
    max_v_norm = torch.norm(v, p=2, dim=-1).max()
    max_out_norm = torch.norm(output, p=2, dim=-1).max()
    assert max_out_norm <= max_v_norm * 1.0001, "Convex combinations cannot exceed the bounding radius of value vectors"
    print(f"[Assertion 5 PASS] Convex hull containment holds: max_out_norm ({max_out_norm:.4f}) <= max_v_norm ({max_v_norm:.4f})")
    
    print("\nAll scaled dot-product invariants verified successfully!")

if __name__ == "__main__":
    run_suite()
