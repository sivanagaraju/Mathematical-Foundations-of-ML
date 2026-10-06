"""
Multi-Head Attention (MHA) Mathematical Verification Suite.

This script mathematically verifies the multi-head attention formulation:
    MHA(Q, K, V) = Concat(head_1, ..., head_M) W^O
    where head_j = Attention(Q W_j^Q, K W_j^K, V W_j^V)

Key invariants validated:
1. Subspace decomposition: D-dimensional space splits into M heads of size D_k = D / M.
2. Tensor shape flow: [B, T, D] -> [B, M, T, D_k] -> [B, M, T, T] -> [B, T, D].
3. Simplex invariant: each head's attention matrix independently satisfies row sums = 1.0.
4. Output projection W^O restores nominal embedding dimension D.
5. Numerical parity with torch.nn.MultiheadAttention under identical weights.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # D_k = D / M
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Fused linear projections for Query, Key, and Value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Output projection matrix W^O
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input shape: [B, T, D]
        B, T, D = x.shape  # Shape: [B, T, D]
        
        # 1. Project to Query, Key, and Value
        q = self.q_proj(x)  # Shape: [B, T, D]
        k = self.k_proj(x)  # Shape: [B, T, D]
        v = self.v_proj(x)  # Shape: [B, T, D]
        
        # 2. Reshape and transpose to expose head dimension: [B, T, M, D_k] -> [B, M, T, D_k]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: [B, M, T, D_k]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: [B, M, T, D_k]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: [B, M, T, D_k]
        
        # 3. Scaled Dot-Product Attention per head
        # Shape: [B, M, T, D_k] @ [B, M, D_k, T] -> [B, M, T, T]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # Shape: [B, M, T, T]
        attn_weights = F.softmax(scores, dim=-1)  # Shape: [B, M, T, T]
        
        # Context vectors per head: [B, M, T, T] @ [B, M, T, D_k] -> [B, M, T, D_k]
        head_outputs = torch.matmul(attn_weights, v)  # Shape: [B, M, T, D_k]
        
        # 4. Concatenate heads back to [B, T, D]
        # [B, M, T, D_k] -> [B, T, M, D_k] -> [B, T, D]
        concat_outputs = head_outputs.transpose(1, 2).contiguous().view(B, T, D)  # Shape: [B, T, D]
        
        # 5. Output linear projection W^O
        output = self.out_proj(concat_outputs)  # Shape: [B, T, D]
        
        return output, attn_weights

def run_suite():
    torch.manual_seed(42)
    print("=" * 70)
    print("Lecture 50: Multi-Head Attention Verification Suite")
    print("=" * 70)
    
    B, T, D, M = 2, 8, 64, 8
    D_k = D // M  # 8
    
    x = torch.randn(B, T, D)  # Shape: [2, 8, 64]
    mha = CustomMultiHeadAttention(embed_dim=D, num_heads=M)
    
    output, attn_weights = mha(x)
    
    # Check 1: Output tensor shape
    assert output.shape == (B, T, D), f"Expected shape {(B, T, D)}, got {output.shape}"
    assert attn_weights.shape == (B, M, T, T), f"Expected weights shape {(B, M, T, T)}, got {attn_weights.shape}"
    print(f"[Assertion 1 PASS] Output shape [B, T, D] = {list(output.shape)} verified.")
    print(f"                  Attention weights shape [B, M, T, T] = {list(attn_weights.shape)} verified.")
    
    # Check 2: Simplex invariant across every individual head
    # Each row of [B, M, T, T] along dim=-1 must sum to 1.0
    row_sums = attn_weights.sum(dim=-1)  # Shape: [B, M, T]
    ones = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, ones, atol=1e-6), "Every attention head must strictly form a row probability simplex"
    print(f"[Assertion 2 PASS] Simplex invariant holds across all {M} heads: max row error = {(row_sums - 1.0).abs().max().item():.2e}")
    
    # Check 3: FLOP and parameter equivalence with PyTorch nn.MultiheadAttention
    py_mha = nn.MultiheadAttention(embed_dim=D, num_heads=M, bias=False, batch_first=True)
    
    # Copy weights to ensure numerical comparison parity
    with torch.no_grad():
        py_mha.in_proj_weight.copy_(torch.cat([mha.q_proj.weight, mha.k_proj.weight, mha.v_proj.weight], dim=0))
        py_mha.out_proj.weight.copy_(mha.out_proj.weight)
        
    py_output, py_weights = py_mha(x, x, x, need_weights=True, average_attn_weights=False)
    assert torch.allclose(output, py_output, atol=1e-5), "Custom MHA must numerically match PyTorch nn.MultiheadAttention"
    assert torch.allclose(attn_weights, py_weights, atol=1e-5), "Attention weights must match PyTorch native weights"
    print(f"[Assertion 3 PASS] Parity with PyTorch nn.MultiheadAttention: max diff = {(output - py_output).abs().max().item():.2e}")
    
    # Check 4: Diversity of attention patterns across heads
    # Different heads should attend to different tokens (not identical attention maps)
    head_1 = attn_weights[0, 0]  # Shape: [T, T]
    head_2 = attn_weights[0, 1]  # Shape: [T, T]
    head_diff = (head_1 - head_2).abs().sum().item()
    assert head_diff > 0.1, "Different heads must produce distinct attention distributions"
    print(f"[Assertion 4 PASS] Multi-head specialization confirmed: Head 0 vs Head 1 divergence = {head_diff:.4f}")
    
    print("\nAll Multi-Head Attention mathematical invariants verified successfully!")

if __name__ == "__main__":
    run_suite()
