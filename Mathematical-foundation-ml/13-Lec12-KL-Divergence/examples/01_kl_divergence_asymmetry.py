#!/usr/bin/env python3
"""
01_kl_divergence_asymmetry.py
=============================
Mathematical simulation demonstrating core properties of Kullback-Leibler (KL) Divergence:
  1. Identity / Gibbs' Inequality: D_KL(P || Q) >= 0 with equality iff P == Q
  2. Asymmetry: D_KL(P || Q) != D_KL(Q || P) (KL is a directed pre-metric, not a metric)
  3. Absolute Continuity: D_KL(P || Q) -> infinity if Q assigns zero mass to any event where P > 0
  4. Cross-Entropy Decomposition: D_KL(P || Q) = H(P, Q) - H(P)

Corresponds to Mathematical Foundations of ML — Lecture 12: KL-Divergence.
Zero external dependencies outside PyTorch and NumPy.
"""

import sys
import numpy as np
import torch


def discrete_kl(p: torch.Tensor, q: torch.Tensor, base: float = 2.0) -> float:
    """Compute D_KL(P || Q) with safe handling of support conditions."""
    # Shape: [K], [K] -> scalar float
    assert torch.isclose(p.sum(), torch.tensor(1.0))
    assert torch.isclose(q.sum(), torch.tensor(1.0))

    # Absolute continuity violation: P > 0 where Q == 0
    if torch.any((p > 0.0) & (q == 0.0)):
        return float("inf")

    mask = p > 0.0
    p_nz = p[mask]
    q_nz = q[mask]

    log_ratio = torch.log(p_nz / q_nz) / np.log(base)
    return float(torch.sum(p_nz * log_ratio).item())


def main() -> int:
    print("=== Simulation 01: KL Divergence Asymmetry & Axiomatic Properties ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Gibbs' Inequality & Self-Divergence D_KL(P || P) = 0
    # -------------------------------------------------------------
    P = torch.tensor([0.70, 0.20, 0.10])
    kl_self = discrete_kl(P, P)
    print(f"[Identity] D_KL(P || P) = {kl_self:.6f} bits")
    assert np.isclose(kl_self, 0.0, atol=1e-6), "Self-divergence must be identically zero"

    # -------------------------------------------------------------
    # 2. Strict Asymmetry: D_KL(P || Q) != D_KL(Q || P)
    # -------------------------------------------------------------
    P_asym = torch.tensor([0.90, 0.08, 0.02])
    Q_asym = torch.tensor([0.20, 0.40, 0.40])

    kl_PQ = discrete_kl(P_asym, Q_asym)
    kl_QP = discrete_kl(Q_asym, P_asym)

    print(f"[Asymmetry] D_KL(P || Q) = {kl_PQ:.4f} bits")
    print(f"[Asymmetry] D_KL(Q || P) = {kl_QP:.4f} bits")
    print(f"[Discrepancy] |D_KL(P||Q) - D_KL(Q||P)| = {abs(kl_PQ - kl_QP):.4f} bits")

    # Assert strict positivity and strong asymmetry
    assert kl_PQ > 0.0, "KL divergence between distinct distributions must be strictly positive"
    assert kl_QP > 0.0, "KL divergence between distinct distributions must be strictly positive"
    assert abs(kl_PQ - kl_QP) > 0.4, (
        f"KL divergence must exhibit significant asymmetry, got discrepancy {abs(kl_PQ - kl_QP)}"
    )

    # -------------------------------------------------------------
    # 3. Cross-Entropy Decomposition: D_KL(P || Q) = H(P, Q) - H(P)
    # -------------------------------------------------------------
    h_P = -float(torch.sum(P_asym * torch.log2(P_asym)).item())
    ce_PQ = -float(torch.sum(P_asym * torch.log2(Q_asym)).item())
    kl_decomposed = ce_PQ - h_P

    print(f"[Decomposition] H(P, Q) = {ce_PQ:.4f} | H(P) = {h_P:.4f} | Diff = {kl_decomposed:.4f}")
    assert np.isclose(kl_PQ, kl_decomposed, atol=1e-5), "D_KL must match H(P, Q) - H(P)"

    # -------------------------------------------------------------
    # 4. Support Condition Violation: Infinite Penalty
    # -------------------------------------------------------------
    # Q_truncated assigns 0 mass to outcome 2 where P > 0
    Q_truncated = torch.tensor([0.80, 0.20, 0.00])
    kl_infinite = discrete_kl(P, Q_truncated)
    print(f"[Support Violation] D_KL(P || Q_truncated) = {kl_infinite}")
    assert kl_infinite == float("inf"), "KL must blow up to infinity when Q has zero support where P > 0"

    print("Verified: KL divergence satisfies Gibbs' inequality, decomposition, and strict asymmetry.")
    print("SUCCESS: 01_kl_divergence_asymmetry executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
