#!/usr/bin/env python3
"""
02_forward_vs_reverse_kl.py
===========================
Mathematical simulation comparing Forward KL vs Reverse KL Optimization:
  Target: Bimodal Gaussian Mixture P(x) = 0.5 * N(-2.5, 0.6^2) + 0.5 * N(+2.5, 0.6^2)
  Model:  Unimodal Gaussian Q_theta(x) = N(mu, sigma^2)

Demonstrates:
  1. Forward KL: D_KL(P || Q_theta) -> Mode-Covering / Mean-Seeking (mu ~ 0.0, wide sigma ~ 2.57)
  2. Reverse KL: D_KL(Q_theta || P) -> Mode-Seeking / Zero-Forcing (mu ~ +2.5 or -2.5, sharp sigma ~ 0.6)
  3. Explains why MLE/supervised learning uses Forward KL, while VAEs/distillation utilize Reverse KL.

Corresponds to Mathematical Foundations of ML — Lecture 13: Minimization of KL.
Zero external dependencies outside PyTorch and NumPy.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def mixture_log_prob(x: torch.Tensor, m1: float = -2.5, m2: float = 2.5, s: float = 0.6) -> torch.Tensor:
    """Log probability density of 50-50 bimodal Gaussian mixture."""
    # Shape: [N] -> [N]
    comp1 = (1.0 / (s * np.sqrt(2.0 * np.pi))) * torch.exp(-0.5 * ((x - m1) / s) ** 2)
    comp2 = (1.0 / (s * np.sqrt(2.0 * np.pi))) * torch.exp(-0.5 * ((x - m2) / s) ** 2)
    p = 0.5 * comp1 + 0.5 * comp2
    return torch.log(p + 1e-30)


def main() -> int:
    print("=== Simulation 02: Forward KL vs Reverse KL Divergence Optimization ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Forward KL: min_theta D_KL(P || Q_theta)
    #    Equivalent to E_{x ~ P}[ -log Q_theta(x) ]
    # -------------------------------------------------------------
    # Draw N samples from target mixture P
    N = 30000
    components = torch.randint(0, 2, (N,))
    samples_P = torch.zeros(N)
    samples_P[components == 0] = -2.5 + 0.6 * torch.randn((components == 0).sum())
    samples_P[components == 1] = 2.5 + 0.6 * torch.randn((components == 1).sum())

    mu_fwd = nn.Parameter(torch.tensor(0.5))
    log_sig_fwd = nn.Parameter(torch.tensor(0.0))
    opt_fwd = optim.Adam([mu_fwd, log_sig_fwd], lr=0.08)

    for _ in range(300):
        opt_fwd.zero_grad()
        sig = torch.exp(log_sig_fwd)
        loss = 0.5 * np.log(2.0 * np.pi) + log_sig_fwd + 0.5 * torch.mean(((samples_P - mu_fwd) / sig) ** 2)
        loss.backward()
        opt_fwd.step()

    fwd_mu_val = mu_fwd.item()
    fwd_sig_val = torch.exp(log_sig_fwd).item()
    print(f"[Forward KL - Mode Covering] Mean: {fwd_mu_val:.4f} (Expect ~0.0) | Sigma: {fwd_sig_val:.4f} (Expect ~2.57)")

    # -------------------------------------------------------------
    # 2. Reverse KL: min_theta D_KL(Q_theta || P)
    #    E_{x ~ Q_theta}[ log Q_theta(x) - log P(x) ]
    #    Use reparameterization trick: x = mu + sig * eps, eps ~ N(0, 1)
    # -------------------------------------------------------------
    # Initialize near one mode (e.g. at 2.0)
    mu_rev = nn.Parameter(torch.tensor(2.0))
    log_sig_rev = nn.Parameter(torch.tensor(0.0))
    opt_rev = optim.Adam([mu_rev, log_sig_rev], lr=0.05)

    for _ in range(400):
        opt_rev.zero_grad()
        sig = torch.exp(log_sig_rev)
        # Reparameterized sampling from Q_theta
        eps = torch.randn(1000)
        x_q = mu_rev + sig * eps

        # log Q(x_q) = -0.5*log(2pi) - log_sig - 0.5*eps^2
        log_q = -0.5 * np.log(2.0 * np.pi) - log_sig_rev - 0.5 * (eps ** 2)
        log_p = mixture_log_prob(x_q, m1=-2.5, m2=2.5, s=0.6)

        rev_kl_loss = torch.mean(log_q - log_p)
        rev_kl_loss.backward()
        opt_rev.step()

    rev_mu_val = mu_rev.item()
    rev_sig_val = torch.exp(log_sig_rev).item()
    print(f"[Reverse KL - Mode Seeking]  Mean: {rev_mu_val:.4f} (Expect ~2.5) | Sigma: {rev_sig_val:.4f} (Expect ~0.60)")

    # -------------------------------------------------------------
    # 3. Contrastive Mathematical Verification
    # -------------------------------------------------------------
    # Forward KL covers the entire spread: mu near 0, sigma wide
    assert abs(fwd_mu_val) < 0.15, f"Forward KL must center near 0, got {fwd_mu_val}"
    assert fwd_sig_val > 2.4, f"Forward KL must inflate variance to cover both modes, got {fwd_sig_val}"

    # Reverse KL collapses onto a single mode: mu near +2.5, sigma narrow (~0.6)
    assert abs(rev_mu_val - 2.5) < 0.25, f"Reverse KL must isolate the mode at 2.5, got {rev_mu_val}"
    assert abs(rev_sig_val - 0.6) < 0.15, f"Reverse KL must match the mode width ~0.6, got {rev_sig_val}"

    print("Verified: Forward KL exhibits mode-covering; Reverse KL exhibits mode-seeking.")
    print("SUCCESS: 02_forward_vs_reverse_kl executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
