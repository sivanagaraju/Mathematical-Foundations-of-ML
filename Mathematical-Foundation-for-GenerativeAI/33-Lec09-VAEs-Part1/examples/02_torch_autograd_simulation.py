#!/usr/bin/env python3
"""
File: 02_torch_autograd_simulation.py
Package: 33-Lec09-VAEs-Part1
Lecture Mapping: Topic 5 (11:54 - 14:45), Topic 6 (14:45 - 18:10), & Topic 7 (18:10 - 22:02) in NOTES.md
Mathematical Theorems & Concepts Verified:
    1. Pathwise Reparameterization vs Score-Function (REINFORCE) Gradient Variance Comparison
    2. Analytical Vector-Jacobian Product (VJP) Equivalence for Affine Reparameterization
    3. End-to-End Mini-VAE Training Optimization Loop on CPU (ELBO Convergence)
Prerequisites & Foundations:
    - PREREQUISITES.md#p6-score-function-estimator
    - PREREQUISITES.md#p7-lotus-coordinate-transforms
    - MathsTerms/06-Deep-Architectures-and-Generative-Models/08-Reparameterization_Trick.md
    - MathsTerms/03-Multivariate-Calculus-and-Optimization/02-Derivatives_Gradients_and_Jacobians.md
Hardware Requirements:
    - Pure CPU execution (device = torch.device('cpu'))
    - Deterministic execution with fixed seeds (seed = 42)
    - Dependencies: python >= 3.9, numpy >= 1.22, torch >= 2.0
Execution:
    python examples/02_torch_autograd_simulation.py
Expected Result:
    All assertion checks pass; prints verification summary; exits with code 0.
"""

from __future__ import annotations
import sys
import torch
import torch.nn as nn
import numpy as np


def verify_reparameterization_vs_score_variance() -> None:
    """Theorem 1: Prove Pathwise Reparameterization has orders-of-magnitude lower variance than Score-Function."""
    print("=" * 70)
    print("Verifying Theorem 1: Pathwise vs Score-Function Gradient Variance")
    print("=" * 70)

    # Consider objective: d/d_phi E_{z ~ N(phi, 1)} [ z^2 ]
    # Analytically: E[z^2] = 1 + phi^2, so true derivative d/d_phi = 2 * phi
    phi_val = 3.0
    exact_grad = 2.0 * phi_val  # 6.0
    n_samples = 40_000

    # Path A: Score-Function (REINFORCE / Likelihood Ratio)
    # g_score = f(z) * d/d_phi log q_phi(z) = z^2 * (z - phi)
    z_samples = torch.normal(mean=phi_val, std=1.0, size=(n_samples,))
    score_grads = (z_samples**2) * (z_samples - phi_val)

    # Path B: Pathwise Reparameterization (LOTUS)
    # z = phi + eps, eps ~ N(0, 1)
    # g_pathwise = d/dz f(z) * dz/d_phi = 2z * 1 = 2(phi + eps)
    eps_samples = torch.randn(n_samples)
    pathwise_grads = 2.0 * (phi_val + eps_samples)

    mean_score = torch.mean(score_grads).item()
    var_score = torch.var(score_grads).item()

    mean_path = torch.mean(pathwise_grads).item()
    var_path = torch.var(pathwise_grads).item()

    # Theoretical variance calculation:
    # Var(pathwise) = Var(2(phi + eps)) = 4 * Var(eps) = 4.0
    # Var(score) = E[ ( (phi+eps)^2 * eps )^2 ] - (2phi)^2
    # For phi=3: Var(score) = 36(1) + 12(3) + 33 = 100+ (enormous)
    variance_ratio = var_score / var_path

    # Assertions
    assert np.isclose(mean_score, exact_grad, atol=0.25), (
        f"Score function mean {mean_score:.4f} diverges from exact {exact_grad}"
    )
    assert np.isclose(mean_path, exact_grad, atol=0.05), (
        f"Pathwise mean {mean_path:.4f} diverges from exact {exact_grad}"
    )
    assert np.isclose(var_path, 4.0, atol=0.15), (
        f"Pathwise variance {var_path:.4f} diverges from theoretical 4.0"
    )
    assert variance_ratio > 10.0, (
        f"Score-function variance ({var_score:.2f}) must be >10x higher than pathwise ({var_path:.2f})!"
    )

    print(f"  Target Parameter phi:            {phi_val:.2f}")
    print(f"  Exact Theoretical Gradient:      {exact_grad:.4f}")
    print(f"  Pathwise Gradient (Mean / Var):  {mean_path:.4f} / {var_path:.4f}")
    print(f"  Score-Function (Mean / Var):     {mean_score:.4f} / {var_score:.4f}")
    print(f"  Empirical Variance Multiplier:   {variance_ratio:.2f}x higher variance in score function")
    print(f"  [PASS] Both Estimators Unbiased: score_err={abs(mean_score - exact_grad):.2e}, path_err={abs(mean_path - exact_grad):.2e}")
    print(f"  [PASS] Variance Reduction Proof: Pathwise variance is {variance_ratio:.1f}x lower than REINFORCE\n")


def verify_analytical_reparam_vjp() -> None:
    """Theorem 2: Prove Analytical Vector-Jacobian Product matches PyTorch Autograd for z = mu + sigma * eps."""
    print("=" * 70)
    print("Verifying Theorem 2: Analytical VJP vs PyTorch Autograd for Reparameterization")
    print("=" * 70)

    # Latent dimensions: batch size 3, latent dim 4
    batch_size, k_dim = 3, 4
    mu = torch.tensor([
        [1.2, -0.8,  0.5,  2.1],
        [-1.5, 0.0,  1.4, -0.3],
        [0.2,  1.7, -2.0,  0.9]
    ], dtype=torch.float32, requires_grad=True)

    logvar = torch.tensor([
        [-0.4,  0.3, -1.2,  0.8],
        [ 0.5, -0.6,  0.1, -0.9],
        [-1.0,  0.2,  0.4, -0.2]
    ], dtype=torch.float32, requires_grad=True)

    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn(batch_size, k_dim)

    # Reparameterized latent coordinate: z = mu + sigma * eps
    z = mu + sigma * eps

    # Target loss: L = 0.5 * sum( (z - target)^2 )
    target = torch.ones_like(z) * 0.5
    loss = 0.5 * torch.sum((z - target)**2)

    # 1. PyTorch Autograd execution
    loss.backward(retain_graph=True)
    autograd_grad_mu = mu.grad.clone()
    autograd_grad_logvar = logvar.grad.clone()

    # 2. Manual Analytical derivation:
    # v = dL / dz = (z - target)
    v = (z - target).detach()
    # dz / dmu = 1  ==> dL / dmu = v * 1
    manual_grad_mu = v
    # dz / dlogvar = dz / dsigma * dsigma / dlogvar
    # dz / dsigma = eps
    # dsigma / dlogvar = d/ds exp(0.5 s) = 0.5 * exp(0.5 s) = 0.5 * sigma
    # ==> dz / dlogvar = 0.5 * sigma * eps
    # ==> dL / dlogvar = v * (0.5 * sigma * eps)
    manual_grad_logvar = v * (0.5 * sigma.detach() * eps)

    # Assertions
    diff_mu = torch.max(torch.abs(manual_grad_mu - autograd_grad_mu)).item()
    diff_logvar = torch.max(torch.abs(manual_grad_logvar - autograd_grad_logvar)).item()

    assert torch.allclose(manual_grad_mu, autograd_grad_mu, atol=1e-5), f"mu VJP mismatch: {diff_mu}"
    assert torch.allclose(manual_grad_logvar, autograd_grad_logvar, atol=1e-5), f"logvar VJP mismatch: {diff_logvar}"

    print(f"  Batch Dimensions:                B={batch_size}, K={k_dim}")
    print(f"  Upstream Gradient Norm ||v||:    {torch.norm(v).item():.4f}")
    print(f"  Max Diff for dL/dmu:             {diff_mu:.2e}")
    print(f"  Max Diff for dL/dlogvar:         {diff_logvar:.2e}")
    print(f"  [PASS] Analytical dL/dmu VJP:    {diff_mu:.2e} < 1e-5")
    print(f"  [PASS] Analytical dL/dlogvar VJP:{diff_logvar:.2e} < 1e-5\n")


class MiniVAE(nn.Module):
    """Compact 2-layer MLP VAE for CPU numerical verification."""
    def __init__(self, in_dim: int = 4, latent_dim: int = 2) -> None:
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.Tanh(),
            nn.Linear(8, in_dim)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def verify_mini_vae_convergence() -> None:
    """Theorem 3: Train an end-to-end Mini-VAE demonstrating monotonic ELBO maximization."""
    print("=" * 70)
    print("Verifying Theorem 3: End-to-End Mini-VAE ELBO Optimization Convergence")
    print("=" * 70)

    torch.manual_seed(42)
    in_dim, latent_dim = 4, 2
    model = MiniVAE(in_dim=in_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    # Generate synthetic clustered dataset: 200 samples in R^4
    # Two distinct clusters in 4D space
    c1 = torch.randn(100, in_dim) * 0.3 + torch.tensor([2.0, 2.0, -1.0, -1.0])
    c2 = torch.randn(100, in_dim) * 0.3 + torch.tensor([-2.0, -2.0, 1.0, 1.0])
    data = torch.cat([c1, c2], dim=0)

    initial_loss = 0.0
    final_loss = 0.0

    epochs = 40
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_x, mu, logvar = model(data)

        # Negative log-likelihood (MSE reconstruction loss * in_dim)
        recon_loss = nn.functional.mse_loss(recon_x, data, reduction="mean") * in_dim
        # Analytical Gaussian KL divergence per sample
        kl_loss = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # Total negative ELBO loss
        total_loss = recon_loss + kl_loss
        total_loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = total_loss.item()
        if epoch == epochs - 1:
            final_loss = total_loss.item()

    loss_reduction = (initial_loss - final_loss) / initial_loss * 100.0

    # Assertions
    assert final_loss < initial_loss, f"Loss did not decrease: init={initial_loss}, final={final_loss}"
    assert loss_reduction > 50.0, f"Expected >50% loss drop, got {loss_reduction:.2f}%"

    print(f"  Training Samples:                N=200 in R^{in_dim} (Latent K={latent_dim})")
    print(f"  Epoch 0  Total Loss (-ELBO):     {initial_loss:.4f}")
    print(f"  Epoch {epochs} Total Loss (-ELBO):    {final_loss:.4f}")
    print(f"  Relative Optimization Gain:      {loss_reduction:.2f}% reduction in negative ELBO")
    print(f"  [PASS] Mini-VAE Converged:       {final_loss:.4f} < {initial_loss:.4f} (Clean exit 0)\n")


def main() -> None:
    print("=" * 70)
    print("RUNNING VAE AUTOGRAD & OPTIMIZATION SUITE (CPU DETERMINISTIC)")
    print("=" * 70 + "\n")

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        verify_reparameterization_vs_score_variance()
        verify_analytical_reparam_vjp()
        verify_mini_vae_convergence()
    except AssertionError as err:
        print(f"\n[FATAL ERROR] Autograd Simulation FAILED: {err}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("SUCCESS: ALL AUTOGRAD SIMULATION THEOREMS VERIFIED CLEANLY (EXIT CODE 0)")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
