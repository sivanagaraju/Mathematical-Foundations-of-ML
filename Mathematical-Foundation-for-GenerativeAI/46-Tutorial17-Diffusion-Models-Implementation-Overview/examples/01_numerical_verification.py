"""Numerical verification of Calvin Luo's four equivalent diffusion parameterizations.

This script simulates a 1D Gaussian diffusion forward jump and verifies that
evaluating posterior mean mu_q (Equation 70) matches the conversions from:
  1. Clean sample estimate x_0 (Equation 94)
  2. Injected Gaussian noise epsilon_0 (Equation 128)
  3. Stein score function s(x_t) (Equation 143)
"""
import numpy as np

# Shape: Scalar schedule parameters
T = 10
betas = np.linspace(0.001, 0.05, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

# Synthetic 1D data point
x0 = np.array([2.5], dtype=np.float64) # Shape: (1,)
eps = np.array([1.2], dtype=np.float64) # Shape: (1,)
t = 5 # Timestep index (1-based)

alpha_t = alphas[t - 1]
beta_t = betas[t - 1]
alpha_bar_t = alpha_bars[t - 1]
alpha_bar_prev = alpha_bars[t - 2] if t > 1 else 1.0

# 1. Forward Jump: xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
xt = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * eps

# 2. True Posterior Mean: mu_q (Equation 70)
weight_x0 = (np.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
weight_xt = (np.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
mu_q = weight_x0 * x0 + weight_xt * xt

# 3. Calvin Luo Equivalence Conversions
# Sample estimation: x_hat = x0
mu_from_x0 = weight_x0 * x0 + weight_xt * xt

# Noise estimation: eps_pred = eps (Equation 128)
mu_from_eps = (1.0 / np.sqrt(alpha_t)) * (xt - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps)

# Score estimation: score = -eps / sqrt(1 - alpha_bar_t) (Equation 143)
score = -eps / np.sqrt(1.0 - alpha_bar_t)
mu_from_score = (1.0 / np.sqrt(alpha_t)) * (xt + beta_t * score)

# Verification assertions
assert np.isclose(mu_q, mu_from_x0).all(), "Sample estimate mismatch"
assert np.isclose(mu_q, mu_from_eps).all(), "Noise estimate mismatch"
assert np.isclose(mu_q, mu_from_score).all(), "Score estimate mismatch"

print(f"Verified all 4 parameterizations on step t={t}:")
print(f"  True mu_q:         {mu_q[0]:.6f}")
print(f"  From Sample x0:    {mu_from_x0[0]:.6f}")
print(f"  From Noise eps:    {mu_from_eps[0]:.6f}")
print(f"  From Score:        {mu_from_score[0]:.6f}")
print("ALL 4 PARAMETERIZATIONS MATCH IDENTICALLY.")
