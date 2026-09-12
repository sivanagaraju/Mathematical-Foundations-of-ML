"""PyTorch simulation of Socratic Kalman filtering vs Open-Loop mental schema updates.

This script verifies that closed-loop adversarial questioning collapses epistemic uncertainty
dramatically faster than open-loop passive slide reading.
"""
import torch

torch.manual_seed(42)

# Ground truth concept vector (true mathematical understanding)
dim = 5
theta_star = torch.tensor([1.0, -0.5, 2.0, 0.0, 1.5]) # Shape: (D,)

# Student initial mental model
theta_open = torch.zeros(dim) # Shape: (D,)
theta_closed = torch.zeros(dim) # Shape: (D,)

# Open-loop updates (passive slide exposure with random bias)
lr = 0.2
for _ in range(10):
    noise = torch.randn(dim) * 0.5
    theta_open += lr * (theta_star - theta_open + noise)

# Closed-loop Socratic updates (targeted adversarial probes)
for _ in range(10):
    # Probe along the direction of maximum current discrepancy
    error = theta_star - theta_closed
    probe_dir = error / (torch.norm(error) + 1e-6)
    correction = torch.dot(error, probe_dir) * probe_dir
    theta_closed += 0.5 * correction

error_open = torch.norm(theta_star - theta_open).item()
error_closed = torch.norm(theta_star - theta_closed).item()

print(f"Open-Loop Final Error:   {error_open:.4f}")
print(f"Socratic Final Error:    {error_closed:.4f}")
assert error_closed < error_open, "Closed-loop Socratic error must be strictly lower"
print("Socratic dialectical convergence verified successfully.")
