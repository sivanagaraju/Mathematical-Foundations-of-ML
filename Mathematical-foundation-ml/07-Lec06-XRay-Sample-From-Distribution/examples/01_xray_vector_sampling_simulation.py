"""
01_xray_vector_sampling_simulation.py
=====================================
Lecture 06 Simulation: Chest X-Ray as Random Vector Realization

Demonstrates the core physical insight:
1. An image grid (e.g. 8x8) stacked into a coordinate vector x in R^64 is a single
   realization of high-dimensional random variable X(omega).
2. TRAP EXPOSED: Pixel intensity values in [0, 1] are NOT probabilities.
   They are coordinate numbers.
3. Evaluates class-conditional Gaussian likelihoods p(x | Y=0) and p(x | Y=1)
   and computes the true posterior diagnostic probability P(Y=1 | x) via Bayes' Theorem.

Mathematical Invariant:
    P(Y=1 | x) = p(x | Y=1) P(Y=1) / [p(x | Y=0) P(Y=0) + p(x | Y=1) P(Y=1)]
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 01: Chest X-Ray as Random Vector Realization ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Define image grid dimensions: 8x8 -> d = 64
    d = 64
    # Class priors
    p_Y0 = 0.85  # Healthy prevalence
    p_Y1 = 0.15  # Pneumonia prevalence

    # Class-conditional generative parameters
    # Healthy lungs: lower central opacity
    mu_0 = torch.zeros(d) + 0.2
    # Pneumonia: higher opacity in central lung fields (indices 20..44)
    mu_1 = torch.zeros(d) + 0.2
    mu_1[20:45] = 0.75  # Consolidation opacity

    sigma = 0.15

    # 2. Draw a synthetic patient realization with Pneumonia (Y = 1)
    patient_image = mu_1 + sigma * torch.randn(d)
    patient_image = torch.clamp(patient_image, 0.0, 1.0)  # Bound to [0, 1]

    # PROVE THE TRAP: Inspect high intensity pixel
    high_pixel_val = patient_image[25].item()
    print(f"Sample pixel intensity at index 25: {high_pixel_val:.4f}")
    print("CRITICAL TRAP NOTE: This pixel intensity (e.g. 0.78) is a PHYSICAL SENSOR NUMBER, NOT A PROBABILITY!")

    # 3. Compute log-likelihood under both generative hypotheses
    # log p(x | Y) = -0.5 * sum((x - mu)^2 / sigma^2) - (d/2) * log(2*pi*sigma^2)
    const_term = -0.5 * d * np.log(2.0 * np.pi * (sigma**2))
    log_p_x_given_Y0 = -0.5 * torch.sum((patient_image - mu_0)**2) / (sigma**2) + const_term
    log_p_x_given_Y1 = -0.5 * torch.sum((patient_image - mu_1)**2) / (sigma**2) + const_term

    print(f"Log Likelihood log p(x | Healthy):   {log_p_x_given_Y0.item():.2f}")
    print(f"Log Likelihood log p(x | Pneumonia): {log_p_x_given_Y1.item():.2f}")

    # 4. Compute Posterior Diagnostic Probability via Bayes' Rule in log-space
    # log [p(x | Y) * P(Y)]
    log_joint_0 = log_p_x_given_Y0 + np.log(p_Y0)
    log_joint_1 = log_p_x_given_Y1 + np.log(p_Y1)

    # LogSumExp normalization
    max_log = torch.maximum(log_joint_0, log_joint_1)
    log_evidence = max_log + torch.log(torch.exp(log_joint_0 - max_log) + torch.exp(log_joint_1 - max_log))

    posterior_Y1 = torch.exp(log_joint_1 - log_evidence).item()
    posterior_Y0 = torch.exp(log_joint_0 - log_evidence).item()

    print(f"Posterior P(Healthy | x):   {posterior_Y0:.6f}")
    print(f"Posterior P(Pneumonia | x): {posterior_Y1:.6f}")

    # Posterior probabilities must sum to 1
    assert np.isclose(posterior_Y0 + posterior_Y1, 1.0, atol=1e-6), (
        "Posterior probabilities must sum to 1.0"
    )
    # The image was generated from Y=1, so posterior of pneumonia should be high
    assert posterior_Y1 > 0.90, f"Expected strong pneumonia posterior, got {posterior_Y1}"

    print("SUCCESS: 01_xray_vector_sampling_simulation executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
