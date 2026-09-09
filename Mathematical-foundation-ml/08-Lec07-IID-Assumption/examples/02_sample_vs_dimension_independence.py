"""
02_sample_vs_dimension_independence.py
======================================
Lecture 07 Simulation: Sample Independence vs. Coordinate Dimension Correlation

Rigorously demonstrates the single most important distinction in Lecture 07:
    - INDEPENDENCE IS ACROSS SAMPLES: Patient 1 is independent of Patient 2.
    - DIMENSIONS WITHIN A SAMPLE ARE HIGHLY CORRELATED: Pixel 1 is strongly
      correlated with neighboring Pixel 2 in a chest X-ray.

Verification:
    Computes sample-to-sample cross-correlation (converging to 0) vs
    within-sample feature cross-correlation (persisting at high values).
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 02: Sample Independence vs Dimension Correlation ===")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Define 2-pixel correlated image patch: [Pixel_1, Pixel_2]
    # Strong positive covariance: adjacent pixels in tissue share opacity
    mu = torch.tensor([0.5, 0.5])
    rho = 0.88  # High spatial correlation within image
    cov = torch.tensor([[1.0, rho],
                        [rho, 1.0]])

    # 2. Draw N = 50,000 independent patients X_i ~ Normal(mu, cov)
    N = 50000
    L = torch.linalg.cholesky(cov)
    Z = torch.randn(N, 2)
    samples = Z @ L.T + mu  # Shape: [N, 2]

    pixel_1_all = samples[:, 0]
    pixel_2_all = samples[:, 1]

    # 3. Test Intra-Sample Feature Correlation: Corr(Pixel_1, Pixel_2) within same patient
    sample_corr_matrix = torch.corrcoef(samples.T)
    emp_within_corr = sample_corr_matrix[0, 1].item()
    print(f"Intra-Sample Correlation (Pixel 1 vs Pixel 2): {emp_within_corr:.4f} (True: {rho:.4f})")

    assert np.isclose(emp_within_corr, rho, atol=0.02), (
        f"Intra-sample feature correlation should be high: {emp_within_corr} vs {rho}"
    )

    # 4. Test Inter-Sample Cross-Correlation: Corr(Patient_i, Patient_{i+1}) across different patients
    # Compare Pixel 1 of patient i with Pixel 1 of patient (i + 1)
    p1_current = pixel_1_all[:-1]
    p1_next = pixel_1_all[1:]

    stacked_inter = torch.stack([p1_current, p1_next], dim=0)
    inter_corr_matrix = torch.corrcoef(stacked_inter)
    emp_inter_corr = inter_corr_matrix[0, 1].item()

    print(f"Inter-Sample Correlation (Patient i vs Patient i+1): {emp_inter_corr:.6f} (Expected: ~0.0)")
    assert abs(emp_inter_corr) < 0.015, (
        f"Inter-sample correlation must be approximately 0 under IID: {emp_inter_corr}"
    )

    print("Verified: High correlation within image dimensions coexist with total independence across patients.")
    print("SUCCESS: 02_sample_vs_dimension_independence executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
