"""
01_random_variable_mapping.py
=============================
Lecture 03 Simulation: Random Variable as Measurable Numerical Mapping

Demonstrates why engineers require random variables: physical outcomes omega in Omega
are qualitative, while machine learning algorithms require numerical coordinate vectors.
Simulates a vector-valued sensor mapping X: Omega -> R^3 (body temperature, WBC count, O2 saturation)
and computes exact preimages X^{-1}(B) and pushforward probabilities.

Mathematical Invariant:
    P_X(B) = P({omega in Omega : X(omega) in B}) = P(X^{-1}(B))
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 01: Random Variable Measurable Mapping ===")
    np.random.seed(42)

    # 1. Define qualitative sample space Omega (patient clinical states)
    omega = ["healthy", "mild_flu", "severe_pneumonia", "septic_shock"]
    prob_weights = np.array([0.60, 0.25, 0.10, 0.05], dtype=np.float64)
    P_omega = {s: prob_weights[i] for i, s in enumerate(omega)}

    # 2. Define vector-valued random variable X: Omega -> R^3
    # X(omega) = [Temperature (C), WBC Count (10^9/L), O2 Saturation (%)]
    sensor_map = {
        "healthy": torch.tensor([36.6, 6.5, 98.5], dtype=torch.float32),
        "mild_flu": torch.tensor([38.1, 9.2, 96.0], dtype=torch.float32),
        "severe_pneumonia": torch.tensor([39.2, 14.8, 89.0], dtype=torch.float32),
        "septic_shock": torch.tensor([40.1, 22.0, 82.5], dtype=torch.float32),
    }

    # 3. Define target clinical diagnostic set B in R^3:
    # B = {x : Temp >= 38.5 AND O2 <= 92.0}
    def in_target_set_B(x: torch.Tensor) -> bool:
        return bool(x[0] >= 38.5 and x[2] <= 92.0)

    # 4. Compute Preimage X^{-1}(B) = {omega in Omega : X(omega) in B}
    preimage_B = {s for s in omega if in_target_set_B(sensor_map[s])}
    print(f"Target condition: Temp >= 38.5 C and O2 <= 92%")
    print(f"Preimage X^{{-1}}(B) in Omega: {preimage_B}")

    # Expected preimage: severe_pneumonia and septic_shock
    assert preimage_B == {"severe_pneumonia", "septic_shock"}, (
        f"Preimage calculation error, got: {preimage_B}"
    )

    # 5. Pushforward Probability P_X(B) = P(X^{-1}(B))
    prob_P_X_B = sum(P_omega[s] for s in preimage_B)
    expected_prob = P_omega["severe_pneumonia"] + P_omega["septic_shock"]
    print(f"Pushforward Probability P_X(B): {prob_P_X_B:.4f}")

    assert np.isclose(prob_P_X_B, expected_prob, atol=1e-7), (
        f"Pushforward measure must equal preimage probability sum: {prob_P_X_B} vs {expected_prob}"
    )

    # 6. Verify Expectation of Random Variable: E[X] = sum_omega X(omega) P(omega)
    expected_vector = torch.zeros(3, dtype=torch.float32)
    for s in omega:
        expected_vector += sensor_map[s] * float(P_omega[s])

    print(f"Expected vital vector E[X]: {expected_vector.tolist()}")
    assert expected_vector[0] > 36.0 and expected_vector[2] > 90.0

    print("SUCCESS: 01_random_variable_mapping executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
