"""
01_joint_marginal_conditional_simulation.py
===========================================
Lecture 05 Simulation: Joint, Marginal & Conditional Probability Verification

Simulates a joint probability distribution over symptoms X and clinical diagnosis Y,
verifying marginalization, conditional distributions, and Bayes' Theorem.

Mathematical Invariants Verified:
    1. Joint Normalization: sum_{x, y} P(X=x, Y=y) == 1.0.
    2. Marginalization:     P(X=x) == sum_y P(X=x, Y=y).
    3. Law of Total Prob:   P(Y=y) == sum_x P(Y=y | X=x) * P(X=x).
    4. Bayes' Rule:         P(X=x | Y=y) == P(Y=y | X=x) * P(X=x) / P(Y=y).
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 01: Joint, Marginal & Conditional Distributions ===")
    np.random.seed(42)

    # 1. Define discrete joint distribution matrix P(X=x, Y=y)
    # X: Symptom severity {1, 2, 3, 4, 5}
    # Y: Pneumonia presence {0: Negative, 1: Positive}
    # Joint table of shape [5, 2]
    raw_counts = np.array([
        [0.35, 0.02],  # Severity 1: rarely positive
        [0.25, 0.05],  # Severity 2
        [0.15, 0.08],  # Severity 3
        [0.05, 0.12],  # Severity 4
        [0.01, 0.12],  # Severity 5: mostly positive
    ], dtype=np.float64)

    joint_P = raw_counts / np.sum(raw_counts)

    # Invariant 1: Normalization
    assert np.isclose(np.sum(joint_P), 1.0, atol=1e-7), "Joint distribution must sum to 1.0"
    print(f"Verified Joint Normalization: sum = {np.sum(joint_P):.6f}")

    # Invariant 2: Marginal Distributions
    # Marginal P(X): sum across Y (columns) -> shape [5]
    marginal_X = np.sum(joint_P, axis=1)
    # Marginal P(Y): sum across X (rows) -> shape [2]
    marginal_Y = np.sum(joint_P, axis=0)

    print(f"Marginal P(Y=0) [Healthy]:   {marginal_Y[0]:.4f}")
    print(f"Marginal P(Y=1) [Pneumonia]: {marginal_Y[1]:.4f}")
    assert np.isclose(np.sum(marginal_X), 1.0, atol=1e-7)
    assert np.isclose(np.sum(marginal_Y), 1.0, atol=1e-7)

    # Invariant 3: Conditional Distributions P(Y | X)
    # P(Y=y | X=x) = P(X=x, Y=y) / P(X=x)
    cond_Y_given_X = joint_P / marginal_X[:, np.newaxis]
    # Check each row sums to 1
    assert np.allclose(np.sum(cond_Y_given_X, axis=1), 1.0, atol=1e-7)

    # Invariant 4: Law of Total Probability
    # P(Y=y) = sum_x P(Y=y | X=x) * P(X=x)
    total_prob_Y = np.dot(marginal_X, cond_Y_given_X)
    assert np.allclose(total_prob_Y, marginal_Y, atol=1e-7), (
        f"Total probability violated: {total_prob_Y} vs {marginal_Y}"
    )
    print("Verified Law of Total Probability.")

    # Invariant 5: Bayes' Theorem
    # P(X=x | Y=y) = P(Y=y | X=x) * P(X=x) / P(Y=y)
    cond_X_given_Y_bayes = np.zeros_like(joint_P)
    for x in range(5):
        for y in range(2):
            cond_X_given_Y_bayes[x, y] = (cond_Y_given_X[x, y] * marginal_X[x]) / marginal_Y[y]

    # Direct conditional from joint definition: P(X=x, Y=y) / P(Y=y)
    cond_X_given_Y_direct = joint_P / marginal_Y[np.newaxis, :]

    assert np.allclose(cond_X_given_Y_bayes, cond_X_given_Y_direct, atol=1e-7), (
        "Bayes' rule inversion must match direct conditional from joint distribution"
    )
    print("Verified Bayes' Theorem Inversion.")

    print("SUCCESS: 01_joint_marginal_conditional_simulation executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
