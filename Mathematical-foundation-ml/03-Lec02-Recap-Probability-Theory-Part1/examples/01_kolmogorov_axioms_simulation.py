"""
01_kolmogorov_axioms_simulation.py
==================================
Lecture 02 Simulation: Kolmogorov Axioms & Probability Measure Verification

Simulates a formal probability space (Omega, F, P) over a discrete sample space
and programmatically verifies Kolmogorov's foundational axioms and derived theorems.

Mathematical Invariants Verified:
    1. Axiom 1 (Non-negativity): P(A) >= 0 for all events A in F.
    2. Axiom 2 (Unit Measure): P(Omega) == 1.0.
    3. Axiom 3 (Additivity): If A and B are disjoint (A cap B == empty),
       then P(A cup B) == P(A) + P(B).
    4. Complement Theorem: P(A^c) == 1 - P(A).
    5. Monotonicity: If A subset of B, then P(A) <= P(B).

Verification:
    Uses torch.allclose and np.allclose across arbitrary subsets. Exits code 0.
"""

import sys
import numpy as np
import torch


def main() -> int:
    print("=== Simulation 01: Kolmogorov Axioms Verification ===")
    np.random.seed(42)

    # 1. Define discrete sample space Omega = {w1, w2, w3, w4, w5}
    outcomes = ["w1", "w2", "w3", "w4", "w5"]
    n_outcomes = len(outcomes)

    # 2. Assign valid ground probability mass to elementary outcomes
    raw_weights = np.array([0.15, 0.25, 0.20, 0.10, 0.30], dtype=np.float64)
    prob_mass = raw_weights / np.sum(raw_weights)
    prob_dict = {outcomes[i]: prob_mass[i] for i in range(n_outcomes)}

    # Function evaluating P(A) for any event A subset of Omega
    def P(event: set) -> float:
        return sum(prob_dict[w] for w in event)

    # Axiom 2: Normalization P(Omega) == 1.0
    omega_event = set(outcomes)
    p_omega = P(omega_event)
    print(f"P(Omega): {p_omega:.6f}")
    assert np.isclose(p_omega, 1.0, atol=1e-7), f"Axiom 2 violated: P(Omega)={p_omega}"

    # Axiom 1: Non-negativity for all 2^N subsets
    # Generate all subsets of Omega
    subsets = []
    for i in range(1 << n_outcomes):
        subset = {outcomes[j] for j in range(n_outcomes) if (i & (1 << j))}
        subsets.append(subset)

    for A in subsets:
        p_A = P(A)
        assert p_A >= -1e-9, f"Axiom 1 violated: P(A)={p_A} < 0"
        assert p_A <= 1.0 + 1e-9, f"Probability bound violated: P(A)={p_A} > 1"

    print(f"Verified Axiom 1 (Non-negativity) across all {len(subsets)} events.")

    # Axiom 3: Additivity on disjoint events
    # Test pairs of events
    disjoint_checks = 0
    for A in subsets:
        for B in subsets:
            if len(A.intersection(B)) == 0:
                p_union = P(A.union(B))
                p_sum = P(A) + P(B)
                assert np.isclose(p_union, p_sum, atol=1e-7), (
                    f"Axiom 3 violated: P(A U B)={p_union} != P(A)+P(B)={p_sum}"
                )
                disjoint_checks += 1

    print(f"Verified Axiom 3 (Additivity) across {disjoint_checks} disjoint event pairs.")

    # Complement Theorem: P(A^c) == 1 - P(A)
    for A in subsets:
        A_comp = omega_event - A
        p_A = P(A)
        p_comp = P(A_comp)
        assert np.isclose(p_A + p_comp, 1.0, atol=1e-7), (
            f"Complement theorem violated: P(A)={p_A}, P(A^c)={p_comp}"
        )

    # Monotonicity: A subset B ==> P(A) <= P(B)
    for A in subsets:
        for B in subsets:
            if A.issubset(B):
                assert P(A) <= P(B) + 1e-9, (
                    f"Monotonicity violated: {A} subset of {B} but P(A)={P(A)} > P(B)={P(B)}"
                )

    print("Verified Complement Theorem and Monotonicity.")
    print("SUCCESS: 01_kolmogorov_axioms_simulation executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
