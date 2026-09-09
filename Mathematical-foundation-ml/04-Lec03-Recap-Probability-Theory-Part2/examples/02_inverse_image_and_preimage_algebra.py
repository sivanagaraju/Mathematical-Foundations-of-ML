"""
02_inverse_image_and_preimage_algebra.py
========================================
Lecture 03 Simulation: Preimage Algebra & Measurability Laws

Programmatically proves that inverse images (preimages) preserve all Boolean set operations,
which is the foundational theorem guaranteeing that if F is a sigma-algebra on Omega,
the pushforward collection on R forms a valid sigma-algebra.

Mathematical Invariants Verified:
    1. Complementation: X^{-1}(B^c) == Omega \ X^{-1}(B)
    2. Union:           X^{-1}(B_1 U B_2) == X^{-1}(B_1) U X^{-1}(B_2)
    3. Intersection:    X^{-1}(B_1 cap B_2) == X^{-1}(B_1) cap X^{-1}(B_2)
    4. Monotonicity:    B_1 subset B_2 ==> X^{-1}(B_1) subset X^{-1}(B_2)
"""

import sys


def main() -> int:
    print("=== Simulation 02: Preimage Algebra & Measurability Laws ===")

    # 1. Define discrete sample space Omega = {1, 2, ..., 10}
    omega = set(range(1, 11))

    # 2. Define scalar random variable X: Omega -> R
    # X(w) = (w - 5)^2  (parabolic mapping)
    def X(w: int) -> int:
        return (w - 5) ** 2

    # Preimage helper: X^{-1}(B) = {w in Omega : X(w) in B}
    def preimage(B: set) -> set:
        return {w for w in omega if X(w) in B}

    # 3. Define target subsets in Range(X)
    range_values = {X(w) for w in omega}
    print(f"Range space Range(X): {sorted(list(range_values))}")

    B1 = {0, 1, 4}       # Small deviations
    B2 = {4, 9, 16, 25}  # Moderate-to-large deviations

    # Invariant 1: Preimage of Union
    inv_union_direct = preimage(B1 | B2)
    inv_union_parts = preimage(B1) | preimage(B2)
    print(f"X^{{-1}}(B1 U B2): {inv_union_direct}")
    assert inv_union_direct == inv_union_parts, (
        f"Union law violated: {inv_union_direct} != {inv_union_parts}"
    )

    # Invariant 2: Preimage of Intersection
    inv_inter_direct = preimage(B1 & B2)
    inv_inter_parts = preimage(B1) & preimage(B2)
    print(f"X^{{-1}}(B1 cap B2): {inv_inter_direct}")
    assert inv_inter_direct == inv_inter_parts, (
        f"Intersection law violated: {inv_inter_direct} != {inv_inter_parts}"
    )

    # Invariant 3: Preimage of Complement
    # B1 complement with respect to full Range
    B1_comp = range_values - B1
    inv_comp_direct = preimage(B1_comp)
    inv_comp_parts = omega - preimage(B1)
    print(f"X^{{-1}}(B1^c): {inv_comp_direct}")
    assert inv_comp_direct == inv_comp_parts, (
        f"Complement law violated: {inv_comp_direct} != {inv_comp_parts}"
    )

    # Invariant 4: Monotonicity
    B_sub = {0, 1}
    B_super = {0, 1, 4, 9}
    assert preimage(B_sub).issubset(preimage(B_super)), (
        "Preimage monotonicity violated"
    )

    print("All four preimage algebraic preservation laws verified successfully.")
    print("SUCCESS: 02_inverse_image_and_preimage_algebra executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
