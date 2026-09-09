"""
02_sigma_algebra_closure_verification.py
========================================
Lecture 02 Simulation: Sigma-Algebra Closure & Generation Verification

Simulates the construction of a mathematical sigma-algebra F over finite sample space Omega.
Verifies the three formal axioms of a sigma-algebra:
    1. Omega in F and empty_set in F.
    2. Closure under complementation: A in F ==> (Omega \ A) in F.
    3. Closure under countable/finite unions: A, B in F ==> (A U B) in F.
    4. Closure under intersections (via De Morgan's laws): A, B in F ==> (A cap B) in F.

Verification:
    Validates generated sigma-algebra against power set properties and asserts clean termination.
"""

import sys
from typing import Set, FrozenSet


def is_valid_sigma_algebra(F: Set[FrozenSet[int]], omega: FrozenSet[int]) -> bool:
    """Checks whether collection of sets F forms a valid sigma-algebra over omega."""
    # Axiom 1: Empty set and Omega must be present
    if frozenset() not in F or omega not in F:
        return False

    # Axiom 2: Closure under complement
    for A in F:
        A_comp = omega - A
        if A_comp not in F:
            return False

    # Axiom 3: Closure under union
    for A in F:
        for B in F:
            if (A | B) not in F:
                return False

    return True


def generate_sigma_algebra(generators: list[FrozenSet[int]], omega: FrozenSet[int]) -> Set[FrozenSet[int]]:
    """Iteratively computes the smallest sigma-algebra containing the generator sets."""
    F: Set[FrozenSet[int]] = set(generators)
    F.add(frozenset())
    F.add(omega)

    changed = True
    while changed:
        old_size = len(F)
        # Add complements
        complements = {omega - A for A in F}
        F.update(complements)

        # Add pairwise unions
        unions = {A | B for A in F for B in F}
        F.update(unions)

        changed = len(F) > old_size

    return F


def main() -> int:
    print("=== Simulation 02: Sigma-Algebra Closure Verification ===")
    omega = frozenset([1, 2, 3, 4])

    # 1. Test trivial sigma-algebra: F_trivial = {empty_set, Omega}
    F_trivial = {frozenset(), omega}
    assert is_valid_sigma_algebra(F_trivial, omega), "Trivial sigma-algebra must be valid"
    print(f"Trivial sigma-algebra verified (size={len(F_trivial)}).")

    # 2. Test an INVALID collection: F_invalid = {empty_set, Omega, {1}} (missing {2,3,4})
    F_invalid = {frozenset(), omega, frozenset([1])}
    assert not is_valid_sigma_algebra(F_invalid, omega), "Collection missing complement must be invalid"
    print("Correctly rejected invalid collection missing complement.")

    # 3. Generate minimal sigma-algebra from generator G = {{1, 2}}
    G = [frozenset([1, 2])]
    F_gen = generate_sigma_algebra(G, omega)
    assert is_valid_sigma_algebra(F_gen, omega), "Generated collection must be a valid sigma-algebra"
    # Should contain: empty_set, {1,2}, {3,4}, {1,2,3,4} -> exactly 4 elements
    assert len(F_gen) == 4, f"Expected 4 elements, got {len(F_gen)}: {F_gen}"
    print(f"Generated sigma-algebra from {{1,2}} verified: size={len(F_gen)}.")

    # 4. Generate from non-trivial intersecting generators: G2 = {{1,2}, {2,3}}
    G2 = [frozenset([1, 2]), frozenset([2, 3])]
    F_gen2 = generate_sigma_algebra(G2, omega)
    assert is_valid_sigma_algebra(F_gen2, omega), "Generated collection from G2 must be valid"
    # Atoms are {1}, {2}, {3}, {4} -> generated sigma-algebra is the full power set of size 2^4 = 16
    assert len(F_gen2) == 16, f"Expected full power set of size 16, got {len(F_gen2)}"
    print(f"Generated sigma-algebra from {{1,2}}, {{2,3}} verified: full power set size={len(F_gen2)}.")

    # 5. Verify De Morgan's Law: (A U B)^c == A^c cap B^c
    A = frozenset([1, 2])
    B = frozenset([2, 3])
    lhs = omega - (A | B)
    rhs = (omega - A) & (omega - B)
    assert lhs == rhs, "De Morgan's law must hold"
    print("De Morgan's law identity verified.")

    print("SUCCESS: 02_sigma_algebra_closure_verification executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
