# Master Formulae Sheet & Mathematical Invariants: Lec 02 Probability Recap Part 1

> **Package:** 03-Lec02-Recap-Probability-Theory-Part1  
> **Role:** High-density mathematical quick-reference sheet for axiomatic probability, Kolmogorov measures, and event algebras.  
> **Audience:** Machine learning engineers and researchers mastering measure-theoretic foundations.

---

## 1. Master Equations Index

### 1.1 Kolmogorov's Axioms of Probability

For a measurable space $(\Omega, \mathcal{F})$, a probability measure $P: \mathcal{F} \to [0, 1]$ satisfies:

$$
\begin{aligned}
\text{Axiom 1 (Non-negativity):} \quad & P(A) \ge 0, \quad \forall A \in \mathcal{F} \\
\text{Axiom 2 (Normalization):} \quad & P(\Omega) = 1 \\
\text{Axiom 3 (Countable Additivity):} \quad & P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i), \quad \text{if } A_i \cap A_j = \emptyset \text{ for } i \neq j
\end{aligned}
$$

> **In words:** Every measurable event receives a non-negative score; the entire universe scores exactly one; and the probability of any countable collection of mutually disjoint events equals the arithmetic sum of their individual probabilities.

---

### 1.2 The Inclusion-Exclusion Principle

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

> **In words:** The probability that either event $A$ or event $B$ occurs equals the sum of their individual probabilities minus the probability of their simultaneous occurrence (preventing double-counting).

---

### 1.3 Continuity of Probability Measures

$$
\begin{aligned}
\text{Monotone Increasing:} \quad & A_1 \subseteq A_2 \subseteq \dots \implies P\left(\lim_{n \to \infty} A_n\right) = \lim_{n \to \infty} P(A_n) \\
\text{Monotone Decreasing:} \quad & B_1 \supseteq B_2 \supseteq \dots \implies P\left(\lim_{n \to \infty} B_n\right) = \lim_{n \to \infty} P(B_n)
\end{aligned}
$$

> **In words:** Probability measures are continuous set functions: taking limits of expanding or contracting event sequences commutes with evaluating probabilities.

---

## 2. Input/Output Tensor Dimensionality Table

| Variable / Construct | Mathematical Domain | Computational Representation | Semantic Meaning |
| :--- | :--- | :--- | :--- |
| $\Omega$ (Sample Space) | Set of outcomes | `set[Outcome]` or `range(N)` | Universal domain of all experimental executions. |
| $A \in \mathcal{F}$ (Event) | Subset of $\Omega$ | `frozenset[int]` | A measurable condition or subset filter. |
| $\mathbb{I}_A(\omega)$ (Indicator) | $\{0, 1\}$ | `torch.BoolTensor[B]` | Boolean activation checking if sample $\omega$ satisfies event $A$. |
| $P(A)$ (Probability) | $[0, 1] \subset \mathbb{R}$ | `float32` scalar | Quantified certainty or likelihood measure. |
| $\mathbf{p} = [P(\omega_1), \dots, P(\omega_K)]$ | $\Delta^{K-1}$ (Simplex) | `torch.FloatTensor[K]` | Discrete probability mass vector summing strictly to 1.0. |

---

## 3. Mathematical Guarantees & Invariants Table

| Invariant / Property | Mathematical Formulation | Physical & Operational Meaning |
| :--- | :--- | :--- |
| **Probability Bounds** | $0 \le P(A) \le 1, \quad \forall A \in \mathcal{F}$ | Probabilities can never be negative or exceed unity. |
| **Empty Set Invariant** | $P(\emptyset) = 0$ | Impossible events carry zero mathematical measure. |
| **Complement Rule** | $P(A^c) = 1 - P(A)$ | The certainty of an event not occurring is one minus its probability. |
| **Monotonicity** | $A \subseteq B \implies P(A) \le P(B)$ | A sub-event can never have higher probability than its enclosing event. |
| **Union Bound (Boole's)** | $P\left(\bigcup_{i=1}^n A_i\right) \le \sum_{i=1}^n P(A_i)$ | Total probability of overlapping events is upper-bounded by their individual sum. |

---

## 4. Contrastive "Why X, Not Y" Decision Table

| Chosen Formulation (X) | Naive Alternative (Y) | Why We Choose X over Y (Mathematical Rationale) |
| :--- | :--- | :--- |
| **$\sigma$-Algebra $\mathcal{F}$** | **Power Set $\mathcal{P}(\Omega)$** | On continuous domains (e.g. $[0, 1]$), Vitali sets prove that assigning probabilities to all subsets leads to fatal contradictions (Banach-Tarski paradox). |
| **Countable Additivity** | **Finite Additivity Only** | Finite additivity cannot evaluate infinite limits, tail distributions, or convergence in probability (e.g. Law of Large Numbers). |
| **Set Measure $P(A)$** | **Pointwise Scores $f(\omega)$** | On continuous spaces, the probability of any single real point is zero ($P(\{x\}) = 0$); probabilities must be defined over measurable intervals. |
| **Axiomatic Triplet $(\Omega, \mathcal{F}, P)$** | **Informal Frequency Heuristics** | Provides a rigorous deductive framework capable of proving convergence guarantees in statistical learning theory. |

---

## 5. Hardware Realities & Numerical Stability

- **Probability Simplex Normalization:** Discrete probability vectors $\mathbf{p}$ must sum to 1. In floating-point arithmetic, accumulated rounding errors cause $\sum p_i \neq 1.0$. Always re-normalize via `p = p / p.sum()` or compute logits directly using `torch.log_softmax`.
- **Underflow in Joint Products:** Computing $P(\bigcap_{i=1}^N A_i) = \prod_{i=1}^N P(A_i)$ for large $N$ leads to immediate float32 underflow (values $< 10^{-45}$ round to zero). Always perform computations in log-space: $\log P = \sum \log P(A_i)$.
