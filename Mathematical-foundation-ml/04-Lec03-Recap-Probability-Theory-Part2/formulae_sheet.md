# Master Formulae Sheet & Mathematical Invariants: Lec 03 Probability Recap Part 2

> **Package:** 04-Lec03-Recap-Probability-Theory-Part2  
> **Role:** High-density mathematical quick-reference sheet for random variables, measurable mappings, and preimages.  
> **Audience:** Machine learning engineers and researchers bridging abstract probability spaces to Euclidean coordinate data.

---

## 1. Master Equations Index

### 1.1 The Formal Definition of a Random Variable

$$
X: \Omega \to \mathbb{R}^d, \quad \text{such that } \forall B \in \mathcal{B}(\mathbb{R}^d), \quad X^{-1}(B) \in \mathcal{F}
$$

> **In words:** A random variable is not a variable and not a random number; it is a deterministic measurable mapping from the abstract sample space $\Omega$ to real coordinate space $\mathbb{R}^d$, whose preimages of Borel sets are valid measurable events in $\mathcal{F}$.

---

### 1.2 Inverse Image (Preimage) Formulation

$$
X^{-1}(B) \triangleq \{ \omega \in \Omega : X(\omega) \in B \} \subseteq \Omega
$$

> **In words:** The inverse image of a numerical subset $B \subseteq \mathbb{R}^d$ collects all underlying physical experimental outcomes $\omega$ whose sensor measurement lands inside $B$.

---

### 1.3 The Induced Pushforward Probability Measure

$$
P_X(B) \triangleq P\left( X^{-1}(B) \right) = P\left( \{ \omega \in \Omega : X(\omega) \in B \} \right)
$$

> **In words:** The probability of a numerical condition $B$ on the range space equals the probability measure assigned by $P$ to its preimage event in the original sample space $\Omega$.

---

## 2. Input/Output Tensor Dimensionality Table

| Entity / Function | Mathematical Space | Computational Representation | Semantic Interpretation |
| :--- | :--- | :--- | :--- |
| $\omega$ | $\Omega$ | Raw physical state / index | Abstract outcome of experiment. |
| $X(\omega)$ | $\mathbb{R}$ | `float32` scalar | Real-valued scalar feature measurement. |
| $\mathbf{X}(\omega)$ | $\mathbb{R}^d$ | `torch.FloatTensor[d]` | High-dimensional sensor coordinate vector (e.g. image vector). |
| $B \in \mathcal{B}(\mathbb{R}^d)$ | Borel set in $\mathbb{R}^d$ | Threshold predicate `x <= c` | Measurable geometric region or hyper-rectangle. |
| $X^{-1}(B)$ | Event in $\mathcal{F}$ | `set[Outcome]` | Subset of original sample space. |
| $P_X(B)$ | $[0, 1] \subset \mathbb{R}$ | `float32` scalar | Induced probability assigned to numerical set $B$. |

---

## 3. Mathematical Guarantees & Invariants Table

| Invariant / Property | Mathematical Formalism | Operational Meaning & Guarantees |
| :--- | :--- | :--- |
| **Preimage Union Preserving** | $X^{-1}\left(\bigcup_i B_i\right) = \bigcup_i X^{-1}(B_i)$ | Preimages commute exactly with arbitrary unions of numerical sets. |
| **Preimage Intersection Preserving** | $X^{-1}\left(\bigcap_i B_i\right) = \bigcap_i X^{-1}(B_i)$ | Preimages commute exactly with intersections of numerical sets. |
| **Preimage Complement Preserving** | $X^{-1}(B^c) = \Omega \setminus X^{-1}(B)$ | The preimage of the complement is the complement of the preimage. |
| **Measurability Sufficiency** | $X^{-1}((-\infty, c]) \in \mathcal{F} \quad \forall c \in \mathbb{R}$ | To prove measurability on $\mathbb{R}$, it is sufficient to verify preimages of half-open rays $(-\infty, c]$. |

---

## 4. Contrastive "Why X, Not Y" Decision Table

| Chosen Formulation (X) | Naive Concept (Y) | Why We Choose X over Y (Mathematical Rationale) |
| :--- | :--- | :--- |
| **$X$ as a Deterministic Function** | **$X$ as a Single Random Float** | Outcomes $\omega$ are stochastic; the mapping $X$ is completely deterministic. Conflating the two prevents defining pushforward measures and conditioning. |
| **Measurable Mapping ($X^{-1}(B) \in \mathcal{F}$)** | **Arbitrary Unconstrained Mapping** | If an arbitrary mapping pulls a Borel set back to a non-measurable set in $\Omega$, $P(X \in B)$ is undefined, crashing probability calculus. |
| **Euclidean Range $\mathbb{R}^d$** | **Direct Computing on $\Omega$** | Abstract outcomes (e.g. "patient is infected") cannot be multiplied by weights, inverted in matrices, or differentiated by autograd. |
| **Interval Generators $(-\infty, x]$** | **Individual Points $\{x\}$** | In continuous distributions, $P(X = x) = 0$ for all individual points; cumulative rays $(-\infty, x]$ provide non-trivial measures. |

---

## 5. Hardware Realities & Numerical Stability

- **ADC Quantization & Continuous Approximations:** Physical hardware sensors convert continuous analog signals into finite-precision digital floats (12-bit, 16-bit, or IEEE-754 float32). The theoretical range $\mathbb{R}^d$ is practically mapped to discrete floating-point grids, requiring care when testing boundary inequalities (`<=` vs `<`).
- **Memory Footprint of Vector RVs:** For an image random variable with $P \times Q = 1024 \times 1024$ pixels, a single realization consumes 4 MB of float32 memory. Processing batches of $N=256$ realizations requires over 1 GB of VRAM.
