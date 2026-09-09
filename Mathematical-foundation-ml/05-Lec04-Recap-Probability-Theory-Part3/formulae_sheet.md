# Master Formulae Sheet & Mathematical Invariants: Lec 04 Probability Recap Part 3

> **Package:** 05-Lec04-Recap-Probability-Theory-Part3  
> **Role:** High-density mathematical quick-reference sheet for pushforward measures, cumulative distribution functions, and product geometry.  
> **Audience:** Machine learning engineers and researchers mastering continuous distributions and empirical estimation.

---

## 1. Master Equations Index

### 1.1 The Pushforward Probability Measure

$$
P_X(B) \triangleq P\left(X^{-1}(B)\right) = P\left(\{ \omega \in \Omega : X(\omega) \in B \}\right), \quad \forall B \in \mathcal{B}(\mathbb{R}^d)
$$

> **In words:** The pushforward measure $P_X$ translates probability mass from the abstract domain $(\Omega, \mathcal{F})$ onto the measurable Euclidean range $(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d))$ via preimages.

---

### 1.2 Univariate Cumulative Distribution Function (CDF)

$$
F_X(x) \triangleq P_X((-\infty, x]) = P(X \le x), \quad x \in \mathbb{R}
$$

> **In words:** The Cumulative Distribution Function evaluated at point $x$ equals the total probability mass that the random variable takes values less than or equal to $x$.

---

### 1.3 Multivariate Product Geometry & 2D Rectangle Formula

$$
\begin{aligned}
F_{X_1, X_2}(x_1, x_2) &= P(X_1 \le x_1, X_2 \le x_2) \\
P(a_1 < X_1 \le b_1, a_2 < X_2 \le b_2) &= F(b_1, b_2) - F(a_1, b_2) - F(b_1, a_2) + F(a_1, a_2)
\end{aligned}
$$

> **In words:** The joint CDF accumulates mass in lower-left quadrants of $\mathbb{R}^2$. The probability mass of any axis-aligned rectangle equals the top-right corner minus the two adjacent corners plus the bottom-left corner.

---

### 1.4 Marginal Recovery from Joint CDF

$$
F_{X_1}(x_1) = \lim_{x_2 \to \infty} F_{X_1, X_2}(x_1, x_2)
$$

> **In words:** Taking the limit of the joint CDF as other coordinates approach positive infinity cleanly recovers the marginal CDF of the remaining coordinate.

---

## 2. Input/Output Tensor Dimensionality Table

| Entity / Operator | Mathematical Space | Computational Representation | Description |
| :--- | :--- | :--- | :--- |
| $\mathbf{x} = [x_1, \dots, x_d]^\top$ | $\mathbb{R}^d$ | `torch.FloatTensor[d]` | Continuous coordinate evaluation vector. |
| $F_X(x)$ | $[0, 1] \subset \mathbb{R}$ | `float32` scalar | Cumulative scalar probability score. |
| $F_{\mathbf{X}}(\mathbf{x}_{\text{grid}})$ | $[0, 1]^{G_1 \times \dots \times G_d}$ | `torch.FloatTensor[G1, ..., Gd]` | Joint cumulative mass evaluated across coordinate evaluation grids. |
| $\mathbf{X}_{\text{batch}}$ | $\mathbb{R}^{N \times d}$ | `torch.FloatTensor[B, d]` | Batch matrix of $N$ empirical realizations sampled from $P_X$. |
| $F_N(x)$ (Empirical CDF) | $[0, 1]$ | `torch.FloatTensor[M]` | Non-parametric step function computed over $M$ evaluation thresholds. |

---

## 3. Mathematical Guarantees & Invariants Table

| Invariant / Axiom | Mathematical Formulation | Operational Significance |
| :--- | :--- | :--- |
| **CDF Normalization** | $\lim_{x \to -\infty} F_X(x) = 0 \quad \text{and} \quad \lim_{x \to +\infty} F_X(x) = 1$ | Every valid distribution function begins at 0 and saturates at 1. |
| **Monotonicity** | $x_1 \le x_2 \implies F_X(x_1) \le F_X(x_2)$ | Additional coordinate volume can never reduce cumulative probability. |
| **Right-Continuity** | $\lim_{h \downarrow 0} F_X(x + h) = F_X(x)$ | Ensures CDF values match boundary conditions at point masses. |
| **Non-Negative Rectangles** | $\Delta_{a_1}^{b_1} \Delta_{a_2}^{b_2} F(x_1, x_2) \ge 0$ | A bivariate function is a valid joint CDF only if all rectangle masses are non-negative. |

---

## 4. Contrastive "Why X, Not Y" Decision Table

| Chosen Formulation (X) | Naive Conception (Y) | Why We Choose X over Y (Mathematical Rationale) |
| :--- | :--- | :--- |
| **Cumulative Distribution Function (CDF)** | **Probability Density Function (PDF)** | The CDF always exists for every random variable (discrete, continuous, or mixed); the PDF requires absolute continuity and does not exist for discrete point masses without Dirac deltas. |
| **Pushforward Triplet $(\mathbb{R}^d, \mathcal{B}, P_X)$** | **Abstract Space $(\Omega, \mathcal{F}, P)$** | Algorithms cannot compute on abstract sample spaces $\Omega$; the pushforward triplet moves probability calculus to numerical coordinates where gradient descent operates. |
| **Empirical CDF $F_N$** | **Histogram Density Binning** | Empirical CDF requires zero hyperparameter tuning (no bin-width bias) and converges uniformly by the Glivenko-Cantelli theorem. |

---

## 5. Hardware Realities & Numerical Stability

- **Integral Image Acceleration:** Evaluating 2D rectangle probabilities via $F(b_1, b_2) - F(a_1, b_2) - F(b_1, a_2) + F(a_1, a_2)$ is identical to Viola-Jones integral image lookups, running in $\mathcal{O}(1)$ time regardless of rectangle size.
- **Logit Transforms for Sigmoidal CDFs:** When modeling CDFs with sigmoids $\sigma(x) = \frac{1}{1 + e^{-x}}$, computing $1 - \sigma(x)$ for large $x > 15$ causes catastrophic float32 cancellation. Always evaluate tail probabilities using `torch.special.expit(-x)`.
