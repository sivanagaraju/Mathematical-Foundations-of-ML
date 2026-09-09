# Master Formulae Sheet & Mathematical Invariants: Lec 05 Probability Recap Part 2

> **Package:** 06-Lec05-Recap-Probability-Theory-Part2  
> **Role:** High-density mathematical quick-reference sheet for joint distributions, marginalization, conditioning, and Bayes' Theorem.  
> **Audience:** Machine learning engineers and researchers modeling complex multi-variable systems.

---

## 1. Master Equations Index

### 1.1 The Sum Rule (Marginalization)

$$
\begin{aligned}
\text{Discrete:} \quad & P_X(x) = \sum_{y \in \mathcal{Y}} P_{X,Y}(x, y) \\
\text{Continuous:} \quad & p_X(x) = \int_{-\infty}^{\infty} p_{X,Y}(x, y) \, dy
\end{aligned}
$$

> **In words:** The marginal probability of random variable $X$ is obtained by accumulating (summing or integrating) the joint probability across all possible realizations of the other random variable $Y$.

---

### 1.2 The Product Rule & Conditional Probability

$$
P(X = x, Y = y) = P(Y = y \mid X = x) \cdot P(X = x) = P(X = x \mid Y = y) \cdot P(Y = y)
$$

> **In words:** The joint probability of two variables equals the conditional probability of the second given the first, multiplied by the marginal probability of the first.

---

### 1.3 Bayes' Theorem

$$
P(Y = y \mid X = x) = \frac{P(X = x \mid Y = y) \, P(Y = y)}{P(X = x)} = \frac{P(X = x \mid Y = y) \, P(Y = y)}{\sum_{y'} P(X = x \mid Y = y') \, P(Y = y')}
$$

> **In words:** The posterior probability of cause $Y$ given observed evidence $X$ equals the likelihood of evidence given cause times prior cause probability, normalized by total marginal evidence.

---

### 1.4 Analytical Bivariate Normal Conditioning

For joint Gaussian $\begin{bmatrix} X \\ Y \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mu_X \\ \mu_Y \end{bmatrix}, \begin{bmatrix} \sigma_X^2 & \rho \sigma_X \sigma_Y \\ \rho \sigma_X \sigma_Y & \sigma_Y^2 \end{bmatrix} \right)$:

$$
\begin{aligned}
Y \mid X = x &\sim \mathcal{N}\left( \mu_{Y \mid X}, \sigma_{Y \mid X}^2 \right) \\
\mu_{Y \mid X} &= \mu_Y + \rho \frac{\sigma_Y}{\sigma_X}(x - \mu_X) \\
\sigma_{Y \mid X}^2 &= \sigma_Y^2 (1 - \rho^2)
\end{aligned}
$$

> **In words:** Conditioning a bivariate Gaussian on a specific coordinate slice yields an exact univariate Gaussian whose mean shifts linearly with correlation $\rho$ and whose variance contracts by factor $(1 - \rho^2)$.

---

## 2. Input/Output Tensor Dimensionality Table

| Construct / Variable | Mathematical Domain | PyTorch Shape | Description |
| :--- | :--- | :--- | :--- |
| $\mathbf{X}_{\text{joint}}$ | $\mathbb{R}^{d_X + d_Y}$ | `[B, D_X + D_Y]` | Concatenated feature and target vector. |
| $\mathbf{P}_{X,Y}$ (Joint Table) | $\Delta^{K_X \times K_Y - 1}$ | `[K_X, K_Y]` | Discrete 2D joint probability mass matrix summing to 1. |
| $\mathbf{P}_{Y \mid X}$ | $[0, 1]^{K_X \times K_Y}$ | `[K_X, K_Y]` | Row-stochastic transition matrix where each row sums to 1. |
| $\mathbf{P}_X$ (Marginal) | $\Delta^{K_X - 1}$ | `[K_X]` | Marginal probability vector obtained by summing columns. |
| $\boldsymbol{\Sigma}$ (Covariance) | $\mathbb{S}_{++}^{d \times d}$ | `[D, D]` | Symmetric positive-definite covariance matrix. |

---

## 3. Mathematical Guarantees & Invariants Table

| Invariant / Property | Mathematical Formalism | Physical & Computational Significance |
| :--- | :--- | :--- |
| **Variance Reduction under Conditioning** | $\text{Var}(Y \mid X) \le \text{Var}(Y)$ (in expectation) | Observing features $X$ can only decrease or maintain uncertainty about target $Y$. |
| **Statistical Independence** | $X \perp Y \iff P(X, Y) = P(X)P(Y)$ | When independent, conditional distribution collapses to marginal: $P(Y \mid X) = P(Y)$. |
| **Joint Preimage Intersection** | $\{\mathbf{X} \in B_1 \times B_2\} = X_1^{-1}(B_1) \cap X_2^{-1}(B_2)$ | Multi-variable events are exact intersections of single-variable cylinder sets in $\Omega$. |
| **Law of Total Variance** | $\text{Var}(Y) = \mathbb{E}[\text{Var}(Y \mid X)] + \text{Var}(\mathbb{E}[Y \mid X])$ | Total variance decomposes into unexplained within-slice variance plus explained across-slice variance. |

---

## 4. Contrastive "Why X, Not Y" Decision Table

| Chosen Formulation (X) | Naive Concept (Y) | Why We Choose X over Y (Mathematical Rationale) |
| :--- | :--- | :--- |
| **Full Joint Distribution $P(X, Y)$** | **Independent Marginal Product $P(X)P(Y)$** | Marginal products discard all correlation and statistical coupling, making classification or regression from $X$ to $Y$ mathematically impossible. |
| **Vector RV $\mathbf{X}: \Omega \to \mathbb{R}^d$** | **Isolated Scalar RVs** | Treating components as a single vector RV preserves linear transformation properties, matrix operations, and autograd gradients. |
| **Bayes' Rule Inversion** | **Assuming $P(Y \mid X) \approx P(X \mid Y)$** | The "Prosecutor's Fallacy": confusing likelihood $P(X \mid Y)$ with posterior $P(Y \mid X)$ ignores prior class imbalance $P(Y)$, causing massive misclassification. |

---

## 5. Hardware Realities & Numerical Stability

- **Log-Sum-Exp for Marginalization:** Computing continuous or high-cardinality marginals $P(X) = \sum_y P(X, y) = \sum_y e^{\log P(X, y)}$ directly leads to underflow. Always use the LogSumExp trick: $\log P(X) = m + \log \sum_y e^{\log P(X, y) - m}$ where $m = \max_y \log P(X, y)$.
- **Covariance Inversion & Cholesky:** Conditioning high-dimensional Gaussians requires inverting covariance submatrices. Never compute $\boldsymbol{\Sigma}^{-1}$ explicitly; always solve linear systems via Cholesky factor `torch.linalg.solve_triangular`.
