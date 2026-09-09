# Master Formulae Sheet & Mathematical Invariants: Lec 01 Overview of Function Approximation

> **Package:** 02-Lec01-Overview-Function-Approximation  
> **Role:** High-density mathematical and architectural quick-reference sheet for function approximation and vector embeddings.  
> **Audience:** Machine learning engineers, researchers, and students reviewing core mathematical formulations.

---

## 1. Master Equations Index

### 1.1 The Classical Function Approximation Problem

$$
y_i = f(\mathbf{x}_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2), \quad i = 1, \dots, N
$$

> **In words:** An unobserved target mapping $f: \mathcal{X} \to \mathcal{Y}$ generates outputs $y_i$ corrupted by additive observational noise $\epsilon_i$. The learning goal is to estimate $\hat{f} \approx f$ from finite dataset $D = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$.

**Parameter Breakdown:**
- $\mathbf{x}_i \in \mathbb{R}^d$: Input sensory feature vector ($d$-dimensional).
- $y_i \in \mathbb{R}$ or $\{0, 1\}$: Target response or categorical label.
- $\epsilon_i$: Stochastic noise perturbation reflecting measurement error or unmodeled latent factors.
- $\hat{f}(\cdot; \theta)$: Parameterized hypothesis function evaluated with weights $\theta$.

---

### 1.2 Empirical Risk Minimization (Ordinary Least Squares)

$$
\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{f}(\mathbf{x}_i; \theta) \right)^2 = \frac{1}{N} \|\mathbf{y} - \mathbf{X}\theta\|_2^2
$$

> **In words:** The empirical mean squared error loss computes the average squared Euclidean distance between true labels and model predictions. Under linear parameterization $\hat{f}(\mathbf{x}) = \mathbf{x}^\top \theta$, the analytical minimizer is the normal equations solution $\theta^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$.

---

### 1.3 Spatial Image Vector Stacking (Isomorphism)

$$
\mathbf{x} = \text{vec}(\mathbf{I}) \in \mathbb{R}^{P \cdot Q}, \quad \text{where } \mathbf{I} \in \mathbb{R}^{P \times Q}, \quad x_{k} = I_{i, j}, \quad k = i \cdot Q + j
$$

> **In words:** A 2D spatial grid (e.g. chest X-ray image) is mapped bijectively into a 1D Euclidean coordinate vector by stacking rows end-to-end, preserving all inner products and Euclidean norms: $\|\text{vec}(\mathbf{I})\|_2 = \|\mathbf{I}\|_F$.

---

## 2. Input/Output Tensor Dimensionality Table

| Tensor Variable | Mathematical Domain | PyTorch Shape | Description & Interpretation |
| :--- | :--- | :--- | :--- |
| $\mathbf{I}$ | $\mathbb{R}^{P \times Q}$ | `[H, W]` | Raw 2D sensory intensity grid (e.g., medical radiograph image). |
| $\mathbf{x} = \text{vec}(\mathbf{I})$ | $\mathbb{R}^{d}$ ($d = P \cdot Q$) | `[D]` | Flattened high-dimensional feature coordinate vector. |
| $\mathbf{X}_{\text{batch}}$ | $\mathbb{R}^{N \times d}$ | `[B, D]` | Batch design matrix stacking $N$ independent training samples. |
| $\mathbf{W}$ | $\mathbb{R}^{C \times d}$ | `[C, D]` | Linear projection matrix mapping $d$ input features to $C$ output logits. |
| $\mathbf{y}$ | $\mathbb{R}^{N}$ | `[B]` or `[B, 1]` | Ground-truth continuous regression values or target class indices. |
| $\hat{\mathbf{y}}$ | $\mathbb{R}^{N \times C}$ | `[B, C]` | Model output logit tensor before softmax activation. |

---

## 3. Mathematical Guarantees & Invariants Table

| Invariant / Property | Mathematical Statement | Operational Meaning & Boundary Conditions |
| :--- | :--- | :--- |
| **Norm Preservation** | $\|\text{vec}(\mathbf{I})\|_2^2 = \text{Tr}(\mathbf{I}^\top \mathbf{I}) = \|\mathbf{I}\|_F^2$ | Vector stacking is a Hilbert space isometry; Euclidean distances between images are strictly preserved. |
| **Universal Approximation** | $\sup_{\mathbf{x} \in K} |f(\mathbf{x}) - \hat{f}(\mathbf{x}; \theta)| < \epsilon$ | Feedforward neural networks with non-linear activations can approximate any continuous function on compact set $K$ arbitrarily closely. |
| **Table-Lookup Generalization Bound** | $\mathbb{E}_{\mathbf{x} \sim P}[|f(\mathbf{x}) - f_{\text{lookup}}(\mathbf{x})|] = \mathcal{O}(N^{-1/d})$ | Nearest-neighbor memorization error decays with sample size $N$ at a rate that suffers catastrophic curse of dimensionality in high dimensions $d$. |
| **Convexity of Linear MSE** | $\nabla^2_\theta \mathcal{L}_{\text{MSE}}(\theta) = \frac{2}{N}\mathbf{X}^\top \mathbf{X} \succeq 0$ | The Hessian of linear least squares is positive semi-definite, guaranteeing a unique global minimum when $\mathbf{X}$ has full column rank. |

---

## 4. Contrastive "Why X, Not Y" Decision Table

| Chosen Approach (X) | Alternative Considered (Y) | Why We Choose X over Y (Mathematical Rationale) |
| :--- | :--- | :--- |
| **Continuous Function Approximator $\hat{f}(x; \theta)$** | **Lookup Table / Exact Memorization** | Memorization achieves zero training error but fails completely on novel test queries $x \notin D$, suffering $\mathcal{O}(h)$ nearest-neighbor interpolation errors. |
| **Parametric Model Family $\mathcal{H}$** | **Unconstrained Function Search** | Without structural inductive bias (e.g. smoothness, linearity), an unconstrained search over arbitrary functions is ill-posed by the No Free Lunch theorem. |
| **Vector Stacking $\mathbf{x} \in \mathbb{R}^d$** | **Ad-hoc Qualitative Rules** | Coordinate vector representations enable linear algebra, differentiable matrix calculus, and inner product projections. |
| **Probabilistic Modeling $p(y \mid \mathbf{x})$** | **Deterministic Physics Modeling** | High-level diagnostic semantics cannot be solved by first-principle differential equations; statistical distributions rigorously handle noise and label ambiguity. |

---

## 5. Hardware Realities & Numerical Stability

- **Memory Layout & Strides:** When flattening 2D tensors in PyTorch, `x.flatten()` or `x.view(-1)` returns a contiguous view if input memory is C-contiguous. Non-contiguous tensors require `tensor.contiguous().view(-1)` to avoid runtime memory segmentation faults.
- **Normal Equations Inversion:** Directly computing $(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$ suffers numerical instability if condition number $\kappa(\mathbf{X}^\top \mathbf{X}) \gg 1$. Always use QR decomposition or `torch.linalg.lstsq` (SVD-based driver) in production code.
