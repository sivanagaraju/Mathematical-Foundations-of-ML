# Master Formulae Sheet & Mathematical Invariants: Lec 06 Chest X-Ray as Sample from Distribution

> **Package:** 07-Lec06-XRay-Sample-From-Distribution  
> **Role:** High-density mathematical quick-reference sheet for high-dimensional image realizations, joint feature-label spaces, and medical sampling.  
> **Audience:** Machine learning engineers and researchers processing high-dimensional sensory data.

---

## 1. Master Equations Index

### 1.1 High-Dimensional Image Coordinate Stacking

$$
\mathbf{x} = \text{vec}(\mathbf{I}) \in \mathbb{R}^d, \quad \text{where } \mathbf{I} \in [0, 1]^{P \times Q}, \quad d = P \cdot Q
$$

> **In words:** An image grid of $P \times Q$ pixels is stacked into a single continuous coordinate vector in $d$-dimensional Euclidean space, where each coordinate $x_j \in [0, 1]$ represents normalized sensor photon transmittance.

---

### 1.2 The Joint Data Probability Measure

$$
P(\mathbf{X} \in B, Y = y) = \int_B p(\mathbf{x} \mid Y = y) \, P(Y = y) \, d\mathbf{x}, \quad B \in \mathcal{B}(\mathbb{R}^d), \quad y \in \{0, 1\}
$$

> **In words:** The joint probability that a patient's radiograph vector lands in geometric region $B$ and has disease status $y$ equals the class-conditional density integrated over $B$ weighted by class prior $P(Y=y)$.

---

### 1.3 Posterior Diagnostic Inference (Bayes' Rule)

$$
P(Y = 1 \mid \mathbf{X} = \mathbf{x}) = \frac{p(\mathbf{x} \mid Y = 1) \, P(Y = 1)}{p(\mathbf{x} \mid Y = 0) \, P(Y = 0) + p(\mathbf{x} \mid Y = 1) \, P(Y = 1)}
$$

> **In words:** The posterior probability that a patient has pneumonia given their observed image vector $\mathbf{x}$ combines the disease-conditional radiographic likelihood with population disease prevalence.

---

### 1.4 Population vs Empirical Dataset Means

$$
\mathbb{E}[\mathbf{X}] = \sum_{y \in \{0,1\}} P(Y = y) \, \mathbb{E}[\mathbf{X} \mid Y = y], \quad \bar{\mathbf{x}}_N = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i \xrightarrow{\text{a.s.}} \mathbb{E}[\mathbf{X}]
$$

> **In words:** By the Law of Total Expectation, the population average X-ray is a weighted mixture of healthy and diseased mean images. By the Law of Large Numbers, the empirical sample mean converges almost surely to this population expectation.

---

## 2. Input/Output Tensor Dimensionality Table

| Construct | Mathematical Domain | PyTorch Shape | Description |
| :--- | :--- | :--- | :--- |
| $\mathbf{I}$ | $[0, 1]^{P \times Q}$ | `[H, W]` | Raw 2D radiographic image. |
| $\mathbf{x} = \text{vec}(\mathbf{I})$ | $\mathbb{R}^d$ ($d=H \cdot W$) | `[D]` | Flattened high-dimensional feature vector. |
| $\mathbf{X}_{\text{batch}}$ | $\mathbb{R}^{N \times d}$ | `[B, D]` | Batch matrix stacking $N$ independent patient images. |
| $\mathbf{y}_{\text{batch}}$ | $\{0, 1\}^N$ | `[B]` | Vector of binary diagnostic indicator labels. |
| $p(\mathbf{x})$ | $\mathbb{R}_{\ge 0}$ | `float32` scalar | Joint image probability density score. |

---

## 3. Mathematical Guarantees & Invariants Table

| Invariant / Property | Mathematical Formalism | Physical & Operational Significance |
| :--- | :--- | :--- |
| **Pixel Intensity Non-Equivalence** | $x_j \in [0, 1] \not\implies x_j = P(E)$ | Pixel intensities are physical photon counts; they do not satisfy Kolmogorov additivity and are not probabilities. |
| **Subspace Manifold Concentration** | $\text{Vol}(\mathcal{M}) \ll \text{Vol}(\mathbb{R}^d)$ | Anatomically valid chest X-rays occupy a tiny, measure-zero non-linear manifold within $\mathbb{R}^d$. |
| **Empirical Consistency** | $\frac{1}{N}\sum_{i=1}^N \mathbb{I}(y_i = 1) \xrightarrow{\text{a.s.}} P(Y=1)$ | Empirical class frequencies converge to true disease prevalence as cohort size $N \to \infty$. |
| **Likelihood Positivity** | $p(\mathbf{x}) \ge 0, \quad \int_{\mathbb{R}^d} p(\mathbf{x}) d\mathbf{x} = 1$ | Continuous probability density integrates to 1 across the infinite volume of $\mathbb{R}^d$. |

---

## 4. Contrastive "Why X, Not Y" Decision Table

| Chosen Formulation (X) | Naive Conception (Y) | Why We Choose X over Y (Mathematical Rationale) |
| :--- | :--- | :--- |
| **X-Ray as Realization of RV $\mathbf{X}(\omega)$** | **X-Ray as a Fixed Constant Matrix** | Treating images as fixed deterministic constants prevents calculating generalization bounds, sampling likelihoods, or Bayesian updates. |
| **Joint Distribution $P(\mathbf{X}, Y)$** | **Only Estimating $f: \mathbf{X} \to Y$** | Pure discriminative fitting misses data corruption, generative synthesis, out-of-distribution detection, and prior shifts. |
| **Probability Density $p(\mathbf{x})$** | **Pixel Magnitude $x_j$** | High intensity $x_j = 0.99$ (dense bone) does not imply 99% probability; likelihood depends on distribution dispersion across all $d$ dimensions. |

---

## 5. Hardware Realities & Numerical Stability

- **Dynamic Range Normalization:** Medical radiograph sensors capture 12-bit (0–4095) or 16-bit integers. Directly feeding unnormalized integer pixels into neural networks causes explosive gradient magnitudes. Always normalize to $[0.0, 1.0]$ or zero-mean unit-variance via $(x - \mu) / \sigma$.
- **Curse of Dimensionality in Likelihoods:** In $d=1024$ dimensions, evaluating Gaussian density $p(\mathbf{x}) = (2\pi)^{-d/2} |\boldsymbol{\Sigma}|^{-1/2} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})}$ causes extreme float32 underflow ($e^{-500} \to 0.0$). Always evaluate densities exclusively in log-space: $\log p(\mathbf{x})$.
