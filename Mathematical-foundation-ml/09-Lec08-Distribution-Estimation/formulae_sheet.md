# Master Formulae Sheet & Mathematical Invariants: Lec 08 Distribution Estimation

> **Package:** 09-Lec08-Distribution-Estimation  
> **Role:** High-density mathematical quick-reference sheet for distribution estimation, generative modeling, moment matching, and inference queries.  
> **Audience:** Machine learning engineers and researchers mastering density estimation and generative formulations.

---

## 1. Master Equations Index

### 1.1 The Foundational Distribution Estimation Objective

$$
\text{Given } D = \{\mathbf{x}_i\}_{i=1}^N \sim_{\text{iid}} P_{\text{data}}, \quad \text{find } \hat{P} \in \mathcal{P} \quad \text{such that } d(\hat{P}, P_{\text{data}}) \le \epsilon
$$

> **In words:** Given a finite training dataset drawn IID from an unknown ground-truth distribution, find an estimate $\hat{P}$ within hypothesis class $\mathcal{P}$ that minimizes a statistical discrepancy metric $d$.

---

### 1.2 Method of Moments vs Maximum Likelihood

$$
\begin{aligned}
\text{Method of Moments:} \quad & \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i^k = \mathbb{E}_{\mathbf{x} \sim \hat{P}}[\mathbf{x}^k], \quad k = 1, \dots, K \\
\text{Maximum Likelihood:} \quad & \theta^* = \arg\max_\theta \frac{1}{N} \sum_{i=1}^N \log p_\theta(\mathbf{x}_i)
\end{aligned}
$$

> **In words:** Method of moments matches low-order empirical powers to analytical expectations; maximum likelihood minimizes the statistical divergence between empirical data and parametric model $p_\theta$.

---

### 1.3 Downstream Inference Query Triplet

Given estimated joint model $\hat{p}(\mathbf{x}, y)$:

$$
\begin{aligned}
\text{1. Supervised Classification / Regression:} \quad & \hat{p}(y \mid \mathbf{x}) = \frac{\hat{p}(\mathbf{x}, y)}{\int \hat{p}(\mathbf{x}, y') dy'} \\
\text{2. Class-Conditional Generation / Inpainting:} \quad & \hat{p}(\mathbf{x} \mid y) = \frac{\hat{p}(\mathbf{x}, y)}{\hat{p}(y)} \\
\text{3. Unconditional Density / Anomaly Scoring:} \quad & \hat{p}(\mathbf{x}) = \sum_{y} \hat{p}(\mathbf{x}, y)
\end{aligned}
$$

> **In words:** Estimating the full joint probability distribution subsumes supervised classification, generative synthesis, and outlier detection into standard marginalization and conditioning operations.

---

### 1.4 Coordinate Inpainting Decomposition

$$
p(\mathbf{x}_{\text{missing}} \mid \mathbf{x}_{\text{observed}}) = \frac{p(\mathbf{x}_{\text{missing}}, \mathbf{x}_{\text{observed}})}{p(\mathbf{x}_{\text{observed}})} = \frac{p(\mathbf{x})}{\int p(\mathbf{x}_{\text{missing}}', \mathbf{x}_{\text{observed}}) d\mathbf{x}_{\text{missing}}'}
$$

> **In words:** Reconstructing corrupted sensory dimensions is formal Bayesian conditioning of unobserved coordinates on observed coordinates under the estimated joint density.

---

## 2. Input/Output Tensor Dimensionality Table

| Entity / Model | Mathematical Space | PyTorch Shape | Description |
| :--- | :--- | :--- | :--- |
| $\mathbf{x} \in \mathcal{X}$ | $\mathbb{R}^d$ | `[D]` | Multi-dimensional continuous observation vector. |
| $\mathbf{X}_{\text{batch}}$ | $\mathbb{R}^{N \times d}$ | `[B, D]` | Empirical training batch matrix. |
| $\hat{p}_\theta(\mathbf{x})$ | $\mathbb{R}_{\ge 0}$ | `float32` scalar | Estimated probability density score. |
| $\hat{p}(y \mid \mathbf{x})$ | $\Delta^{C-1}$ (Simplex) | `[B, C]` | Calibrated categorical prediction vector over $C$ classes. |
| $\tilde{\mathbf{x}} \sim \hat{P}$ | $\mathbb{R}^d$ | `[B, D]` | Newly generated synthetic samples synthesized from estimated model. |

---

## 3. Mathematical Guarantees & Invariants Table

| Invariant / Property | Mathematical Formalism | Computational Significance |
| :--- | :--- | :--- |
| **Sufficiency of Moments (Gaussian)** | $\mathcal{N}(\mu, \Sigma)$ uniquely determined by moments 1 and 2 | For Gaussian families alone, matching mean and covariance completely identifies the distribution. |
| **Incompleteness of Moments (Mixtures)** | $\exists P \neq Q \text{ such that } \mathbb{E}_P[X^k] = \mathbb{E}_Q[X^k]$ for $k \le 2$ | In multimodal distributions, matching two moments fails to resolve mode separation and internal topology. |
| **Probability Query Distinction** | $P(Y) \neq P(Y \mid X) \neq P(X \mid Y)$ | Conflating marginals, likelihoods, and posteriors leads to fatal model mis-specification. |
| **Bayesian Conditioning Coherence** | $\int_{\mathcal{Y}} p(y \mid \mathbf{x}) dy = 1, \quad \forall \mathbf{x} \in \mathcal{X}$ | Valid conditional distribution estimates must integrate to unity for every possible input conditioning point. |

---

## 4. Contrastive "Why X, Not Y" Decision Table

| Chosen Method (X) | Naive Approach (Y) | Why We Choose X over Y (Mathematical Rationale) |
| :--- | :--- | :--- |
| **Full Distribution Estimation ($p_\theta(x)$)** | **Moment Matching ($\mu, \sigma^2$)** | In multimodal real-world data (e.g. healthy vs diseased), moment matching places peak density in empty valleys between modes where no data exists. |
| **Estimating Joint Distribution $P(X, Y)$** | **Isolated Task-Specific Heuristics** | The joint distribution is a master model that mathematically answers classification, generation, and inpainting simultaneously. |
| **Continuous Density Modeling** | **Discrete PMF Approximation** | Continuous sensory spaces $\mathbb{R}^d$ have uncountably infinite points; discrete histograms suffer exponential cell explosion $\mathcal{O}(B^d)$. |

---

## 5. Hardware Realities & Numerical Stability

- **Log-Determinant Stability in Gaussian Likelihoods:** Evaluating $\log p(\mathbf{x}) = -\frac{1}{2}\log|\boldsymbol{\Sigma}| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ requires computing $\log|\boldsymbol{\Sigma}|$. Never compute determinant $|\boldsymbol{\Sigma}|$ directly (underflows to 0 in high dimensions); compute $2 \sum \log L_{ii}$ from Cholesky factor $\mathbf{L}$.
- **Mode Collapse in Generative Sampling:** In high-dimensional generative networks, numerical instability in sampling causes generator collapse onto a single mode. Enforcing entropy regularization or diffusion noise schedules preserves full distribution coverage.
