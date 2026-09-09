# ⚡ Rapid Revision Formulae Sheet: Kullback-Leibler Divergence

Mathematical core references, dimensional signatures, invariant guarantees, and contrastive decision rubrics for **Lecture 12: KL-Divergence**.

---

## 1. Equations Index

### Discrete Definitions & Cross-Entropy Decomposition
$$D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}$$

$$H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \log Q(x) = \mathbb{E}_{X \sim P}[-\log Q(X)]$$

$$H(P) = -\sum_{x \in \mathcal{X}} P(x) \log P(x)$$

$$D_{KL}(P \parallel Q) = H(P, Q) - H(P)$$

### Continuous Density Formulation
$$D_{KL}(p \parallel q) = \int_{-\infty}^\infty p(x) \log \frac{p(x)}{q(x)}\,dx = \int_{-\infty}^\infty p(x) \log p(x)\,dx - \int_{-\infty}^\infty p(x) \log q(x)\,dx$$

### Univariate Gaussian Closed-Form
$$\text{Let } P = \mathcal{N}(\mu_1, \sigma_1^2), \quad Q = \mathcal{N}(\mu_2, \sigma_2^2)$$

$$D_{KL}(P \parallel Q) = \ln\left(\frac{\sigma_2}{\sigma_1}\right) + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### Multivariate Gaussian Closed-Form
$$\text{Let } P = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1), \quad Q = \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) \quad (\text{in } \mathbb{R}^d)$$

$$D_{KL}(P \parallel Q) = \frac{1}{2}\left[ \text{tr}(\boldsymbol{\Sigma}_2^{-1} \boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \boldsymbol{\Sigma}_2^{-1} (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) - d + \ln\frac{\det \boldsymbol{\Sigma}_2}{\det \boldsymbol{\Sigma}_1} \right]$$

---

## 2. Tensor Dimensionality & Shape Signatures

| Mathematical Object | Symbolic Notation | Dimension / Tensor Shape | Operational Semantic |
| :--- | :--- | :--- | :--- |
| **True Distribution Batch** | $\mathbf{P}$ | `(B, K)` | True discrete target probabilities over $K$ categories |
| **Model Distribution Batch** | $\mathbf{Q}_\theta$ | `(B, K)` | Predicted model probabilities from softmax layer |
| **Batch Cross-Entropy** | $\mathbf{H}(\mathbf{P}, \mathbf{Q})$ | `(B,)` | Per-instance cross-entropy values $-\sum_k P_{ik} \log Q_{ik}$ |
| **Batch Divergence Vector** | $\mathbf{D}_{KL}(\mathbf{P} \parallel \mathbf{Q})$ | `(B,)` | Evaluated KL divergence per batch sample |
| **Scalar Objective** | $\mathcal{L}_{KL}(\theta)$ | `()` | Batch-averaged scalar loss for backpropagation |

---

## 3. Guarantees & Invariants

1. **Gibbs' Inequality (Non-negativity):** $D_{KL}(P \parallel Q) \ge 0$ for all valid distributions $P, Q$. $D_{KL}(P \parallel Q) = 0$ if and only if $P(x) = Q(x)$ almost everywhere.
2. **Asymmetry (Pre-Metric):** In general, $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$. KL divergence is not a distance metric in the strict topological sense (it does not satisfy symmetry or the triangle inequality).
3. **Absolute Continuity Requirement:** $D_{KL}(P \parallel Q) < \infty$ requires $P \ll Q$. If there exists any event $A$ where $P(A) > 0$ but $Q(A) = 0$, $D_{KL}(P \parallel Q) = +\infty$.
4. **Information Monotonicity:** Coarse-graining / grouping outcomes into partitions cannot increase KL divergence (data processing inequality).

---

## 4. Contrastive Decision Table: Forward KL vs Reverse KL

| Property | Forward KL: $D_{KL}(P_{\text{true}} \parallel Q_\theta)$ | Reverse KL: $D_{KL}(Q_\theta \parallel P_{\text{true}})$ | Practical Consequence in ML |
| :--- | :--- | :--- | :--- |
| **Weighting Measure** | Expectation under true distribution $P_{\text{true}}$ | Expectation under model distribution $Q_\theta$ | Forward KL requires samples from data; Reverse KL requires sampling from model |
| **Zero Penalty Condition** | Severe penalty if $Q_\theta(x) \approx 0$ where $P_{\text{true}}(x) > 0$ | Severe penalty if $Q_\theta(x) > 0$ where $P_{\text{true}}(x) \approx 0$ | Forward avoids false negatives; Reverse avoids false positives |
| **Multi-Modal Behavior** | **Mean-seeking / Mode-covering:** $Q_\theta$ stretches to cover all modes | **Mode-seeking / Mode-dropping:** $Q_\theta$ collapses onto a single dominant mode | Variational Autoencoders (VAEs) use forward KL; GAN discriminators / RL use reverse KL |
| **Empirical Evaluability** | Directly estimable from training data: $\frac{1}{N}\sum -\log Q_\theta(x_i)$ | Requires explicit unnormalized density $P_{\text{true}}(x)$ evaluation | Forward KL enables Empirical Risk Minimization (ERM) |

---

## 5. Numerical Stability & Traps

1. **Catastrophic Log of Zero:** If $Q(x) = 0$ and $P(x) > 0$, $\log(P(x) / Q(x))$ evaluates to `inf` or `NaN`. Use PyTorch's `F.kl_div(q_logits.log_softmax(), p_targets)` which operates in log-space.
2. **PyTorch Argument Ordering Convention:** `F.kl_div(input, target)` takes `input` as log-probabilities of $Q$ and `target` as probabilities of $P$, calculating $P \log(P / Q)$. Reversing inputs reverses the mathematical direction.
3. **Batch Reduction Flag:** `F.kl_div` defaults to `reduction='mean'`, which historically divides by $B \times K$ rather than $B$. Always pass `reduction='batchmean'` to adhere to formal mathematical summation over categories and mean over batch.
