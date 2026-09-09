# ⚡ Rapid Revision Formulae Sheet: Challenges of Machine Learning

Mathematical core references, dimensional signatures, invariant guarantees, and contrastive decision rubrics for **Lecture 10: Challenges of ML**.

---

## 1. Equations Index

### The Canonical Three-Step Recipe
1. **Model Family Choice:**
   $$\mathcal{M} = \{p_\theta(x) \mid \theta \in \Theta\}$$

2. **Distance / Divergence Objective:**
   $$D(p_{\text{true}} \parallel p_\theta) \quad \text{or} \quad \mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\text{true}}}[-\log p_\theta(x)]$$

3. **Optimization / Parameter Recovery:**
   $$\theta^* = \arg\min_{\theta \in \Theta} D(p_{\text{true}} \parallel p_\theta)$$

### Empirical Risk Minimization (ERM)
$$\hat{R}_N(\theta) = \frac{1}{N}\sum_{i=1}^N \ell(x_i; \theta)$$

$$\hat{\theta}_N = \arg\min_{\theta \in \Theta} \hat{R}_N(\theta)$$

### Population Risk & Generalization Bound
$$R(\theta) = \mathbb{E}_{X \sim p_{\text{true}}}[\ell(X; \theta)]$$

$$\lvert \hat{R}_N(\theta) - R(\theta) \rvert \le \mathcal{O}\left(\sqrt{\frac{\text{VC}(\mathcal{M}) + \log(1/\delta)}{N}}\right)$$

### First-Order Gradient Update Rule
$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \hat{R}_N(\theta_t)$$

---

## 2. Tensor Dimensionality & Shape Signatures

| Mathematical Object | Symbolic Notation | Dimension / Tensor Shape | Operational Semantic |
| :--- | :--- | :--- | :--- |
| **Observation Matrix** | $\mathbf{X}$ | `(N, d)` | $N$ i.i.d. observations of dimension $d$ |
| **Parameter Vector** | $\boldsymbol{\theta}$ | `(P,)` | Flattened learnable model parameters across all layers |
| **Batch Loss Vector** | $\boldsymbol{\ell}$ | `(N,)` | Per-sample scalar loss evaluations before sample mean reduction |
| **Scalar Objective** | $\hat{R}_N(\theta)$ | `()` | Scalar average empirical risk value for backward gradient graph |
| **Gradient Tensor** | $\nabla_\theta \hat{R}_N$ | `(P,)` | First-order gradient matching parameter shape for optimizer step |

---

## 3. Guarantees & Invariants

1. **UFA Existence vs Learnability:** Universal Function Approximation guarantees that there *exists* a parameter configuration $\theta^*$ that approximates a target continuous function within $\epsilon$; it does *not* guarantee that gradient descent will find it from random initialization.
2. **Asymptotic Consistency:** As sample size $N \to \infty$, by the Uniform Law of Large Numbers, $\sup_{\theta \in \Theta} |\hat{R}_N(\theta) - R(\theta)| \to 0$ in probability (under bounded loss and finite capacity).
3. **Argmin Invariance:** Shifting the loss function by an additive constant independent of $\theta$ leaves the argmin set invariant: $\arg\min_\theta (\mathcal{L}(\theta) + C) = \arg\min_\theta \mathcal{L}(\theta)$.
4. **Curse of Dimensionality:** Accurately covering a $d$-dimensional continuous space with localized density estimators requires sample size $N = \mathcal{O}(e^d)$ scaling exponentially in $d$.

---

## 4. Contrastive Decision Table: Model Family vs Distance Metric vs Optimization

| Component | Role in Three-Step Recipe | Core Failure Mode | Practical Remediation in Modern ML |
| :--- | :--- | :--- | :--- |
| **Model Family $\{p_\theta\}$** | Expressive hypothesis space | Underfitting / Inductive Bias Deficit | Increase model capacity (depth/width), use transformer/residual architectures |
| **Distance Metric $D(p \parallel q)$** | Quantitative measurement of mismatch | Vanishing gradients / intractable densities | Replace true divergence with sample-evaluable surrogate (NLL, contrastive loss, score matching) |
| **Optimization Algorithm** | Algorithmic parameter traversal | Saddle points, local minima, divergence | Adaptive learning rates (Adam, AdamW), learning rate warmups, weight decay |

---

## 5. Numerical Stability & Traps

1. **Log-Sum-Exp Underflow:** Evaluating continuous likelihoods $\prod_{i=1}^N p_\theta(x_i)$ directly causes catastrophic underflow to floating-point zero. Always optimize the sum of log-densities: $\sum_{i=1}^N \log p_\theta(x_i)$.
2. **Exploding/Vanishing Gradients:** In deep architectures, $\nabla_\theta \mathcal{L}$ can grow exponentially or vanish; always enforce gradient clipping `torch.nn.utils.clip_grad_norm_`.
3. **Variance Parameter Degeneracy:** In Gaussian mixtures and continuous density models, optimizing $\sigma \to 0$ around a training sample drives likelihood to $+\infty$ (spurious singularity). Parameterize log-variance $\log \sigma$ with a lower bound clamp $\sigma \ge \epsilon_{\text{min}}$.
