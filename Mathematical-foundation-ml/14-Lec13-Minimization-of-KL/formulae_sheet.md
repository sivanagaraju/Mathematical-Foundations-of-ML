# ⚡ Rapid Revision Formulae Sheet: Minimization of KL Divergence

Mathematical core references, dimensional signatures, invariant guarantees, and contrastive decision rubrics for **Lecture 13: Minimization of KL**.

---

## 1. Equations Index

### Complete Mathematical Derivation of MLE from KL Divergence
1. **Starting Objective (Step 2 of 3-Step Recipe):**
   $$\theta^* = \arg\min_{\theta \in \Theta} D_{KL}(p_{\text{data}} \parallel p_\theta)$$

2. **Expansion via Integral Definition:**
   $$D_{KL}(p_{\text{data}} \parallel p_\theta) = \int p_{\text{data}}(x) \log \frac{p_{\text{data}}(x)}{p_\theta(x)}\,dx$$
   $$= \int p_{\text{data}}(x) \log p_{\text{data}}(x)\,dx - \int p_{\text{data}}(x) \log p_\theta(x)\,dx$$
   $$= -H(p_{\text{data}}) + \mathbb{E}_{X \sim p_{\text{data}}}[-\log p_\theta(X)]$$

3. **Argmin Simplification (Dropping Data Entropy):**
   $$\text{Since } \nabla_\theta H(p_{\text{data}}) \equiv \mathbf{0} \implies \arg\min_{\theta \in \Theta} D_{KL}(p_{\text{data}} \parallel p_\theta) = \arg\min_{\theta \in \Theta} \mathbb{E}_{X \sim p_{\text{data}}}[-\log p_\theta(X)]$$
   $$= \arg\max_{\theta \in \Theta} \mathbb{E}_{X \sim p_{\text{data}}}[\log p_\theta(X)]$$

4. **Empirical Approximation via Law of Large Numbers (LOTUS):**
   $$\mathbb{E}_{X \sim p_{\text{data}}}[\log p_\theta(X)] \approx \frac{1}{N}\sum_{i=1}^N \log p_\theta(x_i)$$

5. **Final Maximum Likelihood Equivalence:**
   $$\theta^*_{\text{min-KL}} \equiv \arg\max_{\theta \in \Theta} \sum_{i=1}^N \log p_\theta(x_i) \equiv \theta^*_{\text{MLE}}$$

---

## 2. Tensor Dimensionality & Shape Signatures

| Mathematical Object | Symbolic Notation | Dimension / Tensor Shape | Operational Semantic |
| :--- | :--- | :--- | :--- |
| **Observed Training Matrix** | $\mathbf{X}$ | `(N, d)` | $N$ i.i.d. continuous data vectors |
| **Model Parameters** | $\boldsymbol{\theta}$ | `(P,)` | Trainable parameters governing density $p_\theta$ |
| **Log-Likelihood Batch Vector** | $\log p_{\boldsymbol{\theta}}(\mathbf{X})$ | `(N,)` | Evaluated log-density per training observation |
| **Average NLL Loss** | $\mathcal{L}_{\text{NLL}}(\boldsymbol{\theta})$ | `()` | Scalar average empirical cross-entropy $-\frac{1}{N}\sum \log p_\theta(x_i)$ |
| **Score Function Gradient** | $\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\mathbf{X})$ | `(N, P)` | Per-sample score vectors whose average drives SGD updates |

---

## 3. Guarantees & Invariants

1. **Argmin Invariance under Additive Shifts:** If $c$ does not depend on $\theta$, then $\arg\min_\theta (f(\theta) + c) \equiv \arg\min_\theta f(\theta)$. This exact property allows removing $-H(p_{\text{data}})$ from the optimization objective.
2. **Consistency of MLE:** Under standard regularity conditions (identifiability, compactness, smoothness), $\hat{\theta}_{\text{MLE}} \xrightarrow{p} \theta^*$ as $N \to \infty$.
3. **Equivalence of Forward KL to MLE:** Minimizing forward KL divergence to the empirical data distribution is mathematically identical to maximizing the likelihood of the training data.
4. **Mode-Covering vs Mode-Seeking Asymmetry:** Forward KL $D_{KL}(p_{\text{data}} \parallel p_\theta)$ penalizes under-estimation of probability where $p_{\text{data}} > 0$ (mode-covering); Reverse KL $D_{KL}(p_\theta \parallel p_{\text{data}})$ penalizes over-estimation of probability where $p_{\text{data}} \approx 0$ (mode-seeking).

---

## 4. Contrastive Decision Table: Forward KL vs Reverse KL vs Maximum Likelihood

| Method / Objective | Optimization Direction | Sampling Requirement | Practical Behavior on Multimodal Targets | ML Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Maximum Likelihood Estimation (MLE)** | $\arg\max_\theta \sum_{i=1}^N \log p_\theta(x_i)$ | Samples from true data $\mathcal{D} \sim p_{\text{data}}$ | Mode-covering (spreads variance across all training clusters) | Supervised learning, language models, autoregressive models |
| **Forward KL Divergence** | $\arg\min_\theta D_{KL}(p_{\text{data}} \parallel p_\theta)$ | Evaluates expectation under $p_{\text{data}}$ | Mean-seeking / Mode-covering (zero-avoiding) | Population-level formalization of MLE |
| **Reverse KL Divergence** | $\arg\min_\theta D_{KL}(p_\theta \parallel p_{\text{data}})$ | Samples from generative model $p_\theta$ | Mode-seeking / Mode-dropping (zero-forcing) | Variational Autoencoders (ELBO), policy gradient RL, distillation |

---

## 5. Numerical Stability & Traps

1. **Product Underflow to Zero:** Never compute $\prod_{i=1}^N p_\theta(x_i)$. For $N=1000$, probabilities in $[0, 1]$ underflow float64 to exact 0. Always calculate the sum of logs: $\sum_{i=1}^N \log p_\theta(x_i)$.
2. **Data Entropy Constant Trap:** Confusing $H(p_{\text{data}})$ with $H(p_\theta)$. While $H(p_{\text{data}})$ is constant with respect to $\theta$ and drops out of forward KL minimization, $H(p_\theta)$ is NOT constant in reverse KL and must be explicitly differentiated.
3. **Spurious Singularities in Continuous MLE:** In mixture models or unconstrained continuous distributions, placing a component directly on a single data point and letting $\sigma \to 0$ causes $p_\theta(x_i) \to \infty$, artificially driving the objective to infinity. Always constrain minimum variance ($\sigma \ge \epsilon_{\text{min}}$).
