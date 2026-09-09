# Master Formulae Sheet & Mathematical Invariants: Lec 07 IID Assumption

> **Package:** 08-Lec07-IID-Assumption  
> **Role:** High-density mathematical quick-reference sheet for the IID assumption, sample factorization, and empirical risk minimization.  
> **Audience:** Machine learning engineers and researchers deriving generalization bounds and optimization losses.

---

## 1. Master Equations Index

### 1.1 The I.I.D. Joint Factorization Law

$$
p(\mathbf{x}_1, \dots, \mathbf{x}_N) = \prod_{i=1}^N p(\mathbf{x}_i), \quad \text{where } \mathbf{x}_i \sim_{\text{iid}} P_{\text{data}}
$$

> **In words:** Under the assumption that samples are mutually independent and identically distributed, the joint probability density over the entire dataset equals the simple scalar product of individual sample densities.

---

### 1.2 Additive Log-Likelihood Decomposition

$$
\log p(\mathbf{X}_{\text{batch}}) = \log \left( \prod_{i=1}^N p(\mathbf{x}_i) \right) = \sum_{i=1}^N \log p(\mathbf{x}_i)
$$

> **In words:** Taking the natural logarithm converts the mathematically unstable multiplicative product of small probabilities into a numerically well-behaved sum of log-likelihoods.

---

### 1.3 Sample Independence vs Intra-Sample Covariance

$$
\begin{aligned}
\text{Across Samples:} \quad & \text{Cov}(\mathbf{X}_i, \mathbf{X}_j) = \mathbf{0} \in \mathbb{R}^{d \times d}, \quad \forall i \neq j \\
\text{Within Sample:} \quad & \boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}_i) = \mathbb{E}[(\mathbf{X}_i - \boldsymbol{\mu})(\mathbf{X}_i - \boldsymbol{\mu})^\top] \neq \mathbf{0}
\end{aligned}
$$

> **In words:** Distinct patients are statistically uncoupled (zero cross-covariance), but coordinate dimensions (pixels) within any single patient's image are intensely correlated.

---

### 1.4 Empirical Risk as Monte Carlo Approximation

$$
\mathcal{R}_{\text{emp}}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(\mathbf{x}_i), y_i) \xrightarrow[N \to \infty]{\text{a.s.}} \mathbb{E}_{(\mathbf{X}, Y) \sim P}[\ell(f_\theta(\mathbf{X}), Y)] = \mathcal{R}_{\text{true}}(\theta)
$$

> **In words:** The empirical loss averaged over an IID training set converges almost surely to the true expected generalization risk by the Strong Law of Large Numbers.

---

## 2. Input/Output Tensor Dimensionality Table

| Construct / Variable | Mathematical Space | PyTorch Shape | Description |
| :--- | :--- | :--- | :--- |
| $\mathbf{X}_{\text{batch}}$ | $\mathbb{R}^{N \times d}$ | `[B, D]` | Batch tensor of $N$ independent data vectors. |
| $\log p(\mathbf{x}_i)$ | $\mathbb{R}$ | `[B]` | Vector of individual sample log-densities. |
| $\sum_i \log p(\mathbf{x}_i)$ | $\mathbb{R}$ | `float32` scalar | Aggregated batch total log-likelihood. |
| $\boldsymbol{\Sigma}_{\text{intra}}$ | $\mathbb{R}^{d \times d}$ | `[D, D]` | Non-zero feature covariance matrix within each sample. |
| $\mathbf{C}_{\text{inter}}$ | $\mathbb{R}^{N \times N}$ | `[B, B]` | Identity-diagonal sample correlation matrix ($C_{ij} \approx 0$ for $i \neq j$). |

---

## 3. Mathematical Guarantees & Invariants Table

| Invariant / Theorem | Mathematical Formulation | Operational Consequence |
| :--- | :--- | :--- |
| **Sum of Variances (IID)** | $\text{Var}\left(\frac{1}{N}\sum_{i=1}^N X_i\right) = \frac{\sigma^2}{N}$ | The variance of the empirical mean estimator shrinks as $\mathcal{O}(1/N)$ under independence. |
| **Failure under Positive Correlation** | $\text{Var}(\bar{X}) = \frac{\sigma^2}{N} + \frac{N-1}{N}\rho \sigma^2 \to \rho \sigma^2 > 0$ | If training samples are correlated ($\rho > 0$), increasing dataset size $N$ fails to eliminate estimation variance. |
| **PAC Learnability Guarantee** | $P(\mathcal{R}_{\text{true}}(\hat{\theta}) - \mathcal{R}_{\text{emp}}(\hat{\theta}) > \epsilon) \le 2M e^{-2N\epsilon^2}$ | Finite training error generalises to test error only when both distributions match identically. |
| **Linearity of Expectation** | $\mathbb{E}\left[\sum_{i=1}^N \log p(\mathbf{x}_i)\right] = N \mathbb{E}[\log p(\mathbf{x})]$ | Sample average loss is a strictly unbiased estimator of expected population loss. |

---

## 4. Contrastive "Why X, Not Y" Decision Table

| Chosen Concept (X) | Misconception (Y) | Why We Choose X over Y (Mathematical Rationale) |
| :--- | :--- | :--- |
| **Independence Across Samples ($X_i \perp X_j$)** | **Independence Across Pixels ($X_{i,a} \perp X_{i,b}$)** | Pixels inside a meaningful image are heavily correlated; assuming pixel independence reduces image models to uninformative white noise static. |
| **Sum of Log-Likelihoods ($\sum \log p_i$)** | **Product of Direct Likelihoods ($\prod p_i$)** | Multiplying $N=1000$ probabilities causes catastrophic underflow to zero in IEEE float32; log sums remain stable and differentiable. |
| **Identically Distributed Assumption** | **Training on Out-of-Distribution Data** | When training and deployment distributions diverge ($P_{train} \neq P_{test}$), empirical risk minimization offers zero theoretical generalization guarantees. |

---

## 5. Hardware Realities & Numerical Stability

- **DataLoader Worker Seeds & Shuffling:** Multi-process data loaders (`num_workers > 1`) must seed each worker uniquely. If workers share identical PRNG seeds, duplicate batches will be yielded, destroying the IID independence assumption and causing gradient overfitting.
- **Log-Likelihood Underflow Protection:** Individual continuous densities can exceed 1 or dip to $10^{-30}$. Always evaluate densities in log-space (`dist.log_prob(x)`) rather than computing `torch.log(dist.prob(x))`.
