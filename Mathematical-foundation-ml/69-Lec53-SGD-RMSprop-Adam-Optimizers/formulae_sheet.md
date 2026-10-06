# Formulae Sheet: Lecture 53 (SGD, RMSprop, Adam: Optimizers)

A rapid revision ledger containing core mathematical equations, tensor shape signatures, analytical invariants, contrastive decision tables, and numerical stability guidelines for First-Order Stochastic Optimization, Heavy Ball Momentum, Coordinate-Wise Scaling, Adam, Hessian Eigenvalue Geometry, and Generalization Dynamics.

---

## 1. Equations Index

### Empirical Risk Minimization (ERM)
$$\hat{R}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f(x_i; \theta), y_i)$$

### Minibatch Stochastic Gradient Estimator
For a random minibatch $B_t \subset \{1, \dots, N\}$ of batch size $|B_t| = B \in \{32, 64\}$:
$$g_t = \nabla \hat{R}_{B_t}(\theta_t) = \frac{1}{B} \sum_{i \in B_t} \nabla_\theta \ell(f(x_i; \theta_t), y_i)$$
$$\mathbb{E}[g_t] = \nabla \hat{R}(\theta_t)$$

### Vanilla Gradient Descent Contraction & Stability Limit
Along the principal eigenvector of Hessian $H$ with maximum eigenvalue $\lambda_{\max}$:
$$\theta_{t+1} = (1 - \alpha \lambda_{\max}) \theta_t \implies |1 - \alpha \lambda_{\max}| < 1 \iff \alpha < \frac{2}{\lambda_{\max}}$$

### SGD with Momentum (Heavy Ball Dynamics)
First raw moment recurrence:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
Unrolled geometric series ($m_0 = \mathbf{0}$):
$$m_t = (1 - \beta_1) \sum_{k=0}^{t-1} \beta_1^k g_{t-k}$$
Parameter update:
$$\theta_{t+1} = \theta_t - \alpha m_t$$
Effective memory horizon:
$$\tau_{\text{eff}} = \frac{1}{1 - \beta_1}$$

### RMSprop (Root Mean Square Normalization)
Second uncentered raw moment recurrence:
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{where } g_t^2 = g_t \odot g_t$$
Parameter update:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} \odot g_t$$

### Physical Dimensional Consistency Ratio
$$[g_t] = \left[\frac{\text{Loss}}{\Theta}\right], \quad [\sqrt{v_t}] = \left[\frac{\text{Loss}}{\Theta}\right] \implies \left[ \frac{g_t}{\sqrt{v_t}} \right] = 1 \text{ (Dimensionless)}$$
$$[\Delta \theta] = [\alpha] \cdot 1 = [\Theta]$$

### The Adam Optimizer (Adaptive Moment Estimation)
1. First moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2. Second moment: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3. First moment bias correction: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
4. Second moment bias correction: $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
5. Parameter update:
   $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

### High-Dimensional Critical Points & Hessian Matrix
Stationarity:
$$\nabla \hat{R}(\theta^*) = \mathbf{0} \in \mathbb{R}^P$$
Hessian second derivative tensor:
$$H \in \mathbb{R}^{P \times P}, \quad H_{j, k} = \frac{\partial^2 \hat{R}}{\partial \theta_j \partial \theta_k}$$
Spectral decomposition:
$$H = Q \Lambda Q^T = \sum_{i=1}^P \lambda_i q_i q_i^T$$
Local minimum criterion:
$$H \succ 0 \iff \lambda_i > 0 \quad \forall i \in \{1, \dots, P\}$$

### Prof. Prathosh's Bernoulli Eigenvalue Theorem
Modeling the sign of each eigenvalue $\operatorname{sign}(\lambda_i)$ as an independent Bernoulli random variable with success probability $p \in (0, 1)$:
$$\mathbb{P}(\text{Local Minimum}) = \prod_{i=1}^P \mathbb{P}(\lambda_i > 0) = p^P$$
$$\lim_{P \to \infty} p^P = 0 \quad \text{for } p < 1$$
Expected negative-curvature escape directions:
$$\mathbb{E}[k_{\text{escape}}] = P(1 - p)$$

### Generalization Dynamics & Checkpoints
Generalization gap:
$$\Delta_{\text{gen}}(\theta_t) = \mathcal{R}_{\text{val}}(\theta_t) - \hat{\mathcal{R}}_{\text{train}}(\theta_t)$$
Optimal early stopping checkpoint:
$$t^* = \arg\min_{t \in \{1, \dots, T\}} \mathcal{R}_{\text{val}}(\theta_t)$$

---

## 2. Tensor Shapes & Dimension Signatures

| Tensor / Symbol | Shape Signature | Physical / Computational Description |
|:----------------|:----------------|:-------------------------------------|
| Parameter Vector $\theta$ | $[P]$ | Flattened model weights ($P \approx 10^6$ to $10^{11}$) |
| Minibatch Input $X_B$ | $[B, D_{\text{in}}]$ | Batch of input vectors with batch size $B \in \{32, 64\}$ |
| Minibatch Gradient $g_t$ | $[P]$ | Instantaneous gradient vector $\nabla_\theta \hat{R}_B(\theta_t)$ |
| First Moment Accumulator $m_t$ | $[P]$ | Velocity buffer matching parameter dimension |
| Second Moment Accumulator $v_t$ | $[P]$ | Squared gradient buffer matching parameter dimension |
| Bias-Corrected Vectors $\hat{m}_t, \hat{v}_t$ | $[P]$ | Unbiased moment estimators |
| Hessian Matrix $H$ | $[P, P]$ | Symmetric second-order curvature matrix |
| Eigenvalues $\Lambda$ | $[P]$ | Curvatures along principal orthogonal axes |
| Checkpoint State Dict $\mathcal{M}$ | Mapping `str -> Tensor` | Serialized parameter weights $\theta^* \in \mathbb{R}^P$ |

---

## 3. Analytical Invariants

1. **Unbiased Expectation Guarantee:** Under stationary gradient distributions, $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t]$ and $\mathbb{E}[\hat{v}_t] = \mathbb{E}[g_t^2]$ for all $t \ge 1$.
2. **Initial Step Unit Normalization:** At step $t=1$, with bias correction, $\frac{\hat{m}_1}{\sqrt{\hat{v}_1}} = \frac{g_1}{\sqrt{g_1^2}} = \operatorname{sign}(g_1)$, enforcing exact step magnitude $\alpha$.
3. **Effective Memory Horizons:**
   - First moment horizon: $\tau_1 = \frac{1}{1 - \beta_1} = \frac{1}{1 - 0.9} = 10$ steps.
   - Second moment horizon: $\tau_2 = \frac{1}{1 - \beta_2} = \frac{1}{1 - 0.999} = 1000$ steps.
4. **Coordinate Invariance:** Updates $\Delta \theta_i$ depend only on coordinate history $g_t^{(i)}$ and remain invariant to gradient scale transformations $\tilde{\mathcal{L}} = c \mathcal{L}$.

---

## 4. Contrastive Decision Table

| Scenario / Landscape Geometry | Recommended Optimizer | Key Hyperparameter Profile | Rationale |
|:---|:---|:---|:---|
| **Well-conditioned convex / Linear regression** | Vanilla SGD | $\alpha \in [10^{-2}, 10^{-1}]$ | Low compute overhead; no moment tensors required. |
| **Deep Convolutional Networks (ResNets)** | SGD with Momentum | $\alpha = 0.1, \beta_1 = 0.9$, Step decay | Smooths directional noise; proven inductive bias for vision generalization. |
| **Recurrent Architectures / RL Policies** | RMSprop | $\alpha = 10^{-3}, \beta_2 = 0.99, \epsilon = 10^{-8}$ | Handles non-stationary and rapidly shifting gradient magnitudes. |
| **Transformers, LLMs, Multimodal Networks** | Adam / AdamW | $\alpha = 10^{-3} \dots 10^{-4}, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ | Normalizes heterogeneous layer scales; escapes high-dimensional saddles. |

---

## 5. Numerical Stability & Production Guidelines

1. **Epsilon Selection:** Use $\epsilon = 10^{-8}$ for FP32/TF32 training. For mixed-precision FP16 or BF16 training, increase to $\epsilon = 10^{-6}$ or $10^{-5}$ to avoid underflow into zero division.
2. **Learning Rate Warm-up:** In large transformers, deploy linear warm-up over the first 2,000–5,000 steps to allow $\hat{v}_t$ to accumulate accurate variance statistics before applying full step size $\alpha$.
3. **Gradient Clipping:** Enforce $\|\mathbf{g}\|_2 \le 1.0$ via `torch.nn.utils.clip_grad_norm_` to protect adaptive accumulators from catastrophic outlier minibatches.
4. **Decoupled Weight Decay (AdamW):** Always use `torch.optim.AdamW` rather than `torch.optim.Adam` when applying weight decay $\lambda$, preventing gradient-dependent regularizer distortion.
