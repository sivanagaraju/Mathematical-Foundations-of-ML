# Formulae Sheet — Tutorial 17: Implementation Overview of Diffusion Models

## 1. Forward Process & Jump Equations
- **Single-step transition:**
  $$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t) \mathbf{I})$$
- **Variance parameterization:**
  $$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$
- **Arbitrary-step direct jump (One-shot):**
  $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_0, \quad \epsilon_0 \sim \mathcal{N}(0, \mathbf{I})$$

---

## 2. Tractable Reverse Posterior (Equation 70)
Conditioned on clean sample $x_0$, the reverse distribution is exactly Gaussian:
$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \mu_q(x_t, x_0), \sigma_q^2(t) \mathbf{I})$$
- **Posterior Mean:**
  $$\mu_q(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$
- **Posterior Variance:**
  $$\sigma_q^2(t) = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$

---

## 3. Four Equivalent Parameterizations & Conversions to $\mu_\theta$

| Parameterization | Network Output Target | Loss Objective | Conversion to Parametric Mean $\mu_\theta(x_t, t)$ |
|---|---|---|---|
| **Mean Estimation** | $\mu_q(x_t, x_0)$ | $\|\mu_q(x_t, x_0) - \mu_\theta(x_t, t)\|^2$ | $\mu_\theta(x_t, t) = \text{Output}$ (Eq 93) |
| **Sample Estimation** | Clean data $x_0$ | $\|x_0 - \hat{x}_\theta(x_t, t)\|^2$ | $\mu_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} \hat{x}_\theta + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$ (Eq 94) |
| **Noise Estimation** | Gaussian noise $\epsilon_0$ | $\|\epsilon_0 - \epsilon_\theta(x_t, t)\|^2$ | $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$ (Eq 128) |
| **Score Estimation** | Score $\nabla_{x_t} \log q(x_t)$ | $\|s(x_t) - s_\theta(x_t, t)\|^2$ | $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left( x_t + \beta_t s_\theta(x_t, t) \right)$ (Eq 143) |

---

## 4. Tensor Dimensionality
- Data tensor $x_0$: `(B, C, H, W)`
- Perturbation noise $\epsilon_0$: `(B, C, H, W)`
- Noisy latent $x_t$: `(B, C, H, W)`
- Discrete timestep $t$: `(B,)` (integer indices $\in [1, T]$)
- Schedule scalars $\beta_t, \alpha_t, \bar{\alpha}_t$: shape `(T,)`, indexed and broadcast as `(B, 1, 1, 1)`

---

## 5. Guarantees & Invariants
- **Dimensionality Conservation:** $\text{dim}(x_t) = \text{dim}(x_0)$ across all $t \in [0, T]$.
- **Variance Boundary:** At $t = 1$, $\bar{\alpha}_0 \equiv 1.0 \implies \sigma_q^2(1) = 0$, guaranteeing a deterministic final transition to $x_0$.
- **Prior Convergence:** As $t \to \infty$ or $T \ge 1000$, $\bar{\alpha}_T \approx 0 \implies x_T \sim \mathcal{N}(0, \mathbf{I})$.

---

## 6. Contrastive Decision Table

| Property | Mean Estimation | Clean Sample Estimation | Noise Estimation (DDPM) | Score Estimation |
|---|---|---|---|---|
| **Target Stability** | Time-varying mean | Severe blur at $t \to T$ | Stationary $\mathcal{N}(0, \mathbf{I})$ across all $t$ | Singular as $t \to 0$ ($1/\sigma_t \to \infty$) |
| **Practical Adoption** | Rare | Popular in low-latency distilled models | Universal standard in image diffusion | Prevalent in continuous SDE research |
| **Loss Formulation** | Weighted MSE on $\mu$ | Weighted MSE on $x_0$ | Unweighted simplified MSE $\|\epsilon - \epsilon_\theta\|^2$ | Denoising Score Matching MSE |

---

## 7. Numerical Stability & Traps
- **Precomputing Buffers:** Always precompute $\sqrt{\bar{\alpha}_t}$ and $\sqrt{1 - \bar{\alpha}_t}$ using double precision (`float64`) before casting to `float32` to prevent compounding roundoff drift.
- **Div-by-Zero Protection:** At $t=0$, ensure calculations clamp denominators $(1 - \bar{\alpha}_t)$ or guard indices so that divisions by zero do not inject `NaN` into autograd gradients.
