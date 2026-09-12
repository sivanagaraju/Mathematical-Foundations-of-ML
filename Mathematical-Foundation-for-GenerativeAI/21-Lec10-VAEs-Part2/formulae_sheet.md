# Formulae & Rapid Revision Sheet: Lec 10 VAEs Part 2

This rapid-revision cheat sheet collects all foundational equations, tensor shape signatures, theoretical guarantees, decision boundaries, and numerical stability recipes for Variational Autoencoders Part 2.

---

## Equations Index

### 1. The Evidence Lower Bound (ELBO)
$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z))$$
$$\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + D_{KL}(q_\phi(z|x) \parallel p_\theta(z|x)) \ge \mathcal{L}(\theta, \phi; x)$$

### 2. Gaussian Reparameterization (LOTUS)
$$z = g_\phi(\epsilon, x) = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_K)$$
$$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{p(\epsilon)} \left[ \nabla_z f(z) \nabla_\phi g_\phi(\epsilon, x) \right]$$

### 3. Closed-Form Gaussian Relative Entropy (KL Divergence)
$$q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2)), \quad p(z) = \mathcal{N}(0, I)$$
$$D_{KL}(q_\phi(z|x) \parallel p(z)) = -\frac{1}{2} \sum_{k=1}^K \left( 1 + \log(\sigma_k^2) - \mu_k^2 - \sigma_k^2 \right)$$

### 4. Likelihood Decoders and Equivalent Losses
$$\text{Gaussian: } p_\theta(x|z) = \mathcal{N}(\hat{x}_\theta(z), I) \implies -\log p_\theta(x|z) = \frac{1}{2}\|x - \hat{x}_\theta(z)\|^2 + \frac{D}{2}\log(2\pi)$$
$$\text{Bernoulli: } p_\theta(x|z) = \prod_{d=1}^D (\hat{x}_d)^{x_d} (1 - \hat{x}_d)^{1 - x_d} \implies -\log p_\theta(x|z) = -\sum_{d=1}^D \left[ x_d \log \hat{x}_d + (1 - x_d) \log(1 - \hat{x}_d) \right]$$

---

## Tensor Dimensionality

| Tensor Name | Mathematical Symbol | PyTorch Shape | Description |
|:---|:---|:---|:---|
| Input Mini-batch | $X$ | `[B, D]` | Batch of $B$ samples with ambient dimension $D$ (e.g., 784 for MNIST) |
| Latent Mean | $\mu_\phi(x)$ | `[B, K]` | Predicted posterior mean in $K$-dimensional latent space |
| Latent Log-Variance | $\log \sigma_\phi^2(x)$ | `[B, K]` | Unconstrained log-variance vector output by recognition encoder |
| Auxiliary Noise | $\epsilon$ | `[B, K]` | Standard normal random draws sampled independently from $\mathcal{N}(0, I)$ |
| Reparameterized Latent | $Z$ | `[B, K]` | Differentiable latent sample passed to generative decoder |
| Reconstruction Mean | $\hat{X}$ | `[B, D]` | Decoded observation parameters (mean pixel intensities) |
| Per-Sample Reconstruction Loss | $\mathcal{L}_{\text{recon}}$ | `[B]` | Mean squared error or binary cross entropy per sample |
| Per-Sample KL Loss | $\mathcal{L}_{KL}$ | `[B]` | Analytical relative entropy per sample summed over $K$ dimensions |

---

## Guarantees & Invariants

| Mathematical Property | Invariant Condition | Practical System Guarantee |
|:---|:---|:---|
| Lower Bound Guarantee | $\mathcal{L}(\theta, \phi; x) \le \log p(x)$ | Maximizing ELBO strictly pushes true marginal data log-likelihood upward |
| Non-Negativity of KL | $D_{KL}(q_\phi \parallel p) \ge 0$ | Regularization loss is bounded below by 0; zero achieved iff $q = p$ |
| Differentiable Path | $\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon$ | Pathwise gradients possess orders-of-magnitude lower variance than score function estimators |
| Information Bottleneck | $I(X; Z) \le H(X)$ | Low latent dimension $K \ll D$ forces compact semantic compression |

---

## Contrastive Decision Table

| Modeling Dimension | Choice X: Gaussian Likelihood (MSE) | Choice Y: Bernoulli Likelihood (BCE) | Strategic Decision Guideline |
|:---|:---|:---|:---|
| Observation Data Domain | Continuous, real-valued images $x \in \mathbb{R}^D$ | Binary or normalized pixel intensities $x \in [0, 1]^D$ | Use Gaussian MSE for unconstrained continuous data; use BCE for normalized pixel rasters |
| Decoder Output Layer | Linear or unconstrained activations | Sigmoid activation $\sigma(a) \in (0, 1)$ | Gaussian decoders predict unbounded coordinates; Bernoulli decoders predict probabilities |
| Loss Gradient Behavior | Linear gradients: $\nabla = \hat{x} - x$ | Steep cross-entropy gradients: $\nabla = \frac{\hat{x} - x}{\hat{x}(1 - \hat{x})}$ (cancels with sigmoid) | BCE avoids saturation plateaus when pixels are near 0 or 1 |
| Sample Blurriness | High (due to mean-squared averaging over modes) | Moderate (preserves edge contrast in binarized domains) | Pair Gaussian decoders with perceptual or adversarial losses to mitigate blur |

---

## Numerical Stability & Traps

1. **Log-Variance vs Direct Standard Deviation:**
   - **Trap:** Never directly output $\sigma$ from an unconstrained linear layer. Negative values will crash square roots and logarithms.
   - **Numerical Fix:** Output unconstrained scalar $s = \log(\sigma^2) \in \mathbb{R}$. Compute standard deviation safely via `std = torch.exp(0.5 * log_var)`.
2. **Log-Variance Clamping:**
   - **Trap:** During early optimization, extreme gradients can cause `log_var` to explode towards $+20$ or $-20$, triggering exponential overflow or underflow (`inf` or `0.0`).
   - **Numerical Fix:** Clamp log-variance to stable numerical intervals: `log_var = torch.clamp(log_var, min=-15.0, max=10.0)`.
3. **BCE Logarithm Underflow:**
   - **Trap:** Computing raw $\log(\hat{x})$ when $\hat{x} \in \{0.0, 1.0\}$ produces `-inf` and `NaN` gradients.
   - **Numerical Fix:** Use PyTorch's numerically fused `F.binary_cross_entropy_with_logits` which operates entirely in stable log-domain using the log-sum-exp trick.
