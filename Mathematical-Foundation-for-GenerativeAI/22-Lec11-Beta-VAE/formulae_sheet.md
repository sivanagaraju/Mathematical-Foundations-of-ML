# Formulae & Rapid Revision Sheet: Lec 11 Beta- VAE

This cheat sheet compiles key mathematical formulations, tensor dimensions, theoretical guarantees, decision criteria, and numerical stability guidelines for Beta-VAE.

---

## Equations Index

### 1. The Beta-VAE Objective Function
$$\mathcal{L}_\beta(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta D_{KL}(q_\phi(z|x) \parallel p(z))$$
$$\text{Minimization Loss: } \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}}(x, \hat{x}) + \beta \cdot \mathcal{L}_{KL}(\mu, \sigma^2)$$

### 2. Constrained Optimization Formulation (Karush-Kuhn-Tucker)
$$\max_{\phi, \theta} \mathbb{E}_{x \sim p_{\text{data}}}\left[ \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] \right] \quad \text{subject to } D_{KL}(q_\phi(z|x) \parallel p(z)) \le \epsilon$$
$$\text{Lagrangian: } \mathcal{F}(\theta, \phi, \beta) = \mathbb{E}[\log p_\theta(x|z)] - \beta \left( D_{KL}(q_\phi(z|x) \parallel p(z)) - \epsilon \right)$$

### 3. Total Correlation Decomposition of Aggregate Relative Entropy
$$\mathbb{E}_{p(x)}[D_{KL}(q_\phi(z|x) \parallel p(z))] = I(X; Z) + \text{TC}(Z) + \sum_{k=1}^K D_{KL}(q(z_k) \parallel p(z_k))$$
$$\text{where Total Correlation } \text{TC}(Z) = D_{KL}\left( q(z) \parallel \prod_{k=1}^K q(z_k) \right)$$

### 4. Variational Mixture of Posteriors (VampPrior)
$$p(z) = \frac{1}{K} \sum_{k=1}^K q_\phi(z | u_k), \quad u_k \in \mathbb{R}^D \text{ are learnable pseudo-inputs}$$

---

## Tensor Dimensionality

| Tensor Name | Mathematical Symbol | PyTorch Shape | Description |
|:---|:---|:---|:---|
| Input Batch | $X$ | `[B, D]` | Input batch with ambient dimension $D$ (e.g. 784 for MNIST) |
| Latent Means | $\mu_\phi(x)$ | `[B, K]` | Predicted posterior mean in $K$-dimensional latent space |
| Latent Log-Variances | $\log \sigma_\phi^2(x)$ | `[B, K]` | Unconstrained log-variance vector output by recognition network |
| Sampled Latents | $Z$ | `[B, K]` | Reparameterized latent codes $z = \mu + \sigma \odot \epsilon$ |
| Reconstructed Means | $\hat{X}$ | `[B, D]` | Decoded observation parameters emitted by generative model |
| Per-Dimension KL | $\text{KL}_k$ | `[K]` | Average relative entropy per latent dimension (used for collapse diagnosis) |

---

## Guarantees & Invariants

| Mathematical Property | Invariant Condition | Practical System Guarantee |
|:---|:---|:---|
| Lower Bound Invariant | $\beta = 1.0$ | The objective is mathematically guaranteed to be a strict lower bound on $\log p(x)$ |
| Independent Latents | $\beta > 1.0$ | Information bottleneck penalizes cross-dimension mutual information, driving coordinate disentanglement |
| Posterior Collapse Condition | $D_{KL} \to 0$ | If $q_\phi(z\|x) = p(z)$, mutual information $I(X; Z) = 0$, and the decoder output becomes independent of input $x$ |
| KKT Complementary Slackness | $\beta^*(D_{KL} - \epsilon) = 0$ | When the information budget $\epsilon$ is not fully consumed, the optimal multiplier $\beta$ vanishes |

---

## Contrastive Decision Table

| Strategy Parameter | Choice X: High Beta ($\beta > 1$) | Choice Y: Low Beta ($\beta < 1$) | Strategic Recommendation |
|:---|:---|:---|:---|
| Optimization Focus | Latent factor disentanglement & compactness | High-resolution, sharp pixel-level reconstruction | Use $\beta > 1$ for representation learning / feature discovery; use $\beta < 1$ for image reconstruction |
| Latent Manifold Structure | Dense, smooth, aligned with $\mathcal{N}(0, I)$ | Discontinuous clusters with potential latent holes | High $\beta$ allows smooth interpolation; low $\beta$ causes artifact gaps during ancestral sampling |
| Sample Visual Quality | Slightly blurry (due to consensus averaging) | Sharp and clear (captures high-frequency edges) | For sharp generation with high $\beta$, pair with hierarchical latents or adversarial perceptual losses |
| Vulnerability | Severe posterior collapse in uninformative dims | Overfitting to training samples (Dirac delta lookup) | Apply KL annealing schedule $\beta_t = \min(1.0, t/T)$ to avoid early collapse |

---

## Numerical Stability & Traps

1. **Premature Collapse under Constant High Beta:**
   - **Trap:** Starting training immediately with $\beta = 4.0$ from epoch 1 causes the encoder to collapse to $\mathcal{N}(0, I)$ before the decoder can establish useful reconstructions.
   - **Numerical Fix:** Use a linear or monotonic warm-up schedule: $\beta(t) = \beta_{\text{max}} \cdot \min(1.0, t / T_{\text{warmup}})$.
2. **Loss Scale Inbalance in High Ambient Dimensions:**
   - **Trap:** If $D = 128 \times 128 \times 3 = 49,152$, raw MSE loss will be orders of magnitude larger than a $K=32$ KL term, rendering standard $\beta = 1$ virtually ineffective.
   - **Numerical Fix:** Normalize the reconstruction loss by the ambient dimension $D$ (mean per pixel) or scale $\beta$ proportionally to $D/K$.
