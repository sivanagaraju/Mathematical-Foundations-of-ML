# Formulae & Tensor Mechanics: Tutorial 15 VAE & Beta-VAE Implementation

## Core Mechanics & Equations
### Forward Reparameterization
Given encoder outputs $\mu_\phi(x) \in \mathbb{R}^K$ and $\log \sigma_\phi^2(x) \in \mathbb{R}^K$:
$$\sigma_\phi(x) = \exp\left( 0.5 \times \text{clamp}(\log \sigma_\phi^2(x), -15.0, 10.0) \right)$$
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_K)$$

### Scaled $\beta$-VAE Objective Function
$$\mathcal{L}_{\beta}(\theta, \phi; x) = \mathcal{L}_{\text{recon}}(x, \hat{x}) + \beta \times D_{KL}(q_\phi(z|x) \parallel p(z))$$
where:
- **Binary Cross-Entropy (BCE) Reconstruction:**
  $$\mathcal{L}_{\text{recon}}^{\text{BCE}} = -\sum_{i=1}^D \left[ x_i \log \hat{x}_i + (1 - x_i) \log(1 - \hat{x}_i) \right]$$
- **Mean Squared Error (MSE) Reconstruction:**
  $$\mathcal{L}_{\text{recon}}^{\text{MSE}} = \frac{1}{2} \sum_{i=1}^D (x_i - \hat{x}_i)^2$$
- **Closed-Form Gaussian Relative Entropy:**
  $$D_{KL}(q_\phi(z|x) \parallel \mathcal{N}(\mathbf{0}, \mathbf{I})) = -\frac{1}{2} \sum_{k=1}^K \left( 1 + \log \sigma_k^2 - \mu_k^2 - \sigma_k^2 \right)$$

### Latent Coordinate Traversal Generation
To inspect dimension $k \in \{1, \dots, K\}$:
$$z_{\text{trav}}(k, v) = [0, \dots, 0, \underbrace{v}_{\text{position } k}, 0, \dots, 0]^T, \quad v \in [-3.0, +3.0]$$
$$\hat{x}_{\text{trav}} = \text{Decoder}(z_{\text{trav}}(k, v))$$

## Guarantees & Invariants
- **Strict Positivity of Standard Deviation:** Parameterizing via $\log \sigma^2$ guarantees $\sigma > 0$ everywhere without numerical clamping of standard deviations.
- **Identical Prior Recovery:** When $\mu_k = 0$ and $\log \sigma_k^2 = 0$, $D_{KL} = 0.0$ nats exactly.
- **Sample Independence:** Reparameterization noise $\epsilon^{(i)}$ is drawn independently for each sample in the batch: $\epsilon^{(i)} \perp \epsilon^{(j)}$ for $i \ne j$.

## Contrastive Decision Table
| Objective / Hyperparameter | Standard VAE ($\beta = 1.0$) | Disentangled $\beta$-VAE ($\beta > 1.0$) | High-Fidelity VAE ($\beta < 1.0$) |
|---|---|---|---|
| Optimization Focus | Strict lower bound on marginal likelihood $\log p(x)$ | Information bottleneck compression | Maximum visual reconstruction sharpness |
| Latent Space Geometry | Moderately smooth; slight factor entanglement | Highly factorized, independent coordinate axes | Dense, potentially fragmented with prior gaps |
| Reconstructed Sharpness | Moderate (Prone to consensus blur) | Softer (Consensus blur increases with $\beta$) | Crisp and sharp |
| Posterior Collapse Risk | Low to moderate | High (Requires warmup schedules) | Negligible |
| Latent Traversal Clarity | Features blend across multiple coordinates | Clean, isolated factor manipulation | Entangled, chaotic feature shifts |

## Numerical Stability & Traps
- **LogVar Clamping:** Always clamp $\log \sigma^2$ between $[-15.0, 10.0]$. Without clamping, large activations produce $\exp(0.5 \times 100) \to \infty$, causing `NaN` gradients.
- **Reduction Mode Alignment:** When combining reconstruction loss with KL divergence, ensure both use the same reduction mode (`reduction='sum'` or both divided by batch size $B$). Mixing `reduction='mean'` for reconstruction with `reduction='sum'` for KL skews $\beta$ by a factor of $D / K$.
- **Annealing Warmup Schedule:** When using $\beta > 1.0$, ramp $\beta$ linearly: $\beta_t = \min(\beta_{\text{target}}, \beta_{\text{target}} \times \frac{t}{T_{\text{warmup}}})$.

## Tensor Shapes & Dimensionality Lifecycle
| Stage | Tensor Variable | Shape | Description |
|---|---|---|---|
| Input Image Batch | $x$ | `(B, C, H, W)` | Mini-batch of input images |
| Encoder Feature Grid | $h_{\text{enc}}$ | `(B, C_feat, H', W')`| Output of strided convolutional layers |
| Flattened Features | $h_{\text{flat}}$ | `(B, C_feat * H' * W')`| Vectorized feature representations |
| Latent Mean Head | $\mu$ | `(B, K)` | Mean vectors of variational posteriors |
| Latent LogVar Head | $\log \sigma^2$ | `(B, K)` | Log-variance vectors of variational posteriors |
| Auxiliary Noise | $\epsilon$ | `(B, K)` | Standard normal samples $\epsilon \sim \mathcal{N}(0, I)$ |
| Sampled Latent Code | $z$ | `(B, K)` | Reparameterized latent representations $\mu + \sigma \odot \epsilon$ |
| Reconstructed Image | $\hat{x}$ | `(B, C, H, W)` | Output of convolutional decoder network |
| Per-Sample KL Loss | $D_{KL}$ | `(B,)` | Analytical relative entropy per sample |
| Total Loss | $\mathcal{L}$ | Scalar `()` | Scaled loss for backpropagation |
