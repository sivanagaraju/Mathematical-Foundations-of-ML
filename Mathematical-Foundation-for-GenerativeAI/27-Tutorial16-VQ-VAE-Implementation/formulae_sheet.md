# Formulae & Tensor Mechanics: Tutorial 16 VQ-VAE Implementation

## Core Mechanics & Equations
### Vectorized Pairwise Distance Computation
Given flattened encoder representations $z_{\text{flat}} \in \mathbb{R}^{N \times D}$ (where $N = B \times H' \times W'$) and codebook weights $\mathcal{E} \in \mathbb{R}^{K \times D}$:
$$\|z_e - e_k\|_2^2 = \|z_e\|_2^2 + \|e_k\|_2^2 - 2 \langle z_e, e_k \rangle$$
In PyTorch tensor operations:
$$D_{\text{dist}} = \sum_{d=1}^D z_{\text{flat}}^2 + \sum_{d=1}^D \mathcal{E}^2 - 2 (z_{\text{flat}} \mathcal{E}^T) \in \mathbb{R}^{N \times K}$$

### Discrete Quantization & STE Gradient Identity
$$k^*(n) = \arg\min_{k \in \{0, \dots, K-1\}} D_{\text{dist}}(n, k)$$
$$z_q = \mathcal{E}[k^*]$$
$$z_q^{\text{STE}} = z_e + (z_q - z_e).\text{detach}()$$

### Tripartite Loss Objective
$$\mathcal{L} = \mathcal{L}_{\text{recon}}(x, \hat{x}) + \|z_e.\text{detach}() - z_q\|_2^2 + \beta \|z_e - z_q.\text{detach}()\|_2^2$$
where $\beta \in [0.1, 0.5]$ is the commitment multiplier.

### Codebook Usage Perplexity
Let $p_k = \frac{1}{N} \sum_{n=1}^N \mathbb{I}[k^*(n) = k]$ denote empirical cluster assignment probability.
$$\text{Entropy: } H(p) = -\sum_{k=1}^K p_k \log(p_k + \epsilon)$$
$$\text{Perplexity: } P = \exp(H(p))$$

## Guarantees & Invariants
- **Exact Gradient Routing:** Autograd evaluates $\frac{\partial z_q^{\text{STE}}}{\partial z_e} = \mathbf{I}$, guaranteeing that downstream reconstruction gradients flow directly into the convolutional encoder.
- **Strict Decoupling of Loss Terms:** Detaching inputs in the VQ loss ensures codebook updates do not backpropagate to encoder weights, while detaching quantized targets in the commitment loss ensures encoder updates do not alter codebook embeddings.
- **Perplexity Bounds:** $1.0 \le \text{Perplexity} \le K$. If perplexity approaches $1.0$, codebook collapse has occurred.

## Contrastive Decision Table
| Feature / Implementation | PyTorch VQ-VAE Module | Standard Continuous VAE |
|---|---|---|
| Latent Representation | Discrete Codebook Grid $z_q \in \mathbb{R}^{B \times D \times H' \times W'}$ | Continuous Gaussian $\mu, \log \sigma^2 \in \mathbb{R}^{B \times K}$ |
| Sampling Mechanism | Deterministic Nearest-Neighbor Argmin | Differentiable Stochastic Reparameterization |
| Backpropagation Through Latents| Straight-Through Estimator (`.detach()`) | Analytical Jacobian through $\mu + \sigma \odot \epsilon$ |
| Regularization Objective | Tripartite Loss: $\mathcal{L}_{vq} + \beta \mathcal{L}_{commit}$ | Analytical Relative Entropy: $\beta D_{KL}$ |
| Posterior Collapse Risk | Zero (No Gaussian prior during Stage 1) | High (Decoders easily ignore latents) |
| Usage Metric | Codebook Perplexity $\exp(H(p))$ | Active Latent Dimensions ($D_{KL} > 0.1$ nats) |

## Numerical Stability & Traps
- **Contiguous Memory Ordering:** Always call `.contiguous()` after `x.permute(...)` before invoking `.view(...)`. Without `.contiguous()`, PyTorch raises `RuntimeError: view size is not compatible with input tensor's size and stride`.
- **Epsilon in Perplexity Logarithm:** Add $\epsilon = 10^{-10}$ inside $\log(p_k + \epsilon)$ to prevent `NaN` when unused codebook entries have probability $p_k = 0$.
- **Negative Distance Rounding:** Due to floating-point rounding, expanded distances $\|z\|^2 + \|e\|^2 - 2ze^T$ can occasionally produce tiny negative numbers (e.g. $-10^{-7}$). Clamp with `torch.clamp(min=0.0)`.

## Tensor Shapes & Dimensionality Lifecycle
| Stage | Tensor Variable | Shape | Description |
|---|---|---|---|
| Input Image | $x$ | `(B, C, H, W)` | Mini-batch of training images |
| Encoder Output | $z_e$ | `(B, D, H', W')` | Continuous spatial representation grid |
| Permuted Latent | $z_{\text{perm}}$ | `(B, H', W', D)` | Rearranged for feature-last indexing |
| Flattened Latent | $z_{\text{flat}}$ | `(B * H' * W', D)` | Continuous vectors to quantize |
| Codebook Weights | $\mathcal{E}$ | `(K, D)` | Dictionary prototypes in `nn.Embedding` |
| Distance Matrix | $D_{\text{dist}}$ | `(B * H' * W', K)` | Squared Euclidean distances to all prototypes |
| Quantized Indices | $\mathbf{k}^*$ | `(B * H' * W',)` | Chosen nearest prototype indices |
| Quantized Tensor | $z_q$ | `(B, D, H', W')` | Discretized spatial latent grid |
| Reconstructed Image| $\hat{x}$ | `(B, C, H, W)` | Synthesized output from convolutional decoder |
