# Formulae & Tensor Mechanics: Lec 12 Vector Quantised VAE

## Core Mechanics & Equations
The continuous convolutional encoder maps input observation $x$ to feature grid $z_e(x) \in \mathbb{R}^{H 	imes W 	imes D}$.
For each spatial location $(i, j)$:
$$k^*(i, j) = rg\min_{k \in \{1, \dots, K\}} ||z_e(x)_{i,j} - e_k||_2^2$$
$$z_q(x)_{i, j} = e_{k^*(i, j)}$$

### Straight-Through Estimator (STE) Identity
To route gradients around the non-differentiable argmin:
$$z_q(x) = z_e(x) + 	ext{sg}[z_q(x) - z_e(x)]$$
Forward evaluation yields $z_q(x)$, while autograd computes:
$$rac{\partial z_q(x)}{\partial z_e(x)} = \mathbf{I}, \quad 
abla_{z_e} \mathcal{L} = 
abla_{z_q} \mathcal{L}$$

### Tripartite VQ-VAE Loss Function
$$\mathcal{L}(x, 	heta, \mathcal{E}) = \mathcal{L}_{	ext{recon}}(x, 	ext{Dec}(z_q)) + ||	ext{sg}[z_e(x)] - e||_2^2 + eta ||z_e(x) - 	ext{sg}[e]||_2^2$$
- **Reconstruction Loss:** $\mathcal{L}_{	ext{recon}} = -\log p(x | z_q) = rac{1}{2\sigma^2} ||x - \hat{x}||_2^2$ (Gaussian observation)
- **VQ Codebook Loss:** $||	ext{sg}[z_e(x)] - e||_2^2$ trains dictionary embeddings $\mathcal{E}$.
- **Commitment Loss:** $eta ||z_e(x) - 	ext{sg}[e]||_2^2$ prevents encoder space explosion; $eta \in [0.1, 0.5]$.

### Exponential Moving Average (EMA) Update Equations
For codebook vector $k$ at training step $t$:
$$N_k^{(t)} = \gamma N_k^{(t-1)} + (1 - \gamma) \sum_{i,j} \mathbb{I}[k^*(i, j) = k]$$
$$m_k^{(t)} = \gamma m_k^{(t-1)} + (1 - \gamma) \sum_{i,j} \mathbb{I}[k^*(i, j) = k] z_e(x)_{i,j}$$
$$e_k^{(t)} = rac{m_k^{(t)}}{N_k^{(t)}}$$
where $\gamma \in [0.9, 0.999]$ is the decay factor.

## Guarantees & Invariants
- **Zero Posterior Collapse:** Unlike continuous VAEs, no KL divergence term penalizes the latent code during Stage 1 autoencoding. The mutual information $I(X; Z_q)$ remains strictly positive.
- **Finite Latent Information Channel:** The discrete latent grid represents exactly $H 	imes W 	imes \log_2 K$ bits of information.
- **Gradient Identity under STE:** The gradient received by the continuous encoder is strictly equal to the gradient emitted by the continuous decoder with respect to the quantized representation.

## Contrastive Decision Table
| Objective / Component | Active Parameters Updated | Frozen / Stop-Gradient Parameters | Primary Failure if Omitted |
|---|---|---|---|
| Reconstruction Loss $\mathcal{L}_{recon}$ | Decoder $	heta_{dec}$ & Encoder $	heta_{enc}$ (via STE) | Codebook $\mathcal{E}$ | Model produces blank / uninformative outputs |
| VQ Codebook Loss $\mathcal{L}_{vq}$ | Codebook Dictionary $\mathcal{E}$ | Encoder $z_e(x)$ via $	ext{sg}[z_e]$ | Codebook never moves toward data representations |
| Commitment Loss $\mathcal{L}_{commit}$ | Continuous Encoder $	heta_{enc}$ | Codebook $e$ via $	ext{sg}[e]$ | Encoder output space explodes unboundedly |
| EMA Update Routine | Codebook Dictionary $\mathcal{E}$ | Non-differentiable (No Autograd) | Codebook updates suffer high variance under small batches |

## Numerical Stability & Traps
- **Pairwise Distance Expansion:** Compute $||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 z_e e_k^T$. To prevent negative values caused by floating-point rounding, clamp with `torch.clamp(min=0.0)`.
- **Epsilon in EMA Denominator:** Always add $\epsilon = 10^{-5}$ to $N_k^{(t)}$ before division to avoid catastrophic division-by-zero when cluster counts approach zero.
- **Dead Code Re-initialization:** If $N_k^{(t)} < 1.0$, re-initialize $e_k$ by drawing a random continuous encoder vector from the current batch.

## Tensor Shapes & Dimensionality Lifecycle
| Stage | Tensor Variable | Shape | Description |
|---|---|---|---|
| Input Image | $x$ | `(B, 3, H_in, W_in)` | Raw batch of input images |
| Encoder Output | $z_e(x)$ | `(B, D, H, W)` | Continuous representation grid |
| Flattened Latent | $z_{flat}$ | `(B * H * W, D)` | Vectors to be quantized against codebook |
| Codebook Weights | $\mathcal{E}$ | `(K, D)` | Learnable discrete prototype dictionary |
| Pairwise Distances | $D_{dist}$ | `(B * H * W, K)` | Squared Euclidean distances |
| Quantized Indices | $\mathbf{k}^*$ | `(B, H, W)` | Categorical token indices $\in \{1, \dots, K\}$ |
| Quantized Latent | $z_q(x)$ | `(B, D, H, W)` | Discretized spatial latent grid |
| Reconstructed Image| $\hat{x}$ | `(B, 3, H_in, W_in)` | Output of convolutional decoder |
