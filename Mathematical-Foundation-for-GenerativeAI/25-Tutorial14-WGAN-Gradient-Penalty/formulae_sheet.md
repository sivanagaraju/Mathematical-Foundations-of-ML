# Formulae & Tensor Mechanics: Tutorial 14 WGAN Gradient Penalty

## Core Mechanics & Equations
### WGAN-GP Objective Formulation
$$\min_G \max_D \mathcal{L}(D, G) = \mathbb{E}_{x \sim P_r}[D(x)] - \mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})] - \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}} \left[ \left( ||\nabla_{\hat{x}} D(\hat{x})||_2 - 1 \right)^2 \right]$$
where:
- $x \sim P_r$ denotes real data.
- $\tilde{x} = G(z)$ with $z \sim p(z)$ denotes generated data.
- $\hat{x} = \epsilon x + (1 - \epsilon)\tilde{x}$ with $\epsilon \sim U(0, 1)$ defines straight-line interpolates.
- $\lambda = 10.0$ is the standard penalty coefficient.

### Straight-Line Interpolation Geometry
For each sample in the mini-batch:
$$\hat{x}^{(i)} = \epsilon^{(i)} x^{(i)} + (1 - \epsilon^{(i)}) \tilde{x}^{(i)}, \quad \epsilon^{(i)} \sim \text{Uniform}(0, 1)$$

### Gradient Norm Calculation
$$\nabla_{\hat{x}} D(\hat{x}) = \left[ \frac{\partial D(\hat{x})}{\partial \hat{x}_1}, \frac{\partial D(\hat{x})}{\partial \hat{x}_2}, \dots, \frac{\partial D(\hat{x})}{\partial \hat{x}_D} \right]^T$$
$$||\nabla_{\hat{x}} D(\hat{x})||_2 = \sqrt{\sum_{j=1}^D \left( \frac{\partial D(\hat{x})}{\partial \hat{x}_j} \right)^2 + \epsilon_{\text{stab}}}$$
where $\epsilon_{\text{stab}} = 10^{-12}$ prevents division by zero in backpropagation.

### Two-Sided Gradient Penalty
$$\mathcal{L}_{GP} = \left( ||\nabla_{\hat{x}} D(\hat{x})||_2 - 1 \right)^2$$

## Guarantees & Invariants
- **Exact Unit Gradient Norm on Geodesics:** Optimal transport theory proves that under the $L_2$ ground metric, the optimal Critic $D^*(x)$ satisfies $||\nabla_x D^*(x)||_2 = 1$ almost everywhere along straight lines connecting $x \sim P_r$ and $\tilde{x} \sim P_g$.
- **Smooth Optimization Landscape:** Gradient penalty imposes a continuous soft penalty, eliminating the non-smooth parameter truncations of weight clipping and enabling Adam optimization.
- **Sample Independence:** Because the penalty is computed per-sample on individual interpolates $\hat{x}^{(i)}$, the Critic satisfies point-wise Lipschitz evaluation.

## Contrastive Decision Table
| Mechanism | WGAN (Weight Clipping) | WGAN-GP (Gradient Penalty) |
|---|---|---|
| Lipschitz Enforcement | Hard parameter clamp $w \in [-c, c]$ | Soft gradient norm regularizer $(\|\nabla D\| - 1)^2$ |
| Effective Capacity | Severe underuse (weights stick to $\pm c$) | Full network capacity preserved |
| Computational Overhead| Zero extra backward passes | One extra backward pass with `create_graph=True` |
| Normalization Allowed | None or LayerNorm | LayerNorm or SpectralNorm (NO BatchNorm) |
| Optimizer Supported | RMSprop (No momentum) | Adam ($\beta_1 = 0.0, \beta_2 = 0.9$) |
| Gradient Behavior | Prone to vanishing/exploding gradients | Highly stable across 100+ layer architectures |

## Numerical Stability & Traps
- **Epsilon Inside Sqrt:** When computing $\|\nabla D\|_2 = \sqrt{\sum (\nabla D)^2}$, always add $\epsilon = 10^{-12}$ inside the square root. Without $\epsilon$, if the gradient evaluates to zero, $\frac{d}{dx} \sqrt{x} = \frac{1}{2\sqrt{x}}$ produces `NaN` gradients.
- **The `create_graph=True` Requirement:** When calling `torch.autograd.grad(outputs=critic_hat, inputs=hat_x, create_graph=True)`, failing to set `create_graph=True` disconnects Critic weights from the gradient penalty, rendering $\mathcal{L}_{GP}$ inert.
- **Prohibition of BatchNorm:** Never include `nn.BatchNorm2d` in the Critic network. BatchNorm couples samples across the batch, invalidating the per-sample Lipschitz condition.

## Tensor Shapes & Dimensionality Lifecycle
| Stage | Tensor Variable | Shape | Description |
|---|---|---|---|
| Real Samples | $x$ | `(B, C, H, W)` | Batch of genuine training images |
| Fake Samples | $\tilde{x}$ | `(B, C, H, W)` | Batch of generated images from $G(z)$ |
| Uniform Weights | $\epsilon$ | `(B, 1, 1, 1)` | Random scalars drawn from $\text{Uniform}(0, 1)$ |
| Interpolated Samples | $\hat{x}$ | `(B, C, H, W)` | Straight-line points $\epsilon x + (1-\epsilon)\tilde{x}$ |
| Interpolate Critic Output| $D(\hat{x})$ | `(B, 1)` | Unbounded scalar evaluations of interpolates |
| Interpolate Gradients | $\nabla_{\hat{x}} D(\hat{x})$ | `(B, C, H, W)` | Output of `torch.autograd.grad` |
| Flattened Gradients | $g_{\text{flat}}$ | `(B, C * H * W)` | Reshaped gradients for norm calculation |
| Gradient Norms | $\|\nabla D\|_2$ | `(B,)` | L2 norm of gradient for each sample |
| Gradient Penalty | $\mathcal{L}_{GP}$ | Scalar `()` | `mean((grad_norms - 1.0) ** 2)` |
