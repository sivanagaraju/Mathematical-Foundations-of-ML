# Formulae & Tensor Mechanics: Tutorial 13 WGAN Weight Clipping

## Core Mechanics & Equations
### Primal Optimal Transport Formulation
$$W_1(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma}[||x - y||]$$
where $\Pi(P_r, P_g)$ is the set of all joint distributions with marginals $P_r$ and $P_g$.

### Kantorovich-Rubinstein Dual Objective
$$W_1(P_r, P_g) = \sup_{||f||_L \le 1} \left( \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{y \sim P_g}[f(y)] \right)$$
where $||f||_L \le 1$ denotes the 1-Lipschitz condition:
$$|f(x_1) - f(x_2)| \le ||x_1 - x_2||, \quad \forall x_1, x_2 \in \mathbb{R}^D$$

### WGAN Value Function with Parameterized Critic
$$\min_\theta \max_{w \in \mathcal{W}} V(f_w, G_\theta) = \mathbb{E}_{x \sim P_r}[f_w(x)] - \mathbb{E}_{z \sim p(z)}[f_w(G_\theta(z))]$$
where $\mathcal{W} = \{w : ||w||_\infty \le c\}$ is enforced via weight clipping.

### Weight Clipping Step
For each parameter tensor $w$ in the Critic network:
$$w \leftarrow \text{clamp}(w, -c, c)$$
typically with clipping parameter $c = 0.01$.

## Guarantees & Invariants
- **Non-Vanishing Gradients:** Under optimal Critic $f_w^*$, the gradient with respect to generator parameters $\theta$ satisfies:
  $$\nabla_\theta W_1(P_r, P_\theta) = -\mathbb{E}_{z \sim p(z)} [\nabla_\theta f_w^*(G_\theta(z))] = -\mathbb{E}_{z \sim p(z)} [\nabla_x f_w^*(x)|_{x=G_\theta(z)} \nabla_\theta G_\theta(z)]$$
  which does not vanish even when supports of $P_r$ and $P_\theta$ are completely disjoint.
- **Metric Invariant:** $W_1(P_r, P_g) \ge 0$ with equality if and only if $P_r = P_g$.
- **Critic Score Invariant:** Unlike probabilities, the Critic output $f_w(x)$ is unconstrained and can take any real value in $(-\infty, +\infty)$.

## Contrastive Decision Table
| Objective / Component | Standard GAN (Goodfellow et al.) | Wasserstein GAN (Arjovsky et al.) |
|---|---|---|
| Optimization Objective | $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$ | $\min_G \max_{w \in \mathcal{W}} \mathbb{E}[f_w(x)] - \mathbb{E}[f_w(G(z))]$ |
| Divergence Minimized | Jensen-Shannon Divergence $D_{JS}(P_r \parallel P_g)$ | Earth Mover's Distance $W_1(P_r, P_g)$ |
| Discriminator Output | Probability score $D(x) \in [0, 1]$ via Sigmoid | Unbounded scalar potential $f_w(x) \in \mathbb{R}$ (Linear) |
| Disjoint Supports Behavior| Gradients vanish completely ($D_{JS} = \log 2$) | Gradients remain non-zero and linear with distance |
| Constraint Mechanism | None | Hard parameter weight clipping $w \in [-c, c]$ |
| Recommended Optimizer | Adam ($\beta_1 = 0.5, \beta_2 = 0.999$) | RMSprop ($\alpha = 0.00005$, no momentum) |

## Numerical Stability & Traps
- **Clipping Parameter Sensitivity:** If $c$ is set too small ($c < 0.001$), gradients vanish exponentially through layers. If $c$ is too large ($c > 0.1$), weights explode and optimization destabilizes. Default to $c = 0.01$.
- **No Momentum with Clipping:** Avoid using Adam with high momentum ($\beta_1 = 0.9$). When weights hit $[-c, c]$, momentum pushes against boundaries, causing erratic oscillations. Use RMSprop.
- **No Sigmoid or BatchNorm in Output:** The Critic must be free to scale its output to match true metric distances. Never apply Sigmoid to the final layer.

## Tensor Shapes & Dimensionality Lifecycle
| Stage | Tensor Variable | Shape | Description |
|---|---|---|---|
| Real Image Batch | $x$ | `(B, C, H, W)` | Batch of training images |
| Latent Noise | $z$ | `(B, Z_dim)` | Standard normal noise vector $z \sim \mathcal{N}(0, I)$ |
| Generated Fake Batch | $\tilde{x} = G_\theta(z)$ | `(B, C, H, W)` | Output of convolutional generator |
| Real Critic Score | $f_w(x)$ | `(B, 1)` | Unbounded scalar evaluations of real data |
| Fake Critic Score | $f_w(\tilde{x})$ | `(B, 1)` | Unbounded scalar evaluations of generated data |
| Critic Loss | $\mathcal{L}_C$ | Scalar `()` | `mean(f_w(fake)) - mean(f_w(real))` |
| Generator Loss | $\mathcal{L}_G$ | Scalar `()` | `-mean(f_w(fake))` |
