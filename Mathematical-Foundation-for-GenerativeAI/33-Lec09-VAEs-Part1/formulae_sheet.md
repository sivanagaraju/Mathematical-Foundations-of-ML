# Master Formulae Sheet & Mathematical Invariants: VAEs Part 1

> **Package:** 33-Lec09-VAEs-Part1  
> **Role:** High-density mathematical and architectural quick-reference cheat sheet for Variational Autoencoders.  
> **Audience:** Machine learning engineers and researchers implementing generative probabilistic models.

---

## 1. Master Equations Index

### 1.1 The Marginal Likelihood Integral (Evidence)

$$
p_\theta(\mathbf{x}) = \int_{\mathbb{R}^k} p_\theta(\mathbf{x}, \mathbf{z}) \, d\mathbf{z} = \int_{\mathbb{R}^k} p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) \, d\mathbf{z}
$$

> **In words:** The marginal probability of observing data vector $\mathbf{x}$ equals the product of the conditional decoder likelihood $p_\theta(\mathbf{x} \mid \mathbf{z})$ and the latent prior $p(\mathbf{z})$, accumulated across the entire continuous latent space $\mathbb{R}^k$.

**Parameter Breakdown:**
- $\mathbf{x} \in \mathbb{R}^d$: High-dimensional observed data vector ($d$-dimensional).
- $\mathbf{z} \in \mathbb{R}^k$: Unobserved continuous latent vector ($k$-dimensional, typically $k \ll d$).
- $p(\mathbf{z})$: Fixed generative prior over latent codes (standardly $\mathcal{N}(\mathbf{0}, \mathbf{I}_k)$).
- $p_\theta(\mathbf{x} \mid \mathbf{z})$: Generative decoding model parameterized by deep network weights $\theta$.
- $p_\theta(\mathbf{x})$: Net observable evidence density (analytically intractable in deep models).

**Mental Reconstruction Hook:**
*Think of taking an x-ray: $p(\mathbf{x} \mid \mathbf{z})$ is the shadow cast by each internal bone slice $\mathbf{z}$, and the marginal $p(\mathbf{x})$ is the total exposed film after accumulating all slices.*

---

### 1.2 The Evidence Lower Bound (ELBO) Decomposition

$$
\mathcal{L}_{\text{ELBO}}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}))
$$

> **In words:** The Evidence Lower Bound equals the expected reconstruction log-likelihood of data $\mathbf{x}$ under the variational posterior minus the Kullback-Leibler divergence contracting the variational posterior toward the prior.

**Parameter Breakdown:**
- $q_\phi(\mathbf{z} \mid \mathbf{x})$: Variational posterior approximation parameterized by encoder weights $\phi$.
- $\mathbb{E}_{\mathbf{z} \sim q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]$: Reconstruction fidelity score measuring how accurately the decoder unrolls latents into data.
- $D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}))$: Regularization penalty enforcing latent manifold continuity and preventing memorization.
- Fundamental Bound Guarantee: $\log p_\theta(\mathbf{x}) = \mathcal{L}_{\text{ELBO}}(\theta, \phi; \mathbf{x}) + D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p_\theta(\mathbf{z} \mid \mathbf{x})) \ge \mathcal{L}_{\text{ELBO}}(\theta, \phi; \mathbf{x})$.

**Mental Reconstruction Hook:**
*Reconstruction pulls the model toward sharp, memorized data; KL divergence pushes it toward a standardized, smooth Gaussian ball. Maximizing ELBO strikes an equilibrium between fidelity and entropy.*

---

### 1.3 Closed-Form Diagonal Gaussian KL Divergence

$$
D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2)) \parallel \mathcal{N}(\mathbf{0}, \mathbf{I}_k)) = -\frac{1}{2} \sum_{j=1}^k \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)
$$

> **In words:** The statistical information divergence from a standard isotropic Gaussian prior to a diagonal Gaussian posterior evaluates analytically as negative one-half the sum of one plus log-variance minus mean squared minus variance.

**Parameter Breakdown:**
- $\boldsymbol{\mu} \in \mathbb{R}^k$: Predicted latent mean vector output by the encoder network.
- $\boldsymbol{\sigma}^2 \in \mathbb{R}^k_{>0}$: Predicted latent coordinate variances output by the encoder network.
- $\log(\sigma_j^2) \in \mathbb{R}$: Unconstrained log-variance emitted by the network head to ensure variance positivity.

**Mental Reconstruction Hook:**
*When $\mu = 0$ and $\sigma^2 = 1$, the bracket evaluates to $1 + 0 - 0 - 1 = 0$, giving exact zero divergence. Any departure from mean zero or unit variance incurs a strictly positive quadratic and logarithmic penalty.*

---

### 1.4 The Pathwise Reparameterization Gradient via LOTUS

$$
\nabla_\phi \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})}[f(\mathbf{z})] = \mathbb{E}_{\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})}\left[ \left. \nabla_\mathbf{z} f(\mathbf{z}) \right|_{\mathbf{z}=g_\phi(\boldsymbol{\epsilon}, \mathbf{x})} \cdot \nabla_\phi g_\phi(\boldsymbol{\epsilon}, \mathbf{x}) \right]
$$

> **In words:** The gradient of a parameterized expectation equals the expectation over a parameter-free noise source of the upstream loss gradient multiplied by the local Jacobian of the coordinate mapping.

**Parameter Breakdown:**
- $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$: Auxiliary noise vector drawn from a fixed distribution carrying zero functional dependence on $\phi$.
- $g_\phi(\boldsymbol{\epsilon}, \mathbf{x}) = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}$: Differentiable coordinate transformation.
- $\nabla_\mathbf{z} f(\mathbf{z})$: Upstream sensitivity vector transmitted back through the decoder.
- $\nabla_\phi g_\phi(\boldsymbol{\epsilon}, \mathbf{x})$: Local coordinate Jacobian propagating sensitivities through encoder layers.

**Mental Reconstruction Hook:**
*Unhitch the random number generator: let noise enter as a static leaf node $\boldsymbol{\epsilon}$, leaving an unbroken deterministic highway connecting loss $f(\mathbf{z})$ to encoder weights $\phi$.*

---

### 1.5 Probability Integral Transform & Inverse CDF Sampling

$$
U = F_X(X) \sim \mathcal{U}(0, 1) \iff X = F_X^{-1}(U) \sim P_X
$$

> **In words:** Any continuous random variable transformed through its own cumulative distribution function becomes uniformly distributed on the unit interval, and conversely, uniform noise mapped through an inverse CDF recovers the exact target distribution.

**Parameter Breakdown:**
- $X \sim P_X$: Continuous target random variable with Cumulative Distribution Function $F_X(x) = P(X \le x)$.
- $U \sim \mathcal{U}(0, 1)$: Standard uniform random variable on the interval $[0, 1]$.
- $F_X^{-1}(u)$: Quantile function mapping probability $u \in [0, 1]$ to physical coordinate values.

**Mental Reconstruction Hook:**
*The CDF flattens any irregular rolling distribution into a level uniform runway; the inverse CDF bends the flat runway back into the target probability landscape.*

---

## 2. Input/Output Tensor Dimensionality

| Stage / Module | Input Tensor Shape | Output Tensor Shape | Mathematical Meaning | PyTorch Layer Syntax |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder Input** | `(B, d)` | `(B, d)` | Batch of flattened input data vectors $\mathbf{x}$ | `x` (raw features) |
| **Encoder Backbone** | `(B, d)` | `(B, h_dim)` | Intermediate compressed representation | `h = F.relu(self.fc1(x))` |
| **Encoder Head (Mean)** | `(B, h_dim)` | `(B, k)` | Predicted conditional mean $\boldsymbol{\mu}_\phi(\mathbf{x})$ | `mu = self.fc_mu(h)` |
| **Encoder Head (LogVar)** | `(B, h_dim)` | `(B, k)` | Predicted log-variance $\log \boldsymbol{\sigma}^2_\phi(\mathbf{x})$ | `logvar = self.fc_logvar(h)` |
| **Auxiliary Noise Source** | N/A | `(B, k)` | Parameter-free standard normal noise $\boldsymbol{\epsilon}$ | `eps = torch.randn_like(mu)` |
| **Reparameterization Node** | `(B, k), (B, k), (B, k)` | `(B, k)` | Latent embeddings $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ | `z = mu + torch.exp(0.5*logvar)*eps` |
| **Decoder Backbone** | `(B, k)` | `(B, h_dim)` | Unrolling latent vectors into hidden state | `h_dec = F.relu(self.fc2(z))` |
| **Decoder Head (Recon)** | `(B, h_dim)` | `(B, d)` | Reconstructed data parameters $\hat{\mathbf{x}} = \boldsymbol{\mu}_\theta(\mathbf{z})$ | `x_hat = self.fc_out(h_dec)` |
| **Reconstruction Loss** | `(B, d), (B, d)` | `(B,)` or `()` | Negative conditional log-likelihood $-\log p_\theta$ | `F.mse_loss(x_hat, x, reduction='sum')` |
| **KL Divergence Loss** | `(B, k), (B, k)` | `(B,)` or `()` | Closed-form Gaussian relative entropy penalty | `-0.5 * torch.sum(1 + logvar - mu**2 - exp(logvar))` |

---

## 3. Mathematical Guarantees & Invariants Table

| Invariant / Property | Mathematical Formalism | Guarantee / Condition | Failure Mode if Violated |
| :--- | :--- | :--- | :--- |
| **Non-Negativity of Evidence** | $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}, \mathbf{z}) d\mathbf{z} \ge 0$ | Guaranteed by Kolmogorov's probability axioms | Negative densities break all likelihood-based training |
| **Strict Lower Bound** | $\mathcal{L}_{\text{ELBO}}(\theta, \phi; \mathbf{x}) \le \log p_\theta(\mathbf{x})$ | Guaranteed by Jensen's inequality for concave $\log$ | Optimizing a loose upper bound provides no likelihood guarantees |
| **Tightness Condition** | $\mathcal{L}_{\text{ELBO}} = \log p_\theta(\mathbf{x}) \iff q_\phi(\mathbf{z} \mid \mathbf{x}) = p_\theta(\mathbf{z} \mid \mathbf{x})$ | Occurs when variational posterior matches true posterior | Approximation gap equals $D_{\text{KL}}(q_\phi \parallel p_\theta(\mathbf{z} \mid \mathbf{x})) \ge 0$ |
| **Gibbs' Inequality** | $D_{\text{KL}}(q \parallel p) \ge 0$, with equality iff $q = p$ a.e. | Guaranteed by concavity of logarithm | Negative divergence produces unbounded loss minimization |
| **Zero-Sum Score Integral** | $\int_{\mathcal{Z}} \nabla_\phi q_\phi(\mathbf{z}) d\mathbf{z} \equiv \mathbf{0}$ | Total probability must integrate to 1 across all $\phi$ | Proves $\nabla_\phi q_\phi$ is not a density and cannot be sampled |
| **Parameter-Free Noise** | $\nabla_\phi p(\boldsymbol{\epsilon}) \equiv \mathbf{0}$ | Base noise $\boldsymbol{\epsilon}$ must carry zero functional dependence on $\phi$ | Differentiating through $p(\boldsymbol{\epsilon})$ resurrects the measure gradient trap |
| **Affine Gaussian Closure** | $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\Sigma}^{1/2}\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | Guaranteed by linear properties of Gaussian measures | Non-affine operations destroy Gaussian posterior preservation |

---

## 4. Contrastive "Why X, Not Y" Quick Table

| Chosen Architecture (X) | Rejected Alternative (Y) | Why Alternative (Y) Fails Catastrophically | Core Mathematical / Engineering Mechanism |
| :--- | :--- | :--- | :--- |
| **Pathwise Reparameterization** | Score-Function (REINFORCE) | Catastrophic gradient variance ($50\times$ higher) halts deep network convergence | Score estimator scales with $\|f(\mathbf{z})\|^2$, treating decoder as a black box |
| **Diagonal Covariance $\text{diag}(\boldsymbol{\sigma}^2)$** | Full Covariance Matrix $\boldsymbol{\Sigma}$ | Memory explosion ($k^2$ outputs) and $O(k^3)$ Cholesky inversion per sample | For $k=512$, full covariance requires 131,328 outputs vs 1,024 for diagonal |
| **Predicting Log-Variance $\log \boldsymbol{\sigma}^2$** | Predicting Raw Variance $\boldsymbol{\sigma}^2$ | Unconstrained gradient descent predicts negative variance, crashing autograd | Logarithm maps $(0, \infty)$ variance support to $(-\infty, \infty)$ unconstrained space |
| **Probabilistic Representation** | Deterministic Bottleneck Autoencoder | Latent space develops unmapped voids; random generation produces corrupted noise | Deterministic mapping forms isolated delta spikes with divergent KL divergence |
| **Analytical Gaussian KL** | Monte Carlo Sampled KL | Sample noise introduces variance into the regularization penalty | Closed-form formula evaluates exact relative entropy with zero variance |
| **Mini-Batch Sampling ($L=1$)** | Large Latent Sampling ($L=100$) | Redundant computation; $L=1$ sample per item is proven sufficient across batches | Batch stochasticity provides sufficient empirical smoothing for gradient descent |

---

## 5. Hardware Realities & Stability

### 5.1 Floating-Point 32 Dynamic Range & Safe Exponentiation
In single-precision IEEE 754 floating-point arithmetic (`float32`):
- Maximum representable finite value: $\approx 3.4028 \times 10^{38}$.
- Exponent overflow threshold: $e^{88.7228} \approx 3.4028 \times 10^{38}$.
- **The Crash Mechanism:** If the encoder emits an unconstrained log-variance logit exceeding $88.7$, calling `torch.exp(logvar)` evaluates to `inf`. Subsequent subtraction in the KL divergence ($\text{inf} - \text{inf}$) outputs `NaN`, immediately destroying all model weights.
- **Hardware Clamping Rule:** In all production forward passes, enforce:
  ```python
  logvar_clamped = torch.clamp(logvar, min=-30.0, max=20.0)
  ```
  `exp(20.0)` $\approx 4.85 \times 10^8$ (completely safe from overflow); `exp(-30.0)` $\approx 9.35 \times 10^{-14}$ (prevents division by zero).

### 5.2 Vector-Jacobian Product Memory Efficiency
When evaluating the pathwise chain rule $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}}$:
- **Materializing Full Jacobian Matrix:** Storing $\frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}} \in \mathbb{R}^{B \times k \times k}$ consumes $O(B \cdot k^2)$ VRAM.
- **VJP Contraction in Autograd:** Because $\frac{\partial z_j}{\partial \mu_j} = 1$ and $\frac{\partial z_j}{\partial \mu_i} = 0$ for $i \neq j$, the Jacobian is strictly diagonal. PyTorch contracts the vector-Jacobian product directly into a 1D tensor of shape `(B, k)` without ever allocating an $N \times N$ matrix in GPU memory.

### 5.3 Deterministic Pseudo-Random Seeding on Accelerators
To ensure bit-exact reproducibility across distributed GPU workers:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Standard isotropic Gaussian draws $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ must be generated using `torch.randn_like()` to ensure tensor device and memory stride alignment.
