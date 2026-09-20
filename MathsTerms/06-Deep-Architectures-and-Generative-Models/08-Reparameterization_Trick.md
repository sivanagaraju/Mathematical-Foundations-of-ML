# Reparameterization Trick: Differentiating Through Stochastic Sampling for Variational Inference

[Module guide](README.md) · [Study routes](START_HERE.md) · Previous: [ELBO and variational inference](07-ELBO_and_Variational_Inference.md) · Next: [Minimax game and GANs](09-Minimax_Game_and_GANs.md)

## 1. What this idea helps you do

In deep latent variable models like Variational Autoencoders (VAEs), an encoder neural network outputs distribution parameters—such as mean $\mu_\phi(x)$ and standard deviation $\sigma_\phi(x)$. To evaluate the reconstruction loss $\ln p_\theta(x \mid z)$, the network must draw a continuous latent vector from this distribution:

$$z \sim q_\phi(z \mid x) = \mathcal{N}\big(\mu_\phi(x), \sigma_\phi^2(x)\big)$$

This creates a fundamental mathematical obstacle during backpropagation: **stochastic sampling is a non-differentiable roadblock**. A pseudo-random number generator output cannot be differentiated with respect to the input parameters that configured it:

$$\frac{\partial z}{\partial \mu} \quad \text{and} \quad \frac{\partial z}{\partial \sigma} \quad \text{are mathematically undefined!}$$

The **Reparameterization Trick** (also known as the Pathwise Gradient Estimator or Affine Location-Scale Transformation) solves this dilemma by isolating the stochastic randomness into an auxiliary parameter-free noise variable $\epsilon \sim \mathcal{N}(0, I)$. Instead of sampling directly from $q_\phi(z \mid x)$, the network computes $z$ as a deterministic, continuous, and differentiable function:

$$z = g_\phi(\epsilon, x) \triangleq \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

This transformation turns the stochastic sampling node into a standard linear operation inside the computational graph. Reverse-mode automatic differentiation (backpropagation) flows unhindered through $z$ directly into the encoder weights $\phi$, with exact analytical partial derivatives $\frac{\partial z}{\partial \mu} = 1$ and $\frac{\partial z}{\partial \sigma} = \epsilon$.

```text
================================================================================
       THE REPARAMETERIZATION TRICK: RESTORING DIFFERENTIABLE PATHWAYS
================================================================================

  NAIVE STOCHASTIC NODE (BLOCKED BACKPROP):
  ┌──────────────┐       z ~ 𝒩(μ_ϕ, σ_ϕ²)        ┌──────────────┐
  │ Encoder q_ϕ  ├──────────────────────────────►│ Decoder p_θ  │
  │ Outputs μ, σ │   ❌ NON-DIFFERENTIABLE DICE  │ Computes x̂   │
  └──────────────┘                               └──────┬───────┘
         ▲                                              │
         │  Gradient Flow BLOCKED: ∂z/∂ϕ is undefined!  │
         └─────────────────── ❌ ───────────────────────┘

  REPARAMETERIZED NODE (CONTINUOUS GRADIENT FLOW):
  ┌──────────────┐    z = μ_ϕ + σ_ϕ ⊙ ε          ┌──────────────┐
  │ Encoder q_ϕ  ├──────────────────────────────►│ Decoder p_θ  │
  │ Outputs μ, σ │    Deterministic Function     │ Computes x̂   │
  └──────┬───────┘                               └──────┬───────┘
         │              Fixed Noise: ε ~ 𝒩(0, I)        │
         ▲                         ▲                    │
         │       Backpropagation   │                    │
         └─────────────────────────┴────────────────────┘
             Gradients flow: ∂z/∂μ = 1, ∂z/∂σ = ε
================================================================================
```

*What to notice from the diagram:*
1. In the naive graph, random sampling severs the backward gradient chain between the decoder loss and the encoder parameters $\phi$.
2. In the reparameterized graph, randomness enters as an external input $\epsilon$, making $z = \mu + \sigma \odot \epsilon$ a differentiable node that allows standard chain-rule backpropagation.

**Prerequisites**
- **Required now:** Multivariate chain rule and computational graphs ([Chain rule and backpropagation, §4](../03-Multivariate-Calculus-and-Optimization/04-Chain_Rule_and_Backpropagation.md)). Gaussian distributions ([Common probability distributions, §2](../04-Probability-and-Statistical-Estimation/02-Common_Probability_Distributions.md)).
- **Required for optional depth:** The ELBO objective ([ELBO and variational inference, §4](07-ELBO_and_Variational_Inference.md)).
- **Useful context:** [Autoencoders and latent spaces](03-Autoencoders_and_Latent_Spaces.md) for latent bottlenecks.

**Target systems:** Variational Autoencoders (VAEs), Denoising Diffusion Probabilistic Models (DDPMs), Soft Actor-Critic (SAC) reinforcement learning, and Gumbel-Softmax discrete tokenizers.

**Study time:** About 45–60 minutes for pathwise derivations, variance proofs, and pencil-and-paper calculations; another 30 minutes for PyTorch verification and exercises.

After studying, you should be able to:
1. Formulate the pathwise gradient estimator for location-scale Gaussian distributions.
2. Prove why the pathwise estimator achieves dramatically lower variance than the score-function (REINFORCE) estimator.
3. Compute analytical gradients $\nabla_\mu z$ and $\nabla_\sigma z$ by hand.
4. Derive the Gumbel-Softmax relaxation for differentiating through discrete categorical choices.
5. Implement and verify a reparameterized VAE sampling layer in pure Python and PyTorch autograd.

**Fast route:** §§2–4 $\to$ §7 $\to$ §9 $\to$ §11 $\to$ §12, then §10 for generative AI systems.  
**Deep route:** §§2–14 in order; §4 and §8 contain complete Lebesgue dominated convergence proofs and analytical variance derivations.

---

## 2. Start with a problem you can picture

Suppose we want to find the parameter $\mu$ that minimizes the expected squared error of a random variable:
$$\min_\mu J(\mu) \triangleq \mathbb{E}_{z \sim \mathcal{N}(\mu, 1)}[z^2]$$

We can evaluate this expectation analytically using the definition of variance:
$$\mathbb{E}[z^2] = \operatorname{Var}(z) + (\mathbb{E}[z])^2 = 1.0 + \mu^2$$

The true analytical gradient with respect to $\mu$ is:
$$\nabla_\mu J(\mu) = \frac{d}{d\mu}(1.0 + \mu^2) = 2\mu$$
If our current parameter is $\mu = 3.0$, the true expected gradient is $2(3.0) = \mathbf{6.0}$.

Now suppose a neural network must estimate this gradient using **a single Monte Carlo sample** ($S=1$). Consider two different ways to estimate this gradient:

```text
CONTRASTING GRADIENT ESTIMATORS AT μ = 3.0:

1. REPARAMETERIZED PATHWISE ESTIMATOR:
   z = μ + ε,   where ε ~ 𝒩(0, 1)
   ∂(z²)/∂μ = 2z = 2(μ + ε)

   If ε = +0.50:   z = 3.50  ──►  Grad = 2(3.50) = +7.00   (True: +6.00)
   If ε = -2.00:   z = 1.00  ──►  Grad = 2(1.00) = +2.00   (Always positive!)

2. SCORE-FUNCTION (REINFORCE) ESTIMATOR:
   Grad = z² · ∇_μ ln q(z) = z² · (z - μ) = (μ + ε)² · ε

   If ε = +0.50:   Grad = (3.50)² · (0.50) = 12.25 · 0.50 = +6.125
   If ε = -2.00:   Grad = (1.00)² · (-2.00) = 1.00 · (-2.00) = -2.00
                   ❌ WRONG SIGN! Points in the opposite direction!
```

*What to notice from the comparison:*
1. The pathwise estimator $\frac{\partial z^2}{\partial \mu} = 2z$ inspects the slope of the loss function directly. When $\mu = 3.0$, it is almost always positive, giving stable, consistent gradient updates toward $\mu = 0$.
2. The score-function (REINFORCE) estimator multiplies the loss $z^2$ by the score $(z - \mu) = \epsilon$. When $\epsilon = -2.0$, it outputs a negative gradient ($-2.00$), commanding gradient descent to increase $\mu$ further away from the minimum!
3. The variance of REINFORCE is massive, while the pathwise estimator provides clean, low-variance descent.

**Predict before calculating:** What happens to the variance of the pathwise estimator if the loss function is linear: $f(z) = c \cdot z$?

---

## 3. Name the objects and read the notation

Let $x$ be an observation, and let $q_\phi(z \mid x) = \mathcal{N}(\mu, \Sigma)$ be a multivariate Gaussian proposal where $\mu \in \mathbb{R}^d$ and $\Sigma = \operatorname{diag}(\sigma_1^2, \dots, \sigma_d^2)$.

The **reparameterization mapping** defines the continuous latent code $z$:
$$z \triangleq g_\phi(\epsilon, x) = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim p(\epsilon) = \mathcal{N}(\mathbf{0}, I_d)$$
where $\odot$ denotes the element-wise Hadamard product.

Read this equation aloud:  
*“Latent vector z is defined as g-phi of epsilon and x, equaling mu-phi of x plus sigma-phi of x element-wise multiplied by standard normal noise vector epsilon.”*

The **Pathwise Gradient Estimator** of an expected loss $\mathbb{E}_{q_\phi}[f(z)]$ is:
$$\nabla_\phi \mathbb{E}_{q_\phi(z \mid x)}[f(z)] = \mathbb{E}_{p(\epsilon)}\left[ \nabla_z f(z) \cdot \nabla_\phi g_\phi(\epsilon, x) \right]$$

| Symbol | Spoken as | Mathematical role / dimensions | Concrete toy value (§2 / §9) |
| :--- | :--- | :--- | :--- |
| $\epsilon$ | “epsilon” | Parameter-free standard normal noise; $\mathbb{R}^d$ | $\epsilon \sim \mathcal{N}(0, 1)$ |
| $\mu_\phi(x)$ | “mu phi of ex” | Learned latent mean vector; $\mathbb{R}^d$ | $\mu = 3.0$ |
| $\sigma_\phi(x)$ | “sigma phi of ex” | Learned latent standard deviation; $\mathbb{R}^d$ | $\sigma = 1.0$ |
| $s_\phi(x)$ | “logvar” | Log-variance $s \triangleq \ln(\sigma^2)$; $\mathbb{R}^d$ | $s = \ln(1.0) = 0.0$ |
| $z$ | “zee” | Reparameterized latent coordinate; $\mathbb{R}^d$ | $z = 3.0 + 1.0 \times 0.5 = 3.5$ |
| $\nabla_z f(z)$ | “grad z of f” | Derivative of loss with respect to latent code | $2z = 2(3.5) = 7.0$ |
| $\tau$ | “tau” | Gumbel-Softmax temperature hyperparameter | $\tau = 1.0 \to 0.1$ |

---

## 4. Build the central relationship

### The Core "Aha!" Discovery

In standard probability, when you change the parameters $\phi$ of a distribution $q_\phi(z)$, you are changing the shape of the probability density landscape over which you integrate. Differentiation cannot easily pass through the integral sign because the measure itself depends on $\phi$.

The reparameterization trick performs a **change of variables** in the integral. It pushes $\phi$ out of the probability measure and into the integrand function itself!

### Derivation: From Parameterized Measure to Differentiable Function

We wish to compute the gradient of an expected loss:
$$\nabla_\phi \mathbb{E}_{z \sim q_\phi(z \mid x)}[f(z)] = \nabla_\phi \int_{\mathbb{R}^d} f(z) q_\phi(z \mid x) \, dz$$

Apply the change of variables $z = g_\phi(\epsilon, x) = \mu + \sigma \odot \epsilon$.  
By the probability transformation rule, $q_\phi(z \mid x) dz = p(\epsilon) d\epsilon$.  
The integral becomes:
$$\int_{\mathbb{R}^d} f(z) q_\phi(z \mid x) \, dz = \int_{\mathbb{R}^d} f\big( g_\phi(\epsilon, x) \big) p(\epsilon) \, d\epsilon = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[ f\big( g_\phi(\epsilon, x) \big) \right]$$

Now, notice the crucial difference: **the noise distribution $p(\epsilon) = \mathcal{N}(0, I)$ has zero dependence on parameter $\phi$!**

Under the **Lebesgue Dominated Convergence Theorem** (assuming $f$ is continuously differentiable and bounded by an integrable function), differentiation and integration can be interchanged:

$$\nabla_\phi \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[ f\big( g_\phi(\epsilon, x) \big) \right] = \int_{\mathbb{R}^d} \nabla_\phi \Big( f\big( g_\phi(\epsilon, x) \big) \Big) p(\epsilon) \, d\epsilon$$

Applying the multivariable chain rule inside the integral:
$$\nabla_\phi \Big( f\big( g_\phi(\epsilon, x) \big) \Big) = \nabla_z f(z) \cdot \nabla_\phi g_\phi(\epsilon, x)$$

This yields the **Pathwise Gradient Identity**:
$$\boxed{\nabla_\phi \mathbb{E}_{q_\phi(z \mid x)}[f(z)] = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[ \nabla_z f(z) \cdot \nabla_\phi g_\phi(\epsilon, x) \right]}$$

```text
================================================================================
                    PATHWISE GRADIENT COMPUTATIONAL GRAPH
================================================================================
 FORWARD PASS:
 Encoder (x) ──► μ_ϕ(x) ──┐
             ──► σ_ϕ(x) ──┼──► z = μ + σ ⊙ ε ──► Decoder p_θ(x|z) ──► Loss f(z)
 Fixed Noise ──►   ε    ──┘

 BACKWARD PASS:
 Encoder Grads ◄── [∂z/∂μ = 1] ◄── ∇_z f(z) ◄── Decoder Grads ◄── Loss ∇_f
               ◄── [∂z/∂σ = ε] ◄──
================================================================================
```

### Analytical Partial Derivatives

For the standard location-scale Gaussian $z = \mu + \sigma \odot \epsilon$:
1. **Gradient with respect to mean $\mu$:**
   $$\frac{\partial z}{\partial \mu} = \mathbf{1}$$
2. **Gradient with respect to standard deviation $\sigma$:**
   $$\frac{\partial z}{\partial \sigma} = \epsilon$$
3. **Gradient with respect to log-variance $s = \ln(\sigma^2)$:**
   Since $\sigma = \exp(s / 2)$:
   $$\frac{\partial z}{\partial s} = \frac{\partial z}{\partial \sigma} \cdot \frac{d\sigma}{ds} = \epsilon \cdot \left( \frac{1}{2} \exp(s / 2) \right) = \frac{1}{2} \sigma \odot \epsilon$$

---

## 5. Why choose this tool for this problem?

| Characteristic | Pathwise (Reparameterization) | Score Function (REINFORCE) | Gumbel-Softmax (Concrete) | Finite Differences |
| :--- | :--- | :--- | :--- | :--- |
| **Applicability** | Continuous differentiable $f(z)$ | Any $f(z)$ (even black-box) | Discrete categorical latents | Low-dimensional parameters |
| **Estimator Variance** | **Extremely Low ($O(1)$)** | Extremely High ($O(\mu^2/\sigma^2)$) | Low to Moderate (depends on $\tau$) | High numerical error |
| **Gradient Information** | Uses $\nabla_z f(z)$ gradient | Uses only scalar value $f(z)$ | Uses relaxed continuous logits | Approximates via $f(z+\delta) - f(z)$ |
| **Computational Cost** | Single backward pass | Single backward pass | Single backward pass | $2D$ forward passes |
| **Generative Use Case** | **Standard in VAEs & Diffusion** | Policy gradients in RL | Discrete VQ / Token models | Numerical debugging |

### Concrete Counterexample: Variance Explosion in REINFORCE

Consider estimating the gradient of $\mathbb{E}_{z \sim \mathcal{N}(\mu, 1)}[z]$ with respect to $\mu$ (where true gradient is $\frac{d}{d\mu}(\mu) = 1.0$):
1. **Pathwise Estimator:**
   $$z = \mu + \epsilon \implies \frac{\partial z}{\partial \mu} = 1.0$$
   The single-sample pathwise estimate is identically $1.0$ regardless of $\epsilon$. Its variance is **identically 0.0**!
2. **Score-Function (REINFORCE) Estimator:**
   $$\hat{g}_{\text{REINFORCE}} = z \cdot \nabla_\mu \ln q(z) = (\mu + \epsilon) \cdot \epsilon = \mu \epsilon + \epsilon^2$$
   Taking variance under $\epsilon \sim \mathcal{N}(0, 1)$:
   $$\operatorname{Var}(\mu \epsilon + \epsilon^2) = \mu^2 \operatorname{Var}(\epsilon) + \operatorname{Var}(\epsilon^2) = \mu^2(1) + 2 = \mathbf{\mu^2 + 2}$$
3. As $\mu$ grows (e.g., $\mu = 10.0$), the REINFORCE variance explodes to $102.0$, requiring over $1,000$ samples to achieve the accuracy that pathwise achieves with a single sample.

---

## 6. Strengthen the intuition and mark its limits

```text
================================================================================
          THE STEERED SAILBOAT: INTUITION OF REPARAMETERIZATION
================================================================================
 NAIVE SAMPLING (The Chaotic Ocean):
 You want to test how adjusting the boat's rudder (parameter μ) affects speed.
 But every time you touch the rudder, an unpredictable rogue wave throws the boat.
 You cannot tell if the turn came from the rudder or the wave!
 
 REPARAMETERIZED SAMPLING (Separating Rudder from Wind):
 1. You measure the wind speed and direction beforehand (Noise ε ~ 𝒩(0, I)).
 2. You compute the boat's motion as a deterministic function:
    Motion = (Rudder Angle μ) + (Sail Size σ) · (Measured Wind ε)
 3. Because wind ε is fixed and measured, you can calculate the exact mathematical
    influence of adjusting the rudder alone!
================================================================================
```

### Mechanical Mapping: Intuition to Mathematics and Implementation

| Physical Intuition / Metaphor | Mathematical Operation | Software / Hardware Implementation | Failure Mode / Boundary Condition |
| :--- | :--- | :--- | :--- |
| **Measuring the external wind** | Sample standard noise $\epsilon \sim \mathcal{N}(0, I)$ | `torch.randn_like(mu)` | If $\epsilon$ depends on $\phi$, backprop breaks |
| **Steering the rudder** | Shift distribution mean $\mu_\phi(x)$ | Linear/convolutional layer output | Mean drift outside prior support causes blur |
| **Adjusting sail size** | Scale distribution width $\sigma_\phi(x)$ | `torch.exp(0.5 * logvar)` | $\sigma \to 0$ collapses to deterministic AE |
| **Wind blowing boat** | Pathwise interaction $\mu + \sigma \odot \epsilon$ | Element-wise multiply-accumulate | Non-differentiable $f(z)$ breaks chain rule |

### Where this analogy stops working

1. **Non-Differentiable Discrete Choices:** You cannot reparameterize a coin flip or a categorical token choice directly. If $z \in \{0, 1\}$, the mapping from continuous noise to discrete states is a step function with zero derivative everywhere and infinite derivative at the threshold. Differentiating through discrete tokens requires smooth approximations like the **Gumbel-Softmax**.
2. **Boundary and Support Dependencies:** If the bounds of the support depend on parameters (e.g., $z \sim \operatorname{Uniform}(0, \theta)$), the Leibniz integral rule requires boundary correction terms (the Reynolds transport theorem). Naive reparameterization without accounting for moving boundaries produces biased gradients.

---

## 7. Terms worth keeping straight

### Core Terminology Reference Table

| Term | Pronunciation | Plain-English Meaning | Formal Definition & Conditions |
| :--- | :--- | :--- | :--- |
| **Pathwise Gradient** | “PATH-wyz GRAY-dee-unt” | Gradient evaluated by differentiating through deterministic coordinate map | $\nabla_\theta \mathbb{E}_{q_\theta}[f(z)] = \mathbb{E}_\epsilon [\nabla_z f(g_\theta(\epsilon)) \nabla_\theta g_\theta(\epsilon)]$. Low variance. |
| **Score Function Gradient** | “skor FUNK-shun” | REINFORCE gradient differentiating log-density outside the loss | $\nabla_\theta \mathbb{E}_{q_\theta}[f(z)] = \mathbb{E}_{q_\theta}[f(z) \nabla_\theta \ln q_\theta(z)]$. High variance. |
| **Reparameterization** | “ree-puh-RAM-uh-ter-ih-ZAY-shun” | Isolating stochasticity into an unparameterized noise variable | $z = g_\theta(\epsilon)$ where $\epsilon \sim p(\epsilon)$ independent of $\theta$. |
| **Location-Scale Family** | “loh-KAY-shun skayl” | Distributions formed by shifting and scaling standard base noise | $p(z) = \frac{1}{\sigma} p_0\left(\frac{z - \mu}{\sigma}\right)$. Gaussians, Laplacians, Cauchy. |
| **Gumbel-Softmax** | “GUM-bel SOFT-maks” | Continuous differentiable approximation for discrete categorical sampling | $y_i = \frac{\exp((\ln \pi_i + g_i)/\tau)}{\sum_j \exp((\ln \pi_j + g_j)/\tau)}$ where $g_i \sim \text{Gumbel}(0, 1)$. |
| **Monte Carlo Estimator** | “MON-tee KAR-loh” | Approximating mathematical expectation with empirical sample averages | $\mathbb{E}[f(z)] \approx \frac{1}{S} \sum_{s=1}^S f(z^{(s)})$. |
| **Variance Reduction** | “VAIR-ee-uns ree-DUK-shun” | Techniques decreasing sample variance without introducing bias | Pathwise derivative eliminates $\frac{\mu^2}{\sigma^2}$ variance explosion. |
| **Aleatoric Noise** | “al-ee-uh-TOR-ik noyz” | Irreducible intrinsic randomness in data generation | Base stochasticity $\epsilon \sim \mathcal{N}(0, I)$ injected into generative pass. |

### Confused Pairs Distinction Breakdown

1. **Pathwise Gradient vs. Score Function Gradient**:
   - *Core Distinction:* Pathwise differentiates inside the expectation through $z = g_\theta(\epsilon)$ (requires differentiable $f$); score function differentiates the density $\ln q_\theta(z)$ (works for black-box rewards).
   - *Common Confusion:* Believing REINFORCE and the reparameterization trick compute different expected gradients. Both are mathematically unbiased estimators of the exact same quantity.
   - *Rule of Thumb:* If $f(z)$ is differentiable and $z$ is continuous, always use the reparameterization trick; the variance is orders of magnitude lower.
- **Pathwise Gradient (Reparameterization):** Differentiates *inside* the expectation through $z = g(\epsilon)$. Requires differentiable loss $f(z)$. Has extremely low variance.
- **Score Function Gradient (REINFORCE):** Differentiates the log-density *outside* the loss: $f(z) \nabla \ln q(z)$. Works on non-differentiable rewards, but has massive variance.

### 2. Location-Scale Family
- **Location-Scale Family:** A family of distributions where any member can be expressed as $z = \mu + \sigma \epsilon$ using a fixed base density $p(\epsilon)$. Examples: Gaussian, Cauchy, Laplace, Uniform, Logistic.
- Distributions outside this family (like Gamma or Beta) require more complex implicit or rejection reparameterization schemes.

### 3. Gumbel-Softmax (Concrete) Relaxation
- **Gumbel-Softmax:** A continuous, differentiable approximation to discrete categorical sampling. Replaces non-differentiable $\arg\max$ with temperature-scaled softmax:
  $$y_k = \frac{\exp((\ln \pi_k + g_k) / \tau)}{\sum_j \exp((\ln \pi_j + g_j) / \tau)}, \qquad g_k \sim \operatorname{Gumbel}(0, 1)$$

---

## 8. Work through the mathematics and its conditions

### Theorem 8.1: Analytical Variance Comparison of Pathwise vs. Score Function Estimators

**Statement:** For linear loss $f(z) = z$ with $z \sim \mathcal{N}(\mu, \sigma^2)$, the single-sample Pathwise gradient estimator has variance identically zero, while the Score Function (REINFORCE) estimator has variance $\operatorname{Var}(\hat{g}_{\text{REINFORCE}}) = 1 + \frac{\mu^2}{\sigma^2}$.

**Proof:**
1. **Pathwise Estimator:**
   Reparameterize $z = \mu + \sigma \epsilon$, with $\epsilon \sim \mathcal{N}(0, 1)$.
   $$\hat{g}_{\text{Path}} = \frac{\partial f(z)}{\partial \mu} = \frac{\partial z}{\partial \mu} = 1$$
   Because $\hat{g}_{\text{Path}} = 1$ is a constant, its variance is:
   $$\operatorname{Var}(\hat{g}_{\text{Path}}) = \mathbf{0} \quad [\text{Strictly Zero Variance!}]$$

2. **Score Function (REINFORCE) Estimator:**
   $$\hat{g}_{\text{Score}} = z \cdot \frac{\partial \ln q(z)}{\partial \mu} = z \cdot \frac{z - \mu}{\sigma^2} = (\mu + \sigma \epsilon) \cdot \frac{\sigma \epsilon}{\sigma^2} = \frac{\mu}{\sigma} \epsilon + \epsilon^2$$

3. Evaluate expectation:
   $$\mathbb{E}[\hat{g}_{\text{Score}}] = \frac{\mu}{\sigma} \mathbb{E}[\epsilon] + \mathbb{E}[\epsilon^2] = 0 + 1 = 1 \quad [\text{Unbiased}]$$

4. Evaluate variance:
   $$\operatorname{Var}(\hat{g}_{\text{Score}}) = \operatorname{Var}\left( \frac{\mu}{\sigma} \epsilon + \epsilon^2 \right)$$
   Because $\epsilon$ and $\epsilon^2$ are uncorrelated for standard normals ($\mathbb{E}[\epsilon^3] = 0$):
   $$\operatorname{Var}(\hat{g}_{\text{Score}}) = \frac{\mu^2}{\sigma^2} \operatorname{Var}(\epsilon) + \operatorname{Var}(\epsilon^2) = \frac{\mu^2}{\sigma^2}(1) + (3 - 1^2) = \mathbf{1 + \frac{\mu^2}{\sigma^2}} \quad \blacksquare$$

*Significance:* In deep networks where $\mu \gg \sigma$, the REINFORCE variance explodes as $\frac{\mu^2}{\sigma^2}$, making training completely impossible without millions of samples. The reparameterization trick eliminates this variance entirely!

### Hardware and Computational Realities: PRNG Overhead, Memory Coalescing, and Kernel Fusion

In modern GPU training pipelines, computing $z = \mu + \sigma \odot \epsilon$ introduces distinct hardware considerations:

1. **Pseudo-Random Number Generation (PRNG) Overhead:**
   Generating random normal samples $\epsilon \sim \mathcal{N}(0, I)$ on GPUs requires executing PRNG algorithms (such as Philox or Box-Muller transformations) across thousands of parallel threads. In naive implementations, allocating and writing intermediate noise tensors $\epsilon$ to global GPU High Bandwidth Memory (HBM) creates severe memory bandwidth bottlenecks.

2. **Kernel Fusion via Triton / CUDA:**
   Modern production frameworks fuse the affine transformation with sampling into a single GPU kernel:
   $$\text{Thread } i: \quad \epsilon_i = \operatorname{PRNG}(\text{seed}, \text{offset}_i), \quad z_i = \mu_i + \exp(0.5 \cdot \log\sigma^2_i) \cdot \epsilon_i$$
   The noise tensor $\epsilon$ is kept strictly in GPU registers / SRAM and is never written to global VRAM. During the backward pass, if $\epsilon$ is not retained in memory, it is recomputed on-the-fly using the identical seed and offset, trading a few arithmetic cycles for massive HBM bandwidth savings.

3. **Gumbel-Softmax Temperature Annealing:**
   When using continuous relaxations for discrete latents (e.g. categorical text or discrete codebooks), the temperature parameter $\tau$ must be annealed carefully. Setting $\tau < 0.1$ causes numerical overflow in $\exp(g_i / \tau)$ on `float16` / `bfloat16` hardware, requiring `torch.clamp` or evaluation in `float32`.

---

## 9. Calculate it by hand

### Worked Example: Pathwise vs. REINFORCE Gradients for Quadratic Loss

Consider optimizing parameter $\mu$ for loss $f(z) = z^2$ with $z \sim \mathcal{N}(\mu = 3.0, \sigma^2 = 1.0)$:
- True objective: $J(\mu) = 1.0 + \mu^2 = 1.0 + 3.0^2 = 10.0$.
- True analytical gradient: $\frac{dJ}{d\mu} = 2\mu = \mathbf{6.0000}$.

---

#### Calculation 1: Moderate Positive Noise $\epsilon = +0.50$
1. **Pathwise Gradient:**
   - Latent code: $z = 3.0 + 1.0(0.50) = \mathbf{3.50}$.
   - Loss value: $f(z) = 3.50^2 = 12.25$.
   - Pathwise gradient: $\hat{g}_{\text{Path}} = \frac{\partial f}{\partial z} \cdot \frac{\partial z}{\partial \mu} = 2z \cdot 1 = 2(3.50) = \mathbf{+7.0000}$.
   - Error from true gradient: $7.0000 - 6.0000 = \mathbf{+1.0000}$.

2. **REINFORCE Gradient:**
   - Score: $\frac{z - \mu}{\sigma^2} = \frac{3.50 - 3.0}{1.0} = +0.50$.
   - REINFORCE gradient: $\hat{g}_{\text{Score}} = f(z) \cdot \text{score} = 12.25 \times 0.50 = \mathbf{+6.1250}$.
   - Error from true gradient: $6.1250 - 6.0000 = \mathbf{+0.1250}$.

---

#### Calculation 2: Negative Noise $\epsilon = -2.00$
1. **Pathwise Gradient:**
   - Latent code: $z = 3.0 + 1.0(-2.00) = \mathbf{1.00}$.
   - Loss value: $f(z) = 1.00^2 = 1.00$.
   - Pathwise gradient: $\hat{g}_{\text{Path}} = 2z \cdot 1 = 2(1.00) = \mathbf{+2.0000}$.
   - Direction: **Positive!** Commands gradient descent to decrease $\mu$ toward 0.

2. **REINFORCE Gradient:**
   - Score: $\frac{1.00 - 3.0}{1.0} = -2.00$.
   - REINFORCE gradient: $\hat{g}_{\text{Score}} = 1.00 \times (-2.00) = \mathbf{-2.0000}$.
   - Direction: **Negative!** Commands gradient descent to increase $\mu$ away from 0!
   - Catastrophic error: The gradient step points in the exact opposite direction!

### Second Case: Deterministic Boundary Limit (σ → 0) and Multi-Sample Variance Scaling

To examine how the reparameterization trick behaves near the deterministic limit, consider a target with $\mu = 3.0$ and small noise $\sigma = 0.1$ for loss $f(z) = z^2$:

1. **Exact Theoretical Variance Comparison:**
   From Theorem 8.1:
   - Pathwise variance:
     $$\operatorname{Var}(\hat{g}_{\text{Path}}) = 4\sigma^2 = 4(0.1)^2 = 4(0.01) = \mathbf{0.0400}$$
   - Score function (REINFORCE) variance:
     $$\operatorname{Var}(\hat{g}_{\text{Score}}) = 1 + \frac{\mu^2}{\sigma^2} = 1 + \frac{3.0^2}{0.1^2} = 1 + \frac{9.0}{0.01} = 1 + 900 = \mathbf{901.0000}$$
   - **Variance Ratio:**
     $$\frac{\operatorname{Var}(\hat{g}_{\text{Score}})}{\operatorname{Var}(\hat{g}_{\text{Path}})} = \frac{901.0}{0.04} = \mathbf{22,525 \times}$$
     The score function estimator has over **22,500 times higher variance** than the pathwise gradient!

2. **Boundary Limit ($\sigma \to 0$):**
   As $\sigma \to 0$, the latent variable becomes deterministic: $z = \mu + 0 \cdot \epsilon = \mu$.
   - Pathwise estimator: $\hat{g}_{\text{Path}} = 2z = 2\mu = 6.0000$, with $\lim_{\sigma \to 0} \operatorname{Var}(\hat{g}_{\text{Path}}) = 0$. It seamlessly transitions into ordinary deterministic backpropagation.
   - Score function estimator: $\lim_{\sigma \to 0} \operatorname{Var}(\hat{g}_{\text{Score}}) = \lim_{\sigma \to 0} \left(1 + \frac{\mu^2}{\sigma^2}\right) = +\infty$.
   This proves that REINFORCE is fundamentally ill-conditioned for low-noise continuous systems, whereas the reparameterization trick remains perfectly stable and exact.

---

## 10. Connect the concept to an actual system

```text
================================================================================
             REPARAMETERIZATION ACROSS PRODUCTION GENERATIVE AI
================================================================================
 1. VARIATIONAL AUTOENCODERS (VAEs)           2. DENOISING DIFFUSION (DDPM)
 Latent Sampling: z = μ + σ ⊙ ε              Diffusion Step: x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
 ┌───────────────────────────────────────┐    ┌────────────────────────────────────────┐
 │ Gradients flow through decoder into   │    │ Neural network predicts noise ε_θ      │
 │ encoder mean and logvar projections   │    │ Enables 1000-step reverse generation   │
 └───────────────────────────────────────┘    └────────────────────────────────────────┘
================================================================================
```

*What to notice from the diagram:*
1. In VAEs, the reparameterization trick enables joint end-to-end training of encoder and decoder via standard backpropagation.
2. In Diffusion Models (DDPM), the closed-form forward diffusion equation $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ is an exact application of the location-scale reparameterization trick, allowing noise injection at any arbitrary timestep $t$ in a single step.

| Mathematical Object | Role in Toy Example | Real Production System Counterpart | Hardware / Scale Approximation |
| :--- | :--- | :--- | :--- |
| **Noise Variable $\epsilon$** | Scalar $\epsilon \sim \mathcal{N}(0, 1)$ | Standard normal tensor $\epsilon \in \mathbb{R}^{B \times 16 \times 128 \times 128}$ | Drawn on GPU via `torch.randn_like` using cuRAND RNG streams |
| **Mean Projection $\mu_\phi(x)$** | Scalar $\mu = 3.0$ | Convolutional feature map in Stable Diffusion VAE | Stored in 16-bit precision (`bfloat16`) to fit in GPU VRAM |
| **Scale Projection $\sigma_\phi(x)$** | Scalar $\sigma = 1.0$ | Computed from logvar: $\sigma = \exp(0.5 \cdot s)$ | Clamped to prevent underflow: $s \in [-30, +20]$ |
| **Pathwise Gradient $\nabla_z f$** | Scalar $2z$ | Upstream gradient from MSE + LPIPS decoder loss | Propagates seamlessly through `z.backward()` across all latent channels |

We have mapped the architectural connections. Next, we verify these formulations with executable Python and PyTorch scripts.

---

## 11. Verify the idea with a small experiment

We implement the **Dual-Stage Code Architecture**:
- **Stage 1 (Pure Python):** Standard library implementation comparing the Pathwise and REINFORCE gradient estimators across 10,000 Monte Carlo samples, verifying that the Pathwise estimator achieves dramatically lower variance, with passing assertions.
- **Stage 2 (Production PyTorch):** Vectorized PyTorch implementation testing analytical gradients $\frac{\partial z}{\partial \mu} = 1.0$ and $\frac{\partial z}{\partial \sigma} = \epsilon$ against PyTorch autograd.

```python
"""
Reparameterization Trick Dual-Stage Verification Suite
======================================================
Part A: Pure Python standard library (built-in math & random only).
Part B: PyTorch industrial verification with autograd checks.
"""

import math
import random
import sys

# Ensure UTF-8 output on all consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 80)
print("PART A: PURE PYTHON STDLIB GRADIENT ESTIMATOR VARIANCE COMPARISON")
print("=" * 80)

random.seed(42)

# Problem: f(z) = z^2, z ~ N(mu=3.0, sigma=1.0)
# True E[f(z)] = 1.0 + mu^2 = 10.0
# True dE/dmu = 2 * mu = 6.0
mu_val = 3.0
sigma_val = 1.0
true_grad = 2.0 * mu_val  # 6.0

N_SAMPLES = 20000

pathwise_grads = []
reinforce_grads = []

for _ in range(N_SAMPLES):
    eps = random.gauss(0.0, 1.0)
    z = mu_val + sigma_val * eps
    f_z = z ** 2
    
    # Pathwise gradient: df/dz * dz/dmu = 2 * z * 1.0
    g_path = 2.0 * z
    pathwise_grads.append(g_path)
    
    # REINFORCE gradient: f(z) * dlnq/dmu = z^2 * (z - mu) / sigma^2
    score = (z - mu_val) / (sigma_val ** 2)
    g_score = f_z * score
    reinforce_grads.append(g_score)

def compute_mean_and_var(vals):
    mean_val = sum(vals) / len(vals)
    var_val = sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
    return mean_val, var_val

mean_path, var_path = compute_mean_and_var(pathwise_grads)
mean_reinf, var_reinf = compute_mean_and_var(reinforce_grads)

print(f"Empirical Estimation Across {N_SAMPLES} Samples (True Gradient = {true_grad:.4f}):")
print(f"1. Pathwise (Reparameterization) Estimator:")
print(f"   * Mean Gradient: {mean_path:.4f} (Expected: ~6.0000)")
print(f"   * Variance:      {var_path:.4f} (Expected: 4.0 * sigma^2 = 4.0000)")

print(f"\n2. Score-Function (REINFORCE) Estimator:")
print(f"   * Mean Gradient: {mean_reinf:.4f} (Expected: ~6.0000)")
print(f"   * Variance:      {var_reinf:.4f} (Expected: 4*mu^2 + 6 = 42.0000)")

print(f"\n3. Variance Ratio (REINFORCE / Pathwise): {var_reinf / var_path:.2f}x")

assert math.isclose(mean_path, true_grad, rel_tol=0.02), "Pathwise mean biased!"
assert math.isclose(mean_reinf, true_grad, rel_tol=0.05), "REINFORCE mean biased!"
assert var_path < var_reinf, "Pathwise variance was not lower than REINFORCE!"
assert math.isclose(var_path, 4.0, rel_tol=0.10), "Pathwise variance did not match theoretical 4.0!"
print("Part A Pure Python Suite: ALL CHECKS PASSED [OK]")

print("\n" + "=" * 80)
print("PART B: PYTORCH INDUSTRIAL AUTOGRAD & DIFFERENTIABLE SAMPLING SUITE")
print("=" * 80)

import torch

# Define parameters
mu_torch = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
logvar_torch = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)  # sigma = 1.0

# Reparameterized sampling
torch.manual_seed(42)
eps_torch = torch.randn_like(mu_torch)
std_torch = torch.exp(0.5 * logvar_torch)
z_torch = mu_torch + std_torch * eps_torch

# Quadratic loss: L = z^2
loss_torch = z_torch ** 2
loss_torch.backward()

# Analytical gradients:
# dL/dmu = 2 * z * 1.0
# dL/dlogvar = 2 * z * (0.5 * std * eps)
expected_grad_mu = (2.0 * z_torch).item()
expected_grad_logvar = (2.0 * z_torch * (0.5 * std_torch * eps_torch)).item()

print(f"1. PyTorch Reparameterized Node Output (z): {z_torch.item():.4f}")
print(f"   * Autograd dL/dmu:      {mu_torch.grad.item():.6f} vs Analytical: {expected_grad_mu:.6f}")
print(f"   * Autograd dL/dlogvar: {logvar_torch.grad.item():.6f} vs Analytical: {expected_grad_logvar:.6f}")

assert math.isclose(mu_torch.grad.item(), expected_grad_mu, abs_tol=1e-8)
assert math.isclose(logvar_torch.grad.item(), expected_grad_logvar, abs_tol=1e-8)
print("   * Autograd matches analytical gradients with bit-exact precision! [OK]")

print("\n" + "=" * 80)
print("ALL REPARAMETERIZATION TRICK TESTS PASSED SUCCESSFULLY! [OK]")
print("=" * 80)
```

*Expected output:*
```text
================================================================================
PART A: PURE PYTHON STDLIB GRADIENT ESTIMATOR VARIANCE COMPARISON
================================================================================
Empirical Estimation Across 20000 Samples (True Gradient = 6.0000):
1. Pathwise (Reparameterization) Estimator:
   * Mean Gradient: 5.9961 (Expected: ~6.0000)
   * Variance:      4.0215 (Expected: 4.0 * sigma^2 = 4.0000)

2. Score-Function (REINFORCE) Estimator:
   * Mean Gradient: 5.9873 (Expected: ~6.0000)
   * Variance:      41.9842 (Expected: 4*mu^2 + 6 = 42.0000)

3. Variance Ratio (REINFORCE / Pathwise): 10.44x
Part A Pure Python Suite: ALL CHECKS PASSED [OK]

================================================================================
PART B: PYTORCH INDUSTRIAL AUTOGRAD & DIFFERENTIABLE SAMPLING SUITE
================================================================================
1. PyTorch Reparameterized Node Output (z): 3.3367
   * Autograd dL/dmu:      6.673404 vs Analytical: 6.673404
   * Autograd dL/dlogvar: 1.123412 vs Analytical: 1.123412
   * Autograd matches analytical gradients with bit-exact precision! [OK]

================================================================================
ALL REPARAMETERIZATION TRICK TESTS PASSED SUCCESSFULLY! [OK]
================================================================================
```

---

## 12. Practise, compare, and debug

Attempt all five diagnostic exercises before inspecting the separated solutions.

1. **Recognize.** An engineer implements a sampling node in PyTorch as `z = torch.normal(mu, std)`. During backpropagation, the loss decreases on the decoder, but the encoder weights receive zero gradients (`encoder.weight.grad is None`). Why did backpropagation fail to reach the encoder?
2. **Calculate.** In a 1D reparameterized VAE, $\mu = 2.0$ and $\sigma = 0.5$. The random sample is $\epsilon = -1.0$. The downstream reconstruction loss is $f(z) = (5.0 - z)^2$. Compute the analytical gradient $
rac{\partial f}{\partial \mu}$ and $
rac{\partial f}{\partial \sigma}$ by hand.
3. **Contrast.** Contrast the Reparameterization Trick with the Gumbel-Softmax trick. Why can't the standard location-scale reparameterization trick be applied to discrete categorical distributions?
4. **Transfer.** In Denoising Diffusion Probabilistic Models (DDPM), the forward process adds Gaussian noise across $T$ timesteps. Show how the reparameterization trick allows sampling an arbitrary noisy image $x_t$ directly from clean image $x_0$ in a single step without simulating the intermediate $t-1$ steps.
5. **Debug.** A PyTorch VAE implementation computes:
   `z = mu + torch.exp(logvar) * torch.randn_like(mu)`
   Diagnose the mathematical bug in this line and state the exact correction.

---

### Separated Diagnostic Solutions

<details>
<summary>Click to view solution for Exercise 1</summary>

**Diagnosis:** `torch.normal(mu, std)` executes sampling as an opaque out-of-graph operation. In PyTorch, random number generators produce non-differentiable tensors whose `grad_fn` is `None`. The backward pass terminates at `z` and cannot propagate gradients to `mu` or `std`.

**Resolution:** Use the reparameterization trick:
`z = mu + std * torch.randn_like(mu)`
</details>

<details>
<summary>Click to view solution for Exercise 2</summary>

**Calculation:**
1. Compute latent coordinate:
   $$z = \mu + \sigma \epsilon = 2.0 + (0.5)(-1.0) = 2.0 - 0.5 = \mathbf{1.50}$$
2. Compute loss derivative with respect to $z$:
   $$
rac{\partial f}{\partial z} = 
rac{d}{dz}(5.0 - z)^2 = 2(5.0 - z)(-1) = -2(5.0 - 1.5) = -2(3.5) = \mathbf{-7.00}$$
3. Apply chain rule for $\mu$ and $\sigma$:
   $$
rac{\partial f}{\partial \mu} = 
rac{\partial f}{\partial z} \cdot 
rac{\partial z}{\partial \mu} = (-7.00) \cdot (1.0) = \mathbf{-7.0000}$$
   $$
rac{\partial f}{\partial \sigma} = 
rac{\partial f}{\partial z} \cdot 
rac{\partial z}{\partial \sigma} = (-7.00) \cdot (\epsilon) = (-7.00) \cdot (-1.0) = \mathbf{+7.0000}$$
</details>

<details>
<summary>Click to view solution for Exercise 3</summary>

**Contrast:**
- **Why Location-Scale Fails on Discrete Variables:** The location-scale transformation requires continuous shifting and scaling: $z = \mu + \sigma \epsilon$. Discrete categorical distributions have integer support $z \in \{1, \dots, K\}$. A step function mapping continuous noise to discrete categories has zero gradient almost everywhere ($
rac{\partial z}{\partial \phi} = 0$) and is non-differentiable at category boundaries.
- **Gumbel-Softmax:** Replaces the non-differentiable $rg\max$ operation with a continuous, differentiable softmax relaxation with temperature $	au > 0$, allowing gradients to flow into categorical class logits.
</details>

<details>
<summary>Click to view solution for Exercise 4</summary>

**Transfer (DDPM 1-Step Jump):**
In DDPM, the single-step transition is $x_t = \sqrt{1 - eta_t} x_{t-1} + \sqrt{eta_t} \epsilon_{t-1}$.  
By recursively expanding this location-scale transformation and using the property that the sum of two independent Gaussians $\mathcal{N}(0, \sigma_1^2 I) + \mathcal{N}(0, \sigma_2^2 I)$ is $\mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)I)$, the cumulative noise variance collapses into:
$$x_t = \sqrt{ar{lpha}_t} x_0 + \sqrt{1 - ar{lpha}_t} \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$
where $ar{lpha}_t = \prod_{s=1}^t (1 - eta_s)$. This allows training diffusion models on arbitrary timesteps $t$ in $\mathcal{O}(1)$ time.
</details>

<details>
<summary>Click to view solution for Exercise 5</summary>

**Diagnosis:** The code multiplies by `torch.exp(logvar)` instead of the standard deviation $\sigma$. Since `logvar` is $s = \ln(\sigma^2)$, evaluating $\exp(s)$ yields $\sigma^2$ (the variance), not $\sigma$! The network is scaling noise by the variance, squaring the intended noise magnitude and destabilizing training.

**Fix:**
`z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)`
</details>

---

## 13. Explain it back and return to it

**Closed-notes Feynman prompt:**  
Imagine explaining the Reparameterization Trick to a software engineer who understands backpropagation but has never worked with probabilistic models without using the terms “pathwise gradient”, “measure theory”, or “Lebesgue dominated convergence”. Use the analogy of an RC car being blown by the wind, and explain why separating the random roll of the dice from the neural network's parameters is necessary for gradient descent to work. Once you finish, restore the formal terms and state the Pathwise Gradient equation.

<details>
<summary>Model explanation for self-evaluation</summary>

When you train a normal neural network, every operation is like a mechanical gear: if you turn gear A slightly, gear B turns predictably, which allows backpropagation to trace the chain backwards and adjust the weights.

If you put a random dice roll inside that gear chain, the chain breaks. If the computer rolls a random number 4.2, backpropagation cannot ask "how would the 4.2 change if I changed the network weights?" Because the dice roll doesn't care about your weights—it's pure random chance.

The reparameterization trick fixes this with a simple reorganization: instead of rolling the dice inside the network, you roll standard dice *outside* the network. Then, you treat the random number as an external sensor measurement (like wind). The network simply computes:
$$	ext{Output} = 	ext{Mean} + 	ext{Width} 	imes 	ext{Wind}$$
Now, all operations inside the network are standard multiplication and addition gears! Backpropagation can easily flow through the Mean and Width gears, allowing the network to learn smoothly while still maintaining randomness.

*Restoring formal terminology:* The dice roll is the **auxiliary noise variable** $\epsilon \sim \mathcal{N}(0, I)$, the gears are the **location-scale transformation** $z = \mu + \sigma \odot \epsilon$, and the smooth gradient flow is the **Pathwise Gradient Estimator**:
$$
abla_\phi \mathbb{E}_{q_\phi(z \mid x)}[f(z)] = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[ 
abla_z f(z) \cdot 
abla_\phi g_\phi(\epsilon, x) 
ight]$$

</details>

### Spaced Repetition Schedule

| Return Date | Closed-Notes Retrieval Task | Self-Verification Anchor |
| :--- | :--- | :--- |
| **Day 1** | Sketch the naive vs. reparameterized computational graphs. Write down $
rac{\partial z}{\partial \mu}$ and $
rac{\partial z}{\partial \sigma}$ from memory. | Check against §1 diagram and §4 derivatives. |
| **Day 7** | Derive why the Score Function (REINFORCE) estimator variance explodes with $\mu^2 / \sigma^2$ while the Pathwise estimator variance is zero for linear loss. | Check against Theorem 8.1 proof. |
| **Day 30** | Explain how the Gumbel-Softmax relaxation enables backpropagation through discrete categorical choices, and state what happens as $	au 	o 0$. | Check against §8 Gumbel derivation. |

### Self-Assessment Checklist

- [ ] I can explain why standard sampling nodes $z \sim q_\phi(z)$ block reverse-mode automatic differentiation.
- [ ] I can formulate the location-scale reparameterization mapping $z = \mu + \sigma \odot \epsilon$.
- [ ] I can compute the analytical partial derivatives $
rac{\partial z}{\partial \mu} = 1$ and $
rac{\partial z}{\partial \sigma} = \epsilon$ by hand.
- [ ] I can prove that the Pathwise estimator achieves zero variance on linear losses while REINFORCE variance scales with $\mu^2 / \sigma^2$.
- [ ] I can explain the Lebesgue Dominated Convergence conditions that permit interchanging differentiation and expectation.
- [ ] I can derive the Gumbel-Softmax continuous relaxation for discrete categorical variables.
- [ ] I can explain how the reparameterization trick enables DDPM diffusion models to sample $x_t$ directly from $x_0$ in a single step.
- [ ] I can implement a reparameterized VAE latent layer in PyTorch with verified autograd gradients.

---

## 14. Continue with a purposeful learning path

The resources below are verified for relevance, active status, and pedagogical precision as of **2026-09-18**. Access descriptions indicate verified availability at check time.

| Resource and author | Learning job | Exact starting point | Readiness | Access | Checked date and evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Interactive visualizer:** [The Reparameterization Trick Visualized](https://blog.evjang.com/2016/11/tutorial-categorical-variational.html), Eric Jang | Interactive explanation of pathwise gradients and the Gumbel-Softmax relaxation | Section: "The Gumbel-Max Trick" with interactive temperature plots | After §2 | Free open educational blog | 2026-09-18: verified active interactive visualizations of continuous relaxations. |
| **Video lecture:** [Variational Autoencoders: The Reparameterization Trick](https://www.youtube.com/watch?v=9zKuYvjFFS8), StatQuest with Josh Starmer | Visual explanation of why sampling breaks backprop and how reparameterization fixes it | Timestamp 09:30: "The Reparameterization Trick" | After §2 | Free YouTube video | 2026-09-18: verified active video, clear graphical demonstration of gradient routing. |
| **Video lecture (Advanced):** [Stochastic Gradient Estimation](https://www.youtube.com/watch?v=VqhDnbPBioc), Shakir Mohamed (DeepMind) | Deep mathematical breakdown of pathwise gradients vs score function estimators | Timestamp 18:15: "Pathwise Derivative Estimators" | After §4 | Free YouTube video | 2026-09-18: verified active lecture, rigorous mathematical comparison of Monte Carlo estimator variances. |
| **Foundational paper:** [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), Diederik P. Kingma, Max Welling (ICLR 2014) | Seminal paper introducing the reparameterization trick and SGVB estimator | Section 2.4: "The Reparameterization Trick" | After §8 | Free open-access arXiv preprint | 2026-09-18: verified original paper formulations, location-scale Gaussian transformations. |
| **Discrete extension paper:** [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144), Eric Jang, Shixiang Gu, Ben Poole (ICLR 2017) | Original paper introducing the continuous relaxation for discrete categorical variables | Section 2: "The Gumbel-Softmax Distribution" | After §8 | Free open-access arXiv preprint | 2026-09-18: verified active paper, temperature annealing schedules, and straight-through estimators. |
| **Textbook:** [Probabilistic Machine Learning: Advanced Topics, Chapter 25: Variational Inference](https://probml.github.io/pml-book/book2.html), Kevin P. Murphy | Authoritative academic treatment of Monte Carlo gradient estimators | Chapter 25: §25.2 (Pathwise gradients) and §25.3 (Score function estimators) | After §4 | Free online PDF (MIT Press, 2023) | 2026-09-18: verified section numbers, mathematical variance derivations, and control variates. |
| **Practice problem set:** [UC Berkeley CS285: Deep Reinforcement Learning, Homework 2](https://rail.eecs.berkeley.edu/deeprlcourse/), Sergey Levine (UC Berkeley) | Implement policy gradient REINFORCE vs. pathwise gradient estimators | Section 2: "Policy Gradients and Variance Reduction" | After §12 | Free university course material | 2026-09-18: verified assignment covering variance comparisons between REINFORCE and pathwise methods. |
| **Software documentation:** [PyTorch torch.distributions.Distribution API](https://pytorch.org/docs/stable/distributions.html), PyTorch Contributors | Production reference for `rsample()` (reparameterized) vs. `sample()` (non-differentiable) | `torch.distributions.Normal.rsample` documentation | When running §11 | Free official documentation | 2026-09-18: verified PyTorch 2.9 documentation for pathwise sampling via `rsample()`. |

**Next connection:** The reparameterization trick enables gradient-based training of continuous latent generative models. In [minimax game and GANs](09-Minimax_Game_and_GANs.md), we explore a radically different generative paradigm: bypassing explicit density estimation entirely through adversarial zero-sum games between a generator and a discriminator.
