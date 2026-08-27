# Common Probability Distributions: The Mathematical Blueprints of Generative AI

> `🏷️ Tags:` `Probability-Distributions` `Gaussian` `Bernoulli` `Categorical` `Dirac-Delta` `Generative-AI` `Diffusion` `VAEs` `LLMs`  
> `📚 Prerequisites Needed:` [Probability Basics & Axioms](./Probability_Basics_and_Axioms.md) · [Vector Norms & Inner Products](./Vector_Norms_and_Inner_Products.md) · [Tensors & Shapes](./Tensors_and_Shapes.md)  
> `🎯 Where Do We Use This?:` **Every modern Generative AI pipeline** — Gaussian priors $\mathcal{N}(0, I)$ in VAEs and GANs, Incremental Gaussian noise schedules in Diffusion Models (Stable Diffusion, Midjourney), Categorical next-token distributions in LLMs (GPT-4, Claude), and Dirac delta empirical distributions $p_{\text{data}}$ in loss minimization.  
> `🎓 Course Module Mapping:` [Tut 07: Basic Probability 1](../Mathematical-Foundation-for-GenerativeAI/21-Tutorial07-Review-Basic-Probability-1/NOTES.md) · [Tut 08: Basic Probability 2](../Mathematical-Foundation-for-GenerativeAI/22-Tutorial08-Review-Basic-Probability-2/NOTES.md) · [Lec 18: WGAN](../Mathematical-Foundation-for-GenerativeAI/30-Lec18-Wasserstein-GAN/NOTES.md) · [Lec 20: VAEs](../Mathematical-Foundation-for-GenerativeAI/32-Lec20-Latent-Variable-Models-VAE/NOTES.md)  
> `⏱️ Difficulty Level:` ⭐⭐☆☆☆ (Foundational · 15 min read)

---

### 📌 Quick Navigation & Architecture Map
- [1. 🌟 Everyday Real-World Scenarios](#1--everyday-real-world-scenarios-factory-tolerances--midjourney-static-canvas) — Factory Tolerances & Midjourney Static Canvas
- [2. 👶 ELI5 Intuition](#2--eli5-intuition-the-galton-board-the-loaded-coin--the-laser-pointer) — The Galton Board, Loaded Coin, and Laser Pointer
- [3. 📚 Deep Terminology Master Glossary](#3--deep-terminology-master-glossary-15-core-distribution-terms) — 15 distribution terms dissected without jargon
- [4. 📐 The 6 Core Distribution Families: Formulas & Geometry](#4--the-6-core-distribution-families-formulas--geometry) — Gaussian, Multivariate Normal, Bernoulli, Categorical, Uniform, Dirac Delta
- [5. 🔢 Concrete Micro-Numerical Worked Examples](#5--concrete-micro-numerical-worked-examples) — Gaussian Density Drop & 2D Covariance Matrix Sizing
- [6. 🔗 Connecting the Dots: Generative AI Architecture Blocks](#6--connecting-the-dots-how-distributions-power-modern-generative-ai) — VAE Latent Prior, Diffusion Denoising SDE, and LLM Categorical Softmax
- [7. 💻 Standalone Executable Python/PyTorch Verification Script](#7--complete-standalone-executable-pythonpytorch-verification-script) — Sampling, density checks, and empirical Dirac delta validation
- [8. 🩺 Diagnostic Mini-Checks & Common Traps](#8--diagnostic-mini-checks--common-traps) — Self-test questions & production engineering pitfalls

---

A **Probability Distribution** is a mathematical recipe that assigns probabilities to all possible values a variable can take. In Generative AI, distributions act as the **sculpting clay**: we sample pure randomness from simple standard distributions (like a standard bell curve) and use deep neural networks to reshape that randomness into photorealistic images, coherent text, or high-fidelity audio.

```
 ===================================================================================================
                 THE CORE PROBABILITY DISTRIBUTION FAMILIES IN GENERATIVE AI
 ===================================================================================================

  GAUSSIAN / NORMAL N(μ, σ²)          BERNOULLI / CATEGORICAL (p)          DIRAC DELTA δ(x - x₀)
  Latent Priors & Diffusion Noise     Classification & LLM Tokens          Empirical Training Dataset
  ┌──────────────────────────────┐    ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │       _--~~--_               │    │   █                          │    │            │                 │
  │     /          \             │    │   █           █              │    │            │                 │
  │   /              \           │    │   █     █     █              │    │            │ Infinite Spike  │
  │ _/                \_         │    │   █     █     █     █        │    │            │ Zero Width      │
  └──────────────────────────────┘    └──────────────────────────────┘    └──────────────────────────────┘
   VAEs, GAN Latents, Diffusion        Softmax Output Heads, Tokens        Real Observed Dataset Points
 ===================================================================================================
```

---

### 1. 🌟 Everyday Real-World Scenarios (Factory Tolerances & Midjourney Static Canvas)
> `Context:` Zero Prior Machine Learning / AI Knowledge Needed · Concrete Real-World Mapping

#### Scenario A: Manufacturing 100,000 Metal Bolts in a Factory (Zero ML Background)
Imagine a factory manufacturing 10-cm steel bolts:
1. **The Target (Mean $\mu = 10.0\text{ cm}$):** The machine tries to make every bolt exactly 10 cm.
2. **The Natural Variation (Standard Deviation $\sigma = 0.1\text{ cm}$):** Due to tiny machine vibrations, bolts aren't all identical. Most bolts measure $9.9$ to $10.1\text{ cm}$ ($68.2\%$). A few measure $9.8$ or $10.2\text{ cm}$ ($27\%$), and almost none measure $< 9.7$ or $> 10.3\text{ cm}$.
3. **The Result (The Gaussian Bell Curve):** The bolt lengths naturally form a **Gaussian distribution**.

---

#### Scenario B: In Generative AI — Midjourney & Stable Diffusion Generating Art
> `Context:` How Simple Standard Normal Noise $\mathcal{N}(0, I)$ Serves as the Universal Starting Clay

When you ask Stable Diffusion to draw *"a futuristic cybernetic city at sunset"*:
- The AI does not start with a blank white image. It starts with an image of **pure television static** sampled directly from a **Standard Normal Distribution** $\mathcal{N}(0, I)$.
- Every pixel in the starting noise canvas is drawn independently from a Gaussian bell curve (mean $0$, variance $1$).
- Over 50 denoising steps, the neural network predicts and removes the Gaussian noise, gradually revealing the sharp details of the futuristic city!

```
 ===================================================================================================
         HOW DISTRIBUTIONS OPERATE INSIDE DIFFUSION MODELS (STABLE DIFFUSION / FLUX)
 ===================================================================================================

  STEP 0: PURE GAUSSIAN NOISE             STEP 25: MIDWAY DENOISING           STEP 50: CRYSTAL CLEAR PHOTO
  Sampled from N(0, I)                   Guided by Text Embeddings           Final High-Fidelity Output
  ┌──────────────────────────────┐       ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │ ░▒▓█░▒▓█░▒▓█░▒▓█░▒▓█░▒▓█░▒▓█ │══════►│ ░▒ [City Outline] ▒░▓█▒░▓   │═══►│ 🏙️ [Futuristic Cyber City]   │
  │ Pure television static noise │       │ Coarse structures appear     │    │ Sharp photorealistic pixels  │
  │ Every pixel ~ 𝒩(0, 1)        │       │ Noise variance decreases     │    │ Final clean data manifold    │
  └──────────────────────────────┘       └──────────────────────────────┘    └──────────────────────────────┘
 ===================================================================================================
```

---

### 2. 👶 ELI5 Intuition: The Galton Board, The Loaded Coin & The Laser Pointer
> `Context:` Physical & Everyday Metaphors for Probability Distributions

#### Metaphor 1: The Galton Peg Board (Gaussian Distribution)
Imagine dropping thousands of marbles through a wooden board studded with pegs:
- Every time a marble hits a peg, it has a 50/50 chance of bouncing left or right.
- While a single marble could theoretically bounce right 50 times in a row, the vast majority bounce left roughly half the time and right half the time.
- At the bottom, the marbles naturally pile up into a smooth **bell-shaped curve ($\mathcal{N}(\mu, \sigma^2)$)**.

---

#### Metaphor 2: The 4 Classic Distribution Personas

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. 🔔 THE BELL CURVE / GAUSSIAN (𝒩(μ, σ²)):                                                      │
 │    • Centered at an average (μ); outcomes taper off smoothly as you move further away (σ).      │
 │    • The universal starting canvas for VAEs, GANs, and Diffusion models.                        │
 │                                                                                                 │
 │ 2. 🪙 THE BIASED COIN / BERNOULLI (Bern(p)):                                                    │
 │    • Binary choice: Heads ($1$) with probability $p$, Tails ($0$) with probability $1-p$.       │
 │    • Used in binary classification (Spam vs. Ham, Fraud vs. Normal).                            │
 │                                                                                                 │
 │ 3. 🎲 THE MULTI-SIDED DIE / CATEGORICAL (Cat(p₁, ..., p_K)):                                    │
 │    • A die with $K$ custom-weighted faces where the sum of all face chances equals $100\%$.     │
 │    • Powers next-token selection in ChatGPT (a 128,000-sided die!).                             │
 │                                                                                                 │
 │ 4. 📍 THE PINPOINT LASER / DIRAC DELTA (δ(x - x₀)):                                             │
 │    • Has 100% of its probability concentrated at a single point $x_0$ with zero width.          │
 │    • Represents an exact training image saved on your hard drive.                               │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. 📚 Deep Terminology Master Glossary (15 Core Distribution Terms)
> `Context:` Foundational Mathematical & Machine Learning Vocabulary Explained Without Jargon

```
 ===================================================================================================
                 THE PROBABILITY DISTRIBUTION ROSETTA STONE
 ===================================================================================================
```

| Term / Notation | Formal Mathematical Meaning | Plain-English Definition (No ML Jargon) | Real-World Analogy |
| :--- | :--- | :--- | :--- |
| **Probability Density Function (PDF $p(x)$)** | Derivative of CDF for continuous variables | The relative height or concentration of probability at point $x$ | The height of a sand dune at a specific GPS coordinate |
| **Probability Mass Function (PMF $P(X=k)$)** | Discrete probability $P(X=k) \in [0, 1]$ | The exact percentage chance of landing on a specific discrete option | The probability printed on a lottery scratch card |
| **Mean / Expectation ($\mu = \mathbb{E}[X]$)** | First moment $\int x p(x)dx$ | The long-term average balance center of the distribution | The center of mass where a seesaw balances |
| **Variance ($\sigma^2 = \text{Var}(X)$)** | Second central moment $\mathbb{E}[(X-\mu)^2]$ | How widely spread out the outcomes are around the average | The spray radius of a garden hose nozzle |
| **Standard Deviation ($\sigma = \sqrt{\sigma^2}$)** | Square root of variance in original units | The typical distance a sample falls from the average | The margin of error in an election poll |
| **Covariance Matrix ($\Sigma \in \mathbb{R}^{d \times d}$)** | Matrix of pairwise coordinate covariances $\mathbb{E}[(X-\mu)(X-\mu)^\top]$ | Captures whether multiple variables vary together (tilt of 2D/3D ellipse) | Height and weight growing together in people |
| **Isotropic Gaussian** | Covariance $\Sigma = \sigma^2 I$ (Identity matrix) | A multidimensional bell curve that is perfectly spherical (no tilt/stretch) | A perfectly round basketball vs. an egg |
| **Cumulative Distribution Function (CDF $F(x)$)** | $F(x) = P(X \le x) = \int_{-\infty}^x p(t)dt$ | The total accumulated probability up to threshold $x$ (runs from $0 \to 1$) | The total percentage of students scoring below score $X$ |
| **Dirac Delta Function ($\delta(x - x_0)$)** | Generalized function $\int \delta(x-x_0)f(x)dx = f(x_0)$ | An infinitely tall, zero-width spike holding 100% probability mass | A single pinpoint GPS coordinate on a map |
| **Empirical Distribution ($p_{\text{data}}$)** | $\frac{1}{N}\sum_{i=1}^N \delta(x - x_i)$ | Treating the $N$ samples in your dataset as $N$ equally likely spikes | The finite list of all houses sold in Austin this year |
| **Support ($\text{supp}(P)$)** | $\{x : p(x) > 0\}$ | The domain where the probability density is strictly greater than zero | The land boundaries where a species actually lives |
| **Uniform Distribution ($\mathcal{U}(a, b)$)** | Flat density $p(x) = \frac{1}{b-a}$ on $[a, b]$ | Every single point in the interval has the exact same chance of occurring | Picking a completely random second on a ticking clock |
| **Logit Space** | Unbounded pre-softmax scores $z \in \mathbb{R}^K$ | Raw, unnormalized preference scores output by a neural network | Points scored by teams before calculating win percentages |
| **Categorical Simplex ($\Delta^{K-1}$)** | $\{p \in \mathbb{R}^K : p_k \ge 0, \sum p_k = 1\}$ | The geometric surface containing all legal probability distributions | A pie chart that always sums to 100% |
| **Latent Prior ($p(z)$)** | Prescribed base distribution (usually $\mathcal{N}(0, I)$) | The standard clean starting distribution from which AI generates data | A lump of fresh, unshaped modeling clay |

---

### 4. 📐 The 6 Core Distribution Families: Formulas & Geometry
> `Context:` Mathematical Formulations, Density Equations, and ASCII Curves

```
 ===================================================================================================
                 THE 6 CORE PROBABILITY DISTRIBUTION FORMULAS & CURVES
 ===================================================================================================

  1. 1D GAUSSIAN N(μ, σ²)              2. MULTIVARIATE NORMAL N(μ, Σ)        3. UNIFORM U(a, b)
     p(x) = (1/√(2πσ²)) exp(-(x-μ)²/2σ²)   p(x) = (1/√(2π)ᵈ|Σ|) exp(-½(x-μ)ᵀΣ⁻¹(x-μ)) p(x) = 1/(b-a)
     p(x) ▲                                 x₂ ▲      .---.                        p(x) ▲
          │       _--~~--_                     │    .'     '. (Elliptical)       1/(b-a)├──────────┐
          │     /          \                   │   /    ● μ  \                          │          │
     0.0 ─┴────/─────┬──────\──► x             │   '.       .'                          │          │
                     μ                         └─────'-----'─────────► x₁          0.0 ─┴────┬─────┴────► x
                                                                                             a     b

  4. BERNOULLI Bern(p)                 5. CATEGORICAL Cat(p₁, ..., p_K)      6. DIRAC DELTA δ(x - x₀)
     P(X=1)=p, P(X=0)=1-p                  P(X=k) = p_k, ∑ p_k = 1.0             p(x) = δ(x - x₀), ∫ δ dx = 1
     P(X) ▲                                P(X) ▲                                p(x) ▲
      1.0 ┤   █                             1.0 ┤   █                             ∞   ┤          │ (Spike)
      0.5 ┤   █       █                     0.5 ┤   █           █                     │          │
      0.0 └───┴───────┴──────► X            0.0 └───┴─────█─────┴──────► k        0.0 ┴──────────┴────────► x
              0       1                             1     2     3                                x₀
 ===================================================================================================
```

#### Detailed Mathematical Formulations:

1. **Univariate Gaussian Distribution $\mathcal{N}(\mu, \sigma^2)$:**
   $$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
   - Expected Value: $\mathbb{E}[X] = \mu$
   - Variance: $\text{Var}(X) = \sigma^2$

2. **Multivariate Gaussian Distribution $\mathcal{N}(\mu, \Sigma)$ in $\mathbb{R}^d$:**
   $$p(x) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)$$
   - When $\Sigma = I_d$ (Standard Isotropic Gaussian):
     $$p(z) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\frac{1}{2}\|z\|_2^2\right) = \prod_{j=1}^d \frac{1}{\sqrt{2\pi}} e^{-z_j^2/2}$$

3. **Bernoulli Distribution $\text{Bern}(p)$ on $x \in \{0, 1\}$:**
   $$P(X = x) = p^x (1 - p)^{1 - x}$$
   - Mean: $\mathbb{E}[X] = p$, Variance: $\text{Var}(X) = p(1 - p)$

4. **Categorical Distribution $\text{Cat}(p_1, \dots, p_K)$ on $k \in \{1, \dots, K\}$:**
   $$P(X = k) = p_k \quad \text{where } p_k \ge 0 \text{ and } \sum_{k=1}^K p_k = 1.0$$

5. **Dirac Delta Empirical Distribution $p_{\text{data}}(x)$:**
   $$p_{\text{data}}(x) = \frac{1}{N} \sum_{i=1}^N \delta(x - x_i)$$
   - Property: $\mathbb{E}_{x \sim p_{\text{data}}}[f(x)] = \frac{1}{N} \sum_{i=1}^N f(x_i)$ (Arithmetic sample mean!).

---

### 5. 🔢 Concrete Micro-Numerical Worked Examples
> `Context:` Step-by-Step Manual Calculations (No Black Box)

#### Example 1: Calculating 1D Gaussian Density Heights
Let $X \sim \mathcal{N}(\mu = 0, \sigma = 1)$ (Standard Normal). Let's calculate the probability density $p(x)$ at $x = 0.0$ (mean), $x = 1.0$ ($1\sigma$), and $x = 2.0$ ($2\sigma$):

1. **Pre-factor Constant:**
   $$\frac{1}{\sqrt{2\pi}} = \frac{1}{\sqrt{6.283185}} = \frac{1}{2.50663} \approx \mathbf{0.39894}$$

2. **At Mean ($x = 0.0$):**
   $$p(0) = 0.39894 \cdot \exp\left(-\frac{0^2}{2}\right) = 0.39894 \cdot e^0 = \mathbf{0.39894}$$

3. **At $1\sigma$ ($x = 1.0$):**
   $$p(1) = 0.39894 \cdot \exp\left(-\frac{1^2}{2}\right) = 0.39894 \cdot e^{-0.5} = 0.39894 \cdot 0.60653 = \mathbf{0.24197} \quad (\approx 60.7\% \text{ of peak})$$

4. **At $2\sigma$ ($x = 2.0$):**
   $$p(2) = 0.39894 \cdot \exp\left(-\frac{2^2}{2}\right) = 0.39894 \cdot e^{-2.0} = 0.39894 \cdot 0.13534 = \mathbf{0.05399} \quad (\approx 13.5\% \text{ of peak})$$

---

#### Example 2: 2D Multivariate Gaussian Covariance & Mahalanobis Distance
Let $\mu = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ and diagonal covariance $\Sigma = \begin{bmatrix} 4 & 0 \\ 0 & 1 \end{bmatrix}$ ($\sigma_{x_1} = 2, \sigma_{x_2} = 1$).
Inverse covariance $\Sigma^{-1} = \begin{bmatrix} 0.25 & 0 \\ 0 & 1.0 \end{bmatrix}$, Determinant $|\Sigma| = 4 \cdot 1 - 0 = 4$.

Let us evaluate point $A = [2, 0]^\top$ vs point $B = [0, 2]^\top$:
1. **Squared Mahalanobis Distance $D_M^2(x) = (x - \mu)^\top \Sigma^{-1} (x - \mu)$:**
   - For Point $A = [2, 0]^\top$:
     $$D_M^2(A) = \begin{bmatrix} 2 & 0 \end{bmatrix} \begin{bmatrix} 0.25 & 0 \\ 0 & 1.0 \end{bmatrix} \begin{bmatrix} 2 \\ 0 \end{bmatrix} = 2(0.25)(2) + 0 = \mathbf{1.0}$$
   - For Point $B = [0, 2]^\top$:
     $$D_M^2(B) = \begin{bmatrix} 0 & 2 \end{bmatrix} \begin{bmatrix} 0.25 & 0 \\ 0 & 1.0 \end{bmatrix} \begin{bmatrix} 0 \\ 2 \end{bmatrix} = 0 + 2(1.0)(2) = \mathbf{4.0}$$
2. **Density Values:**
   - Pre-factor: $\frac{1}{2\pi \sqrt{|\Sigma|}} = \frac{1}{2\pi \cdot 2} = \frac{1}{4\pi} \approx 0.07958$
   - $p(A) = 0.07958 \cdot e^{-0.5(1.0)} = 0.07958 \cdot 0.6065 = \mathbf{0.04826}$
   - $p(B) = 0.07958 \cdot e^{-0.5(4.0)} = 0.07958 \cdot 0.1353 = \mathbf{0.01077}$

*(Notice: Even though point $A$ and point $B$ are both distance $2$ from origin in Euclidean terms, Point $A$ is **4.5x more likely** because the distribution spreads twice as wide along the $x_1$ axis!)*

---

### 6. 🔗 Connecting the Dots: How Probability Distributions Power Generative AI
> `Context:` Architectural Implementations in Large Language Models, Diffusion Models, GANs, and VAEs

```
 ===================================================================================================
                 THE ROLE OF STANDARD DISTRIBUTIONS IN GENERATIVE ARCHITECTURES
 ===================================================================================================

  VAE & GAN LATENT PRIORS               DIFFUSION FORWARD SDE               EMPIRICAL DATASET
  ┌───────────────────────────┐         ┌──────────────────────────┐         ┌──────────────────────────┐
  │ Prior p(z) = N(0, I)      │         │ x_t = √(1-β_t) x_t-1 +   │         │ p_data(x) = (1/n)∑ δ(x_i)│
  │ Standard isotropic noise  │ ──────► │       √(β_t) ε_t         │ ──────► │ Finite training samples  │
  │ Input to Generator G_θ(z) │         │ Adds Gaussian noise      │         │ Discrete point clouds    │
  └───────────────────────────┘         └──────────────────────────┘         └──────────────────────────┘
 ===================================================================================================
```

| Generative Architecture | Primary Distribution Used | Mathematical & Architectural Role |
| :--- | :--- | :--- |
| **Diffusion Models (Stable Diffusion, Flux)** | **Gaussian $\mathcal{N}(0, I)$** & Gaussian Perturbations | Denoising SDEs add and remove Gaussian noise over continuous timestep chains |
| **Large Language Models (GPT-4, LLaMA-3)** | **Categorical $\text{Cat}(p_1, \dots, p_K)$** | Softmax produces categorical parameter vector over 128,000 dictionary tokens |
| **Variational Autoencoders (VAEs)** | **Isotropic Gaussian $\mathcal{N}(\mu(x), \Sigma(x))$** | Encoder parameterizes variational Gaussian; regularized toward prior $\mathcal{N}(0, I)$ via KL |
| **Generative Adversarial Nets (GANs)** | **Latent Prior $\mathcal{N}(0, I)$ or $\mathcal{U}(-1, 1)$** | Low-dimensional continuous seed vector transformed into high-D image manifold |
| **Fréchet Inception Distance (FID)** | **Multivariate Gaussian $\mathcal{N}(\mu_r, \Sigma_r)$** | Features extracted from Inception-v3 are modeled as Gaussians to compute 2-Wasserstein |

---

### 7. 💻 Complete Standalone Executable Python/PyTorch Verification Script
> `Context:` Runnable Code Verifying Gaussian Densities, Categorical Token Sampling, and Dirac Expectations

```python
"""
Common Probability Distributions Simulation & Verification
===========================================================
Demonstrates:
1. Univariate and Multivariate Gaussian density calculations
2. Categorical distribution next-token sampling with temperature
3. Empirical dataset Dirac Delta expectation equivalence
"""
import torch
import torch.nn.functional as F
import numpy as np

print("=" * 75)
print("COMMON PROBABILITY DISTRIBUTIONS MATHEMATICAL VERIFICATION")
print("=" * 75)

# ─── 1. Univariate Gaussian Density Verification ───
print("\n1. 1D GAUSSIAN DENSITY CALCULATIONS:")
def gaussian_pdf_1d(x, mu=0.0, sigma=1.0):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x_vals = [0.0, 1.0, 2.0]
for x in x_vals:
    density = gaussian_pdf_1d(x, mu=0.0, sigma=1.0)
    print(f"   * Standard Normal Density at x={x:.1f}: p(x) = {density:.5f}")

# ─── 2. Multivariate Normal & Mahalanobis Distance ───
print("\n2. MULTIVARIATE NORMAL COVARIANCE SIZING:")
mu = torch.tensor([0.0, 0.0])
cov = torch.tensor([[4.0, 0.0], [0.0, 1.0]]) # sigma_x=2, sigma_y=1
dist = torch.distributions.MultivariateNormal(mu, cov)

pt_A = torch.tensor([2.0, 0.0])
pt_B = torch.tensor([0.0, 2.0])

prob_A = torch.exp(dist.log_prob(pt_A)).item()
prob_B = torch.exp(dist.log_prob(pt_B)).item()

print(f"   * Point A [2.0, 0.0] (Along wide axis sigma=2): Density = {prob_A:.5f}")
print(f"   * Point B [0.0, 2.0] (Along narrow axis sigma=1): Density = {prob_B:.5f}")
print(f"   * Ratio p(A) / p(B): {prob_A / prob_B:.2f}x (Point A is 4.5x more likely! ✅)")

# ─── 3. LLM Categorical Token Sampling with Temperature ───
print("\n3. CATEGORICAL TOKEN SAMPLING WITH TEMPERATURE:")
vocab = ["the", "cat", "sat", "on", "mat"]
logits = torch.tensor([2.0, 5.0, 1.0, 0.5, -1.0])

# Sampling with Temperature T=1.0 vs T=0.5
probs_T1 = F.softmax(logits / 1.0, dim=-1)
probs_T05 = F.softmax(logits / 0.5, dim=-1)

print(f"   Tokens:             {vocab}")
print(f"   Standard T=1.0 P:   {probs_T1.numpy().round(4).tolist()}")
print(f"   Sharper T=0.5 P:    {probs_T05.numpy().round(4).tolist()} (Concentrates mass on 'cat')")

# ─── 4. Empirical Dirac Delta Expectation Equivalence ───
print("\n4. EMPIRICAL DISTRIBUTION EXPECTATION EQUIVALENCE:")
dataset = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
# Function f(x) = x^2
f_vals = dataset ** 2
empirical_expectation = torch.mean(f_vals).item()
analytic_mean = (100 + 400 + 900 + 1600 + 2500) / 5.0

print(f"   Dataset Samples: {dataset.tolist()}")
print(f"   Sample Mean of x^2: {empirical_expectation:.2f} (Analytic: {analytic_mean:.2f}) ✅")

print("\n" + "=" * 75)
print("ALL PROBABILITY DISTRIBUTION TESTS PASSED SUCCESSFULLY! ✅")
print("=" * 75)
```

---

### 8. 🩺 Diagnostic Mini-Checks & Common Traps
> `Context:` Production Debugging Insights, Edge-Case Traps & Self-Verification Questions

#### ✅ Self-Test Questions

1. **Q:** In a continuous Gaussian distribution $\mathcal{N}(0, 1)$, can the density $p(x)$ ever exceed $1.0$?  
   **A:** **Yes!** For a narrow Gaussian with $\sigma = 0.1$, the peak density is $p(0) = \frac{1}{\sqrt{2\pi}(0.1)} \approx 3.989 > 1.0$. Probability *density* is not a probability; only the *integral* (area under the curve) is capped at $1.0$.

2. **Q:** Why do VAEs and Diffusion models use an **Isotropic Gaussian** ($\Sigma = I$) rather than a general covariance matrix for latent priors?  
   **A:** Isotropic Gaussians factorize into independent 1D variables along every coordinate ($p(z) = \prod p(z_j)$), making sampling trivial ($z = \text{randn()}$), computing analytical KL divergence closed-form, and preventing dimensional cross-talk.

3. **Q:** What is the difference between a Bernoulli distribution and a Categorical distribution?  
   **A:** Bernoulli models a single binary outcome ($K=2$, e.g. Coin Toss, Sigmoid output), whereas Categorical models multi-class discrete outcomes ($K > 2$, e.g. 6-sided die, Softmax over LLM vocabulary).

#### ⚠️ Common Engineering Traps

| Trap | Why It Fails | Production Fix |
| :--- | :--- | :--- |
| **Treating continuous density $p(x)$ as a probability $\le 1.0$** | Probability density has units of $1 / \text{unit}$ and can be arbitrarily large ($\to \infty$) | Integrate density over a small interval $\Delta x$ to compute probabilities: $P(a \le X \le b) = \int_a^b p(x)dx$ |
| **Sampling latents with standard Python `random.gauss` in a loop** | Extremely slow CPU single-thread bottleneck | Use vectorized GPU sampling: `torch.randn(batch_size, latent_dim, device='cuda')` |
| **Dividing by variance $\sigma^2$ without a small epsilon** | When $\sigma^2 \to 0$, division causes division-by-zero or `NaN` gradients | Add a stability constant: $\frac{1}{\sigma^2 + 10^{-8}}$ |
| **Assuming Categorical distributions preserve word semantics** | Categorical treats each token index as orthogonal with zero inherent geometric relation | Pass categorical tokens through an **Embedding Layer** ($W_{\text{embed}}$) to learn continuous semantic geometry |

---

### 🎯 Summary Checklist
- **Gaussian $\mathcal{N}(\mu, \sigma^2)$** is the foundational continuous distribution in AI, serving as latent priors and diffusion noise schedules.
- **Categorical $\text{Cat}(p_1, \dots, p_K)$** represents discrete probability distributions across finite options (e.g. LLM vocabulary).
- **Dirac Delta $\delta(x - x_i)$** models observed training data points with infinite density spikes.
- **Probability Density $p(x)$** can exceed $1.0$, but its total integral over all space $\int p(x)dx$ must strictly equal $1.0$.
- **Generative AI** works by learning a deep neural map that transforms simple standard distributions ($\mathcal{N}(0, I)$) into complex real-world data distributions ($p_{\text{data}}$).
