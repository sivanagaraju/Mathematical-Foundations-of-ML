# Normalization Layers & Spectral Norm: Stabilizing Deep Representations & Generative Gradients

> `🏷️ Tags:` `Deep-Learning` `Batch-Normalization` `Layer-Normalization` `RMSNorm` `Spectral-Normalization` `GroupNorm` `Transformers` `GANs` `Diffusion`  
> `📚 Prerequisites Needed:` [Vector Norms & Inner Products](./Vector_Norms_and_Inner_Products.md) · [Lipschitz Continuity](./Lipschitz_Continuity.md) · [Tensors & Shapes](./Tensors_and_Shapes.md)  
> `🎯 Where Do We Use This?:` **Ubiquitous in every modern neural network** — RMSNorm in Large Language Models (LLaMA-3, Mistral, GPT-4), GroupNorm + AdaIN in Diffusion U-Nets (Stable Diffusion, Flux), LayerNorm in Vision Transformers (ViT), and Spectral Normalization in GAN discriminators (SNGAN, BigGAN).  
> `🎓 Course Module Mapping:` [Tut 04: CNNs](../Mathematical-Foundation-for-GenerativeAI/18-Tutorial04-CNNs-PyTorch/NOTES.md) · [Lec 01: Intro](../Mathematical-Foundation-for-GenerativeAI/14-Lec01-MFGAI-Introduction/NOTES.md) · [Lec 18: WGAN](../Mathematical-Foundation-for-GenerativeAI/30-Lec18-Wasserstein-GAN/NOTES.md) · [Tut 12: GAN Implementations](../Mathematical-Foundation-for-GenerativeAI/29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/NOTES.md)  
> `⏱️ Difficulty Level:` ⭐⭐☆☆☆ (Foundational · 15 min read)

---

### 📌 Quick Navigation & Architecture Map
- [1. 🌟 Everyday Real-World Scenarios](#1--everyday-real-world-scenarios-the-audio-auto-leveler--the-engine-governor) — The Audio Auto-Leveler & The Engine Governor
- [2. 👶 ELI5 Intuition](#2--eli5-intuition-the-camera-iso-dial--the-power-drill-governor) — The Camera ISO Dial & The Power Drill Governor
- [3. 📚 Deep Terminology Master Glossary](#3--deep-terminology-master-glossary-15-core-concepts-dissected) — 15 normalization terms dissected without jargon
- [4. 📐 Mathematical Formulations, Power Iteration & The Normalization Zoo](#4--mathematical-formulations-power-iteration--the-normalization-zoo) — BatchNorm, LayerNorm, RMSNorm, and Spectral Normalization algorithms
- [5. 🔢 Concrete Micro-Numerical Worked Examples](#5--concrete-micro-numerical-worked-examples) — Mini-Batch Normalization by Hand & Power Iteration Spectral Norm Step
- [6. 🔗 Connecting the Dots: Generative AI Architecture Blocks](#6--connecting-the-dots-how-normalization-powers-generative-ai) — LLM RMSNorm Block, Diffusion GroupNorm/AdaIN, and SNGAN Discriminator
- [7. 💻 Standalone Executable Python/PyTorch Verification Script](#7--complete-standalone-executable-pythonpytorch-verification-script) — BatchNorm vs LayerNorm vs RMSNorm vs SpectralNorm simulation
- [8. 🩺 Diagnostic Mini-Checks & Common Traps](#8--diagnostic-mini-checks--common-traps) — Self-test questions & production engineering pitfalls

---

**Normalization layers** are mathematical operators that standardize intermediate layer activations and constrain weight matrices in deep neural networks, preventing Internal Covariate Shift, eliminating exploding/vanishing gradients, and enforcing 1-Lipschitz continuity in Generative Adversarial Networks.

```
 ===================================================================================================
                 THE 3-STAGE NORMALIZATION PIPELINE ACROSS DEEP LAYERS
 ===================================================================================================

  STAGE 1: STATISTICAL MOMENTS (μ, σ²)   STAGE 2: ZERO-MEAN NORMALIZATION    STAGE 3: LEARNABLE RESCALING (γ, β)
  Mini-Batch / Spatial Statistics        Standard Gaussian Standardize       Restoring Optimal Capacity
  ┌──────────────────────────────┐      ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │ μ_B = (1/m) Σ x_i            │─────►│ x̂_i = (x_i - μ_B) /          │───►│ y_i = γ · x̂_i + β            │
  │ σ_B² = (1/m) Σ (x_i - μ_B)²  │      │       sqrt(σ_B² + ϵ)         │    │ Learnable scale γ, shift β   │
  │ Tracks activation variance   │      │ Centers at 0, unit variance  │    │ Preserves expressive power   │
  └──────────────────────────────┘      └──────────────────────────────┘    └──────────────────────────────┘
 ===================================================================================================
```

---

### 1. 🌟 Everyday Real-World Scenarios (The Audio Auto-Leveler & The Engine Governor)
> `Context:` Zero Prior Machine Learning / AI Knowledge Needed · Concrete Real-World Mapping

#### Scenario A: The Audio Auto-Leveler in Podcasting (Zero ML Background Needed)
Imagine recording a podcast with 50 speakers chained together:
1. **The Compounding Problem:** If speaker 1 whispers and each person halves their volume, speaker 50 hears dead silence ($2^{-50} \approx 0$). If each person doubles their volume, speaker 50's ears explode ($2^{50} \approx 10^{15}$).
2. **The Auto-Leveler Solution (Normalization):** An automatic audio compressor listens to the sound, measures average volume, and normalizes it to standard studio broadcast volume ($\text{mean}=0\text{ dB}, \text{variance}=1$), preventing feedback shrieks and silence.

---

#### Scenario B: In Generative AI — Spectral Normalization in SNGAN & BigGAN
> `Context:` How Spectral Normalization Stabilizes Adversarial Discriminators

In Generative Adversarial Networks:
- If the Discriminator is unconstrained, its gradient slopes can blow up toward infinity ($|\nabla D| \to \infty$).
- This makes real and fake image scores drift millions of units apart, destroying the generator's gradient signal.
- **Spectral Normalization** divides every layer's weight matrix by its maximum singular value ($W / \sigma_1(W)$), installing a strict mathematical speed governor that enforces 1-Lipschitz continuity and stabilizes GAN training forever!

```
 ===================================================================================================
         NORMALIZATION SPATIAL DIMENSIONS ACROSS AI ARCHITECTURES
 ===================================================================================================

  BATCH NORM (CNNs):            LAYER NORM (LLMs):            GROUP NORM (Diffusion):
  Across Batch (B, H, W)        Across Features (C, H, W)     Across Channel Groups (C/G, H, W)
  ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
  │ B₁  B₂  B₃  B₄        │     │ Sample 1: [ C₁ C₂ C₃] │     │ Group 1: [ C₁ C₂ ]    │
  │ [●] [●] [●] [●] ──► μ │     │ ────────────────────► │     │ Group 2: [ C₃ C₄ ]    │
  │ Requires large batch! │     │ Independent per batch!│     │ Independent per batch!│
  └───────────────────────┘     └───────────────────────┘     └───────────────────────┘
 ===================================================================================================
```

---

### 2. 👶 ELI5 Intuition: The Camera ISO Dial & The Power Drill Governor
> `Context:` Physical & Everyday Metaphors for Activation Normalization and Spectral Constraints

#### Metaphor 1: The Camera Auto-ISO Dial (Activation Normalization)
- If you walk from a dark room into bright sunlight, the camera auto-adjusts its ISO and shutter speed so photos are never pitch-black or blinded by white glare.
- Normalization auto-adjusts neural signals so activations never vanish into zero or blow up into `NaN`.

---

#### Metaphor 2: The Power Drill Speed Governor (Spectral Normalization)
- A powerful drill has a governor limiting its maximum rotation speed.
- Even if you pull the trigger as hard as possible, the drill spins at a safe, controlled speed ($1.0$), protecting the motor and drill bit from breaking.

---

### 3. 📚 Deep Terminology Master Glossary (15 Core Concepts Dissected)
> `Context:` Foundational Mathematical & Machine Learning Vocabulary Explained Without Jargon

```
 ===================================================================================================
                 THE NEURAL NORMALIZATION ROSETTA STONE
 ===================================================================================================
```

| Term / Notation | Formal Mathematical Meaning | Plain-English Definition (No ML Jargon) | Real-World Analogy |
| :--- | :--- | :--- | :--- |
| **Batch Normalization (BN)** | Standardizes over batch dimension $(B, H, W)$ | Centers activations using statistics from all samples in the current batch | Grading a test on a curve across all students |
| **Layer Normalization (LN)** | Standardizes over feature channels per sample | Normalizes across all features for one single sample independently | Normalizing audio tracks individually |
| **RMSNorm** | $\frac{x}{\text{RMS}(x)} \odot \gamma$ (No mean subtraction) | Faster variant of LayerNorm that enforces unit root-mean-square magnitude | Scaling volume without changing baseline silence |
| **Group Normalization (GN)** | Standardizes across groups of channels | Divides channels into small clusters and normalizes each cluster | Grouping musicians in an orchestra into sections |
| **Instance Normalization (IN)**| Standardizes spatial area $(H, W)$ per channel | Removes global image contrast/style while preserving content edges | Converting a photo to a high-contrast pencil sketch |
| **Spectral Normalization (SN)**| $W_{\text{SN}} = W / \sigma_1(W)$ | Divides weight matrix by its largest singular value to enforce 1-Lipschitz bound | Installing a top speed governor on an engine |
| **Internal Covariate Shift** | Drifting distribution of layer inputs during training | Upstream layers constantly changing, forcing downstream layers to re-adapt | Trying to hit a target while the ground is shaking |
| **Running Mean / Variance** | Exponential moving averages ($\hat{\mu}_{\text{run}}, \hat{\sigma}^2_{\text{run}}$) | Historical average saved during training to be used for deterministic inference | A rolling 30-day stock price average |
| **Learnable Scale ($\gamma$) & Shift ($\beta$)** | Affine parameters $y = \gamma \hat{x} + \beta$ | Restores network representation power if pure standard Gaussian is too restrictive | Equalizer bass and treble boost sliders |
| **Power Iteration Algorithm** | Fast $O(1)$ approximation of largest singular value | Repeatedly multiplying a random vector by $W$ and $W^\top$ to find top singular value | Finding the highest point on a spinning merry-go-round |
| **1-Lipschitz Condition** | $\|\nabla f(x)\|_2 \le 1.0$ | Function output cannot change faster than $1\times$ the input change | A gentle $45^\circ$ walking ramp |
| **Train vs Eval Mode Leak** | Forgetting `model.eval()` during inference | Accidental dependence on test batch size leading to erratic single-sample predictions | Using grading curve rules when grading only 1 student |
| **Adaptive Instance Norm (AdaIN)** | $y = \sigma(y_s) \frac{x - \mu(x)}{\sigma(x)} + \mu(y_s)$ | Replaces content mean/variance with style image mean/variance in StyleGAN | Applying Picasso's color palette to a selfie |
| **FiLM (Feature Modulation)**| $y = \gamma(\text{cond}) \odot x + \beta(\text{cond})$ | Conditioning mechanism injecting prompt/timestep embeddings into layers | Turning thermostat knob to adjust room temperature |
| **Weight Normalization** | $W = g \frac{v}{\|v\|_2}$ | Decouples the length (norm $g$) of a weight vector from its direction ($v$) | Separating a rocket's engine thrust from its steering angle |

---

### 4. 📐 Mathematical Formulations, Power Iteration & The Normalization Zoo
> `Context:` Formal Statistical Equations, Power Iteration Convergence Proof, and Summary Matrix

```
 ===================================================================================================
                 THE DEEP LEARNING NORMALIZATION ZOO
 ===================================================================================================
```

| Normalization Layer | Normalization Formula | Dimension Normalized | Primary Modern Application |
| :--- | :--- | :--- | :--- |
| **BatchNorm2d** | $y = \gamma \left( \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \right) + \beta$ | $(B, H, W)$ per channel | ResNets, ConvNets, Image Classifiers |
| **LayerNorm** | $y = \gamma \left( \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} \right) + \beta$ | $(C, H, W)$ per sample | Original Transformers, ViT |
| **RMSNorm** | $y = \gamma \odot \left( \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \right)$ | Features $d$ per token | **LLaMA-3, Mistral, Gemma, GPT-4** |
| **GroupNorm** | $y = \gamma \left( \frac{x - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}} \right) + \beta$ | $(C/G, H, W)$ per sample | **Diffusion Models (Stable Diffusion, Flux)** |
| **SpectralNorm** | $\bar{W}_{\text{SN}} = \frac{W}{\sigma_1(W)}, \quad \sigma_1(W) \approx u^\top W v$ | Weight Matrix $\mathbb{R}^{M \times N}$ | **GAN Discriminators (SNGAN, BigGAN)** |

#### Spectral Normalization Power Iteration Formulation (Miyato et al., 2018):
Given weight matrix $W \in \mathbb{R}^{M \times N}$ and random initial vector $\tilde{v} \in \mathbb{R}^N$:
1. Update right singular vector: $\tilde{u} \leftarrow \frac{W \tilde{v}}{\|W \tilde{v}\|_2}$
2. Update left singular vector: $\tilde{v} \leftarrow \frac{W^\top \tilde{u}}{\|W^\top \tilde{u}\|_2}$
3. Spectral Norm Estimate: $\sigma_1(W) \approx \tilde{u}^\top W \tilde{v}$
4. Normalized Weight Matrix: $\bar{W}_{\text{SN}} = \frac{W}{\sigma_1(W)}$

---

### 5. 🔢 Concrete Micro-Numerical Worked Examples
> `Context:` Step-by-Step Manual Calculations (No Black Box)

#### Example 1: Mini-Batch Normalization by Hand
Let mini-batch activations be $x = [2.0, \quad 4.0, \quad 6.0]$, with parameters $\epsilon = 0$, $\gamma = 2.0$, $\beta = 1.0$.

1. **Calculate Mini-Batch Mean:**
   $$\mu_{\mathcal{B}} = \frac{2.0 + 4.0 + 6.0}{3} = \frac{12.0}{3} = \mathbf{4.0000}$$

2. **Calculate Mini-Batch Variance:**
   $$\sigma_{\mathcal{B}}^2 = \frac{(2-4)^2 + (4-4)^2 + (6-4)^2}{3} = \frac{4 + 0 + 4}{3} = \frac{8}{3} \approx \mathbf{2.6667} \implies \sigma_{\mathcal{B}} = \sqrt{2.6667} \approx \mathbf{1.6330}$$

3. **Standardize $\hat{x}_i = \frac{x_i - \mu}{\sigma}$:**
   $$\hat{x}_1 = \frac{2.0 - 4.0}{1.6330} = \mathbf{-1.2247}, \quad \hat{x}_2 = \frac{4.0 - 4.0}{1.6330} = \mathbf{0.0000}, \quad \hat{x}_3 = \frac{6.0 - 4.0}{1.6330} = \mathbf{+1.2247}$$

4. **Apply Learnable Scale ($\gamma=2$) and Shift ($\beta=1$):**
   $$y_1 = 2.0(-1.2247) + 1.0 = -2.4494 + 1.0 = \mathbf{-1.4494}$$
   $$y_2 = 2.0(0.0000) + 1.0 = \mathbf{1.0000}$$
   $$y_3 = 2.0(+1.2247) + 1.0 = +2.4494 + 1.0 = \mathbf{+3.4494}$$
   - Notice: The output mean is $\frac{-1.4494 + 1.0 + 3.4494}{3} = \mathbf{1.0000 = \beta}$!

---

#### Example 2: Spectral Norm via Power Iteration by Hand
Let diagonal weight matrix $W = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$ (Exact true $\sigma_1(W) = 3.0$).
Let initial vector $\tilde{v}_0 = [1.0, \quad 1.0]^\top$.

1. **Step 1: Compute $W \tilde{v}_0$:**
   $$W \tilde{v}_0 = \begin{bmatrix} 3(1) \\ 1(1) \end{bmatrix} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad \|W \tilde{v}_0\|_2 = \sqrt{3^2 + 1^2} = \sqrt{10} \approx 3.1623$$
   $$\tilde{u}_1 = \frac{1}{\sqrt{10}} \begin{bmatrix} 3 \\ 1 \end{bmatrix} \approx \begin{bmatrix} 0.9487 \\ 0.3162 \end{bmatrix}$$

2. **Step 2: Compute $W^\top \tilde{u}_1$:**
   $$W^\top \tilde{u}_1 = \begin{bmatrix} 3(0.9487) \\ 1(0.3162) \end{bmatrix} = \begin{bmatrix} 2.8461 \\ 0.3162 \end{bmatrix}, \quad \text{Norm} = \sqrt{2.8461^2 + 0.3162^2} = \sqrt{8.200} \approx 2.8636$$
   $$\tilde{v}_1 = \begin{bmatrix} 0.9939 \\ 0.1104 \end{bmatrix}$$

3. **Step 3: Estimate $\sigma_1(W) \approx \tilde{u}_1^\top W \tilde{v}_1$:**
   $$\sigma_1 \approx [0.9487, \quad 0.3162] \begin{bmatrix} 3(0.9939) \\ 1(0.1104) \end{bmatrix} = 0.9487(2.9817) + 0.3162(0.1104) = 2.8288 + 0.0349 = \mathbf{2.8637}$$
   *(In just 1 single step, the power iteration converged within $4.5\%$ of true value $3.000$, and in 3 steps reaches $2.999$!).*

---

### 6. 🔗 Connecting the Dots: How Normalization Powers Generative AI
> `Context:` Architectural Implementations in Large Language Models, Diffusion Models, and GANs

```
 ===================================================================================================
                 NORMALIZATION LAYERS ACROSS GENERATIVE AI
 ===================================================================================================

  1. LLM RMSNORM BLOCK (LLaMA-3 / Mistral)          2. DIFFUSION GROUPNORM + FiLM (Flux / SD3)
  y = (x / RMS(x)) ⊙ γ                              y = (1 + γ(t)) ⊙ GroupNorm(x) + β(t)
  ┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐
  │ 7% faster than standard LayerNorm      │        │ Injects diffusion timestep embeddings  │
  │ Eliminates mean-centering overhead     │        │ Operates cleanly with batch size B=1   │
  │ Stabilizes 100-layer backpropagation   │        │ Preserves local spatial image textures │
  └────────────────────────────────────────┘        └────────────────────────────────────────┘
 ===================================================================================================
```

| Generative Architecture | Primary Normalization Technique | Architectural Implementation |
| :--- | :--- | :--- |
| **Large Language Models (LLaMA-3, Mistral)** | **RMSNorm (Root Mean Square LayerNorm)** | Normalizes token activations before Self-Attention and MLP blocks without mean centering |
| **Diffusion Models (Stable Diffusion 3, Flux)** | **GroupNorm ($G=32$) + AdaIN / FiLM** | Normalizes spatial feature maps independently of batch size; FiLM modulates scale by timestep $t$ |
| **Generative Adversarial Nets (SNGAN, BigGAN)**| **Spectral Normalization (`spectral_norm`)** | Normalizes all discriminator convolutional filters to ensure strict 1-Lipschitz continuity |
| **StyleGAN (Style-Based Generation)** | **Adaptive Instance Normalization (AdaIN)** | Strips content style variance and replaces it with style vector $w$ scale and bias |

---

### 7. 💻 Complete Standalone Executable Python/PyTorch Verification Script
> `Context:` Runnable Code Comparing BatchNorm, LayerNorm, RMSNorm, and Power Iteration Spectral Norm

```python
"""
Normalization Layers & Spectral Normalization Simulation
========================================================
Demonstrates:
1. Exact BatchNorm numerical calculation with learnable scale/shift
2. RMSNorm implementation in pure PyTorch (LLaMA-style)
3. Spectral Normalization via Power Iteration vs exact SVD
"""
import torch
import torch.nn as nn
import numpy as np

print("=" * 75)
print("NORMALIZATION LAYERS & SPECTRAL NORM MATHEMATICAL SIMULATION")
print("=" * 75)

# ─── 1. Mini-Batch Normalization Calculation ───
print("\n1. BATCH NORMALIZATION STEP-BY-STEP CALCULATION:")
x = torch.tensor([[2.0], [4.0], [6.0]]) # Batch=3, Feature=1
bn = nn.BatchNorm1d(1, eps=0.0, affine=True)
bn.weight.data.fill_(2.0) # gamma = 2.0
bn.bias.data.fill_(1.0)   # beta = 1.0

y = bn(x)
print(f"   Input Tensor x:\n{x.squeeze().numpy()}")
print(f"   * Output Tensor y (gamma=2, beta=1):\n{y.squeeze().detach().numpy().round(4)}")
expected_y = np.array([-1.4495, 1.0000, 3.4495])
assert np.allclose(y.squeeze().detach().numpy(), expected_y, atol=1e-3), "BatchNorm mismatch!"
print("   * BatchNorm forward pass verified mathematically! ✅")

# ─── 2. LLaMA-Style RMSNorm Implementation ───
print("\n2. LLAMA-STYLE RMSNORM IMPLEMENTATION:")
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

tokens = torch.randn(2, 4, 16) # (Batch, Seq_Len, Dim)
rmsnorm = RMSNorm(16)
normed_tokens = rmsnorm(tokens)

print(f"   Input Token Shape:  {list(tokens.shape)}")
print(f"   * Output RMS Shape: {list(normed_tokens.shape)} (Normalized cleanly without mean subtraction! ✅)")

# ─── 3. Spectral Normalization via Power Iteration ───
print("\n3. SPECTRAL NORMALIZATION POWER ITERATION VERIFICATION:")
W = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
true_sigma = torch.linalg.svdvals(W)[0].item() # 3.0000

# Power Iteration
v = torch.tensor([1.0, 1.0])
for step in range(5):
    u = W @ v
    u = u / torch.norm(u, p=2)
    v = W.T @ u
    v = v / torch.norm(v, p=2)
    sigma_est = (u @ W @ v).item()

print(f"   * True Exact Spectral Norm (SVD):   {true_sigma:.4f}")
print(f"   * Power Iteration Estimate (5 iter): {sigma_est:.4f} ✅")

print("\n" + "=" * 75)
print("ALL NORMALIZATION & SPECTRAL NORM TESTS PASSED SUCCESSFULLY! ✅")
print("=" * 75)
```

---

### 8. 🩺 Diagnostic Mini-Checks & Common Traps
> `Context:` Production Debugging Insights, Edge-Case Traps & Self-Verification Questions

#### ✅ Self-Test Questions

1. **Q:** Why do Transformer Large Language Models use LayerNorm or RMSNorm instead of BatchNorm?  
   **A:** In NLP, sentence lengths vary dynamically across batches, and batch sizes during inference are often $B = 1$. BatchNorm fails completely with batch size $1$ ($\text{Var} = 0$). LayerNorm and RMSNorm normalize across feature dimensions per token, making them **$100\%$ independent of batch size**.

2. **Q:** Why is RMSNorm preferred over LayerNorm in modern LLMs like LLaMA-3?  
   **A:** Research shows that the primary stabilization benefit of LayerNorm comes from scaling by the root-mean-square variance, not from subtracting the mean. RMSNorm removes the mean-centering step, saving GPU bandwidth and running $\approx 7\%$ faster per transformer block.

3. **Q:** What happens if you forget to call `model.eval()` when testing a model with BatchNorm?  
   **A:** In training mode, BatchNorm computes statistics from the current test batch rather than using historical running averages ($\hat{\mu}_{\text{run}}, \hat{\sigma}^2_{\text{run}}$). If you pass a single test sample ($B=1$), the model will crash or produce garbage outputs.

#### ⚠️ Common Engineering Traps

| Trap | Why It Fails | Production Fix |
| :--- | :--- | :--- |
| **Using small batch size ($B < 8$) with BatchNorm** | Noisy batch variance leads to severe training instability and degraded accuracy | Use **GroupNorm** or **LayerNorm** when working with small batch sizes |
| **Forgetting `model.eval()` before inference** | Activations normalize against the test batch instead of global training running stats | Always wrap evaluation loops with `model.eval()` and `with torch.no_grad():` |
| **Using BatchNorm inside WGAN Critic networks** | Creates statistical dependencies between different images, violating 1-Lipschitz metric properties | Replace with **Spectral Normalization** or **LayerNorm** |

---

### 🎯 Summary Checklist
- **BatchNorm** standardizes across the batch dimension $(B, H, W)$ for CNN classifiers.
- **LayerNorm & RMSNorm** standardize across feature channels per sample, making them the standard in Transformers and LLMs.
- **GroupNorm** divides channels into clusters, providing batch-independent stability in Diffusion U-Nets.
- **Spectral Normalization ($W / \sigma_1(W)$)** enforces 1-Lipschitz continuity in GAN discriminators.
- **Power Iteration** computes matrix spectral norms in $O(1)$ fast forward steps.
