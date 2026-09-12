> [!NOTE]
> **Orientation**: This package examines U-Net architecture, continuous time embeddings, and guided sampling. For prerequisites on positional encodings and Bayes' conditional score decomposition, see [./PREREQUISITES.md](./PREREQUISITES.md).

# Lecture 17: Diffusion Models - Part 3 (Architecture, Sampling & Guided Diffusion)

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Master End-to-End Simulation](#master-end-to-end-simulation)
3. [Topic 1: Network Architecture Selection: U-Net and Vision Transformers for Isomorphic Spatial Regression](#topic-1-network-architecture-selection-u-net-and-vision-transformers-for-isomorphic-spatial-regression)
4. [Topic 2: Multi-Scale Noise Conditioning: Shared Weights Across Noise Intensities via Time Embeddings](#topic-2-multi-scale-noise-conditioning-shared-weights-across-noise-intensities-via-time-embeddings)
5. [Topic 3: Deterministic ODE Drift vs Stochastic Langevin Diffusion: The Role of Noise Injection](#topic-3-deterministic-ode-drift-vs-stochastic-langevin-diffusion-the-role-of-noise-injection)
6. [Topic 4: The Full Sampling Trajectory: Walking the Reverse Chain from x_T to Clean Sample x_0](#topic-4-the-full-sampling-trajectory-walking-the-reverse-chain-from-x_t-to-clean-sample-x_0)
7. [Topic 5: Transition to Controllable Generation: Introduction of Guided Diffusion and Conditional Priors](#topic-5-transition-to-controllable-generation-introduction-of-guided-diffusion-and-conditional-priors)
8. [Topic 6: Taxonomy of Guidance Mechanisms: Classifier Guidance vs Classifier-Free Guidance (CFG)](#topic-6-taxonomy-of-guidance-mechanisms-classifier-guidance-vs-classifier-free-guidance-cfg)
9. [Workplace Debugging Scenarios](#workplace-debugging-scenarios)
10. [References & Further Reading](#references--further-reading)

---

## Executive Summary

This lecture bridges mathematical theory to practical deep learning systems. We dissect the neural backbone (U-Net and Vision Transformers) that enables isomorphic spatial regression, explore how time embeddings modulate feature activations across $T$ noise scales, and formulate guided diffusion (Classifier Guidance and Classifier-Free Guidance) to turn unconditional denoisers into prompt-controlled generative engines.

```
                    GUIDANCE DYNAMICS: UNCONDITIONAL VS GUIDED
                    
   Unconditional Score:   nabla_{x_t} log p(x_t)        ---> Broad distribution mode
                                                                   |
   Classifier / Prompt:   nabla_{x_t} log p(y | x_t)    ---> Specific target mode ("Dog")
                                                                   |
   Guided Vector (CFG):   eps_uncond + s * (eps_cond - eps_uncond)
                          (Extrapolates along prompt vector)
```
*Figure 1: Geometric interpretation of Classifier-Free Guidance: extrapolating beyond the conditional prediction along the prompt vector direction.*

### Scenario Walkthrough
1. **Receive Condition**: User inputs prompt text $y$ (e.g., "A golden retriever").
2. **Inject Time & Class**: Embed $t$ via sinusoidal encodings; embed $y$ via text encoders (e.g., CLIP).
3. **Dual Model Evaluation**: Compute unconditional prediction $\epsilon_\theta(x_t, t, \emptyset)$ and conditional prediction $\epsilon_\theta(x_t, t, y)$.
4. **CFG Extrapolation**: Calculate $\tilde{\epsilon} = \epsilon_\emptyset + s(\epsilon_y - \epsilon_\emptyset)$ with guidance scale $s \in [3, 7.5]$.
5. **Ancestral Step**: Step backwards along guided trajectory $x_{t-1} = \mu(x_t, \tilde{\epsilon}) + \sigma_t z$.

### Failure / Contrast Path
Without guidance ($s=1$), diffusion models suffer from mode averaging where complex text prompts are partially ignored in favor of generic image statistics. With guidance set too high ($s > 15$), numerical values exceed dynamic ranges, causing severe pixel saturation and harsh contrast blowouts.

### STOP / Out of Scope
This lecture establishes **U-Net design, sampling dynamics, and guidance principles**. Latent space diffusion (LDM / Stable Diffusion) and full continuous-time SDE proofs are developed in Lecture 18.

### Comparative Feature & Tradeoff Matrix

| Method | Architecture Type | Spatial Resolution | Parameter Efficiency | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Standard CNN** | Contracting Bottleneck | Destroyed by Pooling | High | Classification & Detection |
| **U-Net** | Isomorphic with Skips | Perfectly Preserved | High | Pixel-Level Denoising |
| **DiT (Transformer)** | Patchified Tokens | Scalable via Attention | Very High | Large-Scale Synthesis |

### Load-Bearing Claims
- The diffusion network must be **strictly isomorphic**, mapping an input tensor to an identical output tensor shape.
- Sinusoidal time embeddings are necessary because model weights $\theta$ are shared across all $T$ noise levels.
- Classifier-Free Guidance eliminates the need for external classifier networks by training with random label dropping ($10\text{--}20\%$ rate).

### Comparative Feature & Tradeoff Matrix

| Guidance Paradigm | External Classifier Required? | Inference Forward Passes | Adherence to Prompt | Risk of Adversarial Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **Unconditional** | No | 1 per step | None (Random Sample) | None |
| **Classifier Guidance** | Yes (trained on noisy $x_t$) | 1 denoiser + 1 classifier | High | High (classifier gradient exploits) |
| **Classifier-Free (CFG)** | No (uses label dropout) | 2 per step ($\epsilon_u, \epsilon_c$) | Extremely High | Zero (fully generative) |

### Common Traps & Numerical Fixes
- **Classifier Trained Only on Clean Images**: Using an off-the-shelf ImageNet classifier for guidance fails because it cannot handle severe Gaussian noise at $t > 100$. Guidance classifiers must be explicitly trained on noisy $x_t$.
- **Over-Saturated CFG**: High guidance scale $s$ pushes pixel values outside $[-1, 1]$. Apply static thresholding or dynamic thresholding to prevent color burning.

---

## Master End-to-End Simulation

```python
import torch
import torch.nn as nn

def master_guided_diffusion_simulation():
    torch.manual_seed(42)
    B, D = 4, 8
    
    # 1. Mock inputs: noisy state xt, time t, and class condition y
    xt = torch.randn(B, D)
    t = torch.tensor([500, 500, 500, 500])
    
    # 2. Sinusoidal time embedding
    half_dim = D // 2
    freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(0, half_dim) / half_dim)
    t_emb = torch.cat([torch.sin(t[:, None] * freqs[None, :]), torch.cos(t[:, None] * freqs[None, :])], dim=-1)
    
    # 3. Denoiser evaluating conditional and unconditional branches
    # Unconditional noise prediction
    eps_uncond = torch.tanh(xt + t_emb)
    # Conditional noise prediction (steered by prompt vector)
    prompt_bias = torch.ones_like(xt) * 0.5
    eps_cond = torch.tanh(xt + t_emb + prompt_bias)
    
    # 4. Classifier-Free Guidance extrapolation
    s = 3.5 # Guidance scale
    eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
    
    # 5. Reverse step
    alpha_t = 0.95
    beta_t = 0.05
    alpha_bar_t = 0.60
    drift = (1.0 / (alpha_t**0.5)) * (xt - (beta_t / ((1.0 - alpha_bar_t)**0.5)) * eps_guided)
    
    assert eps_guided.shape == (B, D)
    assert drift.shape == (B, D)
    print(f"Master guidance simulation PASSED: guidance scale={s}, drift mean={drift.mean().item():.4f}")

master_guided_diffusion_simulation()
```

---

## Topic 1: Network Architecture Selection: U-Net and Vision Transformers for Isomorphic Spatial Regression

### Where this sits on the master map
We investigate the deep neural network backbones required to execute noise prediction. Because diffusion models map continuous image tensors $\mathbb{R}^{B \times C \times H \times W}$ back to identical image tensor spaces, architectures must preserve spatial correspondence.

### Board / screenshot
```
                ISOMORPHIC U-NET ARCHITECTURE
                
    Input x_t (32x32x3) -----------------[ Skip Conn ]-----------------> Output eps (32x32x3)
            |                                                                    ^
       [ ResBlock ]                                                         [ ResBlock ]
            v                                                                    |
    Downsample (16x16x64) --------------[ Skip Conn ]-----------------> Upsample (16x16x64)
            |                                                                    ^
            +------------------> [ Self-Attention Mid ] ------------------------+
```
*Notice: Skip connections bypass the bottleneck, directly forwarding high-frequency coordinate information to the decoder.*

### What he is establishing
The instructor establishes why standard classification or autoencoder architectures fail for diffusion and why isomorphic networks like U-Nets and Vision Transformers (DiT) are mandatory. In standard classification tasks, aggressive spatial pooling layers intentionally destroy spatial coordinates to compress the entire image down to a single categorical label. In generative diffusion, however, the neural network must output an individual noise prediction for every single pixel coordinate in the original image. If we forced this prediction through a severe compressive bottleneck, high-frequency spatial details such as hair strands, fabric weaves, and sharp boundary edges would be irreversibly blurred.

To solve this coordinate preservation problem, the U-Net employs long-range skip connections that copy feature maps from the contracting downsampling path directly to the corresponding stages of the expanding upsampling path. This architecture allows deep convolutional residual blocks to learn hierarchical feature representations while preserving pixel-level coordinate alignment. Concurrently, self-attention layers placed in the low-resolution middle blocks allow the network to capture global contextual relationships across distant parts of the image, such as ensuring bilateral symmetry in human faces and aligning lighting angles across the entire scene.

### Contrastive Analysis: Why X, Not Y?
- **Why not standard autoencoders without skip connections?**  
  Standard autoencoders force all information through a tight bottleneck, blurring sharp boundaries and losing fine hair and fabric details. Skip connections provide a lossless high-frequency bypass.

### Active Comprehension Checks
1. *Question*: What is the primary function of skip connections in a diffusion U-Net?  
   *Answer*: To transfer fine spatial details directly from encoder levels to corresponding decoder levels, bypassing low-resolution bottlenecks.

### Analogy for this topic only
An architect designing a skyscraper. They create a high-level master blueprint of the building (bottleneck attention), but they keep the original contractor blueprints pinned to the wall (skip connections) so workers know the exact plumbing alignment down to the millimeter.
*In lecture words: The U-Net combines macro scene understanding with micro coordinate precision.*

### Local picture
```
   [ High-Res Features ] ===(Skip Bypass)===> [ Decoder Reconstruction ]
```
*Notice: Direct feature concatenation.*

### Bridge
Because this single U-Net is shared across all 1,000 steps of diffusion, it needs a mechanism to adjust its behavior dynamically. We explore this via time conditioning.

---

## Topic 2: Multi-Scale Noise Conditioning: Shared Weights Across Noise Intensities via Time Embeddings

### Where this sits on the master map
We analyze how time embeddings allow a single network to behave as 1,000 different specialized denoisers.

### Board / screenshot
```
                TIME CONDITIONING MODULATION
                
    Timestep t in [1, T] ---> [ Sinusoidal Embedding ] ---> [ Linear MLP ] ---> [ AdaGN Modulation ]
                                                                                      |
                                                                                      v
                                                                           Feature Maps in ResBlock
```
*Notice: Continuous time embeddings modulate batch normalization or group normalization scale and shift parameters.*

### What he is establishing
1. **Weight Sharing Necessity**: Storing 1,000 separate neural networks for 1,000 diffusion steps would require terabytes of GPU memory.
2. **Behavioral Divergence Across $t$**:
   - At $t \approx 1000$, the input is pure Gaussian static. The network must infer high-level semantics (e.g., "this blob is a car").
   - At $t \approx 5$, the input is $99\%$ clean image. The network must only remove faint pixel fuzz.
3. **Adaptive Group Normalization (AdaGN)**:
   The time embedding vector modulates features via learned scale $\gamma(t)$ and shift $\beta(t)$:
   $$\text{AdaGN}(h, t) = (1 + \gamma(t)) \cdot \text{GroupNorm}(h) + \beta(t)$$

### Contrastive Analysis: Why X, Not Y?
- **Why continuous sinusoidal frequencies rather than integer one-hot encodings?**  
  One-hot encodings treat step 500 and step 501 as completely orthogonal, preventing parameter sharing. Sinusoidal embeddings provide smooth geometric continuity, allowing the network to interpolate between neighboring time steps.

### Active Comprehension Checks
1. *Question*: How does AdaGN incorporate the time embedding into a residual block?  
   *Answer*: By projecting the time embedding into affine scale $\gamma(t)$ and shift $\beta(t)$ parameters that modulate normalized activations.

### Analogy for this topic only
A master chef cooking a 10-course banquet. Instead of hiring 10 different chefs, one chef listens to an announcer calling out the course number ("Course 1: Soup", "Course 9: Dessert"), instantly switching their cooking style accordingly.
*In lecture words: Time conditioning acts as an operational mode switch for the shared network.*

### Local picture
```
    [ Normalized Features h ] * (1 + gamma(t)) + beta(t) ---> Modulated Features
```
*Notice: Time parameters directly rescale channel activations.*

### Bridge
Next, we examine what happens during reverse inference: does the sampling path follow deterministic trajectory curves or stochastic random walks?

---

## Topic 3: Deterministic ODE Drift vs Stochastic Langevin Diffusion: The Role of Noise Injection

### Where this sits on the master map
We investigate the difference between deterministic reverse sampling (Probability Flow ODE) and stochastic ancestral sampling (SDE Langevin diffusion).

### Board / screenshot
```
                ODE DRIFT VS SDE RANDOM WALK
                
    Deterministic ODE:   x_T -------------------------------> x_0 (Unique 1-to-1 curve)
    
    Stochastic SDE:      x_T ---\  /---\  /-----------------> x_0 (Branching multi-modal path)
                                 \/     \/
```
*Notice: Adding stochastic noise $\sigma_t z$ allows trajectories to correct errors and discover diverse image modes.*

### What he is establishing
The reverse transition is:
$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$
- **If $\sigma_t = 0$**: The reverse process becomes a deterministic Ordinary Differential Equation (ODE), known as the Probability Flow ODE (Song et al., 2021). A given noise vector $x_T$ maps deterministically to a unique image $x_0$. This allows exact latent inversion.
- **If $\sigma_t > 0$**: The reverse process is a Stochastic Differential Equation (SDE). Each step injects Langevin thermal agitation, correcting small errors made by $\mu_\theta$ and exploring distinct artistic modes.

### Contrastive Analysis: Why X, Not Y?
- **Why inject noise if deterministic ODE sampling is faster?**  
  Stochastic noise injection provides error tolerance. If a neural network prediction at step $t=800$ has a slight directional flaw, the random noise injected across subsequent steps dampens the accumulated error, whereas an ODE may integrate the error catastrophically.

### Active Comprehension Checks
1. *Question*: What is the advantage of setting $\sigma_t = 0$ in DDIM sampling?  
   *Answer*: It creates a deterministic mapping between latent noise and data, enabling image inversion, editing, and significant step reduction (e.g. 50 steps instead of 1000).

### Analogy for this topic only
Driving down a bumpy country road. A rigid steering lock (ODE) will drift off course if you hit one pothole. A driver making gentle random steering corrections (SDE) continuously self-centers back into the lane.
*In lecture words: Langevin noise injection provides dynamic error self-correction.*

### Local picture
```
   [ Single Drift Vector ] + [ Isotropic Random Jitter sigma*z ] ---> Corrected State
```
*Notice: Random perturbation prevents trajectory locking.*

### Bridge
Let us trace the complete ancestral sampling trajectory from pure Gaussian noise down to the pristine image.

---

## Topic 4: The Full Sampling Trajectory: Walking the Reverse Chain from $x_T$ to Clean Sample $x_0$

### Where this sits on the master map
We detail the mechanical sequence executed during deployment when a user requests an image from an unconditional diffusion model.

### Board / screenshot
```
                THE COMPLETE ANCESTRAL SAMPLING TRAJECTORY
                
    t = 1000:  x_1000 ~ N(0, I)           (Pure white static noise)
       |
    t = 750:   x_750 = mu(x_750) + sig*z  (Coarse light/dark color blobs appear)
       |
    t = 250:   x_250 = mu(x_250) + sig*z  (Recognizable objects and shapes form)
       |
    t = 1:     x_0 = mu(x_1)              (Sharp textures, pores, and hair complete)
```
*Notice: The sampling trajectory moves from low-frequency global structure to high-frequency micro-textures.*

### What he is establishing
The ancestral sampling loop:
1. Initialize $x_T \sim \mathcal{N}(0, I)$.
2. For $t = T, T-1, \dots, 1$:
   - Predict noise: $\hat{\epsilon} = \epsilon_\theta(x_t, t)$.
   - Compute drift mean:
     $$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon} \right)$$
   - Sample $z \sim \mathcal{N}(0, I)$ if $t > 1$, else $z = 0$.
   - Update $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$.
3. Clamp or decode $x_0$ to pixel space $[-1, 1]$.

### Contrastive Analysis: Why X, Not Y?
- **Why can't we evaluate all steps in parallel?**  
  Because $x_{t-1}$ is strictly conditioned on $x_t$. The trajectory is autoregressive in time, requiring sequential execution.

### Active Comprehension Checks
1. *Question*: At what stage in the trajectory are global scene semantics determined?  
   *Answer*: In the early steps ($t \in [1000, 700]$), where noise levels are highest.

### Analogy for this topic only
An artist sketching a portrait. In the first 10 minutes, they draw rough oval outlines and eye lines. In the last 10 minutes, they cross-hatch individual eyelash strands.
*In lecture words: Diffusion decomposes generation from coarse semantics to fine details across time.*

### Local picture
```
   [ Coarse Global Form (t ~ 1000) ] ===> [ Fine Microscopic Texture (t ~ 1) ]
```
*Notice: Hierarchical resolution of details.*

### Bridge
Unconditional sampling creates random realistic images. But how do we direct the generation to produce a specific desired object? We enter Guided Diffusion.

---

## Topic 5: Transition to Controllable Generation: Introduction of Guided Diffusion and Conditional Priors

### Where this sits on the master map
We formulate conditional diffusion modeling: synthesizing images conditioned on class labels, text prompts, or segmentation masks.

### Board / screenshot
```
                CONDITIONAL SCORE FORMULATION
                
    Goal: Sample from p(x_0 | y)
    
    Bayes' Theorem:
    log p(x_t | y) = log p(x_t) + log p(y | x_t) - log p(y)
    
    Spatial Gradient (Score):
    nabla_{x_t} log p(x_t | y) = nabla_{x_t} log p(x_t) + nabla_{x_t} log p(y | x_t)
```
*Notice: The gradient of the log-prior $\log p(y)$ vanishes because it does not depend on image coordinates $x_t$.*

### What he is establishing
To generate an image matching condition $y$:
1. We need the conditional reverse transition $p_\theta(x_{t-1} \mid x_t, y)$.
2. By Bayes' theorem, the conditional score decomposes into:
   $$\nabla_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y \mid x_t)$$
3. The first term $\nabla_{x_t} \log p(x_t)$ is the unconditional score, modeled by standard DDPM.
4. The second term $\nabla_{x_t} \log p(y \mid x_t)$ is the gradient of an image classifier predicting label $y$ from noisy image $x_t$.

By adding this classifier gradient to the reverse drift, we push the sampling trajectory toward regions of image space that maximize the probability of label $y$.

### Contrastive Analysis: Why X, Not Y?
- **Why not train the diffusion model with conditional inputs directly from day one?**  
  Direct conditioning without guidance scaling often yields weak prompt adherence; the model prefers matching generic image statistics over strict prompt fidelity. Guidance allows explicit mathematical amplification of the condition.

### Active Comprehension Checks
1. *Question*: What does the classifier gradient $\nabla_{x_t} \log p(y \mid x_t)$ represent geometrically?  
   *Answer*: A vector field indicating how to modify each pixel in $x_t$ to increase the classifier's confidence that the image represents class $y$.

### Analogy for this topic only
Hiking on a mountain. The unconditional score tells you which paths are safe, stable walking trails. The classifier gradient points in the direction of the mountain peak where the scenic view is located.
*In lecture words: The classifier gradient acts as an external steering force.*

### Local picture
```
   [ Unconditional Realism Drift ] + [ Classifier Steering Gradient ] ---> Guided Step
```
*Notice: Vector addition steers generation.*

### Bridge
We conclude by examining the two primary mechanisms for computing this steering force: Classifier Guidance vs Classifier-Free Guidance.

---

## Topic 6: Taxonomy of Guidance Mechanisms: Classifier Guidance vs Classifier-Free Guidance (CFG)

### Where this sits on the master map
We contrast Dhariwal & Nichol's Classifier Guidance with Ho & Salimans' state-of-the-art Classifier-Free Guidance (CFG).

### Board / screenshot
```
                GUIDANCE TAXONOMY: CLASSIFIER VS CLASSIFIER-FREE
                
   [ Classifier Guidance ] (Dhariwal & Nichol 2021)
   Denoiser eps_theta(x_t, t)  +  s * nabla_{x_t} log p(y | x_t)  (External Classifier)
   
   [ Classifier-Free Guidance (CFG) ] (Ho & Salimans 2021)
   eps_uncond + s * ( eps_cond - eps_uncond )                     (No Classifier!)
```
*Notice: CFG achieves superior guided synthesis without requiring an external classifier network.*

### What he is establishing
1. **Classifier Guidance (Dhariwal & Nichol 2021)**:
   Uses an auxiliary classifier $p_\phi(y \mid x_t)$ trained on noisy images. At each step, backpropagation computes $\nabla_{x_t} \log p_\phi(y \mid x_t)$.
   - *Drawback*: Requires training and storing a separate noise-aware classifier; classifier gradients can be noisy and vulnerable to adversarial shortcuts.
2. **Classifier-Free Guidance (Ho & Salimans 2021)**:
   A single diffusion network is trained to handle both conditional and unconditional generation by randomly dropping the label $y \to \emptyset$ with probability $p_{\text{drop}} \approx 0.1\text{--}0.2$.
   During inference, evaluate the network twice:
   $$\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, t, \emptyset) + s \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset) \right)$$
   where $s \ge 1$ is the guidance scale.
   - *Advantage*: No external classifier needed; works seamlessly with text embeddings (CLIP); state-of-the-art visual quality across all modern models (Stable Diffusion, DALL-E 3, Midjourney).

### Contrastive Analysis: Why X, Not Y?
- **Why is CFG universally preferred over Classifier Guidance?**  
  Training a separate classifier for open-ended text prompts ("a neon cyberpunk city in the rain") is virtually impossible. CFG natively operates on text embeddings from pretrained models without requiring an explicit classifier.

### Active Comprehension Checks
1. *Question*: In CFG, what is the effect of setting $s = 0$?  
   *Answer*: The model generates completely unconditional images, ignoring condition $y$.

### Analogy for this topic only
Hiring a private tutor (Classifier Guidance) versus self-correcting by comparing your rough draft to your best draft (CFG). The self-correcting student needs no outside coach.
*In lecture words: CFG turns the denoiser into its own implicit classifier.*

### Local picture
```
   [ eps_uncond ] --------------------------------> [ eps_cond ] -----------> [ Extrapolated CFG ]
                                                     (Distance)      s * Dist
```
*Notice: Linear extrapolation along conditioning difference vector.*

### Bridge
We have covered U-Net isomorphic structure, time embeddings, sampling dynamics, and guided diffusion. Now let's explore workplace debugging incidents.

---

## Workplace Debugging Scenarios

### Scenario 1: The High Guidance Contrast Burnout Artifact
**Incident:** An AI image generator with text prompting creates distorted images where skin appears radioactive orange and shadows are crushed to jet black.  
**Mathematical Root Cause:** The user set the CFG guidance scale $s = 25.0$. The extrapolation $\tilde{\epsilon} = \epsilon_u + s(\epsilon_c - \epsilon_u)$ produced extreme vector magnitudes, driving pixel values to $\pm 15.0$. When clamped to $[-1, 1]$ for display, color clipping destroyed all tonal gradients.  
**Debugging Steps:**
1. Inspect the maximum absolute value of `eps_guided`. Observe values exceeding $10.0$.
2. Reduce guidance scale to the standard operating sweet spot $s \in [3.0, 7.5]$. Alternatively, apply dynamic thresholding (Imagen).  
**Code Fix:**
```python
# DYNAMIC THRESHOLDING FIX:
def dynamic_clip(x, percentile=99.5):
    s = torch.quantile(torch.abs(x), percentile / 100.0)
    s = torch.maximum(s, torch.tensor(1.0))
    return torch.clamp(x, -s, s) / s
```

---

### Scenario 2: The Silent Guidance Dropout Failure
**Incident:** A developer trains a conditional diffusion model with CFG, but during inference, varying the prompt condition produces identical random images with zero correlation to the text.  
**Mathematical Root Cause:** During training, the developer set the label dropout probability `p_drop = 1.0` instead of `0.15` due to a configuration bug. The model never trained on conditioned pairs $(x_t, y)$, making $\epsilon_c(x_t, t, y) \equiv \epsilon_u(x_t, t, \emptyset)$ at all times.  
**Debugging Steps:**
1. Evaluate `torch.norm(eps_cond - eps_uncond)`. It evaluates to $0.0$, indicating no condition sensitivity.
2. Check training dropout probability config: verify `p_drop` is set to $0.15$.  
**Code Fix:**
```python
# CORRECT TRAINING LABEL DROPOUT:
p_uncond = 0.15
mask = torch.rand(batch_size) < p_uncond
conditioned_y = torch.where(mask, empty_token, input_y)
```

---

## References & Further Reading

For complete citations, seminal papers, academic lecture links, and official implementations, see [./references.md](./references.md).
