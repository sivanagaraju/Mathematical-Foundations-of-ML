> [!NOTE]
> **Orientation**: This package concludes the theoretical study of diffusion models. For mathematical prerequisites on Tweedie's formula, score matching duality, and continuous SDEs, see [./PREREQUISITES.md](./PREREQUISITES.md).

# Lecture 18: Diffusion Models - Part 4 (Score Theory, Guidance & Latent Diffusion)

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Master End-to-End Simulation](#master-end-to-end-simulation)
3. [Topic 1: Classical Statistics & Tweedie's Formula: The Empirical Bayes Bridge from Posterior Mean to Score](#topic-1-classical-statistics--tweedies-formula-the-empirical-bayes-bridge-from-posterior-mean-to-score)
4. [Topic 2: The Score-Matching Duality: Formulating DDPM as a Continuous Vector Field Estimator](#topic-2-the-score-matching-duality-formulating-ddpm-as-a-continuous-vector-field-estimator)
5. [Topic 3: Classifier-Guided Diffusion: Steering Reverse Trajectories with External Noise-Robust Classifiers](#topic-3-classifier-guided-diffusion-steering-reverse-trajectories-with-external-noise-robust-classifiers)
6. [Topic 4: Classifier-Free Guidance (CFG): Derivation and Mechanics of Implicit Score Guidance](#topic-4-classifier-free-guidance-cfg-derivation-and-mechanics-of-implicit-score-guidance)
7. [Topic 5: Latent Diffusion Models (LDM): High-Resolution Synthesis via Pretrained Latent Space Autoencoders](#topic-5-latent-diffusion-models-ldm-high-resolution-synthesis-via-pretrained-latent-space-autoencoders)
8. [Topic 6: Continuous-Time Diffusion: Unifying VP and VE SDEs with the Reverse-Time Stochastic Differential Equation](#topic-6-continuous-time-diffusion-unifying-vp-and-ve-sdes-with-the-reverse-time-stochastic-differential-equation)
9. [Workplace Debugging Scenarios](#workplace-debugging-scenarios)
10. [References & Further Reading](#references--further-reading)

---

## Executive Summary

This lecture unifies the entire theoretical canon of diffusion models under continuous-time score matching and stochastic differential equations. We explore Tweedie's formula as the foundational bridge connecting empirical Bayes to neural score estimation, derive Classifier-Free Guidance (CFG) as an algebraic implicit score manipulator, and examine Latent Diffusion Models (LDM / Stable Diffusion) as the production paradigm for efficient high-resolution generative AI.

```
                    THE UNIFIED DIFFUSION SCORE PARADIGM
                    
   +-------------------------------------------------------------------------+
   |   CONTINUOUS TIME:  dx = f(x, t)dt + g(t) dw   (Forward Degradation)    |
   |   REVERSE SDE:      dx = [ f(x, t) - g(t)^2 nabla log p ] dt + g(t) dw~ |
   +-------------------------------------------------------------------------+
                                        ||
                    [ Tweedie's Formula & Score Duality ]
                                        vv
   +-------------------------------------------------------------------------+
   |   Score: nabla log p(x_t) = - eps_theta(x_t, t) / sqrt(1 - alpha_bar_t) |
   +-------------------------------------------------------------------------+
                                        ||
                    [ Conditioning & Latent Space Acceleration ]
                                        vv
   +-------------------------------------+   +-------------------------------+
   | Classifier-Free Guidance (CFG):     |   | Latent Diffusion (LDM):       |
   | eps_u + s * ( eps_c - eps_u )       |   | Diffusion in VAE Latent Space |
   | High prompt fidelity without class. |   | 50x compute reduction         |
   +-------------------------------------+   +-------------------------------+
```
*Figure 1: Conceptual unification: Continuous-time reverse SDEs, score-noise equivalence, CFG extrapolation, and latent space compression.*

### Scenario Walkthrough
1. **Compress to Latent**: Encode high-resolution image $x_0 \in \mathbb{R}^{3 \times 512 \times 512}$ via pretrained VAE encoder into $z_0 = \mathcal{E}(x_0) \in \mathbb{R}^{4 \times 64 \times 64}$.
2. **Diffuse in Latent Space**: Apply standard forward noise schedule to $z_0$, generating noisy latent $z_t$.
3. **Condition on Text**: Pass prompt text through CLIP encoder; feed text embeddings into latent U-Net cross-attention layers.
4. **CFG Latent Sampling**: Extrapolate latent noise $\tilde{\epsilon} = \epsilon_\emptyset + s(\epsilon_y - \epsilon_\emptyset)$ and iterate ancestral steps down to $z_0$.
5. **Decode to Pixels**: Pass synthesized clean latent $z_0$ through VAE decoder $\mathcal{D}(z_0)$ to render final $512 \times 512$ image.

### Failure / Contrast Path
Pixel-space diffusion on $512 \times 512$ images requires high-end supercomputing clusters and days of training due to calculating self-attention on $262,144$ pixel coordinates. Attempting high-resolution generation directly in pixel space wastes 95% of GPU memory computing imperceptible sub-pixel noise. Latent diffusion compresses pixel redundancy first, reducing compute by 48x while preserving photorealism.

### STOP / Out of Scope
This lecture provides the **complete unified theoretical and practical architecture**. Specific fine-tuning methods (LoRA, ControlNet, DreamBooth) and video diffusion extensions are engineering specializations built upon these core pillars.

### Load-Bearing Claims
- Tweedie's formula proves that optimal Gaussian denoising $\mathbb{E}[x_0 \mid x_t]$ is purely a function of the observation $x_t$ and the score $\nabla_{x_t} \log p(x_t)$.
- Predicting noise $\epsilon_\theta$ in DDPM is mathematically identical to estimating the score vector $\nabla_{x_t} \log p(x_t)$ up to a known variance scalar.
- Latent Diffusion Models separate generative modeling into two distinct stages: perceptual compression via autoencoders and semantic modeling via latent score diffusion.

### Comparative Feature & Tradeoff Matrix

| Generation Domain | Training Compute | Spatial Resolution | Attention Sequence Length | Primary Modern Implementations |
| :--- | :--- | :--- | :--- | :--- |
| **Pixel Diffusion** | Extremely Heavy ($>100$ GPU-years) | Up to $256 \times 256$ | $65,536$ tokens | DDPM, Imagen (base) |
| **Cascaded Diffusion** | Heavy | $1024 \times 1024$ (multi-stage) | Multi-resolution U-Nets | DALL-E 2, Imagen (SR) |
| **Latent Diffusion (LDM)** | Efficient ($10\text{--}20\times$ faster) | $1024 \times 1024+$ | $4,096$ tokens ($8\times$ down) | Stable Diffusion (1.5, XL, SD3), Flux |

### Common Traps & Numerical Fixes
- **VAE Latent Scale Drift**: Unnormalized VAE latents can have variance $> 5.0$, crashing standard DDPM schedules. Always scale latents by the empirical scaling factor (e.g. `z = z * 0.18215` in Stable Diffusion) to enforce unit variance.
- **Score Sign Confusion**: Remember that score is $\nabla \log p(x)$, pointing toward modes. Noise prediction points away from modes: $\nabla \log p(x_t) = -\frac{\epsilon}{\sigma_t}$. Forgetting the negative sign causes gradient ascent into pure static.

---

## Master End-to-End Simulation

```python
import torch
import torch.nn as nn

def master_ldm_simulation():
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 32, 32
    
    # 1. Simulate Pretrained VAE Encoder: (B, 3, 32, 32) -> (B, 4, 8, 8)
    encoder = nn.Conv2d(3, 4, kernel_size=4, stride=4) # 4x spatial compression
    decoder = nn.ConvTranspose2d(4, 3, kernel_size=4, stride=4)
    
    x0 = torch.randn(B, C, H, W)
    z0 = encoder(x0) * 0.18215 # Scale latent to unit variance
    assert z0.shape == (B, 4, 8, 8)
    
    # 2. Diffusion in Latent Space
    T = 10
    betas = torch.linspace(0.01, 0.1, T)
    alphas_bar = torch.cumprod(1.0 - betas, dim=0)
    
    t = 5
    eps_latent = torch.randn_like(z0)
    zt = torch.sqrt(alphas_bar[t]) * z0 + torch.sqrt(1.0 - alphas_bar[t]) * eps_latent
    
    # 3. Score-based Tweedie reconstruction of clean latent
    # Injected score: s(zt) = -eps_latent / sqrt(1 - alpha_bar_t)
    sigma_t = torch.sqrt(1.0 - alphas_bar[t])
    score = -eps_latent / sigma_t
    
    # Tweedie: E[z0 | zt] = (zt + sigma_t^2 * score) / sqrt(alpha_bar_t)
    recovered_z0 = (zt + (sigma_t**2) * score) / torch.sqrt(alphas_bar[t])
    assert torch.allclose(z0, recovered_z0, atol=1e-5)
    
    # 4. Decode recovered latent back to pixel space
    recovered_x0 = decoder(recovered_z0 / 0.18215)
    assert recovered_x0.shape == (B, C, H, W)
    
    print(f"Master LDM simulation PASSED: pixel shape={x0.shape} -> latent shape={z0.shape} -> recovered={recovered_x0.shape}")

master_ldm_simulation()
```

---

## Topic 1: Classical Statistics & Tweedie's Formula: The Empirical Bayes Bridge from Posterior Mean to Score

### Where this sits on the master map
We begin by establishing the statistical bedrock that unifies Gaussian denoising with score matching: Tweedie's identity in empirical Bayes analysis.

### Board / screenshot
```
                TWEEDIE'S IDENTITY IN EMPIRICAL BAYES
                
      Observation:  x_t ~ N( x_0, sigma_t^2 * I )
      
      Theorem:      E[ x_0 | x_t ] = x_t + sigma_t^2 * nabla_{x_t} log p(x_t)
                    \____________/   \_/   \________________________________/
                      Optimal         Noisy         Score Correction Vector
                     Denoised        Current          (Pushes toward data)
                      Vector          State
```
*Notice: Tweedie's formula reveals that the optimal denoiser is an additive correction determined by the score of the marginal distribution.*

### What he is establishing
The instructor introduces Tweedie's formula from classical statistics. When a clean signal $x_0$ is perturbed by additive Gaussian noise with variance $\sigma_t^2$, the marginal distribution of the noisy measurement $x_t$ is the convolution of the data distribution with a Gaussian.

Tweedie showed that the conditional expectation $\mathbb{E}[x_0 \mid x_t]$—which is the mathematically optimal Minimum Mean Squared Error (MMSE) estimator—can be calculated directly from the gradient of the log-marginal density:
$$\mathbb{E}[x_0 \mid x_t] = x_t + \sigma_t^2 \nabla_{x_t} \log p(x_t)$$

This is a profound revelation for deep learning. It proves that learning to denoise noisy images is mathematically equivalent to learning the score function $\nabla_{x_t} \log p(x_t)$ of the perturbed data distribution. The denoiser does not need to know the explicit prior $p(x_0)$; it only needs to learn the local score gradient.

### Contrastive Analysis: Why X, Not Y?
- **Why is Tweedie's formula preferred over naive Bayesian inversion?**  
  Naive Bayesian inversion requires computing the intractable integral $p(x_t) = \int p(x_0) p(x_t \mid x_0) dx_0$. Tweedie bypasses the integral entirely, proving that the conditional expectation depends only on the spatial gradient $\nabla_{x_t} \log p(x_t)$.

### Active Comprehension Checks
1. *Question*: What happens to Tweedie's correction $\sigma_t^2 \nabla \log p(x_t)$ when $\sigma_t \to 0$?  
   *Answer*: The variance $\sigma_t^2 \to 0$, so the correction vanishes and $\mathbb{E}[x_0 \mid x_t] \to x_t$, confirming that clean data requires zero denoising.

### Analogy for this topic only
A submarine navigating through murky ocean water with a depth sounder. Rather than knowing the topography of the entire ocean floor, the captain simply measures the local seabed slope beneath the hull to steer away from the ocean trench.
*In lecture words: Tweedie connects local gradient slopes to global Bayesian recovery.*

### Local picture
```
   [ Noisy Input x_t ] + [ Variance * Local Score Gradient ] ===> [ Clean Reconstruction ]
```
*Notice: Denoising is a gradient-directed translation.*

### Bridge
With Tweedie's formula established, we now formalize how DDPM's noise predictor $\epsilon_\theta$ functions as an exact score estimator.

---

## Topic 2: The Score-Matching Duality: Formulating DDPM as a Continuous Vector Field Estimator

### Where this sits on the master map
We bridge Ho et al.'s DDPM formulation to Song & Ermon's Score-Based Generative Models (SGM), demonstrating that both are identical algorithms with different notation.

### Board / screenshot
```
                THE DDPM-SCORE MATCHING DUALITY
                
   DDPM Parameterization:            Score Matching Parameterization:
   x_t = sqrt(a_bar)*x_0 + sig*eps   s_theta(x_t, t) = nabla_{x_t} log p(x_t)
   Loss: || eps - eps_theta ||^2     Loss: || s_theta - nabla log q ||^2
   
                 Equivalence:  s_theta(x_t, t) = - eps_theta(x_t, t) / sigma_t
```
*Notice: The noise prediction network $\epsilon_\theta$ is simply the negative score vector scaled by the standard deviation.*

### What he is establishing
The instructor details the exact mathematical duality between DDPM and score-based modeling.
In DDPM, the forward jump is:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
Dividing by $\sqrt{\bar{\alpha}_t}$ normalizes the data coefficient:
$$\frac{x_t}{\sqrt{\bar{\alpha}_t}} = x_0 + \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \epsilon \equiv x_0 + \sigma_t \epsilon$$
The true conditional score of this Gaussian perturbation is:
$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t} = -\frac{\sqrt{1 - \bar{\alpha}_t}\epsilon}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

Therefore, when the neural network $\epsilon_\theta(x_t, t)$ minimizes the MSE loss $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$, it is learning:
$$s_\theta(x_t, t) \approx \nabla_{x_t} \log p(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

This duality establishes that DDPM is not merely an empirical heuristic; it is a rigorous estimator of the continuous Stein score of the data distribution smoothed by Gaussian noise.

### Contrastive Analysis: Why X, Not Y?
- **Why did Ho et al. predict $\epsilon$ rather than predicting score $s$ directly?**  
  Predicting $\epsilon$ ensures the regression target has unit variance $\text{Var}(\epsilon) = 1$ across all $t$. The score $s = -\epsilon / \sigma_t$ scales as $\mathcal{O}(1/\sigma_t)$, exploding to $\pm \infty$ as $\sigma_t \to 0$, causing extreme numerical instability in neural network loss gradients.

### Active Comprehension Checks
1. *Question*: Why does the score vector point in the direction opposite to the injected noise ($-\epsilon$)?  
   *Answer*: Because noise $\epsilon$ knocks the image away from the high-probability data manifold; the score points uphill toward higher probability, which is the negative of the noise direction.

### Analogy for this topic only
Newton's third law of motion: if a cannonball is fired south ($\epsilon$), the recoil vector points north ($-\epsilon$). The score is the recoil vector pointing back to the cannon.
*In lecture words: The score vector points in the exact opposite direction of the corruption noise.*

### Local picture
```
   Data Manifold <====(Score Vector: -eps/sigma)==== Noisy Point x_t
```
*Notice: Direction points toward data density.*

### Bridge
Now that we view the denoiser as a score estimator, we examine how to guide this vector field using external classifiers.

---

## Topic 3: Classifier-Guided Diffusion: Steering Reverse Trajectories with External Noise-Robust Classifiers

### Where this sits on the master map
We study Dhariwal & Nichol's Classifier Guidance, analyzing how an auxiliary classifier steers unconditional diffusion sampling toward specific class labels.

### Board / screenshot
```
                CLASSIFIER GUIDANCE VECTOR FIELD
                
    Unconditional Score:      nabla_{x_t} log p(x_t)         (Makes image realistic)
                                        +
    Classifier Gradient:  s * nabla_{x_t} log p(y | x_t)     (Steers toward class y)
                                       ===
    Guided Score:             nabla_{x_t} log p(x_t | y)     (Realistic image of class y)
```
*Notice: The hyperparameter $s$ amplifies the classifier gradient, trading off diversity for fidelity.*

### What he is establishing
The instructor derives Classifier Guidance using Bayes' rule. To sample from the conditional distribution $p(x_0 \mid y)$, we need the score of the conditional distribution at every noisy step:
$$\nabla_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y \mid x_t)$$

To give the user control over how strongly the condition is enforced, Dhariwal & Nichol introduced a guidance scale $s \ge 1$:
$$\tilde{\nabla}_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + s \nabla_{x_t} \log p_\phi(y \mid x_t)$$

In terms of the noise predictor:
$$\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t) - s \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p_\phi(y \mid x_t)$$

To implement this, one must train a separate image classifier $p_\phi(y \mid x_t)$ on noisy images $x_t$ across all noise levels $t \in [1, T]$. At every inference step, backpropagation through the classifier computes $\nabla_{x_t} \log p_\phi(y \mid x_t)$ to nudge the reverse mean $\mu_\theta$.

### Contrastive Analysis: Why X, Not Y?
- **Why can't we use a pretrained ResNet-50 trained on ImageNet?**  
  Standard ResNets are trained on clean images. When presented with an image at $t=500$ (where signal-to-noise ratio is $< 0.1$), standard classifiers output uniform random probabilities ($1/K$) with meaningless, chaotic gradients. The classifier must be explicitly trained with noisy images $x_t$ and time embeddings $t$.

### Active Comprehension Checks
1. *Question*: What is the computational cost of Classifier Guidance at each reverse step?  
   *Answer*: One forward pass through the diffusion U-Net, plus one forward and one backward pass through the classifier network.

### Analogy for this topic only
An airplane autopilot. The diffusion model keeps the plane airborne and stable (unconditional score); an air traffic controller sends steering instructions over the radio (classifier gradient) to align the plane with runway 24R.
*In lecture words: The classifier provides an external steering gradient.*

### Local picture
```
   [ Denoiser Forward ] ---> [ Classifier Forward & Backward ] ---> [ Guided Update ]
```
*Notice: Requires computing gradients through external classifier.*

### Bridge
While effective, training noise-robust classifiers for text prompts is impractical. Next, we derive Classifier-Free Guidance (CFG), which eliminates the external classifier entirely.

---

## Topic 4: Classifier-Free Guidance (CFG): Derivation and Mechanics of Implicit Score Guidance

### Where this sits on the master map
We examine Ho & Salimans' Classifier-Free Guidance (CFG), the industry-standard guidance mechanism powering modern text-to-image synthesis.

### Board / screenshot
```
                CLASSIFIER-FREE GUIDANCE (CFG) MECHANICS
                
    Train:  Single network eps_theta(x_t, t, c) with random label dropout (c -> empty)
    
    Sample: Evaluate both conditional and unconditional branches:
            eps_uncond = eps_theta(x_t, t, empty)
            eps_cond   = eps_theta(x_t, t, y)
            
    Extrapolate:  eps_guided = eps_uncond + s * ( eps_cond - eps_uncond )
```
*Notice: Extrapolating beyond $\epsilon_{\text{cond}}$ by scale $s$ creates an implicit classifier without training one.*

### What he is establishing
The instructor details the elegant derivation of CFG. Using Bayes' rule, the classifier score is:
$$\nabla_{x_t} \log p(y \mid x_t) = \nabla_{x_t} \log p(x_t \mid y) - \nabla_{x_t} \log p(x_t)$$
Substituting this into the guided score formula:
$$\tilde{\nabla}_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + s \left( \nabla_{x_t} \log p(x_t \mid y) - \nabla_{x_t} \log p(x_t) \right)$$
$$= (1 - s) \nabla_{x_t} \log p(x_t) + s \nabla_{x_t} \log p(x_t \mid y)$$

Converting this score equation directly to noise predictions ($\nabla \log p \propto -\epsilon$):
$$\tilde{\epsilon}_\theta(x_t, t, y) = (1 - s) \epsilon_\theta(x_t, t, \emptyset) + s \epsilon_\theta(x_t, t, y)$$
$$= \epsilon_\theta(x_t, t, \emptyset) + s \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset) \right)$$

This remarkable result proves that we do not need a classifier! A single diffusion model trained with conditional inputs $y$ and unconditional inputs $\emptyset$ (via $10\text{--}20\%$ label dropout) can compute its own implicit guidance vector.

### Contrastive Analysis: Why X, Not Y?
- **Why does CFG produce better images than Classifier Guidance?**  
  Classifiers can be fooled by adversarial high-frequency noise that tricks the classifier without looking like the target class. CFG computes guidance entirely within the generative image manifold, making adversarial shortcuts impossible.

### Active Comprehension Checks
1. *Question*: What happens in CFG when the guidance scale is set to $s = 1$?  
   *Answer*: $\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, t, y)$, which is standard conditional diffusion without guidance extrapolation.

### Analogy for this topic only
An artist painting with a reference photo. The unconditional branch is their muscle memory of natural lighting. The difference vector $(\epsilon_c - \epsilon_u)$ is what makes this specific scene unique. Turning up the guidance knob $s$ makes the unique features stand out vividly.
*In lecture words: CFG implicitly computes the classifier gradient directly from generative predictions.*

### Local picture
```
   [ eps_uncond ] ===(Difference Vector)===> [ eps_cond ] ===(Scale s Extrapolation)===> [ Guided eps ]
```
*Notice: Linear vector extrapolation.*

### Bridge
Now we tackle the computational barrier of diffusion: how can we scale these models to high-resolution images without massive compute costs? We enter Latent Diffusion Models.

---

## Topic 5: Latent Diffusion Models (LDM): High-Resolution Synthesis via Pretrained Latent Space Autoencoders

### Where this sits on the master map
We investigate Latent Diffusion Models (LDM), the breakthrough architecture introduced by Rombach et al. (2022) that underlies Stable Diffusion and modern generative pipelines.

### Board / screenshot
```
                LATENT DIFFUSION ARCHITECTURE (LDM)
                
    RGB Image x_0 (3 x 512 x 512)                     Reconstructed Image (3 x 512 x 512)
          |                                                           ^
     [ VAE Encoder E ]                                           [ VAE Decoder D ]
          v                                                           |
    Clean Latent z_0 (4 x 64 x 64) <-------------------------- Denoised Latent z_0
          |                                                           ^
          +---> [ Latent Forward SDE ] ---> z_t ---> [ Latent U-Net ]-+
                                                           ^
                                             Text Prompt --+ (Cross-Attention)
```
*Notice: The computationally intensive diffusion process runs entirely within the $64\times$ smaller latent space.*

### What he is establishing
The instructor explains why pixel-space diffusion is computationally inefficient. Digital images contain massive spatial redundancy: adjacent pixels in a blue sky share virtually identical values. Processing $512 \times 512$ pixel tensors through 1,000 steps of a U-Net wastes enormous GPU memory on high-frequency noise that human vision cannot perceive.

LDM solves this by separating image synthesis into two distinct stages:
1. **Perceptual Compression Stage**:
   A regularized autoencoder (KL-VAE or VQ-GAN) is trained with perceptual and adversarial losses. The encoder $\mathcal{E}$ compresses an image $x_0 \in \mathbb{R}^{3 \times H \times W}$ into a compact latent representation $z_0 \in \mathbb{R}^{c \times (H/f) \times (W/f)}$, typically with downsampling factor $f = 8$.
2. **Semantic Generative Stage**:
   A diffusion U-Net operates entirely in this low-dimensional latent space:
   $$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
   The latent U-Net uses cross-attention layers to condition on text tokens, class labels, or layout bounding boxes.
3. **Synthesis & Decoding**:
   At inference time, reverse diffusion generates a clean latent $z_0$. The decoder $\mathcal{D}(z_0)$ translates the latent back to full-resolution RGB pixel space in a single feedforward pass.

### Contrastive Analysis: Why X, Not Y?
- **Why not train the VAE and the diffusion model end-to-end together?**  
  Training them end-to-end causes training instability; the encoder's latent space would shift continuously while the diffusion model tries to learn a stationary distribution over it. Freezing the VAE provides a fixed, stationary latent manifold for the diffusion model.

### Active Comprehension Checks
1. *Question*: What is the spatial dimension of a latent tensor in Stable Diffusion for a $512 \times 512$ RGB input with $f=8$?  
   *Answer*: $64 \times 64$, with 4 channels: $(4, 64, 64)$.

### Analogy for this topic only
Shipping freight across the ocean. Instead of loading loose, unorganized cargo onto a ship (raw pixels), workers pack everything into standardized metal shipping containers (latent vectors $z_0$). The cargo ship handles containers with maximum efficiency; at the destination port, the containers are unpacked (decoder $\mathcal{D}$).
*In lecture words: LDM compresses perceptual redundancy before running semantic diffusion.*

### Local picture
```
   [ Pixel Space: 786,432 floats ] ===(VAE 8x Down)===> [ Latent Space: 16,384 floats ]
```
*Notice: 48-fold reduction in spatial tensor elements.*

### Bridge
We conclude by placing all these discrete models into the grand continuous-time framework of Stochastic Differential Equations.

---

## Topic 6: Continuous-Time Diffusion: Unifying VP and VE SDEs with the Reverse-Time Stochastic Differential Equation

### Where this sits on the master map
We examine Song et al.'s (2021) continuous-time formulation, showing how discrete diffusion steps are Euler-Maruyama discretizations of underlying continuous stochastic differential equations.

### Board / screenshot
```
                CONTINUOUS-TIME SDE FRAMEWORK
                
    Forward SDE:   dx = f(x, t) dt + g(t) dw
    
    Reverse SDE:   dx = [ f(x, t) - g(t)^2 * nabla_x log p_t(x) ] dt + g(t) dw~
    
    Probability
    Flow ODE:      dx = [ f(x, t) - 1/2 * g(t)^2 * nabla_x log p_t(x) ] dt
```
*Notice: The Probability Flow ODE shares the exact same marginal probability densities $p_t(x)$ as the stochastic SDE.*

### What he is establishing
The instructor introduces the continuous-time framework. When the time step $\Delta t \to 0$, the discrete forward chain becomes an Itô SDE:
$$dx = f(x, t) dt + g(t) d\mathbf{w}$$
where $\mathbf{w}$ is standard Brownian motion.
1. **Variance Preserving (VP) SDE**:
   Matches DDPM with linear beta schedule:
   $$dx = -\frac{1}{2} \beta(t) x dt + \sqrt{\beta(t)} d\mathbf{w}$$
2. **Variance Exploding (VE) SDE**:
   Matches Score-Based Generative Models (SGM) with geometric noise:
   $$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} d\mathbf{w}$$

By Anderson's reverse-time theorem, the continuous reverse process is also an SDE:
$$dx = \left[ f(x, t) - g(t)^2 \nabla_x \log p_t(x) \right] dt + g(t) d\bar{\mathbf{w}}$$
where $d\bar{\mathbf{w}}$ is reverse-time Brownian motion.

Crucially, Song et al. proved there exists an associated **Probability Flow ODE**:
$$dx = \left[ f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x) \right] dt$$
This deterministic ODE trajectory has the exact same marginal probability distribution $p_t(x)$ at every point in time as the stochastic SDE! This enables exact likelihood computation, deterministic latent encoding, and ultra-fast ODE integration via adaptive step-size solvers (Runge-Kutta, DPMSolver).

### Contrastive Analysis: Why X, Not Y?
- **Why use the Probability Flow ODE instead of the reverse SDE?**  
  The Probability Flow ODE allows exact inversion: taking an existing image $x_0$ and integrating forward to find its unique latent noise representation $x_T$, which enables high-precision image editing, inpainting, and style transfer.

### Active Comprehension Checks
1. *Question*: What is the difference between the drift coefficient in the reverse SDE versus the Probability Flow ODE?  
   *Answer*: The reverse SDE subtracts $g(t)^2 \nabla \log p_t(x)$, whereas the ODE subtracts $\frac{1}{2} g(t)^2 \nabla \log p_t(x)$ because it contains no stochastic noise term.

### Analogy for this topic only
A flowing river. The forward SDE is water flowing downhill into the ocean. The reverse SDE is swimming upstream against the current while fighting turbulent eddies ($d\bar{\mathbf{w}}$). The Probability Flow ODE is a smooth glass tube built through the riverbed, allowing a boat to travel upstream with zero turbulence.
*In lecture words: The Probability Flow ODE is the deterministic twin of the stochastic diffusion process.*

### Local picture
```
   Forward SDE (Noise Injection) <=======> Reverse SDE / Probability Flow ODE (Synthesis)
```
*Notice: Complete continuous-time duality.*

### Bridge
We have now mastered the comprehensive mathematical theory of diffusion models from discrete ELBOs to continuous SDEs, CFG, and Latent Diffusion. Let's inspect critical workplace debugging scenarios.

---

## Workplace Debugging Scenarios

### Scenario 1: The Latent Variance Mismatch Crash
**Incident:** An engineer builds a custom Latent Diffusion pipeline using a pretrained VAE and a DDPM U-Net. During training, the U-Net loss fails to converge and generated images look like multicolored salt-and-pepper static.  
**Mathematical Root Cause:** The developer fed raw VAE latent encodings $z = \mathcal{E}(x)$ directly into the diffusion schedule without scaling. The pretrained VAE produced latents with variance $\text{Var}(z) \approx 5.4$. Because standard DDPM schedules assume input data has unit variance ($\text{Var}=1.0$), the noise schedule injected negligible relative noise, corrupting the signal-to-noise ratio $\bar{\alpha}_t / (1 - \bar{\alpha}_t)$.  
**Debugging Steps:**
1. Compute `z.var()`. Observe variance is $5.4$ instead of $1.0$.
2. Scale latents by the standard normalization constant $1 / \sqrt{5.4} \approx 0.18215$ (or compute batch standard deviation).  
**Code Fix:**
```python
# CORRECT LATENT NORMALIZATION:
scale_factor = 0.18215
latent = encoder(image) * scale_factor
# At inference:
image = decoder(sampled_latent / scale_factor)
```

---

### Scenario 2: The Classifier Gradient Disconnection Bug
**Incident:** During classifier guidance inference, varying the class label $y$ produces identical unconditional samples with zero classifier steering.  
**Mathematical Root Cause:** The developer called `torch.no_grad()` across the entire sampling loop to save memory. This disabled gradient calculation, causing `torch.autograd.grad(logits[y], xt)` to return zero or fail silently, effectively dropping the classifier guidance term.  
**Debugging Steps:**
1. Check `classifier_grad.norm()`. Observe it evaluates to $0.0$.
2. Ensure input `xt` has `requires_grad=True` and enable autograd context when evaluating the classifier.  
**Code Fix:**
```python
# CORRECT CLASSIFIER GRADIENT COMPUTATION:
with torch.enable_grad():
    xt_in = xt.detach().requires_grad_(True)
    logits = classifier(xt_in, t)
    log_prob = torch.log_softmax(logits, dim=-1)
    classifier_grad = torch.autograd.grad(log_prob[:, y].sum(), xt_in)[0]
guided_eps = model_eps - s * torch.sqrt(1.0 - alpha_bar_t) * classifier_grad
```

---

## References & Further Reading

For complete citations, seminal papers, academic lecture links, and official implementations, see [./references.md](./references.md).
