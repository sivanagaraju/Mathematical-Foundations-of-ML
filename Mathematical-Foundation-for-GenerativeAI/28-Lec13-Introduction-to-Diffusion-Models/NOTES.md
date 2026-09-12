> [!NOTE]
> **Orientation**: This package builds upon foundational probability concepts. For rigorous definitions of Gaussian kernels, Markov chains, and variance preservation, see [./PREREQUISITES.md](./PREREQUISITES.md).

# Lecture 13: Introduction to Diffusion Models (DDPM Foundations)

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Master End-to-End Simulation](#master-end-to-end-simulation)
3. [Topic 1: Generative Modeling Taxonomy: Failure Modes of GANs, VAEs, and Normalizing Flows](#topic-1-generative-modeling-taxonomy-failure-modes-of-gans-vaes-and-normalizing-flows)
4. [Topic 2: Thermodynamics Inspiration: Non-Equilibrium Diffusion and Structure Destruction](#topic-2-thermodynamics-inspiration-non-equilibrium-diffusion-and-structure-destruction)
5. [Topic 3: The Forward Markov Diffusion Process: Step-by-Step Perturbation Kernel q(x_t|x_{t-1})](#topic-3-the-forward-markov-diffusion-process-step-by-step-perturbation-kernel-qx_tx_t-1)
6. [Topic 4: Closed-Form Arbitrary Step Sampling: Jump Kernel via Cumulative Variance alpha_bar_t](#topic-4-closed-form-arbitrary-step-sampling-jump-kernel-via-cumulative-variance-alpha_bar_t)
7. [Topic 5: The Reverse Generative Markov Process: Denoising Path p_theta(x_{t-1}|x_t)](#topic-5-the-reverse-generative-markov-process-denoising-path-p_thetax_t-1x_t)
8. [Topic 6: Noise Schedules and Variance Dynamics: Linear vs Cosine Beta Schedules](#topic-6-noise-schedules-and-variance-dynamics-linear-vs-cosine-beta-schedules)
9. [Workplace Debugging Scenarios](#workplace-debugging-scenarios)
10. [References & Further Reading](#references--further-reading)

---

## Executive Summary

Diffusion models revolutionize generative modeling by replacing single-step mapping with a multi-step thermodynamic transport process. By decomposing generation into a sequence of small denoising steps, the model circumvents the adversarial instability of GANs and the blurry reconstructions of VAEs.

```
                  THE DIFFUSION PARADIGM: FORWARD VS REVERSE
                  
   q(x_0) : Clean Data                    q(x_t|x_{t-1})                p(x_T) ~ N(0, I)
 +-----------------------+        FORWARD PROCESS (Fixed)           +-----------------------+
 |  Structured Image     | --------------------------------------> |  Pure Gaussian Noise  |
 |  x_0 ~ q(x_0)         | <-------------------------------------- |  x_T ~ N(0, I)        |
 +-----------------------+        REVERSE PROCESS (Learned)         +-----------------------+
                                  p_theta(x_{t-1}|x_t)
```
*Figure 1: High-level architectural duality of Diffusion Models: Fixed forward entropy maximization vs learned reverse denoising transport.*

### Scenario Walkthrough
1. **Forward Pass (Data $\to$ Noise)**: Take an image $x_0$. At each step $t \in \{1, \dots, T\}$, add a calibrated sliver of Gaussian noise according to schedule $\beta_t$. By $t=T$, all data structure is eradicated into $\mathcal{N}(0, I)$.
2. **Jump Calculation**: Using $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$, generate intermediate state $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$ in $\mathcal{O}(1)$ time.
3. **Reverse Generation (Noise $\to$ Data)**: Sample pure noise $x_T \sim \mathcal{N}(0, I)$. Sequentially query neural denoiser $p_\theta(x_{t-1} \mid x_t)$ to strip off noise step-by-step until reaching high-fidelity sample $\hat{x}_0$.

### Failure / Contrast Path
In GANs, the generator tries to transform standard normal noise to a complex 100-manifold image in one single feedforward pass. When the discriminator outpaces the generator, vanishing gradients occur; when the generator finds a local minimum, mode collapse strikes. In diffusion models, each transition solves a well-conditioned regression problem.

### STOP / Out of Scope
This lecture establishes the **foundational formulation, forward kernel, jump shortcut, and reverse process intuition**. Detailed variational lower bound (ELBO) expansions, score matching derivations, and classifier-free guidance are covered in subsequent lectures.

### Load-Bearing Claims
- The forward process $q(x_{1:T} \mid x_0)$ requires **zero learnable parameters**; it is entirely governed by predefined schedules $\beta_t$.
- The forward process is a **first-order Markov chain** that guarantees asymptotic convergence to standard isotropic Gaussian noise.
- The arbitrary step jump kernel $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$ enables training on arbitrary time steps without iterating through $1, \dots, t-1$.

### Comparative Feature & Tradeoff Matrix

| Method | Training Stability | Sample Diversity | Inference Speed | Likelihood Bounds | Mode Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GANs** | Extremely Unstable (Minimax) | High (when successful) | Instant ($\mathcal{O}(1)$) | None (Implicit) | Prone to Mode Collapse |
| **VAEs** | Stable (ELBO) | Medium (Blurry) | Instant ($\mathcal{O}(1)$) | Yes (ELBO) | Complete Coverage |
| **Normalizing Flows** | Stable (Exact Likelihood) | High | Instant ($\mathcal{O}(1)$) | Exact Analytical | Complete Coverage |
| **Diffusion (DDPM)** | Highly Stable (MSE Regression) | State-of-the-Art | Slow ($\mathcal{O}(T)$ steps) | Yes (Variational Bound) | Unmatched Multi-Modal Mode Coverage |

### Common Traps & Numerical Fixes
- **Variance Explosion**: Adding noise without scaling the previous state blows up variance to $\infty$. Always multiply $x_{t-1}$ by $\sqrt{1 - \beta_t}$ to preserve unit variance.
- **Off-by-one Time Indexing**: Remember $\alpha_t = 1 - \beta_t$. $\bar{\alpha}_t$ at $t=1$ is $\alpha_1$, not $\alpha_0$.
- **Dynamic Range Mismatch**: Diffusion equations assume $x_0 \in [-1, 1]$. Feeding raw $[0, 255]$ pixel values corrupts variance tracking.

---

## Master End-to-End Simulation

```python
import torch
import torch.nn as nn

def master_diffusion_simulation():
    # Set seed for reproducible validation
    torch.manual_seed(42)
    B, D, T = 16, 8, 100
    
    # 1. Generate clean data x0 in [-1, 1]
    x0 = torch.tanh(torch.randn(B, D))
    assert x0.shape == (B, D)
    
    # 2. Linear noise schedule beta_t from 1e-4 to 0.02
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    
    # 3. Simulate step-by-step Markov forward process
    xt_step = x0.clone()
    for t in range(T):
        noise = torch.randn_like(xt_step)
        xt_step = torch.sqrt(alphas[t]) * xt_step + torch.sqrt(betas[t]) * noise
        
    # 4. Simulate closed-form jump to t=T-1
    t_target = T - 1
    eps = torch.randn_like(x0)
    xt_jump = torch.sqrt(alphas_bar[t_target]) * x0 + torch.sqrt(1.0 - alphas_bar[t_target]) * eps
    
    # 5. Verify asymptotic convergence to N(0, I)
    assert xt_jump.shape == (B, D)
    assert alphas_bar[-1] < 0.20, "Cumulative alpha did not decay sufficiently"
    print(f"Simulation success: x0 norm={x0.norm(dim=-1).mean():.3f}, "
          f"x_T jump norm={xt_jump.norm(dim=-1).mean():.3f}, "
          f"alpha_bar_T={alphas_bar[-1]:.4f}")

master_diffusion_simulation()
```

---

## Topic 1: Generative Modeling Taxonomy: Failure Modes of GANs, VAEs, and Normalizing Flows

### Where this sits on the master map
We begin by surveying the landscape of deep generative modeling. Understanding the structural limitations of VAEs (blurry images, prior hole), GANs (adversarial instability, mode collapse), and Normalizing Flows (restrictive invertible architectures) provides the direct motivation for diffusion models.

### Board / screenshot
```
                      TAXONOMY OF GENERATIVE MODELS
                      
       [ VAEs ]                 [ GANs ]             [ Normalizing Flows ]
  P(x) >= E[log p(x|z)]      Minimax Game           Exact Invertible Transform
     - Blurry Samples         - Mode Collapse          - Massive Parameter Cost
     - Posterior Collapse     - Divergent Gradients    - Restricted Architectures
            \                     |                     /
             \                    |                    /
              +-------------------+-------------------+
                                  |
                                  v
                       [ DIFFUSION MODELS ]
                   - Stable Regression Loss (MSE)
                   - Complete Mode Coverage
                   - Unconstrained Neural Architectures (U-Net)
```
*Notice: Diffusion models bypass the architectural restrictions of Flows and the adversarial saddle points of GANs by framing generation as iterative denoising.*

### What he is establishing
The instructor contrasts the generative paradigms. In traditional approaches:
- **VAEs**: Optimize the Evidence Lower Bound ([01-Latent_Variable_Models_and_ELBO.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/05-Latent_Variable_Models.md)). The Gaussian assumption on the posterior and the latent bottleneck force the decoder to output an average of plausible images, resulting in blurriness.
- **GANs**: Pits a generator against a discriminator in a zero-sum game. Optimization frequently falls into cyclic dynamics or mode collapse where the generator outputs only a fraction of the data modes.
- **Normalizing Flows**: Requires bijective mappings where the Jacobian determinant is easily computable, drastically constraining the network layers.

Diffusion models abandon the attempt to produce a complex image from a latent vector in a single step. Instead, they distribute the generative workload over hundreds of infinitesimal steps, each of which is easily modeled by an unconstrained neural network.

### Contrastive Analysis: Why X, Not Y?
- **Why multi-step diffusion rather than single-step GAN generator?**  
  In a single step, the neural network must map a simple distribution to a highly convoluted, multi-modal manifold. One mistake creates unrealistic artifacts. In diffusion, each step only has to undo a tiny sliver of Gaussian noise, which is well-modeled by a standard MSE regression loss.

### Active Comprehension Checks
1. *Question*: Why do Normalizing Flows require invertible transformations while Diffusion models do not?  
   *Answer*: Flows compute exact log-likelihoods using the change-of-variables formula ($\log p(x) = \log p(z) + \log |\det J_f^{-1}|$), which requires $f$ to be bijective with a tractable Jacobian. Diffusion models use a Markov latent hierarchy and optimize an ELBO, allowing arbitrary architectures like U-Nets.

### Analogy for this topic only
Imagine sculpting a marble statue. A GAN attempts to take a single sledgehammer strike to carve David from a raw boulder—if the angle is slightly off, the statue shatters. A diffusion model acts like fine sandpaper, gently polishing away fractions of a millimeter across 1,000 passes.
*In lecture words: We trade single-shot speed for multi-step stability and fidelity.*

### Local picture
```
   Step-by-Step Polishing (Diffusion) vs Single Strike (GAN)
   [Raw Stone] ---> [Rough Outline] ---> [Refined Form] ---> [David]
   (Diffusion: 1000 gentle, highly stable denoising adjustments)
```
*Notice: Micro-adjustments guarantee numerical stability.*

### Bridge
Having diagnosed why existing models struggle, we now examine the physics foundation that inspired this multi-step solution: non-equilibrium thermodynamics.

---

## Topic 2: Thermodynamics Inspiration: Non-Equilibrium Diffusion and Structure Destruction

### Where this sits on the master map
Diffusion models did not originate in computer science; they were directly adapted from non-equilibrium statistical mechanics by Jascha Sohl-Dickstein et al. in 2015.

### Board / screenshot
```
                THERMODYNAMICS: FORWARD DISPERSION VS REVERSE CONDENSATION
                
     t = 0 (Low Entropy)                   t = T/2                        t = T (Max Entropy)
   +-----------------------+        +-----------------------+        +-----------------------+
   |   * * *               |  --->  |    *    *   *   *     |  --->  |   *   *   *   *   *   |
   |  * * * *              |        |   *   *   *   *       |        | *   *   *   *   *   * |
   |   * * *               |        |     *   *   *         |        |   *   *   *   *   *   |
   +-----------------------+        +-----------------------+        +-----------------------+
   Pristine Data Droplet               Intermediate Dispersion           Uniform Thermal Noise
```
*Notice: The forward process monotonically increases entropy until the data distribution becomes indistinguishable from pure noise.*

### What he is establishing
In physical diffusion, a substance dissolved in a liquid diffuses from regions of high concentration to regions of low concentration until uniform equilibrium is reached. Mathematically:
- The forward process is governed by Langevin dynamics or Brownian motion.
- Information (order, low entropy) is systematically destroyed.
- A remarkable theorem in statistical mechanics states that if the perturbation rate is sufficiently slow, the **time-reversed process** also possesses the same functional form as the forward process—namely, Gaussian transitions!

The instructor emphasizes: if we can learn the reverse drift vector at every time step, we can run the film backwards, converting thermal chaos back into ordered structure.

### Contrastive Analysis: Why X, Not Y?
- **Why continuous noise addition rather than deterministic blurring?**  
  Deterministic blurring (e.g., Gaussian blur filter) removes high frequencies but retains low frequencies without mapping to a known, easily sampleable prior distribution like $\mathcal{N}(0, I)$. Adding Gaussian noise simultaneously erodes structure and drives the distribution to an analytically known standard normal prior.

### Active Comprehension Checks
1. *Question*: What happens to the marginal distribution $q(x_T)$ if $T \to \infty$ and $\sum \beta_t = \infty$?  
   *Answer*: The marginal distribution converges strictly in distribution to an isotropic standard normal Gaussian $\mathcal{N}(0, I)$, irrespective of the complexity of the initial distribution $q(x_0)$.

### Analogy for this topic only
Drop a dye cube into warm water. At first, the boundaries of the dye cube are sharp. Over time, thermal agitation knocks dye molecules apart until the liquid is a uniform tint. Reversing this requires an intelligent demon pushing every molecule back toward its cluster center.
*In lecture words: Forward process increases entropy; reverse process learns the anti-entropy vector field.*

### Local picture
```
   [Low Entropy: Distinct Image] ===(Entropy Increase)===> [High Entropy: White Noise]
```
*Notice: Maximum entropy corresponds to standard Gaussian distribution.*

### Bridge
Now that we understand the thermodynamic concept, let us write out the exact discrete-time mathematical equations governing the forward chain.

---

## Topic 3: The Forward Markov Diffusion Process: Step-by-Step Perturbation Kernel $q(x_t \mid x_{t-1})$

### Where this sits on the master map
We formalize the forward process as a discrete-time Markov chain defined over $T$ steps, specifying the exact conditional transition kernel $q(x_t \mid x_{t-1})$.

### Board / screenshot
```
                THE DISCRETE FORWARD MARKOV CHAIN
                
       q(x_1|x_0)             q(x_2|x_1)                     q(x_T|x_{T-1})
  x_0 ------------> x_1 --------------> x_2 ... ----------> x_T ~ N(0, I)
   |                 |                   |                   |
 Clean            Slight             Moderate             Complete
 Data              Haze                Noise                Noise
```
*Notice: Each state depends only on the state immediately preceding it ($x_{t-1}$).*

### What he is establishing
The forward process is defined by:
$$q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$$
where each transition is a Gaussian distribution:
$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

Key components:
1. **Variance schedule** $\beta_1, \beta_2, \dots, \beta_T$: Hyperparameters chosen such that $\beta_1$ is very small (e.g., $10^{-4}$) and $\beta_T$ is larger (e.g., $0.02$).
2. **Signal retention scale** $\sqrt{1 - \beta_t}$: Ensures that $\text{Var}(x_t) = 1$ if $\text{Var}(x_{t-1}) = 1$.
3. **No learnable parameters**: The forward process is fixed. No gradient descent is performed on $q$.

### Contrastive Analysis: Why X, Not Y?
- **Why set mean to $\sqrt{1 - \beta_t} x_{t-1}$ instead of $x_{t-1}$?**  
  If we set mean to $x_{t-1}$, then $\text{Var}(x_t) = \text{Var}(x_{t-1}) + \beta_t$. Over 1,000 steps, variance would grow to $1 + \sum \beta_t \approx 1 + 20 = 21$. Numbers would explode outside the numerical stability range of neural networks.

### Active Comprehension Checks
1. *Question*: If $x_{t-1} = 2.0$, $\beta_t = 0.19$, and sampled noise $\epsilon = 1.0$, compute $x_t$.  
   *Answer*: $\sqrt{1 - \beta_t} = \sqrt{0.81} = 0.9$. $\sqrt{\beta_t} = \sqrt{0.19} \approx 0.43589$.  
   $x_t = 0.9 \times 2.0 + 0.43589 \times 1.0 = 1.8 + 0.43589 = 2.23589$.

### Analogy for this topic only
A vintage photograph left out in the desert sun. Every day ($t$), 1% of the silver emulsion pigment fades away ($\sqrt{1 - \beta}$), and a fine mist of sand particles ($\sqrt{\beta} \epsilon$) settles onto the surface.
*In lecture words: The forward process is a fixed, parameterized destruction channel.*

### Local picture
```
   x_{t-1} -----[ Scale by sqrt(1-beta_t) ]-----+
                                                |-----> x_t
   eps     -----[ Scale by sqrt(beta_t)   ]-----+
```
*Notice: Two incoming vectors combine linearly.*

### Bridge
Simulating 1,000 steps one-by-one during neural network training would be prohibitively slow. Next, we derive the critical mathematical shortcut: jumping directly from $x_0$ to $x_t$ in one shot.

---

## Topic 4: Closed-Form Arbitrary Step Sampling: Jump Kernel via Cumulative Variance $\bar{\alpha}_t$

### Where this sits on the master map
This is the single most important computational efficiency trick in diffusion models. It allows a training algorithm to sample any random time step $t \sim \mathcal{U}(1, T)$ and evaluate the model in $\mathcal{O}(1)$ time.

### Board / screenshot
```
                THE O(1) JUMP KERNEL: SHORTCUTTING THE CHAIN
                
       x_0 -----------------------------------------------------> x_t
        |                     Closed Form:                         |
        |         q(x_t | x_0) = N( sqrt(alpha_bar_t) * x_0,       |
        |                           (1 - alpha_bar_t) * I )        |
        +----------------------------------------------------------+
```
*Notice: We skip computing all intermediate states $x_1, x_2, \dots, x_{t-1}$.*

### What he is establishing
Let $\alpha_t = 1 - \beta_t$, and let $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.
Using the reparameterization trick:
$$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \epsilon_0$$
$$x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1 - \alpha_2} \epsilon_1$$
Substituting $x_1$ into $x_2$:
$$x_2 = \sqrt{\alpha_2}(\sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \epsilon_0) + \sqrt{1 - \alpha_2} \epsilon_1$$
$$x_2 = \sqrt{\alpha_1 \alpha_2} x_0 + \sqrt{\alpha_2(1 - \alpha_1)} \epsilon_0 + \sqrt{1 - \alpha_2} \epsilon_1$$

Since $\epsilon_0, \epsilon_1 \sim \mathcal{N}(0, I)$ are independent, the sum of these two Gaussians is itself a Gaussian with zero mean and variance:
$$\sigma^2 = \left(\sqrt{\alpha_2(1 - \alpha_1)}\right)^2 + \left(\sqrt{1 - \alpha_2}\right)^2 = \alpha_2 - \alpha_1 \alpha_2 + 1 - \alpha_2 = 1 - \alpha_1 \alpha_2$$
By mathematical induction, for any arbitrary time step $t$:
$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### Contrastive Analysis: Why X, Not Y?
- **Why closed-form sampling rather than rolling out the Markov chain?**  
  If training required rolling out the chain, computing gradients at step $t=800$ would require backpropagating through 800 sequential operations, causing catastrophic GPU memory exhaustion and vanishing/exploding gradients. The jump kernel allows training each batch at arbitrary random time steps $t$ completely independently in $\mathcal{O}(1)$ memory and time.

### Active Comprehension Checks
1. *Question*: If $\bar{\alpha}_t = 0.64$, and clean data scalar $x_0 = 10.0$ with sampled noise $\epsilon = -1.0$, what is $x_t$?  
   *Answer*: $\sqrt{\bar{\alpha}_t} = \sqrt{0.64} = 0.8$. $\sqrt{1 - \bar{\alpha}_t} = \sqrt{0.36} = 0.6$.  
   $x_t = 0.8(10.0) + 0.6(-1.0) = 8.0 - 0.6 = 7.4$.

### Analogy for this topic only
If you know the compound interest rate formula $A = P(1 + r)^t$, you do not need to recalculate your bank account balance month-by-month for 30 years to find your retirement sum; you compute $(1+r)^{360}$ directly.
*In lecture words: The product of Gaussian convolutions collapses into a single closed-form convolution.*

### Local picture
```
    x_0  -----[ * sqrt(alpha_bar_t)     ]-----+
                                              |-----> x_t
    eps  -----[ * sqrt(1 - alpha_bar_t) ]-----+
```
*Notice: Direct two-input linear combination.*

### Bridge
With the forward destruction path fully characterized analytically, we now formulate the reverse generative path that our neural network must learn.

---

## Topic 5: The Reverse Generative Markov Process: Denoising Path $p_\theta(x_{t-1} \mid x_t)$

### Where this sits on the master map
The reverse process is where generative modeling actually happens. We define the model distribution $p_\theta(x_{0:T})$ that takes pure Gaussian noise and turns it into clean images.

### Board / screenshot
```
                THE REVERSE GENERATIVE PROCESS
                
      p_theta(x_{T-1}|x_T)           p_theta(x_0|x_1)
  x_T -------------------> x_{T-1} ... -----------> x_0 (Generated Image)
   ^                                                 ^
 Pure Gaussian                                  Pristine Data
 Noise ~ N(0, I)                                 Sample
```
*Notice: The reverse process starts from pure noise $p(x_T) = \mathcal{N}(0, I)$ and works backward to $x_0$.*

### What he is establishing
The true reverse transition $q(x_{t-1} \mid x_t)$ requires knowledge of the entire data distribution $q(x_0)$, which is intractable:
$$q(x_{t-1} \mid x_t) = \frac{q(x_t \mid x_{t-1}) q(x_{t-1})}{q(x_t)}$$
However, when $\beta_t$ is sufficiently small, $q(x_{t-1} \mid x_t)$ is also Gaussian!
We approximate this intractable distribution with a parameterized neural network $p_\theta(x_{t-1} \mid x_t)$:
$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)$$
$$p(x_T) = \mathcal{N}(x_T; 0, I)$$
$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

The neural network takes as input the noisy image $x_t$ and the current time index $t$, and outputs the predicted mean $\mu_\theta(x_t, t)$.

### Contrastive Analysis: Why X, Not Y?
- **Why condition the neural network on time index $t$?**  
  At $t=999$, the input is nearly pure noise, so the network must predict coarse, global structures. At $t=5$, the input is almost a clean image, so the network must predict fine-grained high-frequency textures. The network weights $\theta$ are shared across all $T$ time steps, so conditioning on $t$ (via sinusoidal embeddings) tells the network what scale of noise to remove.

### Active Comprehension Checks
1. *Question*: What is the starting distribution $p(x_T)$ for ancestral sampling at inference time?  
   *Answer*: Standard isotropic Gaussian $\mathcal{N}(0, I)$.

### Analogy for this topic only
A sculptor looking at a foggy silhouette versus looking at a finished bust. In heavy fog ($t \approx T$), they only decide whether the object is a human or a horse. In light mist ($t \approx 1$), they polish the eyelashes.
*In lecture words: The time embedding modulates the multi-scale attention of the denoiser.*

### Local picture
```
    x_t ----+
            |-----> [ U-Net Denoiser f_theta ] -----> mu_theta(x_t, t)
    t   ----+
```
*Notice: Time embedding $t$ is an essential input alongside noisy tensor $x_t$.*

### Bridge
Finally, how do we select the values of $\beta_1, \dots, \beta_T$? The choice of variance schedule governs the entire dynamics of information destruction.

---

## Topic 6: Noise Schedules and Variance Dynamics: Linear vs Cosine $\beta$ Schedules

### Where this sits on the master map
We examine how the schedule of $\beta_t$ controls the decay rate of $\bar{\alpha}_t$. A poorly chosen schedule destroys information too fast or leaves residual data at $t=T$.

### Board / screenshot
```
                VARIANCE SCHEDULE COMPARISON
                
  alpha_bar_t
    1.0 +-----------\
        |            \  (Cosine Schedule: Smooth Sigmoidal Retention)
        |             \
        |  Linear:     \
        |  Drops Fast   \
        |  at start      \
    0.0 +-----------------\-------------+
        t=0                            t=T
```
*Notice: The linear schedule destroys fine details too rapidly in early steps, while the cosine schedule preserves structural signal smoothly.*

### What he is establishing
1. **Linear Schedule (Ho et al. 2020)**:
   $$\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1), \quad \beta_1 = 10^{-4}, \beta_T = 0.02, T = 1000$$
   Under this schedule, $\bar{\alpha}_t$ drops sharply in the first 200 steps, causing 64x64 images to turn into unrecognizable blobs prematurely.
2. **Cosine Schedule (Nichol & Dhariwal 2021)**:
   $$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$
   Provides a linear drop in information across all time steps, preventing sudden collapse of structure.

### Contrastive Analysis: Why X, Not Y?
- **Why not constant $\beta_t = \beta$?**  
  A constant $\beta$ means $\bar{\alpha}_t = (1-\beta)^t$, an exponential decay curve. This leads to either incomplete destruction at $T$ or premature destruction within the first 50 steps.

### Active Comprehension Checks
1. *Question*: What is the condition that $\bar{\alpha}_T$ must satisfy at step $T$?  
   *Answer*: $\bar{\alpha}_T \approx 0$ (typically $< 10^{-3}$), ensuring that $x_T$ is statistically indistinguishable from pure Gaussian noise $\mathcal{N}(0, I)$.

### Analogy for this topic only
Pouring milk into coffee. If you dump a gallon in at once, all coffee contrast is lost instantly. If you add it drop by drop with a calibrated dropper, you can control the gradient of opacity smoothly.
*In lecture words: The schedule governs the rate of information decay per time interval.*

### Local picture
```
   [ Linear Schedule: Rapid Early Drop ] vs [ Cosine Schedule: Uniform Information Loss ]
```
*Notice: Cosine schedule maintains image coherence longer into the trajectory.*

### Bridge
We have covered the foundational theory, Markov kernels, closed-form sampling, and variance dynamics. Now let's explore real-world engineering failures and how to debug them.

---

## Workplace Debugging Scenarios

### Scenario 1: The Exploding Latent Scale Bug
**Incident:** An engineer training a custom diffusion model notices that training loss diverges to `NaN` within 50 iterations. Inspecting tensor activations shows intermediate $x_t$ values reaching $\pm 1000.0$.  
**Mathematical Root Cause:** The developer implemented the forward step as:
```python
# BUGGY IMPLEMENTATION: Missing sqrt(1 - beta) scaling
xt = xt_prev + torch.sqrt(beta_t) * torch.randn_like(xt_prev)
```
Without the signal contraction factor $\sqrt{1 - \beta_t}$, the variance grows additively: $\text{Var}(x_t) = \text{Var}(x_0) + \sum_{s=1}^t \beta_s$. Over hundreds of steps, the latent vector norm explodes uncontrollably.  
**Debugging Steps:**
1. Print `xt.norm()` across steps $t=0, 10, 50, 100$. Observe unbounded growth.
2. Check variance preservation condition: $\text{Var}(aX + bY) = a^2 \text{Var}(X) + b^2 \text{Var}(Y) = 1$.  
**Code Fix:**
```python
# CORRECT IMPLEMENTATION:
xt = torch.sqrt(1.0 - beta_t) * xt_prev + torch.sqrt(beta_t) * torch.randn_like(xt_prev)
```

---

### Scenario 2: The Foggy Generation Artifact at Inference
**Incident:** After converging to a low training loss, the model produces generations that look like faint image outlines obscured by heavy static haze.  
**Mathematical Root Cause:** The inference sampling loop stopped at $t=1$ instead of running down to $t=0$, or used unscaled pure noise at the final step. Alternatively, the linear noise schedule set $\beta_T$ too small, leaving $\bar{\alpha}_T = 0.35$ instead of $\bar{\alpha}_T \approx 0$. Because $x_T$ retained 35% of the prior data structure, starting inference from $\mathcal{N}(0, I)$ creates an out-of-distribution initialization clash.  
**Debugging Steps:**
1. Inspect $\bar{\alpha}_T$: calculate `torch.prod(1.0 - betas)`. If it is $> 0.01$, the prior is not pure Gaussian.
2. Verify reverse sampling loop bounds: ensure loop executes for $t = T-1, T-2, \dots, 0$.  
**Code Fix:**
```python
# Ensure beta_T is sufficiently large to drive alpha_bar to 0
betas = torch.linspace(1e-4, 0.02, T) # alpha_bar_T drops to ~0.0001
assert alphas_bar[-1] < 1e-3, "Increase beta_T or step count T"
```

---

## References & Further Reading

For complete citations, seminal papers, academic lecture links, and official implementations, see [./references.md](./references.md).
