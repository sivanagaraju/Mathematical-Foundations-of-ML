> [!NOTE]
> **Orientation**: This package derives the noise-prediction parameterization and $L_{\text{simple}}$ loss. For foundational algebraic inversions and posterior substitutions, see [./PREREQUISITES.md](./PREREQUISITES.md).

# Lecture 15: Diffusion Models - Part 2 (Reparameterization & Sampling)

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Master End-to-End Simulation](#master-end-to-end-simulation)
3. [Topic 1: Analytical Gaussian Target: Transforming L_{t-1} into Explicit Mean Squared Error](#topic-1-analytical-gaussian-target-transforming-l_t-1-into-explicit-mean-squared-error)
4. [Topic 2: Inverting the Jump Kernel: Expressing Clean Data x_0 in Terms of Injected Noise epsilon](#topic-2-inverting-the-jump-kernel-expressing-clean-data-x_0-in-terms-of-injected-noise-epsilon)
5. [Topic 3: Reparameterizing the Posterior Mean: From State Prediction to Noise-Prediction Network epsilon_theta](#topic-3-reparameterizing-the-posterior-mean-from-state-prediction-to-noise-prediction-network-epsilon_theta)
6. [Topic 4: Loss Equivalence Derivation: Bridging Variational ELBO to Denoising Score Matching](#topic-4-loss-equivalence-derivation-bridging-variational-elbo-to-denoising-score-matching)
7. [Topic 5: The Simplified Loss L_simple: Dropping Analytical Weights to Emphasize Perceptual Quality](#topic-5-the-simplified-loss-l_simple-dropping-analytical-weights-to-emphasize-perceptual-quality)
8. [Topic 6: The Ancestral Reverse Sampling Algorithm: Iterative Denoising from Noise to Pristine Data](#topic-6-the-ancestral-reverse-sampling-algorithm-iterative-denoising-from-noise-to-pristine-data)
9. [Workplace Debugging Scenarios](#workplace-debugging-scenarios)
10. [References & Further Reading](#references--further-reading)

---

## Executive Summary

This lecture establishes the engineering breakthrough that made diffusion models practical: converting intractable variational inference into elementary noise prediction. By parameterizing the neural network to predict the injected Gaussian noise $\epsilon$ rather than the posterior mean, the variational objective reduces to a simple, unweighted Mean Squared Error ($L_{\text{simple}}$).

```
                    THE DDPM TRAINING & INFERENCE LOOP
                    
   =========================== TRAINING (O(1)) ===========================
   Clean Data x_0  ----+
                       |-----> x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps
   Random Noise eps ---+
                                    |
                       [ U-Net Denoiser epsilon_theta ]
                                    v
                           Predicted Noise eps_hat
                                    |
               Loss = || eps - eps_theta(x_t, t) ||^2  (Backprop)
               
   =========================== INFERENCE (O(T)) ===========================
   x_T ~ N(0, I) ---> [ Loop t = T ... 1: x_{t-1} = 1/sqrt(alpha_t)*(x_t - ...) + sigma*z ] ---> Generated x_0
```
*Figure 1: Complete lifecycle of DDPM: O(1) random-time noise prediction during training versus iterative O(T) ancestral denoising during inference.*

### Scenario Walkthrough
1. **Sample Data & Noise**: Draw $x_0 \sim q(x_0)$, $t \sim \mathcal{U}(1, T)$, and $\epsilon \sim \mathcal{N}(0, I)$.
2. **Corrupt in One Shot**: Compute $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
3. **Predict Noise**: Forward pass through neural network $\hat{\epsilon} = \epsilon_\theta(x_t, t)$.
4. **Compute $L_{\text{simple}}$**: Evaluate $\|\epsilon - \hat{\epsilon}\|^2$ and take gradient step on $\theta$.
5. **Generate at Test Time**: Sample $x_T \sim \mathcal{N}(0, I)$, then iterate reverse sampling down to $t=0$.

### Failure / Contrast Path
If you configure the network to predict $x_0$ directly ($\hat{x}_0 = f_\theta(x_t, t)$), early in training and at high $t$ where noise dominates, the network collapses to predicting the blurry conditional dataset average $\mathbb{E}[x_0 \mid x_t]$. By predicting $\epsilon$, the network is always tasked with predicting a standardized, zero-mean Gaussian vector, preventing posterior collapse.

### STOP / Out of Scope
This lecture focuses on **unconditional DDPM, $L_{\text{simple}}$, and ancestral sampling**. Classifier-free guidance (CFG), DDIM accelerated sampling, and continuous-time SDE formulations are reserved for Lectures 17 and 18.

### Load-Bearing Claims
- The true posterior mean $\tilde{\mu}_t(x_t, x_0)$ can be expressed strictly as a linear function of $x_t$ and injected noise $\epsilon$.
- Ho et al.'s simplified loss $L_{\text{simple}}$ drops the analytical variance prefactors, delivering significantly higher sample quality and FID scores.
- Sampling requires injecting Gaussian noise $z \sim \mathcal{N}(0, I)$ at every step $t > 1$; setting $z=0$ produces overly smooth, low-diversity samples.

### Comparative Feature & Tradeoff Matrix

| Objective Formulation | Theoretical Purity | Sample Visual Quality | Code Complexity | Training Stability |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Weighted ELBO** | Maximum Likelihood exact | Moderate (focuses on micro-noise) | High (dynamic coefficients) | Prone to loss spiking |
| **$L_{\text{simple}}$ (Ho et al.)** | Reweighted Variational Bound | State-of-the-Art (high fidelity) | Minimal (standard MSE) | Extremely Stable |
| **Direct $x_0$ Prediction** | Variational Bound | Poor (blurry at high $t$) | Moderate | Prone to mode averaging |

### Common Traps & Numerical Fixes
- **Adding Noise at $t=1$**: At the very last reverse step $t=1 \to 0$, set $z = 0$. Adding noise at $t=1$ leaves static grain on the final output image.
- **Mismatching $\sqrt{1 - \bar{\alpha}_t}$ Scaling**: When computing reverse mean, forgetting to divide by $\sqrt{1 - \bar{\alpha}_t}$ leads to drastic over-smoothing.

---

## Master End-to-End Simulation

```python
import torch
import torch.nn as nn

def master_ddpm_simulation():
    torch.manual_seed(42)
    B, D, T = 16, 4, 20
    
    # 1. Linear beta schedule
    betas = torch.linspace(0.01, 0.05, T)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    
    # 2. Synthetic data batch x0
    x0 = torch.randn(B, D)
    t = torch.randint(0, T, (B,))
    
    # 3. Sample injected noise
    eps = torch.randn_like(x0)
    a_bar = alphas_bar[t].unsqueeze(-1)
    xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps
    
    # 4. Toy Denoiser predicting epsilon
    model = nn.Sequential(nn.Linear(D, 32), nn.SiLU(), nn.Linear(32, D))
    pred_eps = model(xt)
    
    # 5. L_simple loss
    loss = nn.functional.mse_loss(pred_eps, eps)
    loss.backward()
    
    # 6. Ancestral Reverse Step at t=T-1
    step_t = T - 1
    beta_curr = betas[step_t]
    alpha_curr = alphas[step_t]
    alpha_bar_curr = alphas_bar[step_t]
    
    with torch.no_grad():
        drift = (1.0 / torch.sqrt(alpha_curr)) * (xt - (beta_curr / torch.sqrt(1.0 - alpha_bar_curr)) * pred_eps)
        sigma = torch.sqrt(beta_curr)
        z = torch.randn_like(xt)
        xt_prev = drift + sigma * z
        
    assert xt_prev.shape == (B, D)
    print(f"Master simulation PASSED: loss={loss.item():.4f}, xt_prev shape={xt_prev.shape}")

master_ddpm_simulation()
```

---

## Topic 1: Analytical Gaussian Target: Transforming $L_{t-1}$ into Explicit Mean Squared Error

### Where this sits on the master map
We take the abstract KL divergence derived in Lecture 14 and demonstrate how it collapses to a straightforward Euclidean distance between the analytical mean $\tilde{\mu}_t$ and the network mean $\mu_\theta$.

### Board / screenshot
```
                GAUSSIAN KL TO EUCLIDEAN DISTANCE
                
    D_KL( q(x_{t-1}|x_t, x_0) || p_theta(x_{t-1}|x_t) )
                             ||
              1 / (2 * sigma_t^2) * || \tilde{\mu}_t - \mu_theta ||^2
```
*Notice: Because both distributions are spherical Gaussians with identical covariance, all determinant and trace terms cancel out, leaving only the squared difference of means.*

### What he is establishing
From the tripartite decomposition:
$$L_{t-1} = D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t))$$
We set the model covariance to match the forward posterior variance: $\Sigma_\theta(x_t, t) = \sigma_t^2 I$, where $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ (or $\sigma_t^2 = \beta_t$).
Substituting into the Gaussian KL formula:
$$L_{t-1} = \frac{1}{2\sigma_t^2} \|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2 + C$$
where $C$ is independent of $\theta$.

This proves that training a diffusion model does not require complex density ratio estimators or adversarial discriminators; it is strictly a supervised regression task targeting $\tilde{\mu}_t$.

### Contrastive Analysis: Why X, Not Y?
- **Why fix covariance $\Sigma_\theta = \sigma_t^2 I$ rather than learn it?**  
  Ho et al. discovered that learning the covariance alongside the mean causes training instability with negligible visual quality improvement. Fixing it to $\sigma_t^2 I$ allows the network to dedicate 100% of its capacity to accurately learning the mean drift vector.

### Active Comprehension Checks
1. *Question*: Why does the determinant ratio $\log \frac{\det \Sigma_\theta}{\det \Sigma_q}$ disappear from $L_{t-1}$?  
   *Answer*: Because we choose the model variance to match the forward variance ($\Sigma_\theta = \Sigma_q = \sigma_t^2 I$), making their ratio identity and its logarithm zero.

### Analogy for this topic only
Two archery targets placed on the same bullseye with identical target sizes. The distance between the arrow groupings depends solely on the offset between their centers.
*In lecture words: Matching Gaussians with fixed variances is identical to MSE between their means.*

### Local picture
```
   Target: \tilde{mu}_t(x_t, x_0) <---- MSE Distance ----> Model: mu_theta(x_t, t)
```
*Notice: Gradient pulls model mean directly toward analytical posterior target.*

### Bridge
To eliminate the dependence on $x_0$ inside $\tilde{\mu}_t$, we now invert the forward jump kernel.

---

## Topic 2: Inverting the Jump Kernel: Expressing Clean Data $x_0$ in Terms of Injected Noise $\epsilon$

### Where this sits on the master map
We express the ground-truth image $x_0$ as a function of the noisy input $x_t$ and the noise $\epsilon$ that corrupted it.

### Board / screenshot
```
                THE ALGEBRAIC INVERSION OF JUMP KERNEL
                
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
                                 |
                                 v  (Algebraic rearrangement)
        x_0 = 1 / sqrt(alpha_bar_t) * [ x_t - sqrt(1 - alpha_bar_t) * eps ]
```
*Notice: Knowing the noisy state $x_t$ and the injected noise $\epsilon$ completely determines $x_0$.*

### What he is establishing
Recall the closed-form forward jump kernel:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
Subtracting $\sqrt{1 - \bar{\alpha}_t} \epsilon$ from both sides:
$$\sqrt{\bar{\alpha}_t} x_0 = x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon$$
Dividing by $\sqrt{\bar{\alpha}_t}$:
$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}$$

This expresses the clean data point $x_0$ as a deterministic function of $(x_t, \epsilon)$. During training, because we explicitly sampled $\epsilon$ to create $x_t$, we know $\epsilon$ exactly!

### Contrastive Analysis: Why X, Not Y?
- **Why express $x_0$ in terms of $\epsilon$ instead of keeping $x_0$?**  
  Because $\epsilon$ has a standard normal distribution $\mathcal{N}(0, I)$ at every time step $t$, whereas $x_0$ has an arbitrary, complex data distribution. Working with $\epsilon$ standardizes the target representation.

### Active Comprehension Checks
1. *Question*: What happens to the denominator $\sqrt{\bar{\alpha}_t}$ as $t \to T$?  
   *Answer*: As $t \to T$, $\bar{\alpha}_t \to 0$, so $\frac{1}{\sqrt{\bar{\alpha}_t}} \to \infty$. Attempting to reconstruct $x_0$ directly from pure noise blows up numerically, which is why we predict noise $\epsilon$ instead.

### Analogy for this topic only
Deciphering an audio recording masked by static. If you know the exact frequency pattern of the static generator, you subtract the static rather than trying to guess the speaker's vocal cords from scratch.
*In lecture words: Inverting the forward equation connects the clean source to the injected noise.*

### Local picture
```
   [ Noisy Input x_t ] - [ Injected Noise eps ] ===(Divide by sqrt(alpha_bar))===> [ Clean Data x_0 ]
```
*Notice: Clean image is revealed by stripping scaled noise.*

### Bridge
Now we substitute this expression for $x_0$ into the posterior mean formula to derive the noise-prediction parameterization.

---

## Topic 3: Reparameterizing the Posterior Mean: From State Prediction to Noise-Prediction Network $\epsilon_\theta$

### Where this sits on the master map
We complete the substitution of $x_0$ into $\tilde{\mu}_t(x_t, x_0)$ to reveal the canonical DDPM parameterization.

### Board / screenshot
```
                THE NOISE-PREDICTION REPARAMETERIZATION
                
    \tilde{\mu}_t = 1 / sqrt(alpha_t) * [ x_t - beta_t / sqrt(1 - alpha_bar_t) * eps ]
                                   |
                                   v  (Parametrize model identically)
    mu_theta(x_t, t) = 1 / sqrt(alpha_t) * [ x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t) ]
```
*Notice: The neural network only needs to output $\epsilon_\theta(x_t, t)$. The rest of the reverse step is fixed arithmetic.*

### What he is establishing
Substituting $x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}$ into the posterior mean:
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}} \right) + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$
Using $\sqrt{\bar{\alpha}_t} = \sqrt{\alpha_t}\sqrt{\bar{\alpha}_{t-1}}$ and combining like terms:
$$\tilde{\mu}_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$$

This reveals that the optimal reverse transition mean $\tilde{\mu}_t$ is simply $x_t$ minus a scaled version of the noise $\epsilon$!
Therefore, we define our neural network model $\mu_\theta$ to match this structure:
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$
where $\epsilon_\theta(x_t, t)$ is our neural network.

### Contrastive Analysis: Why X, Not Y?
- **Why predict $\epsilon_\theta$ instead of $\mu_\theta$ directly?**  
  $\mu_\theta$ must scale dynamically with $x_t$, which varies wildly in magnitude across time steps. By parameterizing $\mu_\theta$ through $\epsilon_\theta$, the network always predicts a zero-mean, unit-variance Gaussian target $\epsilon \sim \mathcal{N}(0, I)$ for all $t$.

### Active Comprehension Checks
1. *Question*: What is the output shape of $\epsilon_\theta(x_t, t)$?  
   *Answer*: Exactly identical to the input shape $x_t$ (e.g. $(B, C, H, W)$).

### Analogy for this topic only
An art restorer removing grime from a Renaissance painting. It is much easier to describe the chemical soot layer ($\epsilon$) than to paint the Mona Lisa from memory.
*In lecture words: The denoiser predicts the corruption, not the underlying masterpiece.*

### Local picture
```
    x_t, t ----> [ U-Net Backbone ] ----> eps_theta(x_t, t)
```
*Notice: Input and output tensors have identical dimensions.*

### Bridge
Now we substitute this parameterization into our loss function and examine its connection to score matching.

---

## Topic 4: Loss Equivalence Derivation: Bridging Variational ELBO to Denoising Score Matching

### Where this sits on the master map
We show that minimizing the Gaussian KL divergence between the parameterized model and the posterior target is mathematically equivalent to Denoising Score Matching.

### Board / screenshot
```
                THE LOSS REDUCTION EQUIVALENCE
                
     L_{t-1} = 1 / (2 * sigma_t^2) * || \tilde{\mu}_t - \mu_theta ||^2
                                   ||
     gamma_t * || eps - eps_theta(x_t, t) ||^2
     
     where gamma_t = beta_t^2 / ( 2 * sigma_t^2 * alpha_t * (1 - alpha_bar_t) )
```
*Notice: The subtraction cancels the $x_t$ term, leaving only the squared error between injected and predicted noise.*

### What he is establishing
Subtracting $\mu_\theta(x_t, t)$ from $\tilde{\mu}_t(x_t, \epsilon)$:
$$\tilde{\mu}_t - \mu_\theta = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right) - \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$
$$= -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}} \left( \epsilon - \epsilon_\theta(x_t, t) \right)$$

Squaring the norm:
$$\|\tilde{\mu}_t - \mu_\theta\|^2 = \frac{\beta_t^2}{\alpha_t(1 - \bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$
Multiplying by $\frac{1}{2\sigma_t^2}$:
$$L_{t-1} = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

This proves that the full variational lower bound (ELBO) is a weighted sum of MSE denoising score matching objectives across time steps $t \in \{2, \dots, T\}$.

### Contrastive Analysis: Why X, Not Y?
- **Why is this called Score Matching?**  
  Tweedie's formula dictates that for Gaussian perturbation, the score function is $\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$. Therefore, predicting noise $\epsilon_\theta$ is mathematically identical to estimating the score $\nabla_{x_t} \log q(x_t)$.

### Active Comprehension Checks
1. *Question*: What is the relationship between the predicted noise $\epsilon_\theta$ and the score function?  
   *Answer*: $\nabla_{x_t} \log p_\theta(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$.

### Analogy for this topic only
Two routes up a mountain: the rigorous thermodynamic climb (ELBO) and the local topographical slope meter (Score Matching). Both arrive at the exact same summit formula.
*In lecture words: Variational inference and score matching are dual formulations of diffusion.*

### Local picture
```
   [ Variational ELBO ] <=====(Dual Equivalence)=====> [ Denoising Score Matching ]
```
*Notice: Theoretical convergence of two independent AI paradigms.*

### Bridge
Now we examine Ho et al.'s pivotal empirical insight: what happens if we throw away the theoretical weighting factor $\gamma_t$?

---

## Topic 5: The Simplified Loss $L_{\text{simple}}$: Dropping Analytical Weights to Emphasize Perceptual Quality

### Where this sits on the master map
We analyze the transition from the theoretically rigorous weighted ELBO to the unweighted objective $L_{\text{simple}}$ that powers modern generative models.

### Board / screenshot
```
                THE SIMPLIFIED LOSS FUNCTION (Ho et al. 2020)
                
    L_simple(theta) = E_{t, x_0, eps} [ || eps - eps_theta( x_t(x_0, eps), t ) ||^2 ]
    
    (Set weight gamma_t = 1.0 for all t in {1, ..., T})
```
*Notice: No coefficients, no variances, no schedule constants—pure unweighted Mean Squared Error.*

### What he is establishing
In the exact ELBO:
$$L_{\text{ELBO}}(\theta) = \sum_{t=1}^T \gamma_t \|\epsilon - \epsilon_\theta(x_t, t)\|^2, \quad \gamma_t = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1 - \bar{\alpha}_t)}$$
When $\sigma_t^2 = \tilde{\beta}_t$, the weight $\gamma_t$ evaluates to $\frac{1}{2(1 - \bar{\alpha}_{t-1})}$.
Notice that as $t \to 1$ (tiny noise), $\bar{\alpha}_{t-1} \to 1$, so $\gamma_1 \to \infty$!
The exact ELBO forces the neural network to spend 99% of its capacity optimizing imperceptible sub-pixel noise at $t=1$, while heavily downweighting large $t$ where global semantic structure (faces, animals, geometry) is decided.

Ho et al. proposed discarding $\gamma_t$ entirely:
$$L_{\text{simple}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(1, T), x_0, \epsilon \sim \mathcal{N}(0, I)} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$
Setting $\gamma_t = 1$ acts as a perceptual reweighting that emphasizes larger $t$, dramatically improving sample fidelity, sharp edges, and FID scores.

### Contrastive Analysis: Why X, Not Y?
- **Does $L_{\text{simple}}$ still optimize the variational bound?**  
  Technically, $L_{\text{simple}}$ optimizes a reweighted variational lower bound. While exact negative log-likelihood (NLL) on test data is slightly worse, the visual perceptual quality of generated samples improves by an order of magnitude.

### Active Comprehension Checks
1. *Question*: Why does $L_{\text{simple}}$ sample $t$ uniformly from $\{1, \dots, T\}$?  
   *Answer*: Uniform sampling of $t$ ensures equal gradient expectation across all noise scales without requiring full sum computation.

### Analogy for this topic only
An art school grading rubric. The strict academic rubric ($L_{\text{ELBO}}$) deducts 50 points if canvas threads are visible under a microscope. The simplified rubric ($L_{\text{simple}}$) gives equal weight to whether the painting actually looks like a human face.
*In lecture words: Dropping the weights rebalances model capacity toward perceptual features.*

### Local picture
```
   Exact ELBO:   [ Heavy Weight on Tiny t ] ---> [ Microscopic Texture Focus ]
   L_simple:     [ Equal Weight Across all t ] ---> [ Balanced Semantic & Texture Focus ]
```
*Notice: Uniform weighting yields superior aesthetic generation.*

### Bridge
With the training objective fully simplified, we conclude by deriving the step-by-step inference algorithm used to generate images.

---

## Topic 6: The Ancestral Reverse Sampling Algorithm: Iterative Denoising from Noise to Pristine Data

### Where this sits on the master map
We formalize Algorithm 2 from Ho et al. (2020): generating novel images from pure Gaussian noise at test time.

### Board / screenshot
```
                ANCESTRAL SAMPLING ALGORITHM (Ho et al. 2020)
                
    1: x_T ~ N(0, I)
    2: for t = T, T-1, ..., 1 do:
    3:     z ~ N(0, I) if t > 1, else z = 0
    4:     x_{t-1} = 1 / sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t) ) + sigma_t * z
    5: end for
    6: return x_0
```
*Notice: Noise injection $z$ occurs for all steps except the final step $t=1$, where the mean is returned deterministically.*

### What he is establishing
The ancestral sampling procedure:
1. **Sample pure prior**: Draw $x_T \sim \mathcal{N}(0, I)$.
2. **Sequential Iteration**: For each step $t = T, T-1, \dots, 1$:
   - Compute predicted noise: $\hat{\epsilon} = \epsilon_\theta(x_t, t)$.
   - Compute reverse mean:
     $$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon} \right)$$
   - If $t > 1$, sample $z \sim \mathcal{N}(0, I)$; if $t = 1$, set $z = 0$.
   - Update state:
     $$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$$
3. **Return pristine sample**: $x_0$ is the generated image.

### Contrastive Analysis: Why X, Not Y?
- **Why set $z=0$ at $t=1$?**  
  At $t=1$, the model has removed all noise except the final microscopic layer. Adding random noise $z$ at $t=1$ would permanently contaminate the output image with visible pixel noise.

### Active Comprehension Checks
1. *Question*: What is the computational cost of generating one image with $T=1000$?  
   *Answer*: Exactly 1,000 sequential neural network forward evaluations.

### Analogy for this topic only
Developing a Polaroid photograph. You pull the film from the camera (pure black, $x_T$). Over 60 seconds (steps $t$), hazy shapes emerge, slowly sharpening into crisp focus until the chemical reaction completes ($x_0$).
*In lecture words: Generation is an ancestral random walk reversing entropy step-by-step.*

### Local picture
```
   x_T (Noise) ---> x_{T-1} ---> ... ---> x_1 ---> x_0 (Clean Image)
```
*Notice: Gradual emergence of structure.*

### Bridge
We have now mastered the mathematical journey from abstract variational inference to practical noise prediction and sampling. Let's examine critical workplace debugging scenarios.

---

## Workplace Debugging Scenarios

### Scenario 1: The Stochastic Grain Defect at Terminal Step $t=1$
**Incident:** An AI team deploys a diffusion model for photo editing. While generated compositions are aesthetically stunning, zoomed-in inspection reveals high-frequency colored grain across every image.  
**Mathematical Root Cause:** The sampling loop injected random Gaussian noise $\sigma_t z$ at every iteration including $t=1$. The loop condition was written as `for t in range(T, 0, -1): xt = drift + sigma * torch.randn_like(xt)` without zeroing $z$ when $t=1$.  
**Debugging Steps:**
1. Check pixel-level standard deviation in uniform color regions. It measures $\approx \sigma_1 = \sqrt{\beta_1} \approx 0.01$, confirming injected residual noise.
2. Add a conditional check to enforce $z=0$ at $t=1$.  
**Code Fix:**
```python
# CORRECT SAMPLING NOISE INJECTION:
z = torch.randn_like(xt) if t > 1 else torch.zeros_like(xt)
xt_prev = drift + sigma_t * z
```

---

### Scenario 2: The Blurry Average Collapse from Direct $x_0$ Prediction
**Incident:** An engineer switches the U-Net target from predicting noise $\epsilon$ to predicting clean image $x_0$ directly via MSE loss $\|x_0 - f_\theta(x_t, t)\|^2$. At inference time, generated faces have washed-out, ghostly skin tones and no sharp facial features.  
**Mathematical Root Cause:** At large $t$, $x_t$ contains virtually zero mutual information with $x_0$. When minimizing MSE against an uninformative input, the optimal Bayes estimator is the conditional mean $\mathbb{E}[x_0 \mid x_t] \approx \mathbb{E}[x_0]$. The network is mathematically forced to predict the blurry average of all faces in the dataset.  
**Debugging Steps:**
1. Evaluate predicted $\hat{x}_0$ at $t=900$. Observe that the output is an identical blurry beige oval across all random noise seeds.
2. Restore the noise-prediction parameterization $\epsilon_\theta(x_t, t)$, where the target is always full-rank Gaussian noise $\mathcal{N}(0, I)$.  
**Code Fix:**
```python
# REVERT TO NOISE PREDICTION:
pred_eps = model(xt, t)
loss = torch.nn.functional.mse_loss(pred_eps, eps)
```

---

## References & Further Reading

For complete citations, seminal papers, academic lecture links, and official implementations, see [./references.md](./references.md).
