> [!NOTE]
> **Orientation**: This package derives the full Evidence Lower Bound (ELBO) decomposition. For prerequisites on Jensen's inequality and Gaussian KL divergence, see [./PREREQUISITES.md](./PREREQUISITES.md).

# Lecture 14: Diffusion Models - Part 1 (ELBO & Posterior Derivation)

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Master End-to-End Simulation](#master-end-to-end-simulation)
3. [Topic 1: Hierarchical Variational Formulation: Defining Trajectories q(x_{1:T}|x_0) and p_theta(x_{0:T})](#topic-1-hierarchical-variational-formulation-defining-trajectories-qx_1tx_0-and-p_thetax_0t)
4. [Topic 2: The Naive Variational Lower Bound (ELBO): High Monte Carlo Variance Across T Steps](#topic-2-the-naive-variational-lower-bound-elbo-high-monte-carlo-variance-across-t-steps)
5. [Topic 3: Bayes Conditioning Trick: Reformulating Forward Transitions q(x_t|x_{t-1}, x_0)](#topic-3-bayes-conditioning-trick-reformulating-forward-transitions-qx_tx_t-1-x_0)
6. [Topic 4: The Telescoping Product Cancellation: Collapsing Joint Path Quotients into Per-Step Ratios](#topic-4-the-telescoping-product-cancellation-collapsing-joint-path-quotients-into-per-step-ratios)
7. [Topic 5: The Tripartite ELBO Decomposition: Prior Matching L_T, Denoising L_{t-1}, and Reconstruction L_0](#topic-5-the-tripartite-elbo-decomposition-prior-matching-l_t-denoising-l_t-1-and-reconstruction-l_0)
8. [Topic 6: Tractable Posterior q(x_{t-1}|x_t, x_0): Gaussian Expansion and Variance Parameters](#topic-6-tractable-posterior-qx_t-1x_t-x_0-gaussian-expansion-and-variance-parameters)
9. [Workplace Debugging Scenarios](#workplace-debugging-scenarios)
10. [References & Further Reading](#references--further-reading)

---

## Executive Summary

This lecture establishes the complete mathematical architecture of the Evidence Lower Bound (ELBO) for diffusion probabilistic models. By applying Bayes' conditioning on $x_0$, the high-variance trajectory-level expectation collapses via telescoping cancellation into $T$ independent, analytically tractable Gaussian KL divergences.

```
                    THE ELBO DECOMPOSITION ROADMAP
                    
   +-----------------------------------------------------------------+
   |  Naive Trajectory ELBO: E_q [ log ( p_theta(x_{0:T}) / q(x_{1:T}|x_0) ) ]  |
   +-----------------------------------------------------------------+
                                   |
                   [ Bayes Conditioning on x_0 ]
                                   v
   +-----------------------------------------------------------------+
   |     Telescoping Expansion: Sum_{t=2}^T [ log q(x_t|x_0) - log q(x_{t-1}|x_0) ] |
   +-----------------------------------------------------------------+
                                   |
                   [ Regrouping into 3 Canonical Terms ]
                                   v
   +-------------------+   +-----------------------+   +-------------------+
   |   Prior Match L_T |   |  Denoising Match L_t  |   |  Reconstruct L_0  |
   | D_KL(q(x_T|x_0)|| |   | D_KL(q(x_{t-1}|x_t,x0)|   |  -log p_theta(    |
   |       p(x_T))     |   |   || p_theta(x_{t-1}))|   |      x_0 | x_1)   |
   +-------------------+   +-----------------------+   +-------------------+
```
*Figure 1: Mathematical roadmap of the ELBO reduction from high-variance trajectory sampling to three closed-form Gaussian terms.*

### Scenario Walkthrough
1. **Express Log-Likelihood**: Write $\log p_\theta(x_0)$ as the marginal of the joint reverse chain.
2. **Apply Jensen's Inequality**: Bound the marginal with expectation over forward trajectory $q(x_{1:T} \mid x_0)$.
3. **Condition on $x_0$**: Rewrite $q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0)$ using Bayes' rule.
4. **Telescoping Cancellation**: Intermediate terms $\frac{q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$ cancel throughout the chain.
5. **Formulate Closed-Form Loss**: Each intermediate term $t$ becomes an explicit Gaussian KL divergence $D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t))$.

### Failure / Contrast Path
If you attempt to train a diffusion model using the raw trajectory ratio $\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}$, Monte Carlo estimates over 1,000 Gaussian steps suffer from catastrophic variance. Gradients will explode or fluctuate randomly, causing total training failure. The telescoping derivation provides an analytic zero-variance formulation for every single step.

### STOP / Out of Scope
This lecture stops after deriving the **tripartite ELBO and the posterior distribution parameters $\tilde{\mu}_t, \tilde{\beta}_t$**. The reparameterization to noise prediction $\epsilon_\theta(x_t, t)$ and the simplified loss $L_{\text{simple}}$ are derived in Lecture 15.

### Load-Bearing Claims
- Conditioning on $x_0$ renders the reverse transition $q(x_{t-1} \mid x_t, x_0)$ strictly tractable and analytically Gaussian.
- The prior matching term $L_T = D_{\text{KL}}(q(x_T \mid x_0) \parallel p(x_T))$ contains **zero learnable parameters** and is treated as a constant during training.
- The entire learnable component of diffusion training resides in the denoising matching terms $L_{t-1}$ for $t \in \{2, \dots, T\}$.

### Comparative Feature & Tradeoff Matrix

| Loss Formulation | Variance | Computational Cost | Analytical Tractability | Training Feasibility |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Path ELBO** | Catastrophic ($\mathcal{O}(T)$ noise) | High (requires full rollout) | Low (no closed-form per step) | Impossible in practice |
| **Telescoped ELBO** | Zero per-step variance | Moderate (evaluates step $t$) | Exact Closed-Form Gaussian KL | High (production standard) |
| **Score Matching** | Low | Low | Equivalent to Gaussian KL | Highly stable |

### Common Traps & Numerical Fixes
- **Sign Error in ELBO**: Maximizing the ELBO means minimizing $-\text{ELBO} = L_T + \sum L_{t-1} + L_0$. Confusing loss and ELBO leads to gradient ascent instead of descent.
- **Ignoring the Posterior Variance**: Setting $\tilde{\beta}_t = \beta_t$ is an approximation; the exact posterior variance is $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$.

---

## Master End-to-End Simulation

```python
import torch

def master_elbo_simulation():
    torch.manual_seed(42)
    B, D = 8, 4
    x0 = torch.randn(B, D)
    
    # 1. Hyperparameters for 10 steps
    T = 10
    betas = torch.linspace(0.01, 0.1, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # 2. Pick a step t = 5
    t = 5
    a_t = alphas[t]
    b_t = betas[t]
    a_bar_t = alpha_bars[t]
    a_bar_prev = alpha_bars[t - 1]
    
    # 3. Sample noisy state xt from jump kernel
    eps = torch.randn_like(x0)
    xt = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * eps
    
    # 4. Compute analytical posterior mean \tilde{\mu}_t(xt, x0)
    coef_x0 = (torch.sqrt(a_bar_prev) * b_t) / (1.0 - a_bar_t)
    coef_xt = (torch.sqrt(a_t) * (1.0 - a_bar_prev)) / (1.0 - a_bar_t)
    mu_tilde = coef_x0 * x0 + coef_xt * xt
    
    # 5. Compute analytical posterior variance \tilde{\beta}_t
    beta_tilde = ((1.0 - a_bar_prev) / (1.0 - a_bar_t)) * b_t
    
    assert mu_tilde.shape == (B, D)
    assert beta_tilde.item() > 0.0
    print(f"Master simulation PASSED: t={t}, beta_tilde={beta_tilde.item():.5f}, mu_tilde mean={mu_tilde.mean():.4f}")

master_elbo_simulation()
```

---

## Topic 1: Hierarchical Variational Formulation: Defining Trajectories $q(x_{1:T} \mid x_0)$ and $p_\theta(x_{0:T})$

### Where this sits on the master map
We begin by establishing diffusion models as deep hierarchical latent variable models ([05-Latent_Variable_Models.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/05-Latent_Variable_Models.md)), where the latents $x_1, \dots, x_T$ share the exact same dimensionality as the input data $x_0$.

### Board / screenshot
```
                THE HIERARCHICAL LATENT VARIABLE SYSTEM
                
    Forward Inference Path (Fixed):
    q(x_{1:T}|x_0) = q(x_1|x_0) * q(x_2|x_1) * ... * q(x_T|x_{T-1})
    
    Reverse Generative Path (Model):
    p_theta(x_{0:T}) = p(x_T) * p_theta(x_{T-1}|x_T) * ... * p_theta(x_0|x_1)
```
*Notice: In VAEs, there is 1 latent layer $z$. In Diffusion, there are $T$ latent states $x_1, \dots, x_T$, all in data space $\mathbb{R}^D$.*

### What he is establishing
The instructor formalizes both Markov chains:
1. **Forward joint distribution**:
   $$q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$$
2. **Reverse joint distribution**:
   $$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t), \quad p(x_T) = \mathcal{N}(x_T; 0, I)$$

The goal of generative modeling is to maximize the marginal data log-likelihood:
$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T}) dx_{1:T}$$
Because integrating over $\mathbb{R}^{D \times T}$ is completely intractable, we construct a variational lower bound.

### Contrastive Analysis: Why X, Not Y?
- **Why keep latents $x_t$ in the exact same dimension $\mathbb{R}^D$ as data $x_0$?**  
  In standard VAEs, $z$ lives in a compressed lower-dimensional space $\mathbb{R}^d$ ($d \ll D$), forcing lossy compression. In diffusion models, each latent maintains full resolution $\mathbb{R}^D$. The model does not compress spatial information; it only compresses statistical entropy.

### Active Comprehension Checks
1. *Question*: How many latent variables exist in a DDPM with $T=1000$ for a $256 \times 256 \times 3$ image?  
   *Answer*: Exactly 1,000 latent variables, each of shape $(256, 256, 3)$.

### Analogy for this topic only
A multi-stage relay race where every runner runs in the exact same stadium lane, passing a slightly noisier baton to the next runner.
*In lecture words: Diffusion is a hierarchical Markovian latent variable model without dimensionality reduction.*

### Local picture
```
   x_0 (Image) ---> x_1 ---> x_2 ---> ... ---> x_T (Noise) [All same shape]
```
*Notice: Dimensions never shrink or expand across the entire chain.*

### Bridge
With the joint distributions defined, let us write out the standard variational lower bound and discover why its naive evaluation is impossible.

---

## Topic 2: The Naive Variational Lower Bound (ELBO): High Monte Carlo Variance Across $T$ Steps

### Where this sits on the master map
We apply Jensen's inequality to obtain the unrolled ELBO and analyze why naive trajectory sampling fails.

### Board / screenshot
```
                THE NAIVE PATH-WISE EXPECTATION
                
   log p_theta(x_0) >= E_{q(x_{1:T}|x_0)} [ log ( p_theta(x_{0:T}) / q(x_{1:T}|x_0) ) ]
                           ^
                           | High Variance Disaster:
                           | Sample 1000 continuous vectors x_1...x_T
                           | to estimate a single scalar gradient!
```
*Notice: Estimating this expectation via Monte Carlo sampling produces gradients with variance proportional to $T$.*

### What he is establishing
Applying Jensen's inequality:
$$\log p_\theta(x_0) \ge \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right]$$
Expanding the joint products:
$$\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} = \log \left( \frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)}{\prod_{t=1}^T q(x_t \mid x_{t-1})} \right)$$
$$= \log p(x_T) + \sum_{t=1}^T \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}$$

If we try to optimize this directly, we must sample a complete path $x_1, x_2, \dots, x_T \sim q(x_{1:T} \mid x_0)$. Each sample path wanders randomly through a 1,000-dimensional Gaussian space, making the variance of the Monte Carlo gradient catastrophic.

### Contrastive Analysis: Why X, Not Y?
- **Why can't we just use reparameterization gradients on the full path?**  
  Backpropagating through 1,000 sequential stochastic nodes leads to compounding variance and GPU memory explosion. We need a mathematical restructuring that breaks the chain into decoupled individual steps.

### Active Comprehension Checks
1. *Question*: What happens to the variance of a Monte Carlo estimator when summing 1,000 independent noisy log-ratios?  
   *Answer*: The variance of the sum equals the sum of the variances, growing linearly or super-linearly with $T$.

### Analogy for this topic only
Trying to measure the accuracy of a rifle by shooting in a pitch-black hurricane across 1,000 yards. Every yard of crosswind adds random drift, making the bullet's landing position virtually uncorrelated with your aim.
*In lecture words: Naive trajectory sampling yields gradient variance that drowns the training signal.*

### Local picture
```
   Path 1: [x_1]---->[x_2]---->[x_3] ... (Huge divergence between paths)
   Path 2: [x_1']--->[x_2']--->[x_3'] ...
```
*Notice: Random paths diverge dramatically.*

### Bridge
To eliminate this variance, the instructor introduces the clever conditioning trick: conditioning every forward transition on the clean data point $x_0$.

---

## Topic 3: Bayes Conditioning Trick: Reformulating Forward Transitions $q(x_t \mid x_{t-1}, x_0)$

### Where this sits on the master map
This is the pivotal mathematical breakthrough in the derivation. We inject the ground-truth observation $x_0$ into the forward Markov transition using Bayes' theorem.

### Board / screenshot
```
                THE BAYES CONDITIONING IDENTITY
                
                     q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})
                                    ||
                q(x_{t-1} | x_t, x_0) * q(x_t | x_0) / q(x_{t-1} | x_0)
```
*Notice: Because the forward chain is Markovian, adding condition $x_0$ on the left changes nothing, but allows us to rewrite it using Bayes rule on the right.*

### What he is establishing
By the Markov property:
$$q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$$
Now apply Bayes' rule to invert the time direction of the step:
$$q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0) q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$$
Substituting this into the forward transition $q(x_t \mid x_{t-1})$ inside the ELBO:
$$\log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})} = \log \left( \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_{t-1} \mid x_t, x_0)} \cdot \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} \right)$$
$$= \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_{t-1} \mid x_t, x_0)} + \log \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

This splits each transition into two pieces: a ratio comparing the model reverse step to the true posterior $q(x_{t-1} \mid x_t, x_0)$, and a ratio of marginal jump distributions.

### Contrastive Analysis: Why X, Not Y?
- **Why introduce $x_0$ when the forward process doesn't need it?**  
  Conditioning on $x_0$ makes the reverse transition $q(x_{t-1} \mid x_t, x_0)$ completely tractable! Without $x_0$, $q(x_{t-1} \mid x_t)$ is intractable. With $x_0$, both endpoints are known, making the intermediate state a simple Gaussian interpolation.

### Active Comprehension Checks
1. *Question*: Why is $q(x_{t-1} \mid x_t)$ intractable, whereas $q(x_{t-1} \mid x_t, x_0)$ is tractable?  
   *Answer*: $q(x_{t-1} \mid x_t) = \int q(x_{t-1}, x_0 \mid x_t) dx_0$ requires integrating over the unknown data distribution $q(x_0)$. Conditioning on $x_0$ removes the integral.

### Analogy for this topic only
If a traveler arrives in Paris ($x_t$), guessing which town they were in an hour ago ($x_{t-1}$) is impossible. But if you know their passport originated in Berlin ($x_0$), their trajectory is constrained to a narrow, predictable highway corridor.
*In lecture words: Conditioning on the origin turns an intractable distribution into a closed-form Gaussian.*

### Local picture
```
   Origin x_0 (Known) ----------> Intermediate x_{t-1} ----------> Destination x_t
                                  ^
                                  | Constrained Gaussian corridor
```
*Notice: Knowing $x_0$ pins down the intermediate distribution.*

### Bridge
Now watch the miracle of the second term: when summed over all $t$, the marginal ratios form a telescoping sequence where everything cancels out!

---

## Topic 4: The Telescoping Product Cancellation: Collapsing Joint Path Quotients into Per-Step Ratios

### Where this sits on the master map
We take the sum over $t=2, \dots, T$ of the log-ratio terms $\log \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$ and execute the telescoping cancellation.

### Board / screenshot
```
                THE TELESCOPING CANCELLATION DIAGRAM
                
    t=2:  log q(x_1|x_0) - log q(x_2|x_0)
    t=3:  log q(x_2|x_0) - log q(x_3|x_0)    <--- log q(x_2|x_0) cancels!
    t=4:  log q(x_3|x_0) - log q(x_4|x_0)    <--- log q(x_3|x_0) cancels!
    ...
    t=T:  log q(x_{T-1}|x_0) - log q(x_T|x_0) <--- log q(x_{T-1}|x_0) cancels!
    -------------------------------------------------------------------
    Sum:  log q(x_1|x_0) - log q(x_T|x_0)    (Only endpoints survive!)
```
*Notice: Every intermediate distribution from $t=2$ to $t=T-1$ vanishes completely.*

### What he is establishing
Consider the sum:
$$\sum_{t=2}^T \log \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} = \sum_{t=2}^T \left[ \log q(x_{t-1} \mid x_0) - \log q(x_t \mid x_0) \right]$$
Writing out the terms:
$$= [\log q(x_1 \mid x_0) - \log q(x_2 \mid x_0)] + [\log q(x_2 \mid x_0) - \log q(x_3 \mid x_0)] + \dots + [\log q(x_{T-1} \mid x_0) - \log q(x_T \mid x_0)]$$
Notice that $\log q(x_2 \mid x_0)$ cancels with $-\log q(x_2 \mid x_0)$, $\log q(x_3 \mid x_0)$ cancels with $-\log q(x_3 \mid x_0)$, and so forth.
The entire sum collapses to:
$$= \log q(x_1 \mid x_0) - \log q(x_T \mid x_0)$$

Combining this with the terminal prior $\log p(x_T)$:
$$\log p(x_T) - \log q(x_T \mid x_0) = -\log \frac{q(x_T \mid x_0)}{p(x_T)}$$
Taking the expectation yields the negative KL divergence: $-D_{\text{KL}}(q(x_T \mid x_0) \parallel p(x_T))$.

### Contrastive Analysis: Why X, Not Y?
- **Why did we start the sum at $t=2$ rather than $t=1$?**  
  At $t=1$, the denominator $q(x_{t-1} \mid x_0)$ would be $q(x_0 \mid x_0)$, which is a Dirac delta function ($\delta(0)$) with infinite log-density. Separating $t=1$ avoids singular distributions and gives us the reconstruction term $L_0$.

### Active Comprehension Checks
1. *Question*: What survived the telescoping cancellation?  
   *Answer*: Only $\log q(x_1 \mid x_0)$ and $-\log q(x_T \mid x_0)$.

### Analogy for this topic only
An elevator moving from floor 1 to floor 100. Rather than recording the altitude change between every pair of floors, you subtract the altitude of floor 1 from the altitude of floor 100.
*In lecture words: Telescoping simplifies the path integral into a boundary evaluation.*

### Local picture
```
   [ Floor 1 ] --------------------(Skip all internal floors)--------------------> [ Floor T ]
```
*Notice: Internal states do not affect the net displacement.*

### Bridge
Now we assemble all remaining pieces into the canonical tripartite ELBO decomposition.

---

## Topic 5: The Tripartite ELBO Decomposition: Prior Matching $L_T$, Denoising $L_{t-1}$, and Reconstruction $L_0$

### Where this sits on the master map
We organize the resulting lower bound into three distinct, easily interpretable loss components: $L_T$, $L_{t-1}$, and $L_0$.

### Board / screenshot
```
                THE TRIPARTITE ELBO LOSS DECOMPOSITION
                
    ELBO = - [ L_T + Sum_{t=2}^T L_{t-1} + L_0 ]
               |              |             |
               |              |             +---> L_0: Reconstruction Loss
               |              |                   -E_{q(x_1|x_0)}[log p_theta(x_0|x_1)]
               |              |
               |              +-----------------> L_{t-1}: Denoising Matching Terms
               |                                  D_KL( q(x_{t-1}|x_t, x_0) || p_theta(x_{t-1}|x_t) )
               |
               +--------------------------------> L_T: Prior Matching Loss
                                                  D_KL( q(x_T|x_0) || p(x_T) ) [Constant, no theta]
```
*Notice: The total negative ELBO separates into three decoupled, structurally clean terms.*

### What he is establishing
The lower bound simplifies to:
$$\text{ELBO} = \mathbb{E}_q \left[ \log p_\theta(x_0 \mid x_1) \right] - D_{\text{KL}}(q(x_T \mid x_0) \parallel p(x_T)) - \sum_{t=2}^T \mathbb{E}_{q(x_t \mid x_0)} \left[ D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t)) \right]$$

Defining the loss as $-\text{ELBO}$:
$$\mathcal{L} = L_0 + L_T + \sum_{t=2}^T L_{t-1}$$
where:
1. **$L_T = D_{\text{KL}}(q(x_T \mid x_0) \parallel p(x_T))$**: Prior matching term. Since $q(x_T \mid x_0) \approx \mathcal{N}(0, I)$ and $p(x_T) = \mathcal{N}(0, I)$, this term is close to zero and has no trainable parameters $\theta$.
2. **$L_{t-1} = D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t))$**: Denoising transition matching term. Trains the neural network $p_\theta$ to match the true posterior.
3. **$L_0 = -\log p_\theta(x_0 \mid x_1)$**: Reconstruction term. Evaluates data likelihood given the final slightly noisy state $x_1$.

### Contrastive Analysis: Why X, Not Y?
- **Why can we ignore $L_T$ during gradient descent?**  
  Because $L_T$ depends only on the fixed forward schedule $\beta_t$ and data $x_0$; it contains zero model parameters $\theta$. Therefore, $\nabla_\theta L_T = 0$.

### Active Comprehension Checks
1. *Question*: Which of the three terms ($L_0, L_{t-1}, L_T$) contains the vast majority of the training optimization workload?  
   *Answer*: The denoising matching sum $\sum_{t=2}^T L_{t-1}$, which trains the network across all intermediate noise scales.

### Analogy for this topic only
A 3-part exam. Part 1 ($L_T$) is graded on whether you showed up with a blank piece of paper (always 100%). Part 2 ($L_{t-1}$) tests your step-by-step reasoning on 998 math questions. Part 3 ($L_0$) tests whether your final answer is legible.
*In lecture words: The bulk of the objective is teaching the model to match 999 intermediate Gaussians.*

### Local picture
```
   [ L_T: Fixed constant ] + [ Sum L_{t-1}: Neural Training ] + [ L_0: Final Polish ]
```
*Notice: Optimization focuses entirely on $L_{t-1}$ and $L_0$.*

### Bridge
To compute $L_{t-1}$ analytically, we must derive the exact mathematical formula for $q(x_{t-1} \mid x_t, x_0)$.

---

## Topic 6: Tractable Posterior $q(x_{t-1} \mid x_t, x_0)$: Gaussian Expansion and Variance Parameters

### Where this sits on the master map
We complete the lecture by analytically computing the Gaussian parameters of the true conditional posterior $q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$.

### Board / screenshot
```
                THE POSTERIOR GAUSSIAN EXPANSION
                
    q(x_{t-1} | x_t, x_0) = q(x_t | x_{t-1}) * q(x_{t-1} | x_0) / q(x_t | x_0)
                          proportional to
    exp( - 1/2 [ (x_t - sqrt(alpha_t)*x_{t-1})^2 / beta_t
               + (x_{t-1} - sqrt(alpha_bar_{t-1})*x_0)^2 / (1 - alpha_bar_{t-1})
               - (x_t - sqrt(alpha_bar_t)*x_0)^2 / (1 - alpha_bar_t) ] )
```
*Notice: Expanding the exponents and completing the square for $x_{t-1}$ yields a standard Gaussian density.*

### What he is establishing
By Bayes' rule:
$$q(x_{t-1} \mid x_t, x_0) \propto q(x_t \mid x_{t-1}) q(x_{t-1} \mid x_0)$$
Both terms on the right are Gaussians:
$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t I\right) \propto \exp\left(-\frac{1}{2\beta_t} \|x_t - \sqrt{\alpha_t} x_{t-1}\|^2\right)$$
$$q(x_{t-1} \mid x_0) = \mathcal{N}\left(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_0, (1 - \bar{\alpha}_{t-1}) I\right) \propto \exp\left(-\frac{1}{2(1 - \bar{\alpha}_{t-1})} \|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0\|^2\right)$$

Combining terms and grouping powers of $x_{t-1}$:
1. **Coefficient of $x_{t-1}^2$**:
   $$\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} = \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})} = \frac{\alpha_t - \bar{\alpha}_t + 1 - \alpha_t}{\beta_t(1 - \bar{\alpha}_{t-1})} = \frac{1 - \bar{\alpha}_t}{\beta_t(1 - \bar{\alpha}_{t-1})} \equiv \frac{1}{\tilde{\beta}_t}$$
   Therefore, the posterior variance is:
   $$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

2. **Coefficient of $x_{t-1}$**:
   $$\left( \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0 \right) \cdot \tilde{\beta}_t$$
   Multiplying by $\tilde{\beta}_t = \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$:
   $$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

The posterior mean is an explicit, linear interpolation between the clean origin $x_0$ and the current noisy state $x_t$!

### Contrastive Analysis: Why X, Not Y?
- **Why is $\tilde{\mu}_t$ a weighted sum of both $x_0$ and $x_t$?**  
  $x_t$ provides the immediate noisy location, while $x_0$ provides the ground-truth anchor. As $t \to 1$, $\bar{\alpha}_t \to 1$, so the coefficient of $x_0$ dominates, driving the prediction to clean data. As $t \to T$, $\bar{\alpha}_t \to 0$, so $x_t$ dominates.

### Active Comprehension Checks
1. *Question*: What happens to $\tilde{\beta}_t$ when $t = 1$? (Note: $\bar{\alpha}_0 = 1$).  
   *Answer*: At $t=1$, $1 - \bar{\alpha}_0 = 1 - 1 = 0$. Thus $\tilde{\beta}_1 = 0$. Given $x_0$ and $x_1$, $x_0$ is deterministically known with zero uncertainty!

### Analogy for this topic only
A tug-of-war rope between where you are standing ($x_t$) and your destination ($x_0$). The weights on the rope are governed by how much noise has accumulated so far.
*In lecture words: The posterior mean is a precision-weighted combination of the past and the future.*

### Local picture
```
   x_0 -----[ Coef: sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t) ]-----+
                                                                             |-----> \tilde{mu}_t
   x_t -----[ Coef: sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)]-----+
```
*Notice: Both coefficients sum to approximately 1.*

### Bridge
We have derived the complete mathematical foundation of the ELBO and the analytical posterior parameters. Now let's explore practical debugging scenarios encountered when implementing this math in code.

---

## Workplace Debugging Scenarios

### Scenario 1: The Off-by-One Indexing Bug in Posterior Variance $\tilde{\beta}_t$
**Incident:** During implementation of the posterior variance function, the model generates corrupted images with pixel values wrapping around to extreme values at $t=1$.  
**Mathematical Root Cause:** The developer indexed $\bar{\alpha}_{t-1}$ using standard 0-indexing without handling the boundary condition at $t=1$ where $t-1=0$. In Python, `alpha_bars[-1]` was accessed instead of treating $\bar{\alpha}_0 \equiv 1.0$, pulling the terminal step $T$ variance instead of the initial state.  
**Debugging Steps:**
1. Check value of `beta_tilde` at $t=1$. It evaluated to $0.019$ instead of $0.0$.
2. Inspect the indexing guard: verify `alpha_bar_prev` at $t=1$ returns $1.0$.  
**Code Fix:**
```python
# CORRECT POSTERIOR VARIANCE IMPLEMENTATION:
def get_beta_tilde(t, betas, alpha_bars):
    # If t == 0 (representing first step), alpha_bar_prev is 1.0
    alpha_bar_prev = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0)
    beta_t = betas[t]
    alpha_bar_t = alpha_bars[t]
    return ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * beta_t
```

---

### Scenario 2: The Sign Inversion in KL Divergence Optimization
**Incident:** A research engineer computes the tripartite loss, but during training, the loss drops rapidly to $-\infty$ while generated images degrade into high-contrast static.  
**Mathematical Root Cause:** The engineer formulated the training loss as $\mathcal{L} = \text{ELBO}$ instead of $\mathcal{L} = -\text{ELBO}$. Because deep learning optimizers perform gradient descent, minimizing the ELBO drives likelihood to zero and pushes the model away from the data manifold.  
**Debugging Steps:**
1. Monitor whether the loss is positive or negative. A correct negative log-likelihood surrogate should be bounded below.
2. Confirm optimizer sign: optimization must minimize $-\text{ELBO} = L_T + \sum L_{t-1} + L_0$.  
**Code Fix:**
```python
# CORRECT OBJECTIVE:
loss = loss_prior + loss_denoise.sum() + loss_recon
loss.backward() # Minimizes negative ELBO (maximizes ELBO)
```

---

## References & Further Reading

For complete citations, seminal papers, academic lecture links, and official implementations, see [./references.md](./references.md).
