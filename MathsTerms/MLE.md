# Maximum Likelihood Estimation (MLE): Tuning Parameters to Match Reality

> `🏷️ Tags:` `Statistics` `MLE` `Parameter-Estimation` `Log-Likelihood` `Gaussian-MLE` `Bernoulli-MLE` `Generative-AI`  
> `📚 Prerequisites Needed:` [Likelihood & Log-Likelihood](./Likelihood_and_Log_Likelihood.md) · [Probability Basics & Axioms](./Probability_Basics_and_Axioms.md) · [Derivatives, Gradients & Jacobians](./Derivatives_Gradients_and_Jacobians.md)  
> `🎯 Where Do We Use This?:` **The foundational optimization principle in Machine Learning & Generative AI** — Pre-training Large Language Models (LLaMA-3, GPT-4), Training Diffusion Models via Gaussian score matching, Fitting Gaussian Mixture Models (GMMs), and Deriving Mean Squared Error (MSE) and Cross-Entropy loss functions.  
> `🎓 Course Module Mapping:` [Tut 08: Basic Probability 2](../Mathematical-Foundation-for-GenerativeAI/22-Tutorial08-Review-Basic-Probability-2/NOTES.md) · [Lec 01: Intro](../Mathematical-Foundation-for-GenerativeAI/14-Lec01-MFGAI-Introduction/NOTES.md) · [Lec 20: VAEs](../Mathematical-Foundation-for-GenerativeAI/32-Lec20-Latent-Variable-Models-VAE/NOTES.md)  
> `⏱️ Difficulty Level:` ⭐⭐☆☆☆ (Foundational · 15 min read)

---

### 📌 Quick Navigation & Architecture Map
- [1. 🌟 Everyday Real-World Scenarios](#1--everyday-real-world-scenarios-the-mystery-vault-combination--training-neural-models) — The Mystery Vault Combination & Training Neural Models
- [2. 👶 ELI5 Intuition](#2--eli5-intuition-the-radio-frequency-tuner--the-biased-coin) — The Radio Frequency Tuner & The Biased Coin
- [3. 📚 Deep Terminology Master Glossary](#3--deep-terminology-master-glossary-15-core-concepts-dissected) — 15 MLE terms dissected without jargon
- [4. 📐 Mathematical Formulations, Gaussian & Bernoulli Proofs](#4--mathematical-formulations-gaussian--bernoulli-proofs) — Formal MLE definition, Bernoulli coin-flip proof, and Gaussian sample mean/variance derivation
- [5. 🔢 Concrete Micro-Numerical Worked Examples](#5--concrete-micro-numerical-worked-examples) — 5 Coin Flips Bernoulli Hand Calculation & Gaussian Sample Variance
- [6. 🔗 Connecting the Dots: Generative AI Architecture Blocks](#6--connecting-the-dots-how-mle-powers-generative-ai) — LLM Next-Token MLE, Diffusion Noise Prediction as Gaussian MLE, and VAE ELBO
- [7. 💻 Standalone Executable Python/PyTorch Verification Script](#7--complete-standalone-executable-pythonpytorch-verification-script) — Analytical MLE vs PyTorch Gradient Descent optimization
- [8. 🩺 Diagnostic Mini-Checks & Common Traps](#8--diagnostic-mini-checks--common-traps) — Self-test questions & production engineering pitfalls

---

**Maximum Likelihood Estimation (MLE)** is the foundational statistical optimization principle in machine learning: given an observed empirical dataset $D = \{x_1, \dots, x_n\}$, find the parameter configuration $\theta^*$ that makes the observed data most probable under the model family $p_\theta$.

```
 ===================================================================================================
                 THE 3-STAGE MAXIMUM LIKELIHOOD ESTIMATION (MLE) PIPELINE
 ===================================================================================================

  STAGE 1: OBSERVED DATA (D)           STAGE 2: PARAMETRIC FAMILY (p_θ)     STAGE 3: ARGMAX LOG-LIKELIHOOD
  Fixed in stone from nature           Adjustable Dials / Weights θ         Find optimal setting θ*
  ┌──────────────────────────────┐    ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │ D = {x₁, x₂, ..., xₙ}        │───►│ p_θ(x) (Gaussian, LLM, VAE)  │───►│ θ_MLE = argmax_θ ∑ ln p_θ(xᵢ)│
  │ [ H, H, H, T, H ]            │    │ Dial θ (mean, weights W, b)  │    │ Set derivative to 0 or use   │
  │ Images, Tokens, Audio        │    │ Candidate hypotheses         │    │ Gradient Descent to find peak│
  └──────────────────────────────┘    └──────────────────────────────┘    └──────────────┬───────────────┘
                                                                                         │
                                                                                         ▼
                                                                            OPTIMAL GENERATIVE MODEL
 ===================================================================================================
```

---

### 1. 🌟 Everyday Real-World Scenarios (The Mystery Vault Combination & Training Neural Models)
> `Context:` Zero Prior Machine Learning / AI Knowledge Needed · Concrete Real-World Mapping

#### Scenario A: The Mystery Vault Combination (Zero ML Background Needed)
Imagine a bank vault found open in the morning with a specific combination:
1. **The Observed Data ($D$):** The combination dials show `[7, 3, 9]`. This fact is **fixed in stone**.
2. **The Adjustable Gear Configurations ($\theta$):** The lock mechanism has adjustable internal gear ratios $\theta$.
3. **The Likelihood Test ($L(\theta)$):**
   - If the gear was at setting $\theta_1$, there was only a $0.01\%$ chance the lock would stop at `[7, 3, 9]`.
   - If the gear was at setting $\theta_2$, there was an $85\%$ chance.
4. **The MLE Decision ($\theta^*$):** The investigator concludes configuration $\theta_2$ was active because $\theta_2$ **maximizes the probability of the observed numbers**.

---

#### Scenario B: In Generative AI — Training Billion-Parameter Language Models
> `Context:` How Modern Deep Learning Optimizes Weights via Maximum Likelihood Estimation

When pre-training an LLM:
- The human text corpus $D$ is fixed on disk.
- We initialize 100 billion neural weights $\theta$ randomly.
- **Maximum Likelihood Estimation** updates $\theta$ via gradient ascent:
  $$\theta^* = \arg\max_\theta \sum_{i=1}^N \ln p_\theta(x_i)$$
- When parameters reach $\theta^*$, the model accurately reproduces human grammar, syntax, and logic!

```
 ===================================================================================================
         MLE AS PARAMETER TUNING ACROSS MACHINE LEARNING
 ===================================================================================================

  FIXED DATASET D                    PARAMETRIC MODEL FAMILY              OPTIMAL PARAMETERS θ_MLE
  4 Heads, 1 Tail (Fixed)            Bernoulli(θ) where θ = P(Head)       θ* = 4/5 = 0.80 (80% Heads)
  ┌──────────────────────────────┐   ┌──────────────────────────────┐     ┌──────────────────────────────┐
  │ D = [ H, H, H, T, H ]        │──►│ L(θ) = θ⁴ · (1 - θ)¹         │───► │ Peak of curve at θ = 0.80    │
  │ Real-world observations      │   │ Parameter dial θ ∈ [0, 1]    │     │ Makes 4 Heads most probable! │
  └──────────────────────────────┘   └──────────────────────────────┘     └──────────────────────────────┘
 ===================================================================================================
```

---

### 2. 👶 ELI5 Intuition: The Radio Frequency Tuner & The Biased Coin
> `Context:` Physical & Everyday Metaphors for Maximum Likelihood Estimation

#### Metaphor 1: Tuning a Radio Knob
- The radio song broadcast over the airwaves is the fixed data $D$.
- The tuner dial is the parameter $\theta$.
- Turning the dial left or right changes the static.
- **MLE** is turning the dial until the song comes in with $100\%$ perfect fidelity.

---

#### Metaphor 2: The 5 Coin Flips
- You flip an unknown coin 5 times and get 4 Heads and 1 Tail.
- What is the most plausible chance of getting a Head?
- Common sense says $80\%$ ($4 / 5$). **MLE** is the exact mathematical proof confirming that common-sense intuition!

---

### 3. 📚 Deep Terminology Master Glossary (15 Core Concepts Dissected)
> `Context:` Foundational Mathematical & Machine Learning Vocabulary Explained Without Jargon

```
 ===================================================================================================
                 THE MAXIMUM LIKELIHOOD ESTIMATION (MLE) ROSETTA STONE
 ===================================================================================================
```

| Term / Notation | Formal Mathematical Meaning | Plain-English Definition (No ML Jargon) | Real-World Analogy |
| :--- | :--- | :--- | :--- |
| **Maximum Likelihood (MLE)** | $\arg\max_\theta \sum \ln p_\theta(x_i)$ | Finding parameter values that make observed training data most probable | Picking the suspect whose story best fits the crime scene |
| **Parametric Family ($p_\theta$)**| Distribution family indexed by $\theta$ | The mathematical template (e.g. Gaussian, Transformer) with adjustable knobs | Choosing a cake recipe template before adjusting sugar |
| **Log-Likelihood ($\ell(\theta)$)** | $\sum \ln p(x_i \mid \theta)$ | Additive score measuring model fit; converts multiplication to addition | Adding up test scores on an exam |
| **Negative Log-Likelihood (NLL)** | $-\ell(\theta)$ | The loss function minimized during gradient descent | Penalty points: fewer points means better fit |
| **Analytical MLE** | Setting derivative $\nabla_\theta \ell = 0$ | Finding the exact mathematical formula for optimal weights on paper | Solving a quadratic equation with the quadratic formula |
| **Numerical MLE** | Optimizing via Gradient Descent | Iteratively stepping toward the peak when formulas cannot be solved by hand | Hiking up a mountain in thick fog |
| **Consistency Property** | $\theta_{\text{MLE}} \xrightarrow{P} \theta_0$ as $N \to \infty$ | Guarantee that with infinite data, MLE discovers the true parameter values | Polling every citizen in a country reveals true election winner |
| **Asymptotic Efficiency** | Achieves Cramér-Rao Lower Bound | MLE achieves the lowest possible estimation variance among all estimators | An engine operating at maximum possible Carnot efficiency |
| **Cramér-Rao Lower Bound** | $\text{Var}(\hat{\theta}) \ge \frac{1}{I(\theta)}$ | The theoretical minimum variance achievable by any unbiased estimator | The speed of light limit in physics |
| **Estimator Bias** | $\mathbb{E}[\hat{\theta}] - \theta_0$ | The systematic error or offset of an estimator away from the true value | A bathroom scale that is calibrated 2 lbs too light |
| **Gaussian Mean MLE** | $\hat{\mu} = \frac{1}{N} \sum x_i$ | Proves the sample average is the optimal center of a Gaussian distribution | Taking the average height of students in a class |
| **Gaussian Variance MLE** | $\hat{\sigma}^2 = \frac{1}{N} \sum (x_i - \hat{\mu})^2$ | Proves the sample variance is the optimal spread (biased by $\frac{N-1}{N}$) | Measuring how widely exam scores are scattered |
| **Bernoulli MLE** | $\hat{p} = \frac{\text{Successes}}{N}$ | Proves the fraction of successes is the optimal probability estimate | Calculating batting average in baseball |
| **Maximum A Posteriori (MAP)** | $\arg\max_\theta [ \ell(\theta) + \ln p(\theta) ]$ | Bayesian variant of MLE that incorporates prior beliefs (acts like $L_2$ weight decay) | Guessing suspect with prior criminal record taken into account |
| **KL Minimization Equivalence**| $\arg\max \mathbb{E}[\ln p_\theta] \equiv \arg\min D_{\text{KL}}$ | Proves MLE is mathematically identical to minimizing KL divergence to reality | Shaping clay to match an original sculpture |

---

### 4. 📐 Mathematical Formulations, Gaussian & Bernoulli Proofs
> `Context:` Formal MLE Definition, Bernoulli Coin Proof, and Gaussian Sample Mean/Variance Proof

```
 ===================================================================================================
                 THE TWO MASTER CLOSED-FORM MLE PROOFS
 ===================================================================================================

  1. BERNOULLI FLIP MLE:               2. GAUSSIAN MEAN & VARIANCE MLE:
  D = {H, H, H, T, H} (k=4, N=5)       D = {x₁, x₂, ..., xₙ}
  d/dp [ 4 ln p + 1 ln(1-p) ] = 0      ∂ℓ/∂μ = 0  ──►  μ̂_MLE = (1/n) ∑ xᵢ  (Sample Mean!)
  4/p = 1/(1-p) ──► p̂_MLE = 4/5 = 0.80 ∂ℓ/∂σ² = 0 ──►  σ̂²_MLE = (1/n) ∑ (xᵢ - μ̂)² (Sample Var!)
 ===================================================================================================
```

#### Core Mathematical Proofs:

1. **Formal Definition of Maximum Likelihood Estimator:**
   $$\theta_{\text{MLE}} \triangleq \arg\max_{\theta \in \Theta} \sum_{i=1}^N \ln p_\theta(x_i) = \arg\min_{\theta \in \Theta} \left[ -\sum_{i=1}^N \ln p_\theta(x_i) \right]$$

2. **Proof: Bernoulli Maximum Likelihood (Coin Flipping):**
   Let $D$ contain $k$ heads in $N$ independent flips:
   $$L(p) = p^k (1 - p)^{N - k} \implies \ell(p) = k \ln(p) + (N - k) \ln(1 - p)$$
   Taking derivative w.r.t $p$ and setting to $0$:
   $$\frac{d\ell}{dp} = \frac{k}{p} - \frac{N - k}{1 - p} = 0 \implies \frac{k}{p} = \frac{N - k}{1 - p}$$
   $$k(1 - p) = p(N - k) \implies k - kp = Np - kp \implies Np = k \implies \mathbf{\hat{p}_{\text{MLE}} = \frac{k}{N}}$$

3. **Proof: Gaussian Mean and Variance MLE:**
   Given I.I.D. samples $X \sim \mathcal{N}(\mu, \sigma^2)$:
   $$\ell(\mu, \sigma^2) = -\frac{N}{2}\ln(2\pi) - \frac{N}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (x_i - \mu)^2$$
   - **For Mean $\mu$:**
     $$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^N (x_i - \mu) = 0 \implies \sum_{i=1}^N x_i - N\mu = 0 \implies \mathbf{\hat{\mu}_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N x_i}$$
   - **For Variance $\sigma^2$:**
     $$\frac{\partial \ell}{\partial \sigma^2} = -\frac{N}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^N (x_i - \hat{\mu})^2 = 0 \implies \mathbf{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N (x_i - \hat{\mu})^2}$$

---

### 5. 🔢 Concrete Micro-Numerical Worked Examples
> `Context:` Step-by-Step Manual Calculations (No Black Box)

#### Example 1: Bernoulli Coin-Flip MLE by Hand
Suppose you flip a coin $N = 5$ times and observe $D = [\text{Head}, \text{Head}, \text{Head}, \text{Tail}, \text{Head}]$ ($k = 4$ Heads, $1$ Tail).

1. **Evaluate Likelihood $L(p) = p^4 (1-p)^1$ across candidates:**
   - If $p = 0.10$: $L(0.1) = (0.1)^4(0.9) = 0.0001 \times 0.9 = \mathbf{0.00009}$
   - If $p = 0.50$: $L(0.5) = (0.5)^4(0.5) = 0.0625 \times 0.5 = \mathbf{0.03125}$
   - If $p = 0.80$: $L(0.8) = (0.8)^4(0.2) = 0.4096 \times 0.2 = \mathbf{0.08192\text{ (PEAK!)}}$
   - If $p = 0.90$: $L(0.9) = (0.9)^4(0.1) = 0.6561 \times 0.1 = \mathbf{0.06561}$

2. **Analytical Check:**
   $$\hat{p}_{\text{MLE}} = \frac{k}{N} = \frac{4}{5} = \mathbf{0.8000\text{ (80\% Heads)}}$$

---

#### Example 2: Gaussian Mean & Variance on $\{2.0, 4.0, 6.0\}$
1. **Sample Mean MLE:**
   $$\hat{\mu}_{\text{MLE}} = \frac{2.0 + 4.0 + 6.0}{3} = \frac{12.0}{3} = \mathbf{4.0000}$$
2. **Sample Variance MLE:**
   $$\hat{\sigma}^2_{\text{MLE}} = \frac{(2-4)^2 + (4-4)^2 + (6-4)^2}{3} = \frac{4 + 0 + 4}{3} = \frac{8}{3} \approx \mathbf{2.6667}$$

---

### 6. 🔗 Connecting the Dots: How MLE Powers Generative AI
> `Context:` Architectural Implementations in Large Language Models, Diffusion Models, and VAEs

```
 ===================================================================================================
                 MLE ACROSS GENERATIVE AI ARCHITECTURES
 ===================================================================================================

  1. AUTOREGRESSIVE LLMS (GPT-4 / LLaMA-3)          2. DIFFUSION SCORE MATCHING (Flux / SD3)
  θ* = argmax_θ ∑ ln p_θ(w_t | w_<t)                Gaussian noise model converts MLE into:
  ┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐
  │ Cross-Entropy loss is exact Categorical│        │ min_θ 𝔼[ ||ϵ - ϵ_θ(x_t, t)||² ]        │
  │ Maximum Likelihood Estimation          │        │ Mean Squared Error on noise is exact   │
  │ over the token vocabulary simplex      │        │ Gaussian Maximum Likelihood Estimation!│
  └────────────────────────────────────────┘        └────────────────────────────────────────┘
 ===================================================================================================
```

| Generative System | How MLE is Formulated | Architectural Purpose |
| :--- | :--- | :--- |
| **Large Language Models (LLaMA-3, Claude)** | **Categorical MLE via Cross-Entropy** | Optimizes token transition probabilities to minimize KL divergence to human literature |
| **Diffusion Models (Stable Diffusion, Flux)** | **Gaussian MLE via Noise MSE** | Noise prediction MSE is the analytical Maximum Likelihood objective under Gaussian noise |
| **Variational Autoencoders (VAEs)** | **Approximate Marginal MLE via ELBO** | Maximizes the Evidence Lower Bound when true marginal MLE integral $\int p(x, z) dz$ is intractable |
| **Normalizing Flows (RealNVP, Glow)** | **Exact Analytical MLE** | Invertible architectures compute exact log-likelihood via Jacobian change of variables |

---

### 7. 💻 Complete Standalone Executable Python/PyTorch Verification Script
> `Context:` Runnable Code Verifying Analytical MLE vs PyTorch Gradient Descent Optimization

```python
"""
Maximum Likelihood Estimation (MLE) Simulation
==============================================
Demonstrates:
1. Exact Bernoulli coin-flip MLE derivation vs numerical curve peak
2. Gaussian Mean and Variance closed-form MLE calculation
3. PyTorch Gradient Descent convergence to exact analytical MLE
"""
import torch
import numpy as np

print("=" * 75)
print("MAXIMUM LIKELIHOOD ESTIMATION (MLE) MATHEMATICAL SIMULATION")
print("=" * 75)

# ─── 1. Bernoulli Coin Flip Analytical vs Numerical MLE ───
print("\n1. BERNOULLI COIN FLIP MLE (4 Heads, 1 Tail):")
p_candidates = np.linspace(0.01, 0.99, 100)
likelihoods = (p_candidates ** 4) * ((1.0 - p_candidates) ** 1)
optimal_idx = np.argmax(likelihoods)
optimal_p_numerical = p_candidates[optimal_idx]
optimal_p_analytic = 4.0 / 5.0 # 0.80

print(f"   * Analytical Closed-Form MLE:  {optimal_p_analytic:.4f} (4/5) ✅")
print(f"   * Numerical Grid Search Peak:  {optimal_p_numerical:.4f} ✅")

# ─── 2. Gaussian Sample Mean & Variance Closed-Form MLE ───
print("\n2. GAUSSIAN MEAN & VARIANCE MLE ON DATA D = {2.0, 4.0, 6.0}:")
data = torch.tensor([2.0, 4.0, 6.0])
mu_mle = torch.mean(data).item()
var_mle = torch.mean((data - mu_mle)**2).item()

print(f"   * Sample Mean MLE (mu):        {mu_mle:.4f} (Analytic: 4.0000) ✅")
print(f"   * Sample Variance MLE (sigma^2): {var_mle:.4f} (Analytic: 8/3 = 2.6667) ✅")

# ─── 3. PyTorch Gradient Descent Optimization toward Analytical MLE ───
print("\n3. PYTORCH GRADIENT DESCENT CONVERGENCE TO EXACT MLE:")
mu_param = torch.tensor([0.0], requires_grad=True) # Initialize far from 4.0
optimizer = torch.optim.SGD([mu_param], lr=0.1)

for step in range(50):
    optimizer.zero_grad()
    # Loss = NLL = 0.5 * sum((x - mu)^2) (Ignoring constants)
    nll_loss = 0.5 * torch.sum((data - mu_param)**2)
    nll_loss.backward()
    optimizer.step()

print(f"   * Initial Parameter Value:     0.0000")
print(f"   * 50 Steps Optimized Value:    {mu_param.item():.4f} (Converged to analytical MLE = 4.0000! ✅)")
assert np.isclose(mu_param.item(), 4.0, atol=1e-3), "Gradient descent failed to find MLE!"

print("\n" + "=" * 75)
print("ALL MAXIMUM LIKELIHOOD ESTIMATION TESTS PASSED SUCCESSFULLY! ✅")
print("=" * 75)
```

---

### 8. 🩺 Diagnostic Mini-Checks & Common Traps
> `Context:` Production Debugging Insights, Edge-Case Traps & Self-Verification Questions

#### ✅ Self-Test Questions

1. **Q:** Why is the Gaussian sample variance MLE $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{N}\sum (x_i - \hat{\mu})^2$ a biased estimator?  
   **A:** Because the sample mean $\hat{\mu}$ is estimated from the exact same data, the sum of squared differences systematically underestimates the true population spread by a factor of $\frac{N-1}{N}$ ($\mathbb{E}[\hat{\sigma}^2] = \frac{N-1}{N}\sigma^2$). Bessel's correction uses $\frac{1}{N-1}$ to make it unbiased.

2. **Q:** What is the difference between Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP)?  
   **A:** **MLE** maximizes only data likelihood ($\arg\max_\theta \ln p(X \mid \theta)$). **MAP** is Bayesian; it adds a prior belief distribution over parameters ($\arg\max_\theta [ \ln p(X \mid \theta) + \ln p(\theta) ]$). A Gaussian prior on weights is mathematically identical to $L_2$ weight decay!

3. **Q:** Why is Minimizing Cross-Entropy Loss identical to Maximum Likelihood Estimation?  
   **A:** Cross-Entropy loss is defined as $\mathcal{L}_{\text{CE}} = -\sum y_i \ln \hat{p}_i = -\ln \hat{p}_{\text{true}}$. Minimizing $-\ln \hat{p}$ is mathematically identical to maximizing $\ln \hat{p}$, which is the exact definition of MLE.

#### ⚠️ Common Engineering Traps

| Trap | Why It Fails | Production Fix |
| :--- | :--- | :--- |
| **Assuming MLE is robust to extreme outliers** | Gaussian MLE uses squared errors $(x-\mu)^2$; a single massive outlier corrupts the mean estimate | Use **Laplace MLE ($L_1$ / MAE)** or Huber loss for robust estimation |
| **Overfitting on small sample sizes with unregularized MLE** | Maximizing pure likelihood on small datasets causes weights to explode | Add Bayesian priors (MAP) via **$L_2$ Weight Decay / AdamW** |
| **Confusing population variance ($N$) with sample variance ($N-1$)** | Small $N$ estimates underreport true variance if Bessel's correction is omitted | Use `torch.var(data, unbiased=True)` for unbiased sample variance |

---

### 🎯 Summary Checklist
- **Maximum Likelihood Estimation (MLE)** tunes model parameters $\theta$ to make observed data most probable.
- **Analytical MLE:** $\nabla_\theta \ell(\theta) = 0$ provides closed-form equations for simple distributions (Sample Mean $\hat{\mu} = \frac{1}{N}\sum x_i$).
- **Numerical MLE:** Complex deep networks optimize MLE via backpropagation and AdamW gradient descent.
- **MLE is Asymptotically Efficient & Consistent:** With sufficient data, it achieves minimum possible variance.
- **Cross-Entropy & Noise MSE** are exact implementations of Maximum Likelihood Estimation in modern Generative AI.
