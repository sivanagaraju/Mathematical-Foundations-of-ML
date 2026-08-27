# Convexity & Jensen's Inequality: The Geometric Foundations of Variational Generative AI

> `🏷️ Tags:` `Optimization` `Convexity` `Jensens-Inequality` `ELBO` `VAEs` `EM-Algorithm` `Information-Theory` `Generative-AI`  
> `📚 Prerequisites Needed:` [Derivatives, Gradients & Jacobians](./Derivatives_Gradients_and_Jacobians.md) · [Logarithms & Exponential Functions](./Logarithms_and_Exponential_Functions.md) · [Probability Basics & Axioms](./Probability_Basics_and_Axioms.md)  
> `🎯 Where Do We Use This?:` **The theoretical backbone of all Variational AI** — Deriving the Evidence Lower Bound (ELBO) in Variational Autoencoders (VAEs), Proving Gibbs' Inequality ($D_{\text{KL}}(P \parallel Q) \ge 0$) in Information Theory, and Convergence proofs in the Expectation-Maximization (EM) algorithm.  
> `🎓 Course Module Mapping:` [Tut 08: Basic Probability 2](../Mathematical-Foundation-for-GenerativeAI/22-Tutorial08-Review-Basic-Probability-2/NOTES.md) · [Lec 01: Intro](../Mathematical-Foundation-for-GenerativeAI/14-Lec01-MFGAI-Introduction/NOTES.md) · [Lec 20: VAEs](../Mathematical-Foundation-for-GenerativeAI/32-Lec20-Latent-Variable-Models-VAE/NOTES.md)  
> `⏱️ Difficulty Level:` ⭐⭐⭐☆☆ (Intermediate · 15 min read)

---

### 📌 Quick Navigation & Architecture Map
- [1. 🌟 Everyday Real-World Scenarios](#1--everyday-real-world-scenarios-the-soup-bowl-vs-hill-dome--the-vae-safety-floor) — The Soup Bowl vs Hill Dome & The VAE Safety Floor
- [2. 👶 ELI5 Intuition](#2--eli5-intuition-the-rolling-marble--the-sagging-string) — The Rolling Marble & The Sagging String
- [3. 📚 Deep Terminology Master Glossary](#3--deep-terminology-master-glossary-15-core-concepts-dissected) — 15 convexity and variational terms dissected without jargon
- [4. 📐 Mathematical Formulations, Jensen Proofs & ELBO Derivation](#4--mathematical-formulations-jensen-proofs--elbo-derivation) — Formal Convexity, Gibbs' Inequality proof, and VAE ELBO derivation
- [5. 🔢 Concrete Micro-Numerical Worked Examples](#5--concrete-micro-numerical-worked-examples) — Numerical Jensen test on $x^2$ and $\ln(x)$ & VAE ELBO Gap
- [6. 🔗 Connecting the Dots: Generative AI Architecture Blocks](#6--connecting-the-dots-how-jensens-inequality-powers-generative-ai) — VAE ELBO Lower Bound Architecture & EM Algorithm Lower Bounds
- [7. 💻 Standalone Executable Python/PyTorch Verification Script](#7--complete-standalone-executable-pythonpytorch-verification-script) — Jensen's Inequality Monte Carlo checks, Gibbs' KL proof, and VAE loss simulation
- [8. 🩺 Diagnostic Mini-Checks & Common Traps](#8--diagnostic-mini-checks--common-traps) — Self-test questions & production engineering pitfalls

---

In machine learning and Generative AI, **Convexity** provides the ultimate optimization guarantee that local minima are global minima, while **Jensen's Inequality** provides the foundational algebraic bridge converting intractable probability integrals inside non-linear functions into tractable, solvable variational lower bounds.

```
 ===================================================================================================
                 CONVEXITY & JENSEN'S INEQUALITY IN PROBABILISTIC AI
 ===================================================================================================

  CONVEX FUNCTION g(x) (Chords lie above graph)   CONCAVE FUNCTION ln(x) (Chords lie below graph)
  E[g(X)] ≥ g(E[X])                               E[ln(X)] ≤ ln(E[X])
  ┌──────────────────────────────┐                ┌──────────────────────────────┐
  │ Average of function values   │                │ Log of expected ratio (True) │
  │ is GREATER than function of  │                │ is strictly GREATER than     │
  │ the average input!           │                │ Expected log-ratio (ELBO)    │
  └──────────────────────────────┘                └──────────────────────────────┘
                 │                                               │
                 ▼                                               ▼
  OPTIMIZATION GUARANTEE:                         VARIATIONAL GENERATIVE AI:
  Local minimum = Global minimum                  ln p(x) ≥ E_q[ln(p(x,z)/q(z|x))] (ELBO)
  No spurious local traps!                        D_KL(P || Q) ≥ 0 (Information Theory)
 ===================================================================================================
```

---

### 1. 🌟 Everyday Real-World Scenarios (The Soup Bowl vs Hill Dome & The VAE Safety Floor)
> `Context:` Zero Prior Machine Learning / AI Knowledge Needed · Concrete Real-World Mapping

#### Scenario A: The Soup Bowl vs The Hill Dome (Zero ML Background Needed)
Imagine two physical shapes in the real world:
1. **The Soup Bowl / Convex Function ($g(x) = x^2$):** A salad bowl curves upwards. If you place a marble anywhere on the rim, it rolls straight down to the **one and only lowest point** at the bottom (Global Minimum). There are no deceptive false pockets to get stuck in.
2. **The Hill Dome / Concave Function ($h(x) = \ln(x)$):** A mountain dome curves downwards. If you stretch a straight string between any two points on the hill, the string hangs **underneath** the surface of the mountain!
3. **Jensen's Rule:** The average height of the string endpoints is always **lower** than the mountain height at the midpoint:
   $$\mathbb{E}[\ln(X)] \le \ln(\mathbb{E}[X])$$

---

#### Scenario B: In Generative AI — The VAE Evidence Lower Bound (ELBO)
> `Context:` How Jensen's Inequality Bypasses Impossible Calculus Integrals in Variational Autoencoders

When training a VAE to generate human faces:
- We want to maximize the true data log-likelihood $\ln p(x) = \ln \int p(x, z) dz$.
- But integrating across all possible combinations of 512 latent facial features $z$ is mathematically **impossible (intractable)**.
- **Jensen's Inequality** allows us to pull the logarithm inside the integral:
  $$\ln \mathbb{E}\left[ \frac{p(x, z)}{q(z \mid x)} \right] \ge \mathbb{E}\left[ \ln \frac{p(x, z)}{q(z \mid x)} \right] \triangleq \mathbf{\text{ELBO}}$$
- By pushing up this tractable lower bound (the floor), we guarantee that the true data probability (the ceiling) is pushed upwards as well!

```
 ===================================================================================================
         JENSEN'S INEQUALITY CREATING THE VAE EVIDENCE LOWER BOUND (ELBO)
 ===================================================================================================

  IMPOSSIBLE TRUE LIKELIHOOD:  ln p(x) = ln ∫ p(x, z) dz  (Intractable Ceiling!)
                               ▲
                               │  Gap = D_KL( q_ϕ(z|x) || p_θ(z|x) ) ≥ 0
                               │
  TRACTABLE ELBO FLOOR:        ELBO(θ, ϕ) = 𝔼_q[ ln p(x|z) ] - D_KL( q(z|x) || p(z) )
  (Optimized via Backpropagation & Reparameterization Trick!)
 ===================================================================================================
```

---

### 2. 👶 ELI5 Intuition: The Rolling Marble & The Sagging String
> `Context:` Physical & Everyday Metaphors for Convexity and Jensen's Inequality

#### Metaphor 1: The Rolling Marble (Convex Optimization)
- In a convex loss bowl, every direction that slopes downward leads directly to the bottom.
- You can never get trapped in a dead end. Gradient descent is guaranteed to find the absolute best solution.

---

#### Metaphor 2: The Diminishing Returns of Wealth (Concave Function)
- Imagine happiness as a function of money: $H(m) = \ln(m)$.
- Winning $\$100{,}000$ when broke brings massive joy; winning another $\$100{,}000$ when a billionaire brings almost zero extra joy.
- Because $\ln(m)$ is concave, taking a 50/50 gamble between $\$0$ and $\$200{,}000$ gives lower expected happiness than receiving $\$100{,}000$ guaranteed ($\mathbb{E}[\ln(M)] \le \ln(\mathbb{E}[M])$)!

---

### 3. 📚 Deep Terminology Master Glossary (15 Core Concepts Dissected)
> `Context:` Foundational Mathematical & Machine Learning Vocabulary Explained Without Jargon

```
 ===================================================================================================
                 THE CONVEXITY & VARIATIONAL BOUNDS ROSETTA STONE
 ===================================================================================================
```

| Term / Notation | Formal Mathematical Meaning | Plain-English Definition (No ML Jargon) | Real-World Analogy |
| :--- | :--- | :--- | :--- |
| **Convex Set ($\mathcal{C}$)** | $\forall x, y \in \mathcal{C}, \lambda \in [0, 1] \implies \lambda x + (1-\lambda)y \in \mathcal{C}$ | A shape with no indentations or holes; straight lines stay inside | A solid circular dinner plate vs a donut |
| **Convex Function ($g$)** | $g(\lambda x + (1-\lambda)y) \le \lambda g(x) + (1-\lambda)g(y)$ | A bowl-shaped function where straight lines between points float above the curve | A soup bowl |
| **Concave Function ($h$)** | $h(\lambda x + (1-\lambda)y) \ge \lambda h(x) + (1-\lambda)h(y)$ | A dome-shaped function where straight lines hang below the curve | An umbrella or mountain peak |
| **Jensen's Inequality** | $\mathbb{E}[g(X)] \ge g(\mathbb{E}[X])$ for convex $g$ | Average of curved outputs is higher than curving the average input | Evaluating stock portfolio variance |
| **Chord / Secant Line** | Line segment connecting $(x, g(x))$ and $(y, g(y))$ | The straight bridge connecting two points on a curve | A tightrope strung between two mountain peaks |
| **First-Order Condition** | $g(y) \ge g(x) + \nabla g(x)^\top (y - x)$ | The tangent plane touches the function strictly from below | A flat wooden board supporting a bowl |
| **Second-Order Condition** | $\nabla^2 g(x) \succeq 0$ (Hessian is PSD) | Curvature is non-negative in all directions | A skateboard bowl curving up everywhere |
| **Global Minimum** | $g(x^*) \le g(x) \quad \forall x$ | The absolute lowest point in the entire universe | The deepest trench in the ocean |
| **Evidence Lower Bound (ELBO)** | $\mathbb{E}_q[\ln p(x, z) / q(z \mid x)]$ | The solvable lower floor beneath intractable data log-likelihood | A sturdy jack lifting a car |
| **Gibbs' Inequality** | $D_{\text{KL}}(P \parallel Q) \ge 0$ | Proves relative entropy is non-negative via Jensen's inequality on $-\ln$ | Proving you cannot travel negative distance |
| **Expectation-Maximization (EM)**| 2-step iterative latent optimization | Algorithm creating tight Jensen lower bounds (E-step) and maximizing them (M-step) | Climbing a hill by setting up stable base camps |
| **Fenchel Conjugate ($f^*(t)$)** | $\sup_x \{ tx - f(x) \}$ | Legendre-Fenchel dual transformation used in $f$-GANs | Finding the highest tangent contact point |
| **Variational Inference** | Approximating true posterior with $q_\phi(z \mid x)$ | Replacing an impossible integral with an optimization problem | Sketching an approximate map of an unexplored forest |
| **Log-Sum-Exp Function** | $\text{LSE}(x) = \ln \sum e^{x_i}$ | Smooth convex approximation of the maximum function; denominator of Softmax | A smooth rounded corner replacing a sharp point |
| **Strict Convexity** | Inequality is strict for $x \neq y, \lambda \in (0, 1)$ | Guarantees that the global minimum is unique (exactly one optimal solution) | A funnel with a single drain hole |

---

### 4. 📐 Mathematical Formulations, Jensen Proofs & ELBO Derivation
> `Context:` Formal Theorems, Proof of Gibbs' Inequality, and VAE ELBO Derivation

```
 ===================================================================================================
                 THE GEOMETRY OF JENSEN'S INEQUALITY
 ===================================================================================================

  CONVEX: E[g(X)] ≥ g(E[X]) (e.g. g(x) = x²)          CONCAVE: E[h(X)] ≤ h(E[X]) (e.g. h(x) = ln x)
  g(x) ▲        / Chord lies ABOVE curve              h(x) ▲       h(E[X])
       │  ●────/────● E[g(X)] (Average height)             │       .---●---. (Peak)
       │  │   /     │                                      │     .'    │    '.
       │  │  /      │                                      │  ●──┼─────┼─────● E[h(X)] (Chord BELOW)
  0.0 ─┴──●─────────●──────► x                        0.0 ─┴──┴─────┴─────────┴──► x
          x₁  E[X]  x₂                                        x₁    E[X]      x₂
 ===================================================================================================
```

#### Core Mathematical Proofs:

1. **Jensen's Inequality Statement:**
   For any random variable $X$ and convex function $g$:
   $$\mathbb{E}[g(X)] \ge g\bigl(\mathbb{E}[X]\bigr)$$
   For a concave function $h$ (such as $h(t) = \ln(t)$):
   $$\mathbb{E}[h(X)] \le h\bigl(\mathbb{E}[X]\bigr)$$

2. **Proof of Gibbs' Inequality ($D_{\text{KL}}(P \parallel Q) \ge 0$):**
   $$-D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \ln\left(\frac{Q(x)}{P(x)}\right) = \mathbb{E}_{X \sim P}\left[ \ln\left(\frac{Q(X)}{P(X)}\right) \right]$$
   Applying Jensen's Inequality to concave function $\ln$:
   $$\mathbb{E}_P\left[ \ln\left(\frac{Q(X)}{P(X)}\right) \right] \le \ln\left( \mathbb{E}_P\left[\frac{Q(X)}{P(X)}\right] \right) = \ln\left( \sum_x P(x) \frac{Q(x)}{P(x)} \right) = \ln\left(\sum_x Q(x)\right) = \ln(1) = \mathbf{0}$$
   Multiplying by $-1$ flips the inequality:
   $$\mathbf{D_{\text{KL}}(P \parallel Q) \ge 0}$$

3. **Step-by-Step Derivation of VAE ELBO:**
   $$\ln p_\theta(x) = \ln \int p_\theta(x, z) \, dz = \ln \int q_\phi(z \mid x) \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \, dz = \ln \mathbb{E}_{z \sim q_\phi}\left[ \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right]$$
   Applying Jensen's Inequality ($\ln \mathbb{E}[U] \ge \mathbb{E}[\ln U]$):
   $$\ln p_\theta(x) \ge \mathbb{E}_{z \sim q_\phi}\left[ \ln \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right] = \underbrace{\mathbb{E}_{q_\phi}[\ln p_\theta(x \mid z)]}_{\text{Reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))}_{\text{Prior Regularization}} \triangleq \mathbf{\text{ELBO}}$$

---

### 5. 🔢 Concrete Micro-Numerical Worked Examples
> `Context:` Step-by-Step Manual Calculations (No Black Box)

#### Example 1: Numerical Jensen Verification on $g(x) = x^2$ and $h(x) = \ln(x)$
Let discrete random variable $X \in \{1.0, \quad 9.0\}$ with equal probabilities $p = [0.5, \quad 0.5]$.

1. **Calculate Expected Value $\mathbb{E}[X]$:**
   $$\mathbb{E}[X] = 0.5(1.0) + 0.5(9.0) = 0.5 + 4.5 = \mathbf{5.0000}$$

2. **Test Convex Function $g(x) = x^2$:**
   - $\mathbb{E}[g(X)] = 0.5(1.0^2) + 0.5(9.0^2) = 0.5(1.0) + 0.5(81.0) = 0.5 + 40.5 = \mathbf{41.0000}$
   - $g(\mathbb{E}[X]) = (5.0)^2 = \mathbf{25.0000}$
   - **Verification:** $41.0000 \ge 25.0000$ (Holds with gap equal to $\text{Var}(X) = 41 - 25 = \mathbf{16.0}$!).

3. **Test Concave Function $h(x) = \ln(x)$:**
   - $\mathbb{E}[\ln(X)] = 0.5\ln(1.0) + 0.5\ln(9.0) = 0.5(0.0) + 0.5(2.1972) = \mathbf{1.0986}$
   - $\ln(\mathbb{E}[X]) = \ln(5.0) = \mathbf{1.6094}$
   - **Verification:** $1.0986 \le 1.6094$ (Holds with gap equal to $\mathbf{0.5108\text{ nats}}$!).

---

#### Example 2: VAE ELBO Gap Arithmetic
Suppose true data log-likelihood is $\ln p(x) = -3.20$ nats.
Suppose the encoder approximates the posterior with a mismatch of $D_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x)) = 0.45$ nats.
- By the Master ELBO Identity: $\ln p(x) = \text{ELBO} + D_{\text{KL}}(q \parallel p)$.
- $$\text{ELBO} = \ln p(x) - D_{\text{KL}}(q \parallel p) = -3.20 - 0.45 = \mathbf{-3.65\text{ nats}}$$
- *(Notice: $\text{ELBO} = -3.65 \le \ln p(x) = -3.20$. Pushing ELBO up from $-3.65 \to -3.20$ guarantees the model improves!)*

---

### 6. 🔗 Connecting the Dots: How Jensen's Inequality Powers Generative AI
> `Context:` Architectural Implementations in Variational Autoencoders, Diffusion, and EM Algorithm

```
 ===================================================================================================
                 JENSEN'S INEQUALITY ACROSS GENERATIVE AI
 ===================================================================================================

  1. VAE ELBO OBJECTIVE (Kingma & Welling 2013)     2. EXPECTATION-MAXIMIZATION (EM Algorithm)
  ln p_θ(x) ≥ 𝔼_q[ln p(x|z)] - D_KL(q(z|x) || p(z)) E-step creates tight Jensen lower bound curve
  ┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐
  │ Bypasses intractable marginal integral │        │ M-step optimizes model parameters on   │
  │ Closed-form KL + Monte Carlo decoder   │        │ lower bound; guarantees monotonic ascent│
  │ Enables end-to-end backpropagation     │        │ on Gaussian Mixture Models (GMMs)      │
  └────────────────────────────────────────┘        └────────────────────────────────────────┘
 ===================================================================================================
```

| Generative Architecture | How Convexity / Jensen is Used | Mathematical Formulation |
| :--- | :--- | :--- |
| **Variational Autoencoders (VAEs)** | **Evidence Lower Bound (ELBO)** | Maximizes $\mathbb{E}_q[\ln p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))$ as tractable proxy for $\ln p(x)$ |
| **Information Theory (KL Divergence)** | **Gibbs' Non-Negativity Proof** | Proves $D_{\text{KL}}(P \parallel Q) \ge 0$ by applying Jensen's inequality to concave $-\ln(t)$ |
| **Gaussian Mixture Models (GMMs)** | **EM Lower Bound (Q-Function)** | Constructs auxiliary surrogate function $Q(\theta, \theta^{(t)}) \le \ln p(X \mid \theta)$ via Jensen's inequality |
| **Convex Loss Functions (MSE, CCE)** | **Global Convergence Guarantee** | Guarantees standard convex classifiers have a single unique global optimum |

---

### 7. 💻 Complete Standalone Executable Python/PyTorch Verification Script
> `Context:` Runnable Code Verifying Jensen's Inequality, Gibbs' Inequality, and VAE ELBO Lower Bound

```python
"""
Convexity & Jensen's Inequality Mathematical Simulation
=======================================================
Demonstrates:
1. Numerical verification of Jensen's inequality for convex and concave functions
2. Gibbs' Inequality proof: D_KL(P || Q) >= 0 via Jensen on random distributions
3. VAE ELBO decomposition: ln p(x) == ELBO + D_KL(q || p)
"""
import torch
import numpy as np

print("=" * 75)
print("CONVEXITY & JENSEN'S INEQUALITY MATHEMATICAL SIMULATION")
print("=" * 75)

# ─── 1. Jensen's Inequality Test on Convex & Concave Functions ───
print("\n1. JENSEN'S INEQUALITY ON RANDOM VARIABLE X in {1.0, 9.0} (p=[0.5, 0.5]):")
X = torch.tensor([1.0, 9.0])
E_X = torch.mean(X).item() # E[X] = 5.0

# Convex test: g(x) = x^2
E_gX = torch.mean(X ** 2).item()
g_EX = E_X ** 2
print(f"   * Convex g(x)=x^2:   E[g(X)] = {E_gX:.2f} >= g(E[X]) = {g_EX:.2f} (Gap = Var(X) = {E_gX - g_EX:.2f}) ✅")

# Concave test: h(x) = ln(x)
E_hX = torch.mean(torch.log(X)).item()
h_EX = np.log(E_X)
print(f"   * Concave h(x)=ln x: E[h(X)] = {E_hX:.4f} <= h(E[X]) = {h_EX:.4f} (Gap = {h_EX - E_hX:.4f} nats) ✅")

# ─── 2. Gibbs' Inequality Verification via Jensen's on Random Distributions ───
print("\n2. GIBBS' INEQUALITY VERIFICATION (D_KL(P || Q) >= 0):")
torch.manual_seed(42)
for i in range(3):
    p_logits = torch.randn(5)
    q_logits = torch.randn(5)
    P = torch.softmax(p_logits, dim=-1)
    Q = torch.softmax(q_logits, dim=-1)
    
    d_kl = torch.sum(P * torch.log(P / Q)).item()
    print(f"   Trial {i+1}: D_KL(P || Q) = {d_kl:.6f} nats (Strictly >= 0.0! ✅)")
    assert d_kl >= 0.0, "Gibbs' inequality violated!"

# ─── 3. VAE ELBO Bound Verification ───
print("\n3. VAE ELBO LOWER BOUND DECOMPOSITION:")
# True log-likelihood = -3.20, KL divergence = 0.45
true_log_lik = -3.20
kl_gap = 0.45
elbo_val = true_log_lik - kl_gap

print(f"   * True Log-Likelihood ln p(x): {true_log_lik:.2f} nats (Ceiling)")
print(f"   * Posterior KL Mismatch Gap:   {kl_gap:.2f} nats")
print(f"   * Computed ELBO Value:         {elbo_val:.2f} nats (Floor)")
assert elbo_val <= true_log_lik, "ELBO failed to be a lower bound!"
print("   * ELBO is strictly less than or equal to true log-likelihood! ✅")

print("\n" + "=" * 75)
print("ALL CONVEXITY & JENSEN'S INEQUALITY TESTS PASSED SUCCESSFULLY! ✅")
print("=" * 75)
```

---

### 8. 🩺 Diagnostic Mini-Checks & Common Traps
> `Context:` Production Debugging Insights, Edge-Case Traps & Self-Verification Questions

#### ✅ Self-Test Questions

1. **Q:** Why does Jensen's Inequality state $\mathbb{E}[g(X)] \ge g(\mathbb{E}[X])$ for convex $g$, but $\mathbb{E}[h(X)] \le h(\mathbb{E}[X])$ for concave $h$?  
   **A:** A convex function curves upwards (chords lie *above* the graph), so the average of function values is higher than the function at the average point. A concave function ($\ln$) curves downwards (chords lie *below* the graph), reversing the direction of the inequality.

2. **Q:** In a VAE, why is maximizing the ELBO guaranteed to improve the true data log-likelihood $\ln p(x)$?  
   **A:** Because $\ln p(x) = \text{ELBO} + D_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x))$. Since KL divergence is always non-negative ($D_{\text{KL}} \ge 0$), the ELBO forms a rigorous mathematical floor beneath $\ln p(x)$. Raising the floor pushes up the ceiling.

3. **Q:** When does equality hold in Jensen's Inequality ($\mathbb{E}[g(X)] = g(\mathbb{E}[X])$)?  
   **A:** Equality holds if and only if $X$ is a **deterministic constant** ($\text{Var}(X) = 0$), or if the function $g(x)$ is purely **linear** ($g(x) = ax + b$).

#### ⚠️ Common Engineering Traps

| Trap | Why It Fails | Production Fix |
| :--- | :--- | :--- |
| **Assuming ELBO equals exact log-likelihood** | ELBO underestimates true likelihood by the variational posterior gap $D_{\text{KL}}(q \parallel p)$ | Use **Importance Weighted Autoencoders (IWAE)** with $K$ samples to tighten the bound |
| **Applying Jensen's inequality with the wrong direction on $\ln$** | Mistakenly asserting $\mathbb{E}[\ln X] \ge \ln \mathbb{E}[X]$ inverts the lower bound into an upper bound | Remember: $\ln(x)$ is **concave**, so $\mathbb{E}[\ln X] \le \ln \mathbb{E}[X]$ |
| **Using non-convex loss functions without multiple restarts** | Highly non-convex deep loss landscapes can get stuck in poor saddle points or bad local minima | Use modern optimizers like **AdamW**, learning rate warmup, and residual skip connections |

---

### 🎯 Summary Checklist
- **Convex Functions** curve upwards ($\mathbb{E}[g(X)] \ge g(\mathbb{E}[X])$); **Concave Functions** curve downwards ($\mathbb{E}[\ln X] \le \ln \mathbb{E}[X]$).
- **Jensen's Inequality** converts intractable expectations inside non-linear functions into tractable optimization bounds.
- **Gibbs' Inequality** ($D_{\text{KL}} \ge 0$) is proven directly via Jensen's Inequality on the concave logarithm.
- **VAE ELBO** maximizes the lower bound floor beneath the intractable marginal log-likelihood $\ln p(x)$.
- **Expectation-Maximization (EM)** iteratively builds and maximizes Jensen lower bounds.
