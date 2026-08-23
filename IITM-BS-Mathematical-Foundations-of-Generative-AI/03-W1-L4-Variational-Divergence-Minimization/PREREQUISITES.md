# Prerequisites — Comprehensive Mathematical Foundations for Variational Divergence Minimization (W1_L4)

> **Welcome to the Elite Mathematical Foundations of Generative AI!**  
> Before diving into the lecture notes in [NOTES.md](./NOTES.md), use this master preparatory guide to build complete mathematical intuition for probability modeling, convex analysis, statistical divergences, pushforward measure theory, and the $f$-divergence family.  
> Written with deep pedagogical clarity: visual ASCII diagrams, intuitive real-world analogies, step-by-step algebraic proofs, concrete numerical calculations, actionable engineering scenarios, runnable Python code, and knowledge-check diagnostics.

```
══════════════════════════════════════════════════════════════════════════════════════════════════
                               THE 8 FOUNDATIONAL PILLARS OF W1_L4
══════════════════════════════════════════════════════════════════════════════════════════════════
 §1 Unknown Population p_x vs Empirical Dataset D ──► Population Law vs Finite Sample Observations
 §2 The Dual Objective: Estimate p_x AND Sample   ──► Density Fitting vs Synthesizing Novel Samples
 §3 Deep Neural Parametric Models p_θ             ──► Universal Approximators & Weight Manifolds
 §4 Statistical Divergences as Discrepancy Scores ──► Distance-Like Penalties on Probability Manifolds
 §5 The Pushforward Generative Engine G_θ(z)      ──► Transforming Latent Priors into High-D Data
 §6 Deep Dive: Convex Analysis & Jensen's Bound   ──► Epigraphs, Secants, Subgradients & Strict Proofs
 §7 The Unified Csiszár-Ali-Silvey f-Divergence   ──► The Integral Form, Scalar Ratios & Non-Negativity
 §8 The Exhaustive f-Divergence Catalog & Scenarios─► KL, Rev-KL, JS, TV, Pearson χ², Hellinger & Alpha
══════════════════════════════════════════════════════════════════════════════════════════════════
```

---

## 1. Unknown Population Law $p_x$ vs Empirical Dataset $\mathcal{D}$

<a id="p1-px"></a>

### Why It Matters for Lecture 4
All generative modeling begins with a fundamental asymmetry in data science: reality generates phenomena through a complex, continuous, multi-dimensional probability density function $p_x$, but the machine learning engineer only ever possesses a finite, discrete collection of historical observations $\mathcal{D} = \{x_1, \dots, x_n\}$. We can never query $p_x(x)$ analytically.

### Formal & Intuitive Architecture

```
         TRUE PHYSICAL UNIVERSE (p_x)                     EMPIRICAL DATASET (D)
   ┌────────────────────────────────────────┐          ┌───────────────────────────┐
   │ Continuous Density Function on ℝᴰ      │  ──IID─► │ Finite set of n vectors   │
   │ Infinite possible high-res photographs │  Draws   │ D = {x₁, x₂, ..., xₙ} ⊂ ℝᴰ│
   │ FORMULA IS COMPLETELY UNKNOWN!         │          │ WE HOLD THESE ON DISK!    │
   └────────────────────────────────────────┘          └───────────────────────────┘
```

* **True Population Distribution ($p_x$):** An unknown probability density function $p_x: \mathbb{R}^D \to \mathbb{R}^+$ satisfying $\int_{\mathbb{R}^D} p_x(x) \, dx = 1$.
* **Empirical Dataset ($\mathcal{D}$):** An unordered set of $n$ independent and identically distributed ($\text{IID}$) vectors $x_i \sim p_x$.
* **The Core Mathematical Paradox:** We want to optimize an objective $D(p_x \parallel p_\theta)$, but we cannot evaluate the function $p_x(x)$ at any arbitrary point $x \in \mathbb{R}^D$.

### Worked Micro-Examples

1. **Discrete Case (Rolling a Biased Die):**
   * *True Law $p_x$:* $P(X=6) = 0.5$, $P(X=k) = 0.1$ for $k \in \{1, 2, 3, 4, 5\}$.
   * *Dataset $\mathcal{D}$ (20 rolls):* $\{6, 6, 2, 6, 1, 6, 4, 6, 6, 3, 5, 6, 2, 6, 6, 1, 6, 3, 6, 4\}$.
   * The list of 20 integers is an empirical realization; it is not the mathematical probability mass function $p_x$.
2. **Continuous Case (MNIST Handwritten Digits):**
   * *True Law $p_x$:* The continuous manifold of all natural handwritten digits embedded in $\mathbb{R}^{784}$ ($28 \times 28$ pixels).
   * *Dataset $\mathcal{D}$:* A static array of $60{,}000$ specific image matrices stored in a binary tensor file.

### Real-World Pedagogical Analogies
* **The Master Baker's Secret Recipe:** The master baker's locked recipe book represents $p_x$. The 500 croissants sitting in the bakery display case represent $\mathcal{D}$. Eating every croissant in the case gives you rich sensory data (samples), but nobody hands you the recipe book ($p_x$).
* **Continental Climate vs Weather Logs:** The continent's climate system represents $p_x$. A binder containing 365 daily temperature logs represents $\mathcal{D}$. Reading the binder does not give you the analytical atmospheric differential equations ($p_x$).

### Python Code Illustration

```python
import numpy as np

# 1. Simulate the hidden world distribution p_x (Bimodal 1D Gaussian Mixture)
def sample_true_world_px(n=1000):
    mode = np.random.binomial(1, 0.35, size=n)
    samples = np.where(mode == 0,
                       np.random.normal(loc=-2.5, scale=0.6, size=n),
                       np.random.normal(loc=2.0, scale=0.9, size=n))
    return samples

# 2. We only hold the finite dataset D
dataset_D = sample_true_world_px(n=500)
print(f"Dataset D size: {len(dataset_D)} samples.")
print(f"Sample mean: {np.mean(dataset_D):.4f} | True density formula remains inaccessible!")
```

### Diagnostic Mini-Check
1. Is a `.png` image file $x_i$ the same mathematical object as the density function $p_x$? *(Answer: No, $x_i$ is a single sample vector; $p_x$ is the continuous density mapping $\mathbb{R}^D \to \mathbb{R}^+$).*
2. Why can we not evaluate $\int_{\mathbb{R}^D} p_x(x) \log p_x(x) \, dx$ directly on Day 1? *(Answer: Because we only have a discrete sample collection $\mathcal{D}$, not an analytical equation for $p_x$).*

---

## 2. The Dual Objective: Estimate $p_x$ AND Learn to Sample

<a id="p2-two-jobs"></a>

### Why It Matters for Lecture 4
Supervised learning models estimate conditional boundaries $p(y \mid x)$ to categorize existing data. Deep generative modeling is strictly defined by **two simultaneous, non-negotiable requirements**: estimating the distribution and constructing an efficient sampling engine.

### Formal System Breakdown

```
                             THE DUAL GOALS OF GENERATIVE AI
                                            │
        ┌───────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                       ▼
   JOB 1: ESTIMATE p_x                                                     JOB 2: LEARN TO SAMPLE
   • Align candidate model density p_θ with p_x                            • Provide an executable procedure
   • Explicit: closed-form log p_θ(x) density                                to synthesize brand-new points
   • Implicit: parameterized generator network G_θ                           x_new ~ p_θ* ≈ p_x (Novel creations!)
```

* **Job 1 (Distribution Estimation):** Optimize parameters $\theta$ such that candidate distribution $p_\theta$ approaches $p_x$ in probability distribution space.
* **Job 2 (Sample Generation):** Provide a fast, tractable sampling algorithm to draw novel data points $x_{\text{new}} \sim p_{\theta^\star}$ that are statistically indistinguishable from genuine data $p_x$, without duplicating entries from $\mathcal{D}$.

### Comparative Architecture Matrix

| Model Class | Primary Goal | Density Evaluation $p(x)$? | Novel Sample Generation? | Typical Failure Mode |
| :--- | :--- | :---: | :---: | :--- |
| **Discriminative** (ResNet, SVM) | Estimate $p(y \mid x)$ | **No** | **No** (Cannot draw $x$) | Adversarial Vulnerability |
| **Explicit Generative** (PixelCNN, Flows) | Maximize $\mathbb{E}[\log p_\theta(x)]$ | **Yes** (Exact density) | Yes (Often slow/autoregressive) | High computational cost |
| **Implicit Generative** (GANs, VDM) | Minimize $D_f(p_x \parallel p_\theta)$ | **No** (Density implicit) | **Yes** (Fast 1-step $G_\theta(z)$) | Mode Collapse / Instability |

### Real-World Pedagogical Analogies
* **Art Critic vs Master Painter:** An art critic (discriminative classifier) inspects a painting and outputs a label: *"Authentic Rembrandt ($y=1$)"* or *"Fake ($y=0$)"*. The critic cannot paint. A master painter (generative model) studies 100 authentic Rembrandts ($\mathcal{D}$) and paints a brand-new masterpiece ($x_{\text{new}}$) in Rembrandt's exact stylistic distribution ($p_x$).
* **Photocopier vs Novelist:** A photocopier memorizes and reproduces training text. A novelist learns the linguistic distribution ($p_x$) and authors an original novel.

### Diagnostic Mini-Check
1. If an algorithm calculates an accurate density value $p(x)$ for any test image but has no algorithm to draw new images, has it solved Job 2? *(Answer: No, it solved Job 1 but failed Job 2).*
2. Does an implicit generative model require calculating $p_\theta(x)$ to draw a sample? *(Answer: No, it maps random noise through a generator $G_\theta(z)$).*

---

## 3. Deep Neural Parametric Models $p_\theta$

<a id="p3-model"></a>

### Why It Matters for Lecture 4
We cannot perform an unconstrained search over the infinite-dimensional universe of all mathematical functions. We constrain our search to a **parametric family** $\mathcal{P} = \{p_\theta \mid \theta \in \Theta\}$ parameterized by the synaptic weight tensors $\theta$ of a deep neural network.

### Universal Approximation & Parameter Manifolds

```
   INFINITE FUNCTION SPACE                          PARAMETRIC NEURAL FAMILY {p_θ}
  ┌─────────────────────────────────┐              ┌───────────────────────────────┐
  │ All continuous probability      │              │ Candidate laws parameterized  │
  │ distributions on ℝᴰ             │ ──RESTRICT──►│ by deep net weights θ ∈ ℝᴹ    │
  │ (Infinite degrees of freedom)   │              │ (Universal Approximators)     │
  └─────────────────────────────────┘              └───────────────┬───────────────┘
                                                                   │
                                                            Gradient Updates on θ
                                                                   ▼
                                                      p_θ₁ ──► p_θ₂ ──► p_θ* ≈ p_x
```

* **The Parametric Family:** A structured collection of probability distributions indexed by a finite weight vector $\theta \in \mathbb{R}^M$.
* **Universal Approximation Theorem (Hornik et al., 1989):** A feedforward neural network with non-linear activation functions and sufficient hidden units can approximate any continuous Borel-measurable function to arbitrary precision $\epsilon > 0$.
* **The Course Terminology:** In this course, the term **"model"** refers specifically to the parametric candidate distribution $p_\theta$ represented by the neural network.

### Real-World Pedagogical Analogies
* **The Modular Audio Synthesizer:** A modular synthesizer has 10,000 knobs and patch cords ($\theta$). At configuration $\theta_1$, it outputs a violin wave; at $\theta_2$, a flute; at $\theta^\star$, an authentic orchestral string section ($p_x$). The synthesizer circuit is the neural architecture; the knob angles are $\theta$.
* **Sculptor's Steel Armature:** A steel armature with thousands of adjustable joints can be configured into any anatomical pose. The armature structure is the neural network; the joint angles are $\theta$.

### Diagnostic Mini-Check
1. What does the variable $\theta$ represent in $p_\theta$? *(Answer: The vector of all trainable synaptic weights and biases in the neural network).*
2. If we double the depth of network $G_\theta$, what happens to the parametric family $\{p_\theta\}$? *(Answer: It expands the expressiveness and representational capacity of the candidate distribution family).*

---

## 4. Statistical Divergences as Discrepancy Scores

<a id="p4-div"></a>

### Why It Matters for Lecture 4
Training a generative model requires a mathematical yardstick that evaluates the distance between two probability distributions. A **statistical divergence** $D(p \parallel q)$ quantifies the discrepancy between two probability density functions on a manifold.

### The Invariant Divergence Axioms

```
        True Distribution p_x            Model Distribution p_θ        Statistical Divergence D(p_x ‖ p_θ)
     ▲                                 ▲                             ▲
     │      ┌───┐                      │               ┌───┐         │   D(p_x ‖ p_θ) >> 0 (High loss!)
     │     ┌┘   └┐                     │              ┌┘   └┐        │
     └─────┴─────┴──────────► x        └──────────────┴─────┴──► x   └──────────────────────────────►
             Region A                                   Region B

                                       AFTER CONVERGENCE (θ = θ*):
     ▲
     │      ┌───┐   <── p_x and p_θ* OVERLAP PERFECTLY!
     │     ┌┘   └┐
     └─────┴─────┴──────────► x        ════════════════════════════►  D(p_x ‖ p_θ*) = 0 (Global Minimum!)
```

1. **Non-Negativity (Axiom 1):**
   $$D(p \parallel q) \ge 0 \quad \forall \, p, q$$
2. **Identity of Indiscernibles (Axiom 2):**
   $$D(p \parallel q) = 0 \iff p = q \quad (\text{almost everywhere})$$
3. **Asymmetry (Not a Metric):**
   In general, $D(p \parallel q) \neq D(q \parallel p)$, and statistical divergences do not satisfy the triangle inequality $D(p \parallel r) \le D(p \parallel q) + D(q \parallel r)$.

### Why These Axioms Enable Generative Learning
Because $D(p_x \parallel p_\theta) \ge 0$ with its absolute global minimum uniquely located at $0$, framing generative training as:
$$\theta^\star = \arg\min_\theta \, D(p_x \parallel p_\theta)$$
guarantees that driving the divergence loss to zero forces the generator's distribution $p_{\theta^\star}$ to match the ground-truth data distribution $p_x$!

### Real-World Pedagogical Analogies
* **Discrepancy Scale for Flour:** You place two bowls of flour ($p_x$ and $p_\theta$) on a sensor. The scale reads $\ge 0.00\text{ kg}$, and reads exactly $0.00\text{ kg}$ if and only if both bowls have identical density profiles. Adjusting recipe knobs $\theta$ until the scale reads $0.00$ guarantees the recipes match.
* **Laser Target Deviation Meter:** A target tracking system measures distance from the bullseye. The deviation is non-negative and zero only when the crosshair hits the exact center.

### Diagnostic Mini-Check
1. If a proposed loss function could evaluate to $-12.8$, why would $\arg\min_\theta D$ fail to align distributions? *(Answer: Minimizing the loss would drive it into negative infinity rather than stopping when distributions match at zero).*
2. Does a statistical divergence satisfy symmetry $D(p \parallel q) = D(q \parallel p)$? *(Answer: Generally no; most divergences like KL are asymmetric).*

---

## 5. The Pushforward Generative Engine: $Z \sim \mathcal{N}(0, I) \to G_\theta(Z)$

<a id="p5-push"></a>
<a id="p6-samples"></a>

### Why It Matters for Lecture 4
How does a neural network transform a vector of simple computer-generated random numbers into a photorealistic $480{,}000$-dimensional image? It uses the **pushforward measure** principle from probability theory.

### Mathematical Formulation & Pipeline Architecture

```
══════════════════════════════════════════════════════════════════════════════════════════════════
                            THE PUSHFORWARD SAMPLING PIPELINE
══════════════════════════════════════════════════════════════════════════════════════════════════

   Tractable Latent Prior               Deterministic Generator                   Generated Data Sample
        Z ∈ ℝᵏ                               G_θ : ℝᵏ ──► ℝᴰ                          X̂ ∈ ℝᴰ
  ┌──────────────────┐                     ┌──────────────────┐                 ┌──────────────────┐
  │ Z ~ N(0, I_k)    │ ──────────────────► │ Deep Neural Net  │ ──────────────► │ X̂ = G_θ(Z)       │
  │ Standard Normal  │    Forward Pass     │ (Synaptic θ)     │                 │ X̂ ~ p_θ(x̂)       │
  └──────────────────┘                     └──────────────────┘                 └──────────────────┘
    Trivial to sample!                        Deterministic                        Complex Natural
    (torch.randn)                              Transformation                       Data Distribution
══════════════════════════════════════════════════════════════════════════════════════════════════
```

1. **Latent Base Prior ($Z \sim p_z$):** We sample from an accessible low-dimensional Gaussian $Z \sim \mathcal{N}(0, I_k)$ in latent space $\mathbb{R}^k$ (where $k \ll D$).
2. **Deterministic Neural Transformation ($G_\theta$):** A deep neural network $G_\theta: \mathbb{R}^k \to \mathbb{R}^D$ parameterized by weights $\theta$.
3. **Induced Output Random Variable ($\hat{X}$):** A deterministic function of a random variable is itself a random variable:
   $$\hat{X} = G_\theta(Z) \in \mathbb{R}^D$$
   The resulting probability distribution of $\hat{X}$ is denoted $p_\theta$.
4. **Pushforward Measure Definition:** For any Borel-measurable set $A \subseteq \mathbb{R}^D$:
   $$p_\theta(A) = P(\hat{X} \in A) = P(G_\theta(Z) \in A) = P(Z \in G_\theta^{-1}(A)) = \int_{G_\theta^{-1}(A)} p_z(z) \, dz$$
5. **The Critical Insight:** Running a forward pass through $G_\theta$ yields **concrete samples from $p_\theta$**, but does not provide an analytical formula for $p_\theta(x)$!

### Worked Micro-Example (1D Analytical Pushforward)

* Let latent variable $Z \sim \text{Uniform}[0, 1]$ (density $p_z(z) = 1$ for $z \in [0, 1]$).
* Let deterministic generator $g(z) = -\frac{1}{\lambda} \ln(1 - z)$ for $\lambda > 0$.
* The output random variable is $\hat{X} = g(Z)$.
* Using the transformation rule for monotonic functions ($p_{\hat{X}}(x) = p_z(g^{-1}(x)) \left|\frac{d}{dx} g^{-1}(x)\right|$):
  $$z = g^{-1}(x) = 1 - e^{-\lambda x} \implies \frac{dz}{dx} = \lambda e^{-\lambda x}$$
  $$p_{\hat{X}}(x) = 1 \cdot \lambda e^{-\lambda x} = \lambda e^{-\lambda x} \quad (x \ge 0)$$
* Pushing uniform noise through $g(z)$ generated an **Exponential Distribution $\text{Exp}(\lambda)$**!

### Real-World Pedagogical Analogies
* **The Industrial Pasta Extruder:** Standardized wheat flour dough ($Z \sim \mathcal{N}(0, I)$) is pushed through an interchangeable steel die ($G_\theta$). If the die is configured for penne, penne noodles emerge; if configured for fusilli, fusilli noodles emerge. You never write a mathematical density equation for pasta; you design the die.
* **Refractive Optical Prism:** Uniform white sunlight ($Z$) enters a finely ground glass prism ($G_\theta$). The deterministic refractive geometry bends the light rays into a rich multi-colored spectrum ($\hat{X}$).

### Python Code Illustration

```python
import torch
import torch.nn as nn

# Define pushforward generator (Latent k=8 -> Data D=2)
class ToyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, z):
        return self.net(z)

generator = ToyGenerator()
z_noise = torch.randn(100, 8)       # 100 samples from N(0, I)
x_fake = generator(z_noise)         # 100 samples from p_theta!
print(f"Generated sample cloud shape: {x_fake.shape}")
```

### Diagnostic Mini-Check
1. If we feed the exact same noise vector $z_0$ into $G_\theta$ twenty times, do we get 20 different images? *(Answer: No, $G_\theta$ is deterministic; identical input produces identical output).*
2. Where does the diversity of generated samples come from? *(Answer: From sampling different latent vectors $z \sim \mathcal{N}(0, I)$).*

---

## 6. Deep Dive: Convex Analysis & Jensen's Bound

<a id="p6-convex"></a>

### Why It Matters for Lecture 4
The entire $f$-divergence framework relies on a generator function $f(u)$ that is **convex on $\mathbb{R}^+$ with $f(1) = 0$**. Understanding convex functions, secant lines, supporting hyperplanes, and Jensen's inequality provides the rigorous mathematical proof of why $D_f(p_x \parallel p_\theta) \ge 0$.

### Mathematical Definitions of Convexity

```
                             GEOMETRY OF A CONVEX FUNCTION
       f(u)
        ▲
        │                                  / (Secant line lies ABOVE curve!)
        │        (u₁, f(u₁))  *───────────* (u₂, f(u₂))
        │                     │  \     /  │
        │                     │   \___/   │
        │                     │     ▲     │
        │                     │     │ Curve f(u) lies strictly BELOW secant!
        0 ────────────────────┴─────┴─────┴────────► u
                             u₁     u*    u₂
```

1. **Secant Inequality Definition:**  
   A function $f: \mathcal{C} \subseteq \mathbb{R} \to \mathbb{R}$ is convex if for all $u_1, u_2 \in \mathcal{C}$ and all $\lambda \in [0, 1]$:
   $$f(\lambda u_1 + (1-\lambda) u_2) \le \lambda f(u_1) + (1-\lambda) f(u_2)$$
   *Strict convexity* holds if the inequality is strict ($<$) for all $u_1 \neq u_2$ and $\lambda \in (0, 1)$.
2. **First-Order Condition (Supporting Hyperplane):**  
   If $f$ is continuously differentiable, $f$ is convex if and only if the tangent line at any point $u_0$ lies entirely below the function graph:
   $$f(u) \ge f(u_0) + f'(u_0)(u - u_0) \quad \forall \, u \in \mathcal{C}$$
3. **Second-Order Condition (Curvature Test):**  
   If $f$ is twice differentiable on an open interval, $f$ is convex if and only if its second derivative is non-negative everywhere:
   $$f''(u) \ge 0 \quad \forall \, u > 0$$

```
   SUPPORTING HYPERPLANE / TANGENT PROPERTY:
   f(u)
    ▲                                    / f(u) curve
    │                     *             /
    │                    / \           /
    │                   /   *─────────* 
    │                  /   /  Tangency point (u₀, f(u₀))
    │                 /  /
    │  Tangent Line: / / y = f(u₀) + f'(u₀)(u - u₀)  <── Lies ENTIRELY BELOW f(u)!
    └───────────────┴─┴──────────────────────────────► u
                     u₀
```

### Complete Mathematical Proof of Jensen's Inequality

**Theorem (Jensen's Inequality):**  
Let $f: \mathbb{R} \to \mathbb{R}$ be a convex function, and let $U$ be an integrable random variable. Then:
$$\mathbb{E}[f(U)] \ge f(\mathbb{E}[U])$$

**Proof via Supporting Hyperplanes:**
1. Let $\mu = \mathbb{E}[U]$ be the expected value of $U$.
2. Since $f$ is convex, there exists a supporting hyperplane (subgradient slope $c$) at $\mu$ such that for all possible real values $u$:
   $$f(u) \ge f(\mu) + c(u - \mu)$$
3. Since this inequality holds for every realization $u$ of the random variable $U$, it holds for the random variable $U$ itself:
   $$f(U) \ge f(\mu) + c(U - \mu)$$
4. Take the mathematical expectation $\mathbb{E}[\cdot]$ on both sides:
   $$\mathbb{E}[f(U)] \ge \mathbb{E}[f(\mu) + c(U - \mu)]$$
5. Using the linearity of the expectation operator:
   $$\mathbb{E}[f(U)] \ge f(\mu) + c\left(\mathbb{E}[U] - \mu\right)$$
6. Since $\mu = \mathbb{E}[U]$, the term $(\mathbb{E}[U] - \mu) = 0$:
   $$\mathbb{E}[f(U)] \ge f(\mu) + c(0) = f(\mathbb{E}[U])$$
7. Substituting $\mu = \mathbb{E}[U]$ completes the proof:
   $$\mathbb{E}[f(U)] \ge f(\mathbb{E}[U]) \quad \blacksquare$$

### Why $f(1) = 0$ Guarantees $D_f(p_x \parallel p_\theta) \ge 0$

Let $U(x) = \frac{p_x(x)}{p_\theta(x)}$ be the density ratio evaluated at random variable $X \sim p_\theta$.  
1. Compute the expectation of the density ratio under the model distribution $p_\theta$:
   $$\mathbb{E}_{X \sim p_\theta}[U(X)] = \int_{\mathcal{X}} p_\theta(x) \left(\frac{p_x(x)}{p_\theta(x)}\right) dx = \int_{\mathcal{X}} p_x(x) \, dx = 1$$
2. Apply Jensen's inequality to the expectation of $f(U)$:
   $$D_f(p_x \parallel p_\theta) = \mathbb{E}_{X \sim p_\theta}[f(U(X))] \ge f\left(\mathbb{E}_{X \sim p_\theta}[U(X)]\right) = f(1)$$
3. Since the generator function satisfies $f(1) = 0$:
   $$D_f(p_x \parallel p_\theta) \ge 0 \quad \forall \, p_x, p_\theta$$

```
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                    THE JENSEN NON-NEGATIVITY GUARANTEE                    │
   │                                                                           │
   │   D_f(p_x ‖ p_θ) = E_{X ~ p_θ}[ f( p_x(X) / p_θ(X) ) ]                    │
   │                  ≥ f( E_{X ~ p_θ}[ p_x(X) / p_θ(X) ] )  (Jensen's Bound)  │
   │                  = f( ∫ p_x(x) dx )                     (Ratio cancels)   │
   │                  = f( 1 )                               (Probability sum) │
   │                  = 0                                    (Axiom f(1) = 0)  │
   │                                                                           │
   │   ⟹ D_f(p_x ‖ p_θ) ≥ 0 with equality if and only if p_x = p_θ!          │
   └───────────────────────────────────────────────────────────────────────────┘
```

### What is Left Semi-Continuity?
A function $f: \mathbb{R}^+ \to \mathbb{R}$ is **left semi-continuous** at $u_0$ if:
$$\liminf_{u \to u_0^-} f(u) \ge f(u_0)$$
Geometrically, as $u$ approaches $u_0$ from the left, the function values do not suddenly jump upwards. This technical property guarantees that convex conjugates $f^*(t) = \sup_{u > 0} \{ut - f(u)\}$ and integrals over $f$ remain well-defined at the boundary $u \to 0^+$.

### Diagnostic Mini-Check
1. If $f(u) = u^2$, why is it not a valid $f$-divergence generator as written? *(Answer: Because $f(1) = 1^2 = 1 \neq 0$; we must shift it to $f(u) = u^2 - 1$ or $(u-1)^2$).*
2. What allows $p_\theta(x)$ to cancel out inside $\mathbb{E}_{p_\theta}[p_x/p_\theta]$? *(Answer: The integral $\int p_\theta(x) \frac{p_x(x)}{p_\theta(x)} dx = \int p_x(x) dx = 1$).*

---

## 7. The Unified Csiszár-Ali-Silvey $f$-Divergence

<a id="p7-f"></a>

### Why It Matters for Lecture 4
Introduced independently by Imre Csiszár (1967) and S. M. Ali & S. D. Silvey (1966), $f$-divergence provides a master mathematical framework that unifies nearly all statistical distances used in information theory, probability theory, and machine learning into a single integral.

### Master Integral Definition

Let $p_x$ and $p_\theta$ be continuous probability density functions supported on domain $\mathcal{X} \subseteq \mathbb{R}^D$. The **$f$-divergence** between $p_x$ and $p_\theta$ is defined as:

$$D_f(p_x \parallel p_\theta) = \int_{\mathcal{X}} p_\theta(x) \, f\left(\frac{p_x(x)}{p_\theta(x)}\right) dx$$

where $f: \mathbb{R}^+ \to \mathbb{R}$ satisfies:
1. **$f$ is convex on $(0, \infty)$**
2. **$f$ is left semi-continuous**
3. **$f(1) = 0$**

```
  THE ANATOMY OF THE f-DIVERGENCE INTEGRAL
  ─────────────────────────────────────────────────────────────────────────────
                ┌── Outer expectation weight: p_θ(x)
                │
   D_f = ∫_X  p_θ(x) · f( p_x(x) / p_θ(x) )  dx
                          └────────┬────────┘
                                   └── Density ratio u(x) is a 1D SCALAR in ℝ⁺!
                                       f evaluates single positive real numbers.
```

### Why the Density Ratio $u(x)$ is a 1D Scalar

* Although data vector $x \in \mathbb{R}^D$ resides in high dimensions (e.g., $D = 784$ for MNIST or $D = 480{,}000$ for high-resolution color images), evaluating probability densities at $x$ yields two scalar heights: $p_x(x) \in \mathbb{R}^+$ and $p_\theta(x) \in \mathbb{R}^+$.
* Therefore, the density ratio:
  $$u(x) = \frac{p_x(x)}{p_\theta(x)}$$
  is a **single positive real scalar**.
* $f(u)$ simply evaluates this 1D scalar ratio. The function $f$ never acts directly on the high-dimensional vector $x$.

### Expectation Representation
The integral can be expressed concisely as an expectation under the model distribution:

$$D_f(p_x \parallel p_\theta) = \mathbb{E}_{x \sim p_\theta}\left[ f\left(\frac{p_x(x)}{p_\theta(x)}\right) \right]$$

### Real-World Pedagogical Analogy
* **City Building Height Discrepancy:** You walk across a city with $D$ coordinates. At every address $x$, there are two towers: Tower A (height $p_x(x)$) and Tower B (height $p_\theta(x)$). You compute their height ratio $u = p_x(x)/p_\theta(x)$. A spring $f(u)$ measures the tension of their mismatch. You multiply by Tower B's base area $p_\theta(x) dx$ and integrate across the entire city to determine the municipal mismatch penalty ($D_f$).

### Diagnostic Mini-Check
1. If $x \in \mathbb{R}^{1000000}$, what is the dimensionality of the input to $f$? *(Answer: Dimension 1 (a scalar), because the ratio of two density values is a single real number).*
2. If $p_x(x) = p_\theta(x)$ at all $x$, what is $u(x)$ and what is $f(u(x))$? *(Answer: $u(x) = 1$, and $f(1) = 0$, so $D_f = 0$).*

---

## 8. The Exhaustive $f$-Divergence Catalog, Scenarios & Mode Dynamics

<a id="p8-kl"></a>

### Why It Matters for Lecture 4
Different choices of generator function $f(u)$ produce different statistical divergences, each with distinct geometric curvature, gradient behaviors, and mode-fitting characteristics in machine learning.

```
══════════════════════════════════════════════════════════════════════════════════════════════════
                               THE MASTER f-DIVERGENCE CATALOG
══════════════════════════════════════════════════════════════════════════════════════════════════
 Divergence Name            Generator Function f(u)                        Integral Formula D_f(p_x ‖ p_θ)
 ────────────────────────────────────────────────────────────────────────────────────────────────
 1. Forward KL              f(u) = u log(u)                                ∫ p_x(x) log( p_x(x) / p_θ(x) ) dx
 2. Reverse KL              f(u) = -log(u)                                 ∫ p_θ(x) log( p_θ(x) / p_x(x) ) dx
 3. Jensen-Shannon (JS)     f(u) = ½ [ u log u - (u+1) log((u+1)/2) ]      ½ D_KL(p_x ‖ M) + ½ D_KL(p_θ ‖ M)
 4. Total Variation (TV)    f(u) = ½ |u - 1|                               ½ ∫ |p_x(x) - p_θ(x)| dx
 5. Pearson χ²              f(u) = (u - 1)²  or  u² - 1                    ∫ (p_x(x) - p_θ(x))² / p_θ(x) dx
 6. Neyman χ²               f(u) = (1 - u)² / u                            ∫ (p_x(x) - p_θ(x))² / p_x(x) dx
 7. Squared Hellinger       f(u) = (√u - 1)²                               ½ ∫ (√p_x(x) - √p_θ(x))² dx
 8. Alpha-Divergence (α≠0,1)f(u) = (u^α - αu + (α - 1)) / (α(α - 1))       1/(α(α-1)) [ ∫ p_x^α p_θ^(1-α) dx - 1 ]
══════════════════════════════════════════════════════════════════════════════════════════════════
```

---

### Detailed Mathematical Derivations

#### 1. Forward KL Divergence Derivation ($f(u) = u \log u$)
1. Substitute $f(u) = u \log u$ into the $f$-divergence integral:
   $$D_f(p_x \parallel p_\theta) = \int_{\mathcal{X}} p_\theta(x) \left[ \left(\frac{p_x(x)}{p_\theta(x)}\right) \log\left(\frac{p_x(x)}{p_\theta(x)}\right) \right] dx$$
2. The outer $p_\theta(x)$ cancels with the denominator of the ratio:
   $$D_f(p_x \parallel p_\theta) = \int_{\mathcal{X}} p_x(x) \log\left(\frac{p_x(x)}{p_\theta(x)}\right) dx = D_{\text{KL}}(p_x \parallel p_\theta)$$

#### 2. Reverse KL Divergence Derivation ($f(u) = -\log u$)
1. Substitute $f(u) = -\log u$ into the $f$-divergence integral:
   $$D_f(p_x \parallel p_\theta) = \int_{\mathcal{X}} p_\theta(x) \left[ -\log\left(\frac{p_x(x)}{p_\theta(x)}\right) \right] dx$$
2. Using logarithm inversion $-\log(a/b) = \log(b/a)$:
   $$D_f(p_x \parallel p_\theta) = \int_{\mathcal{X}} p_\theta(x) \log\left(\frac{p_\theta(x)}{p_x(x)}\right) dx = D_{\text{KL}}(p_\theta \parallel p_x)$$

#### 3. Pearson $\chi^2$ Divergence Derivation ($f(u) = (u - 1)^2$)
1. Substitute $f(u) = (u - 1)^2$ into the $f$-divergence integral:
   $$D_f(p_x \parallel p_\theta) = \int_{\mathcal{X}} p_\theta(x) \left( \frac{p_x(x)}{p_\theta(x)} - 1 \right)^2 dx = \int_{\mathcal{X}} p_\theta(x) \left( \frac{p_x(x) - p_\theta(x)}{p_\theta(x)} \right)^2 dx$$
2. Simplifying the algebra:
   $$D_f(p_x \parallel p_\theta) = \int_{\mathcal{X}} \frac{(p_x(x) - p_\theta(x))^2}{p_\theta(x)} \, dx = \chi^2(p_x \parallel p_\theta)$$

---

### Deep Dive: Forward KL vs Reverse KL Mode Dynamics

```
   GROUND TRUTH DISTRIBUTION p_x (Bimodal with two distinct peaks):
   ▲
   │        ┌───┐             ┌───┐
   │       ┌┘   └┐           ┌┘   └┐
   └───────┴─────┴───────────┴─────┴──────► x
          Mode 1            Mode 2

   FORWARD KL FIT (Mode-Covering / Zero-Avoiding):
   ▲
   │        ┌─────────────────────┐      <── Spreads probability mass across BOTH modes!
   │       ┌┘                     └┐         (Places fake mass in the middle valley -> Blurry images)
   └───────┴───────────────────────┴──────► x

   REVERSE KL FIT (Mode-Seeking / Zero-Forcing):
   ▲
   │        ┌───┐
   │       ┌┘   └┐                       <── Locks tightly onto Mode 1!
   └───────┴─────┴────────────────────────► x (Completely drops Mode 2 -> Mode collapse)
```

* **Forward KL ($D_{\text{KL}}(p_x \parallel p_\theta) = \mathbb{E}_{x \sim p_x}[\log \frac{p_x(x)}{p_\theta(x)}]$):**
  * **Zero-Avoiding Property:** If $p_x(x) > 0$ and $p_\theta(x) \to 0$, the ratio $\frac{p_x(x)}{p_\theta(x)} \to \infty$, causing the loss to explode to $+\infty$.
  * **Behavior:** The model refuses to allow $p_\theta(x) = 0$ anywhere real data exists. It spreads its mass to cover all modes, generating blurry intermediate samples.
* **Reverse KL ($D_{\text{KL}}(p_\theta \parallel p_x) = \mathbb{E}_{x \sim p_\theta}[\log \frac{p_\theta(x)}{p_x(x)}]$):**
  * **Zero-Forcing Property:** If $p_x(x) = 0$ and $p_\theta(x) > 0$, the penalty explodes to $+\infty$.
  * **Behavior:** The model refuses to place generated mass anywhere real data is missing. It tightly encapsulates a single mode, ignoring other modes.

---

### Practical Engineering Scenarios: When to Use Which $f$-Divergence

```
══════════════════════════════════════════════════════════════════════════════════════════════════
                               PRACTICAL SCENARIO SELECTION MATRIX
══════════════════════════════════════════════════════════════════════════════════════════════════
 Scenario / Domain                   Recommended Divergence    Technical Rationale
 ────────────────────────────────────────────────────────────────────────────────────────────────
 Maximum Likelihood / VAE Decoder    Forward KL                Guarantees all data modes are covered;
                                                               avoids omitting rare training samples.
 Variational Inference (ELBO)        Reverse KL                Tractable expectation under candidate q_θ;
                                                               locks onto tight posterior approximations.
 Standard GANs (Goodfellow 2014)     Jensen-Shannon (JS)       Symmetric & bounded in [0, log 2]; discriminator
                                                               acts as smooth density ratio estimator.
 Least Squares GANs (LSGAN)          Pearson χ²                Linear penalty on large errors; prevents vanishing
                                                               gradients on samples far from real boundary.
 Distribution Drift & Data Quality   Total Variation (TV)      Intuitive interpretation: upper bounds the maximum
                                                               probability shift on any arbitrary event.
 Robust Anomaly Detection            Squared Hellinger         Bounded metric in [0, 1]; symmetric and heavily
                                                               resistant to extreme dataset outliers.
 Continuous Tunable Balancing        Alpha-Divergence          Varying α smoothly interpolates between mode-
                                                               covering (α=1) and mode-seeking (α=0).
══════════════════════════════════════════════════════════════════════════════════════════════════
```

---

### Step-by-Step Numerical Calculations on Discrete Distributions

Let True Distribution $P = [0.8, 0.2]$ and Model Distribution $Q = [0.4, 0.6]$.

1. **Forward KL:**
   $$D_{\text{KL}}(P \parallel Q) = 0.8 \ln\left(\frac{0.8}{0.4}\right) + 0.2 \ln\left(\frac{0.2}{0.6}\right) = 0.8 \ln(2) + 0.2 \ln(1/3) = 0.8(0.6931) + 0.2(-1.0986) = 0.5545 - 0.2197 = \mathbf{0.3348}$$
2. **Reverse KL:**
   $$D_{\text{KL}}(Q \parallel P) = 0.4 \ln\left(\frac{0.4}{0.8}\right) + 0.6 \ln\left(\frac{0.6}{0.2}\right) = 0.4 \ln(0.5) + 0.6 \ln(3) = 0.4(-0.6931) + 0.6(1.0986) = -0.2772 + 0.6592 = \mathbf{0.3820}$$
   *(Note: $D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$, numerically proving asymmetry!)*
3. **Total Variation (TV):**
   $$D_{\text{TV}}(P \parallel Q) = \frac{1}{2}\left( |0.8 - 0.4| + |0.2 - 0.6| \right) = \frac{1}{2}(0.4 + 0.4) = \mathbf{0.4000}$$
4. **Pearson $\chi^2$:**
   $$\chi^2(P \parallel Q) = \frac{(0.8 - 0.4)^2}{0.4} + \frac{(0.2 - 0.6)^2}{0.6} = \frac{0.16}{0.4} + \frac{0.16}{0.6} = 0.4000 + 0.2667 = \mathbf{0.6667}$$

---

### Python Verification Script for All $f$-Divergences

```python
import numpy as np

# Define discrete distributions
P = np.array([0.8, 0.2])
Q = np.array([0.4, 0.6])

# Generator functions f(u)
f_dict = {
    "Forward KL": lambda u: u * np.log(u),
    "Reverse KL": lambda u: -np.log(u),
    "Jensen-Shannon": lambda u: 0.5 * (u * np.log(u) - (u + 1) * np.log((u + 1) / 2.0)),
    "Total Variation": lambda u: 0.5 * np.abs(u - 1.0),
    "Pearson Chi-Sq": lambda u: (u - 1.0)**2,
    "Squared Hellinger": lambda u: (np.sqrt(u) - 1.0)**2
}

print("--- Numerical Verification of f-Divergence Values ---")
for name, f_gen in f_dict.items():
    ratio = P / Q
    div_val = np.sum(Q * f_gen(ratio))
    print(f"{name:20s}: {div_val:.4f}")
```

---

## Prerequisite Mastery Matrix

Check off each item before starting [NOTES.md](./NOTES.md):

- [ ] I understand why population density $p_x$ is never available in closed form.
- [ ] I can articulate the dual generative requirements: (1) estimate $p_x$ and (2) learn to sample novel points.
- [ ] I understand that neural network weights $\theta$ define the candidate parametric family $\{p_\theta\}$.
- [ ] I know that divergences satisfy $D(p \parallel q) \ge 0$ and $D(p \parallel q) = 0 \iff p = q$.
- [ ] I can explain the pushforward sampling mechanism $Z \sim \mathcal{N}(0, I) \to G_\theta(Z) \sim p_\theta$.
- [ ] I can write the second-derivative test $f''(u) \ge 0$ and state the supporting hyperplane definition of convexity.
- [ ] I can walk through the algebraic proof of Jensen's inequality $\mathbb{E}[f(U)] \ge f(\mathbb{E}[U])$ and explain why $f(1)=0$ proves $D_f \ge 0$.
- [ ] I can derive Forward KL from $f(u) = u \log u$ by canceling $p_\theta(x)$.
- [ ] I understand the behavioral difference between mode-covering (Forward KL) and mode-seeking (Reverse KL).
- [ ] I know the chalkboard generator for Jensen-Shannon divergence and its historical connection to GANs.

---

**You are fully prepared! Proceed to [NOTES.md](./NOTES.md).**
