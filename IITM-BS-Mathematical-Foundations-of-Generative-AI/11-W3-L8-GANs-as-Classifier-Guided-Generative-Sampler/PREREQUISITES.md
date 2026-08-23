# Warm-up before the lecture (PREREQUISITES.md)

> **Do this first.** Then open [NOTES.md](./NOTES.md) at the **Executive Summary** architecture blueprint.  
> This warm-up builds the core intuitive and mathematical machinery needed to understand how a binary classifier can guide a generative neural network.

```
  After this warm-up you can say in plain words:

  "A continuous probability distribution is a density landscape over feature space."
  "A generator is a differentiable function that bends simple Gaussian noise into complex data shapes."
  "A binary classifier carves feature space into acceptance and rejection regions using a decision boundary."
  "Binary cross-entropy rewards high confidence on real data and penalizes false acceptance of fakes."
  "Variational bounds replace impossible density integrals with solvable neural-network optimization problems."
  "A minimax game pits two networks with conflicting goals against each other over a single shared score."
  "An optimal discriminator computes the exact density ratio between real and generated distributions."
```

---

## 1. Probability Distributions as Density Landscapes in Space

<a id="p1-distributions"></a>

In machine learning, we treat data points (like images, audio frames, or patient measurements) as vectors living in a continuous feature space $\mathbb{R}^d$. A continuous probability distribution $p_x(x)$ is not an abstract formula—it is a density landscape over that space.

Where $p_x(x)$ is high, data points cluster densely. Where $p_x(x) \approx 0$, the space is empty.

```
  x₂ (Feature 2)
   ▲
   │        · · ·  (Dense real data cluster)
   │      · · · · ·
   │        · · ·       px(x) is high here!
   │
   │
   │   (Empty space: px(x) ≈ 0)
   │
   └───────────────────────────► x₁ (Feature 1)
```

**Concrete example:** Suppose $x = (x_1, x_2)^T \in \mathbb{R}^2$ represents the height and weight of adult humans. Real points cluster tightly in a specific diagonal band (e.g., around $(175\text{ cm}, 70\text{ kg})$). If a point appears at $(300\text{ cm}, 10\text{ kg})$, the true distribution density $p_x$ at that coordinate is effectively zero.

**Analogy:** Think of $p_x$ like a population heat map of a continent. High peaks mark dense cities; vast flat valleys mark uninhabited deserts. Sampling from $p_x$ is like picking a random resident: you almost always land inside a city, never in the desert.

---

## 2. The Generative Sampler (Transforming Noise into Structure)

<a id="p2-generative-sampler"></a>

We want an algorithm that can generate brand-new, realistic data points that look like they came from $p_x$, without just copying the training dataset.

Since computers can easily generate simple random numbers (such as standard Gaussian noise $z \sim \mathcal{N}(0, I)$), a **generative sampler** is a neural network $G_\theta: \mathbb{R}^k \to \mathbb{R}^d$ parameterized by weights $\theta$. It takes a simple noise vector $z$ and bends, stretches, and translates it into a synthetic data vector $\hat{x} = G_\theta(z)$.

```
   Source Noise Space (Z)            Generator Network G_θ            Data Space (X)
      z ~ N(0, I)                                                   x̂ = G_θ(z) ~ p_θ
   ┌─────────────────┐               ┌─────────────────┐           ┌─────────────────┐
   │     ·  ·  ·     │   ────────►   │ Neural Layers   │  ──────►  │      · · ·      │
   │   ·   ·   ·   · │               │ W₁, b₁, ReLU,   │           │    · · · · ·    │
   │     ·  ·  ·     │               │ W₂, b₂, ...     │           │      · · ·      │
   └─────────────────┘               └─────────────────┘           └─────────────────┘
    Spherical Gaussian                Non-linear Mapping            Generated Cluster
```

**Concrete example:** If $z \in \mathbb{R}$ is drawn uniformly from $[-1, 1]$, and $G_\theta(z) = 3z + 5$, the output $\hat{x}$ is uniformly distributed on $[2, 8]$. Deep neural networks perform high-dimensional non-linear warping so that spherical noise is molded into photorealistic images.

**Analogy:** Imagine a master pastry chef. The chef starts with a uniform, featureless sphere of dough (Gaussian noise $z$). By stretching, folding, and pressing the dough (the neural network layers $G_\theta$), they produce an intricately shaped croissant (the generated sample $\hat{x}$).

---

## 3. Binary Classification and Decision Boundaries

<a id="p3-classification-boundary"></a>

A binary classifier $D_w: \mathbb{R}^d \to [0, 1]$ is a function parameterized by weights $w$ that takes any point $x$ in feature space and outputs a single scalar between $0$ and $1$.

- $D_w(x) \approx 1$ means the classifier is confident the point belongs to **Class 1** (Real Data).
- $D_w(x) \approx 0$ means the classifier is confident the point belongs to **Class 0** (Fake / Generated Data).

The set of all points where $D_w(x) = 0.5$ forms the **decision boundary** (a hyperplane for linear models, or a curved hypersurface for deep neural networks).

```
  x₂ ▲
     │   CLASS 1 (Real Data: x ~ px)
     │       +   +   +
     │         +   +
     │ ═══════════════════════════════  Decision Boundary: Dw(x) = 0.5
     │         ·   ·
     │       ·   ·   ·
     │   CLASS 0 (Fake Data: x̂ ~ p_θ)
     └────────────────────────────────► x₁
```

**Concrete example:** In $\mathbb{R}^2$, a linear classifier computes $D_w(x) = \sigma(w_1 x_1 + w_2 x_2 + b)$, where $\sigma(u) = 1 / (1 + e^{-u})$ is the logistic sigmoid. If $w = (1, 1)$ and $b = -5$, any point with $x_1 + x_2 > 5$ gets a score $> 0.5$, while any point with $x_1 + x_2 < 5$ gets a score $< 0.5$.

**Analogy:** Think of a security guard at an exclusive venue. The guard examines everyone arriving at the door. If you hold a genuine VIP pass (real data), they let you in ($1$). If you hold a crude photocopy (fake data), they turn you away ($0$).

---

## 4. Binary Cross-Entropy and Log-Likelihood Objectives

<a id="p4-cross-entropy"></a>

To train a binary classifier, we use the principle of **Maximum Likelihood Estimation (MLE)**. We want the classifier to give high probability to real events and low probability to fake events.

For a real sample $x \sim p_x$, the probability assigned to the correct class is $D_w(x)$. The log-likelihood is $\log D_w(x)$.
For a fake sample $\hat{x} \sim p_\theta$, the probability assigned to the correct class (Class 0) is $1 - D_w(\hat{x})$. The log-likelihood is $\log(1 - D_w(\hat{x}))$.

```
  Discriminator Output Dw(x)     log Dw(x) [Real Term]     log(1 - Dw(x̂)) [Fake Term]
  ───────────────────────────────────────────────────────────────────────────────────
  0.99 (Strong Real)              -0.010 (Near zero loss)   -4.605 (Massive penalty)
  0.90 (Confident Real)           -0.105 (Small loss)       -2.302 (Heavy penalty)
  0.50 (Complete Uncertainty)     -0.693 (Moderate loss)    -0.693 (Moderate loss)
  0.10 (Confident Fake)           -2.302 (Heavy penalty)    -0.105 (Small loss)
  0.01 (Strong Fake)              -4.605 (Massive penalty)  -0.010 (Near zero loss)
```

Notice how the logarithm heavily penalizes mistakes: assigning $D_w(x) = 0.01$ to a real image incurs a massive loss penalty of $-4.605$, driving aggressive gradient updates to correct the mistake.

---

## 5. Variational Bounds and Fenchel Conjugate Duality (VDM Recap)

<a id="p5-vdm-duality"></a>

In generative modeling, we want to measure the statistical distance (an $f$-divergence $D_f(p_x \parallel p_\theta)$) between the true data distribution $p_x$ and the model distribution $p_\theta$. However, we cannot compute the integral $\int p_\theta(x) f(p_x(x) / p_\theta(x)) dx$ directly because we do not know the mathematical formula for $p_x(x)$.

**Variational Divergence Minimization (VDM)** solves this by rewriting the divergence through convex duality (the Fenchel conjugate $f^*(t) = \sup_u \{ut - f(u)\}$):

$$D_f(p_x \parallel p_\theta) \ge \mathbb{E}_{x \sim p_x}[T(x)] - \mathbb{E}_{x \sim p_\theta}[f^*(T(x))]$$

Here, $T(x)$ is a variational "witness" function (parameterized by a neural network). Instead of computing an impossible integral, we optimize $T(x)$ over available sample batches.

```
  Divergence Value
     ▲
     │              True Divergence D_f(px ║ p_θ)   [Uncomputable directly]
     │          ════════════════════════════════════
     │                     ▲
     │                     │  Maximizing over T pushes the lower bound up!
     │                     │
     │          ─ ─ ─ ─ ─ ─┴─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
     │          Variational Lower Bound J(θ, w)     [Computable via samples]
     └─────────────────────────────────────────────► Function Class of T
```

**Analogy:** Imagine trying to measure the exact height of an uneven ceiling in a dark room. You cannot see the whole ceiling at once. Instead, you raise an adjustable pole from the floor until it touches the highest possible point. The top of the pole gives you a rigorous lower bound that equals the true ceiling height when perfectly positioned.

---

## 6. Minimax Zero-Sum Games and Saddle Points

<a id="p6-minimax-games"></a>

A **minimax zero-sum game** is an optimization problem involving two competing agents and a single scalar value function $J(\theta, w)$:

$$\min_\theta \max_w J(\theta, w)$$

- **Player 1 (Discriminator $w$):** Acts as the maximizer, adjusting $w$ to increase $J$ as high as possible.
- **Player 2 (Generator $\theta$):** Acts as the minimizer, adjusting $\theta$ to decrease $J$ as low as possible.

The solution is not a standard minimum or maximum, but a **saddle point** $(\theta^*, w^*)$ where neither player can unilaterally improve their payoff.

```
       Payoff Surface J(θ, w)
             ▲
             │       /═════\   (Maximized along w axis)
             │      /       \
             │     │    *    │  <--- SADDLE POINT (θ*, w*)
             │      \       /
             │       \═════/   (Minimized along θ axis)
             └───────────────────────► Parameters (θ, w)
```

**Analogy:** Consider a game between a lock designer (Generator) and a lock picker (Discriminator). The lock picker studies every lock to find vulnerabilities (maximizing detection). The lock designer modifies the lock mechanisms specifically to defeat the picker's latest tools (minimizing detection). At equilibrium, the lock is completely unpickable.

---

## 7. The Density Ratio and Jensen-Shannon Divergence

<a id="p7-density-ratio-jsd"></a>

Why does a binary classifier connect to probability divergence? Consider an optimal classifier $D^*(x)$ trained to distinguish real data $p_x(x)$ from generated data $p_\theta(x)$ with equal prior probabilities ($1/2$ each). By Bayes' theorem, the exact optimal posterior probability is:

$$D^*(x) = \frac{p_x(x)}{p_x(x) + p_\theta(x)}$$

Notice that dividing the numerator and denominator by $p_\theta(x)$ reveals the **density ratio** $r(x) = p_x(x) / p_\theta(x)$:

$$D^*(x) = \frac{r(x)}{r(x) + 1} \iff r(x) = \frac{D^*(x)}{1 - D^*(x)}$$

When this optimal discriminator $D^*(x)$ is plugged back into the binary cross-entropy loss function, the resulting maximum value is exactly proportional to the **Jensen-Shannon Divergence (JSD)**:

$$J(\theta, w^*) = 2 \cdot \text{JSD}(p_x \parallel p_\theta) - 2\log 2$$

Where $\text{JSD}(p_x \parallel p_\theta) = \frac{1}{2} D_{KL}\left(p_x \parallel \frac{p_x + p_\theta}{2}\right) + \frac{1}{2} D_{KL}\left(p_\theta \parallel \frac{p_x + p_\theta}{2}\right)$.

This profound result proves that training a binary classifier to distinguish real and fake data is mathematically equivalent to calculating the Jensen-Shannon distance between the two distributions!

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
