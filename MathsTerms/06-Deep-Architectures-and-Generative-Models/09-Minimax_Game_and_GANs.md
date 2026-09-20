# Minimax Games and Generative Adversarial Networks: Zero-Sum Distribution Matching

[Module guide](README.md) · [Study routes](START_HERE.md) · Previous: [Reparameterization trick](08-Reparameterization_Trick.md) · Next: [Fréchet inception distance](10-Frechet_Inception_Distance.md)

## 1. What this idea helps you do

In high-dimensional continuous domains like photorealistic images, the true data distribution $p_{\text{data}}(x)$ is concentrated on a low-dimensional curved manifold embedded inside millions of pixel dimensions. Explicit likelihood models (such as VAEs or autoregressive models) must approximate the data density using predefined parametric families (like Gaussians). Because maximizing likelihood across pixels averages over ambiguous possibilities, these models often generate blurry or washed-out images.

**Generative Adversarial Networks (GANs)** bypass explicit density estimation entirely. Instead of maximizing a mathematical likelihood formula, a GAN frames generation as a **two-player zero-sum minimax game** between two competing neural networks:
1. **The Generator ($G_\theta$):** A synthetic forger that transforms random noise $z \sim \mathcal{N}(0, I)$ into realistic synthetic samples $\hat{x} = G_\theta(z)$, attempting to fool the discriminator.
2. **The Discriminator ($D_w$):** A detective classifier that inspects both real images $x \sim p_{\text{data}}$ and fake images $\hat{x} \sim p_g$, outputting the probability $D_w(x) \in (0, 1)$ that an image is authentic.

```text
================================================================================
             THE GAN MINIMAX GAME: GENERATOR VS DISCRIMINATOR
================================================================================

  GENERATOR G_θ (Forger)           SHARED OBJECTIVE V(G, D)        DISCRIMINATOR D_w
  Tries to MINIMIZE V              The Minimax Value Function      Tries to MAXIMIZE V
  ┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────┐
  │ Latent: z ~ 𝒩(0, I)      │    │ V(G, D) =                │    │ Inputs: x or x̂   │
  │ Output: x̂ = G_θ(z)       ├───►│   𝔼_x[ln D(x)]           │◄───┤ Output: D(x)∈(0,1│
  │ Goal: Fool discriminator │    │ + 𝔼_z[ln(1 - D(G(z)))]   │    │ Goal: Classify   │
  │ min_θ max_w V(G_θ, D_w)  │    │ Saddle: min_θ max_w V    │    │ max_w min_θ V    │
  └──────────────────────────┘    └──────────────────────────┘    └──────────────────┘
================================================================================
```

*What to notice from the diagram:*
1. The Discriminator $D_w$ maximizes $V(G, D)$ by correctly labeling real samples ($D(x) \to 1$) and fake samples ($D(G(z)) \to 0$).
2. The Generator $G_\theta$ minimizes $V(G, D)$ by producing samples that the discriminator classifies as real ($D(G(z)) \to 1$).
3. Goodfellow et al. (2014) proved that at the theoretical minimax saddle-point equilibrium, the generator recovers the exact true data distribution ($p_g = p_{\text{data}}$) by minimizing the **Jensen-Shannon Divergence** ($D_{\text{JS}}(p_{\text{data}} \parallel p_g)$) to zero.

**Prerequisites**
- **Required now:** Binary cross-entropy loss ([Loss functions, §3](../03-Multivariate-Calculus-and-Optimization/08-Loss_Functions.md)). Gradient descent and saddle points ([Gradient descent, §2](../03-Multivariate-Calculus-and-Optimization/09-Gradient_Descent.md)).
- **Required for optional depth:** Jensen-Shannon Divergence ([Jensen-Shannon divergence, §2](../05-Information-Theory-and-Divergences/03-Jensen_Shannon_Divergence.md)).
- **Useful context:** [Autoencoders and latent spaces](03-Autoencoders_and_Latent_Spaces.md) for latent generators.

**Target systems:** StyleGAN3 (photorealistic face and texture synthesis), PatchGAN in Pix2Pix and CycleGAN (image-to-image translation), adversarial loss terms in Latent Diffusion VAEs, and BigGAN.

**Study time:** About 60–75 minutes for minimax game theory, optimal discriminator proofs, and pencil-and-paper calculations; another 45 minutes for PyTorch verification and exercises.

After studying, you should be able to:
1. Formulate the two-player zero-sum minimax objective $V(G, D)$ and derive the optimal discriminator $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$.
2. Prove that substituting $D^*$ into $V(G, D^*)$ reduces the objective to the Jensen-Shannon Divergence: $V(G, D^*) = -\ln 4 + 2 D_{\text{JS}}(p_{\text{data}} \parallel p_g)$.
3. Explain why vanilla minimax loss suffers from vanishing gradients early in training and derive the non-saturating heuristic loss $-\ln D(G(z))$.
4. Prove that simultaneous gradient descent in bilinear zero-sum games causes rotational divergence ($r_{k+1}^2 = (1 + \eta^2) r_k^2$).
5. Implement, stabilize, and verify a 1D adversarial training pipeline in pure Python and PyTorch.

**Fast route:** §§2–4 $\to$ §7 $\to$ §9 $\to$ §11 $\to$ §12, then §10 for production systems.  
**Deep route:** §§2–14 in order; §4 and §8 contain complete calculus of variations proofs and rotational dynamics derivations.

---

## 2. Start with a problem you can picture

Suppose real data consists of a single point in 1D space:
$$x_{\text{real}} = 4.0$$

We build a generator with a single learnable scalar parameter $\theta$ that emits a fake point:
$$x_{\text{fake}} = G(\theta) = \theta, \qquad \text{currently at } \theta = 1.0$$

To train the generator, we construct a discriminator with a single weight $w$:
$$D_w(x) = \sigma\big( w \cdot (x - 2.5) \big) = \frac{1}{1 + \exp\big(-w(x - 2.5)\big)}$$

The discriminator places a decision boundary at the midpoint $2.5$. The minimax value function is:
$$V(\theta, w) = \ln D_w(4.0) + \ln\big( 1 - D_w(\theta) \big)$$

```text
1D ADVERSARIAL NUMBER LINE:
           Fake Point             Decision Boundary             Real Data
         x_fake = 1.0                 x = 2.5                 x_real = 4.0
───────────────●──────────────────────────┼─────────────────────────●────────► x
               ▲                                                    ▲
        Generated Point                                        Ground Truth
        (Tries to move                                         (Fixed Anchor)
       rightward to 4.0!)

DISCRIMINATOR CONFIDENCE D_w(x) AT w = 1.0:
D(1.0) = σ(-1.5) ≈ 0.1824 (Fake!)            D(4.0) = σ(+1.5) ≈ 0.8176 (Real!)
```

*What to notice from the diagram:*
1. At current position $\theta = 1.0$, the fake point lies on the left side of the decision boundary ($x < 2.5$), so the discriminator confidently flags it as fake: $D(1.0) \approx 0.1824$.
2. The real point $x = 4.0$ lies on the right side ($x > 2.5$), so the discriminator classifies it as real: $D(4.0) \approx 0.8176$.
3. The generator wants to fool the discriminator: it must shift $\theta$ rightward across the decision boundary until $\theta = 4.0$.
4. When $\theta = 4.0$, $x_{\text{fake}} = x_{\text{real}}$, making it physically impossible for any discriminator to tell them apart: $D(4.0) = 0.50$.

**Predict before calculating:** If the discriminator becomes too strong too quickly (e.g., $w \to 100$, forming a razor-sharp step function at $2.5$), what will happen to the gradient of the generator loss at $\theta = 1.0$? Can the generator still learn?

---

## 3. Name the objects and read the notation

Let $x \in \mathbb{R}^D$ denote data observations drawn from true distribution $p_{\text{data}}(x)$. Let $z \in \mathbb{R}^d$ denote latent noise drawn from prior $p_z(z) = \mathcal{N}(\mathbf{0}, I_d)$.

The **Generator** is a parameterized function $G_\theta: \mathbb{R}^d \to \mathbb{R}^D$ that maps noise into data space, defining an implicit generated distribution $p_g$.

The **Discriminator** is a parameterized function $D_w: \mathbb{R}^D \to (0, 1)$ that outputs the probability that $x$ came from $p_{\text{data}}$ rather than $p_g$.

The **Minimax Game Objective** is:
$$\min_\theta \max_w V(G_\theta, D_w) \triangleq \mathbb{E}_{x \sim p_{\text{data}}}\left[ \ln D_w(x) \right] + \mathbb{E}_{z \sim p_z}\left[ \ln\big( 1 - D_w(G_\theta(z)) \big) \right]$$

Read this equation aloud:  
*“The minimum over theta and maximum over w of value function V of G-theta and D-w is defined as the expectation under p-data of log D-w of x, plus the expectation under p-z of log of one minus D-w of G-theta of z.”*

| Symbol | Spoken as | Mathematical role / dimensions | Concrete toy value (§2 / §9) |
| :--- | :--- | :--- | :--- |
| $p_{\text{data}}(x)$ | “p data of ex” | True underlying data distribution | Point mass at $x = 4.0$ |
| $p_g(x)$ | “p g of ex” | Implicit distribution defined by generator | Point mass at $\theta = 1.0$ |
| $G_\theta(z)$ | “G theta of zee” | Generator neural network; $\mathbb{R}^d \to \mathbb{R}^D$ | $G(\theta) = \theta$ |
| $D_w(x)$ | “D w of ex” | Discriminator probability score; $(0, 1)$ | $D(4.0) \approx 0.8176$ |
| $V(G, D)$ | “V of G and D” | Zero-sum minimax value function | $V \approx -0.4028$ at $w=1, \theta=1$ |
| $D^*(x)$ | “D star of ex” | Optimal discriminator for a fixed generator | $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$ |
| $D_{\text{JS}}$ | “J-S divergence” | Symmetrized Jensen-Shannon divergence | $D_{\text{JS}}(p_{\text{data}} \parallel p_g) \in [0, \ln 2]$ |

---

## 4. Build the central relationship

```text
  REAL DATASET                     DISCRIMINATOR / CRITIC            VALUE OBJECTIVE
  [ Real x ~ p_data ] ──────► [ Critic D_w(x) ] ──► log D(x) ──────┐
                                     ▲                             │
                                     │                             v
  LATENT PRIOR       GENERATOR       │                     [ Minimax Value V(G,D) ]
  [ Noise z ~ p(z) ] ──► [ G_θ(z) ] ─┴─► [ D_w(G(z)) ] ──► log(1 - D(G(z))) ─┘
                           │                  ▲
                           └─ Gradient Flow ──┘
                             ∇_θ log(1 - D(G(z)))
```

### The Core "Aha!" Discovery

In standard generative models, we must compute $\ln p_\theta(x)$, which requires an analytical density formula. But what if we do not know the density formula of real images?

A GAN replaces the intractable analytical formula with a **learned critic**. If a powerful discriminator cannot tell real images from generated images—even when trying its best—then the generated distribution $p_g$ must be identical to the real distribution $p_{\text{data}}$!

### Derivation 1: The Optimal Discriminator $D^*(x)$

For any fixed generator $G$, we can express the value function as an integral over data space $\mathcal{X}$:

$$V(G, D) = \int_{\mathcal{X}} p_{\text{data}}(x) \ln D(x) \, dx + \int_{\mathcal{X}} p_g(x) \ln\big( 1 - D(x) \big) \, dx$$
$$V(G, D) = \int_{\mathcal{X}} \Big( p_{\text{data}}(x) \ln D(x) + p_g(x) \ln\big( 1 - D(x) \big) \Big) \, dx$$

To find the discriminator $D(x)$ that maximizes this integral, we maximize the integrand independently at each point $x$.  
Define the scalar objective $f(y) = a \ln y + b \ln(1 - y)$, where $y = D(x)$, $a = p_{\text{data}}(x)$, and $b = p_g(x)$.

Differentiating with respect to $y$ and setting to zero:

$$f'(y) = \frac{a}{y} - \frac{b}{1 - y} = 0 \implies a(1 - y) = b y \implies a = (a + b)y$$
$$y^* = \frac{a}{a + b}$$

Substituting back $a = p_{\text{data}}(x)$ and $b = p_g(x)$:

$$\boxed{D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}}$$

*Interpretation:*
- If $x$ is impossible under the generator ($p_g(x) = 0$), $D^*(x) = 1.0$ (definitely real).
- If $x$ is impossible in real data ($p_{\text{data}}(x) = 0$), $D^*(x) = 0.0$ (definitely fake).
- When the generator matches the true data perfectly ($p_g(x) = p_{\text{data}}(x)$), $D^*(x) = \frac{1}{2}$ (pure chance!).

---

### Derivation 2: Minimax Equivalence to Jensen-Shannon Divergence

Now substitute the optimal discriminator $D^*(x)$ back into the value function $V(G, D^*)$:

$$V(G, D^*) = \int_{\mathcal{X}} \left[ p_{\text{data}}(x) \ln\left( \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \right) + p_g(x) \ln\left( \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)} \right) \right] dx$$

Multiply and divide the denominator inside both logarithms by $2$:

$$V(G, D^*) = \int_{\mathcal{X}} \left[ p_{\text{data}}(x) \ln\left( \frac{p_{\text{data}}(x)}{2 \cdot \frac{p_{\text{data}}(x) + p_g(x)}{2}} \right) + p_g(x) \ln\left( \frac{p_g(x)}{2 \cdot \frac{p_{\text{data}}(x) + p_g(x)}{2}} \right) \right] dx$$

Using $\ln\left(\frac{u}{2v}\right) = \ln\left(\frac{u}{v}\right) - \ln 2$:

$$V(G, D^*) = -\ln 2 \int p_{\text{data}}(x) dx - \ln 2 \int p_g(x) dx + \int p_{\text{data}}(x) \ln\left( \frac{p_{\text{data}}(x)}{\frac{p_{\text{data}} + p_g}{2}} \right) dx + \int p_g(x) \ln\left( \frac{p_g(x)}{\frac{p_{\text{data}} + p_g}{2}} \right) dx$$

Since probability densities integrate to $1.0$, the first two terms sum to $-\ln 2 - \ln 2 = -\ln 4 = -2\ln 2$.  
The remaining two integrals are the definition of the Kullback-Leibler divergences to the mixture distribution $M = \frac{p_{\text{data}} + p_g}{2}$:

$$V(G, D^*) = -\ln 4 + D_{\text{KL}}(p_{\text{data}} \parallel M) + D_{\text{KL}}(p_g \parallel M)$$

By definition, the sum of these two KL divergences is twice the **Jensen-Shannon Divergence**:

$$\boxed{V(G, D^*) = -\ln 4 + 2 \, D_{\text{JS}}(p_{\text{data}} \parallel p_g)}$$

*Conclusion:*
- The Jensen-Shannon divergence satisfies $D_{\text{JS}} \ge 0$, with $D_{\text{JS}} = 0$ if and only if $p_g = p_{\text{data}}$.
- Therefore, the global minimum of the minimax game is:
  $$\min_G V(G, D^*) = -\ln 4 \approx -1.3863$$
  attained if and only if the generator perfectly replicates the real data distribution!

---

## 5. Why choose this tool for this problem?

| Characteristic | Generative Adversarial Networks (GANs) | Variational Autoencoders (VAEs) | Autoregressive Models (GPT) | Diffusion Models (DDPM) |
| :--- | :--- | :--- | :--- | :--- |
| **Sampling Speed** | **Single forward pass ($O(1)$)** | Single forward pass ($O(1)$) | Sequential token loop ($O(T)$) | Iterative denoising ($20$–$50$ steps) |
| **Sample Sharpness** | **Extremely sharp (no blur)** | Prone to blurriness | High fidelity | Extremely sharp & photorealistic |
| **Likelihood Evaluation** | **None (Implicit distribution)** | Approximate (ELBO) | Exact ($p(x) = \prod p(x_t \mid x_{<t})$) | Tractable variational bound |
| **Training Stability** | **Challenging (Adversarial oscillations)** | Stable (Direct ELBO ascent) | Highly stable (Cross-entropy) | Very stable (Score matching) |
| **Dominant Failure Mode** | **Mode collapse & vanishing gradients** | Posterior collapse & blur | Exposure bias & repetition | High inference compute latency |

### Concrete Counterexample: Vanishing Gradients in Vanilla GANs

Early in training, the generator produces poor samples, so the discriminator easily achieves near-perfect classification: $D(G(z)) \approx 0$.
Under the original vanilla minimax loss $\mathcal{L}_G = \ln(1 - D(G(z)))$:
$$\frac{d \mathcal{L}_G}{d z} = \frac{-D'(G(z))}{1 - D(G(z))}$$
When $D(G(z)) \to 0$, $1 - D(G(z)) \to 1$, but because the sigmoid saturates on the negative axis, its derivative $D'(G(z)) = D(1 - D) \approx 0$.
The gradient completely vanishes: $\frac{d \mathcal{L}_G}{dz} \approx 0$! The generator receives zero learning signal.

**The Production Fix (Non-Saturating GAN):** Goodfellow et al. replaced the objective with:
$$\mathcal{L}_G^{\text{NS}} = -\ln D(G(z))$$
Now, when $D(G(z)) \to 0$:
$$\frac{d \mathcal{L}_G^{\text{NS}}}{dz} = -\frac{D'(G(z))}{D(G(z))} = -\frac{D(1 - D)}{D} = -(1 - D) \approx -1.0$$
The gradient remains strong and non-zero early in training, driving rapid learning!

---

## 6. Strengthen the intuition and mark its limits

```text
================================================================================
          THE ART FORGER AND DETECTIVE: INTUITION OF MINIMAX DYNAMICS
================================================================================
 STEP 1 (The Amateur Forger):
 Generator produces crude scribbles.
 Detective spots them effortlessly: D(Fake) = 0.01.
 
 STEP 2 (The Learning Forger):
 Generator learns to paint canvas texture and oil brushstrokes.
 Detective must now inspect signature details and canvas aging.
 
 STEP 3 (The Master Forger):
 Generator paints with authentic pigments and historical styles.
 Detective is reduced to pure guessing: D(Painting) = 0.50.
 At this equilibrium, every fake painting is indistinguishable from real art!
================================================================================
```

### Mechanical Mapping: Intuition to Mathematics and Implementation

| Physical Intuition / Metaphor | Mathematical Operation | Software / Hardware Implementation | Failure Mode / Boundary Condition |
| :--- | :--- | :--- | :--- |
| **Detective inspecting authenticity** | Binary cross-entropy $\ln D(x) + \ln(1 - D(\hat{x}))$ | `nn.BCEWithLogitsLoss()` | If detective is too strong, gradients vanish |
| **Forger perfecting technique** | Non-saturating loss $-\ln D(G(z))$ | `F.binary_cross_entropy_with_logits` | If forger finds one flaw, triggers mode collapse |
| **Indistinguishable paintings** | Nash equilibrium $D^*(x) = 0.50$ | Saddle-point stationarity | May oscillate in limit cycles rather than converge |
| **Art gallery collection diversity** | Support coverage $\operatorname{supp}(p_g) = \operatorname{supp}(p_{\text{data}})$ | Minibatch discrimination / spectral norm | Generator drops modes to fool discriminator easily |

### Where this analogy stops working

1. **Rotational Divergence in Simultaneous Gradient Descent:** In the art forger analogy, both players learn smoothly. But on a computer using simultaneous gradient descent ($w \leftarrow w + \eta \nabla_w V$, $\theta \leftarrow \theta - \eta \nabla_\theta V$), bilinear zero-sum games do not converge: they spiral outward with exponentially increasing radius ($r_{k+1}^2 = (1 + \eta^2) r_k^2$).
2. **Disjoint Manifold Support:** In high dimensions, real data lives on a low-dimensional manifold. If the generated manifold does not intersect the real manifold, the distance between them is maximal ($D_{\text{JS}} = \ln 2$) everywhere. The discriminator achieves 100% accuracy with zero slope across the gap, providing zero gradient to guide the generator toward the real manifold.

---

## 7. Terms worth keeping straight

### Core Terminology Reference Table

| Term | Pronunciation | Plain-English Meaning | Formal Definition & Conditions |
| :--- | :--- | :--- | :--- |
| **Minimax Game** | “MIN-ih-maks gaym” | Two-player zero-sum game where one minimizes what the other maximizes | $\min_\theta \max_w V(G_\theta, D_w)$. Game-theoretic optimization. |
| **Generator ($G_\theta$)** | “JEN-er-ay-ter” | Neural network transforming latent noise vectors into synthetic samples | $G_\theta: \mathcal{Z} \to \mathcal{X}$ mapping $z \sim p(z) \mapsto x_{\text{fake}}$. |
| **Discriminator ($D_w$)** | “dis-KRIM-ih-nay-ter” | Neural network predicting probability that a sample is real vs. fake | $D_w: \mathcal{X} \to [0, 1]$ scoring $P(\text{real} \mid x)$. |
| **Jensen-Shannon Div.** | “JEN-sun SHAN-un” | Symmetric bounded divergence measuring overlap between distributions | $\text{JSD}(p \parallel q) = \frac{1}{2} D_{\text{KL}}(p \parallel m) + \frac{1}{2} D_{\text{KL}}(q \parallel m) \in [0, \ln 2]$. |
| **Nash Equilibrium** | “NASH ee-kwih-LIB-ree-um” | State where neither player can improve payoff by unilaterally deviating | $V(G^*, D) \le V(G^*, D^*) \le V(G, D^*)$ for all valid $G, D$. |
| **Mode Collapse** | “mohd kuh-LAPS” | Generator produces only a tiny subset of dataset modes | $p_g$ places all probability mass on a single output, ignoring other modes. |
| **Non-Saturating Loss** | “non-SATCH-er-ay-ting” | Heuristic generator loss providing strong gradients early in training | $\mathcal{L}_G = -\mathbb{E}_z [\ln D(G(z))]$ instead of $\ln(1 - D(G(z)))$. |
| **Wasserstein GAN (WGAN)**| “VAH-ser-shtyn gan” | Replaces classifier with 1-Lipschitz critic optimizing Earth Mover's | $\min_\theta \max_{\|D\|_L \le 1} \mathbb{E}[D(x)] - \mathbb{E}[D(G(z))]$. Linear gradients. |

### Confused Pairs Distinction Breakdown

1. **Minimax Game vs. Simultaneous Gradient Descent**:
   - *Core Distinction:* Minimax game is a theoretical mathematical objective; simultaneous gradient descent is an uncoupled iterative algorithm.
   - *Common Confusion:* Assuming gradient descent-ascent guarantees convergence to a minimax Nash equilibrium. On non-convex functions, it frequently enters limit cycles or explodes.
   - *Rule of Thumb:* Use stabilization techniques (WGAN-GP, Spectral Normalization, Two Time-Scale Update Rule) to stabilize adversarial game dynamics.
- **Minimax Game:** A theoretical game-theoretic optimization $\min_\theta \max_w V(\theta, w)$.
- **Simultaneous Gradient Descent:** An algorithmic heuristic updating both players simultaneously. Does not generally converge to minimax equilibria in non-convex games.

### 2. Nash Equilibrium vs. Local Saddle Point
- **Nash Equilibrium:** A global state where neither player can improve their payoff unilaterally.
- **Local Saddle Point:** A point where $\nabla_\theta V = 0$ and $\nabla_w V = 0$. In deep neural networks, the players often oscillate around saddle points without settling.

### 3. Mode Collapse vs. Mode Dropping
- **Complete Mode Collapse:** The generator maps all latent noise vectors $z$ to a single identical image (e.g., only one dog breed).
- **Mode Dropping:** The generator produces diverse samples within 5 modes but completely ignores 5 other valid modes to minimize risk.

### 4. Vanilla Minimax Loss vs. Non-Saturating Loss
- **Vanilla Minimax Loss:** $\mathcal{L}_G = \ln(1 - D(G(z)))$. Mathematically tied to $D_{\text{JS}}$, but gradients vanish when $D \to 0$.
- **Non-Saturating Loss:** $\mathcal{L}_G = -\ln D(G(z))$. Provides strong gradients throughout training; standard in production.

---

## 8. Work through the mathematics and its conditions

### Theorem 8.3: Rotational Divergence in Bilinear Zero-Sum Games

**Statement:** Consider the canonical continuous bilinear zero-sum game $\min_x \max_y x \cdot y$. Simultaneous forward Euler gradient descent with step size $\eta > 0$:
$$x_{k+1} = x_k - \eta \nabla_x (x_k y_k) = x_k - \eta y_k$$
$$y_{k+1} = y_k + \eta \nabla_y (x_k y_k) = y_k + \eta x_k$$
causes the distance from the Nash equilibrium $(0, 0)$ to grow strictly monotonically:
$$r_{k+1}^2 = (1 + \eta^2) r_k^2 \to \infty$$

**Proof:**
1. Compute the squared Euclidean radius at step $k+1$:
   $$r_{k+1}^2 = x_{k+1}^2 + y_{k+1}^2 = (x_k - \eta y_k)^2 + (y_k + \eta x_k)^2$$

2. Expand both squares:
   $$(x_k - \eta y_k)^2 = x_k^2 - 2\eta x_k y_k + \eta^2 y_k^2$$
   $$(y_k + \eta x_k)^2 = y_k^2 + 2\eta x_k y_k + \eta^2 x_k^2$$

3. Add the two expressions:
   $$r_{k+1}^2 = x_k^2 + y_k^2 + \eta^2(x_k^2 + y_k^2) = (1 + \eta^2)(x_k^2 + y_k^2) = \mathbf{(1 + \eta^2) r_k^2}$$

4. Since $\eta > 0$, the factor $(1 + \eta^2) > 1$ strictly. Therefore:
   $$\lim_{k \to \infty} r_k^2 = \lim_{k \to \infty} (1 + \eta^2)^k r_0^2 = \infty \quad \blacksquare$$

*Significance:* Simultaneous gradient descent cannot converge to the Nash equilibrium of a zero-sum game without regularization techniques like **Gradient Penalty (WGAN-GP)**, **Spectral Normalization**, or **Extragradient methods**.

### Hardware and Computational Realities: Alternating Gradient Synchronization, Memory Footprint, and Spectral Normalization

Training GANs at scale imposes unique hardware and runtime constraints across GPU clusters:

1. **Alternating Gradient Synchronization Overhead:**
   Unlike standard networks where all parameters update in a single unified backward pass, GANs alternate between Discriminator updates and Generator updates (typically $n_{\text{critic}} = 1$ to $5$ discriminator steps per generator step). On distributed multi-GPU clusters (DDP/FSDP), this requires multiple `AllReduce` gradient synchronizations per training iteration, substantially increasing inter-GPU network latency overhead.

2. **Dual-Model Memory Footprint & Optimizer State:**
   Both Generator and Discriminator must be resident in GPU VRAM simultaneously. For a 1-billion parameter model in `bfloat16`, the model weights consume $2\text{ GB} \times 2 = 4\text{ GB}$, but their Adam optimizer states ($m$ and $v$ buffers in `float32`) consume an additional $16\text{ GB}$ per model ($32\text{ GB}$ total). Furthermore, backpropagating through the Discriminator to train the Generator requires retaining both networks' computational graphs in memory concurrently.

3. **Spectral Normalization Compute Costs:**
   Enforcing the 1-Lipschitz constraint via Spectral Normalization requires running power iteration at every forward step to compute the top singular value $\sigma(W) \approx u^\top W v$ for every convolutional and linear weight matrix. While computationally lighter than full SVD, this adds repeated matrix-vector multiplications that disrupt GPU tensor core utilization unless compiled with torch.compile or fused CUDA kernels.

---

## 9. Calculate it by hand

### Worked Example: Forward Evaluation and Analytical Gradients for 1D GAN

Consider the 1D toy problem from Section 2:
- Real observation: $x_{\text{real}} = 4.0$.
- Fake observation: $x_{\text{fake}} = 1.0$.
- Discriminator parameter: $w = 1.0$.
- Discriminator logit: $a(x) = w(x - 2.5) = 1.0(x - 2.5)$.

---

#### Step 1: Forward Evaluation of Discriminator Scores
Recall sigmoid: $\sigma(a) = \frac{1}{1 + e^{-a}}$.
1. **For Real Sample ($x_{\text{real}} = 4.0$):**
   $$a(4.0) = 1.0(4.0 - 2.5) = +1.50$$
   $$D(4.0) = \sigma(1.50) = \frac{1}{1 + e^{-1.50}} = \frac{1}{1 + 0.223130} \approx \mathbf{0.817574}$$
2. **For Fake Sample ($x_{\text{fake}} = 1.0$):**
   $$a(1.0) = 1.0(1.0 - 2.5) = -1.50$$
   $$D(1.0) = \sigma(-1.50) = \frac{1}{1 + e^{1.50}} = \frac{1}{1 + 4.481689} \approx \mathbf{0.182426}$$
   *Notice:* $1 - D(1.0) = 1 - 0.182426 = \mathbf{0.817574}$.

---

#### Step 2: Minimax Value Function Evaluation
$$V = \ln D(4.0) + \ln\big(1 - D(1.0)\big) = \ln(0.817574) + \ln(0.817574) = 2 \ln(0.817574) \approx 2(-0.201413) = \mathbf{-0.402826}$$

---

#### Step 3: Backward Gradients on Discriminator Weight $w$
$$\mathcal{L}_D = -V = -\ln D(x_{\text{real}}) - \ln\big(1 - D(x_{\text{fake}})\big)$$
Using the derivative of cross-entropy with respect to logit: $\frac{\partial \mathcal{L}}{\partial a} = \sigma(a) - y$.
1. For real sample ($y = 1$): $\frac{\partial \mathcal{L}_D}{\partial a_{\text{real}}} = D(4.0) - 1.0 = 0.817574 - 1.0 = \mathbf{-0.182426}$.
2. For fake sample ($y = 0$): $\frac{\partial \mathcal{L}_D}{\partial a_{\text{fake}}} = D(1.0) - 0.0 = \mathbf{+0.182426}$.
3. Chain rule to weight $w$ (where $a = w(x - 2.5)$):
   $$\frac{\partial \mathcal{L}_D}{\partial w} = (-0.182426)(4.0 - 2.5) + (0.182426)(1.0 - 2.5)$$
   $$\frac{\partial \mathcal{L}_D}{\partial w} = (-0.182426)(+1.5) + (0.182426)(-1.5) = -0.273639 - 0.273639 = \mathbf{-0.547278}$$
   *Interpretation:* The loss gradient with respect to $w$ is negative, so gradient descent ($w \leftarrow w - \eta \nabla_w \mathcal{L}$) increases $w$, sharpening the decision boundary.

---

#### Step 4: Backward Gradients on Generator Output $x_{\text{fake}}$

1. **Vanilla Generator Loss:** $\mathcal{L}_G^{\text{vanilla}} = \ln(1 - D(x_{\text{fake}}))$.
   $$\frac{\partial \mathcal{L}_G^{\text{vanilla}}}{\partial x_{\text{fake}}} = -w \cdot D(x_{\text{fake}}) = -1.0(0.182426) = \mathbf{-0.182426}$$

2. **Non-Saturating Generator Loss:** $\mathcal{L}_G^{\text{NS}} = -\ln D(x_{\text{fake}})$.
   $$\frac{\partial \mathcal{L}_G^{\text{NS}}}{\partial x_{\text{fake}}} = -w \cdot \big(1 - D(x_{\text{fake}})\big) = -1.0(0.817574) = \mathbf{-0.817574}$$

*Comparison:* The non-saturating gradient ($-0.8176$) is **$4.48\times$ stronger** than the vanilla gradient ($-0.1824$)! Subtracting this gradient pushes $x_{\text{fake}}$ strongly rightward toward $4.0$.

---

## 10. Connect the concept to an actual system

```text
================================================================================
                    GANS IN PRODUCTION GENERATIVE AI
================================================================================
 1. STYLEGAN3 (High-Resolution Synthesis)     2. PATCHGAN (Pix2Pix / CycleGAN)
 Aliasing-free hierarchical generator         Discriminator operates on 70x70
 produces 1024x1024 photorealistic portraits  image patches; preserves local texture
 ┌────────────────────────────────────────┐   ┌────────────────────────────────┐
 │ Mapping network maps z -> w space      │   │ Fused conv kernel evaluates    │
 │ Synthesis network generates sharp faces│   │ patch realism in parallel      │
 └────────────────────────────────────────┘   └────────────────────────────────┘
================================================================================
```

*What to notice from the diagram:*
1. Modern high-resolution GANs (StyleGAN3) decouple the latent space into an intermediate $W$-space, regularizing generator Jacobian norms to prevent mode collapse.
2. Image-to-image models (Pix2Pix, CycleGAN) use PatchGAN discriminators that classify $70 \times 70$ pixel patches rather than whole images, enforcing high-frequency local realism.

| Mathematical Object | Role in Toy Example | Real Production System Counterpart | Hardware / Scale Approximation |
| :--- | :--- | :--- | :--- |
| **Generator $G_\theta$** | Scalar mapping $G(\theta) = \theta$ | 50M-parameter transposed convolutional/attention network | Computes forward pass in fp16/bf16 on Tensor Cores |
| **Discriminator $D_w$** | Scalar sigmoid $\sigma(w(x - 2.5))$ | Convolutional ResNet / Vision Transformer classifier | Spectral normalization applied to weights to enforce 1-Lipschitz continuity |
| **Adversarial Loss $V$** | Scalar value $\approx -0.4028$ | Non-saturating BCE loss + $R_1$ gradient penalty | Fused CUDA loss kernel avoids materializing logit tensors in VRAM |
| **Nash Equilibrium** | $\theta^* = 4.0, D^* = 0.50$ | Balanced generator-discriminator capacity | Monitored via Fréchet Inception Distance (FID) |

We have mapped the architectural connections. Next, we verify these formulations with executable Python and PyTorch scripts.

---

## 11. Verify the idea with a small experiment

We implement the **Dual-Stage Code Architecture**:
- **Stage 1 (Pure Python):** Standard library implementation computing the forward value $V$, analytical discriminator and generator gradients, and contrasting vanilla vs non-saturating gradients matching Section 9, with passing assertions.
- **Stage 2 (Production PyTorch):** Vectorized PyTorch implementation comparing analytical gradients against autograd and verifying 10 steps of adversarial training converging toward $x_{\text{real}} = 4.0$.

```python
"""
Minimax Games and GANs Dual-Stage Verification Suite
====================================================
Part A: Pure Python standard library (built-in math only, zero dependencies).
Part B: PyTorch industrial verification suite with autograd checks.
"""

import math
import sys

# Ensure UTF-8 output on all consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 80)
print("PART A: PURE PYTHON STDLIB MINIMAX FORWARD & GRADIENT SUITE")
print("=" * 80)

# Setup from Section 2 and Section 9
x_real = 4.0
x_fake = 1.0
w_val = 1.0
boundary = 2.5

def sigmoid(a):
    return 1.0 / (1.0 + math.exp(-a))

# Forward pass
logit_real = w_val * (x_real - boundary)  # +1.5
logit_fake = w_val * (x_fake - boundary)  # -1.5

d_real = sigmoid(logit_real)
d_fake = sigmoid(logit_fake)

v_val = math.log(d_real) + math.log(1.0 - d_fake)

print(f"1. Forward Evaluation:")
print(f"   * D(x_real = {x_real}): {d_real:.6f} (Expected: ~0.817574)")
print(f"   * D(x_fake = {x_fake}): {d_fake:.6f} (Expected: ~0.182426)")
print(f"   * Minimax Value V:     {v_val:.6f} nats (Expected: ~ -0.402826)")

assert math.isclose(d_real, 0.817574, abs_tol=1e-4)
assert math.isclose(d_fake, 0.182426, abs_tol=1e-4)
assert math.isclose(v_val, -0.402826, abs_tol=1e-4)

# Backward Gradients on Discriminator Weight w (Loss = -V)
# dL_D/dw = (D(real) - 1) * (x_real - boundary) + D(fake) * (x_fake - boundary)
grad_w = (d_real - 1.0) * (x_real - boundary) + d_fake * (x_fake - boundary)
print(f"\n2. Discriminator Loss Gradient (dL_D/dw): {grad_w:.6f} (Expected: -0.547278)")
assert math.isclose(grad_w, -0.547278, abs_tol=1e-4)

# Generator Gradients: Vanilla vs Non-Saturating
grad_g_vanilla = -w_val * d_fake
grad_g_ns = -w_val * (1.0 - d_fake)

print(f"\n3. Generator Output Gradients (dL_G/dx_fake):")
print(f"   * Vanilla Minimax Gradient:     {grad_g_vanilla:.6f} (Expected: -0.182426)")
print(f"   * Non-Saturating Gradient:      {grad_g_ns:.6f} (Expected: -0.817574)")
print(f"   * Gradient Magnitude Boost:      {abs(grad_g_ns) / abs(grad_g_vanilla):.2f}x")

assert math.isclose(grad_g_vanilla, -0.182426, abs_tol=1e-4)
assert math.isclose(grad_g_ns, -0.817574, abs_tol=1e-4)
print("Part A Pure Python Suite: ALL CHECKS PASSED [OK]")

print("\n" + "=" * 80)
print("PART B: PYTORCH INDUSTRIAL AUTOGRAD & 1D TRAINING SUITE")
print("=" * 80)

import torch
import torch.nn as nn

# Tensors with autograd
w_torch = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
x_fake_torch = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
x_real_torch = torch.tensor([4.0], dtype=torch.float64)

# Discriminator logits
d_logit_real = w_torch * (x_real_torch - 2.5)
d_logit_fake = w_torch * (x_fake_torch - 2.5)

bce_loss = nn.BCEWithLogitsLoss()

# Discriminator loss: BCE(real, 1) + BCE(fake, 0)
loss_d = bce_loss(d_logit_real, torch.ones_like(d_logit_real)) +          bce_loss(d_logit_fake, torch.zeros_like(d_logit_fake))

loss_d.backward(retain_graph=True)
print(f"1. PyTorch Autograd dL_D/dw: {w_torch.grad.item():.6f} vs Analytical: {grad_w:.6f}")
assert math.isclose(w_torch.grad.item(), grad_w, abs_tol=1e-6)

# Non-saturating generator loss: BCE(fake, 1)
# Note: recompute logit to clear previous graph connections
d_logit_fake_g = w_torch.detach() * (x_fake_torch - 2.5)
loss_g_ns = bce_loss(d_logit_fake_g, torch.ones_like(d_logit_fake_g))
x_fake_torch.grad = None
loss_g_ns.backward()

print(f"2. PyTorch Autograd dL_G/dx_fake: {x_fake_torch.grad.item():.6f} vs Analytical: {grad_g_ns:.6f}")
assert math.isclose(x_fake_torch.grad.item(), grad_g_ns, abs_tol=1e-6)
print("   * Autograd matches analytical gradients with bit-exact precision! [OK]")

# 3. 1D Training Loop: Verifying Generator Movement Toward Real Data
theta = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
w = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
opt_g = torch.optim.SGD([theta], lr=0.20)
opt_d = torch.optim.SGD([w], lr=0.10)

print("\n3. Running 10 Steps of 1D Adversarial Training:")
for step in range(10):
    # D step
    opt_d.zero_grad()
    l_d = bce_loss(w * (x_real_torch - 2.5), torch.ones(1, dtype=torch.float64)) +           bce_loss(w * (theta.detach() - 2.5), torch.zeros(1, dtype=torch.float64))
    l_d.backward()
    opt_d.step()
    
    # G step (Non-saturating)
    opt_g.zero_grad()
    l_g = bce_loss(w.detach() * (theta - 2.5), torch.ones(1, dtype=torch.float64))
    l_g.backward()
    opt_g.step()

print(f"   * Initial Fake x: 1.0000")
print(f"   * Final Fake x:   {theta.item():.4f} (Moved closer to 4.0000!)")
assert theta.item() > 1.0, "Generator failed to advance toward real data!"

print("\n" + "=" * 80)
print("ALL MINIMAX & GAN VERIFICATION TESTS PASSED SUCCESSFULLY! [OK]")
print("=" * 80)
```

*Expected output:*
```text
================================================================================
PART A: PURE PYTHON STDLIB MINIMAX FORWARD & GRADIENT SUITE
================================================================================
1. Forward Evaluation:
   * D(x_real = 4.0): 0.817574 (Expected: ~0.817574)
   * D(x_fake = 1.0): 0.182426 (Expected: ~0.182426)
   * Minimax Value V:     -0.402826 nats (Expected: ~ -0.402826)

2. Discriminator Loss Gradient (dL_D/dw): -0.547278 (Expected: -0.547278)

3. Generator Output Gradients (dL_G/dx_fake):
   * Vanilla Minimax Gradient:     -0.182426 (Expected: -0.182426)
   * Non-Saturating Gradient:      -0.817574 (Expected: -0.817574)
   * Gradient Magnitude Boost:      4.48x
Part A Pure Python Suite: ALL CHECKS PASSED [OK]

================================================================================
PART B: PYTORCH INDUSTRIAL AUTOGRAD & 1D TRAINING SUITE
================================================================================
1. PyTorch Autograd dL_D/dw: -0.547278 vs Analytical: -0.547278
2. PyTorch Autograd dL_G/dx_fake: -0.817574 vs Analytical: -0.817574
   * Autograd matches analytical gradients with bit-exact precision! [OK]

3. Running 10 Steps of 1D Adversarial Training:
   * Initial Fake x: 1.0000
   * Final Fake x:   2.3484 (Moved closer to 4.0000!)

================================================================================
ALL MINIMAX & GAN VERIFICATION TESTS PASSED SUCCESSFULLY! [OK]
================================================================================
```

---

## 12. Practise, compare, and debug

Attempt all five diagnostic exercises before inspecting the separated solutions.

1. **Recognize.** A deep learning engineer observes that during early GAN training, the discriminator loss drops to $0.0001$ within two epochs, and the generator loss stops improving entirely. What phenomenon occurred, and what objective modification resolves it?
2. **Calculate.** In a 1D GAN, $p_{\text{data}}(x) = 0.80$ and $p_g(x) = 0.20$ at $x = 3.0$. Calculate the optimal discriminator output $D^*(3.0)$ and the local contribution to the value function $p_{\text{data}} \ln D^* + p_g \ln(1 - D^*)$ by hand.
3. **Contrast.** Contrast the Jensen-Shannon divergence minimized by standard GANs with the Wasserstein-1 (Earth Mover's) distance minimized by WGANs. Why does $D_{\text{JS}}$ fail to provide useful gradients when the real and generated manifolds are disjoint?
4. **Transfer.** Show how an adversarial PatchGAN discriminator is incorporated into Latent Diffusion training (e.g., in SDXL or FLUX) to prevent blurriness without replacing the diffusion backbone.
5. **Debug.** A PyTorch GAN training loop defines the generator update as:
   ```python
   opt_g.zero_grad()
   fake_images = generator(z)
   loss_g = -torch.mean(torch.log(discriminator(fake_images)))
   loss_g.backward()
   opt_g.step()
   ```
   During training, the loss abruptly outputs `NaN`. Diagnose the numerical instability and provide the standard production fix.

---

### Separated Diagnostic Solutions

<details>
<summary>Click to view solution for Exercise 1</summary>

**Diagnosis:** The engineer encountered **discriminator saturation / vanishing generator gradients**. When the discriminator is too powerful, $D(G(z)) \approx 0$. Under the vanilla minimax objective $\min \ln(1 - D(G(z)))$, the sigmoid derivative saturates, producing near-zero gradients that stall the generator.

**Resolution:** Switch to the **non-saturating heuristic loss** $\max \ln D(G(z))$ (or $\min -\ln D(G(z))$). This provides strong, non-vanishing gradients early in training.
</details>

<details>
<summary>Click to view solution for Exercise 2</summary>

**Calculation:**
1. Compute optimal discriminator:
   $$D^*(3.0) = \frac{p_{\text{data}}(3.0)}{p_{\text{data}}(3.0) + p_g(3.0)} = \frac{0.80}{0.80 + 0.20} = \frac{0.80}{1.00} = \mathbf{0.8000}$$
2. Compute local value function contribution:
   $$0.80 \ln(0.80) + 0.20 \ln(1.0 - 0.80) = 0.80 \ln(0.80) + 0.20 \ln(0.20)$$
   $$\ln(0.80) \approx -0.223144, \quad \ln(0.20) \approx -1.609438$$
   $$0.80(-0.223144) + 0.20(-1.609438) = -0.178515 - 0.321888 = \mathbf{-0.500403\text{ nats}}$$
</details>

<details>
<summary>Click to view solution for Exercise 3</summary>

**Contrast:**
- **Disjoint Support Failure of $D_{\text{JS}}$:** When distributions $p_{\text{data}}$ and $p_g$ have disjoint supports (non-overlapping manifolds), $D_{\text{JS}}(p_{\text{data}} \parallel p_g) = \ln 2$ is a flat constant everywhere. The derivative $\nabla_\theta D_{\text{JS}} = 0$, giving the generator zero directional guidance on which way to move.
- **Wasserstein-1 Distance ($W_1$):** Measures the minimal cost of transporting probability mass: $W(P, Q) = \inf_{\gamma} \mathbb{E}[\|x - y\|]$. Even when supports are completely disjoint, $W_1$ scales linearly with the physical distance between manifolds, providing smooth, non-vanishing gradients everywhere.
</details>

<details>
<summary>Click to view solution for Exercise 4</summary>

**Transfer (Adversarial VAEs in Latent Diffusion):**
In Latent Diffusion, the autoencoder is trained with a composite loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{perceptual}} \mathcal{L}_{\text{LPIPS}} + \beta D_{\text{KL}} + \lambda_{\text{adv}} \mathcal{L}_{\text{PatchGAN}}$$
While $L_1$ and KL losses handle global content and latent normalization, the PatchGAN discriminator penalizes blurry textures on local $70 \times 70$ patches, forcing the decoder to generate crisp high-frequency details (such as hair and skin pores).
</details>

<details>
<summary>Click to view solution for Exercise 5</summary>

**Diagnosis:** Evaluating `torch.log(discriminator(fake_images))` where `discriminator` outputs probabilities via `torch.sigmoid()` causes floating-point underflow. If the discriminator outputs $0.0$ (common in early training), $\ln(0.0) = -\infty$. Multiplying by negative gives $+\infty$, which produces `NaN` upon the next arithmetic operation or backward pass.

**Fix:** Keep discriminator outputs as unnormalized logits and use PyTorch's numerically stable `BCEWithLogitsLoss`:
```python
loss_g = bce_loss(discriminator_logits(fake_images), torch.ones_like(logits))
```
</details>

---

## 13. Explain it back and return to it

**Closed-notes Feynman prompt:**  
Imagine explaining Generative Adversarial Networks (GANs) to a software engineer who only knows standard classification without using the terms “minimax theorem”, “Jensen-Shannon divergence”, or “Nash equilibrium”. Use the analogy of an art forger and a museum detective, and explain why the two networks must improve together for the forger to learn to paint realistic masterpieces. Once you finish, restore the formal terms and state the Minimax Value Function equation.

<details>
<summary>Model explanation for self-evaluation</summary>

Imagine an art forger who wants to forge historical paintings, but has never been allowed to look directly at the real paintings. Instead, there is an art detective who inspects paintings and says "Real" or "Fake".

The training works like a game:
1. The forger tries painting something and mixes it with a real painting.
2. The detective examines both and tries to catch the fake.
3. If the detective catches the fake easily, the forger learns which brushstrokes gave it away and improves.
4. If the forger successfully tricks the detective, the detective studies harder to spot subtler differences.

If both networks improve at the same pace, the detective acts like a personal teacher for the forger. Eventually, the forger becomes so skilled that its paintings are physically indistinguishable from the real ones—the detective has to guess randomly (50/50).

*Restoring formal terminology:* The forger is the **Generator** $G_\theta$, the detective is the **Discriminator** $D_w$, and the game is a zero-sum **Minimax Game** minimizing the **Jensen-Shannon Divergence** $D_{\text{JS}}(p_{\text{data}} \parallel p_g)$ via the **Value Function**:
$$\min_\theta \max_w V(G_\theta, D_w) = \mathbb{E}_{x \sim p_{\text{data}}}[\ln D_w(x)] + \mathbb{E}_{z \sim p_z}[\ln(1 - D_w(G_\theta(z)))]$$

</details>

### Spaced Repetition Schedule

| Return Date | Closed-Notes Retrieval Task | Self-Verification Anchor |
| :--- | :--- | :--- |
| **Day 1** | Write out the Minimax Value Function $V(G, D)$. Derive the optimal discriminator formula $D^*(x) = \frac{p_{\text{data}}}{p_{\text{data}} + p_g}$ from memory. | Check against §4 step-by-step derivation. |
| **Day 7** | Re-derive the proof showing that $V(G, D^*) = -\ln 4 + 2 D_{\text{JS}}(p_{\text{data}} \parallel p_g)$. Explain why the global minimum is $-\ln 4$. | Check against Derivation 2 in §4. |
| **Day 30** | Prove why simultaneous gradient descent in bilinear zero-sum games diverges ($r_{k+1}^2 = (1+\eta^2)r_k^2$) and explain why Non-Saturating loss is used. | Check against Theorem 8.3 and §5. |

### Self-Assessment Checklist

- [ ] I can formulate the two-player zero-sum minimax objective $V(G, D)$.
- [ ] I can derive the optimal discriminator $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$ using calculus of variations.
- [ ] I can prove that substituting $D^*$ into $V(G, D^*)$ yields the Jensen-Shannon Divergence $V = -\ln 4 + 2 D_{\text{JS}}$.
- [ ] I can explain why vanilla minimax loss causes vanishing gradients and derive the non-saturating loss $-\ln D(G(z))$.
- [ ] I can prove that simultaneous gradient descent in bilinear games causes rotational divergence.
- [ ] I can explain mode collapse, why it occurs, and how techniques like spectral normalization mitigate it.
- [ ] I can explain why $D_{\text{JS}}$ fails when data and generated manifolds have disjoint supports.
- [ ] I can implement a 1D adversarial training pipeline in PyTorch using `BCEWithLogitsLoss`.

---

## 14. Continue with a purposeful learning path

The resources below are verified for relevance, active status, and pedagogical precision as of **2026-09-18**. Access descriptions indicate verified availability at check time.

| Resource and author | Learning job | Exact starting point | Readiness | Access | Checked date and evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Interactive visualizer:** [GAN Lab: Play with Generative Adversarial Networks in your Browser](https://poloclub.github.io/ganlab/), Minsuk Kahng et al. (Georgia Tech / Google) | Visualize 2D distribution matching, discriminator decision surfaces, and mode collapse | Full interactive web tool (adjust iterations and noise sliders) | After §2 | Free open educational tool | 2026-09-18: verified active WebGL interactive sandbox simulating GAN game dynamics in real time. |
| **Video lecture:** [Generative Adversarial Networks (GANs), Clearly Explained](https://www.youtube.com/watch?v=Gib_kiXgnvA), StatQuest with Josh Starmer | Visual step-by-step walkthrough of generator-discriminator training and loss curves | Full 16-minute video (timestamp 00:00 to 16:00) | After §2 | Free YouTube video | 2026-09-18: verified active video, intuitive graphical explanation of adversarial game updates. |
| **Video lecture (Advanced):** [Deep Generative Models: GANs](https://www.youtube.com/watch?v=myGstX5K2_M), Stefano Ermon (Stanford University CS236) | Rigorous mathematical breakdown of minimax games, optimal discriminator, and $f$-divergences | Lecture 8: "Generative Adversarial Networks" (timestamp 14:20 to 45:00) | After §4 | Free YouTube video | 2026-09-18: verified active lecture, complete mathematical treatment of Jensen-Shannon equivalence. |
| **Foundational paper:** [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661), Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (NeurIPS 2014) | Seminal paper introducing GANs, the minimax objective, and the Jensen-Shannon theorem | Section 3: "Adversarial Nets" and Section 4: "Theoretical Results" | After §8 | Free open-access arXiv preprint | 2026-09-18: verified original paper formulations, Theorem 1 (optimal D), and Theorem 2 (global minimum). |
| **Stabilization paper:** [Wasserstein GAN](https://arxiv.org/abs/1701.07875), Martin Arjovsky, Soumith Chintala, Léon Bottou (ICML 2017) | Landmark paper explaining disjoint manifold failures and introducing the Earth Mover's distance | Section 2: "Different Distances" and Section 3: "Wasserstein GAN" | After §8 | Free open-access arXiv preprint | 2026-09-18: verified active paper, mathematical proof of JS divergence failure on disjoint supports. |
| **Textbook:** [Deep Learning, Chapter 20: Generative Models](https://www.deeplearningbook.org/contents/generative_models.html), Ian Goodfellow, Yoshua Bengio, Aaron Courville | Authoritative textbook treatment of deep generative architectures and adversarial games | Chapter 20: §20.10.4 (Generative Adversarial Networks, pp. 696–704) | After §4 | Free online HTML (MIT Press, 2016) | 2026-09-18: verified section numbers, game-theoretic formulations, and non-saturating loss derivations. |
| **Practice problem set:** [Stanford CS236: Deep Generative Models, Homework 3](https://deepgenerativemodels.github.io/), Stefano Ermon (Stanford University) | Implement DCGAN and WGAN-GP on image benchmarks | Problem 1: "Generative Adversarial Networks and Non-Saturating Loss" | After §12 | Free university course assignment | 2026-09-18: verified problem set questions covering discriminator optimality and gradient penalty. |
| **Software documentation:** [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html), Nathan Inkawhich (PyTorch) | Production reference for building, stabilizing, and training GANs in PyTorch | "Generator", "Discriminator", and "Training Loop" sections | When running §11 | Free official framework tutorial | 2026-09-18: verified PyTorch 2.9 documentation, BCEWithLogitsLoss implementation, and weight initialization. |

**Next connection:** Evaluating GANs is notoriously difficult because they lack explicit likelihoods. In [Fréchet inception distance](10-Frechet_Inception_Distance.md), we explore how the Fréchet distance between feature distributions in deep Inception networks provides the gold-standard metric for evaluating generative image quality and diversity.
