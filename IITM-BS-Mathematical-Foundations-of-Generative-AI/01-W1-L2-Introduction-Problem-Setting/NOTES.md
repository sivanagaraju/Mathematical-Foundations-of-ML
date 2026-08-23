# W1_L2 — Introduction & Problem Setting: Mathematical Foundations of Generative AI

> **Course:** IIT Madras B.S. Degree in Data Science & AI · **Mathematical Foundations of Generative AI**  
> **Instructor:** Prof. Prathosh A. P. (IISc / IIT Madras)  
> **Lecture Video:** [W1_L2: Introduction & Problem Setting | Generative AI Basics Explained](https://www.youtube.com/watch?v=HUunmwZfGzc&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=2) (~58:32)  
> **Prerequisites Warm-up:** [PREREQUISITES.md](./PREREQUISITES.md) · **Self-Assessment Quiz:** [quiz.html](./quiz.html)  
> **Course Catalog:** [../NOTES.md](../NOTES.md)

---

## Quick Navigation Matrix

| Topic & Timestamp | Board Focus | Core Mathematical Object | Prerequisite Link |
| :--- | :--- | :--- | :--- |
| [Topic 1: Conditional Generators in the Wild](#topic-1-conditional-generators-in-the-wild-00110453) (00:11–04:53) | Apps & Modalities | $x_{\text{new}} \sim p(x \mid c)$ | [p1-rv](./PREREQUISITES.md#p1-rv) |
| [Topic 2: Data as IID Samples from Unknown $p_x$](#topic-2-data-as-iid-samples-from-unknown-p_x-04530654) (04:53–06:54) | The Dataset | $\mathcal{D} = \{x_1, \dots, x_n\} \overset{\text{iid}}{\sim} \mathbb{P}_x$ | [p2-density](./PREREQUISITES.md#p2-density) |
| [Topic 3: Images as High-$D$ Vectors](#topic-3-images-as-high-d-vectors-06540922) (06:54–09:22) | High Dimensionality | $x_i \in \mathbb{R}^D$, $D = R \times C \times 3$ | [p4-vectors](./PREREQUISITES.md#p4-vectors) |
| [Topic 4: Vector RVs & The IID Principle](#topic-4-vector-random-variables-iid-across-samples-not-pixels-09221645) (09:22–16:45) | True Worldview | $x_i \perp x_j$ vs $x_{i,a} \not\perp x_{i,b}$ | [p3-iid](./PREREQUISITES.md#p3-iid) |
| [Topic 5: Estimate $p_x$ and Learn to Sample](#topic-5-estimate-p_x-and-learn-to-sample-16451947) (16:45–19:47) | The Formal Goal | Estimate $p_x$ $\&$ Sample $x_{\text{new}} \sim p_x$ | [p1-rv](./PREREQUISITES.md#p1-rv) |
| [Topic 6: Parametric Family $p_\theta$ via Deep Nets](#topic-6-parametric-family-p_theta-as-a-deep-neural-network-19472419) (19:47–24:19) | Parametric Family | $p_\theta$ represented by DNNs (UAT) | [p5-parametric](./PREREQUISITES.md#p5-parametric) |
| [Topic 7: Divergence Metric & The Unknown-$p_x$ Trap](#topic-7-divergence-metrics-and-the-unknown-p_x-trap-24192831) (24:19–28:31) | 3-Step Recipe | $d(p_x \parallel p_\theta) \ge 0$, $d=0 \iff p_x = p_\theta$ | [p6-divergence](./PREREQUISITES.md#p6-divergence) |
| [Topic 8: Sampling Engine $z \to G_\theta(z)$](#topic-8-the-sampling-engine-z-to-g_thetaz-28313626) (28:31–36:26) | Pushforward Engine | $z \sim \mathcal{N}(0, I_k) \implies G_\theta(z) \sim p_\theta$ | [p7-transform](./PREREQUISITES.md#p7-transform) |
| [Topic 9: Train $G_{\theta^\star}$, Then Sample from Near $p_x$](#topic-9-train-g_thetastar-then-sample-from-near-p_x-36264754) (36:26–47:54) | Training & Sampling | $\theta^\star = \arg\min_\theta d(p_x \parallel p_\theta)$ | [p8-argmin](./PREREQUISITES.md#p8-argmin) |
| [Topic 10: Four Open Questions & Lecture Recap](#topic-10-four-open-foundational-questions-and-recap-47545832) (47:54–58:32) | Course Roadmap | Q1 (Compute $d$), Q2 (Choice of $d$), Q3 ($G_\theta$), Q4 (Optimizer) | [p8-argmin](./PREREQUISITES.md#p8-argmin) |
| [External References](#external-references) | Multi-Source Study | 2–3 Videos & 2–3 Blogs/Notes per Subtopic | — |

---

## Executive Summary & Master Architecture

This lecture establishes the **foundational mathematical formulation** of generative AI. Rather than treating generative models as specialized software applications (ChatGPT, Stable Diffusion, DALL·E), the course abstracts them into a single, unified probabilistic framework:

$$\text{Given an empirical dataset } \mathcal{D} = \{x_1, \dots, x_n\} \overset{\text{iid}}{\sim} p_x \text{ in } \mathbb{R}^D, \text{ estimate } p_x \text{ and learn a sampler } x \sim p_x.$$

```
══════════════════════════════════════════════════════════════════════════════════════════════════
                            THE MASTER BLUEPRINT OF GENERATIVE AI
══════════════════════════════════════════════════════════════════════════════════════════════════

  1. REAL-WORLD PHENOMENON / APPS
     • Conditional Text / Code: Prompt c  ────────►  Response x ~ p(x | c)   (ChatGPT, Gemini)
     • Conditional Images:      Caption c ────────►  Image x ~ p(x | c)      (Stable Diffusion)
     • Text-to-Speech:          Script c  ────────►  Waveform x ~ p(x | c)   (TTS Engines)

  2. DATASET IN HIGH-DIMENSIONAL SPACE
     • Unordered sample set: D = {x₁, x₂, ..., xₙ} ⊂ ℝᴰ
     • Huge ambient dimension: D = R × C × 3  (e.g., 400 × 400 × 3 = 480,000)
     • Statistical Assumption: x_i ~ p_x IID (Across samples, NOT across pixels!)

  3. THE CORE OBJECTIVE OF GENERATIVE MODELING
     • Primary Goal: (1) Estimate unknown law p_x   AND   (2) Learn to sample from it.
     • Contrast with Discriminative ML: Discriminative models estimate p(y | x); they never mint x.

  4. THE 3-STEP GENERAL RECIPE
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ (i)   Model Selection: Assume a parametric family p_θ represented by a Deep Neural Net.│
     │ (ii)  Score Discrepancy: Define a divergence d(p_x ‖ p_θ) with d ≥ 0, d=0 iff match.   │
     │ (iii) Optimization: Solve θ* = argmin_θ d(p_x ‖ p_θ) to align p_θ with p_x.             │
     └────────────────────────────────────────────────────────────────────────────────────────┘

  5. THE PUSHFORWARD SAMPLING ENGINE
     • Tractable Prior: Draw standard normal noise z ~ N(0, I_k)  (RNG with k ≪ D)
     • Deterministic Generator: Push through neural net x̂ = G_θ(z)  (Outputs x̂ ~ p_θ)
     • After Optimization (θ = θ*): G_θ* transforms z into x_new ~ p_θ* ≈ p_x (Implicit Sampler!)

  6. THE FOUR FOUNDATIONAL OPEN QUESTIONS (Course Syllabus)
     • Q1: How do we compute d(p_x ‖ p_θ) when both densities p_x and p_θ are unknown?
     • Q2: Which divergence metric d should we choose (KL, JS, Wasserstein, f-divergence)?
     • Q3: How should we parameterize G_θ / p_θ (GAN, VAE, Diffusion, Autoregressive, SSM)?
     • Q4: How do we solve the non-convex optimization problem efficiently?
══════════════════════════════════════════════════════════════════════════════════════════════════
```

### Comparative Paradigms Matrix

| Dimension | Discriminative Modeling | Explicit Generative Modeling | Implicit Generative Modeling (This Lecture's Focus) |
| :--- | :--- | :--- | :--- |
| **Primary Goal** | Estimate conditional $p(y \mid x)$ | Estimate $p_x(x)$ analytically and evaluate likelihood | Learn a generative sampler $G_\theta(z) \sim p_x$ |
| **Density Tractability** | $p(y \mid x)$ is evaluated directly | Closed-form formula for $p_\theta(x)$ is computed | Density $p_\theta(x)$ is uncalculated / implicit |
| **Sampling Mechanism** | Not applicable (cannot generate $x$) | Often complex (MCMC, autoregressive steps) | Trivial: $x_{\text{new}} = G_\theta(z)$, $z \sim \mathcal{N}(0, I)$ |
| **Representative Models**| ResNet, BERT, Logistic Regression | PixelCNN, Normalizing Flows, VAE (ELBO) | GANs, Diffusion Models (SDE samplers) |

---

## Complete Chalkboard Algorithm in Python / PyTorch

Although Lecture 2 is delivered on a chalkboard with pure mathematical derivations, the entire framework translates directly into executable deep learning concepts:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ==============================================================================
# 1. THE DATASET & HIGH-DIMENSIONAL SPACE
# ==============================================================================
# n = number of empirical observations; D = flattened pixel dimension
# E.g., for 400x400 RGB images: D = 400 * 400 * 3 = 480,000
n_samples, D_dim = 1000, 480_000
k_latent_dim = 128  # Latent noise dimension (k << D, manifold hypothesis)

# D = {x_1, ..., x_n} ~ IID from unknown p_x (we only hold empirical samples)
# Note: Real training data is loaded from disk; here simulated abstractly:
dataset_D = torch.randn(n_samples, D_dim)

# ==============================================================================
# 2. STEP (i): PARAMETRIC FAMILY p_theta VIA DEEP NEURAL GENERATOR G_theta
# ==============================================================================
class NeuralGenerator(nn.Module):
    """Deterministic map G_theta: R^k -> R^D parameterizing distribution p_theta."""
    def __init__(self, k_dim, d_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, d_dim),
            nn.Tanh()  # Bounds output to valid normalized pixel range
        )
        
    def forward(self, z):
        return self.net(z)

G_theta = NeuralGenerator(k_dim=k_latent_dim, d_dim=D_dim)

# ==============================================================================
# 3. STEPS (ii) & (iii): DIVERGENCE MINIMIZATION (THEORETICAL ABSTRACT LOOP)
# ==============================================================================
# In practice, d(p_x || p_theta) is computed via adversarial critics or ELBOs
# theta* = argmin_theta d(p_x || p_theta)

def abstract_divergence_loss(real_batch, fake_batch):
    """Placeholder for sample-based divergence estimation (e.g., Wasserstein/JS)."""
    return torch.mean((real_batch.mean(dim=0) - fake_batch.mean(dim=0))**2)

optimizer = optim.Adam(G_theta.parameters(), lr=1e-4)

# Abstract Training Loop:
for epoch in range(1):  # Illustrated schematically
    # Sample real batch from D ~ p_x
    real_x = dataset_D[:32]
    
    # Sample known Gaussian noise z ~ N(0, I_k)
    z = torch.randn(32, k_latent_dim)
    
    # Pushforward: x_hat = G_theta(z) ~ p_theta
    fake_x = G_theta(z)
    
    # Evaluate divergence score d(p_x || p_theta)
    loss = abstract_divergence_loss(real_x, fake_x)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==============================================================================
# 4. INFERENCE / SAMPLING AFTER CONVERGENCE (G_theta*)
# ==============================================================================
G_theta.eval()

def sample_novel_data(generator, k_dim=128):
    """Draws fresh samples from near p_x without replaying dataset D."""
    with torch.no_grad():
        # (a) Draw fresh standard normal noise
        z_fresh = torch.randn(1, k_dim)
        # (b) Pass through frozen trained generator G_theta*
        x_new = generator(z_fresh)
    return x_new

# Mint a brand-new high-dimensional vector
x_generated = sample_novel_data(G_theta, k_latent_dim)
print(f"Generated Vector Shape: {x_generated.shape} (Point on p_theta* ~ p_x)")
```

---

## Topic 1: Conditional Generators in the Wild (00:11–04:53)

### Master Map Placement
This introduces the public, application-level face of generative AI before stripping away product wrappers to inspect the underlying mathematical machinery.

### Chalkboard Screenshot
![ChatGPT, Gemini, Claude as conditional text; DALL-E, Stable Diffusion as conditional image; TTS as conditional speech](./screenshots/composites/ch01-topic-01-conditional-generators-panel1of1.png)
*Figure 1.1 (~00:33–04:30):* Prof. Prathosh categorizes modern generative models into three major application modalities: (1) Conditional Text/Code (ChatGPT, Gemini, Claude), (2) Conditional Images (DALL·E, Stable Diffusion), and (3) Conditional Audio/Speech (TTS Waveform synthesizers).

### In-Depth Mathematical & Conceptual Exposition
Prof. Prathosh begins by noting that while generative models are ubiquitous in modern technology, they all share a common mathematical structure: they are **conditional generative models**.

* **Conditional Text & Code Generation:**
  Given a user prompt $c \in \mathcal{C}$ (natural language description, instructions, or incomplete code), the model generates a response sequence $x \in \mathcal{X}$ conditioned on $c$:
  $$x \sim p(x \mid c)$$
  Whether the generated text is English literature, a Python script, or SQL queries, the mathematical mapping is identical: sampling from a high-dimensional conditional distribution over token sequences.

* **Conditional Image Generation:**
  Given a textual description or caption $c$ (e.g., *"An astronaut riding a green horse on Mars"*), models such as DALL·E or Stable Diffusion generate an image tensor $x$ that satisfies the semantic constraints of $c$:
  $$x \sim p(x \mid c) \quad \text{where } x \in \mathbb{R}^{H \times W \times 3}$$

* **Conditional Speech & Audio Synthesis:**
  Given input text $c$, text-to-speech (TTS) systems synthesize continuous pressure waveforms $x \in \mathbb{R}^T$ matching the phonetic, emotional, and prosodic requirements of $c$:
  $$x \sim p(x \mid c)$$

```
  CONDITIONAL MODALITY MATRIX
  ─────────────────────────────────────────────────────────────────────────────
  Input Condition c              Generator Mechanism              Generated Sample x
  ┌────────────────────────┐     ┌──────────────────────┐        ┌─────────────┐
  │ "Write a quicksort fn" │ ──► │ LLM (p_θ(x | c))     │ ─────► │ Python Code │
  ├────────────────────────┤     ├──────────────────────┤        ├─────────────┤
  │ "A cat in a space suit"│ ──► │ Diffusion Generator  │ ─────► │ 512×512 PNG │
  ├────────────────────────┤     ├──────────────────────┤        ├─────────────┤
  │ "Welcome to IIT Madras"│ ──► │ TTS WaveNet Engine   │ ─────► │ .WAV Audio  │
  └────────────────────────┘     └──────────────────────┘        └─────────────┘
```

> [!NOTE]
> **Conditional vs. Unconditional Modeling:**  
> In conditional generation, we sample from $p(x \mid c)$. In unconditional generation, we sample directly from the joint prior $p(x)$ (e.g., generating an arbitrary photorealistic human face without any prompt). Mathematically, unconditional modeling is the foundational primitive; conditional modeling simply augments the generator with conditioning variables $c$.

### Real-World Analogy
* **The Master Restaurant Chef:** You sit at a restaurant and hand the waiter a specific order ticket $c$: *"Medium-rare steak with garlic butter."* The kitchen generator cooks a brand-new dish $x$ adhering strictly to that ticket. If you gave no ticket (unconditional), the chef would simply prepare an arbitrary dish from their entire repertoire of recipes ($p_x$).

---

## Topic 2: Data as IID Samples from Unknown $p_x$ (04:53–06:54)

### Master Map Placement
Transitions from application phenomenology to scientific formulation. What is the exact mathematical input provided to a machine learning engineer?

### Chalkboard Screenshot
![Starting point: Data D = {x1,...,xn} sampled iid from unknown distribution P_x; x_i in R^d](./screenshots/composites/ch02-topic-02-data-iid-px-panel1of1.png)
*Figure 2.1 (~05:02–06:44):* Definition of the starting point: an unordered set of $n$ data points $\mathcal{D} = \{x_1, \dots, x_n\} \sim \text{IID } \mathbb{P}_x$, where $\mathbb{P}_x$ is an unknown probability distribution over $\mathbb{R}^D$.

### In-Depth Mathematical & Conceptual Exposition
The starting point of every statistical machine learning formulation is a dataset $\mathcal{D}$:

$$\mathcal{D} = \{x_1, x_2, \dots, x_n\}$$

1. **Unordered Set:** The index $i \in \{1, \dots, n\}$ is purely an arbitrary bookkeeping label. There is no temporal or spatial sequence between file $x_1$ and file $x_2$.
2. **The Unknown Distribution ($p_x$):** The data points are generated by a true physical, natural, or linguistic process governed by a probability law $p_x$. **Crucially, we do not possess a mathematical formula or closed-form expression for $p_x$.**
3. **Notation Custom:**
   * Script letters ($\mathbb{P}_x$) denote **probability distribution functions** (cumulative measures assigning mass in $[0, 1]$ to sets).
   * Lowercase letters ($p_x$) denote **probability density functions** (continuous height functions whose integral over $\mathbb{R}^D$ is $1$).
   * Prof. Prathosh notes that throughout the course, he will use the terms "distribution" and "density" interchangeably for ease of exposition, but the continuous density $p_x$ is the primary analytical object.

```
       UNKNOWN TRUE WORLD LAW p_x (The River)
                      │
        ┌─────────────┼─────────────┐   IID Sampling Process
        ▼             ▼             ▼
      Sample 1      Sample 2      Sample n
     ┌────────┐    ┌────────┐    ┌────────┐
     │  x₁    │    │  x₂    │    │  xₙ    │  ──►  Dataset D = {x₁, ..., xₙ} ⊂ ℝᴰ
     └────────┘    └────────┘    └────────┘           (The Jar of Pebbles)
```

### Real-World Analogy
* **The Jar of River Pebbles:** You stand by an ancient, unmapped river whose currents deposit polished pebbles on the bank. You scoop 1,000 pebbles into a glass jar ($\mathcal{D}$). You have the pebbles in your hands, but you do not have a hydraulic equation describing the river's entire geology ($p_x$). Generative modeling is deducing the river's fluid dynamics from the jar of pebbles alone.

---

## Topic 3: Images as High-$D$ Vectors (06:54–09:22)

### Master Map Placement
Establishes the geometry and dimensionality of the sample space $\mathcal{X} = \mathbb{R}^D$.

### Chalkboard Screenshot
![Image represented as R x C x 3 tensor; worked example 400x400x3 = 480,000 dimensions](./screenshots/composites/ch03-topic-03-high-d-images-panel1of1.png)
*Figure 3.1 (~07:05–09:10):* Calculation of image dimensionality on the chalkboard. A color photo of resolution $R=400, C=400$ with 3 channels yields $D = 480{,}000$ dimensions.

### In-Depth Mathematical & Conceptual Exposition
While digital images are stored as 3D arrays (tensors) of size $R \times C \times 3$, analytical probability theory treats each image as an individual point (vector) in a Euclidean vector space $\mathbb{R}^D$.

* **Flattening Transformation:**
  Every spatial pixel location $(r, c)$ and color channel $k \in \{\text{Red}, \text{Green}, \text{Blue}\}$ is assigned a unique 1D index $j \in \{1, 2, \dots, D\}$:
  $$j = (r \cdot C + c) \cdot 3 + k$$
  $$x = \begin{bmatrix} x_1 & x_2 & x_3 & \dots & x_D \end{bmatrix}^T \in \mathbb{R}^D$$

* **Chalkboard Calculation:**
  For a modest $400 \times 400$ RGB image:
  $$D = 400 \times 400 \times 3 = 160{,}000 \times 3 = 480{,}000$$
  Each single photograph is a point in a **$480{,}000$-dimensional space**.

```
    3D TENSOR REPRESENTATION                    1D FLATTENED VECTOR
   ┌───────────────────────────┐               ┌───────────────────┐
   │ Red Channel   (400 × 400) │               │ x₁   (Pixel 1, R) │
   ├───────────────────────────┤  ──FLATTEN──► │ x₂   (Pixel 1, G) │
   │ Green Channel (400 × 400) │               │ x₃   (Pixel 1, B) │
   ├───────────────────────────┤               │ ...               │
   │ Blue Channel  (400 × 400) │               │ x_D  (Pixel RC, B)│
   └───────────────────────────┘               └───────────────────┘
     400 Rows × 400 Cols × 3                     Single Point in ℝ⁴⁸⁰⁰⁰⁰
```

> [!IMPORTANT]
> **The Curse of Dimensionality vs. The Manifold Hypothesis:**  
> In $\mathbb{R}^{480000}$, the volume of the space is astronomically vast ($256^{480000}$ possible discrete states). If pixel values were chosen uniformly at random, you would only ever generate white TV static noise. Real natural images occupy an infinitesimally thin, highly curved low-dimensional manifold embedded within $\mathbb{R}^D$.

---

## Topic 4: Vector Random Variables; IID Across Samples, Not Pixels (09:22–16:45)

### Master Map Placement
The core probabilistic worldview of the entire course. Distinguishes statistical independence across observations from structural dependencies within an observation.

### Chalkboard Screenshot
![Vector-valued random variable of size d; x_i ⊥ x_j across 1000 images, but pixels are dependent](./screenshots/composites/ch04-topic-04-vector-rv-iid-panel1of1.png)
*Figure 4.1 (~09:57–16:15):* Prof. Prathosh emphasizes that $x_i \perp x_j$ for $i \neq j$ across $n=1000$ images, but the dimensions within $x_i$ (pixels) are strongly correlated.

### In-Depth Mathematical & Conceptual Exposition
Prof. Prathosh delivers an extensive clarification of the IID assumption:

1. **Vector-Valued Random Variable:**
   Each data point $x_i$ is an instantiation of a vector random variable $X = [X_1, X_2, \dots, X_D]^T$ with joint density $p_X(x_1, \dots, x_D)$.

2. **Independence Across Data Points ($x_i \perp x_j$):**
   If you have $n = 1000$ photos, photo $x_1$ and photo $x_2$ were taken independently at different times and locations. Knowledge of $x_1$ provides zero statistical information about $x_2$:
   $$p(x_1, x_2, \dots, x_n) = \prod_{i=1}^n p_x(x_i)$$

3. **Heavy Dependence Across Dimensions Within a Point ($X_a \not\perp X_b$):**
   The IID assumption **never** claims that coordinate $X_1$ and coordinate $X_{1000}$ inside the same photo are independent!
   $$p_x(x_{i,1}, x_{i,2}, \dots, x_{i,D}) \neq \prod_{j=1}^D p(x_{i,j})$$
   Inside an image, adjacent pixels share textures, colors, and lighting. If pixel $(100, 100)$ is part of a blue sky, pixel $(100, 101)$ has an overwhelming probability of being blue sky.

4. **Why Assume a Single Shared Law $p_x$?**
   * **Mathematical Tractability:** Modeling one unknown distribution $p_x$ is solvable with $n$ samples. Modeling $n$ distinct distributions $p_{x_1}, \dots, p_{x_n}$ with 1 sample each is mathematically ill-posed.
   * **Internet-Scale LLMs:** For GPT-style models, the training corpus comprises billions of web pages. We treat the entire internet as samples drawn from one shared universal human text distribution $p_{\text{web}}$.

```
   DATASET MATRIX (n Samples × D Dimensions)
   ─────────────────────────────────────────────────────────────────────────────
                  Pixel 1      Pixel 2      ...      Pixel D
   Sample x₁   │   0.45   ───   0.46   ───  ...  ───   0.12   │  <── Pixels are
                   ⟂            ⟂                      ⟂             DEPENDENT!
   Sample x₂   │   0.88   ───   0.87   ───  ...  ───   0.94   │
                   ⟂            ⟂                      ⟂
      ...      │   ...          ...                    ...    │
                   ⟂            ⟂                      ⟂
   Sample xₙ   │   0.10   ───   0.11   ───  ...  ───   0.05   │
               ▲
               │
     Samples are INDEPENDENT (x_i ⟂ x_j) and drawn from the SAME law p_x.
```

### Real-World Analogy
* **Author's Library vs. Sentences in a Chapter:** J.K. Rowling writes 7 Harry Potter books. Book 1 and Book 4 can be viewed as distinct independent creative outputs from the author's mind ($p_x$). But *inside* Book 1, Chapter 2 is intricately dependent on Chapter 1, and the word "Voldemort" is followed by specific contextual grammar. IID applies across books, never across consecutive words.

---

## Topic 5: Estimate $p_x$ and Learn to Sample (16:45–19:47)

### Master Map Placement
Formal definition of the ultimate problem statement of generative modeling in machine learning.

### Chalkboard Screenshot
![Goal of Generative Modeling: Given D iid from unknown P_x, Estimate P_x and learn to sample from it](./screenshots/composites/ch05-topic-05-estimate-and-sample-panel1of1.png)
*Figure 5.1 (~16:59–19:32):* Formal definition written on the board: "Goal: Estimate $\mathbb{P}_x$ & learn to sample from it."

### In-Depth Mathematical & Conceptual Exposition
Generative modeling is formally defined as a **two-fold mathematical task**:

$$\text{Given } \mathcal{D} = \{x_1, \dots, x_n\} \overset{\text{iid}}{\sim} p_x \text{ (unknown)},$$
$$\textbf{Goal:} \quad \text{(1) Estimate } p_x \quad \text{AND} \quad \text{(2) Learn to sample } x_{\text{new}} \sim p_x.$$

```
                     THE MACHINE LEARNING FORK
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
   DISCRIMINATIVE                                  GENERATIVE
   • Data: Pairs (x_i, y_i)                        • Data: Points {x_i}
   • Goal: Estimate p(y | x)                       • Goal: Estimate p_x AND Sample!
   • E.g., Given X-ray, predict Tumor/No Tumor    • E.g., Synthesize a brand-new,
   • NEVER needs to generate a new X-ray!            never-before-seen realistic X-ray!
```

* **Explicit vs. Implicit Density Estimation:**
  * **Explicit Density Models:** Provide an analytical, evaluatable formula for $p_\theta(x)$ (e.g., PixelCNN, Normalizing Flows).
  * **Implicit Generative Models:** Never write down or compute the density value $p_\theta(x)$, but construct a sampling algorithm that outputs vectors distributed according to $p_x$ (e.g., GANs).

---

## Topic 6: Parametric Family $p_\theta$ as a Deep Neural Network (19:47–24:19)

### Master Map Placement
Recipe Step (i): Constraining the infinite search space of probability distributions to a tractable parameterized family.

### Chalkboard Screenshot
![General principle of Gen. Models: (i) Assume a parametric family on P_x, denoted by p_θ represented using DNNs](./screenshots/composites/ch06-topic-06-parametric-family-dnn-panel1of1.png)
*Figure 6.1 (~20:08–23:57):* Step (i) of the general principle: Assume a parametric family $p_\theta$ represented using Deep Neural Networks, exploiting their Universal Approximation capacity.

### In-Depth Mathematical & Conceptual Exposition
How do we search for an unknown function $p_x$ over a $480{,}000$-dimensional space? We cannot perform an exhaustive search across all mathematical functions.

* **Parametric Formulation:**
  We assume that $p_x$ can be approximated by a family of density functions indexed by a finite (but large) parameter vector $\theta \in \mathbb{R}^M$:
  $$\mathcal{P} = \{p_\theta \mid \theta \in \Theta \subseteq \mathbb{R}^M\}$$

* **Why Deep Neural Networks? (The Universal Approximation Theorem):**
  Deep neural networks possess universal approximation capabilities (Hornik et al., 1989; Cybenko, 1989). A neural network with non-linear activation functions and sufficient depth/width can approximate any Borel-measurable function to arbitrary precision $\epsilon > 0$.
  Therefore, we represent $p_\theta$ using a deep neural network whose weight tensors and biases form the parameter vector $\theta$.

```
      Infinite Space of All Densities              Parametric Neural Family {p_θ}
   ┌───────────────────────────────────┐          ┌───────────────────────────────┐
   │                                   │          │  Neural Net Architecture      │
   │           p_x (Truth)             │  ──►     │  Parameters θ = {W₁, b₁, ...} │
   │                ★                  │          │                               │
   │                                   │          │   p_θ₁ ──► p_θ₂ ──► p_θ* ≈ p_x│
   └───────────────────────────────────┘          └───────────────────────────────┘
```

> [!IMPORTANT]
> **Course Definition of "Model":**  
> Throughout this course, the term **"model"** refers specifically to the parametric distribution family $p_\theta$ (and its underlying neural network weights $\theta$), not to a complete commercial software package.

---

## Topic 7: Divergence Metrics and the Unknown-$p_x$ Trap (24:19–28:31)

### Master Map Placement
Recipe Steps (ii) and (iii): Defining a loss function between probability distributions and formulating the optimization problem.

### Chalkboard Screenshot
![Three-step principle: assume p_θ; define divergence metric d(p_θ, p_x); solve optimization over θ](./screenshots/composites/ch07-topic-07-divergence-optimize-panel1of1.png)
*Figure 7.1 (~24:39–27:00):* The complete 3-step general principle of generative modeling written across the chalkboard.

### In-Depth Mathematical & Conceptual Exposition
With a parametric family $p_\theta$ defined, we must quantify how closely $p_\theta$ matches the true distribution $p_x$.

* **Step (ii): Define a Statistical Divergence $d(p_x \parallel p_\theta)$:**
  A divergence $d(\cdot \parallel \cdot)$ is a functional that measures the statistical discrepancy between two probability laws. For the purposes of this course, it satisfies two key axioms:
  1. Non-negativity: $d(p_x \parallel p_\theta) \ge 0 \quad \forall \, p_x, p_\theta$
  2. Identity of Indiscernibles: $d(p_x \parallel p_\theta) = 0 \iff p_x = p_\theta \quad (\text{almost everywhere})$

* **Step (iii): Optimization over Parameters $\theta$:**
  We adjust the weights $\theta$ of the neural network to minimize the divergence score:
  $$\theta^\star = \arg\min_\theta \, d(p_x \parallel p_\theta)$$

```
                         THE 3-STEP GENERAL RECIPE
  ─────────────────────────────────────────────────────────────────────────────
   Step (i)    Assume parametric family p_θ via Deep Neural Network (Weights θ)
                  │
   Step (ii)   Define divergence metric d(p_x ‖ p_θ) measuring distance to truth
                  │
   Step (iii)  Solve optimization: θ* = argmin_θ d(p_x ‖ p_θ)
                  │
                  ▼
              p_θ* ≈ p_x  (Model matches reality!)
```

### The Fundamental Paradox ("Begging the Question")
> [!WARNING]
> **The Trap:** We began by establishing that $p_x$ is an **unknown** function. If we do not know $p_x$, how can we evaluate $d(p_x \parallel p_\theta) = \int p_x(x) \log \frac{p_x(x)}{p_\theta(x)} dx$?  
> **Resolution:** Prof. Prathosh acknowledges that writing $d(p_x \parallel p_\theta)$ directly appears to "beg the question." In upcoming lectures, we will develop sample-based estimation techniques (variational duals, adversarial games) that compute $d(p_x \parallel p_\theta)$ using **only finite samples** from $p_x$.

---

## Topic 8: The Sampling Engine $z \to G_\theta(z)$ (28:31–36:26)

### Master Map Placement
Explains how a deterministic neural network transforms simple noise into complex, high-dimensional probability distributions.

### Chalkboard Screenshot
![z ~ N(0, I) in R^k passed through trapezoid G_θ(z) into x̂ ~ p_θ(x̂) in R^D](./screenshots/composites/ch08-topic-08-z-gtheta-sampling-panel1of1.png)
*Figure 8.1 (~29:09–35:48):* The generator network drawn as a expanding trapezoid: Gaussian noise $z \sim \mathcal{N}(0, I_k)$ is transformed by deterministic network $G_\theta$ into high-dimensional output $\hat{x} = G_\theta(z) \sim p_\theta$.

### In-Depth Mathematical & Conceptual Exposition
How do we physically draw samples from a neural network?

1. **Known Latent Distribution ($p_z$):**
   We define an auxiliary random variable $Z \in \mathbb{R}^k$ with a simple, standard distribution that is trivial to sample on a computer (e.g., standard normal $Z \sim \mathcal{N}(0, I_k)$).

2. **Deterministic Neural Mapping ($G_\theta$):**
   Let $G_\theta: \mathbb{R}^k \to \mathbb{R}^D$ be a deterministic neural network parameterized by weights $\theta$.

3. **Transformation of Random Variables (Pushforward Measure):**
   From probability theory, applying a deterministic function $G_\theta$ to a random variable $Z$ creates a new random variable:
   $$\hat{X} = G_\theta(Z) \in \mathbb{R}^D$$
   The probability density function of $\hat{X}$, denoted $p_\theta(\hat{x})$, is entirely determined by the prior $p_z$ and the functional mapping $G_\theta$.

```
     LATENT NOISE SPACE (ℝᵏ)             GENERATOR (DNN)            DATA SPACE (ℝᴰ)
   ┌───────────────────────────┐      ┌──────────────────┐       ┌───────────────────┐
   │ z ~ N(0, I_k)             │ ──►  │ G_θ : ℝᵏ ──► ℝᴰ  │ ──►   │ x̂ = G_θ(z)        │
   │ (k-dim Standard Gaussian) │      │ (Weights θ)      │       │ x̂ ~ p_θ(x̂)        │
   └───────────────────────────┘      └──────────────────┘       └───────────────────┘
      Trivial to sample!                  Deterministic              Complex Image Law!
```

> [!TIP]
> **Implicit Sampling Power:**  
> To generate a valid sample $\hat{x} \sim p_\theta$, **we never need to write down or evaluate $p_\theta(\hat{x})$**. We simply draw $z \sim \mathcal{N}(0, I_k)$ from our random number generator and perform a single forward pass: $\hat{x} = G_\theta(z)$.

---

## Topic 9: Train $G_{\theta^\star}$, Then Sample from Near $p_x$ (36:26–47:54)

### Master Map Placement
Closing the loop: connecting divergence minimization to novel inference sampling.

### Chalkboard Screenshots
![θ* = argmin d(p_x ‖ p_θ); d≥0, d=0 iff p_x=p_θ](./screenshots/composites/ch09-topic-09-train-then-sample-panel1of2.png)
*Figure 9.1 (~37:21–41:28):* Definition of the optimal parameter set $\theta^\star = \arg\min_\theta d(p_x \parallel p_\theta)$, where divergence $d \ge 0$ and $d=0 \iff p_x = p_\theta$.

![After training, p_x is implicitly estimated by G_θ; a Gaussian z through G_θ samples from near p_x](./screenshots/composites/ch09-topic-09-train-then-sample-panel2of2.png)
*Figure 9.2 (~42:51–46:58):* Magenta trajectory on the board: passing fresh $z \sim \mathcal{N}(0, I)$ through trained $G_{\theta^\star}$ produces samples from $p_{\theta^\star} \approx p_x$, completing the generative goal.

### In-Depth Mathematical & Conceptual Exposition

1. **Optimal Parameter Convergence:**
   By optimizing $\theta$ to minimize divergence, we obtain the optimal parameter weights $\theta^\star$:
   $$\theta^\star = \arg\min_\theta \, d(p_x \parallel p_\theta) \implies p_{\theta^\star} \approx p_x$$

2. **Implicit Estimation:**
   Upon convergence, $G_{\theta^\star}$ has **implicitly estimated** $p_x$. We never constructed an explicit table or formula for $p_x(x)$; instead, the geometry of $p_x$ is encoded directly inside the synaptic weights of $G_{\theta^\star}$.

3. **Inference / Sampling Procedure:**
   To generate novel content:
   * Draw fresh noise $z \sim \mathcal{N}(0, I_k)$ (unlimited supply via RNG).
   * Evaluate $x_{\text{new}} = G_{\theta^\star}(z)$.
   * Because $p_{\theta^\star} \approx p_x$, $x_{\text{new}}$ is effectively a sample from the true data distribution $p_x$.

4. **Why $x_{\text{new}}$ is NOT a Copy of Training Data $\mathcal{D}$:**
   A properly trained generator does not memorize the training set $\mathcal{D}$. Each fresh draw $z$ lands on a different point of the continuous latent space, generating novel data points in proportion to their likelihood under $p_x$.

```
  TRAINING PHASE (Adjust θ)                     INFERENCE PHASE (Freeze θ*)
  ─────────────────────────                     ───────────────────────────
  D ~ p_x (True Data)                           z_fresh ~ N(0, I_k)
          │                                              │
          ▼                                              ▼
  d(p_x ‖ p_θ) ──► Optimize ──► θ*              x_new = G_θ*(z_fresh) ~ p_θ* ≈ p_x
                                                (Brand-new sample! Not in D!)
```

---

## Topic 10: Four Open Foundational Questions and Recap (47:54–58:32)

### Master Map Placement
The syllabus blueprint: how the remainder of the course will systematically solve the four unresolved theoretical challenges of generative AI.

### Chalkboard Screenshots
![Four questions: compute d without p_x and p_θ; choice of d; choice of G_θ; how to optimize](./screenshots/composites/ch10-topic-10-open-questions-recap-panel1of2.png)
*Figure 10.1 (~48:45–52:34):* The four open questions explicitly written on the board.

![Recap: examples, three-step recipe, G_θ samples from near p_x, then the same four questions](./screenshots/composites/ch10-topic-10-open-questions-recap-panel2of2.png)
*Figure 10.2 (~53:51–57:40):* Master recap board summarizing application examples, the 3-step recipe, implicit estimation via $G_\theta(z)$, and the four open questions.

### The Four Open Questions (The Course Syllabus)

```
══════════════════════════════════════════════════════════════════════════════════
                     THE FOUR FOUNDATIONAL QUESTIONS OF GENAI
══════════════════════════════════════════════════════════════════════════════════
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ Q1: HOW TO COMPUTE d(p_x ‖ p_θ) WITHOUT DENSITIES?                         │
  │     We hold sample clouds D and {G_θ(z)}, but neither formula. How do we   │
  │     estimate their divergence purely from empirical data points?           │
  │     ──► Solution Preview: Variational f-divergences, Duals, Discriminators │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Q2: WHICH DIVERGENCE METRIC d SHOULD BE CHOSEN?                            │
  │     KL Divergence, Jensen-Shannon, Total Variation, Wasserstein Distance?  │
  │     Different metrics enforce mode-covering vs mode-seeking behavior.      │
  │     ──► Solution Preview: f-GANs, Wasserstein GANs, Score Matching         │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Q3: HOW TO CHOOSE THE ARCHITECTURAL FORM OF G_θ / p_θ?                     │
  │     How do GANs, VAEs, Diffusion Models, Autoregressive Models (GPT),      │
  │     and State Space Models (Mamba) parameterize this mapping?              │
  │     ──► Solution Preview: Latent Variable Models, Markov Chains, Flows     │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Q4: HOW TO SOLVE THE DIVERGENCE MINIMIZATION OPTIMIZATION?                 │
  │     How do we optimize high-dimensional non-convex objectives stably?      │
  │     ──► Solution Preview: Minimax Game Dynamics, ELBO, RLHF / DPO Alignment│
  └────────────────────────────────────────────────────────────────────────────┘
══════════════════════════════════════════════════════════════════════════════════
```

### Comprehensive Closed-Book Lecture Recap

| Milestone | Core Concept Established | Mathematical Formulation |
| :--- | :--- | :--- |
| **1. Application Reality** | Modern GenAI tools are conditional generators | $x \sim p(x \mid c)$ across text, code, image, and speech |
| **2. Starting Data** | Dataset is an unordered set of IID draws | $\mathcal{D} = \{x_1, \dots, x_n\} \overset{\text{iid}}{\sim} p_x$, $x_i \in \mathbb{R}^D$ |
| **3. Vector Space** | Images are high-dimensional vectors | $D = R \times C \times 3$ (e.g., $480{,}000$ dimensions) |
| **4. IID Worldview** | Independence is across files, not pixels | $x_i \perp x_j$ for $i \neq j$; intra-vector coordinates are dependent |
| **5. The Formal Goal** | Two-fold objective | (1) Estimate $p_x$ AND (2) Learn to sample $x \sim p_x$ |
| **6. The 3-Step Recipe** | Systematic framework | (i) Family $p_\theta$ via DNN $\to$ (ii) Divergence $d$ $\to$ (iii) $\theta^\star = \arg\min d$ |
| **7. Sampling Engine** | Transformation of random variables | $z \sim \mathcal{N}(0, I_k) \implies \hat{x} = G_\theta(z) \sim p_\theta$ |
| **8. Trained Inference** | Implicit estimation & novel generation | $x_{\text{new}} = G_{\theta^\star}(z_{\text{fresh}}) \sim p_{\theta^\star} \approx p_x$ (Novel draw) |

---

## External References

> Links and curated learning materials for every subtopic covered in Lecture 2. Each entry includes 2–3 video lectures, 2–3 authoritative blog posts/notes, and key seminal papers.

### Topic 1 — Conditional Generators in the Wild
* **Video Lectures:**
  1. [MIT 6.S191 (2026): Lecture 4 — Deep Generative Modeling](https://www.youtube.com/watch?v=R8V8CbuxryI) — Alexander Amini covers modern generative architectures and conditional generation across modalities.
  2. [3Blue1Brown: How Do AI Images Actually Work? (Diffusion Models)](https://www.youtube.com/watch?v=iv-5mZ_9CPY) — Intuitive visual breakdown of conditional image synthesis.
  3. [Stanford CS25: Transformers United — Lecture 1: Introduction to Transformers](https://www.youtube.com/watch?v=XfpMkf4rD6E) — Deep dive into conditional autoregressive language and code modeling.
* **Articles, Notes & Tutorials:**
  1. [MIT 6.7960 Deep Learning Fall 2024 Lecture 14 Notes by Phillip Isola](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec14.pdf) — Formal treatment of conditional generative vision models.
  2. [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Accessible, visual explanation of conditional sequence generation.
  3. [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — Comprehensive survey of text-conditional diffusion models.

### Topic 2 — Data as IID Samples from Unknown $p_x$
* **Video Lectures:**
  1. [StatQuest: The Main Ideas Behind Probability Distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc) — Intuitive introduction to probability distributions and empirical sampling.
  2. [Harvard Stat 110: Lecture 8 — Random Variables and Their Distributions](https://www.youtube.com/watch?v=k2BB0p8byGA) — Prof. Joe Blitzstein establishes formal probability measures and RV foundations.
  3. [MIT 18.05: Introduction to Probability and Statistics](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Complete OCW foundational video module on continuous distributions.
* **Articles, Notes & Tutorials:**
  1. [Stanford CS236: Deep Generative Models — Lecture 1: Introduction](https://deepgenerativemodels.github.io/notes/introduction/) — Formal definition of datasets as finite samples from unknown data distributions $p_{\text{data}}$.
  2. [Seeing Theory: Chapter 3 — Probability Distributions (Brown University)](https://seeing-theory.brown.edu/probability-distributions/index.html) — Interactive visual walkthrough of continuous densities vs probability measures.
  3. [Terence Tao: A Review of Probability Theory](https://terrytao.wordpress.com/2010/01/01/254a-notes-0-a-review-of-probability-theory/) — Rigorous mathematical notes on sample spaces, measures, and random variables.

### Topic 3 — Images as High-$D$ Vectors
* **Video Lectures:**
  1. [3Blue1Brown: But What is a Neural Network? | Chapter 1, Deep Learning](https://www.youtube.com/watch?v=aircAruvnKk) — Visual demonstration of flattening $28 \times 28$ image grids into 784-dimensional vectors.
  2. [Stanford CS231n: Lecture 1 — Introduction to Computer Vision](https://www.youtube.com/watch?v=vT1JzLTH4G4) — Explains image tensor representation and high-dimensional pixel spaces.
  3. [Khan Academy: Thinking About Multivariable Functions & Vectors](https://www.khanacademy.org/math/multivariable-calculus/thinking-about-multivariable-function) — Intuitive foundation for reasoning about spaces $\mathbb{R}^D$ where $D > 3$.
* **Articles, Notes & Tutorials:**
  1. [Stanford CS231n: Python NumPy Tutorial for Deep Learning](https://cs231n.github.io/python-numpy-tutorial/) — Practical tensor reshaping, slicing, and flattening guide.
  2. [Christopher Olah (Colah's Blog): Visualizing Representations (MNIST & High-D Spaces)](https://colah.github.io/posts/2014-10-Visualizing-MNIST/) — Deep exploration of high-dimensional geometry and manifold projections.
  3. [Deep Learning Book (Goodfellow et al.): Chapter 2 — Linear Algebra & Vector Spaces](https://www.deeplearningbook.org/contents/linear_algebra.html) — Rigorous linear algebra reference for deep learning.

### Topic 4 — Vector RVs & IID Across Samples, Not Pixels
* **Video Lectures:**
  1. [3Blue1Brown: But What is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) — Clear explanation of IID properties and joint summation behavior.
  2. [Harvard Stat 110: Lecture 15 — Joint, Marginal, and Conditional Distributions](https://www.youtube.com/watch?v=GvhKGvgK2w8) — Covers vector-valued distributions and inter-variable dependence.
  3. [Cornell CS 6785: Advanced Machine Learning — Lecture 1: Probabilistic Modeling](https://www.youtube.com/watch?v=FBGj_B6hH9Y) — High-dimensional joint distributions and independence assumptions.
* **Articles, Notes & Tutorials:**
  1. [Stanford CS236: Autoregressive Generative Models Notes](https://deepgenerativemodels.github.io/notes/autoregressive/) — How autoregressive models explicitly model pixel-to-pixel conditional dependencies.
  2. [Distill.pub: Feature Visualization](https://distill.pub/2017/feature-visualization/) — Visual demonstration of spatial correlations and semantic hierarchies in neural representations.
  3. [Michael Nielsen: Using Neural Networks to Recognize Handwritten Digits](http://neuralnetworksanddeeplearning.com/chap1.html) — Classic tutorial on vector inputs and spatial correlations.

### Topic 5 — Estimate $p_x$ and Learn to Sample vs Discriminative
* **Video Lectures:**
  1. [Cornell CS 6785: Lecture 2 — Generative vs. Discriminative Modeling](https://www.youtube.com/watch?v=FBGj_B6hH9Y) — Direct comparison of $p(y \mid x)$ versus joint modeling of $p(x)$.
  2. [Stanford CS229: Machine Learning — Lecture 4: Generative Learning Algorithms (Andrew Ng)](https://www.youtube.com/watch?v=nt63k3bfXS0) — Classic formulation contrasting discriminative classifiers with generative density estimation.
  3. [MIT 6.S191: Lecture 4 — Generative Modeling Foundations](https://www.youtube.com/watch?v=R8V8CbuxryI) — Contrasting supervised classification with generative data synthesis.
* **Articles, Notes & Tutorials:**
  1. [Stanford CS231n (Spring 2025): Lecture 13 Slides — Generative Models](https://cs231n.stanford.edu/slides/2025/lecture_13.pdf) — Structural breakdown of supervised vs unsupervised/generative paradigms.
  2. [Stanford CS236: Generative Modeling Foundations Notes](https://deepgenerativemodels.github.io/notes/introduction/) — In-depth analysis of density estimation versus sampling capabilities.
  3. [Andrew Ng & Michael I. Jordan: On Discriminative vs. Generative Classifiers (NeurIPS Paper)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) — Seminal paper analyzing the asymptotic error rates of generative vs discriminative models.

### Topic 6 — Parametric Family $p_\theta$ via Deep Neural Networks
* **Video Lectures:**
  1. [3Blue1Brown: How Do Neural Networks Learn? | Chapter 2, Deep Learning](https://www.youtube.com/watch?v=IHZwWFHWa-w) — Visual explanation of parameters $\theta$ as knobs in high-dimensional optimization.
  2. [Lex Fridman (MIT 6.S094): Deep Learning Basics](https://www.youtube.com/watch?v=O5xeyoRL95U) — Neural network architectures as universal function approximators.
  3. [StatQuest: Neural Networks Inside the Black Box](https://www.youtube.com/watch?v=CqOfi41LfDw) — Step-by-step breakdown of how neural layers parameterize complex non-linear functions.
* **Articles, Notes & Tutorials:**
  1. [Michael Nielsen: A Visual Proof That Neural Nets Can Compute Any Function](http://neuralnetworksanddeeplearning.com/chap4.html) — Intuitive, visual geometric proof of the Universal Approximation Theorem.
  2. [Hornik, Stinchcombe, & White (1989): Multilayer Feedforward Networks are Universal Approximators](https://www.sciencedirect.com/science/article/pii/0893608089900208) — The foundational mathematical theorem proving UAT for deep networks.
  3. [Distill.pub: Why Momentum Really Works](https://distill.pub/2017/momentum/) — Deep geometric insight into parameter optimization in deep learning.

### Topic 7 — Divergence Metrics and the Unknown-$p_x$ Trap
* **Video Lectures:**
  1. [Cornell CS 6785: Lecture 10 — Variational Divergences and $f$-GANs](https://www.youtube.com/watch?v=Ml15crPldBk) — Prof. Volodymyr Kuleshov explains how to optimize arbitrary divergences using only samples.
  2. [StatQuest: Kullback-Leibler (KL) Divergence and Cross Entropy Explained](https://www.youtube.com/watch?v=SxGYPqCgJWM) — Step-by-step graphical explanation of KL divergence and statistical distance.
  3. [Alexander Amini (MIT 6.S191): Deep Learning Optimization and Loss Landscapes](https://www.youtube.com/watch?v=5tvmMX8r_OM) — Optimization principles for minimizing distributional loss metrics.
* **Articles, Notes & Tutorials:**
  1. [Lilian Weng: From GAN to WGAN (Comprehensive Divergence Guide)](https://lilianweng.github.io/posts/2017-08-20-gan/) — Masterful mathematical breakdown of KL, JS, and Wasserstein divergences.
  2. [Ferenc Huszár: How (Not) to Train Your Generative Model](https://www.inference.vc/how-not-to-train-your-generative-model-scheduled-sampling-likelihood-adversarial-training/) — Deep dive into the failure modes of maximum likelihood and the geometry of divergences.
  3. [Stanford CS236: Generative Adversarial Networks & $f$-Divergences Notes](https://deepgenerativemodels.github.io/notes/gan/) — Rigorous mathematical notes on variational divergence minimization.

### Topic 8 — The Sampling Engine $z \to G_\theta(z)$ & Transformations
* **Video Lectures:**
  1. [Stanford CS236: Lecture 7 — Implicit Generative Models and GANs (Stefano Ermon)](https://www.youtube.com/watch?v=AJVy6Z1pC-c) — Detailed derivation of the pushforward generative engine $z \to G_\theta(z)$.
  2. [MIT 6.S191: Latent Space Arithmetic and Generator Mechanics](https://www.youtube.com/watch?v=R8V8CbuxryI) — Visualizing latent interpolations and pushforward mapping in generative models.
  3. [StatQuest: Principal Component Analysis (PCA) & Latent Dimensions](https://www.youtube.com/watch?v=FgakZw6K1QQ) — Foundational understanding of low-dimensional latent subspaces.
* **Articles, Notes & Tutorials:**
  1. [Stanford CS236: Implicit Models Notes](https://deepgenerativemodels.github.io/notes/gan/) — Mathematical notes on sampling from implicit distributions without tractable densities.
  2. [Lilian Weng: Generative Adversarial Networks Primer](https://lilianweng.github.io/posts/2017-08-20-gan/) — Detailed explanation of the deterministic generator mapping $G_\theta: \mathbb{R}^k \to \mathbb{R}^D$.
  3. [David Foster: Generative Deep Learning (O'Reilly) — Latent Space Transformations](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/) — Practical tutorial on building and sampling from generative latent spaces.

### Topic 9 — Train $G_{\theta^\star}$, Then Sample from Near $p_x$
* **Video Lectures:**
  1. [3Blue1Brown: Gradient Descent, How Neural Networks Learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — Visual explanation of finding optimal parameters $\theta^\star = \arg\min_\theta \mathcal{L}(\theta)$.
  2. [Stanford CS231n: Lecture 13 — Generative Modeling in Practice](https://www.youtube.com/watch?v=5WoItGTWV54) — Complete walkthrough of training generative networks and evaluating sample quality.
  3. [MIT 6.7960: Lecture 14 — Generative Models in Computer Vision (Phillip Isola)](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/resources/mit6_7960f24_lec14_mp4/) — Rigorous video on generative optimization and implicit sampling.
* **Articles, Notes & Tutorials:**
  1. [Ian Goodfellow et al. (2014): Generative Adversarial Nets (The Foundation Paper)](https://arxiv.org/abs/1406.2661) — The landmark paper introducing sample-based implicit generative model training.
  2. [MIT 6.7960 Fall 2024 Lecture 14 Notes by Phillip Isola](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec14.pdf) — Lecture notes covering convergence of $p_\theta$ to $p_{\text{data}}$.
  3. [Stanford CS236: Learning in Implicit Models](https://deepgenerativemodels.github.io/notes/gan/) — Detailed notes on the mechanics of sampling from trained generative networks.

### Topic 10 — Four Open Foundational Questions & Course Taxonomy
* **Video Lectures:**
  1. [Stanford CS236: Lecture 1 — Course Overview & Taxonomy of Generative Models](https://www.youtube.com/watch?v=r0p639Nq_mE) — Prof. Stefano Ermon maps the entire landscape of modern deep generative models.
  2. [MIT 6.S191: Generative Modeling Taxonomy (Autoencoders, VAEs, GANs, Diffusion)](https://www.youtube.com/watch?v=R8V8CbuxryI) — Unified architectural overview answering Questions 1 through 4.
  3. [Yannic Kilcher: Deep Learning Foundations & Generative AI Overview](https://www.youtube.com/watch?v=jDe5BAsT2-Y) — Broad perspective on optimization, divergences, and generative paradigms.
* **Articles, Notes & Tutorials:**
  1. [Stanford CS236: Deep Generative Models Lecture Notes Index](https://deepgenerativemodels.github.io/notes/) — The complete syllabus covering autoregressive models, VAEs, normalizing flows, GANs, and diffusion.
  2. [Yang Song: Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) — Foundational blog post establishing score-based diffusion models as solutions to Q1–Q4.
  3. [Lilian Weng: Flow-based Deep Generative Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/) — Explores exact density evaluation via invertible neural architectures.

---

## Sources & Production Notes

* **Primary Lecture:** [W1_L2: Introduction & Problem Setting | Generative AI Basics Explained](https://www.youtube.com/watch?v=HUunmwZfGzc&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=2) · IIT Madras B.S. Degree Programme · Prof. Prathosh A. P. · Runtime: 58:32
* **Timed Audio Captions:** `raw/captions.en.timed.txt` (Cleaned ASR transcripts)
* **Composite Screenshot Panels:** `./screenshots/composites/` (High-resolution chalkboard captures per topic MM:SS)
* **Claims & Coverage Verification:** `raw/coverage-checklist.md` and `raw/coverage-receipt.md`
