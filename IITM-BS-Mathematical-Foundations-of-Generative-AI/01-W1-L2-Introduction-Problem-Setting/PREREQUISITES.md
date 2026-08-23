# Prerequisites — Foundations for Generative Modeling (W1_L2)

> **Welcome to the Mathematical Foundations of Generative AI!**  
> Before diving into [NOTES.md](./NOTES.md), make sure the core concepts in this guide are second nature.  
> This guide is designed like a masterclass tutorial: clear definitions, intuitive real-world analogies, step-by-step worked numerical examples, ASCII visualizations, runnable code snippets, common pitfalls, and quick self-check questions.

```
══════════════════════════════════════════════════════════════════════════════════
                         THE 8 PILLARS OF W1_L2
══════════════════════════════════════════════════════════════════════════════════
 §1 Random Variable vs Realization    ──► Outcomes vs Numbers in ℝ or ℝᴰ
 §2 Distribution vs Density           ──► Probability Mass vs Height Functions
 §3 The IID Assumption                ──► Independence Across Samples, NOT Pixels
 §4 Vectors & Flattening Tensors      ──► Reshaping Images ℝ^{H×W×C} into ℝᴰ
 §5 Parametric Family p_θ & DNNs      ──► Infinite Function Space ──► Knob Vector θ
 §6 Statistical Divergences d(p ‖ q)  ──► Quantifying Distance Between Distributions
 §7 Functions of Random Variables     ──► Transforming Noise z ~ N(0, I) via G_θ(z)
 §8 Optimization & argmin             ──► Finding θ* that Minimizes Divergence
══════════════════════════════════════════════════════════════════════════════════
```

---

## 1. Random Variable vs a Realized Number

<a id="p1-rv"></a>

### Why It Matters for Lecture 2
When the lecture writes $\mathcal{D} = \{x_1, x_2, \dots, x_n\}$, these data points are **not** static inert files on a hard drive. In probability theory, each $x_i$ is a concrete **realization (instantiation)** of an underlying random mechanism (a random variable $X$). Understanding this distinction prevents you from confusing the *data we happened to observe* with the *generative process that created them*.

### Formal & Intuitive Definitions

```
      Sample Space (Ω)                Measurable Space (ℝᴰ)
   ┌────────────────────┐               ┌────────────────────┐
   │  Hidden Physical   │   X(ω) : Ω ──►│  Concrete Vector   │
   │  Event / Scene (ω) │ ────────────► │  x = [0.42, ...]   │
   └────────────────────┘  Mapping Rule └────────────────────┘
```

* **Sample Space ($\Omega$):** The set of all possible underlying states of reality (e.g., all possible visual scenes that could ever be photographed).
* **Random Variable ($X$):** A deterministic **mathematical rule (function)** $X: \Omega \to \mathbb{R}^D$ that maps an abstract, unobservable physical outcome $\omega \in \Omega$ to a concrete numerical vector $x \in \mathbb{R}^D$.
* **Realization / Sample ($x$):** The specific number or vector obtained after the random experiment has been performed once (e.g., $X(\omega_{\text{today}}) = x$).

### Worked Micro-Examples

1. **Scalar Case (Rolling a Die):**
   * *Outcome $\omega$:* The physical die bounces on the table and stops with a specific face upward.
   * *Random Variable $X$:* The rule "count the number of dots on the top face."
   * *Realization $x$:* Tonight, $X(\omega) = 4$. The number $4$ is the realization; $X$ is the measurement rule.

2. **Vector Case (High-Resolution Camera):**
   * *Outcome $\omega$:* A sunset over the ocean with clouds moving in the wind.
   * *Random Variable $X$:* The camera sensor's rule that converts incoming photons into an array of $400 \times 400 \times 3$ floating-point RGB values.
   * *Realization $x_i$:* A single saved image file `sunset_01.png` containing $480{,}000$ numbers.

### Real-World Analogies

* **Analogy 1 (The Weather Barometer):** The atmosphere is chaotic, vast, and hidden (the sample space $\Omega$). A barometer on your wall is a random variable $X$—a rule that translates the atmospheric pressure state into a single number on a dial (e.g., $1013.25\text{ hPa}$). You record the dial reading (the realization $x$). You never store the entire atmosphere.
* **Analogy 2 (The Lottery Drum vs. The Ticket):** The sealed glass drum tumbling numbered balls is the hidden probabilistic experiment. The ticket printed in your hand reading "73" is tonight's realization. The rule "read the integer printed on the selected ball" is the random variable. Generative modeling is learning the physical behavior of the drum by studying millions of discarded tickets without ever opening the drum.

### Python Code Illustration

```python
import numpy as np

# 1. The Random Variable is the RULE / GENERATOR
def random_variable_rule(sample_size=3):
    """Simulates a 3-dimensional vector random variable X: Ω -> ℝ³."""
    # The internal RNG represents the physical process Ω
    return np.random.normal(loc=0.0, scale=1.0, size=sample_size)

# 2. Realizations are specific concrete numerical draws
x_1 = random_variable_rule()  # Realization 1: e.g., array([ 0.45, -1.21,  0.88])
x_2 = random_variable_rule()  # Realization 2: e.g., array([-0.12,  0.33, -0.74])

print(f"Realization 1: {x_1}")
print(f"Realization 2: {x_2}")
```

### Common Pitfalls & Traps
> [!WARNING]
> **Pitfall:** Thinking that $X$ is a number.  
> **Correction:** $X$ is a **function** (a rule). $x = X(\omega)$ is the number. Generative models learn the probability distribution that governs the *rule* $X$, not just memorizing the realized values $\{x_1, \dots, x_n\}$.

### Mini-Check
1. If a temperature sensor reads $28.4^\circ\text{C}$ at 2:00 PM, is $28.4$ the random variable or an instantiation? *(Answer: An instantiation/realization).*
2. What mathematical entity is $X$ in high-dimensional image generation? *(Answer: A vector-valued function $X: \Omega \to \mathbb{R}^D$).*

---

## 2. Probability Distribution vs Probability Density Function (PDF)

<a id="p2-density"></a>

### Why It Matters for Lecture 2
Prof. Prathosh writes script $\mathbb{P}_x$ for probability **distributions** (measures) and standard lowercase $p_x(x)$ for probability **densities**, noting that he will occasionally interchange the words for colloquial ease. Knowing the rigorous difference keeps you grounded when calculating continuous expectations and divergences.

### Formal & Intuitive Definitions

| Term | Symbol | Formal Meaning | Intuitive Meaning | Can Value Exceed 1? |
| :--- | :---: | :--- | :--- | :---: |
| **Probability Distribution** | $\mathbb{P}_x$ | A probability measure assigning numbers in $[0, 1]$ to subsets $A \subseteq \mathbb{R}^D$: $\mathbb{P}_x(A) = \int_A p_x(x) \, dx$. | The total "mass" or probability budget allocated to a region. | **No** (always $\le 1$) |
| **Probability Density Function (PDF)** | $p_x(x)$ | A non-negative function such that total volume under the curve equals 1: $\int_{\mathbb{R}^D} p_x(x) \, dx = 1$. | The "concentration" or height of probability at an infinitesimal point. | **Yes!** (Can be $> 1$, even $\to \infty$) |

```
        Density p(x)
          ▲
        4 ┼         ┌──┐   <── Height can exceed 1! (e.g., height = 4)
        3 ┼         │  │
        2 ┼         │  │       Area (Probability) = Height × Width
        1 ┼         │  │                          = 4 × 0.25 = 1.00
        0 ┴─────────┴──┴────────► x
                   0  0.25
```

### Worked Micro-Examples

1. **Uniform Distribution on a Small Interval:**
   * Consider $X \sim \text{Uniform}[0, 0.25]$.
   * To have total probability equal to 1, the density must be:
     $$p_x(x) = \frac{1}{0.25 - 0} = 4 \quad \text{for } x \in [0, 0.25]$$
   * Here, the density $p_x(0.1) = 4 > 1$.
   * The probability of a sub-interval $[0, 0.1]$ is:
     $$\mathbb{P}_x(0 \le X \le 0.1) = \int_0^{0.1} 4 \, dx = 4 \times 0.1 = 0.40 \le 1$$

2. **Standard Normal Density in 1D:**
   $$p_x(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$
   * At $x = 0$, $p_x(0) = \frac{1}{\sqrt{2\pi}} \approx 0.3989$.
   * The probability of observing *exactly* $0.000000\dots$ is $0$, but the probability of falling in an $\epsilon$-neighborhood $[-\epsilon, +\epsilon]$ is $\approx p_x(0) \times 2\epsilon$.

### Real-World Analogies

* **Analogy 1 (Jam on Toast):** The total amount of jam in your jar is fixed at $1.0$ (total probability measure $\mathbb{P}$). How thickly you spread that jam on any square centimeter of the toast is the **density** $p(x)$. In places where you spread a thick mountain of jam, the density is very high; on bare crust, it is zero. But the total jam consumed is always exactly 1 jar.
* **Analogy 2 (Population Density vs. Total Headcount):** Manhattan has an astronomical population *density* (over $28{,}000\text{ people/km}^2$). However, if you inspect a tiny $1\text{ m}^2$ phone booth, the actual headcount inside is 1 person. Density is a rate per unit volume, not an absolute count.

### Python Code Illustration

```python
import numpy as np
from scipy import stats

# Continuous distribution: Uniform[0, 0.2]
dist = stats.uniform(loc=0.0, scale=0.2)

# Density height p(x) can be greater than 1:
pdf_val = dist.pdf(0.1)
print(f"Density height p(0.1) = {pdf_val}")  # Outputs: 5.0

# Probability P(0 <= X <= 0.05) is always <= 1:
prob_val = dist.cdf(0.05) - dist.cdf(0.0)
print(f"Probability P(0 <= X <= 0.05) = {prob_val}")  # Outputs: 0.25
```

### Common Pitfalls & Traps
> [!CAUTION]
> **Trap:** Saying "The probability of this image $x$ is $p_x(x)$."  
> **Truth:** In continuous spaces ($\mathbb{R}^D$), the probability of any single exact point is strictly zero ($\mathbb{P}(X = x) = 0$). $p_x(x)$ is a **likelihood density**. We integrate $p_x(x)$ over a small volume $d^D x$ to get a probability.

### Mini-Check
1. Can a probability density function evaluate to $150.0$? *(Answer: Yes, if the probability mass is concentrated in a tiny volume of width $< 1/150$).*
2. What is unknown at the beginning of generative modeling: the dataset $\mathcal{D}$, or the density function $p_x$? *(Answer: The true continuous density function $p_x$).*

---

## 3. The IID Assumption: Across Samples, NOT Pixels

<a id="p3-iid"></a>

### Why It Matters for Lecture 2
This is the single most critical conceptual pillar emphasized by Prof. Prathosh. If you misinterpret IID, you will make the disastrous mistake of assuming pixels inside an image are independent.

```
══════════════════════════════════════════════════════════════════════════════════
                        WHERE IID APPLIES VS WHERE IT DOES NOT
══════════════════════════════════════════════════════════════════════════════════

  ACROSS SAMPLES (YES! IID IS ASSUMED):
  Image 1 (x₁)            Image 2 (x₂)                  Image n (xₙ)
  ┌───────────────┐       ┌───────────────┐             ┌───────────────┐
  │  Cat in Park  │  ⟂   │ Sunset Ocean  │  ⟂   ...   ⟂│ Vintage Car   │
  └───────────────┘       └───────────────┘             └───────────────┘
  All drawn independently from the SAME underlying world distribution p_x.

  WITHIN A SINGLE SAMPLE (NO! ABSOLUTELY NOT INDEPENDENT):
  ┌─────────────────────────────┐
  │ Pixel (100, 100): Blue Sky  │ <── Highly correlated!
  │             ▲               │     Knowing this pixel is blue makes it 99.9%
  │             │ correlated    │     likely that its neighbor is also blue sky!
  │             ▼               │
  │ Pixel (100, 101): Blue Sky  │
  └─────────────────────────────┘
══════════════════════════════════════════════════════════════════════════════════
```

### The Mathematics of IID

Let our dataset be $\mathcal{D} = \{x_1, x_2, \dots, x_n\}$ where each $x_i \in \mathbb{R}^D$.

1. **Independent (I):** The joint probability distribution of the entire dataset factorizes as a product of individual sample densities:
   $$p(x_1, x_2, \dots, x_n) = \prod_{i=1}^n p_x(x_i)$$
   Capturing photo 1 on Monday gives zero information about what photo 2 will capture on Friday.

2. **Identically Distributed (ID):** Every single sample $x_i$ is governed by the **exact same** unknown density $p_x(\cdot)$.
   $$x_i \sim p_x \quad \forall i \in \{1, \dots, n\}$$

3. **Non-Factorization Across Dimensions (Within a Sample):**
   For any single image $x_i = [x_{i,1}, x_{i,2}, \dots, x_{i,D}]^T$:
   $$p_x(x_{i,1}, x_{i,2}, \dots, x_{i,D}) \neq \prod_{j=1}^D p(x_{i,j})$$
   Pixels have intricate spatial correlations, edges, textures, and semantic structure.

### Why Assume a Single Shared $p_x$? (The Internet / GPT Perspective)
* **Mathematical Tractability:** If every image $x_i$ came from a distinct distribution $p_{x_i}$, you would have $n$ unknowns and only 1 sample per distribution—an impossible statistical estimation problem!
* **The Internet Assumption (LLMs):** For models like ChatGPT/Gemini, the training set consists of billions of documents scraped from the internet. We treat the entire internet as samples drawn from one overarching "distribution of human text and thought" $p_{\text{text}}$.

### Real-World Analogies

* **Analogy 1 (Batch of Chocolate Chip Cookies):** A master baker mixes a giant bowl of cookie dough (the underlying distribution $p_x$). Each cookie scooped out and baked on the tray is an independent sample $x_i$ from the same recipe. Pulling cookie #3 doesn't alter the batter of cookie #4 (Independent & Identical). But *inside* cookie #3, the chocolate chips and flour are bound together in rich physical clumps—the ingredients inside one cookie are definitely not independent!
* **Analogy 2 (Music Playlist):** A radio station plays classical music all day ($p_x$). Song 1 and Song 50 are separate independent tracks broadcast under the classical format. Inside Song 1, however, note #100 heavily depends on note #99 (melody and harmony).

### Python Code Illustration

```python
import numpy as np

# 1. Independent samples across the dataset (IID draws)
n_samples = 1000
D = 2  # 2D toy vector for visualization

# True distribution: 2D Gaussian with strong correlation between dimension 1 and 2
mean = [0, 0]
cov = [[1.0, 0.85],   # Covariance = 0.85 indicates strong inter-pixel dependence!
       [0.85, 1.0]]

# Draw n IID samples:
dataset = np.random.multivariate_normal(mean, cov, size=n_samples)

# Sample 0 and Sample 1 are statistically independent:
corr_across_samples = np.corrcoef(dataset[0], dataset[1])[0, 1]

# Dimension 1 and Dimension 2 within the same sample are strongly dependent:
corr_within_sample = np.corrcoef(dataset[:, 0], dataset[:, 1])[0, 1]

print(f"Correlation within vector dimensions (pixels): {corr_within_sample:.3f} (High!)")
```

### Common Pitfalls & Traps
> [!IMPORTANT]
> Never assume pixels are independent. If pixels were independent, an image would look like random TV static noise. Generative modeling is all about learning the complex **joint dependency** between all $D$ dimensions.

### Mini-Check
1. If an image contains an eye at coordinates $(120, 150)$, does that make it more likely to find another eye at $(120, 210)$? *(Answer: Yes, because of spatial correlation within the vector RV).*
2. Does IID claim that the eye coordinates are independent? *(Answer: No, IID only says photo 1 and photo 2 are independent).*

---

## 4. Vectors & Flattening Tensors into High-Dimensional Space $\mathbb{R}^D$

<a id="p4-vectors"></a>

### Why It Matters for Lecture 2
Computers store images as 3D arrays (Height $\times$ Width $\times$ Channels), but the mathematical equations of probability and machine learning operate on vectors in Euclidean space $\mathbb{R}^D$. Prof. Prathosh emphasizes the flattening step where $D = R \times C \times 3$.

### Stacking Mechanics & Dimensions

```
   3D Image Tensor (H × W × C)              1D Flattened Vector x ∈ ℝᴰ
  ┌───────────────────────────┐
  │ Channel 1: Red   (R × C)  │
  ├───────────────────────────┤             ┌─────────────────────────┐
  │ Channel 2: Green (R × C)  │ ──FLATTEN─► │ x₁  (Pixel 1, Red)      │
  ├───────────────────────────┤             │ x₂  (Pixel 1, Green)    │
  │ Channel 3: Blue  (R × C)  │             │ ...                     │
  └───────────────────────────┘             │ x_D (Pixel RC, Blue)    │
      Height R, Width C                     └─────────────────────────┘
                                                Total Length D = R·C·3
```

### Worked Micro-Calculations

| Image Type | Dimensions ($H \times W \times C$) | Math Calculation | Total Dimensionality $D$ |
| :--- | :---: | :--- | :---: |
| **MNIST Digit** | $28 \times 28 \times 1$ (Grayscale) | $28 \times 28 \times 1$ | **784** |
| **CIFAR-10** | $32 \times 32 \times 3$ (RGB) | $32 \times 32 \times 3$ | **3,072** |
| **Lecture Example** | $400 \times 400 \times 3$ (RGB) | $400 \times 400 \times 3 = 160{,}000 \times 3$ | **480,000** |
| **Full HD 1080p** | $1080 \times 1920 \times 3$ (RGB) | $1080 \times 1920 \times 3$ | **6,220,800** |
| **4K Ultra HD** | $2160 \times 3840 \times 3$ (RGB) | $2160 \times 3840 \times 3$ | **24,883,200** |

One single standard $400 \times 400$ color photo is a single point sitting in a **$480{,}000$-dimensional space**.

### Real-World Analogies

* **Analogy 1 (Packing a Multi-Compartment Suitcase):** You have a suitcase with 3 separate layers (shirts, pants, accessories), organized in neat rows and columns. When the airline scales weigh your bag, they don't care about the 2D grid of folded clothes; they collapse everything into a single total weight reading. Flattening unrolls all compartments into a single continuous strip of luggage on a linear conveyor belt.
* **Analogy 2 (Reading a Book):** A page of text is a 2D rectangle of lines and words. When reading aloud or processing tokens, you read linearly left-to-right, line-by-line, converting a 2D page into a 1D sequence of characters.

### Python Code Illustration

```python
import torch

# Create a batch of 4 RGB images: Batch x Channels x Height x Width
B, C, H, W = 4, 3, 400, 400
image_batch = torch.randn(B, C, H, W)

# Flatten each image into a 1D vector of length D = C * H * W
# Shape becomes: [Batch, D] = [4, 480000]
flattened_vectors = image_batch.view(B, -1)

print(f"Original Tensor Shape:   {list(image_batch.shape)}")
print(f"Flattened Vector Shape: {list(flattened_vectors.shape)}")
print(f"Dimensionality D:        {flattened_vectors.shape[1]:,}")
```

### Common Pitfalls & Traps
> [!NOTE]
> Flattening discards the explicit 2D grid index, but **it loses no information**. Any neural network with sufficient capacity can learn the spatial relationships between coordinate index $j$ and coordinate index $j+1$.

### Mini-Check
1. If an audio clip has $16{,}000$ samples per second and lasts $5$ seconds (mono), what is $D$? *(Answer: $D = 16{,}000 \times 5 = 80{,}000$).*
2. If your dataset contains $5{,}000$ images of size $400 \times 400 \times 3$, what is $n$ and what is $D$? *(Answer: $n = 5{,}000$ (number of samples), $D = 480{,}000$ (dimension of each sample)).*

---

## 5. Parametric Family $p_\theta$ & Neural Network Representation

<a id="p5-parametric"></a>

### Why It Matters for Lecture 2
We cannot search the space of "all possible mathematical functions on $\mathbb{R}^{480000}$"—that space is unconstrained and infinite. Instead, we constrain our search to a **parametric family** $\{p_\theta \mid \theta \in \Theta\}$ governed by adjustable parameters (weights) $\theta$.

### What is a Parametric Family?

```
      Infinite Function Space                   Parametric Family {p_θ}
   ┌───────────────────────────────┐          ┌───────────────────────────┐
   │  All arbitrary mathematical   │          │ Candidate laws indexed by │
   │  densities on ℝᴰ              │ ──PICK─► │ parameter vector θ ∈ ℝᴹ   │
   │  (Untractable to search)      │          │ (e.g., Deep Neural Net)   │
   └───────────────────────────────┘          └─────────────┬─────────────┘
                                                            │
                                                     Tuning knobs θ
                                                            ▼
                                              p_θ₁   ──►  p_θ₂   ──►  p_θ*
```

1. **Simple 1D Gaussian Family:**
   * Knobs: $\theta = (\mu, \sigma) \in \mathbb{R} \times \mathbb{R}^+$.
   * Changing $\mu$ slides the center; changing $\sigma$ adjusts the spread.
   * Model capacity is low: cannot model multi-modal data (e.g., cats AND dogs).

2. **Deep Neural Network Family:**
   * Knobs: $\theta = \{W_1, b_1, W_2, b_2, \dots, W_L, b_L\}$, where $M = |\theta|$ can be millions or billions of parameters.
   * **Universal Approximation Theorem (UAT):** A neural network with sufficient width/depth can approximate any continuous function to arbitrary precision.
   * Therefore, parameterizing $p_\theta$ via a deep net allows us to represent unimaginably complex, high-dimensional probability distributions.

> [!IMPORTANT]
> **Course Slang Alert:** Whenever Prof. Prathosh says the word **"model"**, he specifically means the parameterized candidate density $p_\theta$, not a deployed software application.

### Real-World Analogies

* **Analogy 1 (The Mixing Console):** The true song played by an orchestra is $p_x$. You have a massive audio mixing console with 10,000 knobs and sliders ($\theta$). Each distinct setting of the knobs produces a different audio output ($p_\theta$). Your job as the sound engineer is to twist the knobs until your output sounds indistinguishable from the orchestra.
* **Analogy 2 (Sculpting Armature):** An artist uses an adjustable metal wire armature with hundreds of joints ($\theta$). By bending the joints, the armature can take the pose of a human, a horse, or a dragon. The armature design is the parametric family; the specific joint angles are $\theta$.

### Python Code Illustration

```python
import torch
import torch.nn as nn

# A simple parametric neural network representing G_theta
class GeneratorNetwork(nn.Module):
    def __init__(self, latent_dim=128, data_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, data_dim),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.net(z)

# Instantiate the parametric family
model = GeneratorNetwork(latent_dim=128, data_dim=784)

# Count total parameters (knobs in theta)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameter knobs in θ: {total_params:,}")
```

### Mini-Check
1. If we change the weights $\theta$ of our neural network, does the true data distribution $p_x$ change? *(Answer: No, $p_x$ is fixed by reality; only our candidate model $p_\theta$ changes).*
2. What theoretical property of neural networks justifies using them to parameterize $p_\theta$? *(Answer: The Universal Approximation Theorem).*

---

## 6. Statistical Divergences $d(p \parallel q)$ as Distance-Like Metrics

<a id="p6-divergence"></a>

### Why It Matters for Lecture 2
Once we have a candidate family $p_\theta$, how do we know if a given knob setting $\theta$ is good or bad? We need a quantitative score that measures the discrepancy between the true law $p_x$ and our candidate law $p_\theta$. This score is a **statistical divergence** $d(p_x \parallel p_\theta)$.

### Formal Properties of Divergences

```
     True Density p_x             Model Density p_θ          Divergence d(p_x ‖ p_θ)
   ▲                            ▲                          ▲
   │    ┌───┐                   │            ┌───┐         │   d >> 0 (Large penalty!)
   │   ┌┘   └┐                  │           ┌┘   └┐        │
   └───┴─────┴───► x            └───────────┴─────┴───► x  └──────────────────────►
         Region A                              Region B

                                  AFTER TRAINING (θ = θ*):
   ▲
   │    ┌───┐   <── p_x and p_θ* OVERLAP PERFECTLY!
   │   ┌┘   └┐
   └───┴─────┴───► x            ═════════════════════════►  d(p_x ‖ p_θ*) = 0
```

1. **Non-Negativity:**
   $$d(p \parallel q) \ge 0 \quad \forall \, p, q$$
2. **Identity of Indiscernibles:**
   $$d(p \parallel q) = 0 \iff p = q \quad (\text{almost everywhere})$$
3. **Asymmetry (Why it's a Divergence, not a Metric):**
   In general, $d(p \parallel q) \neq d(q \parallel p)$, and it may not satisfy the triangle inequality. Hence we write $d(p \parallel q)$ with parallel vertical bars rather than a metric distance $D(p, q)$.

### Worked Micro-Example (Discrete Divergence)

Suppose $x \in \{0, 1\}$ (coin flip):
* True law $p_x$: $P(X=1) = 0.8, P(X=0) = 0.2$.
* Model law $p_\theta$: $P(X=1) = 0.5, P(X=0) = 0.5$.
* Kullback-Leibler (KL) Divergence:
  $$D_{\text{KL}}(p_x \parallel p_\theta) = \sum_{x \in \{0,1\}} p_x(x) \log\left(\frac{p_x(x)}{p_\theta(x)}\right) = 0.8 \log\left(\frac{0.8}{0.5}\right) + 0.2 \log\left(\frac{0.2}{0.5}\right) \approx 0.1927 > 0$$
* If $p_\theta = p_x$, then $\log(1) = 0$, so $D_{\text{KL}} = 0$.

### Real-World Analogies

* **Analogy 1 (Earth Mover's Shovel Work):** Imagine $p_x$ is a mound of dirt on the ground and $p_\theta$ is a hole shaped elsewhere. The divergence is the minimum amount of physical shovel work (mass of dirt $\times$ distance moved) required to reshape mound $p_\theta$ until it matches mound $p_x$. The shovel work is 0 if and only if the piles already match.
* **Analogy 2 (Hot and Cold Game):** When tuning a musical instrument, the divergence is how out-of-tune the string sounds compared to a reference tuning fork. Zero dissonance means perfect pitch match.

### The Fundamental Paradox (Begging the Question)
> [!WARNING]
> **The Unknown-$p_x$ Trap:** To compute $d(p_x \parallel p_\theta) = \int p_x(x) \log \frac{p_x(x)}{p_\theta(x)} dx$, it seems we must know the exact formula for $p_x(x)$. But $p_x$ is unknown!  
> **Course Resolution:** In later lectures, we will discover ingenious mathematical transformations (variational bounds, dual formulations, discriminators) that allow us to evaluate and minimize $d(p_x \parallel p_\theta)$ using **only finite samples** from $p_x$.

### Mini-Check
1. If $d(p_x \parallel p_\theta) = 0$, what does that tell us about $p_\theta$? *(Answer: $p_\theta$ is identical to the true distribution $p_x$).*
2. Why can't we just evaluate the formula for $d(p_x \parallel p_\theta)$ directly on Day 1? *(Answer: Because $p_x$ is unknown; we only have empirical sample points $\mathcal{D}$).*

---

## 7. Functions of Random Variables & The Pushforward Engine

<a id="p7-transform"></a>

### Why It Matters for Lecture 2
How does a neural network generate new data? It uses the fundamental theorem of transformations of random variables: if $Z$ is a random variable and $G$ is a fixed deterministic function, then $X = G(Z)$ is **also a random variable** with a new induced probability distribution!

```
══════════════════════════════════════════════════════════════════════════════════
                       THE PUSHFORWARD SAMPLING ENGINE
══════════════════════════════════════════════════════════════════════════════════

   Known Simple Noise                    Deterministic Net                 Generated Data
       z ∈ ℝᵏ                                G_θ : ℝᵏ ──► ℝᴰ                  x̂ ∈ ℝᴰ
  ┌──────────────────┐                     ┌──────────────────┐            ┌──────────────────┐
  │ z ~ N(0, I_k)    │ ──────────────────► │ Deep Neural Net  │ ─────────► │ x̂ ~ p_θ(x̂)       │
  │ (Standard Normal)│    Forward Pass     │ (Weights θ)      │            │ (Complex Image)  │
  └──────────────────┘                     └──────────────────┘            └──────────────────┘
    Easy to Sample!                           Deterministic                   Rich Distribution
    (np.random.randn)                          Transformation                  in High-D Space
══════════════════════════════════════════════════════════════════════════════════
```

### Mathematical Principle

1. We define a latent noise variable $Z \in \mathbb{R}^k$ with a **known, tractable distribution** (e.g., standard Gaussian $Z \sim \mathcal{N}(0, I_k)$).
2. We construct a deterministic neural network $G_\theta: \mathbb{R}^k \to \mathbb{R}^D$, parameterized by weights $\theta$.
3. The output vector $\hat{X} = G_\theta(Z) \in \mathbb{R}^D$ is a random variable whose induced distribution is denoted by $p_\theta$.
4. **Key Insight:** To draw a sample from $p_\theta$, **we never need an explicit formula for the density $p_\theta(x)$**. We simply draw $z \sim \mathcal{N}(0, I_k)$ and compute $G_\theta(z)$!

### Worked Micro-Example (1D Transformation)

* Let $Z \sim \text{Uniform}[0, 1]$ (simple uniform noise).
* Let deterministic function $G(z) = 3z + 2$.
* Output $\hat{X} = G(Z)$ is a new random variable uniformly distributed over $[2, 5]$.
* By changing the function $G$, we changed the support, mean, and variance of the output distribution without changing our random number generator!

### Dimensionality Nuance: $k$ vs $D$
* Latent dimension $k$ (e.g., $k = 128$ or $k = 512$).
* Data dimension $D$ (e.g., $D = 480{,}000$).
* $k$ is usually **much smaller** than $D$ ($k \ll D$). This embeds the **manifold hypothesis**: natural images lie on a low-dimensional curved manifold embedded inside high-dimensional pixel space.

### Real-World Analogies

* **Analogy 1 (The Pasta Extruder):** You have a hopper filled with standardized, plain dough ($z \sim \mathcal{N}(0, I)$). The dough passes through a shaped metal die ($G_\theta$). If the die is shaped like a star, star pasta comes out; if shaped like a tube, rigatoni comes out. Changing the die ($G_\theta$) changes the shape of the output ($\hat{x}$) without needing a different dough hopper!
* **Analogy 2 (The Optical Prism):** Simple, uniform white light ($Z$) enters a finely cut glass prism ($G_\theta$). The deterministic refractive geometry of the prism bends and separates the light into a rich, structured rainbow spectrum of colors ($\hat{X}$).

### Python Code Illustration

```python
import torch
import torch.nn as nn

# 1. Define generator mapping low-D noise (k=16) to high-D vector (D=784)
k_dim, D_dim = 16, 784
G_theta = nn.Sequential(
    nn.Linear(k_dim, 128),
    nn.ReLU(),
    nn.Linear(128, D_dim)
)

# 2. Step 1: Draw known standard normal noise z ~ N(0, I)
z_sample = torch.randn(1, k_dim)  # Unlimited RNG

# 3. Step 2: Pushforward through deterministic G_theta
with torch.no_grad():
    x_hat = G_theta(z_sample)  # Shape: [1, 784]

print(f"Generated sample x_hat shape: {x_hat.shape}")
print(f"Sample drawn from p_theta successfully without knowing formula for p_theta!")
```

### Mini-Check
1. If $G_\theta$ is fixed and we feed it the exact same noise vector $z_0$ twice, do we get different outputs? *(Answer: No, $G_\theta$ is a deterministic function; identical input yields identical output).*
2. Does the latent dimension $k$ have to equal the image dimension $D$? *(Answer: No, typically $k \ll D$).*

---

## 8. Optimization & The $\arg\min$ Formulation

<a id="p8-argmin"></a>

### Why It Matters for Lecture 2
The culmination of Lecture 2 is the formalization of generative model training as an optimization problem:
$$\theta^\star = \arg\min_\theta \, d(p_x \parallel p_\theta)$$
Understanding the difference between $\min$ (the smallest value) and $\arg\min$ (the knob setting that achieves that value) is essential.

### $\min$ vs $\arg\min$

```
          Loss d(p_x ‖ p_θ)
            ▲
            │        \               /
            │         \             /
            │          \   p_θ*    /
  min d ──► │───────────\───*───/────────────  <── Minimum divergence value (~0)
            │            \     /
            │             \___/
            └───────────────┬────────────────► Parameter θ
                            ▲
                            │
                       θ* = argmin d
               (The optimal weights we save!)
```

* **$\min_\theta f(\theta)$:** Returns the **minimum value** of the function (e.g., $\min_\theta (\theta - 3)^2 = 0$).
* **$\arg\min_\theta f(\theta)$:** Returns the **argument (parameter setting)** that achieves that minimum (e.g., $\arg\min_\theta (\theta - 3)^2 = 3$).
* In deep learning, $\theta^\star$ represents the **saved checkpoint file (trained model weights)**.

### The Complete End-to-End Training & Sampling Loop

```python
# ==============================================================================
# THE COMPLETE GENERATIVE AI BLUEPRINT (CHALKBOARD ALGORITHM IN PYTHON)
# ==============================================================================
import torch

# 1. GIVEN: Training data D = {x_1, ..., x_n} drawn iid from unknown p_x
# (We only have samples, no mathematical formula for p_x)

# 2. STEP (i) - MODEL: Neural Generator G_theta : R^k -> R^D
# latent_z ~ N(0, I_k) ---> x_hat = G_theta(z) ~ p_theta

# 3. STEP (ii) & (iii) - TRAIN: Find theta* = argmin_theta d(p_x || p_theta)
# (In practice, optimized via stochastic gradient descent on mini-batches)
# theta_star = optimizer_loop(data_samples, G_theta)

# 4. INFERENCE / SAMPLING (After training, freeze G_theta_star):
def sample_new_image(G_theta_star, k_dim=128):
    """Mints a brand-new sample from near p_x."""
    # (a) Draw fresh standard normal noise
    z_fresh = torch.randn(1, k_dim)
    
    # (b) Pass through trained generator
    with torch.no_grad():
        x_new = G_theta_star(z_fresh)
        
    # (c) x_new is a fresh draw from p_theta* ~ p_x (NOT a copy of training set!)
    return x_new
```

### Real-World Analogies

* **Analogy 1 (Focusing a Camera Lens):** The divergence $d$ is the blurriness of the image on the camera sensor. The ring on the lens is $\theta$. $\arg\min$ is the exact physical position $\theta^\star$ of the focus ring where blurriness is zero. Once locked at $\theta^\star$, you leave the ring in place and capture crisp photos all day.
* **Analogy 2 (Dialing a Safe Combination):** Turning the dials until the lock opens. The winning sequence of numbers is $\theta^\star$.

### Common Pitfalls & Traps
> [!CAUTION]
> **Trap:** Believing that a successful generative model simply looks up images in its training set $\mathcal{D}$.  
> **Truth:** Memorization is a failure mode (overfitting). A properly trained generator $G_{\theta^\star}$ samples smoothly from the continuous high-dimensional probability landscape $p_x$, generating plausible instances that **never existed in the training dataset**.

### Mini-Check
1. After training is complete, do we continue updating $\theta$? *(Answer: No, $\theta^\star$ is frozen; we only sample fresh $z$).*
2. Why is generating an exact duplicate of a training image considered a flaw rather than a success? *(Answer: Because the goal is to model the general distribution $p_x$ and sample novel data, not memorize finite points).*

---

## Prerequisite Mastery Matrix

Check off each item before starting [NOTES.md](./NOTES.md):

- [ ] I can clearly explain why an image is a realization of a vector RV $X: \Omega \to \mathbb{R}^D$.
- [ ] I know why a probability density $p(x)$ can exceed 1 while total probability $\mathbb{P}$ cannot.
- [ ] I understand that IID applies **across photos**, while pixels within a photo are **heavily dependent**.
- [ ] I can calculate $D = H \times W \times C$ for any image tensor.
- [ ] I understand why neural networks are used to represent parametric families $p_\theta$ (Universal Approximation).
- [ ] I know that $d(p_x \parallel p_\theta) \ge 0$ and equals $0$ if and only if $p_\theta = p_x$.
- [ ] I can explain how pushing $z \sim \mathcal{N}(0, I)$ through $G_\theta(z)$ creates samples from $p_\theta$ without a closed-form formula for $p_\theta(x)$.
- [ ] I understand the complete 3-step recipe: Family $\to$ Divergence $\to \arg\min_\theta$.

---

**You are ready! Proceed to [NOTES.md](./NOTES.md).**
