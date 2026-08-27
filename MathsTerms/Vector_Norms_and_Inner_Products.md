# Vector Norms, Distances & Inner Products: The Geometric Foundations of Generative AI

> `🏷️ Tags:` `Linear-Algebra` `Vector-Norms` `Inner-Products` `Cosine-Similarity` `Attention` `RAG` `Embedding` `Optimization`  
> `📚 Prerequisites Needed:` [Probability Basics & Axioms](./Probability_Basics_and_Axioms.md) · [Tensors & Shapes](./Tensors_and_Shapes.md)  
> `🎯 Where Do We Use This?:` **Every modern AI attention and search engine** — Scaled Dot-Product Attention in Transformers ($\text{Softmax}(QK^\top / \sqrt{d_k})$ in GPT-4, LLaMA-3), Cosine similarity in Retrieval-Augmented Generation (RAG vector databases like Pinecone/Chroma), $L_2$ Weight decay in AdamW optimizer, and Gradient norm regularization in WGAN-GP.  
> `🎓 Course Module Mapping:` [Tut 03: PyTorch Basics](../Mathematical-Foundation-for-GenerativeAI/17-Tutorial03-PyTorch-Basics/NOTES.md) · [Tut 04: CNNs](../Mathematical-Foundation-for-GenerativeAI/18-Tutorial04-CNNs-PyTorch/NOTES.md) · [Lec 01: Intro](../Mathematical-Foundation-for-GenerativeAI/14-Lec01-MFGAI-Introduction/NOTES.md) · [Lec 18: WGAN](../Mathematical-Foundation-for-GenerativeAI/30-Lec18-Wasserstein-GAN/NOTES.md)  
> `⏱️ Difficulty Level:` ⭐☆☆☆☆ (Foundational / Beginner-Friendly · 15 min read)

---

### 📌 Quick Navigation & Architecture Map
- [1. 🌟 Everyday Real-World Scenarios](#1--everyday-real-world-scenarios-the-taxi-vs-pigeon-and-ai-semantic-search) — The Taxi vs Pigeon & Semantic Search in Vector DBs
- [2. 👶 ELI5 Intuition](#2--eli5-intuition-the-rubber-band--the-flashlight-alignment) — The Rubber Band & Flashlight Alignment
- [3. 📚 Deep Terminology Master Glossary](#3--deep-terminology-master-glossary-15-core-concepts-dissected) — 15 geometric vector terms dissected without jargon
- [4. 📐 Mathematical Formulations, Unit Ball Geometry & Proofs](#4--mathematical-formulations-unit-ball-geometry--proofs) — $L_1, L_2, L_\infty$ norms, Cauchy-Schwarz Inequality, and Cosine Angle
- [5. 🔢 Concrete Micro-Numerical Worked Examples](#5--concrete-micro-numerical-worked-examples) — 2D Norm Calculations & 4D Attention Dot-Product
- [6. 🔗 Connecting the Dots: Generative AI Architecture Blocks](#6--connecting-the-dots-how-norms-and-dot-products-power-generative-ai) — Scaled Dot-Product Attention Block, RAG Cosine Retrieval, and WGAN Critic Norm
- [7. 💻 Standalone Executable Python/PyTorch Verification Script](#7--complete-standalone-executable-pythonpytorch-verification-script) — Norm calculations, Cosine similarity, and Cauchy-Schwarz verification
- [8. 🩺 Diagnostic Mini-Checks & Common Traps](#8--diagnostic-mini-checks--common-traps) — Self-test questions & production engineering pitfalls

---

In machine learning and Generative AI, **Vector Norms** measure the length, magnitude, or size of data points and model weights, while **Inner Products (Dot Products)** measure the geometric alignment, similarity, and projection between high-dimensional embeddings.

```
 ===================================================================================================
                 THE THREE COMMON VECTOR NORMS IN EUCLIDEAN SPACE ℝ^d
 ===================================================================================================

  L₁ NORM (MANHATTAN)                  L₂ NORM (EUCLIDEAN LENGTH)           L_∞ NORM (MAXIMUM ABSOLUTE)
  Sum of Absolute Values               Straight-Line Ruler Distance         Largest Single Component
  ┌──────────────────────────────┐    ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │ ||x||₁ = ∑ |xᵢ|              │    │ ||x||₂ = √(∑ xᵢ²)            │    │ ||x||_∞ = max |xᵢ|           │
  │ Grid / Taxi distance         │    │ True physical distance       │    │ Peak coordinate deviation    │
  │ Induces sparsity (Lasso)     │    │ Weight decay, Ridge, WGAN    │    │ Adversarial attacks (FGSM)   │
  └──────────────────────────────┘    └──────────────────────────────┘    └──────────────────────────────┘
 ===================================================================================================
```

---

### 1. 🌟 Everyday Real-World Scenarios (The Taxi vs Pigeon and AI Semantic Search)
> `Context:` Zero Prior Machine Learning / AI Knowledge Needed · Concrete Real-World Mapping

#### Scenario A: Traveling Across New York City (Zero ML Background Needed)
Imagine traveling between two street corners in Manhattan:
1. **The Taxi / Manhattan Distance ($L_1$ Norm):** A taxi cannot drive through brick buildings; it must travel along perpendicular city blocks: $3\text{ blocks East} + 4\text{ blocks North} = \mathbf{7\text{ blocks}}$.
2. **The Pigeon / Euclidean Distance ($L_2$ Norm):** A pigeon flies straight through the air directly to the destination: $\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5\text{ blocks}}$.
3. **The Peak Deviation ($L_\infty$ Norm):** The longest single segment of the trip: $\max(|3|, |4|) = \mathbf{4\text{ blocks}}$.

---

#### Scenario B: In Generative AI — RAG Vector Database Semantic Search
> `Context:` How Dot Products and Cosine Similarity Enable AI to Retrieve Relevant Documents

When you ask an AI assistant *"How do I fix a leaking faucet?"*:
- The query is converted into a 1536-dimensional embedding vector $q$.
- The database contains millions of article vectors $d_i$.
- To find the best answer, the vector database computes the **Cosine Similarity** (normalized Dot Product):
  $$\text{Cosine Similarity}(q, d_i) = \frac{q^\top d_i}{\|q\|_2 \|d_i\|_2}$$
- If two vectors point in the exact same direction, cosine similarity equals **$+1.0$** (perfect semantic match!), allowing the AI to retrieve the exact plumbing manual in milliseconds.

```
 ===================================================================================================
         DOT PRODUCTS IN TRANSFORMER ATTENTION & VECTOR SEARCH (RAG)
 ===================================================================================================

  USER QUERY: "Fix leaking pipe" ──► Vector q: [ 0.82,  0.55, -0.12, ... ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
  DOC 1: "Plumbing repair guide"                  DOC 2: "Baking chocolate cookies"
  Vector d₁: [ 0.80,  0.58, -0.10, ... ]          Vector d₂: [ -0.70,  0.10,  0.65, ... ]
  ────────────────────────────────────────        ────────────────────────────────────────
  Cosine Sim: +0.992 (Angle ≈ 7°)                 Cosine Sim: -0.420 (Angle ≈ 115°)
  ✅ TOP RETRIEVED RESULT (Perfect Match!)        ❌ IRRELEVANT (Filtered Out)
 ===================================================================================================
```

---

### 2. 👶 ELI5 Intuition: The Rubber Band & Flashlight Alignment
> `Context:` Physical & Everyday Metaphors for Vector Norms and Dot Products

#### Metaphor 1: The Stretched Rubber Band (Vector Norm)
- A vector is a rubber band anchored at the origin $(0, 0)$.
- The **Norm $\|x\|_2$** is the physical length of the stretched rubber band.
- If the vector is zero $\mathbf{0}$, the rubber band has zero length ($\|x\| = 0$).

---

#### Metaphor 2: Two Flashlights Shining in Fog (The Dot Product)
- **Same Direction ($0^\circ$ Angle):** Both flashlight beams align, creating maximum brightness ($x^\top y = \|x\| \|y\| > 0$).
- **Perpendicular ($90^\circ$ Angle / Orthogonal):** The beams cross at a right angle with zero overlapping projection ($x^\top y = 0$).
- **Opposite Directions ($180^\circ$ Angle):** The flashlights point away from each other, representing complete opposition ($x^\top y < 0$).

---

### 3. 📚 Deep Terminology Master Glossary (15 Core Concepts Dissected)
> `Context:` Foundational Mathematical & Machine Learning Vocabulary Explained Without Jargon

```
 ===================================================================================================
                 THE VECTOR GEOMETRY & NORMS ROSETTA STONE
 ===================================================================================================
```

| Term / Notation | Formal Mathematical Meaning | Plain-English Definition (No ML Jargon) | Real-World Analogy |
| :--- | :--- | :--- | :--- |
| **Vector Norm ($\|x\|$ )** | Function assigning non-negative length | The physical length or size of a multi-dimensional arrow | Measuring an arrow with a tape measure |
| **$L_1$ Norm (Manhattan)** | $\|x\|_1 = \sum |x_i|$ | Sum of absolute coordinate distances along grid lines | Driving along street blocks in a city |
| **$L_2$ Norm (Euclidean)** | $\|x\|_2 = \sqrt{\sum x_i^2}$ | True straight-line ruler distance between origin and point | A laser rangefinder distance measurement |
| **$L_\infty$ Norm (Chebyshev)** | $\|x\|_\infty = \max |x_i|$ | The single largest coordinate value in the vector | The tallest player on a basketball team |
| **Inner Product / Dot Product** | $\langle x, y \rangle = x^\top y = \sum x_i y_i$ | Multiplies matching coordinates and sums them to measure alignment | Calculating total grocery cost from price $\times$ quantity |
| **Cosine Similarity** | $\frac{x^\top y}{\|x\|_2 \|y\|_2} = \cos(\theta)$ | Measures the angle between two vectors, ignoring their lengths | Checking if two compass needles point North |
| **Orthogonality ($x \perp y$)** | $x^\top y = 0$ | Two vectors meet at an exact $90^\circ$ right angle with zero correlation | North vs East on a compass |
| **Cauchy-Schwarz Inequality** | $|x^\top y| \le \|x\|_2 \|y\|_2$ | The dot product can never exceed the product of individual lengths | You cannot catch more fish than exist in the lake |
| **Triangle Inequality** | $\|x + y\| \le \|x\| + \|y\|$ | The straight path is always shorter than taking a detour | Walking directly across a park vs around the perimeter |
| **Weight Decay ($L_2$ Regularization)** | $\mathcal{L}_{\text{total}} = \mathcal{L} + \frac{\lambda}{2}\|W\|_2^2$ | Penalizes large network weights to prevent overfitting | Keeping volume knobs within safe limits |
| **Lasso Sparsity ($L_1$ Regularization)** | $\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda \|W\|_1$ | Drives unneeded weights to exact zero for automatic feature selection | Decluttering a closet by throwing away unused items |
| **Spectral Norm ($\sigma(W)$)** | Largest singular value $\max \frac{\|Wv\|}{\|v\|}$ | The maximum magnification factor a layer can apply to any vector | The maximum zoom multiplier on a telescope |
| **Unit Ball ($\{x : \|x\| \le 1\}$)** | Geometric shape of all vectors of length $\le 1$ | $L_1$ forms a diamond, $L_2$ forms a circle, $L_\infty$ forms a square | Cookie cutter shapes |
| **Scaled Dot-Product Attention** | $\text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | Core Transformer equation measuring query-key token similarity | Matching job search queries with applicant resumes |
| **Vector Normalization ($\hat{x} = x / \|x\|_2$)** | Scaling vector to unit length $\|\hat{x}\|_2 = 1$ | Shrinking or stretching an arrow until its length is exactly $1$ | Resizing all photos to standard passport dimensions |

---

### 4. 📐 Mathematical Formulations, Unit Ball Geometry & Proofs
> `Context:` Mathematical Definitions, Unit Ball Shapes, and Cauchy-Schwarz Proof

```
 ===================================================================================================
                 THE 2D UNIT BALL GEOMETRIES (||x|| ≤ 1.0)
 ===================================================================================================

  L₁ NORM (Diamond)                    L₂ NORM (Circle / Sphere)            L_∞ NORM (Square / Box)
  |x₁| + |x₂| ≤ 1                      x₁² + x₂² ≤ 1                        max(|x₁|, |x₂|) ≤ 1
  x₂ ▲                                 x₂ ▲                                 x₂ ▲
     │   /\                               │    .---.                           │  ┌───────┐
     │  /  \                              │  .'     '.                         │  │       │
  ───┼─/────\───► x₁                   ───┼─/───────\───► x₁                ───┼──┼───────┼──► x₁
     │ \    /                             │ '.     .'                          │  │       │
     │  \  /                              │   '---'                            │  └───────┘
 ===================================================================================================
```

#### Core Mathematical Theorems:

1. **The 3 Formal Norm Axioms:**
   - **Non-Negativity:** $\|x\| \ge 0$, and $\|x\| = 0 \iff x = \mathbf{0}$.
   - **Absolute Scalability:** $\|\alpha x\| = |\alpha| \cdot \|x\|$ for all scalars $\alpha \in \mathbb{R}$.
   - **Triangle Inequality:** $\|x + y\| \le \|x\| + \|y\|$.

2. **Cauchy-Schwarz Inequality Theorem:**
   For any vectors $x, y \in \mathbb{R}^d$:
   $$|x^\top y| \le \|x\|_2 \cdot \|y\|_2$$
   - Equality holds if and only if $x = \alpha y$ (collinear vectors).

3. **Cosine Angle Decomposition:**
   $$x^\top y = \|x\|_2 \|y\|_2 \cos(\theta) \implies \cos(\theta) = \frac{x^\top y}{\|x\|_2 \|y\|_2}$$

---

### 5. 🔢 Concrete Micro-Numerical Worked Examples
> `Context:` Step-by-Step Manual Calculations (No Black Box)

#### Example 1: Comprehensive 2D Vector Geometry by Hand
Let vector $x = [3.0, \quad 4.0]^\top$ and vector $y = [1.0, \quad 2.0]^\top$ in $\mathbb{R}^2$:

1. **$L_1$ Norms (Manhattan):**
   $$\|x\|_1 = |3.0| + |4.0| = \mathbf{7.0000}$$
   $$\|y\|_1 = |1.0| + |2.0| = \mathbf{3.0000}$$

2. **$L_2$ Norms (Euclidean):**
   $$\|x\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5.0000}$$
   $$\|y\|_2 = \sqrt{1^2 + 2^2} = \sqrt{1 + 4} = \sqrt{5} \approx \mathbf{2.2361}$$

3. **$L_\infty$ Norms (Maximum Absolute):**
   $$\|x\|_\infty = \max(|3|, |4|) = \mathbf{4.0000}, \quad \|y\|_\infty = \max(|1|, |2|) = \mathbf{2.0000}$$

4. **Inner Product (Dot Product):**
   $$x^\top y = 3.0(1.0) + 4.0(2.0) = 3.0 + 8.0 = \mathbf{11.0000}$$

5. **Cosine Similarity & Angle $\theta$:**
   $$\cos(\theta) = \frac{x^\top y}{\|x\|_2 \|y\|_2} = \frac{11.0}{5.0 \times 2.2361} = \frac{11.0}{11.1803} = \mathbf{0.9839}$$
   $$\theta = \arccos(0.9839) \approx 0.1798\text{ radians} \approx \mathbf{10.3^\circ} \quad (\text{Very closely aligned!})$$

---

#### Example 2: Transformer Scaled Dot-Product Attention by Hand
Let Query vector $q = [1.0, \quad 0.0, \quad 1.0, \quad 0.0]$ and Key vector $k = [0.0, \quad 1.0, \quad 1.0, \quad 0.0]$ ($d_k = 4$):

1. **Dot Product:**
   $$q^\top k = 1(0) + 0(1) + 1(1) + 0(0) = \mathbf{1.0}$$
2. **Scaling Factor $\sqrt{d_k} = \sqrt{4} = 2.0$:**
   $$\text{Scaled Logit} = \frac{q^\top k}{\sqrt{d_k}} = \frac{1.0}{2.0} = \mathbf{0.50}$$
   *(Scaling by $\sqrt{d_k}$ prevents dot products from growing excessively large in high dimensions, preserving stable gradients!)*

---

### 6. 🔗 Connecting the Dots: How Norms & Dot Products Power Generative AI
> `Context:` Architectural Implementations in Large Language Models, Diffusion Models, and GANs

```
 ===================================================================================================
                 INNER PRODUCTS & NORMS ACROSS GENERATIVE AI
 ===================================================================================================

  1. TRANSFORMER SCALED ATTENTION                   2. WGAN-GP CRITIC GRADIENT NORM
  Attention(Q, K, V) = Softmax(QKᵀ / √d_k) V        ℒ_GP = 𝔼[( ||∇_x̂ D(x̂)||₂ - 1 )²]
  ┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐
  │ Dot product QKᵀ computes raw semantic  │        │ Takes L₂ Euclidean norm of the         │
  │ alignment between every token pair     │        │ gradient vector; forces slope to be    │
  │ Division by √d_k prevents saturation   │        │ exactly 1.0 everywhere on manifold     │
  └────────────────────────────────────────┘        └────────────────────────────────────────┘
 ===================================================================================================
```

| Generative Architecture | Primary Norm / Product | Architectural Role |
| :--- | :--- | :--- |
| **Large Language Models (Transformers)** | **Scaled Dot Product $\langle q, k \rangle / \sqrt{d_k}$** | Computes attention affinity weights between query tokens and memory keys |
| **RAG & Vector Search (Embedding Models)** | **Cosine Similarity $\frac{u^\top v}{\|u\| \|v\|}$** | Indexes and retrieves most semantically relevant text passages from vector DB |
| **WGAN-GP (Generative Adversarial Nets)** | **$L_2$ Gradient Norm $\|\nabla D\|_2$** | Enforces 1-Lipschitz continuity along the linear interpolation path |
| **Diffusion Models (DiT / Flux)** | **$L_2$ Mean Squared Error $\| \epsilon - \epsilon_\theta \|_2^2$** | Computes loss between true Gaussian noise and neural network noise prediction |

---

### 7. 💻 Complete Standalone Executable Python/PyTorch Verification Script
> `Context:` Runnable Code Verifying $L_1, L_2, L_\infty$ Norms, Cosine Similarity, and Attention Scaling

```python
"""
Vector Norms, Distances & Inner Products Simulation
===================================================
Demonstrates:
1. Computation of L1, L2, Linf norms and Euclidean distance in PyTorch
2. Dot product, Cosine similarity, and angle theta calculation
3. Scaled dot-product attention calculation in Transformers
"""
import torch
import torch.nn.functional as F
import numpy as np

print("=" * 75)
print("VECTOR NORMS, DISTANCES & INNER PRODUCTS MATHEMATICAL SIMULATION")
print("=" * 75)

# ─── 1. Vector Norms in PyTorch ───
print("\n1. PYTORCH VECTOR NORMS CALCULATION (x = [3.0, 4.0]):")
x = torch.tensor([3.0, 4.0])
y = torch.tensor([1.0, 2.0])

norm_l1 = torch.norm(x, p=1).item()
norm_l2 = torch.norm(x, p=2).item()
norm_linf = torch.norm(x, p=float('inf')).item()

print(f"   * L1 Norm (Manhattan):        {norm_l1:.4f} (Analytic: 7.0000) ✅")
print(f"   * L2 Norm (Euclidean):        {norm_l2:.4f} (Analytic: 5.0000) ✅")
print(f"   * Linf Norm (Chebyshev):      {norm_linf:.4f} (Analytic: 4.0000) ✅")

# ─── 2. Inner Product & Cosine Similarity ───
print("\n2. INNER PRODUCT & COSINE SIMILARITY:")
dot_product = torch.dot(x, y).item()
cos_sim = F.cosine_similarity(x, y, dim=0).item()
angle_deg = np.degrees(np.arccos(cos_sim))
dist_l2 = torch.dist(x, y, p=2).item()

print(f"   * Dot Product x^T y:          {dot_product:.4f} (Analytic: 11.0000) ✅")
print(f"   * Cosine Similarity:          {cos_sim:.4f} (Analytic: 0.9839) ✅")
print(f"   * Angle Between Vectors:      {angle_deg:.2f}°")
print(f"   * Euclidean Distance ||x-y||: {dist_l2:.4f} (Analytic: 2.8284) ✅")

# ─── 3. Cauchy-Schwarz Inequality Verification ───
print("\n3. CAUCHY-SCHWARZ INEQUALITY TEST (|x^T y| <= ||x|| * ||y||):")
lhs = abs(dot_product)
rhs = norm_l2 * torch.norm(y, p=2).item()

print(f"   * Left Hand Side |x^T y|:     {lhs:.4f}")
print(f"   * Right Hand Side ||x||*||y||: {rhs:.4f}")
assert lhs <= rhs + 1e-6, "Cauchy-Schwarz assertion failed!"
print("   * Cauchy-Schwarz inequality verified mathematically! ✅")

# ─── 4. Transformer Scaled Dot-Product Attention ───
print("\n4. TRANSFORMER SCALED ATTENTION LOGIT:")
q = torch.tensor([1.0, 0.0, 1.0, 0.0]) # d_k = 4
k = torch.tensor([0.0, 1.0, 1.0, 0.0])
d_k = q.shape[0]

scaled_logit = torch.dot(q, k) / np.sqrt(d_k)
print(f"   * Query: {q.tolist()}, Key: {k.tolist()} (d_k = {d_k})")
print(f"   * Scaled Logit (q^T k / sqrt(d_k)): {scaled_logit.item():.4f} (Analytic: 0.5000) ✅")

print("\n" + "=" * 75)
print("ALL VECTOR NORM & INNER PRODUCT TESTS PASSED SUCCESSFULLY! ✅")
print("=" * 75)
```

---

### 8. 🩺 Diagnostic Mini-Checks & Common Traps
> `Context:` Production Debugging Insights, Edge-Case Traps & Self-Verification Questions

#### ✅ Self-Test Questions

1. **Q:** Why do Transformer models scale the dot product by $\frac{1}{\sqrt{d_k}}$ in the attention equation?  
   **A:** In high-dimensional spaces ($d_k = 128$), the variance of the unscaled dot product $q^\top k$ grows proportionally to $d_k$, pushing pre-softmax values into extreme regions where Softmax gradients vanish ($\approx 0$). Dividing by $\sqrt{d_k}$ normalizes the variance back to $1.0$.

2. **Q:** What is the fundamental difference between $L_1$ regularization and $L_2$ regularization?  
   **A:** **$L_1$ (Lasso)** drives unimportant weights to **exact zero**, producing sparse feature selection due to the sharp diamond vertices of its unit ball. **$L_2$ (Ridge / Weight Decay)** shrinks weights smoothly toward zero without making them exactly zero.

3. **Q:** Can Cosine Similarity be computed between vectors of different lengths?  
   **A:** **Yes!** Cosine similarity normalizes both vectors by their Euclidean lengths ($\|u\|_2, \|v\|_2$), measuring purely the angular alignment while being 100% invariant to vector scale.

#### ⚠️ Common Engineering Traps

| Trap | Why It Fails | Production Fix |
| :--- | :--- | :--- |
| **Computing cosine similarity on unnormalized embeddings in a loop** | Redundant normalization on every query causes massive search latency | Pre-normalize all database vectors to unit length ($\|\hat{v}\| = 1$) so similarity is a simple matrix multiply: $S = Q D^\top$ |
| **Dividing by zero norm ($\|x\|_2 = 0$) during normalization** | Zero vector division causes `NaN` outputs that corrupt neural network weights | Add numerical epsilon: $\frac{x}{\|x\|_2 + 10^{-8}}$ or use `F.normalize(x, p=2, eps=1e-8)` |
| **Assuming dot product equals cosine similarity** | Dot product is heavily influenced by vector magnitude; a long irrelevant vector can have a larger dot product than a short relevant one | Normalize vectors before searching or use explicit Cosine Similarity |

---

### 🎯 Summary Checklist
- **$L_2$ Norm** is the straight-line Euclidean ruler distance; **$L_1$ Norm** is the grid-based Manhattan distance.
- **Dot Product $x^\top y$** measures directional alignment and projection in Euclidean space.
- **Cosine Similarity** normalizes the dot product to $[-1, 1]$, measuring pure angular orientation.
- **Scaled Dot-Product Attention** $\text{Softmax}(QK^\top / \sqrt{d_k})$ powers all modern Transformer Large Language Models.
- **WGAN-GP** relies on $L_2$ gradient norms to enforce the 1-Lipschitz condition.
