# Prerequisites — Lec 43: Local Receptive Fields and Parameter Sharing

> **Do this first.** Then open [NOTES.md](./NOTES.md) at the **Executive Summary** map.  
> Basics only — not a second lecture. They unlock words on the master map if you are rusty.  
> **How to read:** Use the **3-Minute Executive Fast-Track** below for a rapid ramp-up. Read the 👶 Physical Intuition and 💻 Python snippets in each pillar. Expand the 📐 formal calculus blocks only when you need deep mathematical proofs.

```
  After this warm-up you can say:

  "Images have grid topology where neighboring pixels share high mutual information."
  "Local receptive fields prune 99%+ of dense MLP connections via Dirac delta priors."
  "Parameter sharing ties sliding weights together, creating banded Toeplitz matrices."
  "Convolutional layers are translation equivariant; global pooling makes them invariant."
  "Architectures act as Bayesian priors, restricting search space for tractable ERM."
```

---

## ⚡ 3-Minute Executive Fast-Track

If you have only 3 minutes before starting the lecture, master this visual blueprint:

```
  DENSE MLP (Unordered Confetti Space):
  x ∈ ℝᵈ ────────► [ Dense W: M × d unconstrained params ] ────────► z ∈ ℝᴹ  (O(M·d) weights)

  CNN ARCHITECTURAL INDUCTIVE BIAS:
  x ∈ ℝᵈ ───► [ 1. Local Receptive Field (Banded Sparsity) ] ───► [ 2. Parameter Sharing (Toeplitz) ] ───► z ∈ ℝᴹ
                                                                                                    (Only k weights!)
```

### 🧠 The 3 Core Mental Shifts
1. **Pixels Are Not Unordered Confetti:** Unlike tabular features, images have grid topology where Euclidean coordinate distance carries spatial mutual information; neighboring pixels are strongly autocorrelated, while distant pixels are largely independent.
2. **The Dual Inductive Biases of CNNs:** A convolutional layer is born from two distinct structural regularizers applied to a dense MLP: (1) Local Receptive Fields (forcing 99%+ of weights to zero via Dirac delta priors), and (2) Parameter Sharing (tying non-zero weights across space to form Toeplitz matrices).
3. **Equivariance vs. Invariance:** Convolutions are translation *equivariant* ($f(\mathcal{T}_v x) = \mathcal{T}_v f(x)$—the feature map shifts along with the object). Global spatial pooling turns this into translation *invariance* ($g(\mathcal{T}_v x) = g(x)$—the category label doesn't change when the object moves).

### ⏱️ Instant Readiness Check
1. *If you permute all pixels of an image with a fixed random permutation matrix $P$, which model's performance collapses: a dense MLP or a CNN?*  
   <details><summary><b>Reveal Answer</b></summary><b>The CNN collapses catastrophically</b> because local receptive fields now operate over disjoint, uncorrelated pixels. The dense MLP is unaffected because its unconstrained weights simply reorder to absorb the permutation.</details>
2. *Does parameter sharing reduce the number of floating-point operations (FLOPs) required to compute the forward pass?*  
   <details><summary><b>Reveal Answer</b></summary><b>No.</b> Parameter sharing reduces memory footprint and sample complexity by reusing weights, but each patch still requires $k$ multiply-accumulate operations ($\mathcal{O}(M \cdot k)$ FLOPs).</details>
3. *In 1D convolution with kernel $w = [w_0, w_1, w_2]$ and stride 1, what special matrix structure represents the linear transformation?*  
   <details><summary><b>Reveal Answer</b></summary><b>A banded Toeplitz matrix</b> (where descending diagonals from left to right have constant values).</details>

---

## Math Terminology Rosetta Stone

Before diving into the foundational pillars, use this reference table to decode mathematical shorthand and notation into plain English, spoken phonetics, and software implementations.

| Symbol | Mathematical Concept | Spoken English (Phonetics) | Plain-English Intuition | Computational / Code Representation |
|:-------|:---------------------|:---------------------------|:------------------------|:-----------------------------------|
| $\mathcal{RF}(j)$ | Local Receptive Field Set | "ARR-eff of JAY" | The specific window of input pixels that a single neuron looks at | `input_slice = x[j*s : j*s + k]` |
| $k$ | Kernel / Filter Width | "KAY / filter size" | How many adjacent input units feed into a local neuron | `kernel_size = 3` |
| $s$ | Stride Step-Size | "ESS / stride" | How many pixels we shift the receptive field to reach the next neuron | `stride = 1` |
| $w = [w_0, \dots, w_{k-1}]$ | Shared Parameter Kernel | "DOUBLE-yoo filter vector" | The small template of learnable weights reused across the entire grid | `nn.Parameter(torch.randn(k))` |
| $T_v$ | Spatial Translation Operator | "TEE SUB VEE" | Shifting the entire image canvas by $v$ pixels | `torch.roll(x, shifts=v, dims=-1)` |
| $\delta_0(w)$ | Dirac Delta Prior Density | "DEER-ack DEL-tuh of DOUBLE-yoo" | An infinite probability spike at zero, forcing a weight to be strictly zero | `mask * W` where `mask == 0` |
| $\star$ | Discrete Cross-Correlation | "STAR / cross-correlation operator" | Sliding dot product between a kernel template and local patches of data | `torch.nn.functional.conv1d` |
| $p(\theta)$ | Parameter Prior Distribution | "PEE of THAY-tuh" | Bayesian belief distribution over parameter weights before observing data | `prior_loss = lambda_reg * torch.norm(W)` |

---

## 🌉 Curriculum & Sibling Course Prerequisite Bridges

| Concept / Prerequisite | Where Established | How Used in This Lecture | Sibling Track Bridge |
|:-----------------------|:------------------|:-------------------------|:---------------------|
| Universal Approximation Theorem (UAT) | [`54-Lec41`](../54-Lec41-Neural-Networks-UAT/) | Establishes baseline that single-hidden-layer MLPs are universal; motivates why architectural restrictions are needed for practical ERM | [`MathsTerms: UAT`](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/) |
| Empirical Risk Minimization & Backpropagation | [`55-Lec42`](../55-Lec42-ERM-Neural-Networks-Backpropagation/) | Computational machinery adapted to accumulate gradients across tied weights in convolutional layers | [`04-Chain_Rule_and_Backpropagation.md`](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/04-Chain_Rule_and_Backpropagation.md) |
| MAP Estimation & Bayesian Priors | [`47-Lec34`](../47-Lec34-MAP-Estimation/) | Formal foundation proving that architectural pruning and weight tying are singular Dirac delta priors | [`04-Likelihood_and_Log_Likelihood.md`](../../MathsTerms/03-Probability-and-Statistical-Estimation/04-Likelihood_and_Log_Likelihood.md) |
| Regularization & Weight Decay | [`46-Lec33`](../46-Lec33-Regularization/) | Contrastive baseline comparing soft $L_2$ penalties with hard architectural structural regularizers | [`09-Gradient_Descent.md`](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/09-Gradient_Descent.md) |
| 2D Convolutions & Image Tensors | [`05-Tutorial04`](../../Mathematical-Foundation-for-GenerativeAI/05-Tutorial04-CNNs-PyTorch/) | Practical implementation bridge to PyTorch `nn.Conv2d` tensor operations | [`01-Convolution_and_Pooling.md`](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) |

---

## 1. Grid Topologies vs Permutation-Invariant Vector Spaces

<a id="p1"></a>

### 👶 Physical Analogy & Intuition
Imagine a printed newspaper photograph cut into a million tiny confetti squares. If you throw them randomly into a bucket, a naive observer sees only an unstructured collection of colored specks. But if you keep the grid coordinates of every speck intact, neighboring specks combine into recognizable human eyes, text headlines, and buildings. Fully connected MLPs treat inputs like a bucket of confetti—shuffling pixel locations changes nothing about their internal geometry. Grid topologies, by contrast, assign geometric meaning to neighbor relationships: nearby pixels are physically correlated, while distant pixels are largely independent.

### 🔍 Plain-English Breakdown
- In a generic vector space $\mathbb{R}^d$, feature dimensions are unordered: swapping coordinate 1 and coordinate 100 has no geometric consequence.
- In a **grid topology** (images, audio spectrograms, video frames), coordinates carry spatial or temporal distances.
- Adjacent pixels share high mutual information (smooth surfaces, continuous textures), whereas distant pixels are weakly correlated.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Consider a 1D sequence of 4 pixels: $x = [10, 12, 1, 2]$.
- **Local pixel differences:**
  - Adjacent differences: $|x_1 - x_0| = |12 - 10| = 2$ (smooth region), $|x_3 - x_2| = |2 - 1| = 1$ (smooth region).
  - Distal difference: $|x_2 - x_1| = |1 - 12| = 11$ (sharp boundary edge at coordinate 1.5).
- If an unconstrained permutation $\pi = [0, 2, 1, 3]$ rearranges coordinates, the permuted vector becomes $\tilde{x} = [10, 1, 12, 2]$.
  - The apparent adjacent differences become $|1 - 10| = 9$, $|12 - 1| = 11$, and $|2 - 12| = 10$.
  - The local smooth structure has been completely destroyed into high-frequency noise, proving that arbitrary coordinate permutation destroys grid topology.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# Generate 1D spatially correlated signal
np.random.seed(42)
t = np.linspace(0, 10, 100)
signal = np.sin(t) + np.random.normal(0, 0.05, size=100)

# Lag-1 autocorrelation (adjacent neighbors)
corr_lag1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
# Lag-50 autocorrelation (distant points)
corr_lag50 = np.corrcoef(signal[:-50], signal[50:])[0, 1]

assert corr_lag1 > 0.95, f"Expected high local correlation, got {corr_lag1}"
assert abs(corr_lag50) < 0.2, f"Expected low distal correlation, got {corr_lag50}"
print(f"[P1 PASS] Grid autocorrelation verified: Lag-1={corr_lag1:.3f} >> Lag-50={corr_lag50:.3f}")
```

### 🩺 Diagnostic Mini-Check
**Question:** If we randomly shuffle the pixel positions of all $28 \times 28$ MNIST images with a fixed, deterministic permutation matrix $P$, can a fully-connected MLP still achieve the same training accuracy as on the unpermuted images? Can a standard CNN?  
<details><summary><b>Reveal Answer</b></summary><b>An MLP achieves the exact same training accuracy</b> because its linear layers $W P x = \tilde{W} x$ can absorb the permutation without structural penalty. A CNN suffers catastrophic failure because local receptive fields now operate over disjoint, randomly scattered pixels with zero spatial correlation.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $\Omega = \{1, 2, \dots, d\}$ index input coordinates.
- In a **permutation-invariant metric space**, for any permutation $\pi \in S_d$, the hypothesis class satisfies $\mathcal{H}(x) \cong \mathcal{H}(\pi(x))$ up to relabeled coordinate weights:
  $$\|x_i - x_j\| = \delta_{ij}$$
- In a **topological grid space** $G = \{1, \dots, H\} \times \{1, \dots, W\}$, input points $u = (r_u, c_u)$ and $v = (r_v, c_v)$ have spatial distance metric:
  $$\text{dist}(u, v) = \|u - v\|_2 = \sqrt{(r_u - r_v)^2 + (c_u - c_v)^2}$$
- The spatial autocorrelation function $R(u, v) = \mathbb{E}[X(u) X(v)]$ decays rapidly as $\text{dist}(u, v) \to \infty$, establishing localized statistical dependence.
</details>

---

## 2. Receptive Fields & Structural Weight Matrix Sparsity

<a id="p2"></a>

### 👶 Physical Analogy & Intuition
Think of looking at the night sky through a narrow telescope rather than with your naked eyes. A naked-eye view takes in the entire horizon at once (global receptive field), making it hard to count individual craters on the moon because city lights and distant airplanes distract you. A telescope isolates a tiny patch of the sky (local receptive field), eliminating all irrelevant distal light and allowing your eyes to focus on fine details. In matrix algebra, a telescope corresponds to replacing a dense weight matrix with a banded sparse matrix where 99% of the entries are hard-zeroed out.

### 🔍 Plain-English Breakdown
- In a standard MLP, every neuron in layer $l$ connects to every neuron in layer $l-1$ ($M \times d$ parameters).
- In a **local receptive field** of size $k$, each neuron only connects to a contiguous window of $k$ inputs ($k \ll d$).
- This prunes all non-local connections, transforming the dense weight matrix into a sparse, banded matrix with sparsity ratio $1 - k/d$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let input dimension $d = 6$, filter width $k = 3$, stride $s = 1$, and output dimension $M = d - k + 1 = 4$.
- Input: $x = [x_0, x_1, x_2, x_3, x_4, x_5]^T$.
- Receptive fields:
  - $\mathcal{RF}(0) = \{0, 1, 2\} \implies z_0 = w_{0,0} x_0 + w_{0,1} x_1 + w_{0,2} x_2$
  - $\mathcal{RF}(1) = \{1, 2, 3\} \implies z_1 = w_{1,1} x_1 + w_{1,2} x_2 + w_{1,3} x_3$
  - $\mathcal{RF}(2) = \{2, 3, 4\} \implies z_2 = w_{2,2} x_2 + w_{2,3} x_3 + w_{2,4} x_4$
  - $\mathcal{RF}(3) = \{3, 4, 5\} \implies z_3 = w_{3,3} x_3 + w_{3,4} x_4 + w_{3,5} x_5$
- The full transformation matrix $W \in \mathbb{R}^{4 \times 6}$ is:
  $$W = \begin{bmatrix}
  w_{0,0} & w_{0,1} & w_{0,2} & 0 & 0 & 0 \\
  0 & w_{1,1} & w_{1,2} & w_{1,3} & 0 & 0 \\
  0 & 0 & w_{2,2} & w_{2,3} & w_{2,4} & 0 \\
  0 & 0 & 0 & w_{3,3} & w_{3,4} & w_{3,5}
  \end{bmatrix}$$
- Out of 24 matrix entries, exactly 12 are non-zero. The sparsity is $1 - 12/24 = 50\%$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

d = 8
k = 3
s = 1
M = (d - k) // s + 1  # 6 outputs

# Construct sparse matrix
W_sparse = np.zeros((M, d))
for j in range(M):
    for r in range(k):
        W_sparse[j, j * s + r] = np.random.randn()
        
# Count non-zero entries
nnz = np.count_nonzero(W_sparse)
total_entries = M * d
sparsity = 1.0 - (nnz / total_entries)

assert nnz == M * k, f"Expected {M*k} non-zeros, got {nnz}"
assert np.isclose(sparsity, 1.0 - (k / d))
print(f"[P2 PASS] Local receptive field sparsity verified: {sparsity*100:.1f}%")
```

### 🩺 Diagnostic Mini-Check
**Question:** If an input image has resolution $1000 \times 1000$ ($d = 10^6$ pixels) and the next layer has $10^6$ neurons, how many parameters does a dense MLP weight matrix require? How many parameters does a local receptive field layer with $k = 5 \times 5 = 25$ require (without weight sharing)?  
<details><summary><b>Reveal Answer</b></summary><b>Dense MLP requires $10^{12}$ parameters</b> (1 trillion weights, ~4 Terabytes). With local receptive fields of size 25, non-zero weights drop to $10^6 \times 25 = 2.5 \times 10^7$ (25 million weights, ~100 Megabytes), a <b>$40,000\times$ parameter reduction</b>.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let input vector be $x \in \mathbb{R}^d$ and hidden layer activations be $z \in \mathbb{R}^M$.
- **Full MLP Connectivity (Dense Receptive Field):**
  $$z_j = \sum_{i=1}^d W_{ji} x_i + b_j \iff z = W x + b, \quad W \in \mathbb{R}^{M \times d}$$
  Total non-zero parameters: $\text{nnz}(W) = M \cdot d$.
- **Local Receptive Field of size $k$ with stride $s$ (Sparse Connectivity):**
  Neuron $j \in \{0, 1, \dots, M-1\}$ connects only to input indices in $\mathcal{RF}(j) = \{j \cdot s, j \cdot s + 1, \dots, j \cdot s + k - 1\}$.
  $$W_{ji} = 0 \quad \forall i \notin \mathcal{RF}(j)$$
  Total non-zero parameters (without weight sharing): $\text{nnz}(W) = M \cdot k$.
  Sparsity ratio:
  $$\text{Sparsity} = 1 - \frac{M \cdot k}{M \cdot d} = 1 - \frac{k}{d}$$
</details>

---

## 3. Parameter Sharing & Weight Tying Across Coordinates

<a id="p3"></a>

### 👶 Physical Analogy & Intuition
Imagine a quality inspector testing manufactured light bulbs on an assembly line. Instead of hiring 1,000 separate inspectors—one for each station along the conveyer belt—and training each inspector independently, you hire one master inspector and give them a mobile cart so they apply the exact same testing checklist at every station. Parameter sharing (weight tying) does exactly this: instead of learning independent weights for every spatial patch across an image, one shared filter vector travels across every location, applying the identical feature-detecting rule.

### 🔍 Plain-English Breakdown
- In an unshared local layer, each spatial position has its own unique $k$ weights ($M \cdot k$ total).
- **Parameter sharing** constrains the weights across all positions to be identical: $W_{j, j\cdot s + r} = w_r$.
- This collapses the parameter count to just $k$ weights $+ 1$ bias, transforming $W$ into a **Toeplitz matrix** (where all descending diagonals are constant).

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let shared kernel be $w = [2.0, -1.0, 0.5]$ and bias $b = 1.0$.
Let input be $x = [1, 3, 2, 4, 1]$.
Output dimension $M = 5 - 3 + 1 = 3$.
- **Neuron 0 ($j=0$):** Patch $x[0:3] = [1, 3, 2]$.
  $$z_0 = 2.0(1) + (-1.0)(3) + 0.5(2) + 1.0 = 2 - 3 + 1 + 1 = 1.0$$
- **Neuron 1 ($j=1$):** Patch $x[1:4] = [3, 2, 4]$.
  $$z_1 = 2.0(3) + (-1.0)(2) + 0.5(4) + 1.0 = 6 - 2 + 2 + 1 = 7.0$$
- **Neuron 2 ($j=2$):** Patch $x[2:5] = [2, 4, 1]$.
  $$z_2 = 2.0(2) + (-1.0)(4) + 0.5(1) + 1.0 = 4 - 4 + 0.5 + 1 = 1.5$$
Output feature map: $z = [1.0, 7.0, 1.5]^T$, computed with only 3 shared weights.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
w = np.array([2.0, -1.0, 0.5])
b = 1.0

# 1. Slide window
M = len(x) - len(w) + 1
z_loop = np.zeros(M)
for j in range(M):
    z_loop[j] = np.dot(w, x[j:j+len(w)]) + b
    
# 2. Toeplitz matrix multiplication
W_toeplitz = np.array([
    [2.0, -1.0, 0.5,  0.0, 0.0],
    [0.0,  2.0, -1.0, 0.5, 0.0],
    [0.0,  0.0,  2.0, -1.0, 0.5]
])
z_matrix = W_toeplitz @ x + b

assert np.allclose(z_loop, z_matrix), "Loop and Toeplitz outputs must match!"
assert np.allclose(z_loop, np.array([1.0, 7.0, 1.5]))
print("[P3 PASS] Parameter sharing & Toeplitz matrix verified:", z_loop)
```

### 🩺 Diagnostic Mini-Check
**Question:** Does parameter sharing reduce the number of floating-point multiplication operations (FLOPs) required during the forward pass?  
<details><summary><b>Reveal Answer</b></summary><b>No.</b> Parameter sharing reduces memory footprint and sample complexity by tying parameter values, but each spatial patch still requires $k$ multiply-accumulate operations. The forward FLOP count remains $\mathcal{O}(M \cdot k)$.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $W \in \mathbb{R}^{M \times d}$ be the banded local receptive field matrix.
- **Parameter Sharing Constraint:**
  For all output neurons $j \in \{0, \dots, M-1\}$ and local offsets $r \in \{0, \dots, k-1\}$, the non-zero entries are tied to a single shared parameter vector $w = [w_0, w_1, \dots, w_{k-1}]^T \in \mathbb{R}^k$:
  $$W_{j, j \cdot s + r} = w_r \quad \forall j \in \{0, \dots, M-1\}$$
- **Toeplitz Structure:** When stride $s = 1$, the matrix $W$ becomes a **Toeplitz matrix** (diagonal-constant matrix):
  $$W = \begin{bmatrix}
  w_0 & w_1 & w_2 & 0 & \dots & 0 \\
  0 & w_0 & w_1 & w_2 & \dots & 0 \\
  \vdots & \ddots & \ddots & \ddots & \ddots & \vdots \\
  0 & \dots & 0 & w_0 & w_1 & w_2
  \end{bmatrix}$$
- **Parameter Count:** The parameter count collapses from $M \cdot k$ down to $k$ weights $+ 1$ scalar bias $b$, completely independent of input size $d$ and output size $M$.
</details>

---

## 4. Translation Stationarity, Invariance, and Equivariance

<a id="p4"></a>

### 👶 Physical Analogy & Intuition
Consider identifying a coffee mug on a desk. If you slide the mug 5 inches to the right:
- **Equivariance:** The bounding box detector in your brain tracks the movement: its output shifts 5 inches to the right. The output moves in tandem with the input.
- **Invariance:** The classification label remains "coffee mug", whether the mug sits on the left, center, or right edge of the desk. The output label does not change at all.
Convolutional layers are inherently **equivariant**; when followed by global spatial pooling, the network becomes **invariant**.

### 🔍 Plain-English Breakdown
- **Translation Operator** $\mathcal{T}_v$: Shifts a signal by spatial displacement $v$.
- **Equivariance:** $f(\mathcal{T}_v x) = \mathcal{T}_v (f(x))$. Applying the filter before or after shifting yields the exact same shifted feature map.
- **Invariance:** $g(\mathcal{T}_v x) = g(x)$. The final representation is completely unaffected by spatial displacement.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let input $x = [0, 5, 0, 0]$, kernel $w = [1, 2]$, and bias $b = 0$.
Valid convolution output $z = w \star x$:
- $z_0 = 1(0) + 2(5) = 10$
- $z_1 = 1(5) + 2(0) = 5$
- $z_2 = 1(0) + 2(0) = 0$
Output $z = [10, 5, 0]$.

Now translate input $x$ right by $v = 1$ step: $x_{\text{shifted}} = [0, 0, 5, 0]$.
- $z_0' = 1(0) + 2(0) = 0$
- $z_1' = 1(0) + 2(5) = 10$
- $z_2' = 1(5) + 2(0) = 5$
Output $z' = [0, 10, 5]$.
Notice: $z' = \mathcal{T}_1 z$! The output vector has translated right by exactly 1 position, proving equivariance.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# Signal and kernel
x = np.array([0.0, 0.0, 5.0, 2.0, 0.0, 0.0, 0.0])
w = np.array([1.0, -1.0])

# Standard convolution
z = np.convolve(x, w[::-1], mode='valid')

# Translated signal by 2 steps
x_shift = np.roll(x, 2)
z_shift = np.convolve(x_shift, w[::-1], mode='valid')

# Check inner slice shift
assert np.allclose(z[0:3], z_shift[2:5]), "Convolved output must shift identically!"
print("[P4 PASS] Translation equivariance verified:", z[:3], "->", z_shift[2:5])
```

### 🩺 Diagnostic Mini-Check
**Question:** Is a multi-layer perceptron (MLP) with dense weight matrices equivariant to spatial translations?  
<details><summary><b>Reveal Answer</b></summary><b>No.</b> Because each spatial input coordinate connects to neurons with distinct, unshared weight values, shifting the input activates completely different weight combinations, destroying equivariance.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $\mathcal{T}_v$ denote the spatial translation operator by vector $v \in \mathbb{Z}^d$:
$$(\mathcal{T}_v x)(u) = x(u - v)$$
- **Translation Equivariance:** An operator $f$ is equivariant to translation if translating the input before applying $f$ yields the exact same result as applying $f$ and then translating the output:
  $$f(\mathcal{T}_v x) = \mathcal{T}_v (f(x)) \iff f \circ \mathcal{T}_v = \mathcal{T}_v \circ f$$
- **Proof for Discrete Convolution:**
  Let $f_w(x) = w \star x$, where $(w \star x)(j) = \sum_r w(r) x(j + r)$.
  $$(f_w(\mathcal{T}_v x))(j) = \sum_r w(r) (\mathcal{T}_v x)(j + r) = \sum_r w(r) x(j + r - v) = (w \star x)(j - v) = (\mathcal{T}_v f_w(x))(j)$$
  Thus, discrete convolution commutes with translation.
- **Translation Invariance:** An operator $g$ is invariant if translating the input leaves the output unchanged:
  $$g(\mathcal{T}_v x) = g(x)$$
</details>

---

## 5. Discrete Spatial Filtering & 1D/2D Cross-Correlation

<a id="p5"></a>

### 👶 Physical Analogy & Intuition
Think of a rubber stamp with a carved silhouette of a leaf. To find where leaves appear on a page, you press the stamp against every spot on the paper. Where the ink aligns perfectly with existing marks, the stamp impression creates a dark, high-intensity response; where the paper is blank or has conflicting lines, the impression is weak. In machine learning, the rubber stamp is the convolutional kernel, and stamping across every position on the grid is discrete cross-correlation.

### 🔍 Plain-English Breakdown
- In deep learning libraries (PyTorch, TensorFlow), `nn.Conv2d` actually computes **discrete cross-correlation** (sliding inner product without kernel flipping).
- Since kernel weights are learnable parameters, spatial reflection is absorbed into optimization.
- The output dimensions follow $H_{\text{out}} = \lfloor \frac{H - k_h}{s_h} \rfloor + 1$ and $W_{\text{out}} = \lfloor \frac{W - k_w}{s_w} \rfloor + 1$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let input patch $X \in \mathbb{R}^{3 \times 3}$ and kernel $K \in \mathbb{R}^{2 \times 2}$ be:
$$X = \begin{bmatrix} 1 & 2 & 0 \\ 0 & 3 & 1 \\ 2 & 1 & 4 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & -1 \\ 2 & 0 \end{bmatrix}$$
With stride $s = 1$, output size is $(3 - 2 + 1) \times (3 - 2 + 1) = 2 \times 2$.
- **Position $(0, 0)$:** Sub-patch $X[0:2, 0:2] = \begin{bmatrix} 1 & 2 \\ 0 & 3 \end{bmatrix}$.
  $$Z_{0,0} = 1(1) + (-1)(2) + 2(0) + 0(3) = 1 - 2 + 0 + 0 = -1$$
- **Position $(0, 1)$:** Sub-patch $X[0:2, 1:3] = \begin{bmatrix} 2 & 0 \\ 3 & 1 \end{bmatrix}$.
  $$Z_{0,1} = 1(2) + (-1)(0) + 2(3) + 0(1) = 2 - 0 + 6 + 0 = 8$$
- **Position $(1, 0)$:** Sub-patch $X[1:3, 0:2] = \begin{bmatrix} 0 & 3 \\ 2 & 1 \end{bmatrix}$.
  $$Z_{1,0} = 1(0) + (-1)(3) + 2(2) + 0(1) = 0 - 3 + 4 + 0 = 1$$
- **Position $(1, 1)$:** Sub-patch $X[1:3, 1:3] = \begin{bmatrix} 3 & 1 \\ 1 & 4 \end{bmatrix}$.
  $$Z_{1,1} = 1(3) + (-1)(1) + 2(1) + 0(4) = 3 - 1 + 2 + 0 = 4$$

Resulting feature map:
$$Z = \begin{bmatrix} -1 & 8 \\ 1 & 4 \end{bmatrix}$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

X = np.array([
    [1.0, 2.0, 0.0],
    [0.0, 3.0, 1.0],
    [2.0, 1.0, 4.0]
])
K = np.array([
    [1.0, -1.0],
    [2.0,  0.0]
])

H_out = X.shape[0] - K.shape[0] + 1
W_out = X.shape[1] - K.shape[1] + 1
Z = np.zeros((H_out, W_out))

for i in range(H_out):
    for j in range(W_out):
        patch = X[i:i+K.shape[0], j:j+K.shape[1]]
        Z[i, j] = np.sum(patch * K)
        
expected = np.array([[-1.0, 8.0], [1.0, 4.0]])
assert np.allclose(Z, expected), f"Expected {expected}, got {Z}"
print("[P5 PASS] 2D discrete cross-correlation verified:\n", Z)
```

### 🩺 Diagnostic Mini-Check
**Question:** In classical computer vision, Sobel horizontal filter $K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ is applied to images. What does it compute? How does a CNN differ?  
<details><summary><b>Reveal Answer</b></summary>The Sobel filter computes the discrete spatial partial derivative $\frac{\partial I}{\partial x}$ to detect vertical edges. In a CNN, filter weights are not fixed by humans; they are free parameters initialized randomly and learned automatically via backpropagation.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

- **1D Discrete Cross-Correlation (Valid Padding, Stride $s$):**
  For input $x \in \mathbb{R}^d$ and kernel $w \in \mathbb{R}^k$:
  $$(w \star x)_j = \sum_{r=0}^{k-1} w_r \cdot x_{j \cdot s + r}$$
  Output length: $M = \lfloor \frac{d - k}{s} \rfloor + 1$.
- **2D Discrete Cross-Correlation (Image $X \in \mathbb{R}^{H \times W}$, Kernel $K \in \mathbb{R}^{k_h \times k_w}$):**
  $$(K \star X)_{i, j} = \sum_{u=0}^{k_h - 1} \sum_{v=0}^{k_w - 1} K_{u, v} \cdot X_{i \cdot s_h + u, j \cdot s_w + v}$$
  Output grid dimensions:
  $$H_{\text{out}} = \left\lfloor \frac{H - k_h}{s_h} \right\rfloor + 1, \quad W_{\text{out}} = \left\lfloor \frac{W - k_w}{s_w} \right\rfloor + 1$$
</details>

---

## 6. Bayesian Priors on Parameter Space: Soft Penalties vs Hard Dirac Delta Constraints

<a id="p6"></a>

### 👶 Physical Analogy & Intuition
Imagine hiking down a mountain path in the fog. 
- A **soft regularizer** is like an elastic tether tied to your waist: you can walk anywhere on the mountain, but straying far from basecamp requires pulling against increasing tension (Gaussian prior / $L_2$ weight decay).
- A **hard architectural regularizer** is a 20-foot concrete wall bordering the trail: you are strictly forbidden from stepping off the path, no matter how much you pull (Dirac delta prior / zeroed connections).
CNNs place concrete walls in parameter space, constraining optimization to search only within spatially meaningful subspaces.

### 🔍 Plain-English Breakdown
- In Bayesian Maximum A Posteriori (MAP) estimation, regularizers correspond to log-priors $\log p(\theta)$.
- Standard $L_2$ weight decay corresponds to a smooth Gaussian prior $p(\theta) = \mathcal{N}(0, \sigma^2 I)$.
- Hard architectural inductive biases (such as local receptive fields and weight tying) correspond to **singular Dirac delta priors** $p(w) = \delta_0(w)$, which restrict optimization to a low-dimensional manifold.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Consider a 2-parameter model $\theta = (\theta_1, \theta_2)$. Loss function is $\mathcal{L}(\theta) = (\theta_1 - 4)^2 + (\theta_2 - 4)^2$.
- **Unregularized minimum:** $\theta^* = (4, 4)$, Loss $= 0$.
- **Soft $L_2$ regularization ($\lambda = 0.5$):**
  $$\mathcal{L}_{\text{soft}}(\theta) = (\theta_1 - 4)^2 + (\theta_2 - 4)^2 + 0.5(\theta_1^2 + \theta_2^2)$$
  $$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial \theta_1} = 2(\theta_1 - 4) + \theta_1 = 3\theta_1 - 8 = 0 \implies \theta_1 = \frac{8}{3} \approx 2.67$$
  Optimal point: $\theta_{\text{MAP}} = (2.67, 2.67)$. Both parameters shrink toward zero, but neither is zero.
- **Hard Dirac Delta constraint on $\theta_2$ ($p(\theta_2) = \delta_0(\theta_2) \iff \theta_2 \equiv 0$):**
  $$\mathcal{L}_{\text{hard}}(\theta_1) = (\theta_1 - 4)^2 + (0 - 4)^2 = (\theta_1 - 4)^2 + 16$$
  Optimal point: $\theta_{\text{hard}} = (4, 0)$. Parameter $\theta_2$ is strictly 0 with probability 1.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# True target is (4, 4)
# 1. Soft L2 solution
lam = 0.5
theta_soft = 4.0 / (1.0 + 0.5 * lam)

# 2. Hard constraint (theta_2 = 0)
theta_hard = np.array([4.0, 0.0])

assert np.isclose(theta_soft, 8.0 / 3.0), f"Soft expected 2.667, got {theta_soft}"
assert theta_hard[1] == 0.0, "Hard constraint must be identically zero!"
print(f"[P6 PASS] Soft vs Hard priors verified: Soft={theta_soft:.3f}, Hard={theta_hard}")
```

### 🩺 Diagnostic Mini-Check
**Question:** Why can't standard $L_2$ weight decay transform an arbitrary fully-connected MLP into a Convolutional Neural Network during training?  
<details><summary><b>Reveal Answer</b></summary><b>$L_2$ weight decay softly penalizes all parameter magnitudes uniformly.</b> It lacks the structural geometric knowledge to force specific non-local connections to zero while simultaneously tying remaining weights across space. Only hard architectural constraints enforce Toeplitz geometry.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

In Maximum A Posteriori (MAP) parameter estimation:
$$\theta_{\text{MAP}} = \arg\max_\theta \left[ \sum_{i=1}^n \log p(y_i \mid x_i, \theta) + \log p(\theta) \right]$$
- **Soft Parameter Prior (Gaussian $L_2$ Regularization):**
  $$p(\theta) = \prod_{j=1}^P \frac{1}{\sqrt{2\pi \sigma_0^2}} \exp\left(-\frac{\theta_j^2}{2\sigma_0^2}\right)$$
  Taking logs yields the familiar continuous objective:
  $$\max_\theta \sum_{i=1}^n \log p(y_i \mid x_i, \theta) - \frac{\lambda}{2} \|\theta\|_2^2, \quad \lambda = \frac{1}{\sigma_0^2}$$
- **Hard Architectural Constraint (Dirac Delta Prior):**
  For unallowed connections $w_{ij}$, set the prior as a singular generalized function:
  $$p(w_{ij}) = \delta_0(w_{ij}) = \begin{cases} +\infty & w_{ij} = 0 \\ 0 & w_{ij} \neq 0 \end{cases}$$
  satisfying $\int_{\mathbb{R}} \delta_0(w) dw = 1$.
  The posterior probability is identically zero everywhere except on the linear subspace $\mathcal{M} = \{W \mid W_{ij} = 0 \ \forall (i, j) \notin \mathcal{E}\}$, unconditionally eliminating those parameters from optimization.
</details>
