# Prerequisites — Lec 44: CNNs as Regularized MLPs

> **Do this first.** Then open [NOTES.md](./NOTES.md) at the **Executive Summary** map.  
> Basics only — not a second lecture. They unlock words on the master map if you are rusty.  
> **How to read:** Use the **3-Minute Executive Fast-Track** below for a rapid ramp-up. Read the 👶 Physical Intuition and 💻 Python snippets in each pillar. Expand the 📐 formal calculus blocks only when you need deep mathematical proofs.

```
  After this warm-up you can say:

  "Multi-channel convolutions sum over all input channels to collapse them into one scalar."
  "Deep learning conv2d is cross-correlation; ERM absorbs any needed spatial reflection."
  "Output spatial resolution follows floor((P - k + 2p) / s) + 1."
  "Max pooling backpropagation routes 100% of error to the winning argmax position."
  "Transposed convolutions expand low-resolution bottlenecks back to full pixel canvases."
```

---

## ⚡ 3-Minute Executive Fast-Track

If you have only 3 minutes before starting the lecture, master this visual blueprint:

```
  MULTI-CHANNEL FORWARD CONVOLUTION (Channel Marginalization):
  Input X ∈ ℝ^(P × Q × R) ──► [ Bank of L Filters: k × k × R ] ──► Sum over R ──► Activation Z ∈ ℝ^(P_out × Q_out × L)
                                                                                        │
  SPATIAL DOWNSAMPLING & RECONSTRUCTION:                                               │
  Dense Class Map ◄── Transposed Conv Decoder ◄── Bottleneck ◄── Max/Avg Pooling ◄─────┘
  [P × Q × K]         (Upsample via Stride s)     (Latent)       (Subgradient Routing)
```

### 🧠 The 3 Core Mental Shifts
1. **Multi-Channel Marginalization:** A convolutional filter is not a 2D matrix; it is a 3D tensor block ($k \times k \times R$). Sliding it over a local patch computes a 3D Frobenius inner product that sums across all $R$ channels simultaneously, *marginalizing* (collapsing) the input channels into a single scalar per spatial location. To produce $L$ output channels, you must deploy a bank of $L$ distinct filters.
2. **Convolution vs. Cross-Correlation:** Deep learning `conv2d` does not flip the kernel across spatial axes—it computes sliding cross-correlation. This is harmless because neural network filter weights are initialized randomly and learned via Empirical Risk Minimization (ERM); gradient descent directly learns whichever orientation minimizes the loss.
3. **Argmax Routing in Max Pooling:** Max pooling is non-linear and non-differentiable at ties. Backpropagation resolves this using subgradient selection: 100% of the upstream error $\delta$ routes exclusively to the single winning coordinate that achieved the maximum during the forward pass, while all losing coordinates receive zero gradient.

### ⏱️ Instant Readiness Check
1. *In standard C-contiguous row-major layout, what is the 1D memory index formula for an element at $(i, j, c)$ in a tensor of shape $[P, Q, R]$?*  
   <details><summary><b>Reveal Answer</b></summary>$\text{flat\_idx} = i \cdot (Q \cdot R) + j \cdot R + c$.</details>
2. *When an input image has 32 channels and a convolutional layer uses 64 filters of size $3 \times 3$, what is the exact shape of each individual filter tensor?*  
   <details><summary><b>Reveal Answer</b></summary><b>$[3, 3, 32]$</b> (each filter must span all 32 input channels so they can be summed and marginalized into 1 output slice).</details>
3. *An input image has height $P = 224$. Convolving with filter size $k = 7$, stride $s = 2$, and padding $p = 3$ yields what output height $P_{\text{out}}$?*  
   <details><summary><b>Reveal Answer</b></summary><b>$112$</b> ($\lfloor \frac{224 - 7 + 6}{2} \rfloor + 1 = 111 + 1 = 112$, the classic ResNet-50 conv1 spatial dimension).</details>

---

## Math Terminology Rosetta Stone

Before diving into the foundational pillars, use this reference table to decode mathematical shorthand and notation into plain English, spoken phonetics, and software implementations.

| Mathematical Symbol | Formal Concept | Spoken English (Phonetics) | Plain-English Intuition | Operational / Engineering Role | Master Concept Link |
|:-------------------:|:---------------|:---------------------------|:------------------------|:-------------------------------|:--------------------|
| $\mathcal{X} \in \mathbb{R}^{P \times Q \times R}$ | Multi-Channel Input Tensor | "CAL-i-graf-ik EKS in AHR to the PEE by KYOO by AHR" | Multi-spectral image or volume stacked by color/sensor channels | Spatial grid $P \times Q$ with $R$ physical/spectral channels (e.g., RGB, MRI) | [`Tensors & Shapes`](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md) |
| $\mathcal{W} \in \mathbb{R}^{L \times k \times k \times R}$ | Convolutional Filter Bank | "CAL-i-graf-ik DOUBLE-yoo in AHR to the EL by KAY by KAY by AHR" | Set of feature-detecting magnifying glasses sliding together | Bank of $L$ spatial filters spanning all $R$ input channels simultaneously | [`Convolution & Pooling`](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) |
| $\langle \mathcal{A}, \mathcal{B} \rangle_F$ | Frobenius Tensor Inner Product | "fro-BEN-ee-us IN-ner PROD-ukt of A and B" | Overlap score between kernel and image patch summed over channels | Component-wise sum of products collapsing spatial-channel patch to a scalar | [`Vector Norms & Inner Products`](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/02-Vector_Norms_and_Inner_Products.md) |
| $P_{\text{out}} = \lfloor \frac{P - k + 2p}{s} \rfloor + 1$ | Output Spatial Dimension | "PEE out EQUALS FLOOR of PEE MINUS KAY PLUS TWO PEE over ESS PLUS ONE" | Formula counting how many strides fit across the padded canvas | Formula computing activation grid resolution under stride $s$ and padding $p$ | [`Convolution & Pooling`](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) |
| $\sigma(z)$ | Pointwise Non-Linear Activation | "SIG-muh of ZEE" | Bent-line switch (ReLU) preventing multi-layer collapse to a single linear layer | Element-wise squashing (ReLU, GELU) preventing deep collapse to a single linear map | [`Activation Functions`](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/05-Activation_Functions.md) |
| $\mathcal{P}_{\text{avg}}(X)$ | Average Pooling Operator | "PEE sub AV-er-ij of EKS" | Blurring/smoothing window that replaces a block with its mean value | Fixed low-pass convolution with uniform kernel weights $1/k_p^2$ | [`Convolution & Pooling`](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) |
| $\mathcal{P}_{\text{max}}(X)$ | Max Pooling Operator | "PEE sub MAKS of EKS" | Peak-detector window keeping only the strongest feature response | Non-linear spatial downsampling preserving maximum feature response | [`Convolution & Pooling`](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) |
| $W^T \star X$ | Transposed Convolution (Upsampling) | "DOUBLE-yoo TRANS-pose STAR EKS" | Cinema projector expanding coarse features back to high-resolution canvas | Fractionally strided convolution expanding spatial grid in decoder paths | [`Convolution & Pooling`](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) |

---

## 🌉 Curriculum & Sibling Course Prerequisite Bridges

| Prerequisite Topic | Sibling / Preceding Source | Specific Module / Lecture | Relevance to Lecture 44 |
|:-------------------|:--------------------------|:--------------------------|:------------------------|
| Multi-Layer Perceptrons & UAT | `Mathematical-foundation-ml` | [`54-Lec41`](../54-Lec41-Neural-Networks-UAT/) | Establishes existential density of single-hidden-layer MLPs |
| Error Backpropagation & Chain Rule | `Mathematical-foundation-ml` | [`55-Lec42`](../55-Lec42-ERM-Neural-Networks-Backpropagation/) | Provides multivariable reverse-mode AD framework |
| 1D Local Receptive Fields & Parameter Sharing | `Mathematical-foundation-ml` | [`56-Lec43`](../56-Lec43-Local-Receptive-Field-Parameter-Sharing/) | Direct conceptual precursor defining CNN inductive bias |
| Tensors & Shape Manipulations | `MathsTerms` | [`01-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md`](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md) | Core tensor algebra for multi-dimensional convolutional banks |
| Convolution & Pooling Operators | `MathsTerms` | [`06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md`](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) | Comprehensive architectural reference for 2D filtering |

---

## 1. Multi-Dimensional Tensors, Memory Layouts, & Tensor Contractions

<a id="p1"></a>

### 👶 Physical Analogy & Intuition
Imagine an industrial warehouse with aisles ($P$), shelves ($Q$), and bin compartments ($R$). Computer memory is a single straight conveyor belt on the floor. To store 3D bins on a 1D belt, we lay out aisle 0 shelf 0 bins, then shelf 1 bins, then aisle 1 bins in sequence.

### 🔍 Plain-English Breakdown
- A multi-channel image or medical volume is formally a higher-order tensor $\mathcal{X} \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_N}$.
- Under the hood, all computer memory is linear 1D RAM.
- A tensor $\mathcal{X} \in \mathbb{R}^{P \times Q \times R}$ is mapped to memory via row-major (C-contiguous) ordering through the bijection index formula:
  $$\text{flat\_idx}(i, j, c) = i \cdot (Q \cdot R) + j \cdot R + c$$

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Consider a small tensor with height $P = 2$, width $Q = 2$, and channels $R = 3$:
$$\mathcal{X} = \begin{bmatrix}
\begin{pmatrix} 1 & 2 & 3 \end{pmatrix} & \begin{pmatrix} 4 & 5 & 6 \end{pmatrix} \\
\begin{pmatrix} 7 & 8 & 9 \end{pmatrix} & \begin{pmatrix} 10 & 11 & 12 \end{pmatrix}
\end{bmatrix}$$
The element at spatial coordinate $(i=1, j=0)$ in the blue channel ($c=2$) has value $9$. Its 1D flattened index is:
$$\text{flat\_idx}(1, 0, 2) = 1 \cdot (2 \cdot 3) + 0 \cdot 3 + 2 = 6 + 0 + 2 = 8$$
Verifying in zero-indexed list: $[1, 2, 3, 4, 5, 6, 7, 8, \mathbf{9}, 10, 11, 12]$, element at index 8 is indeed $9$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# Assert memory isomorphism
X = np.arange(1, 13).reshape(2, 2, 3)
assert X[1, 0, 2] == 9
assert X.ravel()[8] == 9
print("[P1 PASS] Tensor memory layout isomorphism verified:", X[1, 0, 2])
```

### 🩺 Diagnostic Mini-Check
**Question:** In an RGB image of size $100 \times 100 \times 3$, what is the 1D flattened memory index of the green channel ($c=1$) at row $i=20$, column $j=50$ under C-contiguous layout?  
<details><summary><b>Reveal Answer</b></summary>$\text{flat\_idx}(20, 50, 1) = 20 \cdot (100 \cdot 3) + 50 \cdot 3 + 1 = 6000 + 150 + 1 = \mathbf{6151}$.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_K}$ be a rank-$K$ tensor. The memory stride vector $s \in \mathbb{Z}^K$ for C-contiguous storage is defined recursively:
$$s_K = 1, \quad s_k = \prod_{m=k+1}^K d_m \quad \forall k \in \{1, \dots, K-1\}$$
The memory offset for multi-index $(i_1, i_2, \dots, i_K)$ is given by the inner product $\sum_{k=1}^K i_k s_k$.
Tensor contraction along mode $k$ against tensor $\mathcal{U}$ corresponds to matrix-matrix GEMM after permuting and reshaping contiguous strides.
</details>

---

## 2. 2D Spatial Cross-Correlation vs Mathematical Convolution (Filter Reflection)

<a id="p2"></a>

### 👶 Physical Analogy & Intuition
Imagine stamping paper with a rubber seal. If you stamp with the seal as carved, you compute cross-correlation. If you flip the stamp horizontally and vertically before stamping, you compute true mathematical convolution. Because neural networks learn the rubber stamp's engraving from scratch, whether the stamp was pre-flipped or not is irrelevant—the network simply learns the final desired orientation.

### 🔍 Plain-English Breakdown
- In signal processing, true convolution flips the kernel across both axes:
  $$(f * g)[i, j] = \sum_{u} \sum_{v} f[u, v] \, g[i - u, j - v]$$
- Deep learning libraries compute **cross-correlation** (no reflection):
  $$(X \star K)[i, j] = \sum_{u=0}^{k_h-1} \sum_{v=0}^{k_w-1} X[i + u, j + v] \, K[u, v]$$
- Because kernel weights are learnable parameters optimized via ERM, any required spatial flip is learned automatically.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let input patch $X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and kernel $K = \begin{bmatrix} 0 & 1 \\ -1 & 2 \end{bmatrix}$.
- **Cross-correlation (Deep Learning "Conv"):**
  $$\langle X, K \rangle_F = (1)(0) + (2)(1) + (3)(-1) + (4)(2) = 0 + 2 - 3 + 8 = 7$$
- **True Mathematical Convolution (Flipped $K_{\text{flip}} = \begin{bmatrix} 2 & -1 \\ 1 & 0 \end{bmatrix}$):**
  $$\langle X, K_{\text{flip}} \rangle_F = (1)(2) + (2)(-1) + (3)(1) + (4)(0) = 2 - 2 + 3 + 0 = 3$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

X = np.array([[1.0, 2.0], [3.0, 4.0]])
K = np.array([[0.0, 1.0], [-1.0, 2.0]])

# Deep learning sliding inner product
corr = np.sum(X * K)
assert corr == 7.0

# Mathematical convolution with double-flipped kernel
K_flip = np.flip(K)
conv = np.sum(X * K_flip)
assert conv == 3.0
print("[P2 PASS] Cross-correlation vs Convolution verified: corr=7.0, conv=3.0")
```

### 🩺 Diagnostic Mini-Check
**Question:** If your PyTorch convolutional layer learns an asymmetric filter $K = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, what physical filter orientation did gradient descent directly learn without spatial reflection?  
<details><summary><b>Reveal Answer</b></summary><b>It learned the cross-correlation template directly.</b> If the optimal feature detector required a flipped orientation, ERM simply flipped the weight values during training without requiring manual inversion.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let involution operator $\mathcal{R}$ reflect a discrete 2D kernel: $(\mathcal{R} K)[u, v] = K[-u, -v]$.
The algebraic relationship between 2D convolution and cross-correlation is:
$$(X \star K)[i, j] = (X * \mathcal{R} K)[i, j]$$
Because the hypothesis space $\mathcal{H}_{\text{corr}} = \{X \mapsto \sigma(X \star K + b) : K \in \mathbb{R}^{k \times k}\}$ is isometric to $\mathcal{H}_{\text{conv}} = \{X \mapsto \sigma(X * K' + b) : K' \in \mathbb{R}^{k \times k}\}$ under the bijection $K' = \mathcal{R} K$, both function classes have identical capacity, VC-dimension, and Rademacher complexity.
</details>

---

## 3. Tensor Inner Products & Channel Marginalization

<a id="p3"></a>

### 👶 Physical Analogy & Intuition
Imagine a multi-sensor weather station measuring temperature, pressure, and humidity simultaneously. A weather-front detector cannot look at temperature alone; it places a 3-sensor probe over a local county, multiplies each sensor by its respective weight, and sums all sensor signals into a single scalar air-mass score. The multi-channel input is marginalized into a single scalar response.

### 🔍 Plain-English Breakdown
- When processing an input tensor $\mathcal{X} \in \mathbb{R}^{P \times Q \times R}$ with $R$ channels, a convolutional filter is a 3D rank-3 tensor $\mathcal{K} \in \mathbb{R}^{k \times k \times R}$.
- At each spatial window, $k \times k \times R$ multiplications are computed and **summed together across all channels**.
- The input channel dimension $R$ is completely **marginalized (collapsed)** into 1 scalar pre-activation.
- To produce $L$ output channels, the layer deploys a bank of $L$ separate 3D filters.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let patch size be $k \times k = 2 \times 2$ across $R = 2$ channels:
$$\text{Channel 1: } X_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad K_1 = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}$$
$$\text{Channel 2: } X_2 = \begin{bmatrix} 2 & 3 \\ 1 & 1 \end{bmatrix}, \quad K_2 = \begin{bmatrix} -1 & 0 \\ 1 & 2 \end{bmatrix}, \quad \text{Bias } b = -1.0$$
- Sum on Channel 1: $(1)(2) + (0)(1) + (0)(0) + (1)(3) = 2 + 3 = 5$.
- Sum on Channel 2: $(2)(-1) + (3)(0) + (1)(1) + (1)(2) = -2 + 0 + 1 + 2 = 1$.
- Total pre-activation: $z = \text{Sum}_1 + \text{Sum}_2 + b = 5 + 1 - 1.0 = 5.0$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

X = np.stack([[[1, 0], [0, 1]], [[2, 3], [1, 1]]], axis=-1)  # [2, 2, 2]
K = np.stack([[[2, 1], [0, 3]], [[-1, 0], [1, 2]]], axis=-1)  # [2, 2, 2]
b = -1.0

z = np.sum(X * K) + b
assert z == 5.0
print("[P3 PASS] Multi-channel inner product verified:", z)
```

### 🩺 Diagnostic Mini-Check
**Question:** When a $3 \times 3 \times 64$ filter convolves over a local receptive field of a 64-channel activation map, how many scalar products are summed to produce the single output pre-activation?  
<details><summary><b>Reveal Answer</b></summary><b>$3 \times 3 \times 64 = 576$ scalar multiplications</b> are summed into 1 scalar pre-activation (plus 1 bias).</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $\mathcal{X} \in \mathbb{R}^{P \times Q \times R}$ and let filter bank $\mathcal{W} \in \mathbb{R}^{L \times k \times k \times R}$ with bias $b \in \mathbb{R}^L$.
The forward 3D convolution mapping is defined component-wise for filter $l \in \{1, \dots, L\}$:
$$\mathcal{Z}[i, j, l] = \sum_{c=1}^R \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} \mathcal{W}[l, u, v, c] \, \mathcal{X}[i \cdot s + u, j \cdot s + v, c] + b[l]$$
The summation over $c \in \{1, \dots, R\}$ contracts the channel dimension of the input tensor, proving that output depth $L$ is strictly determined by the cardinality of the filter bank, independent of $R$.
</details>

---

## 4. Spatial Resolution Arithmetic: Stride, Dilation, and Zero-Padding

<a id="p4"></a>

### 👶 Physical Analogy & Intuition
Think of stepping across paving stones. If the path has length $P$, your shoes span $k$ stones, and you jump $s$ stones per step, padding $p$ adds extra stones at the start and end so your stride fits without tripping at the curb.

### 🔍 Plain-English Breakdown
The spatial dimensions of an activation map after convolution depend strictly on five quantities: input height $P$, kernel size $k$, stride $s$, symmetric zero-padding $p$, and dilation $d$:
$$P_{\text{out}} = \left\lfloor \frac{P + 2p - d(k - 1) - 1}{s} \right\rfloor + 1$$
For standard non-dilated convolution ($d = 1$), this reduces to the classical formula:
$$P_{\text{out}} = \left\lfloor \frac{P - k + 2p}{s} \right\rfloor + 1$$

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let input image size be $P = 7$, filter size $k = 3$, stride $s = 2$, and padding $p = 1$:
$$P_{\text{out}} = \left\lfloor \frac{7 - 3 + 2(1)}{2} \right\rfloor + 1 = \left\lfloor \frac{6}{2} \right\rfloor + 1 = 3 + 1 = 4$$
The valid window starting indices in the padded array of length $7 + 2 = 9$ (indices $0 \dots 8$) with step $s = 2$ are:
- Window 0: indices $0, 1, 2$ (starts at 0)
- Window 1: indices $2, 3, 4$ (starts at 2)
- Window 2: indices $4, 5, 6$ (starts at 4)
- Window 3: indices $6, 7, 8$ (starts at 6)
Total windows = 4, exactly matching the formula.

### 💻 Standalone Executable Python Verification
```python
P, k, s, p = 7, 3, 2, 1
P_out = (P - k + 2 * p) // s + 1
assert P_out == 4
print("[P4 PASS] Spatial resolution arithmetic verified: P_out =", P_out)
```

### 🩺 Diagnostic Mini-Check
**Question:** For input $P=32$, filter size $k=5$, stride $s=1$, what symmetric padding $p$ ensures "Same" spatial resolution ($P_{\text{out}} = 32$)?  
<details><summary><b>Reveal Answer</b></summary>$p = \frac{k - 1}{2} = \frac{5 - 1}{2} = \mathbf{2}$. Verification: $\lfloor \frac{32 - 5 + 4}{1} \rfloor + 1 = 31 + 1 = 32$.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Given input grid index set $\mathcal{I}_{\text{in}} = \{0, 1, \dots, P-1\}$, symmetric padding expands the support to $\{-p, \dots, P - 1 + p\}$ of total cardinality $P + 2p$.
A filter with dilation $d$ spans effective receptive footprint $\tilde{k} = d(k - 1) + 1$.
The number of non-overlapping window translations of step size $s$ fitting inside this support before exceeding the boundary is bounded by the integer floor function:
$$N_{\text{steps}} = \max\left\{m \in \mathbb{N} : m \cdot s + \tilde{k} - 1 \le P + 2p - 1\right\} = \left\lfloor \frac{P + 2p - \tilde{k}}{s} \right\rfloor$$
Adding 1 for the initial window at $m=0$ gives $P_{\text{out}} = N_{\text{steps}} + 1$.
</details>

---

## 5. Downsampling Operators & Subgradient Backpropagation (Avg vs Max Pooling)

<a id="p5"></a>

### 👶 Physical Analogy & Intuition
Imagine water flowing through a drainage basin. Average pooling acts like a porous sponge distributing water evenly across all pores; max pooling acts like a high-water spillway valve where only the single highest wave crest opens the gate, and upstream backpropagation flows exclusively back through that open gate.

### 🔍 Plain-English Breakdown
Spatial downsampling layers reduce feature map resolution to lower computational burden and enforce translation tolerance:
1. **Average Pooling (Fixed Linear Filter):**
   $$\mathcal{P}_{\text{avg}}(X)[i, j] = \frac{1}{k_p^2} \sum_{u=0}^{k_p-1} \sum_{v=0}^{k_p-1} X[i \cdot s_p + u, j \cdot s_p + v]$$
   Distributes error gradient uniformly: $\frac{\partial \mathcal{L}}{\partial X} = \frac{\delta}{k_p^2}$.
2. **Max Pooling (Non-Linear Subgradient):**
   $$\mathcal{P}_{\text{max}}(X)[i, j] = \max_{u, v} X[i \cdot s_p + u, j \cdot s_p + v]$$
   Employs **argmax routing**: 100% of gradient goes to the maximum element, 0% to the rest.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Consider patch $X = \begin{bmatrix} 2.0 & 8.0 \\ 1.0 & 4.0 \end{bmatrix}$ with upstream sensitivity $\delta = 10.0$:
- **Average Pooling ($2 \times 2$):** Forward output is $\frac{2 + 8 + 1 + 4}{4} = 3.75$.
  Gradient matrix is $\begin{bmatrix} 2.5 & 2.5 \\ 2.5 & 2.5 \end{bmatrix}$.
- **Max Pooling ($2 \times 2$):** Forward output is $\max(X) = 8.0$ at coordinate $(0, 1)$.
  Gradient matrix is $\begin{bmatrix} 0.0 & 10.0 \\ 0.0 & 0.0 \end{bmatrix}$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

X = np.array([[2.0, 8.0], [1.0, 4.0]])
delta = 10.0

# Avg Pool Gradient
dX_avg = np.full_like(X, delta / 4.0)
assert np.allclose(dX_avg, 2.5)

# Max Pool Gradient
dX_max = np.zeros_like(X)
argmax_pos = np.unravel_index(np.argmax(X), X.shape)
dX_max[argmax_pos] = delta
assert dX_max[0, 1] == 10.0 and dX_max[0, 0] == 0.0
print("[P5 PASS] Average & Max Pooling gradient routing verified.")
```

### 🩺 Diagnostic Mini-Check
**Question:** In a $2 \times 2$ max pooling window with values $[1.0, 7.0, 3.0, 2.0]$ and upstream gradient $\delta = 4.0$, what is the gradient assigned to the input element with value $1.0$?  
<details><summary><b>Reveal Answer</b></summary><b>$0.0$.</b> Under argmax subgradient routing, 100% of the upstream gradient routes to the maximum element ($7.0$), and all non-maximal elements receive zero.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $f(x) = \max_{i \in \{1, \dots, N\}} x_i$. The subdifferential $\partial f(x)$ is the convex hull of standard basis vectors corresponding to active maximizing indices:
$$\partial f(x) = \operatorname{conv}\left\{e_k : x_k = \max_{j} x_j\right\}$$
When the maximum is unique (almost everywhere with respect to Lebesgue measure on $\mathbb{R}^N$), the subdifferential is a singleton: $\nabla f(x) = e_{\arg\max(x)}$.
This justifies deterministic single-path backward routing in deep learning autograd engines.
</details>

---

## 6. Inverse Spatial Transformations & Transposed Convolutions (Fractional Stride)

<a id="p6"></a>

### 👶 Physical Analogy & Intuition
Imagine an optical projector projecting a miniature slide film onto a large cinema screen. The slide is dilated with spacing, and the projection lens (kernel) broadcasts each tiny pixel into a wide overlapping illuminated pattern on the wall.

### 🔍 Plain-English Breakdown
- In dense prediction (semantic segmentation, generative decoders, U-Net), networks must upsample coarse latent bottleneck features back to full image resolution.
- A **Transposed Convolution** is the adjoint (matrix transpose) operator of a forward convolution:
  $$\text{Forward: } y = W x \implies \text{Transposed: } \tilde{x} = W^T y$$
- For stride $s > 1$, it inserts $s - 1$ zeros between input pixels and convolves with kernel $K$.
- Output spatial dimension expands: $P_{\text{out}} = (P_{\text{in}} - 1) \cdot s - 2p + k$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let input length be $P_{\text{in}} = 2$, kernel size $k = 3$, stride $s = 2$, and padding $p = 0$:
$$P_{\text{out}} = (2 - 1) \cdot 2 - 0 + 3 = 2 + 3 = 5$$
Input $[x_0, x_1]$ is dilated with $s - 1 = 1$ zero: $[x_0, 0, x_1]$. Convolving with filter $[w_0, w_1, w_2]$ of length 3 yields an output of length $3 + 3 - 1 = 5$ elements.

### 💻 Standalone Executable Python Verification
```python
P_in, k, s, p = 2, 3, 2, 0
P_out = (P_in - 1) * s - 2 * p + k
assert P_out == 5
print("[P6 PASS] Transposed convolution output expansion verified: P_out =", P_out)
```

### 🩺 Diagnostic Mini-Check
**Question:** If a feature map has spatial size $P_{\text{in}} = 7$, transposed convolution with kernel $k=4$, stride $s=2$, padding $p=1$ yields what output dimension?  
<details><summary><b>Reveal Answer</b></summary>$P_{\text{out}} = (7 - 1) \cdot 2 - 2(1) + 4 = 12 - 2 + 4 = \mathbf{14}$.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $W \in \mathbb{R}^{M \times N}$ be the sparse Toeplitz matrix implementing forward strided convolution with stride $s > 1$, mapping $x \in \mathbb{R}^N \mapsto y \in \mathbb{R}^M$ where $M < N$.
The adjoint operator $W^T \in \mathbb{R}^{N \times M}$ maps low-dimensional feature vectors $y \in \mathbb{R}^M$ back to high-dimensional space $\mathbb{R}^N$.
By the properties of adjoints, $\langle W x, y \rangle_{\mathbb{R}^M} = \langle x, W^T y \rangle_{\mathbb{R}^N}$, proving that backward propagation through a forward convolution is computationally identical to the forward pass through a transposed convolution.
</details>

---

## Diagnostic Test Suite

Run this self-contained script to verify your numerical and mathematical readiness before proceeding to [NOTES.md](./NOTES.md).

```python
#!/usr/bin/env python3
"""
Diagnostic Pre-computation Test Suite for Lecture 44 Prerequisites.
"""
import numpy as np

def run_diagnostics():
    print("[DIAGNOSTIC] Running 6 mathematical prerequisite checks...")

    # Check 1: Tensor Indexing Isomorphism
    T = np.arange(24).reshape(2, 3, 4)
    assert T[1, 2, 3] == 23
    assert T.ravel()[1 * 12 + 2 * 4 + 3] == 23
    print("  [PASS] Pillar 1: Tensor memory layout validated.")

    # Check 2: Cross-Correlation vs Convolution
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    K = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert np.sum(X * K) == 9.0  # (1*1 + 4*2)
    assert np.sum(X * np.flip(K)) == 6.0  # (1*2 + 4*1)
    print("  [PASS] Pillar 2: Cross-correlation vs convolution distinction validated.")

    # Check 3: Multi-channel Marginalization
    X_mc = np.ones((2, 2, 3))
    K_mc = np.full((2, 2, 3), 0.5)
    z = np.sum(X_mc * K_mc)
    assert z == 6.0  # 2 * 2 * 3 * 0.5
    print("  [PASS] Pillar 3: Multi-channel inner product marginalization validated.")

    # Check 4: Spatial Dimension Floor Recurrence
    P_in, k, s, p = 224, 7, 2, 3
    P_out = (P_in - k + 2 * p) // s + 1
    assert P_out == 112  # Classic ResNet-50 conv1 spatial dimension!
    print("  [PASS] Pillar 4: Spatial dimension arithmetic validated.")

    # Check 5: Pooling Subgradient Masks
    patch = np.array([[1.0, 5.0], [2.0, 3.0]])
    d_out = 4.0
    dx_max = np.zeros_like(patch)
    dx_max[np.unravel_index(np.argmax(patch), patch.shape)] = d_out
    assert dx_max[0, 1] == 4.0 and np.sum(dx_max) == 4.0
    print("  [PASS] Pillar 5: Max pooling argmax subgradient routing validated.")

    # Check 6: Transposed Convolution Expansion
    P_in, k, s, p = 14, 4, 2, 1
    P_out = (P_in - 1) * s - 2 * p + k
    assert P_out == 28  # Classic 2x spatial upsampling
    print("  [PASS] Pillar 6: Transposed convolution spatial upsampling validated.")

    print("\n[SUCCESS] All prerequisite diagnostics passed cleanly.")

if __name__ == "__main__":
    run_diagnostics()
```
