# Rapid Revision Formulae Sheet & Dimensionality Ledger

> **Package:** `56-Lec43-Local-Receptive-Field-Parameter-Sharing`  
> **Course:** NPTEL / IISc — Mathematical Foundations of Machine Learning  
> **Instructor:** Prof. Prathosh A P (IISc Bengaluru)  
> **Skill Standard:** Canonical 7-Pillar Production Learning Suite (`/youtube-lecture-tutor`)

---

## 1. Master Equations Index

### 1. 1D Discrete Cross-Correlation (Local RF + Parameter Sharing)
$$z_j = \sum_{r=0}^{k-1} w_r \cdot x_{j \cdot s + r} + b \quad \forall j \in \{0, 1, \dots, M-1\}$$
where $M = \left\lfloor \frac{d - k}{s} \right\rfloor + 1$.

### 2. 2D Discrete Spatial Cross-Correlation
$$Z_{i, j} = \sum_{u=0}^{k_h - 1} \sum_{v=0}^{k_w - 1} K_{u, v} \cdot X_{i \cdot s_h + u, \; j \cdot s_w + v} + b$$
where $H_{\text{out}} = \left\lfloor \frac{H - k_h}{s_h} \right\rfloor + 1$ and $W_{\text{out}} = \left\lfloor \frac{W - k_w}{s_w} \right\rfloor + 1$.

### 3. Effective Receptive Field Expansion Recurrence
$$RF_l = RF_{l-1} + (k_l - 1) \prod_{i=1}^{l-1} s_i, \quad RF_0 = 1$$
For $L$ identical layers with kernel size $k$ and unit stride $s = 1$:
$$RF_L = 1 + L(k - 1)$$

### 4. Shared Weight Gradient (Multivariable Chain Rule Accumulation)
$$\frac{\partial \mathcal{L}}{\partial w_r} = \sum_{j=0}^{M-1} \delta_j \cdot x_{j \cdot s + r}, \quad \text{where } \delta_j \equiv \frac{\partial \mathcal{L}}{\partial z_j}$$

### 5. Shared Bias Gradient
$$\frac{\partial \mathcal{L}}{\partial b} = \sum_{j=0}^{M-1} \delta_j$$

### 6. Input Sensitivity Gradient (Transposed Convolution)
$$\frac{\partial \mathcal{L}}{\partial x_i} = \sum_{j, r: \; j \cdot s + r = i} \delta_j \cdot w_r$$

---

## 2. Input/Output Dimensionality & Tensor Shapes Ledger

| Operation / Object | Mathematical Symbol | Tensor Shape (Batch Mode) | NumPy / PyTorch Dimension |
|:-------------------|:-------------------:|:-------------------------:|:--------------------------|
| 1D Input Tensor | $x$ | $[B, d]$ or $[B, C_{\text{in}}, d]$ | `(batch_size, channels, length)` |
| 1D Convolutional Kernel | $w$ | $[k]$ or $[C_{\text{out}}, C_{\text{in}}, k]$ | `(out_channels, in_channels, kernel_size)` |
| 1D Pre-activation Output | $z$ | $[B, M]$ or $[B, C_{\text{out}}, M]$ | `(batch_size, out_channels, out_length)` |
| Banded Toeplitz Matrix | $W_{\text{Toeplitz}}$ | $[M, d]$ | `(out_length, in_length)` |
| Upstream Error Sensitivity | $\delta$ | $[B, M]$ or $[B, C_{\text{out}}, M]$ | `(batch_size, out_channels, out_length)` |
| Kernel Gradient | $\nabla_w \mathcal{L}$ | $[k]$ or $[C_{\text{out}}, C_{\text{in}}, k]$ | `(out_channels, in_channels, kernel_size)` |
| 2D Image Tensor | $X$ | $[B, C_{\text{in}}, H, W]$ | `(B, C, H, W)` |
| 2D Kernel Tensor | $K$ | $[C_{\text{out}}, C_{\text{in}}, k_h, k_w]$ | `(C_out, C_in, k_h, k_w)` |

---

## 3. Mathematical Guarantees & Equivalence Theorems

1. **Toeplitz Equivalence Theorem:** A 1D discrete cross-correlation with kernel $w \in \mathbb{R}^k$ and unit stride $s = 1$ is mathematically identical to matrix multiplication by a banded Toeplitz matrix $W \in \mathbb{R}^{M \times d}$:
   $$w \star x + b \equiv W_{\text{Toeplitz}} x + b \mathbf{1}_M$$
2. **Translation Equivariance Guarantee:** For any spatial shift operator $T_v$ and cross-correlation $f_w(x) = w \star x$:
   $$f_w(T_v x) = T_v f_w(x) \iff f_w \circ T_v = T_v \circ f_w$$
3. **Dirac Delta Prior Representation:** Eliminating connection $w_{ij}$ corresponds to MAP estimation under singular measure $p(w_{ij}) = \delta_0(w_{ij})$.
4. **Universal Approximation for CNNs (Zhou 2020):** Deep convolutional networks with sufficiently wide channels and non-linear activations are dense in $C(K)$ for any compact $K \subset \mathbb{R}^d$.

---

## 4. Contrastive Quick-Decision Guide

| Engineering Scenario | Recommended Architecture | Mathematical Rationale |
|:---------------------|:-------------------------|:-----------------------|
| 1D/2D/3D Grid Data (Images, Audio, Lattices) | **Convolutional Neural Network (CNN)** | Exploits local spatial correlation; translation equivariance; $\mathcal{O}(k)$ parameters. |
| Tabular / Unstructured Heterogeneous Features | **Multi-Layer Perceptron (MLP)** | No geometric distance or spatial stationarity between tabular feature columns. |
| Arbitrary Input Canvas Resolution at Test Time | **Fully Convolutional Network (FCN)** | Kernel $w \in \mathbb{R}^k$ evaluates on any input dimension $d \ge k$. |
| Global Coordinate-Dependent Classification (e.g. Centered Facial Landmarks) | **CNN + Coordinate Convolutions (CoordConv)** | Injects Cartesian coordinates $(u, v)$ as explicit input channels to break translation equivariance. |

---

## 5. Numerical Stability, Memory, & Hardware Traps

- **Boundary Truncation Trap (Valid Padding):** Stride and filter width without padding reduce output dimension by $k - 1$. After $L$ layers, spatial resolution collapses by $L(k - 1)$. Use explicit zero-padding ($p = \lfloor k/2 \rfloor$) to preserve spatial canvas size.
- **Transposed Convolution Checkerboard Artifacts:** During gradient backpropagation or generative upsampling, strides $s > 1$ with kernel sizes $k$ not divisible by $s$ produce uneven overlap patterns (checkerboard artifacts).
- **Float Precision Accumulation in Gradient Pooling:** Summing $\sum_{j=0}^{M-1} \delta_j x_{j \cdot s + r}$ across high-resolution feature maps ($M \ge 10^5$) in 16-bit floating point (FP16) causes catastrophic underflow/loss of precision. Accumulate gradient reductions in FP32.
