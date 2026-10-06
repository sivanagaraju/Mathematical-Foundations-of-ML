# Formulae Sheet — Tutorial 14 Part 1: Convolutional Neural Networks (CNNs)

This reference sheet consolidates all mathematical formulas, tensor shapes, parameter counts, and operational mechanics for 2D Convolutional Neural Networks.

---

## 1. 2D Discrete Cross-Correlation (Convolution Operator)

For an input volume $\mathbf{X} \in \mathbb{R}^{D_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$, filter bank $\mathbf{W} \in \mathbb{R}^{K \times D_{\text{in}} \times F \times F}$, bias vector $\mathbf{b} \in \mathbb{R}^K$, stride $S \in \mathbb{N}_{\ge 1}$, and zero padding $P \in \mathbb{N}_{\ge 0}$:

$$Y_k(i, j) = b_k + \sum_{c=1}^{D_{\text{in}}} \sum_{u=0}^{F-1} \sum_{v=0}^{F-1} \tilde{X}_c(i \cdot S + u, \; j \cdot S + v) \cdot W_{k, c}(u, v)$$

where $\tilde{\mathbf{X}} \in \mathbb{R}^{D_{\text{in}} \times (H_{\text{in}} + 2P) \times (W_{\text{in}} + 2P)}$ is the zero-padded input volume:
$$\tilde{X}_c(r, c) = \begin{cases} X_c(r - P, c - P) & \text{if } P \le r < H_{\text{in}} + P \text{ and } P \le c < W_{\text{in}} + P \\ 0 & \text{otherwise} \end{cases}$$

---

## 2. Spatial Dimension Arithmetic

### Height and Width
$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - F + 2P}{S} \right\rfloor + 1$$
$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - F + 2P}{S} \right\rfloor + 1$$

### Channel Depth
$$D_{\text{out}} = K \quad (\text{number of filters in the layer})$$

### Padding Configurations
- **Valid Padding ($P = 0$):**  
  $$H_{\text{out}} = H_{\text{in}} - F + 1 \quad (\text{for } S=1)$$
- **Same Padding ($H_{\text{out}} = H_{\text{in}}$ for $S=1$, odd $F$):**  
  $$P = \frac{F - 1}{2}$$

---

## 3. Parameter Count & FLOP Arithmetic

| Layer Type | Weight Parameters | Bias Parameters | Total Parameters | FLOPs per Sample |
|:---|:---|:---|:---|:---|
| **Conv2D** | $K \cdot D_{\text{in}} \cdot F^2$ | $K$ | $K \cdot (D_{\text{in}} \cdot F^2 + 1)$ | $2 \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot K \cdot D_{\text{in}} \cdot F^2$ |
| **Linear (Dense)** | $N_{\text{in}} \cdot N_{\text{out}}$ | $N_{\text{out}}$ | $N_{\text{out}} \cdot (N_{\text{in}} + 1)$ | $2 \cdot N_{\text{in}} \cdot N_{\text{out}}$ |
| **Spatial Pooling** | $0$ | $0$ | $0$ | $H_{\text{out}} \cdot W_{\text{out}} \cdot D \cdot F_p^2$ |

---

## 4. Spatial Pooling Operations

For an activation tensor $\mathbf{Z} \in \mathbb{R}^{D \times H \times W}$, pooling window size $F_p$, and stride $S_p$:

### Max Pooling
$$P_c(i, j) = \max_{0 \le u < F_p, \; 0 \le v < F_p} Z_c(i \cdot S_p + u, \; j \cdot S_p + v)$$

### Average Pooling
$$P_c(i, j) = \frac{1}{F_p^2} \sum_{u=0}^{F_p-1} \sum_{v=0}^{F_p-1} Z_c(i \cdot S_p + u, \; j \cdot S_p + v)$$

*Note:* $D_{\text{out}} \equiv D_{\text{in}}$ (channel depth is strictly preserved).

---

## 5. Effective Receptive Field (ERF) Growth

For a sequence of $L$ convolutional layers with kernel sizes $F_l$ and strides $S_l$:

$$RF_L = RF_{L-1} + (F_L - 1) \cdot \prod_{i=1}^{L-1} S_i$$

For unit strides ($S_l = 1$ for all $l$):
$$RF_L = 1 + \sum_{l=1}^L (F_l - 1)$$

*Key Implication (VGG Insight):* Stacking two $3 \times 3$ filters ($S=1$) yields:
$$RF_2 = 1 + (3 - 1) + (3 - 1) = 5$$
which matches a single $5 \times 5$ filter, but uses $2 \times 3^2 = 18$ weights instead of $5^2 = 25$ weights (a 28% parameter reduction).

---

## 6. LeNet-5 Layer-by-Layer Progression Table

| Layer | Type | Hyperparameters | Input Shape | Output Shape | Parameters |
|:---|:---|:---|:---|:---|:---|
| **Input** | Grayscale Image | - | - | $(1, 28, 28)$ | 0 |
| **Conv1** | `nn.Conv2d` | $K=6, F=5, P=2, S=1$ | $(1, 28, 28)$ | $(6, 28, 28)$ | $6 \times (1 \times 25 + 1) = 156$ |
| **Act1** | `nn.ReLU` / Sigmoid | - | $(6, 28, 28)$ | $(6, 28, 28)$ | 0 |
| **Pool1** | `nn.AvgPool2d` | $F_p=2, S_p=2$ | $(6, 28, 28)$ | $(6, 14, 14)$ | 0 |
| **Conv2** | `nn.Conv2d` | $K=16, F=5, P=0, S=1$ | $(6, 14, 14)$ | $(16, 10, 10)$ | $16 \times (6 \times 25 + 1) = 2,416$ |
| **Act2** | `nn.ReLU` / Sigmoid | - | $(16, 10, 10)$ | $(16, 10, 10)$ | 0 |
| **Pool2** | `nn.AvgPool2d` | $F_p=2, S_p=2$ | $(16, 10, 10)$ | $(16, 5, 5)$ | 0 |
| **Flatten**| `nn.Flatten` | $16 \times 5 \times 5$ | $(16, 5, 5)$ | $(400,)$ | 0 |
| **FC1** | `nn.Linear` | $N_{\text{in}}=400, N_{\text{out}}=120$ | $(400,)$ | $(120,)$ | $400 \times 120 + 120 = 48,120$ |
| **Act3** | `nn.ReLU` / Sigmoid | - | $(120,)$ | $(120,)$ | 0 |
| **FC2** | `nn.Linear` | $N_{\text{in}}=120, N_{\text{out}}=84$ | $(120,)$ | $(84,)$ | $120 \times 84 + 84 = 10,164$ |
| **Act4** | `nn.ReLU` / Sigmoid | - | $(84,)$ | $(84,)$ | 0 |
| **FC3** | `nn.Linear` | $N_{\text{in}}=84, N_{\text{out}}=10$ | $(84,)$ | $(10,)$ | $84 \times 10 + 10 = 850$ |
| **Total** | - | - | - | - | **61,706** |

---

## 7. Contrastive Decision Guide

| Design Question | Option A | Option B | Mathematical Criterion for Choice |
|:---|:---|:---|:---|
| **Filter Stacking** | Single $5 \times 5$ Conv | Two stacked $3 \times 3$ Convs | Stacked $3 \times 3$ has same $RF=5$, but 28% fewer params and an extra non-linearity. |
| **Downsampling** | Strided Conv ($S=2$) | Max Pooling ($2 \times 2, S=2$) | Strided Conv learns parameterized downsampling; MaxPool preserves hard activations with 0 parameters. |
| **Activation** | `nn.Sigmoid()` | `nn.ReLU()` | Sigmoid derivative saturates ($\le 0.25$), causing vanishing gradients; ReLU derivative is strictly 1 for $z>0$. |
| **Classifier Head**| Multi-layer Dense MLP | Global Average Pooling (GAP) | Dense heads account for >90% of parameters; GAP compresses $(C \times H \times W) \to (C \times 1 \times 1)$ with 0 parameters. |

---

## 8. Guarantees & Invariants

1. **Exact Channel Invariance in Pooling:**
   $$\forall c \in \{1, \dots, D_{\text{in}}\}, \quad P_c = \text{Pool}(Z_c) \implies D_{\text{out}} = D_{\text{in}}$$
   Pooling operators never mix or alter cross-channel information.
2. **Translation Equivariance of Convolutions:**
   $$\text{Conv}(T_{(\Delta x, \Delta y)} \mathbf{X}) = T_{(\Delta x, \Delta y)} \text{Conv}(\mathbf{X})$$
   Shifting an input image produces an identically shifted feature representation.
3. **Receptive Field Monotonicity:**
   $$RF_{l+1} > RF_l \quad \forall F_{l+1} > 1$$
   The effective receptive field increases monotonically with every non-trivial convolutional and pooling layer.

---

## 9. Numerical Stability & Traps

1. **Vanishing Gradient Saturation:**
   Sigmoid activations satisfy $\max \sigma'(z) = 0.25$. Across $L$ layers, backpropagation gradient magnitude scales as $\mathcal{O}((0.25)^L)$. To prevent vanishing gradients, replace sigmoid activations with `nn.ReLU()` whose subgradient is strictly $1.0$ for $z > 0$.
2. **Division by Zero in Normalization:**
   When calculating batch or layer statistics, always include numerical stabilizer $\epsilon = 10^{-5}$:
   $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
3. **Integer Truncation & Odd Dimensions:**
   When applying stride $S=2$ to an odd dimension $H_{\text{in}}$, the spatial floor function $\lfloor (H_{\text{in}} - F + 2P)/S \rfloor + 1$ truncates fractional steps, effectively dropping the last row or column. Always check dimension divisibility or use symmetric padding.
4. **Gradient Clipping & Outliers:**
   In early training of deep networks with large initial learning rates, gradients may spike. Apply $\ell_2$-norm gradient clipping:
   $$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{M}{\|\mathbf{g}\|_2}\right)$$
   where $M \in [1.0, 5.0]$.
