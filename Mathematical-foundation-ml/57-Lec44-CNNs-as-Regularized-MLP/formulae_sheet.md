# Rapid Revision Formulae Sheet & Dimensionality Ledger

> **Package:** `57-Lec44-CNNs-as-Regularized-MLP`  
> **Course:** NPTEL / IISc — Mathematical Foundations of Machine Learning  
> **Instructor:** Prof. Prathosh A P (IISc Bengaluru)  
> **Skill Standard:** Canonical 7-Pillar Production Learning Suite (`/youtube-lecture-tutor`)

---

## 1. Master Equations Index

### 1. Multi-Channel 2D Cross-Correlation with Channel Marginalization
$$\mathcal{Z}[i, j, m] = \sum_{c=1}^{C_{\text{in}}} \sum_{u=0}^{k_h - 1} \sum_{v=0}^{k_w - 1} \mathcal{W}[m, u, v, c] \cdot \mathcal{X}[i \cdot s + u, \; j \cdot s + v, \; c] + b_m$$
where:
$$H_{\text{out}} = \left\lfloor \frac{H - k_h + 2p}{s} \right\rfloor + 1, \quad W_{\text{out}} = \left\lfloor \frac{W - k_w + 2p}{s} \right\rfloor + 1$$

### 2. Output Spatial Dimension Floor Recurrence
$$P_{\text{out}} = \left\lfloor \frac{P - d(k - 1) - 1 + 2p}{s} \right\rfloor + 1$$
For standard non-dilated convolution ($d = 1$):
$$P_{\text{out}} = \left\lfloor \frac{P - k + 2p}{s} \right\rfloor + 1$$

### 3. Transposed Convolution Spatial Expansion Recurrence
$$P_{\text{out}} = (P_{\text{in}} - 1) \cdot s - 2p + d(k - 1) + 1$$
For standard non-dilated transposed convolution ($d = 1$):
$$P_{\text{out}} = (P_{\text{in}} - 1) \cdot s - 2p + k$$

### 4. Average Pooling Operator (Fixed Uniform Kernel)
$$\mathcal{P}_{\text{avg}}(X)[i, j, c] = \frac{1}{k_p^2} \sum_{u=0}^{k_p - 1} \sum_{v=0}^{k_p - 1} X[i \cdot s_p + u, \; j \cdot s_p + v, \; c]$$

### 5. Max Pooling Operator & Subgradient Argmax Routing
$$\mathcal{P}_{\text{max}}(X)[i, j, c] = \max_{u, v} X[i \cdot s_p + u, \; j \cdot s_p + v, \; c]$$
$$\frac{\partial \mathcal{L}}{\partial X[r, c', c]} = \sum_{i, j} \delta[i, j, c] \cdot \mathbb{I}\left[(r, c') = \arg\max_{(u, v)} X[i \cdot s_p + u, j \cdot s_p + v, c]\right]$$

### 6. Weight Bank Gradient Accumulation
$$\frac{\partial \mathcal{L}}{\partial \mathcal{W}[m, u, v, c]} = \sum_{b=1}^B \sum_{i=0}^{H_{\text{out}}-1} \sum_{j=0}^{W_{\text{out}}-1} \delta[b, m, i, j] \cdot \mathcal{X}_{\text{pad}}[b, c, i \cdot s + u, j \cdot s + v]$$

### 7. Residual Identity Highway Gradient
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) \right)$$

---

## 2. Input/Output Dimensionality & Tensor Shapes Ledger

| Operation / Object | Mathematical Symbol | Tensor Shape (Batch Mode) | PyTorch Tensor Dimension |
|:-------------------|:-------------------:|:-------------------------:|:-------------------------|
| Input Image / Volume | $\mathcal{X}$ | $[B, C_{\text{in}}, H, W]$ | `(B, C_in, H, W)` |
| Convolutional Filter Bank | $\mathcal{W}$ | $[C_{\text{out}}, C_{\text{in}}, k_h, k_w]$ | `(C_out, C_in, k_h, k_w)` |
| Layer Bias Vector | $b$ | $[C_{\text{out}}]$ | `(C_out,)` |
| Output Feature Tensor | $\mathcal{Z}$ | $[B, C_{\text{out}}, H_{\text{out}}, W_{\text{out}}]$ | `(B, C_out, H_out, W_out)` |
| Average Pooling Kernel | $K_{\text{avg}}$ | $[C, 1, k_p, k_p]$ | `(C, 1, k_p, k_p)` (constant $1/k_p^2$) |
| Transposed Conv Kernel | $\mathcal{W}_{\text{trans}}$ | $[C_{\text{in}}, C_{\text{out}}, k_h, k_w]$ | `(C_in, C_out, k_h, k_w)` |
| Flattened Representation | $v = \text{vec}(\mathcal{A})$ | $[B, d']$ | `(B, C_out * H_out * W_out)` |
| Dense Classification Head | $W_{\text{fc}}$ | $[K, d']$ | `(num_classes, in_features)` |
| Class Logits / Probabilities | $\hat{y}$ | $[B, K]$ | `(B, num_classes)` |

---

## 3. Boundary Cases & Zero Limits

1. **Kernel Size Equals Input Resolution ($k = P, s = 1, p = 0$):**
   $$P_{\text{out}} = \lfloor \frac{P - P}{1} \rfloor + 1 = 1$$
   The convolutional layer collapses to a Global Average Pooling or dense single-neuron receptive field across the entire canvas.
2. **Pointwise $1 \times 1$ Convolution ($k = 1, s = 1, p = 0$):**
   $$P_{\text{out}} = \lfloor \frac{P - 1}{1} \rfloor + 1 = P$$
   Zero spatial windowing. Performs pure channel-wise cross-projection: maps $[B, C_{\text{in}}, H, W] \to [B, C_{\text{out}}, H, W]$, computing an independent MLP across channels at each pixel coordinate.
3. **Stride Exceeds Kernel Width ($s > k$):**
   Sampling windows become non-overlapping and skip input pixels, creating blind spots (spatial information loss).
4. **Boundary Pad Limit:**
   Padding $p \ge k$ introduces redundant outer boundary zero rings that dilute feature activations and cause border artifact gradients.

---

## 4. Computational Complexity & Memory Footprint

| Metric | Dense MLP Baseline | 2D Multi-Channel CNN | 2D Transposed Conv (Decoder) |
|:-------|:-------------------|:---------------------|:-----------------------------|
| **Parameter Count** | $\mathcal{O}(M \cdot (H W C_{\text{in}}))$ | $\mathcal{O}(C_{\text{out}} \cdot C_{\text{in}} \cdot k_h k_w)$ | $\mathcal{O}(C_{\text{in}} \cdot C_{\text{out}} \cdot k_h k_w)$ |
| **FLOPs (Multiply-Adds)** | $2 \cdot B \cdot M \cdot (H W C_{\text{in}})$ | $2 \cdot B \cdot H_{\text{out}} W_{\text{out}} \cdot (C_{\text{out}} C_{\text{in}} k_h k_w)$ | $2 \cdot B \cdot H_{\text{in}} W_{\text{in}} \cdot (C_{\text{out}} C_{\text{in}} k_h k_w)$ |
| **Activation Memory** | $\mathcal{O}(B \cdot M)$ | $\mathcal{O}(B \cdot C_{\text{out}} \cdot H_{\text{out}} W_{\text{out}})$ | $\mathcal{O}(B \cdot C_{\text{out}} \cdot H_{\text{out}} W_{\text{out}})$ |
| **Weight Gradient Memory** | $\mathcal{O}(M \cdot H W C_{\text{in}})$ | $\mathcal{O}(C_{\text{out}} \cdot C_{\text{in}} \cdot k_h k_w)$ | $\mathcal{O}(C_{\text{in}} \cdot C_{\text{out}} \cdot k_h k_w)$ |

---

## 5. Numerical Stability & Sanity Checks

1. **Variance Scaling Initialization (He / Kaiming):**
   To prevent activation explosion or collapse across deep ReLU convolutions:
   $$\text{Var}(\mathcal{W}) = \frac{2}{\text{fan\_in}} = \frac{2}{C_{\text{in}} \cdot k_h \cdot k_w}$$
2. **Channel-Wise Symmetry Guard:**
   Always assert $\max_{i \neq j} \text{CosineSimilarity}(W^{(i)}, W^{(j)}) < 0.99$ after initialization to guarantee broken symmetry.
3. **Argmax Routing Gradient Conservation:**
   In max pooling, the sum of downstream gradients must equal the sum of upstream sensitivities:
   $$\sum_{b, c, u, v} \frac{\partial \mathcal{L}}{\partial X[b, c, u, v]} = \sum_{b, c, i, j} \delta[b, c, i, j]$$
4. **Finite Difference Gradient Check:**
   Verify analytical gradient against two-sided numerical difference:
   $$\frac{\partial \hat{R}}{\partial \mathcal{W}_{u, v}} \approx \frac{\hat{R}(\mathcal{W}_{u, v} + \epsilon) - \hat{R}(\mathcal{W}_{u, v} - \epsilon)}{2\epsilon}, \quad \text{with } \epsilon = 10^{-6}, \; \text{relative error} < 10^{-5}$$

---

## 6. Contrastive Decision Table: Structural Architectural Trade-offs

| Architectural Operator | Input / Output Mapping | Parameter Complexity | Inductive Bias / Invariance | Primary Failure Mode | Best Used When |
|:-----------------------|:-----------------------|:---------------------|:----------------------------|:---------------------|:---------------|
| **Standard 2D Conv (`Conv2d`)** | $[B, C_{\text{in}}, H, W] \to [B, C_{\text{out}}, H', W']$ | $\mathcal{O}(C_{\text{out}} C_{\text{in}} k^2)$ | Local translation equivariance | Receptive field blind spots if depth is too shallow | Feature extraction on spatial grid structures |
| **Average Pooling (`AvgPool2d`)** | $[B, C, H, W] \to [B, C, H/s, W/s]$ | $0$ (fixed parameter-free) | Smooth local low-pass invariance | Attenuates sharp high-frequency edges and contrast | Global spatial aggregation before final classification head |
| **Max Pooling (`MaxPool2d`)** | $[B, C, H, W] \to [B, C, H/s, W/s]$ | $0$ (fixed parameter-free) | Non-linear peak translation tolerance | Discards $75\%$ of spatial signal; subgradient sparsity | Downsampling intermediate feature maps in encoder |
| **Transposed Conv (`ConvTranspose2d`)** | $[B, C_{\text{in}}, H, W] \to [B, C_{\text{out}}, sH, sW]$ | $\mathcal{O}(C_{\text{in}} C_{\text{out}} k^2)$ | Learnable spatial expansion | Checkerboard artifact banding if stride does not divide kernel | Decoder upsampling in dense segmentation/synthesis |
| **Dense Linear Layer (`Linear`)** | $[B, D_{\text{in}}] \to [B, D_{\text{out}}]$ | $\mathcal{O}(D_{\text{in}} D_{\text{out}})$ | Universal unconstrained capacity | Severe sample complexity; destroys spatial topology | Global class decision boundary from compact latent vectors |

