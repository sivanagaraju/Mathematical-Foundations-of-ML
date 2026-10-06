# Rapid Revision Formulae Sheet: ERM & Error Backpropagation

> **Package:** `55-Lec42-ERM-Neural-Networks-Backpropagation`  
> **Course:** NPTEL / IISc — Mathematical Foundations of Machine Learning  
> **Skill Standard:** Canonical 7-Pillar Production Learning Suite (`/youtube-lecture-tutor`)

---

## 1. Master Equations Index

### 1. Forward Pass Computations
- **Pre-activation (Scalar):**
  $$z_j^{[l]} = \sum_{k=1}^{N_{l-1}} w_{jk}^{[l]} a_k^{[l-1]} + b_j^{[l]}$$
- **Post-activation (Scalar):**
  $$a_j^{[l]} = \sigma\left(z_j^{[l]}\right)$$
- **Forward Pass (Batched Tensor):**
  $$Z^{[l]} = A^{[l-1]} \left(W^{[l]}\right)^T + \mathbf{1}_B \left(b^{[l]}\right)^T, \quad A^{[l]} = \sigma\left(Z^{[l]}\right)$$

### 2. Error Sensitivity Recurrences ($\delta$)
- **Sensitivity Definition:**
  $$\delta_j^{[l]} \equiv \frac{\partial \hat{R}}{\partial z_j^{[l]}}$$
- **Output Layer Base Case (Scalar):**
  $$\delta_j^{[L]} = \frac{\partial \ell(y, a^{[L]})}{\partial a_j^{[L]}} \sigma'\left(z_j^{[L]}\right)$$
- **Output Layer Base Case (Half-MSE):**
  $$\delta_j^{[L]} = \left(a_j^{[L]} - y_j\right) \sigma'\left(z_j^{[L]}\right)$$
- **Hidden Layer Recurrence (Scalar):**
  $$\delta_j^{[l]} = \left( \sum_{i=1}^{N_{l+1}} \delta_i^{[l+1]} w_{ij}^{[l+1]} \right) \sigma'\left(z_j^{[l]}\right)$$
- **Hidden Layer Recurrence (Batched Tensor):**
  $$\Delta^{[l]} = \left( \Delta^{[l+1]} W^{[l+1]} \right) \odot \sigma'\left(Z^{[l]}\right)$$

### 3. Parameter Gradients & Updates
- **Weight Gradient (Scalar):**
  $$\frac{\partial \hat{R}}{\partial w_{jk}^{[l]}} = \delta_j^{[l]} a_k^{[l-1]}$$
- **Bias Gradient (Scalar):**
  $$\frac{\partial \hat{R}}{\partial b_j^{[l]}} = \delta_j^{[l]}$$
- **Weight Gradient (Batched Tensor):**
  $$\nabla_{W^{[l]}} \hat{R} = \frac{1}{B} \left(\Delta^{[l]}\right)^T A^{[l-1]} \in \mathbb{R}^{N_l \times N_{l-1}}$$
- **Bias Gradient (Batched Tensor):**
  $$\nabla_{b^{[l]}} \hat{R} = \frac{1}{B} \sum_{b=1}^B \Delta_{b, :}^{[l]} \in \mathbb{R}^{N_l}$$
- **Regularized Gradient Descent Update ($L_2$ Weight Decay):**
  $$W^{[l]} \leftarrow W^{[l]}(1 - \eta \lambda) - \eta \nabla_{W^{[l]}} \hat{R}, \quad b^{[l]} \leftarrow b^{[l]} - \eta \nabla_{b^{[l]}} \hat{R}$$

---

## 2. Tensor Dimensionality & Shape Ledger

| Variable / Tensor | Notation | Single-Sample Vector Shape | Mini-Batch Matrix Shape | Description |
|:------------------|:---------|:---------------------------|:------------------------|:------------|
| **Input Features** | $x, X$ | $[d]$ or $[d, 1]$ | $[B, d]$ | Raw input observations from dataset |
| **Layer Weights** | $W^{[l]}$ | $[N_l, N_{l-1}]$ | $[N_l, N_{l-1}]$ | Affine projection matrix for layer $l$ |
| **Layer Biases** | $b^{[l]}$ | $[N_l]$ or $[N_l, 1]$ | $[N_l]$ | Additive offset vector for layer $l$ |
| **Pre-activations** | $z^{[l]}, Z^{[l]}$ | $[N_l]$ or $[N_l, 1]$ | $[B, N_l]$ | Inner products before activation |
| **Post-activations** | $a^{[l]}, A^{[l]}$ | $[N_l]$ or $[N_l, 1]$ | $[B, N_l]$ | Activated hidden representation |
| **Error Sensitivity** | $\delta^{[l]}, \Delta^{[l]}$ | $[N_l]$ or $[N_l, 1]$ | $[B, N_l]$ | Partial derivative w.r.t pre-activation |
| **Weight Gradient** | $\nabla_{W^{[l]}} \hat{R}$ | $[N_l, N_{l-1}]$ | $[N_l, N_{l-1}]$ | Outer product $\delta^{[l]} (a^{[l-1]})^T$ |
| **Bias Gradient** | $\nabla_{b^{[l]}} \hat{R}$ | $[N_l]$ or $[N_l, 1]$ | $[N_l]$ | Summed error sensitivity across batch |

*Invariant Check:* The shape of parameter gradient $\nabla_\theta \hat{R}$ strictly equals the shape of parameter $\theta$ across all layers.

---

## 3. Mathematical Guarantees & Complexity Laws

1. **Computational Linearity Law:** Reverse-mode error backpropagation evaluates the gradient with respect to all $P$ parameters in $\mathcal{O}(P)$ operations.
2. **Compute-to-Inference Ratio:** For an $L$-layer network, forward inference requires $\sim 2P$ FLOPs per sample; backpropagation requires $\sim 4P$ FLOPs; total training step requires $\sim 6P$ FLOPs ($\approx 3\times$ inference compute).
3. **Memory Storage Invariant:** Computing backward error signals requires retaining all intermediate pre-activations $z^{[l]}$ and post-activations $a^{[l]}$ in memory throughout forward execution, creating an $\mathcal{O}(B \cdot \sum N_l)$ VRAM footprint.
4. **Bayesian Equivalence of Weight Decay:** Minimizing $\hat{R}(\theta) + \frac{\lambda}{2}\|W\|_F^2$ is mathematically identical to Maximum A Posteriori (MAP) estimation under an isotropic Gaussian prior $\mathcal{N}(0, \sigma_0^2 I)$ where $\lambda = \frac{1}{n \sigma_0^2}$.

---

## 4. Contrastive Quick Decision Guide

| Problem / Situation | Do This (Best Practice) | Avoid This (Anti-Pattern) | Why |
|:-------------------|:------------------------|:--------------------------|:----|
| **Deep Network Activations** | ReLU, LeakyReLU, or GELU in hidden layers | Sigmoid or Tanh in deep hidden layers | Sigmoid derivative $\le 0.25$ causes exponential vanishing gradients in early layers. |
| **Weight Regularization** | Apply weight decay solely to weight matrices $W$ | Applying weight decay to bias vectors $b$ | Biases shift activation thresholds without increasing model complexity or exploding Lipschitz bounds. |
| **Batch Size Selection** | Power-of-2 batch sizes ($32, 64, 128$) | Arbitrary odd batch sizes ($37, 93$) | GPU tensor cores tile memory in 8/16/32-thread warps; non-aligned sizes suffer warp padding penalties. |
| **Gradient Management** | `optimizer.zero_grad()` before `.backward()` | Omitting zero_grad in PyTorch | Gradients accumulate across batches, effectively scaling $\eta$ and triggering explosive divergence. |

---

## 5. Numerical Stability & Hardware Traps

- **Saturated Sigmoid Vanishing:** Because $\max \sigma'(z) = 0.25$, across $L$ layers the gradient signal decays by at least $(0.25)^L$. At $L=10$, $(0.25)^{10} \approx 9.5 \times 10^{-7}$, effectively freezing early layers.
- **VRAM Activation Caching Explosion:** In ultra-deep or high-resolution models, activations consume $>80\%$ of GPU memory. Remedy: Activation checkpointing (recomputing forward activations during backward pass).
- **Floating-Point Underflow in Softmax:** Calculating $\frac{e^{z_k}}{\sum_j e^{z_j}}$ directly produces `NaN` when $z_k > 88$ in float32. Remedy: Use the Log-Sum-Exp trick: $z_{\text{shift}} = z - \max(z)$.
