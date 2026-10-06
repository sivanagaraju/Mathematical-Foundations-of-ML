# Formulae Sheet: Lecture 50 (Multi-Head Attention and Transformer Architecture)

A rapid revision ledger containing core mathematical equations, tensor shape signatures, analytical invariants, and numerical stability guidelines for Multi-Head Attention, Normalization Mechanics, and the complete Transformer architecture.

---

## Equations Index

### 1. Multi-Head Attention (MHA)
$$
\operatorname{MHA}(Q, K, V) = \operatorname{Concat}(\operatorname{head}_1, \dots, \operatorname{head}_M) W^O
$$
$$
\operatorname{head}_j = \operatorname{Attention}(Q W_j^Q, K W_j^K, V W_j^V) = \operatorname{softmax}\left( \frac{(Q W_j^Q)(K W_j^K)^T}{\sqrt{D_k}} \right) (V W_j^V)
$$
- $W_j^Q \in \mathbb{R}^{D \times D_k}, W_j^K \in \mathbb{R}^{D \times D_k}, W_j^V \in \mathbb{R}^{D \times D_v}$
- $W^O \in \mathbb{R}^{(M \cdot D_v) \times D}$
- $D_k = D_v = D / M$

### 2. Layer Normalization
For a token vector $x \in \mathbb{R}^D$:
$$
\mu_L = \frac{1}{D} \sum_{d=1}^D x_d, \quad \sigma_L^2 = \frac{1}{D} \sum_{d=1}^D (x_d - \mu_L)^2
$$
$$
\operatorname{LN}(x) = \gamma \odot \left( \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} \right) + \beta
$$
where $\gamma, \beta \in \mathbb{R}^D$ are learnable affine parameters, and $\epsilon > 0$ is a small numerical stabilizer.

### 3. Batch Normalization (Sequence Tensor Formulation)
For tensor $X \in \mathbb{R}^{B \times T \times D}$, for feature channel $d \in \{1, \dots, D\}$:
$$
\mu_{B, d} = \frac{1}{B \cdot T} \sum_{b=1}^B \sum_{t=1}^T X_{b, t, d}, \quad \sigma_{B, d}^2 = \frac{1}{B \cdot T} \sum_{b=1}^B \sum_{t=1}^T (X_{b, t, d} - \mu_{B, d})^2
$$
$$
\operatorname{BN}(X)_{b, t, d} = \gamma_d \left( \frac{X_{b, t, d} - \mu_{B, d}}{\sqrt{\sigma_{B, d}^2 + \epsilon}} \right) + \beta_d
$$

### 4. Residual Highway Gradient Propagation
For layer formulation $y = x + F(x)$:
$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left( I + \frac{\partial F(x)}{\partial x} \right)
$$
Across $L$ stacked residual layers:
$$
x_L = x_l + \sum_{i=l}^{L-1} F(x_i) \implies \frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \sum_{i=l}^{L-1} \frac{\partial F(x_i)}{\partial x_l} \right)
$$

### 5. Position-wise Feedforward Network (FFN)
$$
\operatorname{FFN}(h) = W_2 \sigma(W_1 h + b_1) + b_2
$$
- $W_1 \in \mathbb{R}^{D \times 4D}, b_1 \in \mathbb{R}^{4D}$
- $W_2 \in \mathbb{R}^{4D \times D}, b_2 \in \mathbb{R}^D$
- $\sigma(x) = \max(0, x)$ (ReLU) or $x \Phi(x)$ (GELU)

### 6. Vision Transformer (ViT) Patch Tokenization
For 2D image $I \in \mathbb{R}^{H \times W \times C}$ with patch size $P \times P$:
$$
T = \frac{H \cdot W}{P^2}, \quad x_p = \operatorname{Flatten}(\operatorname{patch}_p) W_E \in \mathbb{R}^D
$$
where $W_E \in \mathbb{R}^{(P^2 C) \times D}$.

---

## Tensor Dimensionality & Shape Signatures

| Operation / Tensor | Symbol | Shape Signature ($B=2, T=512, D=768, M=12, D_k=64$) | Semantics |
|:-------------------|:-------|:---------------------------------------------------|:----------|
| **Input Sequence** | $X$ | `[B, T, D]` = `[2, 512, 768]` | Token sequence representations |
| **Fused QKV Projections** | $QKV$ | `[B, T, 3 * D]` = `[2, 512, 2304]` | Fused linear projection |
| **Reshaped Query Head** | $Q$ | `[B, M, T, D_k]` = `[2, 12, 512, 64]` | Transposed multi-head query |
| **Transposed Key Head** | $K^T$ | `[B, M, D_k, T]` = `[2, 12, 64, 512]` | Transposed key for batched GEMM |
| **Multi-Head Attention Map** | $A$ | `[B, M, T, T]` = `[2, 12, 512, 512]` | $M$ parallel probability simplexes |
| **Head Context Outputs** | $Z_{\text{heads}}$ | `[B, M, T, D_v]` = `[2, 12, 512, 64]` | Aggregated values per head |
| **Concatenated Heads** | $Z_{\text{concat}}$ | `[B, T, M * D_v]` = `[2, 512, 768]` | Packed multi-head output |
| **Output Projection** | $\operatorname{MHA}(X)$ | `[B, T, D]` = `[2, 512, 768]` | Multi-head fused representation |
| **MLP Intermediate** | $H_{\text{mid}}$ | `[B, T, 4D]` = `[2, 512, 3072]` | Non-linear expanded features |
| **Transformer Block Output** | $Y$ | `[B, T, D]` = `[2, 512, 768]` | Contextualized block output |

---

## Guarantees & Invariants

1. **Subspace FLOP Conservation Invariant:**  
   $$
   \operatorname{FLOPs}(M \text{ heads of size } D/M) \equiv \operatorname{FLOPs}(1 \text{ head of size } D) = 4 T^2 D
   $$
2. **Row Simplex Normalization Guarantee:**  
   $$
   \forall b \in \{1, \dots, B\}, \; m \in \{1, \dots, M\}, \; i \in \{1, \dots, T\}: \quad \sum_{j=1}^T A_{b, m, i, j} = 1.0, \quad A_{b, m, i, j} \ge 0
   $$
3. **LayerNorm Sample Independence Guarantee:**  
   $$
   \frac{\partial \operatorname{LN}(X)_{b_1, t, d}}{\partial X_{b_2, t', d'}} = 0, \quad \forall b_1 \neq b_2
   $$
4. **Residual Gradient Lower Bound:**  
   $$
   \frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F(x_l)}{\partial x_l} \implies \left\| \frac{\partial \mathcal{L}}{\partial x_l} \right\| \ge \left\| \frac{\partial \mathcal{L}}{\partial x_{l+1}} \right\| (1 - \|J_F\|)
   $$
5. **Universal Approximation Guarantee:**  
   Pairing linear multi-head routing with position-wise two-layer MLPs renders the Transformer dense in $C(\mathcal{K}, \mathbb{R}^D)$.

---

## Contrastive Decision Table

| Decision Axis | Selected Approach | Rejected Alternative | Mathematical Justification |
|:--------------|:------------------|:---------------------|:----------------------------|
| **Multi-Head Splitting** | Fused GEMM + View Reshape | $M$ separate `nn.Linear` layers | Fused GEMM launches 1 CUDA kernel instead of $3M$ separate memory-bound kernels. |
| **Sequence Normalization** | Layer Normalization (`dim=-1`) | Batch Normalization (`dim=0,1`) | BatchNorm couples batch samples and is contaminated by padding tokens in variable sequences. |
| **Normalization Placement** | Pre-LN ($X + F(\operatorname{LN}(X))$) | Post-LN ($\operatorname{LN}(X + F(X))$) | Pre-LN keeps the residual gradient highway completely unobstructed, eliminating the need for warm-up. |
| **Sub-Layer Coupling** | Additive Residual ($X + F(X)$) | Multiplicative Gating ($X \odot \sigma(G)$) | Multiplicative gating causes exponential decay across 100+ layers; addition provides exact identity flow. |
| **Cross-Head Fusion** | Linear Projection $W^O$ | Direct Head Summation / Slicing | Direct summation forces all heads into an unweighted sum; $W^O$ learns optimal linear coordinate blending. |

---

## Numerical Stability & Hardware Traps

1. **Epsilon Safeguard in Layer Normalization:**  
   When all features in a token vector are nearly constant, $\sigma_L^2 \to 0$. Division by pure zero produces `NaN`. Always set $\epsilon \ge 10^{-5}$:
   ```python
   z_hat = (x - mean) / torch.sqrt(var + 1e-5)
   ```
2. **Pre-LN vs Post-LN Gradient Stability:**  
   In original Post-LN ($\operatorname{LN}(x + F(x))$), gradients passing through LayerNorm are scaled by $1/\sigma_L$, which decays at early layers, necessitating fragile learning rate warmups. Modern models universally use **Pre-LN**:
   $$
   x_{l+1} = x_l + F(\operatorname{LayerNorm}(x_l))
   $$
   which guarantees an unattenuated identity gradient highway $x_L = x_0 + \sum F(\operatorname{LN}(x_i))$.
3. **Contiguous Memory Buffers Before Reshape:**  
   Calling `tensor.transpose(1, 2)` modifies stride metadata without rearranging physical memory. Calling `.view()` directly on this transposed tensor throws a runtime error. Always enforce `.contiguous()` before calling `.view()`:
   ```python
   Z_merged = Z_heads.transpose(1, 2).contiguous().view(B, T, D)
   ```
4. **Residual Stream Variance Growth:**  
   Because $x_{l+1} = x_l + F(x_l)$, if $F(x_l)$ has variance $1.0$, the variance of $x_L$ grows linearly with depth: $\operatorname{Var}(x_L) \approx L \cdot \operatorname{Var}(x_0)$. Pre-LN prevents activation explosion by normalizing inputs to each sub-layer, ensuring stable variance regardless of depth $L$.
