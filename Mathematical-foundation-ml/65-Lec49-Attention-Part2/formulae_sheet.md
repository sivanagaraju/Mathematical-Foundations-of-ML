# Formulae Sheet: Lecture 49 (Attention Part 2)

A rapid revision ledger containing core mathematical equations, tensor shape signatures, analytical invariants, and numerical stability guidelines for scaled dot-product attention.

---

## Equations Index

### 1. Scaled Dot-Product Attention
$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^T}{\sqrt{D_k}} \right) V
$$
- $Q \in \mathbb{R}^{T \times D_k}$: Query matrix.
- $K \in \mathbb{R}^{T \times D_k}$: Key matrix.
- $V \in \mathbb{R}^{T \times D_v}$: Value matrix.
- $D_k$: Feature dimensionality of the key vectors.

### 2. Dot-Product Variance Normalization Law
For independent zero-mean unit-variance components $q_i, k_i \sim \mathcal{N}(0, 1)$:
$$
\operatorname{Var}(q^T k) = \sum_{i=1}^{D_k} \operatorname{Var}(q_i k_i) = D_k
$$
$$
\operatorname{Var}\left( \frac{q^T k}{\sqrt{D_k}} \right) = \left( \frac{1}{\sqrt{D_k}} \right)^2 \operatorname{Var}(q^T k) = \frac{1}{D_k} \cdot D_k = 1.0
$$

### 3. Row-wise Softmax Probability
$$
A_{i, j} = \frac{\exp\left( \frac{q_i^T k_j}{\sqrt{D_k}} \right)}{\sum_{m=1}^T \exp\left( \frac{q_i^T k_m}{\sqrt{D_k}} \right)}
$$

### 4. Softmax Jacobian Matrix
The derivative of attention weight $A_i$ with respect to logit score $S_j$:
$$
\frac{\partial A_i}{\partial S_j} = A_i (\delta_{i, j} - A_j) = 
\begin{cases}
A_i (1 - A_i), & \text{if } i = j \\
-A_i A_j, & \text{if } i \neq j
\end{cases}
$$

### 5. Position-wise Feedforward Network (FFN)
$$
\operatorname{FFN}(z) = W_2 \sigma(W_1 z + b_1) + b_2
$$
- $W_1 \in \mathbb{R}^{D \times 4D}$, $b_1 \in \mathbb{R}^{4D}$.
- $W_2 \in \mathbb{R}^{4D \times D}$, $b_2 \in \mathbb{R}^D$.
- $\sigma(x)$: Non-linear activation function ($\operatorname{ReLU}(x) = \max(0, x)$ or $\operatorname{GELU}(x) = x \Phi(x)$).

---

## Tensor Dimensionality & Shape Signatures

| Operation / Tensor | Notation | Concrete Shape ($B=2, T=512, D=768, D_k=64$) | Semantics |
|:-------------------|:---------|:---------------------------------------------|:----------|
| **Input Sequence** | $X$ | `[B, T, D]` = `[2, 512, 768]` | Raw contextual sequence embeddings |
| **Query Matrix** | $Q = X W^Q$ | `[B, T, D_k]` = `[2, 512, 64]` | Search query projections |
| **Key Matrix** | $K = X W^K$ | `[B, T, D_k]` = `[2, 512, 64]` | Addressable catalog index projections |
| **Value Matrix** | $V = X W^V$ | `[B, T, D_v]` = `[2, 512, 64]` | Semantic payload representations |
| **Transposed Key** | $K^T$ | `[B, D_k, T]` = `[2, 64, 512]` | Prepared for batched GEMM with $Q$ |
| **Pairwise Logits** | $S = Q K^T / \sqrt{D_k}$ | `[B, T, T]` = `[2, 512, 512]` | Scaled compatibility metric |
| **Attention Matrix** | $A = \operatorname{softmax}(S)$ | `[B, T, T]` = `[2, 512, 512]` | Row-stochastic probability simplex |
| **Context Output** | $Z = A V$ | `[B, T, D_v]` = `[2, 512, 64]` | Convex combination of value vectors |
| **FFN Expansion** | $H_1 = \sigma(Z W_1 + b_1)$ | `[B, T, 4D]` = `[2, 512, 3072]` | Non-linear intermediate projection |
| **FFN Output** | $H_2 = H_1 W_2 + b_2$ | `[B, T, D]` = `[2, 512, 768]` | Transformed token representation |

---

## Guarantees & Invariants

1. **Row Simplex Normalization Invariant:**  
   $$
   \forall i \in \{1, \dots, T\}, \quad A_{i, j} \ge 0 \quad \text{and} \quad \sum_{j=1}^T A_{i, j} = 1.0
   $$
2. **Convex Hull Containment Guarantee:**  
   $$
   z_i \in \operatorname{conv}(v_1, \dots, v_T) \implies \|z_i\|_2 \le \max_{1 \le j \le T} \|v_j\|_2
   $$
3. **Permutation Equivariance:**  
   For any permutation matrix $P \in \{0, 1\}^{T \times T}$ where $P^T P = I$:
   $$
   \operatorname{Attention}(P Q, P K, P V) = P \cdot \operatorname{Attention}(Q, K, V)
   $$
4. **Computational Complexity Bound:**  
   - FLOPs: $\mathcal{O}(T^2 \cdot D_k + T^2 \cdot D_v)$.
   - Memory: $\mathcal{O}(T^2)$ for materializing the full attention probability matrix.

---

## Contrastive Decision Table

| Scenario / Goal | Recommended Choice | Rejected Alternative | Mathematical Justification |
|:----------------|:-------------------|:---------------------|:----------------------------|
| **Logits Normalization** | Divide by $\sqrt{D_k}$ | Divide by $D_k$ | Dividing by $D_k$ reduces variance to $1/D_k \approx 0$, collapsing softmax into uniform noise. |
| **Attention Normalization** | Row-wise Softmax (`dim=-1`) | Independent Sigmoid | Sigmoid does not sum to 1.0, allowing activation explosion or total token signal extinction. |
| **Key Transposition** | `K.transpose(-1, -2)` | `K.transpose(0, 1)` | Swapping dimensions 0 and 1 corrupts the batch dimension, causing unintended outer broadcasts and OOM crashes. |
| **Inter-Token Mixing** | Scaled Dot-Product Attention | Stacking Dense MLPs across $T$ | Dense layers across $T$ require fixed sequence length $T$ and cannot handle variable sequence lengths. |
| **Universal Approximation** | Position-wise MLP with ReLU/GELU | Stacking pure linear Attention | Repeated linear attention collapses into a single bilinear form; MLPs inject essential non-linear coordinate folding. |

---

## Numerical Stability & Hardware Traps

1. **Softmax Overflow Protection (Log-Sum-Exp Trick):**  
   Naive exponentiation $\exp(S_{i, j})$ overflows in FP16 / BF16 if $S_{i, j} > 88.7$. Always subtract the row maximum prior to exponentiation:
   $$
   S'_{i, j} = S_{i, j} - \max_{1 \le m \le T} S_{i, m}, \quad A_{i, j} = \frac{\exp(S'_{i, j})}{\sum_{m=1}^T \exp(S'_{i, m})}
   $$
2. **Attention Masking Negative Infinity Trap:**  
   When masking future tokens or padding, do not use `float("-inf")` in FP16. In FP16, `-inf * 0` produces `NaN`. Instead, use $-10000.0$ or `torch.finfo(dtype).min` (e.g., $-65504.0$ for FP16):
   ```python
   mask_value = torch.finfo(scores.dtype).min
   scores = scores.masked_fill(mask == 0, mask_value)
   ```
3. **Softmax Gradient Clamping:**  
   If $D_k$ is unscaled, $\max(S_{i, :}) - \min(S_{i, :})$ exceeds 50, driving non-maximal probabilities below $10^{-20}$. During backpropagation, this sets $\frac{\partial A_i}{\partial S_j} \approx 0$, freezing network weights. Scaling by $1/\sqrt{D_k}$ maintains logit spreads within $[-4, +4]$, ensuring active gradients $\approx 0.15$.
4. **FlashAttention GPU Memory Optimization:**  
   Standard attention stores $A \in \mathbb{R}^{B \times T \times T}$ in high-bandwidth memory (HBM). For $T = 32768$, $T^2$ elements consume 2 GB per head. FlashAttention computes attention incrementally in SRAM using online softmax scaling, reducing memory complexity from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$.
