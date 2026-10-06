# Formulae Sheet & Rapid Revision — Lecture 48: Attention Part 1

This rapid revision ledger summarizes the core mathematical equations, tensor shapes, invariant guarantees, numerical stability protocols, and computational complexity metrics for representation learning and attention projections.

---

## 1. Master Equations Ledger

### Sequence Matrix Representation
$$
X = \begin{bmatrix} x_1^T \\ x_2^T \\ \vdots \\ x_T^T \end{bmatrix} \in \mathbb{R}^{T \times D}
$$
where $T$ is the sequence length and $D$ is the embedding dimension.

### Linear Subspace Projections (Matrix Level)
$$
\begin{aligned}
Q &= X W^Q \in \mathbb{R}^{T \times D_Q} \quad &\text{(Queries: what tokens actively seek)} \\
K &= X W^K \in \mathbb{R}^{T \times D_K} \quad &\text{(Keys: what tokens offer as match)} \\
V &= X W^V \in \mathbb{R}^{T \times D_V} \quad &\text{(Values: actual communicable content)}
\end{aligned}
$$
where $W^Q \in \mathbb{R}^{D \times D_Q}$, $W^K \in \mathbb{R}^{D \times D_K}$, $W^V \in \mathbb{R}^{D \times D_V}$ are learnable parameters. In standard Transformers: $D_Q = D_K = D_V = D_k$.

### Token-Wise Subspace Projections (Row Level)
For token index $i \in \{1, \dots, T\}$:
$$
\begin{aligned}
q_i &= x_i^T W^Q \in \mathbb{R}^{1 \times D_k} \\
k_i &= x_i^T W^K \in \mathbb{R}^{1 \times D_k} \\
v_i &= x_i^T W^V \in \mathbb{R}^{1 \times D_k}
\end{aligned}
$$

### Recurrent Encoder Latent Bottleneck
$$
h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h), \quad t \in \{1, \dots, T\}
$$
Terminal bottleneck passed to decoder:
$$
c = H_T \in \mathbb{R}^m
$$

### Bahdanau Attention Dynamic Context Vector
$$
c_t = \sum_{i=1}^T \alpha_{t, i} h_i \in \mathbb{R}^m
$$
where alignment weights satisfy the simplex constraint:
$$
\alpha_{t, i} = \frac{\exp(e_{t, i})}{\sum_{j=1}^T \exp(e_{t, j})}, \quad \alpha_t \in \Delta^{T-1}
$$

### Direct Gradient Transmission Comparison
- **Recurrent Bottleneck Gradient:**
  $$
  \frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial H_T} \left( \prod_{k=1}^{T-1} \operatorname{diag}(\sigma'(z_{k+1})) W_{hh} \right) \quad \implies \quad \left\| \frac{\partial \mathcal{L}}{\partial h_1} \right\| \le \gamma^{T-1} \left\| \frac{\partial \mathcal{L}}{\partial H_T} \right\|
  $$
- **Attention Dynamic Combination Gradient:**
  $$
  \frac{\partial \mathcal{L}}{\partial h_i} = \frac{\partial \mathcal{L}}{\partial c_t} \frac{\partial c_t}{\partial h_i} = \alpha_{t, i} \frac{\partial \mathcal{L}}{\partial c_t} \quad \implies \quad O(1) \text{ path length for any token } i
  $$

---

## 2. Input/Output Tensor Dimensionality Table

| Tensor / Symbol | Semantic Meaning | Batch Tensor Shape | PyTorch Tensor Memory |
|:----------------|:-----------------|:-------------------|:----------------------|
| $X$ | Raw Input Token Sequence Matrix | $(B, T, D)$ | $B \times T \times D \times 4$ bytes |
| $W^Q$ | Query Projection Matrix | $(D, D_k)$ | $D \times D_k \times 4$ bytes |
| $W^K$ | Key Projection Matrix | $(D, D_k)$ | $D \times D_k \times 4$ bytes |
| $W^V$ | Value Projection Matrix | $(D, D_v)$ | $D \times D_v \times 4$ bytes |
| $Q$ | Projected Queries Matrix | $(B, T, D_k)$ | $B \times T \times D_k \times 4$ bytes |
| $K$ | Projected Keys Matrix | $(B, T, D_k)$ | $B \times T \times D_k \times 4$ bytes |
| $V$ | Projected Values Matrix | $(B, T, D_v)$ | $B \times T \times D_v \times 4$ bytes |
| $H_T$ | Recurrent Terminal Bottleneck | $(B, m)$ | $B \times m \times 4$ bytes |
| $\alpha_t$ | Attention Softmax Weights | $(B, 1, T)$ | $B \times T \times 4$ bytes |
| $c_t$ | Dynamic Context Vector | $(B, 1, m)$ | $B \times m \times 4$ bytes |

---

## 3. Mathematical Guarantees & Invariants Table

| Invariant / Property | Mathematical Formulation | Operational Guarantee in Deep Systems | Failure Mode if Violated |
|:---------------------|:-------------------------|:--------------------------------------|:-------------------------|
| **Permutation Equivariance** | $(\Pi X) W = \Pi (X W)$ for permutation $\Pi$ | Projections are position-agnostic; token features do not depend on sequence index | Position information is lost without explicit positional encodings |
| **Probability Simplex** | $\sum_{i=1}^T \alpha_{t, i} = 1, \; \alpha_{t, i} \ge 0$ | Context vector $c_t$ is a convex combination strictly bounded within the convex hull of $\{h_i\}$ | Exploding activations if weights fail to sum to 1 or become negative |
| **Direct Path Length** | $\operatorname{dist}(x_i, c_t) = 1$ | Unattenuated gradient flow to early tokens regardless of sequence length $T$ | Exponential gradient vanishing if forced through recurrent recurrence |
| **Data Processing Bound** | $I(X_1; H_T) \le I(X_1; H_t)$ for $t < T$ | Information strictly degrades across sequential Markov transitions | Catastrophic forgetting in long-context recurrent encoders |

---

## 4. Contrastive "Why X, Not Y" Quick Decision Table

| Design Choice ($X$) | Alternative ($Y$) | Critical Failure Mode of Alternative ($Y$) | Primary Mathematical Reason to Choose $X$ |
|:--------------------|:------------------|:------------------------------------------|:------------------------------------------|
| **Learned Projections ($W$)** | Fixed Transforms (Fourier/RBF) | Cannot adapt basis axes to complex empirical distributions | Gradient descent optimizes basis for minimal empirical risk |
| **Attention Combination ($c_t$)** | Recurrent Bottleneck ($H_T$) | Severe information compression loss on sequences $T > 30$ | Direct $O(1)$ access to all intermediate token states |
| **Asymmetric Projections ($W^Q \neq W^K$)** | Symmetric Matching ($X X^T$) | Forces directed relations to be symmetric ($A \to B \equiv B \to A$) | Bilinear form $W^Q (W^K)^T$ models directed dependencies |
| **Dynamic Softmax ($\alpha_t$)** | Uniform Average ($\frac{1}{T} \sum h_i$) | Dilutes critical keywords with irrelevant punctuation and stopwords | Sparse, high-entropy focus on relevant semantic tokens |

---

## 5. Hardware Realities & Numerical Stability

### GEMM Parallelism vs Sequential Barrier
- **Recurrent Encoder Execution:** Requires $T$ sequential CUDA kernel launches. Each recurrent step $h_t$ must await the completion of step $h_{t-1}$, under-utilizing GPU streaming multiprocessors (SMs) and stalling memory pipelines.
- **Attention Projections ($Q, K, V$):** Executed as a single batched General Matrix Multiply (GEMM) of shape $(B \cdot T, D) \times (D, 3 D_k)$. Tensor Cores achieve $>80\%$ peak theoretical compute throughput.

### LogSumExp Numerical Stability Trick
When evaluating softmax over raw compatibility scores $e \in \mathbb{R}^T$:
$$
\operatorname{softmax}(e)_i = \frac{\exp(e_i - \max_j e_j)}{\sum_{k=1}^T \exp(e_k - \max_j e_j)}
$$
Subtracting the maximum scalar $\max_j e_j$ ensures all exponent inputs are $\le 0$, preventing float32 overflow ($e^{89} \to \infty$) while maintaining exact mathematical invariance.
