# Rapid Revision Formulae Sheet & Dimensionality Ledger

> **Package:** `58-Lec45-Recurrent-Neural-Networks-RNNs`  
> **Course:** NPTEL / IISc — Mathematical Foundations of Machine Learning  
> **Instructor:** Prof. Prathosh A P (IISc Bengaluru)  
> **Skill Standard:** Canonical 7-Pillar Production Learning Suite (`/youtube-lecture-tutor`)

---

## 1. Master Equations Index

### 1. Recurrent Cell Pre-activation Linear Fusion
$$z_t = W_{hh} h_{t-1} + W_{xh} x_t + b_h$$
where $W_{hh} \in \mathbb{R}^{m \times m}$, $W_{xh} \in \mathbb{R}^{m \times d}$, $b_h \in \mathbb{R}^m$, with unified block formulation:
$$z_t = \begin{bmatrix} W_{hh} & W_{xh} \end{bmatrix} \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_h$$

### 2. Latent Hidden State Non-linear Transition
$$h_t = \sigma(z_t) = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
with boundary initialization at step $t=1$: $h_0 = \mathbf{0} \in \mathbb{R}^m$ (or learnable prior $h_0 \sim \mathcal{N}(0, \sigma^2 I)$).

### 3. Readout Projection & Output Decoding
$$\hat{y}_t = g(W_{hy} h_t + b_y)$$
- For continuous regression: $\hat{y}_t = W_{hy} h_t + b_y \in \mathbb{R}^{d_{\text{out}}}$
- For $K$-class categorical classification: $\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y) \in \Delta^{K-1}$

### 4. Many-to-One Sequence Classification Output
$$\hat{y} = \text{softmax}(W_{hy} h_T + b_y)$$
Loss is evaluated strictly at the terminal timestep $T$:
$$\mathcal{L}_{\text{seq}} = -\sum_{k=1}^K y_k \log \hat{y}_k$$

### 5. Unrolled Linear Recurrence Expansion
For a linear activation $\sigma(z) = z$, the state at step $t$ decomposes exactly into:
$$h_t = W_{hh}^t h_0 + \sum_{k=1}^t W_{hh}^{t-k} (W_{xh} x_k + b_h)$$
demonstrating that $h_t$ is a discrete convolution of input sequence $x$ with the matrix impulse response $W_{hh}^k W_{xh}$.

### 6. Causal Block-Toeplitz Global Transformation Matrix
$$\begin{bmatrix} h_1 \\ h_2 \\ \vdots \\ h_T \end{bmatrix} = \begin{bmatrix}
W_{xh} & \mathbf{0} & \dots & \mathbf{0} \\
W_{hh} W_{xh} & W_{xh} & \dots & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots \\
W_{hh}^{T-1} W_{xh} & W_{hh}^{T-2} W_{xh} & \dots & W_{xh}
\end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_T \end{bmatrix} + \tilde{b}$$

### 7. Auto-regressive Joint Sequence Factorization
$$P(y_1, y_2, \dots, y_T \mid X) = \prod_{t=1}^T P(y_t \mid y_1, \dots, y_{t-1}, X) = \prod_{t=1}^T P(y_t \mid s_t)$$
where decoder state $s_t = \sigma(W_{ss} s_{t-1} + W_{ys} y_{t-1} + b_s)$ acts as a sufficient statistic summarizing past generated tokens.

### 8. Hierarchical Deep Stacked Recurrence
For layer $l \in \{1, \dots, L\}$ and temporal step $t \in \{1, \dots, T\}$:
$$h_t^{[l]} = \sigma\left(W_{hh}^{[l]} h_{t-1}^{[l]} + W_{xh}^{[l]} h_t^{[l-1]} + b^{[l]}\right)$$
where base layer input $h_t^{[0]} \equiv x_t$.

---

## 2. Input/Output Dimensionality & Tensor Shapes Ledger

| Operation / Object | Mathematical Symbol | Tensor Shape (PyTorch Batch-First) | PyTorch Shape (Time-First) |
|:-------------------|:-------------------:|:----------------------------------:|:---------------------------|
| Input Sequence Tensor | $X$ | `[B, T, d]` | `[T, B, d]` |
| Hidden State (Single Layer) | $h_t$ | `[B, m]` | `[B, m]` |
| Hidden Trajectory (All Steps) | $H$ | `[B, T, m]` | `[T, B, m]` |
| Deep Hidden State (Stack) | $h_t^{[1:L]}$ | `[L, B, m]` | `[L, B, m]` |
| Hidden-to-Hidden Weights | $W_{hh}$ | `[m, m]` | `[m, m]` |
| Input-to-Hidden Weights | $W_{xh}$ | `[m, d]` | `[m, d]` |
| Hidden-to-Output Weights | $W_{hy}$ | `[d_out, m]` | `[d_out, m]` |
| Hidden State Bias | $b_h$ | `[m]` | `[m]` |
| Output Projection Bias | $b_y$ | `[d_out]` | `[d_out]` |
| Step Output Predictions | $\hat{y}_t$ | `[B, d_out]` | `[B, d_out]` |
| Full Sequence Predictions | $\hat{Y}$ | `[B, T, d_out]` | `[T, B, d_out]` |
| Terminal Classification Logits | $\hat{y}_{\text{term}}$ | `[B, K]` | `[B, K]` |

## 3. Guarantees & Invariants

| Invariant / Guarantee | Mathematical Expression | Engineering Meaning |
|:----------------------|:------------------------|:--------------------|
| **Temporal Translation Equivariance** | $\mathcal{F}(S_\tau(X)) = S_\tau(\mathcal{F}(X))$ | Shifting input sequence by $\tau$ timesteps shifts output activations identically without altering dynamics. |
| **Causal Information Flow** | $\mathcal{M}_{i, j} = \mathbf{0} \quad \forall i < j$ | Future inputs cannot influence past or present hidden states; the unrolled matrix is strictly lower-triangular. |
| **Parameter Invariance w.r.t Sequence Length** | $|\theta| = m^2 + m d + d_{\text{out}} m + m + d_{\text{out}}$ | Parameter count remains strictly $O(1)$ regardless of whether $T = 5$ or $T = 50,000$. |
| **Exact Factorization Guarantee** | $P(y_1, \dots, y_T \mid X) = \prod_{t=1}^T P(y_t \mid y_{<t}, X)$ | The probabilistic chain rule guarantees lossless decomposition of the joint sequence likelihood. |

---

## 4. Boundary Cases & Zero Limits

| Condition / Limit | Mathematical Behavior | Practical Significance |
|:------------------|:----------------------|:-----------------------|
| **Initial Step ($t=1$)** | $z_1 = W_{hh} h_0 + W_{xh} x_1 + b_h$; with $h_0 = \mathbf{0}$, $z_1 = W_{xh} x_1 + b_h$. | Initial state acts like a standard feedforward layer receiving first token. |
| **Zero Input Stream ($x_t = \mathbf{0}$)** | $h_t = \tanh(W_{hh} h_{t-1} + b_h)$. | Model evolves as an autonomous non-linear dynamical system driven purely by internal memory. |
| **Contractive State ($\rho(W_{hh}) < 1$)** | $\| W_{hh}^k \| \to 0$ exponentially as $k \to \infty$. | Network forgets distant past context; early inputs decay to zero influence. |
| **Explosive State ($\rho(W_{hh}) > 1$)** | $\| W_{hh}^k \| \to \infty$ exponentially as $k \to \infty$. | Memory states saturate activation bounds, causing severe gradient explosion during BPTT. |
| **Identity Memory ($W_{hh} = I, \sigma=\text{id}$)** | $h_t = h_0 + \sum_{k=1}^t (W_{xh} x_k + b_h)$. | Acts as a perfect cumulative integrator across time without memory decay. |
| **Zero Bias ($b_h = \mathbf{0}, b_y = \mathbf{0}$)** | $h_t$ is strictly zero whenever $h_{t-1}=\mathbf{0}$ and $x_t=\mathbf{0}$. | Preserves null-vector fixed point at origin. |

---

## 4. Computational Complexity & Memory Footprint

| Metric / Resource | Formula | Notes |
|:------------------|:--------|:------|
| **FLOPs per Timestep (Forward)** | $2 m^2 + 2 m d + 2 m d_{\text{out}}$ | Matrix-vector GEMM dominates computation at each step. |
| **Total Forward FLOPs (Sequence $T$)** | $O(T \cdot (m^2 + m d + m d_{\text{out}}))$ | Scales strictly linearly with sequence length $T$. |
| **Total Learnable Parameters ($|\Theta|$)** | $m^2 + m d + m + d_{\text{out}} m + d_{\text{out}}$ | **Strictly $O(1)$ with respect to sequence length $T$** (Parameter sharing). |
| **Memory per Step (Inference)** | $O(m)$ bytes | Only current hidden state $h_t$ must be retained; past inputs discarded. |
| **Memory for Training (BPTT)** | $O(T \cdot m)$ bytes per sample | All intermediate hidden states $\{h_1, \dots, h_T\}$ must be cached for backpropagation. |

---

## 5. Numerical Stability & Sanity Checks

1. **Pre-activation Dynamic Range:**
   Monitor $\max |z_t|$. If $|z_t| > 4.0$, $\tanh(z_t)$ enters extreme saturation:
   $$\tanh'(4.0) = 1 - \tanh^2(4.0) \approx 1 - (0.9993)^2 \approx 0.0013$$
   Error signals passing through saturated units are diminished by over $99.8\%$.

2. **Weight Matrix Initialization Rule:**
   To maintain unit variance of hidden activations under random inputs, initialize $W_{xh}$ using Xavier/Glorot normal:
   $$W_{xh} \sim \mathcal{N}\left(0, \frac{2}{d + m}\right)$$
   and initialize recurrent matrix $W_{hh}$ as an orthogonal matrix:
   $$W_{hh}^T W_{hh} = I \implies |\lambda_i(W_{hh})| = 1$$
   preventing immediate exponential decay or explosion at step $t=1$.

3. **Loop vs Block-Toeplitz Parity Assertion:**
   For linear recurrence verification, check:
   $$\| H_{\text{loop}} - \mathcal{M}_{\text{Toeplitz}} X \|_\infty < 10^{-6}$$

---

## 6. Contrastive Decision Table: Structural Architectural Trade-offs

| Architectural Model | Inductive Prior | Sequence Length Generalization | Parameter Scaling w.r.t. $T$ | Computational Parallelism | Long-Range Dependency Retention |
|:--------------------|:----------------|:-------------------------------|:-----------------------------|:--------------------------|:--------------------------------|
| **Dense MLP** | None (Fully connected) | **Fails** (Fixed input dimension $d \cdot T$) | $O(T)$ parameter explosion | Fully Parallel | Poor (Permutation insensitive) |
| **1D Temporal CNN** | Local receptive field + weight sharing | Partial (Finite receptive field $k \cdot L$) | $O(1)$ w.r.t. $T$ | Fully Parallel across $T$ | Limited to effective receptive field depth |
| **Vanilla RNN** | Recurrent state memory + temporal weight tying | **Full** (Arbitrary $T$ processed sequentially) | $O(1)$ w.r.t. $T$ | **Strictly Sequential** ($O(T)$ latency) | Poor (Vanishing gradients beyond 10-20 steps) |
| **LSTM / GRU** | Additive gradient highways + gating units | **Full** (Arbitrary $T$ processed sequentially) | $O(1)$ w.r.t. $T$ | **Strictly Sequential** ($O(T)$ latency) | High (Gating preserves memory up to ~100 steps) |
| **Transformer** | Content-based pairwise self-attention | Position encoding dependent | $O(1)$ parameters, $O(T^2)$ compute/memory | Fully Parallel across $T$ | **Global** ($O(1)$ path length between any two tokens) |
