# Formulae Sheet: Lecture 51 (Positional Embeddings)

A rapid revision ledger containing core mathematical equations, tensor shape signatures, analytical invariants, contrastive decision tables, and numerical stability guidelines for Positional Embeddings and Cross-Attention.

---

## 1. Equations Index

### Permutation Equivariance Identity
For input sequence $X \in \mathbb{R}^{T \times D}$ and permutation matrix $\Pi \in \mathbb{R}^{T \times T}$ ($\Pi^T \Pi = I$):
$$\operatorname{Attention}(\Pi X) = \Pi \operatorname{Attention}(X)$$

### Vaswani Sinusoidal Positional Encodings
For sequence position $pos \in \{0, 1, \dots, T-1\}$ and channel index $i \in \{0, 1, \dots, D/2 - 1\}$:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/D}}\right) = \sin(\omega_i \cdot pos)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/D}}\right) = \cos(\omega_i \cdot pos)$$
$$\omega_i = 10000^{-2i/D} = \exp\left( -\frac{2i}{D} \ln 10000 \right)$$
$$\lambda_i = \frac{2\pi}{\omega_i} = 2\pi \cdot 10000^{2i/D}$$

### 2D Linear Rotation Matrix Representation
For any fixed integer displacement $k \in \mathbb{Z}$:
$$\begin{pmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{pmatrix} = M_i(k) \begin{pmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{pmatrix}$$
$$M_i(k) = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix} \in \mathrm{SO}(2)$$

### Positional Inner Product Invariant
$$\langle PE_{pos}, PE_{pos+k} \rangle = \sum_{i=0}^{D/2-1} \cos(\omega_i k)$$

### Additive Vector Superposition
$$\tilde{x}_t = x_t + PE_t, \quad \tilde{X} \in \mathbb{R}^{T \times D}$$
$$q_i^T k_j = \underbrace{x_i W^Q (W^K)^T x_j^T}_{\text{Content-Content}} + \underbrace{x_i W^Q (W^K)^T PE_j^T}_{\text{Content-Position}} + \underbrace{PE_i W^Q (W^K)^T x_j^T}_{\text{Position-Content}} + \underbrace{PE_i W^Q (W^K)^T PE_j^T}_{\text{Position-Position}}$$

### Rotary Position Embedding (RoPE)
$$R_{\Theta, m}^d = \operatorname{diag}\left( R_{\theta_1, m}, \dots, R_{\theta_{d/2}, m} \right)$$
$$R_{\theta_i, m} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}, \quad \theta_i = 10000^{-2(i-1)/d}$$
$$\langle R_{\Theta, m}^d q_m, R_{\Theta, n}^d k_n \rangle = q_m^T R_{\Theta, n-m}^d k_n = g(q_m, k_n, m - n)$$

### Cross-Attention Formulation
$$\operatorname{Cross-Attention}(X_{\text{dec}}, X_{\text{enc}}) = \operatorname{softmax}\left( \frac{(X_{\text{dec}} W^Q)(X_{\text{enc}} W^K)^T}{\sqrt{d_k}} \right) (X_{\text{enc}} W^V)$$

---

## 2. Tensor Shapes & Dimension Signatures

| Tensor / Operator | Shape Signature | Physical Description |
|:------------------|:----------------|:---------------------|
| $X$ | $[B, T, D]$ | Batch of token embeddings |
| $PE$ | $[T, D]$ | Sinusoidal coordinate matrix |
| $\tilde{X} = X + PE$ | $[B, T, D]$ | Position-augmented sequence tensor |
| $W^Q, W^K, W^V$ | $[D, D]$ | Linear projection parameter matrices |
| $Q_{\text{dec}}$ | $[B, T_{\text{tgt}}, D]$ | Decoder queries |
| $K_{\text{enc}}, V_{\text{enc}}$ | $[B, T_{\text{src}}, D]$ | Encoder keys and values |
| Cross-Attention Scores $S$ | $[B, \text{num\_heads}, T_{\text{tgt}}, T_{\text{src}}]$ | Pairwise alignment affinity logits |
| Cross-Attention Output | $[B, T_{\text{tgt}}, D]$ | Contextual target representations |

---

## 3. Analytical Invariants

- **Norm Boundedness:** $\|PE_{pos}\|_2 = \sqrt{\sum_{i=0}^{D/2-1} 1.0} = \sqrt{\frac{D}{2}}$ for all $pos \ge 0$.
- **Orthogonal Rotation:** $\det(M_i(k)) = 1.0$ and $M_i(k)^T M_i(k) = I_2$.
- **Wavelength Extremes:** $\lambda_{\min} = 2\pi \approx 6.28$ tokens; $\lambda_{\max} = 2\pi \cdot 10000 \approx 62,832$ tokens.
- **RoPE Inner Product Invariance:** $\langle R_{m+s} q, R_{n+s} k \rangle = \langle R_m q, R_n k \rangle$ for any arbitrary shift $s \in \mathbb{Z}$.

---

## 4. Contrastive Decision Table

| Design Choice | When to Choose | When to Avoid | Mathematical Rationale |
|:--------------|:---------------|:--------------|:-----------------------|
| **Sinusoidal Addition ($x + PE$)** | General NLP encoders, sequence translation with variable lengths. | Ultra-long contexts (>32k) requiring strict relative decay. | Zero trainable parameters, bounded norm $\sqrt{D/2}$, preserves shape $[B, T, D]$. |
| **Learned Lookup Table ($E_{\text{pos}}$)** | Fixed-length classification models (BERT, original GPT-2). | Long documents, streaming generation, unknown test lengths. | High representational flexibility on $T \le T_{\max}$; fails with index errors on $T > T_{\max}$. |
| **Rotary Embeddings (RoPE)** | Modern autoregressive foundation LLMs (LLaMA, Mistral, Gemma). | Architectures requiring static spatial embeddings prior to attention. | Exact relative distance inner products $\langle R_m q, R_n k \rangle = g(m-n)$; extensible via NTK-aware scaling. |
| **Concatenation ($[x \,;\, PE]$)** | Low-dimensional inputs where semantic and spatial features conflict. | High-dimensional Transformers ($D \ge 256$). | Doubles projection parameters from $3D^2$ to $6D^2$ and doubles attention FLOPs. |

---

## 5. Numerical Stability & Traps

1. **Log-Domain Frequency Computation:**
   Do not compute $10000^{2i/D}$ directly in standard floating-point arithmetic because large exponents cause floating-point inaccuracies. Instead, compute denominators in the log domain:
   $$\operatorname{div\_term} = \exp\left( \text{indices} \cdot \left( -\frac{\ln 10000.0}{D} \right) \right)$$
2. **Preventing Division by Zero in LayerNorm ($\epsilon$):**
   When standardizing position-augmented embeddings $\tilde{X} = X + PE$, ensure $\epsilon \ge 10^{-5}$ in $\sqrt{\sigma^2 + \epsilon}$ to prevent numerical overflow in precision-limited half-precision (`torch.float16`).
3. **Logit Clipping in Cross-Attention:**
   Cross-attention across long sequences ($T_{\text{src}} \ge 4096$) can produce extreme attention scores if $d_k$ is small. Always scale scores strictly by $1/\sqrt{d_k}$ and optionally clip extreme values to $[-50.0, 50.0]$ before softmax to prevent softmax gradient saturation.
4. **Persistent Buffer Registration:**
   Register the precomputed $PE$ tensor using `self.register_buffer("pe", pe, persistent=False)` in PyTorch so it is transferred automatically to the GPU with the model without being tracked as a trainable parameter.
