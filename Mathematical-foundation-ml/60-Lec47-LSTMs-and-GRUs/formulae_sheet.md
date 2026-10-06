# Formulae Sheet & Rapid Revision — Lecture 47: LSTMs and GRUs

This rapid revision ledger summarizes the core mathematical equations, tensor shapes, invariant guarantees, numerical stability protocols, and computational complexity metrics for gated recurrent architectures.

---

## 1. Master Equations Ledger

### Unified Gating Abstraction
$$
h_t = \alpha_t \odot h_{t-1} + \beta_t \odot \tilde{h}_t
$$
where:
$$
\tilde{h}_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

### Highway Network Formulation ($\beta_t = \mathbf{1}$)
$$
y = T(x) \odot H(x) + (\mathbf{1} - T(x)) \odot x
$$
where $T(x) = \sigma(W_T x + b_T)$ is the transform gate and $(\mathbf{1} - T(x))$ is the carry gate.

### Gated Recurrent Unit (GRU, $\beta_t = \mathbf{1} - \alpha_t$)
$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \quad &\text{(Update Gate)} \\
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \quad &\text{(Reset Gate)} \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \quad &\text{(Candidate Hidden State)} \\
h_t &= (\mathbf{1} - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad &\text{(Convex State Update)}
\end{aligned}
$$

### Long Short-Term Memory (LSTM, Independent Gates)
$$
\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad &\text{(Forget Gate / } \alpha_t\text{)} \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad &\text{(Input Gate / } \beta_t\text{)} \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \quad &\text{(Candidate Memory Vector)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad &\text{(Additive Cell State Highway)} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad &\text{(Output Readout Gate)} \\
h_t &= o_t \odot \tanh(c_t) \quad &\text{(Observable Output State)}
\end{aligned}
$$

### Additive Single-Step Jacobian
$$
\frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}(\alpha_t) + \mathcal{E}_t(W_{xh}, W_{hh})
$$
where:
$$
\mathcal{E}_t = \operatorname{diag}\left( \beta_t \odot (1 - \tilde{h}_t^2) \right) W_{hh} + \operatorname{diag}(h_{t-1}) \frac{\partial \alpha_t}{\partial h_{t-1}} + \operatorname{diag}(\tilde{h}_t) \frac{\partial \beta_t}{\partial h_{t-1}}
$$

### Multi-Step Cumulative Temporal Jacobian
$$
\frac{\partial h_T}{\partial h_t} = \prod_{k=t}^{T-1} \left( \operatorname{diag}(\alpha_{k+1}) + \mathcal{E}_{k+1} \right) = \operatorname{diag}\left( \bigodot_{k=t}^{T-1} \alpha_{k+1} \right) + \sum_{\text{cross terms}} \dots
$$

---

## 2. Input, Hidden, and Output Tensor Shapes Ledger

| Tensor Variable | Algebraic Description | Standard PyTorch Shape | Physical / Memory Role |
|:----------------|:----------------------|:----------------------|:-----------------------|
| $X$ | Full Sequence Observation Tensor | `[B, T, d]` | Batch of input sequences of length $T$ with feature dimension $d$ |
| $x_t$ | Observation Vector at Timestep $t$ | `[B, d]` | Local temporal slice ingested at step $t$ |
| $h_{t-1}, h_t$ | Latent Recurrent Hidden State | `[B, m]` | Short-term memory vector / recurrent activation |
| $c_{t-1}, c_t$ | Dedicated Additive Cell State (LSTM) | `[B, m]` | Long-term memory carousel carrying unattenuated gradients |
| $W_{ih}$ (or $W_f, W_i, W_c, W_o$) | Fused Input Projection Weights | `[4*m, d]` | Projecting inputs to four parallel gate pre-activations |
| $W_{hh}$ (or $U_f, U_i, U_c, U_o$) | Fused Recurrent Transition Weights | `[4*m, m]` | Projecting previous state to four parallel gate pre-activations |
| $b_{ih} + b_{hh}$ | Fused Gate Biases | `[4*m]` | Affine bias vectors for gates $[i, f, g, o]$ |
| $\operatorname{diag}(\alpha_t)$ | Gating Transmission Operator | `[m, m]` | Diagonal operator preserving unattenuated gradient |

---

## 3. Guarantees & Theoretical Invariants

1. **Unattenuated Identity Transmission:** When forget gates satisfy $(\alpha_k)_i \approx 1.0$, the gradient transmission factor along coordinate $i$ satisfies $\prod_{k=t}^{T-1} (\alpha_k)_i \approx 1.0$, guaranteeing non-vanishing gradient flow over hundreds of steps.
2. **Convex State Bounding (GRU):** Since $z_t \in [0, 1]^m$ and $\tilde{h}_t \in [-1, 1]^m$, if $\|h_{t-1}\|_{\infty} \le 1$, then $\|h_t\|_{\infty} \le (1 - z_t) \cdot 1 + z_t \cdot 1 = 1$. GRU hidden activations remain bounded in $[-1, 1]$ for all $t$.
3. **Decoupled Readout:** In LSTMs, the cell state $c_t$ can grow linearly without bound to accumulate counts, while output state $h_t = o_t \odot \tanh(c_t)$ remains strictly squashed in $(-1, 1)$.

---

## 4. Boundary Cases & Zero Limits

- **Complete Forgetting ($f_t \to \mathbf{0}$ / $z_t \to \mathbf{1}$):** Memory is cleared instantly: $c_t = i_t \odot \tilde{c}_t$ (LSTM) or $h_t = \tilde{h}_t$ (GRU). Backpropagated gradients cannot pass prior to step $t$.
- **Complete Preservation ($f_t \to \mathbf{1}, i_t \to \mathbf{0}$ / $z_t \to \mathbf{0}$):** The network acts as a pure identity buffer: $c_t = c_{t-1}$ and $\frac{\partial c_t}{\partial c_{t-1}} = \mathbf{I}_m$. Gradients flow across infinite temporal horizons with zero attenuation.
- **Vanilla Degeneracy ($f_t = \mathbf{0}, i_t = \mathbf{1}, o_t = \mathbf{1}$):** Collapses an LSTM back into a vanilla RNN, re-introducing the exponential vanishing gradient pathology.

---

## 5. Computational Complexity & Memory Footprint

| Architecture | Parameters per Layer | FLOPs per Timestep (Batch=1) | Forward Memory Footprint | Effective Gradient Horizon |
|:-------------|:---------------------|:-----------------------------|:-------------------------|:---------------------------|
| **Vanilla RNN** | $m^2 + m d + m$ | $2 m^2 + 2 m d$ | $O(T \cdot m)$ | $T \approx 5 - 10$ steps |
| **GRU** | $3(m^2 + m d + m)$ | $6 m^2 + 6 m d$ | $O(T \cdot m)$ | $T \approx 100 - 300$ steps |
| **LSTM** | $4(m^2 + m d + m)$ | $8 m^2 + 8 m d$ | $O(2 T \cdot m)$ ($h_t$ and $c_t$) | $T \approx 500 - 1000$ steps |
| **ResNet-50** | Spatial (no time) | Fixed per image | $O(L \cdot d_{\text{feat}})$ | $L = 100+$ layers |
| **Transformer** | $4 d_{\text{model}}^2$ (attn) | $O(T^2 d + T d^2)$ | $O(T^2 + T d)$ | $T = 10,000+$ tokens |

---

## 6. Numerical Stability Protocols & Common Traps

1. **The Forget Gate Bias Initialization Rule:** Always initialize the forget gate bias to a positive value ($b_f = +1.0$ or $+2.0$). Initializing $b_f = 0$ causes $\sigma(0) = 0.5$, which yields $0.5^{50} \approx 8.8 \times 10^{-16}$, causing artificial gradient vanishing during early epochs.
2. **Cell State Gradient Detachment:** Do not clamp $c_t$ using hard conditional branches in Python; use smooth gating or gradient clipping to preserve Autograd computational graph connectivity.
3. **Fused GEMM Dispatch:** In production, pack $[W_f, W_i, W_c, W_o]$ into a single matrix $W_{ih} \in \mathbb{R}^{4m \times d}$ to execute one large GEMM instead of four separate matrix multiplications per step.

---

## 7. Contrastive Architectural Decision Matrix

| Problem Characteristic | Recommended Architecture | Mathematical Justification |
|:-----------------------|:-------------------------|:---------------------------|
| Short sequences ($T < 15$), low compute budget | **Vanilla RNN / 1D-CNN** | Low parameter footprint; gradient vanishing does not occur over short horizons |
| Moderate sequence dependencies ($T \le 250$), latency sensitive | **Gated Recurrent Unit (GRU)** | 25% fewer parameters and GEMMs than LSTM; single convex update gate |
| Long-horizon credit assignment ($T > 500$), counting tasks | **Long Short-Term Memory (LSTM)** | Dedicated linear cell state $c_t$ decoupled from bounded readout $h_t$ |
| Deep spatial hierarchies without sequence time | **ResNet / DenseNet** | Identity skip connections across depth $y = x + F(x)$ |
| Massive parallel training, full cross-token attention | **Transformer** | $O(1)$ path length between any two sequence tokens via self-attention |
