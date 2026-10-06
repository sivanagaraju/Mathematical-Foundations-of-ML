# Mathematical Formulae & Quick Revision Reference — Lecture 46: Backpropagation in RNNs & Vanishing Gradients

---

## 1. Master Equations Index

### 1. Recurrent Cell State Transition
$$h_t = \sigma(z_t) = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
Where $h_t \in \mathbb{R}^m$, $x_t \in \mathbb{R}^d$, $W_{hh} \in \mathbb{R}^{m \times m}$, $W_{xh} \in \mathbb{R}^{m \times d}$, and $b_h \in \mathbb{R}^m$.

### 2. Dual-Stream Backpropagation Through Time (BPTT)
$$\frac{\partial \mathcal{L}}{\partial h_t} = \left(\frac{\partial \hat{y}_t}{\partial h_t}\right)^T \frac{\partial \mathcal{L}}{\partial \hat{y}_t} + \left(\frac{\partial h_{t+1}}{\partial h_t}\right)^T \frac{\partial \mathcal{L}}{\partial h_{t+1}}$$
Splits total error gradient into local vertical injection and recursive horizontal temporal backpropagation.

### 3. Local Transition Jacobian Matrix
$$J_{k+1, k} \equiv \frac{\partial h_{k+1}}{\partial h_k} = \operatorname{diag}(\sigma'(z_{k+1})) \cdot W_{hh} \in \mathbb{R}^{m \times m}$$
Where $\operatorname{diag}(\sigma'(z_{k+1})) \in \mathbb{R}^{m \times m}$ is the diagonal matrix of instantaneous activation slopes.

### 4. Cumulative Temporal Jacobian Product Chain
$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k} = \prod_{k=t}^{T-1} \left( \operatorname{diag}(\sigma'(z_{k+1})) \cdot W_{hh} \right)$$
Represents the multivariable chain rule derivative of terminal state $h_T$ with respect to state $h_t$.

### 5. Parameter Gradient Accumulation Under Temporal Weight Sharing
$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^T \delta_t h_{t-1}^T, \quad \text{where } \delta_t = \frac{\partial \mathcal{L}}{\partial z_t} = \operatorname{diag}(\sigma'(z_t)) \frac{\partial \mathcal{L}}{\partial h_t}$$
Sums parameter sensitivity across all $T$ unrolled computational steps.

### 6. Sub-Multiplicative Operator Norm Bound (Vanishing Gradient Theorem)
$$\left\| \frac{\partial h_T}{\partial h_t} \right\|_2 \le \prod_{k=t}^{T-1} \|\operatorname{diag}(\sigma'(z_{k+1}))\|_2 \cdot \|W_{hh}\|_2 \le (\gamma \lambda_{\max})^{T-t}$$
Where $\gamma = \sup_z |\sigma'(z)|$ ($\gamma = 1.0$ for $\tanh$, $\gamma = 0.25$ for sigmoid) and $\lambda_{\max} = \sigma_{\max}(W_{hh})$.

### 7. Additive Gradient Highway Transition
$$h_t = \alpha h_{t-1} + F(h_{t-1}, x_t) \implies \frac{\partial h_t}{\partial h_{t-1}} = \alpha I + \frac{\partial F}{\partial h_{t-1}}$$
Introduces an unattenuated identity component $\alpha I$ to bypass multiplicative Jacobian decay.

---

## 2. Tensor Dimensionality & Shapes Ledger

| Tensor / Variable | Symbol | PyTorch Shape (`batch_first=True`) | Standard Mathematical Shape | Operational Role |
|:------------------|:-------|:-----------------------------------|:----------------------------|:-----------------|
| Input Sequence | $X$ | `[B, T, d]` | $X \in \mathbb{R}^{T \times d}$ | Batch of $B$ sequential observations of length $T$ and feature dim $d$ |
| Single Time Input | $x_t$ | `[B, d]` | $x_t \in \mathbb{R}^d$ | External observation injected into the cell at step $t$ |
| Hidden State Sequence | $H$ | `[B, T, m]` | $H \in \mathbb{R}^{T \times m}$ | Complete trajectory of hidden states across all $T$ timesteps |
| Single Hidden State | $h_t$ | `[B, m]` | $h_t \in \mathbb{R}^m$ | Latent context memory vector at temporal index $t$ |
| Recurrent Weight Matrix | $W_{hh}$ | `[m, m]` | $W_{hh} \in \mathbb{R}^{m \times m}$ | Transition operator shared across all time steps |
| Input Projection Matrix | $W_{xh}$ | `[m, d]` | $W_{xh} \in \mathbb{R}^{m \times d}$ | Observation projection operator shared across all time steps |
| Readout Projection Matrix | $W_{hy}$ | `[K, m]` | $W_{hy} \in \mathbb{R}^{K \times m}$ | Output classification / regression head |
| Local Transition Jacobian | $J_{k+1, k}$ | `[B, m, m]` | $J \in \mathbb{R}^{m \times m}$ | Rate of change of $h_{k+1}$ with respect to $h_k$ |
| Cumulative Temporal Jacobian | $\frac{\partial h_T}{\partial h_t}$ | `[B, m, m]` | $\frac{\partial h_T}{\partial h_t} \in \mathbb{R}^{m \times m}$ | Long-range sensitivity matrix across temporal gap $T-t$ |
| Error Sensitivity Vector | $\delta_t$ | `[B, m]` | $\delta_t \in \mathbb{R}^m$ | Error backpropagated to pre-activation $z_t$ |
| Recurrent Parameter Gradient | $\frac{\partial \mathcal{L}}{\partial W_{hh}}$ | `[m, m]` | $\nabla_{W_{hh}} \mathcal{L} \in \mathbb{R}^{m \times m}$ | Cumulative parameter update aggregated across all $T$ steps |

---

## 3. Guarantees & Invariants

| Invariant / Guarantee | Mathematical Expression | Engineering Meaning |
|:----------------------|:------------------------|:--------------------|
| **Multi-Path Gradient Conservation** | $\mathrm{d}\mathcal{L} = \sum_{\text{paths } p} \prod_{e \in p} J_e$ | The multivariable chain rule guarantees that error signals are never lost on a DAG, only partitioned across branches. |
| **Spectral Norm Equivalence** | $\|W_{hh}\|_2 \equiv \sigma_{\max}(W_{hh}) \equiv \sqrt{\lambda_{\max}(W_{hh}^T W_{hh})}$ | The maximum Euclidean dilation factor of the recurrent matrix is strictly its largest singular value. |
| **Activation Derivative Bound ($\tanh$)** | $0 < \frac{\mathrm{d}}{\mathrm{d}z}\tanh(z) \le 1.0$ | Hyperbolic tangent derivatives can never exceed $1.0$, preventing activation-induced gradient amplification. |
| **Activation Derivative Bound (Sigmoid)** | $0 < \sigma'(z) \le 0.25$ | Sigmoid activations inherently shrink backward gradients by at least a factor of $4$ at every single layer. |
| **Identity Highway Conservation** | $\lim_{J_F \to \mathbf{0}} \frac{\partial h_t}{\partial h_{t-1}} = \alpha I$ | In additive architectures, fully saturated non-linearities leave an intact linear transmission channel of scale $\alpha$. |

---

## 4. Boundary Cases & Zero Limits

| Condition / Limit | Mathematical Behavior | Practical Significance |
|:------------------|:----------------------|:-----------------------|
| **Large Temporal Gap ($T - t \to \infty, \gamma \lambda_{\max} < 1$)** | $\lim_{T-t \to \infty} (\gamma \lambda_{\max})^{T-t} = 0$ | Total gradient annihilation; initial inputs exert zero mathematical influence on late predictions. |
| **Large Temporal Gap ($T - t \to \infty, \gamma \lambda_{\max} > 1$)** | $\lim_{T-t \to \infty} (\gamma \lambda_{\max})^{T-t} = \infty$ | Exponential gradient explosion; parameters overflow to `NaN` or diverge uncontrollably. |
| **Zero Pre-Activation ($z = \mathbf{0}$ for $\tanh$)** | $\sigma'(0) = 1 - \tanh^2(0) = 1.0$ | Maximal gradient transmission through activation; zero squashing penalty. |
| **Extreme Saturation ($|z| \to \infty$)** | $\sigma'(z) \to 0$ | Complete blockage of gradient flow regardless of how large $\|W_{hh}\|_2$ is. |
| **Orthogonal Recurrent Weights ($W_{hh}^T W_{hh} = I$)** | $\|W_{hh}\|_2 = 1.0$ | Conserves gradient vector length through the linear transform; decay is governed solely by activation derivatives. |
| **Additive Shortcut with $\alpha = 1.0, F \equiv \mathbf{0}$** | $\frac{\partial h_T}{\partial h_t} = I^{T-t} = I$ | Perfect, infinite-horizon gradient transmission with zero loss or amplification. |

---

## 5. Computational Complexity & Memory Footprint

| Phase / Operation | Time Complexity | Spatial / Memory Complexity | Load-Bearing Mechanism |
|:------------------|:----------------|:----------------------------|:-----------------------|
| **Forward Pass (Unrolled)** | $\mathcal{O}(T \cdot (m^2 + m d))$ | $\mathcal{O}(T \cdot m)$ | Sequential evaluation of $h_t$; all $T$ hidden state vectors must be cached in memory for backward BPTT. |
| **Backward Pass (Full BPTT)** | $\mathcal{O}(T \cdot (m^2 + m d))$ | $\mathcal{O}(T \cdot m)$ | Reverse-mode automatic differentiation; identical FLOP count to forward pass, but traverses backwards in time. |
| **Truncated BPTT ($k_1$ forward, $k_2$ backward)** | $\mathcal{O}(k_2 \cdot (m^2 + m d))$ | $\mathcal{O}(k_2 \cdot m)$ | Truncates backward unrolling to fixed horizon $k_2$, bounding memory and preventing early gradient tracking. |
| **Temporal Jacobian Product ($T$ steps)** | $\mathcal{O}(T \cdot m^3)$ if matrices multiplied explicitly | $\mathcal{O}(m^2)$ | Reverse-mode AD uses vector-Jacobian products $\mathcal{O}(m^2)$, never forming full $m \times m$ Jacobians explicitly. |

---

## 6. Numerical Stability & Sanity Checks

| Trap / Vulnerability | Failure Symptom | Mathematical Root Cause | Production Remediation Protocol |
|:---------------------|:----------------|:------------------------|:--------------------------------|
| **Exploding Gradients** | Loss becomes `NaN` or `Inf`; model weights blow up | $\lambda_{\max} > 1$ and non-saturating paths cause exponential growth $(\lambda_{\max})^{T-t}$ | Apply global gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. |
| **Vanishing Gradients** | Training loss plateaus; early sequence tokens have zero gradient | $\lambda_{\max} < 1$ or saturated $\tanh$ derivatives force $(\gamma \lambda_{\max})^{T-t} \to 0$ | Switch to gated architectures (LSTM/GRU) with additive cell state highways; initialize $W_{hh}$ orthogonally. |
| **Sigmoid in Recurrent Transitions** | Rapid vanishing within 3–5 timesteps | $\sigma'(z) \le 0.25$ forces at least a $4\times$ reduction per timestep ($0.25^5 \approx 0.00097$) | Never use sigmoid for recurrent hidden state transitions; use $\tanh$ or additive identity channels. |
| **OOM on Long Sequences** | CUDA Out-Of-Memory during backward pass | BPTT caches all intermediate hidden states $\{h_1, \dots, h_T\}$ in GPU VRAM | Implement Truncated BPTT (TBPTT) or gradient checkpointing across temporal segments. |

---

## 7. Contrastive Decision Table: Structural Architectural Trade-offs

| Architectural Dimension | Vanilla RNN (Multiplicative) | Residual RNN / Highway | Long Short-Term Memory (LSTM) | Transformer (Self-Attention) |
|:------------------------|:----------------------------|:-----------------------|:-----------------------------|:----------------------------|
| **Gradient Propagation Mechanism** | Multiplicative Jacobian chain $\prod J_k$ | Additive identity path $\alpha I + J_F$ | Additive Cell State $c_t$ with Forget Gate | Direct pairwise paths $O(1)$ path length |
| **Long-Term Memory Horizon** | Very short ($\le 10$ steps) | Moderate ($\le 50$ steps) | Long ($\le 1,000$ steps) | Extremely long ($\ge 100,000$ tokens) |
| **Vanishing Gradient Susceptibility** | Extreme (exponential decay) | Low (mitigated by identity) | Minimal (CEC channel $f_t \approx 1$) | Immune across depth (Residual skips + LN) |
| **Exploding Gradient Susceptibility** | High (mitigated by clipping) | Moderate | Moderate (cell state unbounded) | Low (scale factor $\frac{1}{\sqrt{d_k}}$) |
| **Memory Footprint per Step** | Low: $O(m)$ state | Low: $O(m)$ state | Moderate: $O(2m)$ ($h_t, c_t$) | High: $O(T^2)$ attention cache |
| **Sequential Processing Bottleneck** | Strictly sequential $O(T)$ | Strictly sequential $O(T)$ | Strictly sequential $O(T)$ | Fully parallelizable across $T$ |
