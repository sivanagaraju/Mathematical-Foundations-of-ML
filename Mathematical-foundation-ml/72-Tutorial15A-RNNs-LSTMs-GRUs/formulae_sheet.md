# Formulae Sheet & Rapid Revision — Tutorial 15 Part 1: RNNs, LSTMs and GRUs

This sheet consolidates all mathematical formulations, gating equations, tensor dimensions, and parameter counting derivations for **Tutorial 15 Part 1: Recurrent Neural Networks, LSTMs, and GRUs**.

---

## 1. Input & State Tensor Dimensions

| Tensor Variable | Mathematical Domain | PyTorch `batch_first=True` | Description |
|:----------------|:--------------------|:---------------------------|:------------|
| $\mathbf{X}$ | $\mathbb{R}^{B \times T \times D}$ | `[batch_size, seq_len, input_dim]` | Input sequence batch |
| $x_t$ | $\mathbb{R}^D$ | `[batch_size, input_dim]` | Token vector at time step $t$ |
| $h_t$ | $\mathbb{R}^H$ | `[batch_size, hidden_dim]` | Working hidden state memory at step $t$ |
| $C_t$ | $\mathbb{R}^H$ | `[batch_size, hidden_dim]` | LSTM cell state memory at step $t$ |
| $\text{out\_seq}$ | $\mathbb{R}^{B \times T \times H}$ | `[batch_size, seq_len, hidden_dim]` | Stack of all intermediate hidden states $[h_1, \dots, h_T]$ |
| $h_T$ | $\mathbb{R}^H$ | `[batch_size, hidden_dim]` | Terminal hidden state used for sequence classification |
| $\mathbf{z}_{\text{logits}}$ | $\mathbb{R}^C$ | `[batch_size, num_classes]` | Output logits from linear head $W_y h_T + b_y$ |

---

## 2. Recurrent Cell State Transition Formulations

### 2.1 Vanilla RNN Cell
$$\mathbf{a}_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$
$$h_t = \tanh(\mathbf{a}_t)$$

- $W_{xh} \in \mathbb{R}^{H \times D}$, $W_{hh} \in \mathbb{R}^{H \times H}$, $b_h \in \mathbb{R}^H$.
- Initial condition: $h_0 = \mathbf{0}$.

### 2.2 Gated Recurrent Unit (GRU)
$$r_t = \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r) \quad \text{[Reset Gate]}$$
$$z_t = \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z) \quad \text{[Update Gate]}$$
$$\tilde{h}_t = \tanh(W_{xh} x_t + W_{hh} (r_t \odot h_{t-1}) + b_h) \quad \text{[Candidate Hidden State]}$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{[Convex Combination State Update]}$$

### 2.3 Long Short-Term Memory (LSTM)
$$f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f) \quad \text{[Forget Gate]}$$
$$i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i) \quad \text{[Input Gate]}$$
$$\tilde{C}_t = \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c) \quad \text{[Candidate Cell State]}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{[Linear Additive Cell State Update]}$$
$$o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o) \quad \text{[Output Gate]}$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{[Emitted Hidden State]}$$

---

## 3. Backpropagation Through Time (BPTT) & Gradient Highways

### Vanilla RNN Gradient Decay (Chain Rule):
$$\frac{\partial \mathcal{L}_T}{\partial h_1} = \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{k=2}^T \operatorname{diag}\left(1 - \tanh^2(a_k)\right) W_{hh}^T$$
- Condition for vanishing gradients: $\|W_{hh}\|_2 < 1 \implies \lim_{T \to \infty} \frac{\partial \mathcal{L}_T}{\partial h_1} = 0$.

### LSTM Constant Error Carousel (CEC):
$$\frac{\partial C_T}{\partial C_t} = \prod_{k=t+1}^T \operatorname{diag}(f_k)$$
- If $f_k \approx 1$, then $\frac{\partial C_T}{\partial C_t} \approx \mathbf{I}$, preventing gradient vanishing over arbitrarily long sequences.

---

## 4. Parameter Counting Formulae

For input dimension $D$ and hidden dimension $H$:

| Model Cell | Single-Bias Formulation | PyTorch Dual-Bias (`bias_ih` + `bias_hh`) | Proportionality |
|:-----------|:------------------------|:------------------------------------------|:----------------|
| **Vanilla RNN** | $H(D + H) + H$ | $1 \times [H(D + H) + 2H]$ | $1.00\times$ |
| **GRU** | $3 \times [H(D + H) + H]$ | $3 \times [H(D + H) + 2H]$ | $3.00\times$ |
| **LSTM** | $4 \times [H(D + H) + H]$ | $4 \times [H(D + H) + 2H]$ | $4.00\times$ |

### Numerical Example ($D = 128, H = 256$, PyTorch Layout):
- Per gate: $256 \times (128 + 256) + 2 \times 256 = 256 \times 384 + 512 = 98,816$ parameters.
- **RNN:** $98,816$ parameters.
- **GRU:** $296,448$ parameters.
- **LSTM:** $395,264$ parameters (GRU saves $98,816$ parameters, exactly 25%).

---

## 5. Guarantees & Invariants

1. **State Boundedness Invariant:** In vanilla RNNs and GRUs, $h_t \in (-1, 1)^H$ strictly due to the terminal $\tanh$ squashing. In LSTMs, $C_t \in \mathbb{R}^H$ is unconstrained (can accumulate values $> 1$), while emitted state $h_t = o_t \odot \tanh(C_t) \in (-1, 1)^H$.
2. **Gating Interval Invariant:** For all gates $g \in \{f_t, i_t, o_t, r_t, z_t\}$, activations satisfy $g \in (0, 1)^H$ strictly due to the logistic sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$.
3. **Temporal Parameter Invariance:** For any sequence length $T \ge 1$, total model parameter count $\mathcal{O}(1)$ is strictly constant and independent of $T$.

---

## 6. Contrastive Decision Table

| Decision Factor | Vanilla RNN | Gated Recurrent Unit (GRU) | Long Short-Term Memory (LSTM) |
|:---|:---|:---|:---|
| **Primary Use Case** | Baseline teaching, short sequence toy tasks ($T < 10$) | Latency-critical edge NLP, wearable sensor streams, audio | Complex long-document classification, machine translation |
| **Parameter Budget** | Minimalist ($1.0\times$) | Efficient ($3.0\times$ — saves 25% vs LSTM) | Full capacity ($4.0\times$) |
| **Long-Range Retention** | Fails (decays exponentially) | Excellent (adaptive update gate) | Superior (dedicated Constant Error Carousel) |
| **Memory Bandwidth** | Low (stores 1 state per step) | Moderate (stores 2 vectors per step) | High (stores cell state + hidden state + 4 gates) |

---

## 7. Numerical Stability & Traps

1. **BPTT Gradient Explosion:** When $T > 50$, backpropagation through time accumulates gradients that can cause overflow to `inf` or `NaN`.  
   *Remedy:* Always invoke `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`.
2. **Batch Layout Confusion (`batch_first`):** Forgetting `batch_first=True` causes PyTorch to interpret batch slices as temporal sequences, producing silent training failure.  
   *Remedy:* Always assert input shape `assert x.shape == (B, T, D)` and pass `batch_first=True`.
3. **Forget Gate Initialization Bias:** If $b_f$ is initialized to 0 or negative numbers, the network starts by forgetting everything.  
   *Remedy:* Initialize forget gate bias to positive constants ($+1.0$ or $+2.0$) to guarantee memory retention early in training.

