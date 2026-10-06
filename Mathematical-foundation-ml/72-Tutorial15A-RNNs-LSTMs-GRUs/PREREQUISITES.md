# Prerequisites & Mathematical Foundations — Tutorial 15 Part 1: RNNs, LSTMs and GRUs

This document establishes the rigorous mathematical, geometric, and architectural foundations necessary to master **Tutorial 15 Part 1: Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRUs)**.

---

## ⚡ 3-Minute Fast-Track Card

```
Sequential Input Tensor X in R^{B x T x D}
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      RECURRENT STATE DYNAMICS                          │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Vanilla RNN:  h_t = tanh( W_xh x_t + W_hh h_{t-1} + b_h )           │
│    Problem: Repeated Jacobian products cause exponential decay/blowup   │
├────────────────────────────────────────────────────────────────────────┤
│ 2. GRU (2 Gates):                                                      │
│    Reset gate:  r_t = sigma( W_xr x_t + W_hr h_{t-1} + b_r )           │
│    Update gate: z_t = sigma( W_xz x_t + W_hz h_{t-1} + b_z )           │
│    Candidate:   ~h_t = tanh( W_xh x_t + W_hh (r_t (*) h_{t-1}) + b_h ) │
│    Convex Comb: h_t = (1 - z_t) (*) h_{t-1} + z_t (*) ~h_t             │
├────────────────────────────────────────────────────────────────────────┤
│ 3. LSTM (3 Gates + Cell State Highway):                                │
│    Forget gate: f_t = sigma( W_xf x_t + W_hf h_{t-1} + b_f )           │
│    Input gate:  i_t = sigma( W_xi x_t + W_hi h_{t-1} + b_i )           │
│    Candidate:   ~C_t = tanh( W_xc x_t + W_hc h_{t-1} + b_c )           │
│    Cell update: C_t = f_t (*) C_{t-1} + i_t (*) ~C_t (Additive flow)   │
│    Output gate: o_t = sigma( W_xo x_t + W_ho h_{t-1} + b_o )           │
│    Hidden state: h_t = o_t (*) tanh( C_t )                             │
└────────────────────────────────────────────────────────────────────────┘
       │
       ▼  Terminal State h_T in R^{B x H}
┌──────────────────────────────────────────────┐
│ Linear Head: z = W_y h_T + b_y in R^{B x C}  │
│ Softmax Classification: p in Delta^{C-1}     │
└──────────────────────────────────────────────┘
```

### The 3 Core Conceptual Shifts
1. **From Spatial Filtering to Temporal Memory Accumulation:** Unlike CNNs that process fixed spatial grids with local receptive fields, recurrent networks process variable-length ordered sequences by passing an evolving hidden state vector $h_t$ through time.
2. **From Multiplicative Gradient Decay to Additive Gated Highways:** Vanilla RNNs multiply gradient Jacobians at every time step ($W_{hh}^T$), causing vanishing gradients. LSTMs and GRUs introduce additive updates ($C_t = f_t \odot C_{t-1} + \dots$), establishing uninterrupted gradient highways.
3. **From Many-to-Many Output to Many-to-One Classification:** In sequence classification, intermediate hidden states $h_1, \dots, h_{T-1}$ are pooled or discarded, and only the terminal summary representation $h_T$ is projected into class logits.

### 3 Readiness Check Questions
1. If an input sequence has 10 tokens and embedding size 64 with batch size 8, what is its canonical tensor shape under PyTorch `batch_first=True`? *(Answer: `[8, 10, 64]`)*
2. Why does the forget gate in an LSTM use a sigmoid activation $\sigma$ rather than a tanh activation? *(Answer: Sigmoid outputs strictly between 0 and 1, acting as an exact retention fraction/percentage)*
3. What is the ratio of learnable parameters between an LSTM cell and a vanilla RNN cell with identical input and hidden dimensions? *(Answer: Exactly $4:1$ because LSTM has four internal linear transformations vs one in RNN)*

---

## Math Terminology Rosetta Stone

| Symbol | Mathematical Concept | Spoken English (Phonetics) | Mathematical Domain | Plain-English Intuition | Formal Definition | Reference |
|:-------|:---------------------|:---------------------------|:--------------------|:------------------------|:------------------|:----------|
| $\mathbf{X}$ | Input Sequence Batch | /ɛks/ (Input sequence batch) | $\mathbb{R}^{B \times T \times D}$ | 3D collection of $B$ sequential texts/signals | Batch of $B$ sequences of length $T$ with $D$ features | [Tensors & Vectors](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/01-Vectors_and_Matrices.md) |
| $h_t$ | Recurrent Hidden State | /eɪtʃ tiː/ (Hidden state) | $\mathbb{R}^H$ | Short-term scratchpad memory at step $t$ | Working memory vector summarizing history up to $t$ | [Recurrent Networks](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/02-Recurrent_Neural_Networks.md) |
| $C_t$ | LSTM Cell State | /siː tiː/ (Cell state) | $\mathbb{R}^H$ | Long-term memory conveyor belt in LSTMs | Unattenuated linear memory carrier vector | [Recurrent Networks](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/02-Recurrent_Neural_Networks.md) |
| $\sigma(\cdot)$ | Sigmoid Activation | /ˈsɪɡ.mɔɪd/ (Sigmoid) | $\mathbb{R} \to (0, 1)$ | Soft percentage gate (0% to 100%) | Logistic function $\sigma(z) = \frac{1}{1 + e^{-z}}$ | [Activation Functions](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/05-Activation_Functions.md) |
| $\tanh(\cdot)$ | Hyperbolic Tangent | /tæn ˈeɪtʃ/ (Hyperbolic tangent) | $\mathbb{R} \to (-1, 1)$ | Zero-centered squashing function | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | [Activation Functions](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/05-Activation_Functions.md) |
| $\odot$ | Hadamard Product | /ˈhæd.əˌmɑːrd/ (Hadamard product) | Operator | Elementwise multiplication | Pointwise product: $(A \odot B)_i = A_i B_i$ | [Tensors & Vectors](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/01-Vectors_and_Matrices.md) |
| $f_t, i_t, o_t$ | LSTM Gating Vectors | /ɡeɪt ˈvɛk.tərz/ (LSTM Gates) | $[0, 1]^H$ | Forget, input, and output control dials | Gating modulation vectors in LSTM cells | [Recurrent Networks](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/02-Recurrent_Neural_Networks.md) |
| $r_t, z_t$ | GRU Gating Vectors | /ɡruː ɡeɪts/ (GRU Gates) | $[0, 1]^H$ | Reset and update interpolation dials | Gating modulation vectors in GRU cells | [Recurrent Networks](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/02-Recurrent_Neural_Networks.md) |
| $W_{hh}$ | Recurrent Weight Matrix | /wɜːrk ˈweɪt/ (Recurrent weight) | $\mathbb{R}^{H \times H}$ | Transition matrix passed between steps | Reusable matrix connecting consecutive hidden states | [Recurrent Networks](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/02-Recurrent_Neural_Networks.md) |
| $\frac{\partial \mathcal{L}}{\partial h_t}$ | Temporal Gradient Vector | /ˈɡreɪ.di.ənt/ (Temporal gradient) | $\mathbb{R}^H$ | Error signal traveling back in time | BPTT loss sensitivity with respect to hidden state | [Chain Rule](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/04-Chain_Rule_and_Backpropagation.md) |

---

## Curriculum & Sibling Course Prerequisite Bridges

| Prerequisite Milestone | Core Concept Mastered | Critical Role in Tutorial 15 Part 1 | Direct Disk Link |
|:-----------------------|:----------------------|:------------------------------------|:-----------------|
| **Package 64: Lec 48 Attention Part 1** | Content-based Addressing & QKV Routing | Contrasts fixed recurrent hidden compression with dynamic attention addressing | [`64-Lec48-Attention-Part1`](../64-Lec48-Attention-Part1/) |
| **Package 68: Lec 52 Transfer Learning** | Pretrained Feature Extraction & Decoupling | Demonstrates feeding decoupled CNN vision embeddings $Z$ into RNN decoders | [`68-Lec52-Transfer-Learning-Knowledge-Distillation`](../68-Lec52-Transfer-Learning-Knowledge-Distillation/) |
| **Package 69: Lec 53 Optimizers** | Momentum, RMSprop, Adam & Gradient Dynamics | Explains why Adam and gradient norm clipping are vital for training recurrent nets | [`69-Lec53-SGD-RMSprop-Adam-Optimizers`](../69-Lec53-SGD-RMSprop-Adam-Optimizers/) |
| **Package 70: Tutorial 14 Part 1 CNNs** | Discrete Convolutions & PyTorch Modules | Establishes `nn.Module` subclassing, tensor manipulation, and forward loops | [`70-Tutorial14A-CNNs`](../70-Tutorial14A-CNNs/) |

---

<a id="p1"></a>
## Pillar 1: Temporal Hidden State Transitions & Parameter Sharing

### Tier 1: Concrete Intuition & Visual Breakdown
In static neural networks (MLPs or CNNs), an input $\mathbf{x}$ is mapped directly to an output $\hat{\mathbf{y}}$ with no memory of prior inputs. Sequence problems (text, speech, time series) violate independent and identically distributed (i.i.d.) assumptions: the meaning of token $x_t$ depends critically on preceding tokens $x_1, \dots, x_{t-1}$.

A Recurrent Neural Network solves this by maintaining an internal state vector $h_t \in \mathbb{R}^H$. At each time step $t$, the cell ingests:
1. The current external input vector $x_t \in \mathbb{R}^D$.
2. The internal working memory from the preceding step $h_{t-1} \in \mathbb{R}^H$.

Crucially, the weight matrices $W_{xh} \in \mathbb{R}^{H \times D}$ and $W_{hh} \in \mathbb{R}^{H \times H}$ are **strictly shared across all time steps**. This parameter sharing allows the model to process sequences of arbitrary length $T$ without expanding memory requirements.

### Tier 2: Concrete Numbers & Python Verification

#### Hand-Worked Numerical Calculation
Consider scalar dimensions $D=1, H=1$ at time step $t=1$:
- Input $x_1 = 0.50$, previous state $h_0 = 0.0$ (zero initialization).
- Weights: $W_{xh} = 0.80$, $W_{hh} = 0.50$, bias $b_h = 0.10$.
1. Linear combination:
   $$z_1 = W_{xh} x_1 + W_{hh} h_0 + b_h = 0.80 \times 0.50 + 0.50 \times 0.0 + 0.10 = 0.40 + 0.0 + 0.10 = 0.50$$
2. Activation squashing:
   $$h_1 = \tanh(0.50) \approx 0.4621$$
Now at time step $t=2$ with new input $x_2 = -0.20$:
1. Linear combination:
   $$z_2 = W_{xh} x_2 + W_{hh} h_1 + b_h = 0.80 \times (-0.20) + 0.50 \times 0.4621 + 0.10 = -0.16 + 0.2311 + 0.10 = 0.1711$$
2. Activation squashing:
   $$h_2 = \tanh(0.1711) \approx 0.1695$$

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
cell = nn.RNNCell(input_size=1, hidden_size=1, nonlinearity="tanh")
cell.weight_ih.data.fill_(0.80)
cell.weight_hh.data.fill_(0.50)
cell.bias_ih.data.fill_(0.10)
cell.bias_hh.data.fill_(0.0)

x1 = torch.tensor([[0.50]])
h0 = torch.zeros(1, 1)
h1 = cell(x1, h0)
assert torch.isclose(h1, torch.tensor([[0.4621]]), atol=1e-3)

x2 = torch.tensor([[-0.20]])
h2 = cell(x2, h1)
assert torch.isclose(h2, torch.tensor([[0.1695]]), atol=1e-3)
print(f"[PASS] Hand calculation verified: h1={h1.item():.4f}, h2={h2.item():.4f}")
```

### Tier 3: Folded Deep Formal Mathematical Formulation & Guarantees

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

#### Discrete Non-Linear Dynamical System
A recurrent neural network defines a discrete-time non-linear dynamical system:
$$h_t = \Phi(h_{t-1}, x_t; \Theta)$$
where $\Phi: \mathbb{R}^H \times \mathbb{R}^D \to \mathbb{R}^H$ is contractive or expansive depending on the spectrum of the Jacobian matrix $\mathbf{J}_t = \frac{\partial h_t}{\partial h_{t-1}}$. By the Banach Fixed Point Theorem, if $\|\mathbf{J}_t\|_2 < 1$ uniformly, the system possesses a unique stable attractor, but historical memory decays geometrically with time horizon $\Delta t = |t - k|$.
</details>

### Diagnostic Mini-Check
1. Why must $W_{xh}$ and $W_{hh}$ be shared across all time steps?  
   *Answer:* Weight sharing ensures parameter parsimony ($O(1)$ parameters regardless of sequence length $T$) and enables generalization to sequences of unseen lengths.
2. What is the initial hidden state $h_0$ typically set to?  
   *Answer:* It is conventionally initialized to the zero vector $\mathbf{0} \in \mathbb{R}^H$.

---

<a id="p2"></a>
## Pillar 2: Backpropagation Through Time (BPTT) & Exponential Gradient Decay

### Tier 1: Concrete Intuition & Visual Breakdown
To train a recurrent network on a sequence of length $T$, we unfold the computational graph across time. This training algorithm is called **Backpropagation Through Time (BPTT)**.

Suppose a loss $\mathcal{L}_T$ is evaluated at step $T$. To compute the gradient with respect to the initial hidden state $h_1$, the multivariable chain rule accumulates the product of all intermediate temporal Jacobians:
$$\frac{\partial \mathcal{L}_T}{\partial h_1} = \frac{\partial \mathcal{L}_T}{\partial h_T} \left( \prod_{k=2}^T \frac{\partial h_k}{\partial h_{k-1}} \right)$$
For a vanilla RNN, each Jacobian is:
$$\frac{\partial h_k}{\partial h_{k-1}} = \operatorname{diag}(1 - h_k^2) W_{hh}^T$$
Because $|1 - h_k^2| \le 1$ and repeated multiplication by $W_{hh}^T$ acts as power iteration:
- If the maximum singular value $\sigma_{\max}(W_{hh}) < 1$, the gradient decays exponentially to zero: **Vanishing Gradients**. Early steps receive zero update.
- If $\sigma_{\max}(W_{hh}) > 1$, the gradient grows exponentially: **Exploding Gradients**. Parameters diverge to `NaN`.

### Tier 2: Concrete Numbers & Python Verification

#### Hand-Worked Numerical Calculation
Consider a scalar transition weight $W_{hh} = 0.50$ across $T = 10$ steps without input:
1. Product of gradients:
   $$\left( \frac{\partial h_{10}}{\partial h_1} \right) = (W_{hh})^9 = (0.50)^9 = \frac{1}{512} \approx 0.001953$$
2. Over 20 steps:
   $$(0.50)^{19} = \frac{1}{524288} \approx 0.0000019$$
The gradient signal has decayed by a factor of over 500,000!

```python
import torch

T = 15
W_hh = torch.tensor([[0.50]], requires_grad=True)
h = torch.tensor([[1.0]], requires_grad=True)

curr = h
for t in range(T):
    curr = torch.tanh(curr * W_hh)

loss = curr.sum()
loss.backward()

print(f"Gradient at step 1 for T={T}: {h.grad.item():.8f}")
assert h.grad.item() < 1e-4, "Vanilla RNN gradient should have vanished!"
print("[PASS] Exponential vanishing gradient verified.")
```

### Tier 3: Folded Deep Formal Mathematical Formulation & Guarantees

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

#### Spectral Radius Bounds on Temporal Jacobians
Let $\rho(W_{hh})$ denote the spectral radius of $W_{hh}$. A sufficient condition for the gradient norm to vanish as $(T - t) \to \infty$ is:
$$\| \mathbf{J}_{t \to T} \|_2 \le \prod_{k=t+1}^T \| \operatorname{diag}(1 - h_k^2) \|_2 \cdot \| W_{hh} \|_2 \le \gamma^{T-t}$$
where $\gamma = \|W_{hh}\|_2 < 1$. Conversely, if $\gamma > 1$, there exist directions along the principal eigenvector where $\|\mathbf{J}\|_2 \ge \gamma^{T-t}$, causing gradient explosion.
</details>

### Diagnostic Mini-Check
1. What heuristic protects optimizers against exploding gradients during BPTT?  
   *Answer:* Gradient norm clipping (`torch.nn.utils.clip_grad_norm_`).
2. Why cannot gradient clipping solve the vanishing gradient problem?  
   *Answer:* Clipping caps excessively large gradients; it cannot amplify or restore signals that have already decayed to numerical zero.

---

<a id="p3"></a>
## Pillar 3: Gated Recurrent Units (GRU) & Adaptive Convex Interpolation

### Tier 1: Concrete Intuition & Visual Breakdown
Introduced by Cho et al. (2014), the **Gated Recurrent Unit (GRU)** solves vanishing gradients by replacing the rigid state transition with an adaptive gating mechanism:
1. **Reset Gate ($r_t \in [0, 1]^H$):** Decides whether past context should be ignored when computing new candidate information. If $r_t = 0$, the cell acts as if reading the first token of a sequence.
2. **Update Gate ($z_t \in [0, 1]^H$):** Acts as an interpolation weight deciding whether to copy the previous hidden state $h_{t-1}$ or install the new candidate state $\tilde{h}_t$.
3. **Hidden State Convex Combination:**
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
Notice the additive shortcut: if $z_t \approx 0$, $h_t \approx h_{t-1}$, allowing memory and gradients to propagate across thousands of steps with zero attenuation!

### Tier 2: Concrete Numbers & Python Verification

#### Hand-Worked Numerical Calculation
Consider 1D scalar variables at step $t$:
- Previous state $h_{t-1} = 0.80$, candidate state $\tilde{h}_t = -0.40$.
- Update gate $z_t = 0.25$ (retain 75% old, incorporate 25% new).
1. State combination:
   $$h_t = (1 - 0.25) \times 0.80 + 0.25 \times (-0.40) = 0.75 \times 0.80 + (-0.10) = 0.60 - 0.10 = 0.50$$
2. Backward derivative with respect to previous state $h_{t-1}$:
   $$\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) = 1.0 - 0.25 = 0.75$$
Even if candidate weights vanish, $0.75$ of the gradient flows directly backward!

```python
import torch
import torch.nn as nn

B, D, H = 1, 2, 2
gru_cell = nn.GRUCell(D, H)
x = torch.randn(B, D)
h_prev = torch.randn(B, H)

h_next = gru_cell(x, h_prev)
assert h_next.shape == (B, H)
print(f"[PASS] GRU Cell forward executed cleanly: shape={list(h_next.shape)}")
```

### Tier 3: Folded Deep Formal Mathematical Formulation & Guarantees

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

#### Unattenuated Gradient Highway
Expanding the derivative of the GRU objective:
$$\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) \mathbf{I} + \text{residual terms}$$
When $z_t \to 0$, the Jacobian approaches the identity matrix $\mathbf{I}$. This creates an unimpeded gradient flow similar to residual connections in ResNets ($x + \mathcal{F}(x)$).
</details>

### Diagnostic Mini-Check
1. How many internal gate vectors does a GRU compute per time step?  
   *Answer:* Two: reset gate $r_t$ and update gate $z_t$.
2. What role does $1 - z_t$ play in the hidden state update equation?  
   *Answer:* It controls the fraction of the past hidden state $h_{t-1}$ retained in the current step.

---

<a id="p4"></a>
## Pillar 4: Long Short-Term Memory (LSTM) & Dual State Constant Error Carousels

### Tier 1: Concrete Intuition & Visual Breakdown
Developed by Hochreiter & Schmidhuber (1997), the **Long Short-Term Memory (LSTM)** network decouples memory into two distinct pathways:
1. **Cell State ($C_t \in \mathbb{R}^H$):** The *long-term memory highway*. It flows along the top of the cell with only linear interactions (multiplication by forget gate and addition of input gate).
2. **Hidden State ($h_t \in \mathbb{R}^H$):** The *working output memory*. It is squashed by $\tanh$ and filtered by the output gate $o_t$ to emit predictions.

The Three Gates:
- **Forget Gate ($f_t = \sigma(\dots)$):** Decides what fraction of long-term cell state $C_{t-1}$ to retain.
- **Input Gate ($i_t = \sigma(\dots)$):** Decides what fraction of candidate memory $\tilde{C}_t = \tanh(\dots)$ to store.
- **Output Gate ($o_t = \sigma(\dots)$):** Decides what portion of updated cell state $C_t$ to expose in $h_t$.

Because $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ is an **additive linear update**, the derivative $\frac{\partial C_t}{\partial C_{t-1}} = f_t$. If $f_t = 1$, the gradient flows backward indefinitely without decay—forming a **Constant Error Carousel (CEC)**.

### Tier 2: Concrete Numbers & Python Verification

#### Hand-Worked Numerical Calculation
Consider 1D scalar LSTM states at step $t$:
- Past cell state $C_{t-1} = 4.0$, forget gate $f_t = 0.90$.
- Input gate $i_t = 0.50$, candidate memory $\tilde{C}_t = 0.60$.
- Output gate $o_t = 0.80$.
1. Cell state update:
   $$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t = 0.90 \times 4.0 + 0.50 \times 0.60 = 3.60 + 0.30 = 3.90$$
2. Emitted hidden state:
   $$h_t = o_t \times \tanh(C_t) = 0.80 \times \tanh(3.90) \approx 0.80 \times 0.9993 = 0.7994$$
3. Gradient highway:
   $$\frac{\partial C_t}{\partial C_{t-1}} = f_t = 0.90$$
The cell state retains 90% of both information and gradient!

```python
import torch
import torch.nn as nn

B, D, H = 1, 4, 3
lstm_cell = nn.LSTMCell(D, H)
x = torch.randn(B, D)
h0 = torch.zeros(B, H)
c0 = torch.zeros(B, H)

h1, c1 = lstm_cell(x, (h0, c0))
assert h1.shape == (B, H)
assert c1.shape == (B, H)
print(f"[PASS] LSTM Cell forward executed cleanly: h1={list(h1.shape)}, c1={list(c1.shape)}")
```

### Tier 3: Folded Deep Formal Mathematical Formulation & Guarantees

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

#### Constant Error Carousel (CEC) Theorem
Let $\mathcal{E}$ be the error backpropagated through time from step $T$ to step $t$. Across the cell state channel:
$$\frac{\partial \mathcal{E}}{\partial C_t} = \frac{\partial \mathcal{E}}{\partial C_T} \prod_{k=t+1}^T \operatorname{diag}(f_k)$$
If the forget gate biases are initialized positively ($b_f > 0$ such that $f_k \approx 1$), then $\prod_{k=t+1}^T f_k \approx 1$. Thus, the error carousel remains strictly non-vanishing over arbitrary sequence lengths $T$.
</details>

### Diagnostic Mini-Check
1. Why is initializing the forget gate bias $b_f$ to a positive constant (e.g. $+1.0$ or $+2.0$) standard engineering practice?  
   *Answer:* It forces $f_t \approx 1.0$ initially, ensuring the network remembers past context by default until it learns what to forget.
2. What are the two state tensors returned by an LSTM cell?  
   *Answer:* Working hidden state $h_t$ and long-term cell state $C_t$.

---

<a id="p5"></a>
## Pillar 5: Parameter Counting Arithmetic & Architectural Trade-offs

### Tier 1: Concrete Intuition & Visual Breakdown
When designing neural sequence models, calculating the memory footprint and parameter counts is essential for latency and capacity budgeting.
For any recurrent cell with input dimension $D$ and hidden state dimension $H$:
- The affine projection maps concatenated inputs $[x_t, h_{t-1}] \in \mathbb{R}^{D + H}$ to internal gates in $\mathbb{R}^H$.
- Weight matrix size per gate: $H \times (D + H)$ elements.
- Bias vector size per gate: $H$ elements (or $2H$ in PyTorch due to separate input and recurrent bias buffers).

Comparison:
1. **Vanilla RNN (1 projection):** $N = 1 \times [H(D + H) + 2H]$
2. **GRU (3 projections):** $N = 3 \times [H(D + H) + 2H]$
3. **LSTM (4 projections):** $N = 4 \times [H(D + H) + 2H]$

GRU achieves a **25% parameter reduction** compared to LSTM by merging cell and hidden states and removing the output gate.

### Tier 2: Concrete Numbers & Python Verification

#### Hand-Worked Numerical Calculation
Consider $D = 100$ (word embedding) and $H = 256$ (hidden units) under standard single-bias formulation:
- Per-gate parameter size:
  $$\text{Gate} = H \times (D + H) + H = 256 \times (100 + 256) + 256 = 256 \times 356 + 256 = 91136 + 256 = 91392$$
1. Vanilla RNN:
   $$N_{\text{RNN}} = 1 \times 91392 = 91,392 \text{ parameters}$$
2. GRU:
   $$N_{\text{GRU}} = 3 \times 91392 = 274,176 \text{ parameters}$$
3. LSTM:
   $$N_{\text{LSTM}} = 4 \times 91392 = 365,568 \text{ parameters}$$
Difference: $N_{\text{LSTM}} - N_{\text{GRU}} = 365568 - 274176 = 91,392$ parameters saved.

```python
import torch.nn as nn

D, H = 100, 256
rnn = nn.RNNCell(D, H)
gru = nn.GRUCell(D, H)
lstm = nn.LSTMCell(D, H)

# Note: PyTorch allocates separate bias_ih and bias_hh (2*H per gate)
p_rnn = sum(p.numel() for p in rnn.parameters())
p_gru = sum(p.numel() for p in gru.parameters())
p_lstm = sum(p.numel() for p in lstm.parameters())

print(f"PyTorch Params -> RNN: {p_rnn}, GRU: {p_gru}, LSTM: {p_lstm}")
assert p_gru == 3 * p_rnn
assert p_lstm == 4 * p_rnn
print("[PASS] Exact 1 : 3 : 4 parameter proportionality verified.")
```

### Tier 3: Folded Deep Formal Mathematical Formulation & Guarantees

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

#### Computational Complexity per Step
For a sequence of length $T$, the FLOP complexity for recurrent cells is dominated by matrix-vector multiplications:
$$\mathcal{O}(T \cdot K \cdot H \cdot (D + H))$$
where $K \in \{1, 3, 4\}$ denotes the gate count factor. Memory bandwidth during training scales with the saved activations required for BPTT: LSTM stores $(h_t, C_t, f_t, i_t, \tilde{C}_t, o_t) \in \mathbb{R}^{6H}$ per step, whereas GRU stores $(h_t, r_t, z_t, \tilde{h}_t) \in \mathbb{R}^{4H}$, offering significant VRAM savings.
</details>

### Diagnostic Mini-Check
1. How many parameters does a PyTorch `nn.LSTMCell(input_size=10, hidden_size=20)` have?  
   *Answer:* $4 \times [20 \times (10 + 20) + 2 \times 20] = 4 \times [600 + 40] = 4 \times 640 = 2,560$ parameters.
2. In low-compute edge applications, why might an engineer choose GRU over LSTM?  
   *Answer:* GRU reduces parameters and activation memory by 25% while maintaining comparable representational capacity.

---

## 🗺️ Cross-Reference & Lecture Mapping

| Mathematical Pillar | Direct Lecture Topic | Downstream Course Impact |
|:---|:---|:---|
| [Pillar 1: Temporal Hidden State Transitions](#p1) | Topic 1: Sequence Tensor Geometry & Vanilla RNN Cell | Fundamental sequence processing baseline |
| [Pillar 2: BPTT & Exponential Gradient Decay](#p2) | Topic 2: Vanishing Gradients in Recurrence | Motivates gate designs and Transformer attention |
| [Pillar 3: Gated Recurrent Units (GRU)](#p3) | Topic 2: GRU Mechanics & Gating Equations | Efficient lightweight sequence modeling |
| [Pillar 4: Long Short-Term Memory (LSTM)](#p4) | Topic 3: LSTM Cell States & Triple Gating | State-of-the-art recurrent benchmark for NLP |
| [Pillar 5: Parameter Counting Arithmetic](#p5) | Topic 4: Parameter Counting & Cell Comparisons | Model capacity budgeting and inference optimization |
