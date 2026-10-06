# Prerequisites — Lec 45: Recurrent Neural Networks (RNNs)

> **Do this first.** Then open [NOTES.md](./NOTES.md) at the **Executive Summary** map.  
> Basics only — not a second lecture. They unlock words on the master map if you are rusty.  
> **How to read:** Use the **3-Minute Executive Fast-Track** below for a rapid ramp-up. Read the 👶 Physical Intuition and 💻 Python snippets in each pillar. Expand the 📐 formal calculus blocks only when you need deep mathematical proofs.

```
  After this warm-up you can say:

  "Dual-stream linear fusion is mathematically a single block-matrix GEMM."
  "The hyperbolic tangent derivative peaks at 1.0 at the origin and saturates to 0."
  "RNNs are non-linear autonomous state-space dynamical systems with input driving."
  "The probability chain rule factors joint sequence likelihood into one-step conditionals."
  "Unrolled RNNs are causal block-Toeplitz MLPs with tied subdiagonal weights."
```

---

## ⚡ 3-Minute Executive Fast-Track

If you have only 3 minutes before starting the lecture, master this visual blueprint:

```
  RECURRENT CELL FUSION & TEMPORAL UNROLLING:
  Input x_t ∈ ℝᵈ ────► [ W_xh ] ──┐
                                  ├──► [ + b_h ] ──► z_t ──► σ(·) ──► h_t ∈ ℝᵐ ──► Readout ŷ_t
  Past State h_{t-1} ──► [ W_hh ] ──┘                                   │
                                                                        └──► Next Step h_{t+1}

  GLOBAL UNROLLED OPERATOR: Causal Block-Toeplitz Matrix ℳ ∈ ℝ^((Tm) × (Td))
```

### 🧠 The 3 Core Mental Shifts
1. **Dual-Stream Fusion as Single Joint GEMM:** In an RNN cell, combining memory $W_{hh} h_{t-1}$ and observation $W_{xh} x_t$ is mathematically isomorphic to a single dense affine transformation $[W_{hh}, W_{xh}] \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_h$ on the concatenated direct-sum space $\mathbb{R}^m \oplus \mathbb{R}^d$.
2. **RNNs are Non-Linear Dynamical Systems:** Classical control systems model state transitions as $s_t = A s_{t-1} + B u_t$. An RNN adopts this exact time-invariant principle, but wraps the affine transition inside a non-linear squashing function $\tanh(z)$ to bound memory energy in $(-1, 1)$ and prevent state explosion.
3. **Causality & Weight Tying = Block-Toeplitz:** Unrolling an RNN across $T$ steps reveals it as an MLP whose global transformation matrix is strictly block lower-triangular (causality: future inputs cannot alter past memory) and block-Toeplitz (subdiagonals are tied powers $W_{hh}^k W_{xh}$).

### ⏱️ Instant Readiness Check
1. *If hidden memory $h \in \mathbb{R}^{16}$ and input $x \in \mathbb{R}^{32}$, what are the dimensions of the joint block weight matrix $W_{\text{joint}} = [W_{hh}, W_{xh}]$ mapping to $h_{\text{new}} \in \mathbb{R}^{16}$?*  
   <details><summary><b>Reveal Answer</b></summary><b>$16 \times 48$</b> (concatenating $W_{hh} \in \mathbb{R}^{16 \times 16}$ and $W_{xh} \in \mathbb{R}^{16 \times 32}$ horizontally).</details>
2. *What is the numerical derivative $\frac{d}{dz} \tanh(z)$ at $z = 0$, and what happens as $|z| \to \infty$?*  
   <details><summary><b>Reveal Answer</b></summary><b>$\frac{d}{dz} \tanh(0) = 1.0$</b>. As $|z| \to \infty$, $\tanh(z) \to \pm 1$, causing the derivative $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z) \to 0$ (activation saturation / gradient vanishing).</details>
3. *Why does the global unrolled transformation matrix of an RNN have zeros in all upper-triangular block entries?*  
   <details><summary><b>Reveal Answer</b></summary><b>Temporal causality.</b> In physical reality, future observations $x_{t+k}$ cannot influence past hidden states $h_t$.</details>

---

## Math Terminology Rosetta Stone

Before diving into the foundational pillars, use this reference table to decode mathematical shorthand and notation into plain English, spoken phonetics, and software implementations.

| Symbol | Spoken Phonetics | Math Meaning | Operational Role in Lecture | Plain-English Intuition |
|:-------|:-----------------|:-------------|:----------------------------|:------------------------|
| $x_t \in \mathbb{R}^d$ | "x sub t" | Observation vector at time step $t$ | Input token or sensor reading injected into the RNN cell at step $t$ | The current word you are reading or the current vital sign measured right now. |
| $h_t \in \mathbb{R}^m$ | "h sub t" | Latent recurrent hidden state vector | Evolving internal memory vector summarizing historical context from $1$ to $t$ | The running summary in your working memory after reading up to word $t$. |
| $W_{hh} \in \mathbb{R}^{m \times m}$ | "W sub h h" | Hidden-to-hidden transition weight matrix | Transforms and filters previous memory $h_{t-1}$ into the current state | The memory update rule dictating how past context influences current thought. |
| $W_{xh} \in \mathbb{R}^{m \times d}$ | "W sub x h" | Input-to-hidden observation projection matrix | Embeds external input vector $x_t$ into the latent memory dimension $m$ | The sensory translator that converts raw input tokens into cognitive thoughts. |
| $W_{hy} \in \mathbb{R}^{d_{\text{out}} \times m}$ | "W sub h y" | Hidden-to-output readout projection matrix | Maps latent hidden state $h_t$ to output logits or task predictions $\hat{y}_t$ | The spokesperson translating internal mental state into spoken words or actions. |
| $z_t \in \mathbb{R}^m$ | "z sub t" | Recurrent cell pre-activation vector | Dual-stream linear fusion: $z_t = W_{hh} h_{t-1} + W_{xh} x_t + b_h$ | The raw, uncompressed blend of old memory and new observation before non-linear gating. |
| $\sigma(\cdot)$ | "sigma of dot" | Pointwise non-linear activation ($\tanh$ or sigmoid) | Squashes pre-activations into bounded range $(-1, 1)$ or $(0, 1)$ | The non-linear saturation preventing memory values from exploding to infinity. |
| $\hat{y}_t$ | "y-hat sub t" | Network prediction at temporal step $t$ | Predicted class distribution, continuous value, or token logit | The model's best guess for what label or next word should appear at step $t$. |
| $T \in \mathbb{N}$ | "capital T" | Temporal sequence length | Total number of observation steps in sample $X = (x_1, \dots, x_T)$ | The total word count of a sentence or duration of a recording. |
| $c \equiv h_T \in \mathbb{R}^m$ | "context vector c" | Terminal latent context representation | Summary embedding passed from Seq2Seq encoder to initialize decoder | An executive one-page briefing note summarizing an entire document. |
| $\text{EOS}$ | "E-O-S" | End-Of-Sequence delimiter token | Reserved token signaling the dynamic termination of generated sequences | The final period at the end of a sentence telling the writer to stop. |

---

## 🌉 Curriculum & Sibling Course Prerequisite Bridges

| Prior Course / Package | Mathematical Concept | Bridge to Lecture 45 | Target Topic in NOTES.md |
|:-----------------------|:---------------------|:---------------------|:-------------------------|
| Package 54 (Lec 41) | Universal Approximation Theorem (UAT) | Expressivity guarantees of compositional non-linear hypothesis spaces | [`Topic 2: Supervised Sequence Tasks`](./NOTES.md#topic-2-supervised-sequence-tasks--the-variable-length-failure-of-dense-mlps-05001130) |
| Package 55 (Lec 42) | Error Backpropagation & Chain Rule | Reverse-mode automatic differentiation on DAG computational graphs | [`Topic 4: Computational Graph Unrolling`](./NOTES.md#topic-4-computational-graph-unrolling--parameter-sharing-across-time-18002430) |
| Package 56 (Lec 43) | Spatial Parameter Sharing & Toeplitz Convolutions | Translation equivariance across spatial grid vs temporal sequence order | [`Topic 4: Parameter Sharing Across Time`](./NOTES.md#topic-4-computational-graph-unrolling--parameter-sharing-across-time-18002430) |
| Package 57 (Lec 44) | CNN Multi-Channel Feature Banks & Regularized MLPs | Constrained MLP operator perspective generalized from 2D grids to 1D time | [`Topic 6: RNNs as Structured Causal MLPs`](./NOTES.md#topic-6-deep-stacked-rnns--rnns-as-structured-causal-block-toeplitz-mlps-30003400) |
| Lecture 46 Preview | Backpropagation Through Time (BPTT) | Multiplicative temporal Jacobian decay and vanishing/exploding gradients | [`Topic 7: Bridge to BPTT`](./NOTES.md#topic-7-probabilistic-auto-regressive-generation--multi-modal-vision-language-pipelines-34003805) |

---

## 1. Vector Spaces, Affine Maps & Block-Matrix Algebra

<a id="p1"></a>

### 👶 Physical Analogy & Intuition
Imagine a DJ mixer with two incoming audio cables: Cable 1 carries recorded background music (historical memory $h$), and Cable 2 carries live microphone vocals (new observation $x$). You could buy two completely separate amplifiers, adjust their volume knobs independently, and solder their output wires together. Or, you can plug both cables into a single dual-channel mixing board. Block-matrix algebra is that dual mixing board: it bundles both data streams into one wide vector and processes them with a single combined matrix multiplication.

### 🔍 Plain-English Breakdown
- An RNN cell receives two inputs at each step: the previous hidden state $h_{t-1} \in \mathbb{R}^m$ and the current input $x_t \in \mathbb{R}^d$.
- The separate affine maps $W_{hh} h_{t-1} + W_{xh} x_t + b_h$ can be written compactly by concatenating the matrices horizontally and the vectors vertically:
  $$W_{\text{joint}} = \begin{bmatrix} W_{hh} & W_{xh} \end{bmatrix} \in \mathbb{R}^{m \times (m + d)}, \quad v_t = \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} \in \mathbb{R}^{m + d}$$
- Then $z_t = W_{\text{joint}} v_t + b_h$, enabling high-speed vectorized GEMM execution on modern GPUs.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let $m = 2, d = 2$:
$$W_1 = \begin{bmatrix} 0.5 & -0.2 \\ 0.1 & 0.8 \end{bmatrix}, \quad W_2 = \begin{bmatrix} 1.0 & 0.0 \\ 0.5 & -1.0 \end{bmatrix}, \quad b = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$$
$$h = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}, \quad x = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix}$$
1. Separate stream products:
   $$W_1 h = \begin{bmatrix} 0.5(1.0) - 0.2(0.5) \\ 0.1(1.0) + 0.8(0.5) \end{bmatrix} = \begin{bmatrix} 0.4 \\ 0.5 \end{bmatrix}$$
   $$W_2 x = \begin{bmatrix} 1.0(2.0) + 0.0(-1.0) \\ 0.5(2.0) - 1.0(-1.0) \end{bmatrix} = \begin{bmatrix} 2.0 \\ 2.0 \end{bmatrix}$$
2. Sum with bias:
   $$z = \begin{bmatrix} 0.4 \\ 0.5 \end{bmatrix} + \begin{bmatrix} 2.0 \\ 2.0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 2.5 \\ 2.4 \end{bmatrix}$$
3. Block formulation check:
   $$W_{\text{joint}} = \begin{bmatrix} 0.5 & -0.2 & 1.0 & 0.0 \\ 0.1 & 0.8 & 0.5 & -1.0 \end{bmatrix}, \quad v = \begin{bmatrix} 1.0 \\ 0.5 \\ 2.0 \\ -1.0 \end{bmatrix}$$
   $$W_{\text{joint}} v + b = \begin{bmatrix} 0.5 - 0.1 + 2.0 + 0.0 \\ 0.1 + 0.4 + 1.0 + 1.0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 2.5 \\ 2.4 \end{bmatrix}$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

W1 = np.array([[0.5, -0.2], [0.1, 0.8]])
W2 = np.array([[1.0, 0.0], [0.5, -1.0]])
b = np.array([0.1, -0.1])
h = np.array([1.0, 0.5])
x = np.array([2.0, -1.0])

z_separate = W1 @ h + W2 @ x + b
W_joint = np.hstack([W1, W2])
v = np.concatenate([h, x])
z_joint = W_joint @ v + b

np.testing.assert_allclose(z_separate, z_joint, rtol=1e-6)
assert np.allclose(z_separate, np.array([2.5, 2.4]))
print("[P1 PASS] Block matrix fusion verified:", z_joint)
```

### 🩺 Diagnostic Mini-Check
**Question:** If $h \in \mathbb{R}^{16}$ and $x \in \mathbb{R}^{32}$, what are the dimensions of the concatenated vector $[h; x]$ and joint weight matrix $W_{\text{joint}}$ mapping to $\mathbb{R}^{16}$?  
<details><summary><b>Reveal Answer</b></summary><b>$[h; x] \in \mathbb{R}^{48}$</b> and <b>$W_{\text{joint}} \in \mathbb{R}^{16 \times 48}$</b>.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let vector spaces be $\mathcal{H} = \mathbb{R}^m$ and $\mathcal{X} = \mathbb{R}^d$.
Consider two independent linear maps $L_1: \mathcal{H} \to \mathcal{H}$ represented by matrix $W_1 \in \mathbb{R}^{m \times m}$ and $L_2: \mathcal{X} \to \mathcal{H}$ represented by matrix $W_2 \in \mathbb{R}^{m \times d}$, with shared bias $b \in \mathbb{R}^m$.
The linear combination is:
$$z = W_1 h + W_2 x + b$$
Define the partitioned block matrix $W_{\text{joint}} \in \mathbb{R}^{m \times (m + d)}$ and concatenated column vector $v \in \mathbb{R}^{m + d}$ as:
$$W_{\text{joint}} = \begin{bmatrix} W_1 & W_2 \end{bmatrix}, \quad v = \begin{bmatrix} h \\ x \end{bmatrix}$$
By block matrix multiplication:
$$W_{\text{joint}} v + b = \begin{bmatrix} W_1 & W_2 \end{bmatrix} \begin{bmatrix} h \\ x \end{bmatrix} + b = W_1 h + W_2 x + b = z$$
Thus, dual-stream linear fusion in an RNN cell is mathematically isomorphic to a single dense affine map operating on the direct-sum space $\mathbb{R}^m \oplus \mathbb{R}^d$.
</details>

---

## 2. Feedforward Multi-Layer Perceptrons & Non-linear Activations

<a id="p2"></a>

### 👶 Physical Analogy & Intuition
Think of a rubber spring. If you pull it gently within its normal elastic range, the stretch is linear and easy to control. But if you pull it with extreme force, it reaches its physical limit and resists further stretching—it saturates. In an RNN cell, the hyperbolic tangent $\tanh$ acts as this non-linear spring: it squashes pre-activations into the bounded interval $(-1, 1)$, preventing memory values from compounding out of control across hundreds of time steps.

### 🔍 Plain-English Breakdown
- Without non-linear activations, an unrolled RNN across $T$ steps collapses into a single linear map $W^T x$, rendering depth meaningless.
- In recurrent cells, $\tanh$ is the standard activation function because its outputs are zero-centered around 0.
- The derivative of $\tanh$ is $\sigma'(z) = 1 - \tanh^2(z)$, which reaches its maximum of $1.0$ at $z = 0$ and decays rapidly toward $0$ as $|z|$ grows.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let $z = 0.5$:
$$e^{0.5} \approx 1.648721, \quad e^{-0.5} \approx 0.606531$$
$$\tanh(0.5) = \frac{1.648721 - 0.606531}{1.648721 + 0.606531} = \frac{1.042190}{2.255252} \approx 0.462117$$
Derivative:
$$\frac{d}{dz} \tanh(0.5) = 1 - (0.462117)^2 = 1 - 0.213552 \approx 0.786448$$
Notice that even for a moderate pre-activation of $0.5$, the gradient sensitivity is scaled down by $21.35\%$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

z = 0.5
val = np.tanh(z)
grad = 1.0 - val**2

expected_val = 0.462117
expected_grad = 0.786448

assert abs(val - expected_val) < 1e-5
assert abs(grad - expected_grad) < 1e-5
print("[P2 PASS] Hyperbolic tangent mechanics verified: val =", val, "grad =", grad)
```

### 🩺 Diagnostic Mini-Check
**Question:** Compute the derivative $\frac{d}{dz} \tanh(z)$ evaluated at pre-activation $z = 0$. What is the value?  
<details><summary><b>Reveal Answer</b></summary><b>$1.0$</b> ($\frac{d}{dz} \tanh(0) = 1 - \tanh^2(0) = 1 - 0 = 1.0$).</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

The hyperbolic tangent function $\tanh: \mathbb{R} \to (-1, 1)$ satisfies:
1. **Symmetry:** $\tanh(-z) = -\tanh(z)$ (strictly zero-centered).
2. **Bounds:** $\lim_{z \to \pm\infty} \tanh(z) = \pm 1$.
3. **First Derivative:** Using the quotient rule:
   $$\frac{d}{dz} \tanh(z) = \frac{(e^z + e^{-z})^2 - (e^z - e^{-z})^2}{(e^z + e^{-z})^2} = 1 - \tanh^2(z)$$
Since $\tanh^2(z) \in [0, 1)$, the derivative satisfies $0 < \frac{d}{dz}\tanh(z) \le 1.0$, with peak at $z = 0$.
Under repeated compositions $h_t = \tanh(W_{hh} h_{t-1})$, the temporal Jacobian $\frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}(1 - h_t^2) W_{hh}$ contracts exponentially if singular values of $W_{hh}$ are below 1 or if neurons saturate ($|h_t| \to 1$).
</details>

---

## 3. Empirical Risk Minimization & Computational Graph Execution

<a id="p3"></a>

### 👶 Physical Analogy & Intuition
Imagine a factory conveyor belt where the same robotic arm welds parts at 10 different stations as the product moves down the line. When a finished product fails inspection at the end of the line, how much blame does that robotic arm bear? The arm was involved in 10 different operations! To calculate its total repair adjustment, you must sum the error contributions across all 10 stations where that single robot arm touched the workpiece.

### 🔍 Plain-English Breakdown
- In an RNN, parameter matrices $W_{hh}, W_{xh}, W_{hy}$ are reused at every time step $t \in \{1, \dots, T\}$.
- Because the parameters are shared, their total gradient with respect to the sequence loss is the **sum of gradients across all timesteps**:
  $$\frac{\partial \mathcal{L}}{\partial W} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial z_t} \frac{\partial z_t}{\partial W}$$
- Automatic differentiation traverses the unrolled computation graph in reverse, accumulating gradients at each node where the shared weights appear.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let a 2-step recurrent chain have loss $\mathcal{L} = \frac{1}{2} (\hat{y}_2 - y)^2$ with $\hat{y}_2 = h_2$.
Scalar equations: $h_1 = w x_1$, $h_2 = w h_1 + x_2$.
Inputs: $x_1 = 2.0, x_2 = 1.0, y = 10.0, w = 2.0$.
- **Forward pass:**
  $$h_1 = 2.0(2.0) = 4.0$$
  $$h_2 = 2.0(4.0) + 1.0 = 9.0$$
  $$\mathcal{L} = \frac{1}{2} (9.0 - 10.0)^2 = 0.5$$
- **Backward pass:**
  $$\delta_2 = \frac{\partial \mathcal{L}}{\partial h_2} = h_2 - y = 9.0 - 10.0 = -1.0$$
  $$\delta_1 = \delta_2 \frac{\partial h_2}{\partial h_1} = (-1.0)(w) = (-1.0)(2.0) = -2.0$$
- **Accumulated weight gradient:**
  $$\frac{\partial \mathcal{L}}{\partial w} = \delta_2 \frac{\partial^+ h_2}{\partial w} + \delta_1 \frac{\partial^+ h_1}{\partial w} = (-1.0)(h_1) + (-2.0)(x_1) = (-1.0)(4.0) + (-2.0)(2.0) = -4.0 - 4.0 = -8.0$$

### 💻 Standalone Executable Python Verification
```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x1 = torch.tensor(2.0)
x2 = torch.tensor(1.0)
y = torch.tensor(10.0)

h1 = w * x1
h2 = w * h1 + x2
loss = 0.5 * (h2 - y)**2
loss.backward()

assert abs(loss.item() - 0.5) < 1e-6
assert abs(w.grad.item() - (-8.0)) < 1e-6
print("[P3 PASS] Computational Graph ERM gradient verified:", w.grad.item())
```

### 🩺 Diagnostic Mini-Check
**Question:** What is the computational complexity of evaluating the loss and gradients across a sequence of length $T$ with hidden state dimension $m$?  
<details><summary><b>Reveal Answer</b></summary><b>$\mathcal{O}(T \cdot m^2)$ operations</b>, scaling strictly linearly with sequence length $T$.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let loss $\mathcal{L} = \ell(y, \hat{y})$ where $\hat{y} = g(h_T)$ and $h_t = f(h_{t-1}, x_t; \theta)$.
The gradient of loss with respect to parameters $\theta$ shared across all timesteps is given by the multivariable chain rule on computational graphs:
$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial^+ h_t}{\partial \theta}$$
where $\frac{\partial^+ h_t}{\partial \theta}$ represents the immediate partial derivative holding $h_{t-1}$ constant.
The error sensitivity $\delta_t \equiv \frac{\partial \mathcal{L}}{\partial h_t}$ satisfies the backward recurrence:
$$\delta_{t-1} = \frac{\partial \mathcal{L}}{\partial h_{t-1}} = \delta_t \cdot \frac{\partial h_t}{\partial h_{t-1}}$$
</details>

---

## 4. Autonomous Linear Dynamical Systems & State-Space Transitions

<a id="p4"></a>

### 👶 Physical Analogy & Intuition
Imagine pushing a child on a playground swing. If you push at the right rhythm, energy accumulates and the swing goes higher; if there is friction, the swing gradually slows down when you stop pushing. In physics and engineering, this is a state-space dynamical system: internal motion (the swing's state) evolves based on its own momentum plus external pushes. An RNN works identically: past thoughts evolve over time, driven by incoming sensory tokens.

### 🔍 Plain-English Breakdown
- A discrete Linear Time-Invariant (LTI) system evolves via $s_t = A s_{t-1} + B u_t$.
- Expanding this recurrence yields:
  $$s_t = A^t s_0 + \sum_{k=1}^t A^{t-k} B u_k$$
- If the eigenvalues of transition matrix $A$ have magnitude $|\lambda| < 1$, past memory decays exponentially (forgetting). If $|\lambda| > 1$, memory energy explodes.
- RNNs mirror this exact state-space formulation, stabilizing it by bounding states via $\tanh$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let $A = \begin{bmatrix} 0.8 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}, B = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}, s_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$.
Inputs $u_1 = 1.0, u_2 = 2.0$:
- Step 1:
  $$s_1 = A s_0 + B u_1 = \mathbf{0} + \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} (1.0) = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$$
- Step 2:
  $$s_2 = A s_1 + B u_2 = \begin{bmatrix} 0.8 & 0.0 \\ 0.0 & 0.5 \end{bmatrix} \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} + \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}(2.0) = \begin{bmatrix} 0.8 \\ 0.5 \end{bmatrix} + \begin{bmatrix} 2.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 2.8 \\ 2.5 \end{bmatrix}$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

A = np.array([[0.8, 0.0], [0.0, 0.5]])
B = np.array([[1.0], [1.0]])
s = np.zeros((2, 1))

u_seq = [1.0, 2.0]
for u in u_seq:
    s = A @ s + B * u

expected = np.array([[2.8], [2.5]])
np.testing.assert_allclose(s, expected, rtol=1e-5)
print("[P4 PASS] LTI State-Space Recurrence verified:\n", s)
```

### 🩺 Diagnostic Mini-Check
**Question:** If matrix $A$ in $h_t = A h_{t-1}$ has an eigenvalue $\lambda = 1.5$, what happens to $\|h_t\|$ as $t \to \infty$?  
<details><summary><b>Reveal Answer</b></summary><b>The state norm grows exponentially toward infinity ($\sim 1.5^t$)</b>, causing numerical explosion. In an RNN, non-linear saturation via $\tanh$ prevents unbounded growth.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

A discrete-time Autonomous Linear Time-Invariant (LTI) state-space system is defined by:
$$s_t = A s_{t-1} + B u_t, \quad y_t = C s_t + D u_t$$
where $s_t \in \mathbb{R}^m$ is internal state, $u_t \in \mathbb{R}^d$ is external input, and $A, B, C, D$ are stationary matrices.
By mathematical induction, for any arbitrary time $t \ge 1$:
$$s_t = A^t s_0 + \sum_{k=1}^t A^{t-k} B u_k$$
Spectral stability requires the spectral radius $\rho(A) = \max_i |\lambda_i(A)| \le 1$.
</details>

---

## 5. Discrete Probability & Joint Chain Rule Factorization

<a id="p5"></a>

### 👶 Physical Analogy & Intuition
Think of predicting the weather for the upcoming week. The probability of having rain on Monday, snow on Tuesday, and sunshine on Wednesday is: (Chance of rain on Monday) $\times$ (Chance of snow on Tuesday, GIVEN it rained Monday) $\times$ (Chance of sunshine Wednesday, GIVEN rain Monday and snow Tuesday). You do not need to assume days are independent; the chain rule breaks the complex joint future into a sequence of step-by-step predictions.

### 🔍 Plain-English Breakdown
- The **probabilistic chain rule** decomposes any joint probability distribution over $T$ sequential tokens $(Y_1, \dots, Y_T)$ into a product of one-step conditional probabilities:
  $$P(Y_1, \dots, Y_T \mid X) = \prod_{t=1}^T P(Y_t \mid Y_{<t}, X)$$
- This exact decomposition powers **auto-regressive generative models** (RNNs, LSTMs, Transformers, LLMs): at each step $t$, the network outputs logits parameterizing $P(Y_t \mid Y_{<t})$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Consider a 3-token sentence $Y = (\text{"the"}, \text{"cat"}, \text{"sat"})$:
$$P(Y_1 = \text{"the"}) = 0.10$$
$$P(Y_2 = \text{"cat"} \mid Y_1 = \text{"the"}) = 0.05$$
$$P(Y_3 = \text{"sat"} \mid Y_1 = \text{"the"}, Y_2 = \text{"cat"}) = 0.20$$
Joint sequence probability:
$$P(Y) = 0.10 \times 0.05 \times 0.20 = 0.0010 = 1.0 \times 10^{-3}$$
Sequence negative log-likelihood (base $e$):
$$-\ln P(Y) = -(\ln 0.10 + \ln 0.05 + \ln 0.20) = -(-2.3026 - 2.9957 - 1.6094) = 6.9078 \text{ nats}$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

p1 = 0.10
p2_given_1 = 0.05
p3_given_12 = 0.20

joint_p = p1 * p2_given_1 * p3_given_12
nll = -(np.log(p1) + np.log(p2_given_1) + np.log(p3_given_12))

assert abs(joint_p - 0.0010) < 1e-7
assert abs(nll - 6.907755) < 1e-5
print("[P5 PASS] Probability Chain Rule verified: joint_p =", joint_p, "nll =", nll)
```

### 🩺 Diagnostic Mini-Check
**Question:** How does the probability chain rule express $P(y_1, y_2, y_3)$ without any conditional independence assumptions?  
<details><summary><b>Reveal Answer</b></summary><b>$P(y_1) P(y_2 \mid y_1) P(y_3 \mid y_1, y_2)$</b>.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $Y = (Y_1, Y_2, \dots, Y_T)$ be a sequence of discrete random variables taking values in vocabulary $\mathcal{V}$, conditioned on context $X$.
By repeated application of Kolmogorov's definition of conditional probability:
$$P(Y_1, \dots, Y_T \mid X) = \prod_{t=1}^T P(Y_t \mid Y_1, \dots, Y_{t-1}, X) = \prod_{t=1}^T P(Y_t \mid Y_{<t}, X)$$
This exact identity forms the mathematical foundation of **auto-regressive generative modeling**: maximizing joint sequence log-likelihood is equivalent to minimizing cumulative cross-entropy loss step-by-step.
</details>

---

## 6. Block-Toeplitz Operators & Causal Lower-Triangular Matrices

<a id="p6"></a>

### 👶 Physical Analogy & Intuition
Imagine reading a murder mystery novel. Chapter 3 can be influenced by clues dropped in Chapter 1 and Chapter 2, but Chapter 3 cannot possibly be influenced by events that haven't happened yet in Chapter 10. Time flows in one direction: information only travels forward into the future, never backward into the past. In matrix algebra, this one-way flow forces all upper-triangular matrix connections to be zero—a strictly lower-triangular causal matrix.

### 🔍 Plain-English Breakdown
- When an unrolled linear recurrent network is viewed as a gigantic Multi-Layer Perceptron operating on the entire concatenated sequence vector $X \in \mathbb{R}^{T d}$, its transformation matrix $\mathcal{M} \in \mathbb{R}^{(T m) \times (T d)}$ has two structural properties:
  1. **Strictly Block Lower-Triangular (Causality):** Upper blocks are all zero ($\mathcal{M}_{i, j} = \mathbf{0}$ for $j > i$), ensuring future inputs cannot affect past states.
  2. **Block-Toeplitz (Temporal Parameter Sharing):** Along each sub-diagonal, blocks are identical powers $W_{hh}^{i-j} W_{xh}$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let $T = 3, m = 1, d = 1$ with scalar parameters $W_{hh} = 0.5, W_{xh} = 2.0$:
The block entries are:
- Main diagonal ($k=0$): $W_{xh} = 2.0$
- First subdiagonal ($k=1$): $W_{hh} W_{xh} = (0.5)(2.0) = 1.0$
- Second subdiagonal ($k=2$): $W_{hh}^2 W_{xh} = (0.25)(2.0) = 0.5$
$$\mathcal{M} = \begin{bmatrix} 2.0 & 0.0 & 0.0 \\ 1.0 & 2.0 & 0.0 \\ 0.5 & 1.0 & 2.0 \end{bmatrix}$$
For input $X = [1.0, 2.0, 3.0]^T$:
$$H = \mathcal{M} X = \begin{bmatrix} 2.0(1.0) \\ 1.0(1.0) + 2.0(2.0) \\ 0.5(1.0) + 1.0(2.0) + 2.0(3.0) \end{bmatrix} = \begin{bmatrix} 2.0 \\ 5.0 \\ 8.5 \end{bmatrix}$$
Sequential recurrence check:
- $h_1 = 2.0(1.0) = 2.0$
- $h_2 = 0.5(2.0) + 2.0(2.0) = 1.0 + 4.0 = 5.0$
- $h_3 = 0.5(5.0) + 2.0(3.0) = 2.5 + 6.0 = 8.5$. Matches exactly!

### 💻 Standalone Executable Python Verification
```python
import numpy as np

T = 3
w_hh = 0.5
w_xh = 2.0

M = np.array([
    [w_xh, 0.0, 0.0],
    [w_hh * w_xh, w_xh, 0.0],
    [(w_hh**2) * w_xh, w_hh * w_xh, w_xh]
])
x = np.array([1.0, 2.0, 3.0])
h_matrix = M @ x

# Sequential recurrence
h_seq = []
h_prev = 0.0
for xt in x:
    h_curr = w_hh * h_prev + w_xh * xt
    h_seq.append(h_curr)
    h_prev = h_curr

np.testing.assert_allclose(h_matrix, h_seq, rtol=1e-6)
assert np.allclose(h_matrix, np.array([2.0, 5.0, 8.5]))
print("[P6 PASS] Causal Block-Toeplitz Equivalence verified:", h_matrix)
```

### 🩺 Diagnostic Mini-Check
**Question:** Why does the unrolled RNN matrix $\mathcal{M}$ have zeros in its upper block triangle?  
<details><summary><b>Reveal Answer</b></summary><b>Temporal causality.</b> Future inputs $x_t$ cannot influence past states $h_\tau$ where $\tau < t$.</details>

<details>
<summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let input sequence be $X = [x_1^T, x_2^T, \dots, x_T^T]^T \in \mathbb{R}^{T d}$ and hidden state trajectory be $H = [h_1^T, h_2^T, \dots, h_T^T]^T \in \mathbb{R}^{T m}$.
For a linear recurrent system $h_t = W_{hh} h_{t-1} + W_{xh} x_t$ with $h_0 = \mathbf{0}$:
$$\begin{bmatrix} h_1 \\ h_2 \\ \vdots \\ h_T \end{bmatrix} = \begin{bmatrix}
W_{xh} & \mathbf{0} & \dots & \mathbf{0} \\
W_{hh} W_{xh} & W_{xh} & \dots & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots \\
W_{hh}^{T-1} W_{xh} & W_{hh}^{T-2} W_{xh} & \dots & W_{xh}
\end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_T \end{bmatrix}$$
The global matrix $\mathcal{M} \in \mathbb{R}^{(T m) \times (T d)}$ is:
1. **Causal:** $\mathcal{M}_{i, j} = \mathbf{0}$ for all $j > i$ (block lower-triangular).
2. **Block-Toeplitz:** Subdiagonal blocks $k = i - j$ depend only on the time lag $k$: $\mathcal{M}_{i, j} = W_{hh}^k W_{xh}$.
</details>
