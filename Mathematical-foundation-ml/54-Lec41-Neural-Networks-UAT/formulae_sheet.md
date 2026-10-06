# Formulae Sheet — Lec 41: Neural Networks and Universal Approximation Theorem

A high-density rapid revision sheet covering master equations, tensor dimensions, mathematical invariants, contrastive choices, and numerical stability considerations for multi-layer perceptrons and universal approximation.

---

## 1. Master Equations Index

### The Single-Layer Perceptron (Rosenblatt 1958)
$$\hat{y} = \mathrm{sign}\left(w^T x + b\right) = \mathrm{sign}\left(\sum_{j=1}^d w_j x_j + b\right)$$

### Perceptron Learning Algorithm (PLA) Weight Update
$$w_{t+1} = w_t + y_t x_t \quad \text{if } y_t (w_t^T x_t) \le 0$$

### Multi-Layer Perceptron (MLP) Forward Pass
$$z^{[1]} = W^{[1]} x + b^{[1]}, \quad a^{[1]} = \sigma\left(z^{[1]}\right)$$
$$z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}, \quad a^{[2]} = \sigma\left(z^{[2]}\right)$$
$$h_\theta(x) = W^{[L]} a^{[L-1]} + b^{[L]}$$

### Parameter Counting Formula
$$P = \sum_{l=1}^L \left( \dim(a^{[l]}) \times \dim(a^{[l-1]}) + \dim(a^{[l]}) \right)$$

### The Universal Approximation Theorem (Cybenko 1989, Hornik 1991)
$$\sup_{x \in X} \left| f(x) - \sum_{i=1}^N \beta_i \sigma\left(w_i^T x + b_i\right) \right| < \epsilon$$
$$\text{where } X \subset \mathbb{R}^d \text{ is compact, } f \in C(X), \text{ and } \epsilon > 0.$$

### Multi-Class Output Layer (Softmax)
$$P(Y = k \mid x) = \frac{e^{z_k^{[L]}}}{\sum_{j=1}^K e^{z_j^{[L]}}}$$

---

## 2. Input/Output Tensor Dimensionality Table

| Tensor Variable | Mathematical Description | Batch Tensor Shape | Dimension Semantics |
|:----------------|:-------------------------|:-------------------|:--------------------|
| `X` | Input Data Batch | `[B, D]` | $B$: batch size, $D$: input feature dimensionality |
| `W1` | Layer 1 Weight Matrix | `[L1, D]` | $L1$: hidden layer 1 width, $D$: input dimensions |
| `b1` | Layer 1 Bias Vector | `[L1]` | Broadcasted across batch $B$ |
| `Z1` | Layer 1 Pre-activation | `[B, L1]` | Linear combination $X W_1^T + b_1$ |
| `A1` | Layer 1 Post-activation | `[B, L1]` | Element-wise non-linear activations $\sigma(Z_1)$ |
| `W2` | Layer 2 Weight Matrix | `[L2, L1]` | $L2$: hidden layer 2 width, $L1$: incoming width |
| `Z2` | Layer 2 Pre-activation | `[B, L2]` | Intermediate feature representations |
| `W_out` | Output Weight Matrix | `[K, L2]` | $K$: output classes (classification) or target dimensions |
| `logits` | Raw Output Pre-activations | `[B, K]` | Unnormalized class scores before Softmax |
| `probs` | Normalized Class Probabilities | `[B, K]` | Probability vectors lying on standard simplex $\Delta^{K-1}$ |

---

## 3. Mathematical Guarantees & Invariants Table

| Construct / Law | Mathematical Guarantee | Conditions Required | Failure Mode if Violated |
|:----------------|:-----------------------|:--------------------|:-------------------------|
| **Novikoff's Theorem (PLA)** | Converges in $k \le (R / \gamma)^2$ mistake steps | Linear separability with geometric margin $\gamma > 0$ and radius $\|x\| \le R$ | Cycles infinitely on non-separable data (e.g. XOR) |
| **Linear Collapse Invariant** | Stacking linear layers yields a linear map: $\prod_{l=1}^L W^{[l]} = W_{\text{eff}}$ | Activation is identity $\sigma(z) = z$ | Zero representational capacity beyond linear hyperplanes |
| **UAT Existence Guarantee** | $\exists N < \infty$ achieving uniform error $\sup \|f - h\|_\infty < \epsilon$ | Compact domain $K \subset \mathbb{R}^d$, continuous $f \in C(K)$, non-polynomial $\sigma$ | Diverges on unbounded domains $\mathbb{R}^d$ or polynomial activations |
| **Softmax Simplex Invariant** | $\sum_{k=1}^K p_k = 1$ and $p_k \in (0, 1)$ | Finite real logits $z_k \in \mathbb{R}$ | Floating-point overflow under large positive logits |
| **Sigmoid Output Bound** | $a_i \in (0, 1)$ strictly bounded | Monotonic squashing $\lim_{t \to \infty} \sigma(t) = 1$ | Vanishing gradient when $|z| \gg 0$ due to $\sigma'(z) \to 0$ |

---

## 4. Contrastive "Why X, Not Y" Quick Decision Table

| Architecture / Technique (X) | Naive / Rejected Alternative (Y) | Why X Wins in Modern Deep Learning | Failure Mode of Y |
|:-----------------------------|:---------------------------------|:-----------------------------------|:------------------|
| **Deep & Narrow MLP** | Shallow & Ultra-Wide MLP | Compositional reuse; parameter efficiency $\mathcal{O}(L d^2)$ vs $\mathcal{O}(2^d)$ | Parameter explosion, out-of-memory crashes, inability to generalize |
| **Continuous Non-linear $\sigma(z)$** | Step / Threshold $\mathrm{sign}(z)$ | Smooth differentiable gradients allow gradient descent & backprop | Step function has derivative $\sigma'(z) = 0$ everywhere except at zero, breaking gradient descent |
| **Non-linear Activation** | Linear Layers $\sigma(z) = z$ | Curving decision boundaries to separate complex manifolds | Stacking collapses into a single affine matrix: $W_3 W_2 W_1 x = W_{\text{eff}} x$ |
| **Element-wise Activation** | Polynomial Activation $z^p$ | Prevents explosive numerical scaling and maintains UAT density | High-degree polynomials oscillate wildly outside training points (Runge phenomenon) |
| **Vectorized Layer Matrices** | Individual Scalar Loops | GPU acceleration via parallel BLAS/GEMM tensor operations | CPU loop overhead makes training modern models millions of times slower |

---

## 5. Hardware Realities & Numerical Stability

### Sigmoid Gradient Saturation & Vanishing Gradients
The derivative of the logistic sigmoid is:
$$\sigma'(z) = \sigma(z)\left(1 - \sigma(z)\right)$$
- When pre-activation $|z| > 5.0$, $\sigma(z) \approx 0.0$ or $1.0$, causing $\sigma'(z) < 0.0067$.
- In a deep network, multiplying many sub-unity derivatives cascades exponentially:
  $$\frac{\partial \mathcal{L}}{\partial W^{[1]}} \sim \prod_{l=1}^L \sigma'\left(z^{[l]}\right) \to 0$$
- **Fix:** Use non-saturating activations (ReLU, GELU) or batch normalization to keep pre-activations centered near zero.

### Softmax Numerical Overflow & LogSumExp Stabilization
Computing $e^{z_k}$ directly on raw float32 logits overflows to `+inf` when $z_k > 88.7$.
- **Unstable Naive Softmax:**
  $$\mathrm{Softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$
- **Hardware-Stable LogSumExp Trick:**
  Subtract the maximum logit $m = \max_{j} z_j$ before exponentiating:
  $$\mathrm{Softmax}(z)_k = \frac{e^{z_k - m}}{\sum_{j=1}^K e^{z_j - m}}$$
  Since $z_k - m \le 0$, all exponentiated terms are strictly bounded in $(0, 1]$, completely preventing floating-point overflow.
