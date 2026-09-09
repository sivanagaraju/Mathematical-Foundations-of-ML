# ⚡ Rapid Revision Formulae Sheet: Information Entropy

Mathematical core references, dimensional signatures, invariant guarantees, and contrastive decision rubrics for **Lecture 11: Entropy**.

---

## 1. Equations Index

### Surprisal / Self-Information
$$I(A) = -\log_b P(A) = \log_b \frac{1}{P(A)}$$

$$\text{Base 2 (Bits): } I(A) = -\log_2 P(A), \quad \text{Base } e \text{ (Nats): } I(A) = -\ln P(A)$$

$$\text{Conversion: } 1 \text{ nat} = \frac{1}{\ln 2} \approx 1.442695 \text{ bits}$$

### Discrete Shannon Entropy
$$H(X) = \mathbb{E}_{X \sim p}[I(X)] = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)$$

$$\text{Limit Convention: } \lim_{p \to 0^+} p \log p = 0$$

### Continuous Differential Entropy
$$h(X) = -\int_{-\infty}^\infty f(x) \ln f(x)\,dx$$

$$\text{1D Gaussian: } X \sim \mathcal{N}(\mu, \sigma^2) \implies h(X) = \frac{1}{2} \ln\left(2\pi e \sigma^2\right)$$

$$\text{Critical Zero Threshold: } \sigma^* = \frac{1}{\sqrt{2\pi e}} \approx 0.24197 \implies h(X) = 0$$

### Joint & Conditional Entropy
$$H(X, Y) = -\sum_{x, y} p(x, y) \log_2 p(x, y)$$

$$H(Y \mid X) = \sum_x p(x) H(Y \mid X = x) = -\sum_{x, y} p(x, y) \log_2 p(y \mid x)$$

$$\text{Chain Rule: } H(X, Y) = H(X) + H(Y \mid X)$$

---

## 2. Tensor Dimensionality & Shape Signatures

| Mathematical Object | Symbolic Notation | Dimension / Tensor Shape | Operational Semantic |
| :--- | :--- | :--- | :--- |
| **Probability Simplex Vector** | $\mathbf{p}$ | `(K,)` | Valid discrete distribution satisfying $p_k \ge 0, \sum p_k = 1$ |
| **Surprisal Vector** | $\mathbf{I}(\mathbf{p})$ | `(K,)` | Element-wise information content $-\log_2(\mathbf{p})$ in bits |
| **Batch Logits** | $\mathbf{z}$ | `(B, K)` | Raw unnormalized model outputs passed into softmax |
| **Softmax Probabilities** | $\hat{\mathbf{p}}$ | `(B, K)` | Normalized predicted distribution along class dimension |
| **Batch Entropy Vector** | $\mathbf{H}(\hat{\mathbf{p}})$ | `(B,)` | Evaluated entropy per batch element for confidence tracking |

---

## 3. Guarantees & Invariants

1. **Non-negativity of Discrete Entropy:** $H(X) \ge 0$ strictly for any discrete random variable, with $H(X) = 0$ if and only if $X$ is deterministic ($P(X = x_0) = 1$).
2. **Uniform Upper Bound:** For a discrete support with $K$ distinct outcomes, $0 \le H(X) \le \log_2 K$. Maximum entropy is attained uniquely by the uniform distribution $p_i = 1/K$.
3. **Differential Entropy Can Be Negative:** Continuous differential entropy $h(X)$ is NOT bounded below by 0. A localized density with variance $\sigma^2 < \frac{1}{2\pi e}$ yields $h(X) < 0$.
4. **Independent Information Additivity:** If $X$ and $Y$ are independent random variables, $H(X, Y) = H(X) + H(Y)$ and $I(X \cap Y) = I(X) + I(Y)$.

---

## 4. Contrastive Decision Table: Discrete Shannon Entropy vs Continuous Differential Entropy

| Metric Property | Discrete Shannon Entropy $H(X)$ | Continuous Differential Entropy $h(X)$ | Mathematical Reason |
| :--- | :--- | :--- | :--- |
| **Range of Values** | $[0, \log_2 K]$ (strictly $\ge 0$) | $(-\infty, +\infty)$ (admits negative values) | Continuous density $f(x)$ can exceed 1, causing $\ln f(x) > 0$ and $-\ln f(x) < 0$ |
| **Coding Interpretation** | Minimum average code length (bits) | Relative coordinate entropy (not code length) | Discrete bins correspond to binary decisions; continuous points require infinite precision |
| **Coordinate Invariance** | Invariant under 1-to-1 relabeling | NOT invariant under coordinate scaling | Change of variables $Y = cX$ adds $\ln |c|$ to differential entropy |
| **Measure of Certainty** | $H(X) = 0 \iff$ exact deterministic outcome | $h(X) \to -\infty$ as distribution approaches Dirac delta | Infinite density spike corresponds to minus infinity differential entropy |

---

## 5. Numerical Stability & Traps

1. **Zero Probability in Logarithm:** Evaluating $\log(0)$ generates `NaN` or `-inf`. In code implementations, always mask zero entries `p > 0` or add an infinitesimal epsilon: `torch.clamp(p, min=1e-12)`.
2. **Logit-Space Evaluation:** Never compute entropy by first calculating `p = torch.softmax(z)` and then `p * torch.log(p)`. Compute directly from logits using log-softmax: `sum(p * log_softmax(z))` to prevent intermediate numerical underflow.
3. **Log Base Inconsistency:** Information theory literature alternates between base 2 (bits), base $e$ (nats), and base 10 (hartleys). Mixing bases without conversion factors ($\ln 2$) causes systematic calculation errors.
