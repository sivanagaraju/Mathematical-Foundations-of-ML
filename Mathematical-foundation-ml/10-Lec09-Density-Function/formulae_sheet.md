# ⚡ Rapid Revision Formulae Sheet: Density Functions

Mathematical core references, dimensional signatures, invariant guarantees, and contrastive decision rubrics for **Lecture 09: Density Function**.

---

## 1. Equations Index

### Cumulative Distribution & Fundamental Density Definition
$$F_X(x) = \mathbb{P}(X \le x) = \int_{-\infty}^x f_X(t)\,dt$$

$$f_X(x) = \frac{d}{dx} F_X(x) \quad \text{(almost everywhere)}$$

### Core Continuous Axioms
$$\forall x \in \mathbb{R}, \quad f_X(x) \ge 0$$

$$\int_{-\infty}^\infty f_X(x)\,dx = 1$$

### Interval & Singleton Probability
$$\mathbb{P}(a \le X \le b) = F_X(b) - F_X(a) = \int_a^b f_X(x)\,dx$$

$$\mathbb{P}(X = c) = \lim_{\epsilon \to 0} \int_{c - \epsilon}^{c + \epsilon} f_X(t)\,dt = 0 \quad (\forall c \in \mathbb{R})$$

### Multivariate Joint & Marginal Densities
$$\int_{\mathbb{R}^d} f_{\mathbf{X}}(\mathbf{x})\,d\mathbf{x} = \int_{-\infty}^\infty \cdots \int_{-\infty}^\infty f_{X_1, \dots, X_d}(x_1, \dots, x_d)\,dx_1 \cdots dx_d = 1$$

$$f_{X_1}(x_1) = \int_{-\infty}^\infty f_{X_1, X_2}(x_1, x_2)\,dx_2$$

$$\mathbb{P}(\mathbf{X} \in \mathcal{A}) = \int_{\mathcal{A}} f_{\mathbf{X}}(\mathbf{x})\,d\mathbf{x}$$

---

## 2. Tensor Dimensionality & Shape Signatures

| Mathematical Object | Symbolic Notation | Dimension / Tensor Shape | Operational Semantic |
| :--- | :--- | :--- | :--- |
| **Continuous Sample Batch** | $\mathbf{X}$ | `(N, d)` | $N$ i.i.d. observations drawn in continuous $\mathbb{R}^d$ feature space |
| **Joint Density Evaluation** | $f_{\mathbf{X}}(\mathbf{X})$ | `(N,)` | Evaluated density height at each batch sample coordinate |
| **Coordinate Grid** | $\mathbf{G}$ | `(K, K, ..., K)` | Discretized coordinate mesh for numerical Riemann volume integration |
| **Marginal Density Vector** | $\mathbf{f}_{X_1}$ | `(K,)` | Extracted 1D slice along dimension 1 after summing out dimension 2 |
| **Log-Density Batch** | $\log f_{\mathbf{X}}(\mathbf{X})$ | `(N,)` | Coordinate-wise log-density values summed in continuous likelihood |

---

## 3. Guarantees & Invariants

1. **Non-negativity without Upper Bound:** $f_X(x) \ge 0$ for all $x$, but $f_X(x)$ has NO upper bound constraint. If support length $L < 1$, the density height $f(x) = 1/L > 1$.
2. **Total Area / Volume Normalization:** $\int_{\mathbb{R}^d} f(\mathbf{x}) d\mathbf{x} \equiv 1.0$. Area under the curve is probability mass; height at a point is not.
3. **Nullity of Singletons:** For continuous random variables, singleton sets $\{x\}$ have Lebesgue measure zero, meaning $\mathbb{P}(X = x) \equiv 0$ identically for every real number $x$.
4. **Marginalization Invariance:** Integrating the joint density over all possible values of an unobserved coordinate yields the exact marginal density of the observed coordinates.

---

## 4. Contrastive Decision Table: Why Densities, Not Mass Functions

| Criterion | Discrete Probability Mass Function (PMF) | Continuous Probability Density Function (PDF) | Operational Why / Tradeoff |
| :--- | :--- | :--- | :--- |
| **Upper Bound** | $0 \le p(x) \le 1$ strictly | $0 \le f(x) < \infty$ (no upper bound) | Discrete points hold finite mass; continuous points hold zero mass with finite density rate |
| **Evaluation at Point** | $p(c) = \mathbb{P}(X = c)$ | $f(c) \neq \mathbb{P}(X = c) \equiv 0$ | Point evaluation cannot be interpreted as event probability |
| **Summation vs Integration** | $\sum_{x \in \mathcal{X}} p(x) = 1$ | $\int_{-\infty}^\infty f(x) dx = 1$ | Continuous spaces are uncountable; summation is replaced by Riemann/Lebesgue integration |
| **Measurement Units** | Dimensionless probability $[0, 1]$ | Inverse units of the variable $(1 / \text{unit of } x)$ | Density multiplied by width $dx$ restores dimensionless probability mass |
| **Dimensional Curse in ML** | Enumerating states $K^d$ scales exponentially | High-dimensional volumes scale exponentially; densities become exponentially sharp | Estimating high-dimensional PDFs requires parametric assumptions (Gaussians, normalizing flows) |

---

## 5. Hardware Realities & Numerical Stability

1. **Underflow in Continuous Density Evaluation:** In continuous spaces, direct product of densities $p(\mathbf{x}) = \prod_{i=1}^d f(x_i)$ underflows to zero in IEEE 754 float32 when $d > 50$ (values $< 10^{-45}$). Always accumulate in the log-domain:
   $$\log p(\mathbf{x}) = \sum_{i=1}^d \log f(x_i)$$
2. **Epsilon Clamping on Support Boundaries:** When evaluating densities with bounded support or log-likelihoods, clamp probabilities away from zero to prevent $-\infty$:
   ```python
   # Tensor Shape: [B]
   clamped_density = torch.clamp(density, min=1e-12)
   log_density = torch.log(clamped_density)
   ```
3. **Riemann Mesh Discretization Error:** Numerical integration via grid points incurs truncation errors $\mathcal{O}(\Delta x)$ for rectangle rules and $\mathcal{O}(\Delta x^2)$ for trapezoidal rules (`np.trapezoid` / `torch.trapezoid`). Choose step sizes $\Delta x \le \sigma / 10$ to accurately capture sharp Gaussian peaks.
4. **Non-Invariance under Reparameterization:** Mode/argmax of continuous density changes under non-linear coordinate changes $y = g(x)$ due to the Jacobian determinant $|\frac{dx}{dy}|$. Do not interpret mode peaks as coordinate-free invariant representations.
