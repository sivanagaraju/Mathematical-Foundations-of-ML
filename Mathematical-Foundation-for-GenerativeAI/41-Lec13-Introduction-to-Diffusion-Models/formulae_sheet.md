# Formulae Sheet: Introduction to Diffusion Models

## 1. Forward Diffusion Process (Discrete Markov Chain)
$$q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$$
$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$
- **Tensor Shape**: $x_t \in \mathbb{R}^{B \times C \times H \times W}$
- **Hyperparameters**: $\beta_1 < \beta_2 < \dots < \beta_T$, typically $\beta_1 = 10^{-4}, \beta_T = 0.02$.

## 2. Definitions of $\alpha$ and $\bar{\alpha}$
$$\alpha_t \equiv 1 - \beta_t$$
$$\bar{\alpha}_t \equiv \prod_{s=1}^t \alpha_s = \prod_{s=1}^t (1 - \beta_s)$$

## 3. Closed-Form Arbitrary Step Jump Kernel
$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
- **Computational Complexity**: $\mathcal{O}(1)$ time complexity to sample $x_t$ directly from $x_0$.

## 4. Reverse Generative Process
$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)$$
$$p(x_T) = \mathcal{N}(x_T; 0, I)$$
$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

## 5. Noise Schedule Formulations
- **Linear Schedule**:
  $$\beta_t = \beta_1 + \frac{t-1}{T-1} (\beta_T - \beta_1)$$
- **Cosine Schedule**:
  $$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$
