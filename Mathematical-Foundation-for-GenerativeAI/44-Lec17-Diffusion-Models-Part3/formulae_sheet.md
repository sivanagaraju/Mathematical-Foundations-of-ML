# Formulae Sheet: Diffusion Models Part 3 (Architecture & Guidance)

## 1. Sinusoidal Time Positional Embeddings
$$\tau(t)_{2i} = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \tau(t)_{2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)$$
- **Output Shape**: $\tau(t) \in \mathbb{R}^{B \times d_{\text{emb}}}$

## 2. Adaptive Group Normalization (AdaGN)
$$\text{AdaGN}(h, t) = (1 + \gamma(t)) \odot \left( \frac{h - \mu(h)}{\sqrt{\sigma^2(h) + \epsilon}} \right) + \beta(t)$$
where $[\gamma(t), \beta(t)] = \text{Linear}(\text{SiLU}(\text{Linear}(\tau(t))))$.

## 3. Bayes' Conditional Score Decomposition
$$\nabla_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y \mid x_t)$$

## 4. Classifier Guidance Score
$$\tilde{\nabla}_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + s \nabla_{x_t} \log p_\phi(y \mid x_t)$$
- **Noise Representation**: $\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t) - s \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p_\phi(y \mid x_t)$.

## 5. Classifier-Free Guidance (CFG) Extrapolation
$$\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + s \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset) \right)$$
- **Hyperparameter**: $s \ge 1.0$, typically $s \in [3.0, 7.5]$.
- **Training Dropout Rate**: $p_{\text{uncond}} \in [0.10, 0.20]$.
