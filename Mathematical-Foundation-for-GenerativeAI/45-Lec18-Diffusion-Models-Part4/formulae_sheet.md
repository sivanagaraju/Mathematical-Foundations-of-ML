# Formulae Sheet: Diffusion Models Part 4 (Score Theory & Latent Diffusion)

## 1. Tweedie's Classical Inversion Formula
$$\mathbb{E}[x_0 \mid x_t] = x_t + \sigma_t^2 \nabla_{x_t} \log p(x_t)$$
- For DDPM jump kernel $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$:
  $$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

## 2. Duality Between Score and Noise Prediction
$$s_\theta(x_t, t) = \nabla_{x_t} \log p(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

## 3. Classifier Guidance Formulation
$$\tilde{\nabla}_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + s \nabla_{x_t} \log p_\phi(y \mid x_t)$$
$$\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t) - s \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p_\phi(y \mid x_t)$$

## 4. Classifier-Free Guidance (CFG)
$$\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + s \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset) \right)$$

## 5. Latent Diffusion Models (LDM)
$$z_0 = \mathcal{E}(x_0) \cdot c_{\text{scale}}$$
$$L_{\text{LDM}} = \mathbb{E}_{z_0, \epsilon, t, y} \left[ \|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|^2 \right]$$
$$x_{\text{rec}} = \mathcal{D}\left(\frac{z_0}{c_{\text{scale}}}\right)$$

## 6. Continuous-Time SDE Formulations
- **Forward SDE**: $dx = f(x, t) dt + g(t) d\mathbf{w}$
- **Reverse SDE**: $dx = \left[ f(x, t) - g(t)^2 \nabla_x \log p_t(x) \right] dt + g(t) d\bar{\mathbf{w}}$
- **Probability Flow ODE**: $dx = \left[ f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x) \right] dt$
