# Formulae Sheet: Diffusion Models Part 2 (Reparameterization & Sampling)

## 1. Jump Kernel Inversion
$$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right)$$

## 2. Posterior Mean Parameterization
$$\tilde{\mu}_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$$
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

## 3. Gaussian KL to Noise MSE Equivalence
$$D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t)) = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

## 4. Ho et al. Simplified Training Loss ($L_{\text{simple}}$)
$$L_{\text{simple}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(1, T), x_0, \epsilon \sim \mathcal{N}(0, I)} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$
where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.

## 5. Ancestral Reverse Sampling Algorithm
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$
where $z \sim \mathcal{N}(0, I)$ for $t > 1$, and $z = 0$ for $t = 1$.
$$\sigma_t = \sqrt{\tilde{\beta}_t} = \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t} \quad (\text{or } \sigma_t = \sqrt{\beta_t})$$
