# Formulae Sheet: Diffusion Models Part 1 (ELBO & Posterior)

## 1. Full Evidence Lower Bound (ELBO)
$$\log p_\theta(x_0) \ge \text{ELBO} = - \left( L_0 + L_T + \sum_{t=2}^T L_{t-1} \right)$$
- **Optimization Objective**: $\min_\theta (L_0 + \sum_{t=2}^T L_{t-1})$ (since $\nabla_\theta L_T = 0$).

## 2. Canonical Tripartite Terms
1. **Prior Matching Loss**:
   $$L_T = D_{\text{KL}}(q(x_T \mid x_0) \parallel p(x_T))$$
2. **Denoising Transition Matching Loss**:
   $$L_{t-1} = D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t))$$
3. **Reconstruction Loss**:
   $$L_0 = -\log p_\theta(x_0 \mid x_1)$$

## 3. Bayes Identity for Markov Transition
$$q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0) = q(x_{t-1} \mid x_t, x_0) \frac{q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$$

## 4. Tractable Posterior Distribution $q(x_{t-1} \mid x_t, x_0)$
$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$
- **Boundary Invariant**: At $t=1$, $\bar{\alpha}_0 \equiv 1.0 \implies \tilde{\beta}_1 = 0.0$ and $\tilde{\mu}_1(x_1, x_0) = x_0$.

## 5. Gaussian KL Divergence Reduction
$$D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t)) = \frac{1}{2\sigma_t^2} \|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2 + C$$
