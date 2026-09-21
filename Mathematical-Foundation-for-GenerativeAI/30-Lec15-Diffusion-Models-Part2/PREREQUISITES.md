> [!NOTE]
> **Warm-up & Orientation**: This package covers the reparameterization of DDPM from predicting means $\mu_\theta$ to predicting injected noise $\epsilon_\theta$, culminating in Ho et al.'s celebrated $L_{\text{simple}}$ objective and ancestral sampling.

# Prerequisites & Foundations: Diffusion Models Part 2 (Reparameterization & Sampling)

## Math Terminology Rosetta Stone

| Symbol / Notation | Explicit Meaning | Standard Units / Domain | Common Alternative Notations | Mental Anchor / Reading Translation |
| :--- | :--- | :--- | :--- | :--- |
| $\epsilon \sim \mathcal{N}(0, I)$ | Ground-truth Gaussian noise vector injected into $x_0$ | Dimensionless vector $\mathbb{R}^D$ | $z, \epsilon_t$ | "The exact random static added at step $t$" |
| $\epsilon_\theta(x_t, t)$ | Neural network predicting injected noise | $\mathbb{R}^D$ continuous | $\hat{\epsilon}, \text{score}_{\text{net}}$ | "The denoiser's guess of what noise was added" |
| $\mu_\theta(x_t, t)$ | Neural network predicting reverse transition mean | $\mathbb{R}^D$ continuous | $\hat{\mu}$ | "The reconstructed center of the previous state $x_{t-1}$" |
| $\sigma_t^2$ | Variance of the reverse transition $p_\theta(x_{t-1} \mid x_t)$ | Scalar $> 0$ | $\beta_t \text{ or } \tilde{\beta}_t$ | "The random exploration jitter added during reverse sampling" |
| $L_{\text{simple}}(\theta)$ | Simplified unweighted MSE loss | Dimensionless scalar $\ge 0$ | $\mathcal{L}_{\text{simple}}$ | "The clean MSE between injected noise and network prediction" |
| $z \sim \mathcal{N}(0, I)$ | Standard Gaussian sample drawn at each reverse step | Dimensionless vector $\mathbb{R}^D$ | $\xi, \epsilon_{\text{sample}}$ | "The stochastic kick that keeps generation dynamic" |
| $\bar{\alpha}_t$ | Cumulative signal retention product | Dimensionless $\in (0, 1)$ | $\alpha_t^{\text{cum}}$ | "Fraction of clean signal power remaining in $x_t$" |

---

## Foundational Pillar 1: Inverting the Forward Reparameterization

### Mathematical Derivation & Concept
From the forward jump kernel, we know:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Solving algebraically for $x_0$ in terms of $x_t$ and $\epsilon$:
$$\sqrt{\bar{\alpha}_t} x_0 = x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon$$
$$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right)$$

This elementary algebraic inversion allows us to eliminate the ground-truth variable $x_0$ entirely from our posterior formulas, replacing it with the injected noise vector $\epsilon$!

```
   Forward Direction:
   x_0 (Clean)  ----[ Scale by sqrt(alpha_bar) ]----+
                                                    |----> x_t (Noisy)
   eps (Noise)  ----[ Scale by sqrt(1-alpha_bar)]---+
   
   Inverted Direction:
   x_t (Noisy)  ----+
                    |----[ Subtract noise, divide by sqrt(alpha_bar) ]----> x_0 (Recovered)
   eps (Noise)  ----+
```

### Micro-Number Numerical Verification
Let $\bar{\alpha}_t = 0.25$, so $\sqrt{\bar{\alpha}_t} = 0.5$ and $\sqrt{1 - \bar{\alpha}_t} = \sqrt{0.75} \approx 0.86603$.
Suppose $x_0 = 4.0$ and $\epsilon = 1.0$.
Then $x_t = 0.5(4.0) + 0.86603(1.0) = 2.0 + 0.86603 = 2.86603$.
Now invert:
$$x_0 = \frac{1}{0.5} (2.86603 - 0.86603 \times 1.0) = 2.0 \times 2.0 = 4.0$$
The algebraic inversion is exact.

### Physical Analogy
If you dissolve 5 grams of salt into 100 ml of pure water, you know the final saline solution ($x_t$). If you measure that 3 grams of salt came from an external shaker ($\epsilon$), you can immediately deduce that the water originally contained exactly 2 grams of salt ($x_0$).

### Runnable Python Verification
```python
import torch

def verify_inversion():
    torch.manual_seed(42)
    x0 = torch.tensor([4.0])
    eps = torch.tensor([1.0])
    alpha_bar = torch.tensor([0.25])
    
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * eps
    recovered_x0 = (xt - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
    
    assert torch.allclose(x0, recovered_x0)
    print(f"Inversion verified: x0={x0.item():.4f}, recovered={recovered_x0.item():.4f}")

verify_inversion()
```

---

## Foundational Pillar 2: Substituting $x_0$ into the Posterior Mean $\tilde{\mu}_t$

### Mathematical Derivation & Concept
Recall the analytical posterior mean derived in Lecture 14:
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

Substitute $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right)$ into the first term:
$$\tilde{\mu}_t(x_t, \epsilon) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{(1 - \bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon) + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

Note that $\sqrt{\bar{\alpha}_t} = \sqrt{\alpha_t} \sqrt{\bar{\alpha}_{t-1}}$, so $\frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}} = \frac{1}{\sqrt{\alpha_t}}$.
Grouping the coefficients of $x_t$:
$$\text{Coef}(x_t) = \frac{\beta_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} = \frac{\beta_t + \alpha_t(1 - \bar{\alpha}_{t-1})}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{\beta_t + \alpha_t - \bar{\alpha}_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{1 - \bar{\alpha}_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{1}{\sqrt{\alpha_t}}$$
And the coefficient of $\epsilon$:
$$\text{Coef}(\epsilon) = -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}}$$

Combining both terms gives the iconic formula for the posterior mean:
$$\tilde{\mu}_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$$

### Micro-Number Numerical Verification
Let $\alpha_t = 0.81 \implies \sqrt{\alpha_t} = 0.9, \beta_t = 0.19$.
Let $\bar{\alpha}_t = 0.64 \implies \sqrt{1 - \bar{\alpha}_t} = \sqrt{0.36} = 0.6$.
Suppose $x_t = 3.0$ and $\epsilon = 0.5$.
$$\tilde{\mu}_t = \frac{1}{0.9} \left( 3.0 - \frac{0.19}{0.6} \times 0.5 \right) = \frac{1}{0.9} (3.0 - 0.31667 \times 0.5) = \frac{1}{0.9} (3.0 - 0.15833) = \frac{2.84167}{0.9} \approx 3.1574$$

### Physical Analogy
Rather than calculating where you came from by remembering your birth hospital, you measure your current location and subtract the odometer reading of the taxi ride you just took.

### Runnable Python Verification
```python
import torch

def verify_mu_reparameterization():
    beta_t = 0.1
    alpha_t = 0.9
    alpha_bar_prev = 0.8
    alpha_bar_t = alpha_t * alpha_bar_prev # 0.72
    
    x0 = torch.tensor([2.0])
    eps = torch.tensor([0.5])
    xt = torch.sqrt(torch.tensor(alpha_bar_t)) * x0 + torch.sqrt(torch.tensor(1.0 - alpha_bar_t)) * eps
    
    # Formula 1: using x0
    c1 = (torch.sqrt(torch.tensor(alpha_bar_prev)) * beta_t) / (1.0 - alpha_bar_t)
    c2 = (torch.sqrt(torch.tensor(alpha_t)) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
    mu_1 = c1 * x0 + c2 * xt
    
    # Formula 2: using eps
    mu_2 = (1.0 / torch.sqrt(torch.tensor(alpha_t))) * (xt - (beta_t / torch.sqrt(torch.tensor(1.0 - alpha_bar_t))) * eps)
    
    assert torch.allclose(mu_1, mu_2)
    print(f"Reparameterization exact: mu_1={mu_1.item():.4f}, mu_2={mu_2.item():.4f}")

verify_mu_reparameterization()
```

---

## Foundational Pillar 3: Parametrizing the Neural Denoiser as Noise Predictor $\epsilon_\theta$

### Mathematical Derivation & Concept
Since the true target mean is $\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$, we parameterize our model's mean $\mu_\theta(x_t, t)$ with the exact same functional form:
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$
where $\epsilon_\theta(x_t, t)$ is a deep neural network (typically a U-Net with time embeddings).

Substituting $\tilde{\mu}_t$ and $\mu_\theta$ into the Gaussian KL divergence $L_{t-1} = \frac{1}{2\sigma_t^2} \|\tilde{\mu}_t - \mu_\theta\|^2$:
$$L_{t-1} = \frac{1}{2\sigma_t^2} \left\| \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right) - \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) \right\|^2$$
$$= \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

The complicated Gaussian KL divergence between probability distributions reduces to an ordinary Mean Squared Error (MSE) between the true injected noise $\epsilon$ and the network's predicted noise $\epsilon_\theta(x_t, t)$!

### Micro-Number Numerical Verification
Let the scalar prefactor $\kappa_t = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} = 1.5$.
If injected noise $\epsilon = [1.0, -0.5]$ and predicted noise $\hat{\epsilon} = [0.9, -0.4]$:
$$\|\epsilon - \hat{\epsilon}\|^2 = (1.0 - 0.9)^2 + (-0.5 - (-0.4))^2 = 0.01 + 0.01 = 0.02$$
$$L_{t-1} = 1.5 \times 0.02 = 0.03$$

### Physical Analogy
Tuning a radio. You don't ask a machine to synthesize Mozart from static directly; you ask it to identify the high-frequency hiss of the radio receiver ($\epsilon$), and then you subtract that hiss to reveal Mozart cleanly.

### Runnable Python Verification
```python
import torch
import torch.nn as nn

def verify_loss_reduction():
    eps_true = torch.tensor([1.0, -0.5])
    eps_pred = torch.tensor([0.9, -0.4])
    mse = torch.sum((eps_true - eps_pred)**2)
    assert abs(mse.item() - 0.02) < 1e-5
    print(f"MSE noise prediction verified: {mse.item():.4f}")

verify_loss_reduction()
```

---

## Foundational Pillar 4: Ancestral Sampling with Langevin Jitter

### Mathematical Derivation & Concept
At test time (inference), we start from pure noise $x_T \sim \mathcal{N}(0, I)$ and iterate backwards for $t = T, T-1, \dots, 1$:
$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, I) \text{ if } t > 1, \text{ else } z = 0$$
Substituting our noise predictor parameterization:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

Why add $\sigma_t z$ at each step?
Without the stochastic kick $\sigma_t z$, sampling becomes a deterministic ODE trajectory. Adding Langevin noise $z$ allows the sample to explore multi-modal probability landscapes, escaping saddle points and resolving fine stochastic details (e.g. hair strands, water reflections).

### Micro-Number Numerical Verification
At final step $t=1$, $z = 0$.
If $\mu_\theta(x_1, 1) = [0.8, -0.2]$, then $x_0 = [0.8, -0.2]$ exactly, without adding noise.

### Physical Analogy
Annealing steel. If you cool the molten metal instantly with zero thermal jitter, crystalline defects freeze in place, making the metal brittle. Adding controlled thermal agitation ($\sigma_t z$) while cooling allows atoms to settle into a flawless crystalline lattice.

### Runnable Python Verification
```python
import torch

def verify_sampling_step():
    xt = torch.tensor([1.5])
    eps_pred = torch.tensor([0.2])
    alpha_t = 0.95
    beta_t = 0.05
    alpha_bar_t = 0.7
    sigma_t = 0.1
    z = torch.tensor([0.8])
    
    # Reverse step
    drift = (1.0 / (alpha_t**0.5)) * (xt - (beta_t / (1.0 - alpha_bar_t)**0.5) * eps_pred)
    xt_prev = drift + sigma_t * z
    assert xt_prev.shape == (1,)
    print(f"Sample step verified: drift={drift.item():.4f}, xt_prev={xt_prev.item():.4f}")

verify_sampling_step()
```

---

## Curriculum Bridges

| Concept | Upstream Prerequisite | Downstream Application in Lecture 15 |
| :--- | :--- | :--- |
| Mean Squared Error | [08-Loss_Functions.md](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/08-Loss_Functions.md) | Formulating $L_{\text{simple}}(\theta) = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$ |
| Gaussian Distributions | [02-Common_Probability_Distributions.md](../../MathsTerms/03-Probability-and-Statistical-Estimation/02-Common_Probability_Distributions.md) | Reverse transition kernel $p_\theta(x_{t-1} \mid x_t)$ |
| Score Matching | [04-Lipschitz_Continuity.md](../../MathsTerms/05-Convexity-Duality-and-Metric-Analysis/04-Lipschitz_Continuity.md) | Connection between $\epsilon_\theta(x_t, t)$ and score function $\nabla_{x_t} \log p(x_t)$ |

---

## Diagnostics Self-Assessment

1. **Why does predicting noise $\epsilon$ work significantly better than predicting $x_0$ directly?**  
   *Answer*: Predicting $\epsilon$ standardizes the network target to zero-mean, unit-variance Gaussian noise across all time steps $t \in [1, T]$, stabilizing gradients and preventing the network from predicting blurry averages at large $t$.

2. **Why did Ho et al. drop the analytical weight $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}$ in $L_{\text{simple}}$?**  
   *Answer*: The analytical weight assigns disproportionately high weight to small $t$ (imperceptible details) and downweights large $t$ (global semantic structure). Dropping it emphasizes perceptually salient structures, yielding superior sample quality.
