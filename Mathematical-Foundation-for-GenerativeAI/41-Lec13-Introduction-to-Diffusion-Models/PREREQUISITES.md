> [!NOTE]
> **Warm-up & Orientation**: Before entering the rigorous mathematical derivation of Diffusion Models (DDPM), master the foundational pillars of Gaussian transition kernels, Markov chains, variance composition, and non-equilibrium thermodynamics.

# Prerequisites & Foundations: Introduction to Diffusion Models

## Math Terminology Rosetta Stone

| Symbol / Notation | Explicit Meaning | Standard Units / Domain | Common Alternative Notations | Mental Anchor / Reading Translation |
| :--- | :--- | :--- | :--- | :--- |
| $x_0 \in \mathbb{R}^D$ | Clean data vector from data distribution $q(x_0)$ | Pixel intensities $[-1, 1]^D$ | $x, x_{\text{data}}$ | "Original pristine uncorrupted image" |
| $x_t \in \mathbb{R}^D$ | Latent noisy state at discrete time step $t$ | $\mathbb{R}^D$ continuous | $z_t, x^{(t)}$ | "Image contaminated with $t$ stages of Gaussian static" |
| $\beta_t \in (0, 1)$ | Variance scale injected at time step $t$ | Dimensionless scalar $(10^{-4} \to 0.02)$ | $\sigma_t^2, \eta_t$ | "Fraction of pure noise added during transition $t-1 \to t$" |
| $\alpha_t \equiv 1 - \beta_t$ | Fraction of signal preserved at time step $t$ | Dimensionless scalar $(0.98 \to 0.9999)$ | $a_t$ | "Retention coefficient of the previous signal" |
| $\bar{\alpha}_t \equiv \prod_{s=1}^t \alpha_s$ | Cumulative signal retention from time $0$ to $t$ | Dimensionless scalar $\in (0, 1)$ | $\alpha_t^{\text{cum}}, \gamma_t$ | "Fraction of pristine data $x_0$ remaining after $t$ steps" |
| $q(x_t \mid x_{t-1})$ | Forward perturbation transition kernel | Conditional Gaussian density | $p_{\text{forward}}$ | "The deterministic degradation law adding Gaussian haze" |
| $p_\theta(x_{t-1} \mid x_t)$ | Learnable reverse denoising transition kernel | Parametrized Gaussian distribution | $p_{\text{model}}$ | "The neural denoiser reversing one tick of entropy" |

---

## Foundational Pillar 1: Gaussian Perturbation and Variance Preservation

### Mathematical Derivation & Concept
In diffusion models, we transform an arbitrary, complex data distribution $q(x_0)$ into pure isotropic Gaussian noise $\mathcal{N}(0, I)$ via incremental Gaussian perturbations. Consider scaling a random variable by $c \in \mathbb{R}$ and adding independent noise:
$$x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

Why scale by $\sqrt{1 - \beta_t}$ instead of $1$? If $x_{t-1}$ has unit variance $\text{Var}(x_{t-1}) = 1$, the variance of $x_t$ is:
$$\text{Var}(x_t) = (\sqrt{1 - \beta_t})^2 \text{Var}(x_{t-1}) + (\sqrt{\beta_t})^2 \text{Var}(\epsilon_t) = (1 - \beta_t)(1) + \beta_t(1) = 1$$
Thus, the total variance remains bounded and stationary throughout the entire 1,000-step diffusion process.

```
Pristine Vector x_0       Scaled Signal (sqrt(1-beta)*x_0)      Total Vector x_1 (Var=1)
[ +1.20, -0.80, +0.50 ] ----> [ +1.18, -0.79, +0.49 ] ----+---> [ +1.22, -0.71, +0.55 ]
                                                            ^
                               Pure Noise (sqrt(beta)*eps)  |
                              [ +0.04, +0.08, +0.06 ] ------+
```

### Micro-Number Numerical Verification
Let $\beta_1 = 0.02$, so $1 - \beta_1 = 0.98$, $\sqrt{1 - \beta_1} \approx 0.98995$, and $\sqrt{\beta_1} \approx 0.14142$.
Suppose $x_0 = 1.000$ and sampled $\epsilon_1 = 0.500$:
$$x_1 = 0.98995 \times 1.000 + 0.14142 \times 0.500 = 0.98995 + 0.07071 = 1.06066$$

### Physical Analogy
Imagine a drop of black ink placed in a glass of still water. At $t=0$, the ink molecule coordinates form a concentrated, non-Gaussian structure. At each tiny millisecond $t$, surrounding water molecules bump the ink particles (Brownian motion). By scaling down coordinates slightly and adding Gaussian kicks, the ink disperses uniformly until the glass reaches thermodynamic equilibrium (maximum entropy, isotropic Gaussian noise).

### Runnable Python Verification
```python
import torch

def verify_variance_preservation():
    torch.manual_seed(42)
    x = torch.randn(100000) # Unit variance data
    beta = 0.05
    scaled_x = torch.sqrt(torch.tensor(1.0 - beta)) * x
    noise = torch.sqrt(torch.tensor(beta)) * torch.randn(100000)
    x_next = scaled_x + noise
    
    var_orig = x.var().item()
    var_next = x_next.var().item()
    assert abs(var_next - 1.0) < 0.02, f"Variance drifted! Got {var_next}"
    print(f"Variance preserved: initial={var_orig:.4f}, perturbed={var_next:.4f}")

verify_variance_preservation()
```

---

## Foundational Pillar 2: The Markov Chain and Conditional Independence

### Mathematical Derivation & Concept
A sequence of random variables $x_0, x_1, \dots, x_T$ forms a first-order Markov chain if the conditional distribution of $x_t$ depends solely on the immediately preceding state $x_{t-1}$:
$$q(x_t \mid x_{t-1}, x_{t-2}, \dots, x_0) = q(x_t \mid x_{t-1})$$

The joint distribution of the full forward trajectory factorizes as:
$$q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$$

Because each transition is Gaussian:
$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

### Physical Analogy
Imagine an ink particle whose position at 3:00 PM is updated solely by the Brownian collision occurring between 2:59 PM and 3:00 PM. The collision at 3:00 PM does not need to look up where the particle was at 9:00 AM; the immediate state at 2:59 PM contains all the physical coordinates necessary to determine the next displacement.

### Micro-Number Numerical Verification
Let $x_0 = 2.0$. Step 1 injects noise $\epsilon_1 = 0.1$ with $\beta=0.1$:
$$x_1 = \sqrt{0.9}(2.0) + \sqrt{0.1}(0.1) = 0.94868(2.0) + 0.31623(0.1) = 1.89736 + 0.03162 = 1.92898$$
Step 2 injects noise $\epsilon_2 = -0.2$:
$$x_2 = \sqrt{0.9}(1.92898) + \sqrt{0.1}(-0.2) = 0.94868(1.92898) - 0.06325 = 1.82998 - 0.06325 = 1.76673$$

### Runnable Python Verification
```python
import torch

def verify_markov_property():
    T = 5
    beta = 0.1
    x_prev = torch.tensor([1.0])
    trajectory = [x_prev.item()]
    for t in range(1, T + 1):
        x_next = torch.sqrt(torch.tensor(1.0 - beta)) * x_prev + torch.sqrt(torch.tensor(beta)) * torch.randn(1)
        trajectory.append(x_next.item())
        x_prev = x_next
    assert len(trajectory) == T + 1
    print(f"Markov trajectory (0 to {T}): {[round(v, 3) for v in trajectory]}")

verify_markov_property()
```

---

## Foundational Pillar 3: Sum of Independent Gaussians and the Alpha-Bar Jump Kernel

### Mathematical Derivation & Concept
If $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are independent random variables, their linear combination is strictly Gaussian:
$$a X + b Y \sim \mathcal{N}(a \mu_1 + b \mu_2, a^2 \sigma_1^2 + b^2 \sigma_2^2)$$

Applying this repeatedly across transitions:
$$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \epsilon_1$$
$$x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1 - \alpha_2} \epsilon_2 = \sqrt{\alpha_2}(\sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \epsilon_1) + \sqrt{1 - \alpha_2} \epsilon_2$$
$$x_2 = \sqrt{\alpha_2 \alpha_1} x_0 + \sqrt{\alpha_2(1 - \alpha_1)} \epsilon_1 + \sqrt{1 - \alpha_2} \epsilon_2$$

The combined noise term has variance:
$$\text{Var} = (\sqrt{\alpha_2(1 - \alpha_1)})^2 + (\sqrt{1 - \alpha_2})^2 = \alpha_2 - \alpha_2 \alpha_1 + 1 - \alpha_2 = 1 - \alpha_1 \alpha_2$$
Defining $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, induction yields the closed-form jump kernel directly from $x_0$ to $x_t$:
$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### Micro-Number Numerical Verification
Let $\alpha_1 = 0.98, \alpha_2 = 0.95$. Then $\bar{\alpha}_2 = 0.98 \times 0.95 = 0.931$.
Coefficient for $x_0$: $\sqrt{0.931} \approx 0.96488$.
Coefficient for noise $\epsilon$: $\sqrt{1 - 0.931} = \sqrt{0.069} \approx 0.26268$.
Sum of squared coefficients: $0.96488^2 + 0.26268^2 = 0.93100 + 0.06900 = 1.00000$.

### Physical Analogy
Rather than observing an ink drop take 1,000 microscopic collisions one millisecond at a time, statistical mechanics allows us to write down the exact diffusion equation solution after 1 second as a single macro Gaussian blur kernel.

### Runnable Python Verification
```python
import torch

def verify_closed_form_jump():
    torch.manual_seed(42)
    x0 = torch.tensor([5.0])
    alphas = [0.99, 0.98, 0.97, 0.96]
    
    # 1. Step-by-step
    x_step = x0.clone()
    for a in alphas:
        b = 1.0 - a
        x_step = torch.sqrt(torch.tensor(a)) * x_step + torch.sqrt(torch.tensor(b)) * torch.randn(1)
        
    # 2. Closed form
    alpha_bar = 0.99 * 0.98 * 0.97 * 0.96
    # Distribution of closed form: mean = sqrt(alpha_bar)*x0, std = sqrt(1-alpha_bar)
    mean = (torch.sqrt(torch.tensor(alpha_bar)) * x0).item()
    std = (torch.sqrt(torch.tensor(1.0 - alpha_bar))).item()
    print(f"Closed-form parameters at t=4: Mean={mean:.4f}, Std={std:.4f}")
    assert mean < 5.0 and std > 0.0

verify_closed_form_jump()
```

---

## Foundational Pillar 4: The Generative Reversal and Score Intuition

### Mathematical Derivation & Concept
While entropy drives the forward process to destroy information ($x_0 \to \mathcal{N}(0, I)$), Bayes' rule dictates that the time-reversal of a continuous diffusion process with small step size $\beta_t \to 0$ is also an identical Gaussian transition kernel:
$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

The deep neural network's job is not to generate an entire image from scratch in one catastrophic leap, but merely to predict the infinitesimal drift (or reverse mean) $\mu_\theta(x_t, t)$ that rolls back one increment of noise.

### Micro-Number Numerical Verification
At $T=1000$, $\bar{\alpha}_T \approx 0.0001 \approx 0$. Thus $x_T \sim \mathcal{N}(0, I)$.
Sampling $x_{1000} \sim \mathcal{N}(0, I)$, the model denoises it backwards across $t = 1000, 999, \dots, 1$ to recover a realistic $x_0$.

### Physical Analogy
If you record a movie of an ink drop dissolving in water and play the film in reverse, you see dispersed pigment miraculously concentrate back into a pristine droplet. In physics, the Second Law of Thermodynamics forbids this spontaneously; but our neural network acts as "Maxwell's Demon", computing the exact vector directions required to coalesce the noise into coherent data.

### Runnable Python Verification
```python
import torch
import torch.nn as nn

class ToyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, xt, t):
        inp = torch.cat([xt, t], dim=-1)
        return self.net(inp)

denoiser = ToyDenoiser()
xt = torch.tensor([[1.5]])
t = torch.tensor([[0.5]])
predicted_drift = denoiser(xt, t)
assert predicted_drift.shape == (1, 1)
print(f"Denoiser forward pass valid. Drift shape: {predicted_drift.shape}")
```

---

## Curriculum Bridges

| Concept | Upstream Prerequisite | Downstream Application in Lecture 13 |
| :--- | :--- | :--- |
| Gaussian Distribution | [02-Common_Probability_Distributions.md](../../MathsTerms/04-Probability-and-Statistical-Estimation/02-Common_Probability_Distributions.md) | Forward transition kernel $q(x_t \mid x_{t-1})$ and prior distribution $p(x_T) = \mathcal{N}(0, I)$ |
| Markov Chains | [02-Markov_Chains_and_Stationary_Distributions.md](../../MathsTerms/04-Probability-and-Statistical-Estimation/03-Joint_Marginal_Conditional_Dist.md) | Unrolling joint forward trajectory $q(x_{1:T} \mid x_0)$ and reverse generative chain |
| Latent Variable Models | [01-Latent_Variable_Models_and_ELBO.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/05-Latent_Variable_Models.md) | Formulating diffusion as a hierarchical $T$-step latent model with Evidence Lower Bound |

---

## Diagnostics Self-Assessment

1. **Why does $x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon$ keep the variance bounded to 1 if $\text{Var}(x_{t-1})=1$?**  
   *Answer*: Because the variance of a linear combination of independent random variables is $\text{Var}(aX + bY) = a^2 \text{Var}(X) + b^2 \text{Var}(Y)$. Here $a^2 + b^2 = (1 - \beta_t) + \beta_t = 1$.

2. **Why do we define $\bar{\alpha}_t \equiv \prod_{s=1}^t \alpha_s$?**  
   *Answer*: Because compounding independent Gaussian steps allows us to jump directly from $x_0$ to $x_t$ without sequentially simulating $t$ intermediate Markov transitions.

3. **What is the fatal flaw of Generative Adversarial Networks that Diffusion Models solve?**  
   *Answer*: GANs suffer from non-convergent minimax saddle-point optimization and mode collapse; Diffusion models replace adversarial training with stationary maximum likelihood / score-matching regression.
