> [!NOTE]
> **Warm-up & Orientation**: This package delves into the rigorous derivation of the Evidence Lower Bound (ELBO) for Diffusion Models. Ensure you are familiar with Jensen's inequality, KL divergence between multivariate Gaussians, and Bayes' conditioning rules.

# Prerequisites & Foundations: Diffusion Models Part 1 (ELBO Derivation)

## Math Terminology Rosetta Stone

| Symbol / Notation | Explicit Meaning | Standard Units / Domain | Common Alternative Notations | Mental Anchor / Reading Translation |
| :--- | :--- | :--- | :--- | :--- |
| $\log p_\theta(x_0)$ | Evidence / marginal data log-likelihood | Nats / dimensionless log-prob | $\log p(x)$ | "The total log probability of observing pristine data" |
| $\text{ELBO}$ | Evidence Lower Bound | Nats ($\le \log p_\theta(x_0)$) | $\mathcal{L}_{\text{VLB}}$ | "The tractable surrogate objective we actually maximize" |
| $D_{\text{KL}}(q \parallel p)$ | Kullback-Leibler Divergence | Nats ($\ge 0$) | $D(q \parallel p), \text{KL}(q, p)$ | "The relative entropy penalty between posteriors" |
| $q(x_{t-1} \mid x_t, x_0)$ | Tractable forward posterior conditioned on clean data | Conditional Gaussian | $\tilde{q}_t$ | "Where $x_{t-1}$ came from, given $x_t$ and the known answer $x_0$" |
| $\tilde{\mu}_t(x_t, x_0)$ | Mean of the tractable posterior $q(x_{t-1} \mid x_t, x_0)$ | $\mathbb{R}^D$ | $\mu_{\text{post}}$ | "The exact analytical centroid targeting the previous step" |
| $\tilde{\beta}_t$ | Variance of the tractable posterior $q(x_{t-1} \mid x_t, x_0)$ | Scalar $> 0$ | $\sigma_q^2(t)$ | "The residual uncertainty when unstepping noise with known $x_0$" |
| $L_{t-1}$ | Denoising transition matching term at step $t$ | Nats | $\mathcal{L}_{t-1}$ | "The KL divergence penalty between true and modeled reverse step" |

---

## Foundational Pillar 1: Jensen's Inequality and Hierarchical ELBO

### Mathematical Derivation & Concept
Given a latent variable model with latent trajectory $x_{1:T}$ and observable data $x_0$, the true log-likelihood is intractable because it requires integrating over all possible continuous noise trajectories:
$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T}) dx_{1:T}$$

Introducing the variational forward distribution $q(x_{1:T} \mid x_0)$ and applying Jensen's inequality to the concave logarithm function:
$$\log p_\theta(x_0) = \log \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right] \ge \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right] \equiv \text{ELBO}$$

```
                JENSEN'S INEQUALITY ON TRAJECTORY SPACE
                
   Log-Likelihood log p(x_0)  ============================== (Intractable)
                                         ^
                                         | Gap = D_KL( q(x_{1:T}|x_0) || p_theta(x_{1:T}|x_0) )
                                         v
   ELBO                       ------------------------------ (Tractable Objective)
```

### Micro-Number Numerical Verification
Suppose $\frac{p_\theta}{q}$ takes values $0.20$ and $0.80$ with equal probability $0.5$.
$$\mathbb{E}\left[\frac{p}{q}\right] = 0.5(0.20) + 0.5(0.80) = 0.50 \implies \log(0.50) = -0.6931$$
$$\mathbb{E}\left[\log \frac{p}{q}\right] = 0.5 \log(0.20) + 0.5 \log(0.80) = 0.5(-1.6094) + 0.5(-0.2231) = -0.9163$$
Notice $-0.6931 \ge -0.9163$. The lower bound strictly holds.

### Physical Analogy
Imagine measuring the average height of trees in a foggy forest. You cannot see the ground directly (intractable marginal). Instead, you drop 1,000 plumb lines through the canopy. Calculating the lower bound of each line guarantees you never overestimate the actual average canopy height.

### Runnable Python Verification
```python
import torch

def verify_jensen():
    torch.manual_seed(42)
    # Ratios p / q
    ratios = torch.tensor([0.2, 0.8])
    log_mean = torch.log(ratios.mean())
    mean_log = torch.log(ratios).mean()
    assert log_mean >= mean_log
    print(f"Jensen verified: log(E[X])={log_mean.item():.4f} >= E[log(X)]={mean_log.item():.4f}")

verify_jensen()
```

---

## Foundational Pillar 2: Bayes' Rule on Markov Transitions Conditioned on $x_0$

### Mathematical Derivation & Concept
In a standard Markov chain, $q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$. Conditioning on $x_0$ does not change the forward transition because $x_t$ depends only on $x_{t-1}$.
Using Bayes' rule on the joint distribution of $(x_t, x_{t-1})$ conditioned on $x_0$:
$$q(x_t, x_{t-1} \mid x_0) = q(x_{t-1} \mid x_t, x_0) q(x_t \mid x_0)$$
Equivalently:
$$q(x_t, x_{t-1} \mid x_0) = q(x_t \mid x_{t-1}, x_0) q(x_{t-1} \mid x_0)$$
Equating both sides and solving for $q(x_t \mid x_{t-1})$:
$$q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0) = q(x_{t-1} \mid x_t, x_0) \frac{q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$$

This identity is the master key to decomposing the joint trajectory without computing intractable integrals.

### Physical Analogy
If you know where a suspect was at 3:00 PM ($x_{t-1}$) and where they are at 4:00 PM ($x_t$), knowing where they were born at 8:00 AM ($x_0$) doesn't change the 3:00-to-4:00 PM step. But knowing their home ($x_0$) allows you to deduce which path they likely took backwards from 4:00 PM to 3:00 PM!

### Micro-Number Numerical Verification
Let $P(A \mid B) = 0.4, P(B) = 0.5, P(A) = 0.2$. Then:
$$P(B \mid A) = \frac{P(A \mid B) P(B)}{P(A)} = \frac{0.4 \times 0.5}{0.2} = 1.0$$

### Runnable Python Verification
```python
import torch

def verify_bayes_identity():
    # Verify probability equality on discrete categorical distribution
    q_x0 = torch.tensor([0.6, 0.4])
    # Conditional kernel q(x1 | x0)
    q_x1_given_x0 = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    joint = q_x1_given_x0 * q_x0.unsqueeze(1)
    q_x1 = joint.sum(dim=0)
    q_x0_given_x1 = joint / q_x1.unsqueeze(0)
    
    # Check Bayes reconstruction: q(x1|x0) == q(x0|x1) * q(x1) / q(x0)
    reconstructed = (q_x0_given_x1 * q_x1.unsqueeze(0)) / q_x0.unsqueeze(1)
    assert torch.allclose(q_x1_given_x0, reconstructed)
    print("Bayes identity verified numerically.")

verify_bayes_identity()
```

---

## Foundational Pillar 3: Telescoping Products and Log Cancellation

### Mathematical Derivation & Concept
When multiplying fractions where the numerator of term $t$ matches the denominator of term $t-1$:
$$\prod_{t=2}^T \frac{q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)} = \frac{q(x_2 \mid x_0)}{q(x_1 \mid x_0)} \cdot \frac{q(x_3 \mid x_0)}{q(x_2 \mid x_0)} \cdots \frac{q(x_T \mid x_0)}{q(x_{T-1} \mid x_0)} = \frac{q(x_T \mid x_0)}{q(x_1 \mid x_0)}$$

In log-space, this sum telescopes:
$$\sum_{t=2}^T \left( \log q(x_t \mid x_0) - \log q(x_{t-1} \mid x_0) \right) = \log q(x_T \mid x_0) - \log q(x_1 \mid x_0)$$
All $T-2$ intermediate distributions cancel completely, leaving only the boundary terms at $t=1$ and $t=T$.

```
   log q(x_2) - log q(x_1)
+  log q(x_3) - log q(x_2)    <--- log q(x_2) cancels!
+  log q(x_4) - log q(x_3)    <--- log q(x_3) cancels!
...
+  log q(x_T) - log q(x_{T-1})
--------------------------------
=  log q(x_T) - log q(x_1)    (Only endpoints survive!)
```

### Micro-Number Numerical Verification
Let $T=4$. Values: $\log q_1 = -5.0, \log q_2 = -7.0, \log q_3 = -9.0, \log q_4 = -12.0$.
$$\Delta_2 = -7.0 - (-5.0) = -2.0$$
$$\Delta_3 = -9.0 - (-7.0) = -2.0$$
$$\Delta_4 = -12.0 - (-9.0) = -3.0$$
Sum: $-2.0 + (-2.0) + (-3.0) = -7.0$.
Boundary difference: $\log q_4 - \log q_1 = -12.0 - (-5.0) = -7.0$. Exact match!

### Physical Analogy
An accordion folded tight: when you pull the two outer handles, every internal pleat unfolds and flattens out, leaving only the two outer handles defining the span.

### Runnable Python Verification
```python
import torch

def verify_telescoping():
    vals = torch.tensor([1.2, 3.4, 5.6, 7.8, 9.0])
    deltas = vals[1:] - vals[:-1]
    telescoped_sum = deltas.sum().item()
    direct_diff = (vals[-1] - vals[0]).item()
    assert abs(telescoped_sum - direct_diff) < 1e-5
    print(f"Telescoping verified: sum={telescoped_sum:.4f}, direct={direct_diff:.4f}")

verify_telescoping()
```

---

## Foundational Pillar 4: KL Divergence Between Two Multivariate Gaussians

### Mathematical Derivation & Concept
For two $D$-dimensional Gaussians $q(x) = \mathcal{N}(x; \mu_q, \sigma_q^2 I)$ and $p(x) = \mathcal{N}(x; \mu_p, \sigma_p^2 I)$:
$$D_{\text{KL}}(q \parallel p) = \frac{1}{2} \left[ \sum_{i=1}^D \left( \frac{\sigma_q^2}{\sigma_p^2} - 1 + \log\frac{\sigma_p^2}{\sigma_q^2} \right) + \frac{\|\mu_q - \mu_p\|^2}{\sigma_p^2} \right]$$

When both distributions share identical fixed scalar variance $\sigma_q^2 = \sigma_p^2 = \sigma^2$:
$$D_{\text{KL}}(q \parallel p) = \frac{1}{2\sigma^2} \|\mu_q - \mu_p\|^2$$
The statistical divergence between two Gaussian transitions reduces directly to an ordinary Mean Squared Error (MSE) Euclidean distance between their mean vectors!

### Micro-Number Numerical Verification
Let $\sigma^2 = 1.0$, $\mu_q = [1.0, 2.0]$, $\mu_p = [1.0, 4.0]$.
$\|\mu_q - \mu_p\|^2 = (1-1)^2 + (2-4)^2 = 0 + 4 = 4.0$.
$D_{\text{KL}} = \frac{1}{2(1.0)} (4.0) = 2.0$ nats.

### Physical Analogy
Comparing two flashlights shining on a wall. If both flashlights have identical beam spread (variance), the dissimilarity in where they illuminate is determined entirely by the physical distance between their beam centers (means).

### Runnable Python Verification
```python
import torch

def verify_gaussian_kl():
    mu_q = torch.tensor([1.0, 2.0])
    mu_p = torch.tensor([1.0, 4.0])
    sigma_sq = 1.0
    kl = 0.5 * torch.sum((mu_q - mu_p)**2) / sigma_sq
    assert kl.item() == 2.0
    print(f"Gaussian KL verified: {kl.item():.4f} nats")

verify_gaussian_kl()
```

---

## Curriculum Bridges

| Concept | Upstream Prerequisite | Downstream Application in Lecture 14 |
| :--- | :--- | :--- |
| KL Divergence | [02-KL_Divergence.md](../../MathsTerms/04-Information-Theory-and-Divergences/02-KL_Divergence.md) | Formulating the denoising matching terms $L_{t-1}$ in ELBO |
| Joint & Conditional Distributions | [03-Joint_Marginal_Conditional_Dist.md](../../MathsTerms/03-Probability-and-Statistical-Estimation/03-Joint_Marginal_Conditional_Dist.md) | Expanding $q(x_{1:T} \mid x_0)$ and conditioning on $x_0$ |
| Latent Variable Models | [05-Latent_Variable_Models.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/05-Latent_Variable_Models.md) | Bounding marginal log-likelihood $\log p(x_0)$ via trajectory variational bound |

---

## Diagnostics Self-Assessment

1. **Why does naive Monte Carlo evaluation of $\mathbb{E}_q[\log(p/q)]$ fail for $T=1000$?**  
   *Answer*: Because sampling an entire trajectory of 1,000 high-dimensional random vectors introduces massive variance; the estimator will fluctuate wildly and provide unusable gradient estimates.

2. **What allows intermediate terms $\log q(x_t \mid x_0)$ to cancel out in the ELBO?**  
   *Answer*: The telescoping sum property: expanding $q(x_t \mid x_{t-1})$ using Bayes rule produces a ratio whose numerator and denominator cancel across successive time steps.
