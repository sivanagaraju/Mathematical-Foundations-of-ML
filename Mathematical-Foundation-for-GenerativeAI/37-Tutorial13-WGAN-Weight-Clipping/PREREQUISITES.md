# Pedagogical Prerequisites: Tutorial 13 WGAN Weight Clipping

Welcome to the foundational guide for Wasserstein Generative Adversarial Networks (WGAN). Before implementing weight clipping and training Critic networks, master these fundamental mathematical pillars.

---

## Rosetta Stone Symbol Mapping

| Mathematical Symbol | Spoken Reading | Mathematical Concept | Plain-English Intuition |
|:---|:---|:---|:---|
| $W_1(P_r, P_g)$ | "double-yoo one of pee-ahr and pee-jee" | 1-Wasserstein Distance / Earth Mover's Distance | The minimum physical work needed to move one probability shape into another |
| $\Pi(P_r, P_g)$ | "pie of pee-ahr and pee-jee" | Set of all joint probability couplings | All possible transportation schedules connecting source and destination points |
| $\|f\|_L \le 1$ | "one-Lipschitz norm of eff" | 1-Lipschitz continuity constraint | A strict speed limit forbidding the function from changing faster than 1 unit per unit distance |
| $f_w(x)$ | "eff sub double-yoo of eks" | Parameterized Critic network | The scoring judge that outputs a continuous real number representing data quality |
| $[-c, c]$ | "closed interval minus see to see" | Hard weight clipping hypercube | A compact box bounding all network parameters to enforce Lipschitz continuity |
| $n_{critic}$ | "en-krit-ik" | Critic iterations per generator step | The number of practice rounds the judge gets before the generator makes a single move |
| $\mathbb{E}_{P_r}[f(x)]$ | "expectation of eff of eks under pee-ahr" | Expected score on real data | The average score awarded by the Critic to genuine training samples |

---

## Curriculum & Prerequisite Bridges

| Sibling Module | Core Mathematical Concept | How it Unlocks This Lecture |
|:---|:---|:---|
| [MathsTerms: Wasserstein Distance](../../MathsTerms/05-Information-Theory-and-Divergences/05-Wasserstein_Distance_and_EMD.md) | Primal optimal transport and Earth Mover's Distance | Provides the geometric foundation for the WGAN loss function |
| [MathsTerms: Lipschitz Continuity](../../MathsTerms/01-Primal-Analysis-and-Foundations/04-Lipschitz_Continuity.md) | Gradient norm bounds and Lipschitz constants | Explains why weight clipping bounds the network's rate of change |
| [MathsTerms: Minimax Game and GANs](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/09-Minimax_Game_and_GANs.md) | Zero-sum adversarial games and JS divergence | Highlights the vanishing gradient problem that WGAN solves |
| [Lec 11: Beta-VAE](../35-Lec11-Beta-VAE/NOTES.md) | Variational density fitting trade-offs | Connects latent generation to distribution matching |

---

## Pillar 1: The Earth Mover's Distance (1-Wasserstein Metric)
<a id="p1-wasserstein-metric"></a>

### 👶 ELI5 Intuition
Imagine you have a pile of dirt at position $x = 0$ on a line, and you need to fill a hole of the exact same size at position $x = 5$. The Earth Mover's Distance measures the amount of physical labor: the weight of the dirt multiplied by the distance traveled ($1 \text{ kg} \times 5 \text{ meters} = 5 \text{ meter-kg}$). If the hole is moved to $x = 10$, the work doubles to $10$. It provides a steady, continuous gradient indicating exactly which direction to push the dirt.

### 🔢 Concrete Micro-Numbers
Let real distribution $P_r$ be a point mass at $x = 0$, and generated distribution $P_\theta$ be a point mass at $x = \theta$.
- **Wasserstein Distance:** $W_1(P_0, P_\theta) = |\theta - 0| = |\theta|$.
  Gradient: $\frac{d}{d\theta} W_1 = \text{sign}(\theta) \ne 0$.
- **Jensen-Shannon Divergence:**
  If $\theta \ne 0$, the supports are completely disjoint.
  $D_{JS}(P_0 \parallel P_\theta) = \frac{1}{2} D_{KL}(P_0 \parallel \frac{P_0 + P_\theta}{2}) + \frac{1}{2} D_{KL}(P_\theta \parallel \frac{P_0 + P_\theta}{2}) = \log(2) \approx 0.693$.
  Gradient: $\frac{d}{d\theta} D_{JS} = 0$. Optimization is paralyzed!

### 📐 Formal Math
The 1-Wasserstein distance between probability measures $P_r$ and $P_g$ is:
$$W_1(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma}[||x - y||]$$
where $\Pi(P_r, P_g)$ is the set of joint distributions whose marginals are $P_r$ and $P_g$.

### 💻 Runnable Code Snippet
```python
import numpy as np

theta = 3.5
# 1D Wasserstein between point masses is absolute distance
w1 = abs(theta - 0.0)
# JS divergence for disjoint point masses is log(2)
js = np.log(2.0)
assert np.isclose(w1, 3.5)
assert np.isclose(js, 0.693147, atol=1e-4)
print(f"Pillar 1 Clean: W1={w1:.2f} has active gradient, whereas JS={js:.4f} has zero gradient!")
```

### 🎯 Diagnostic Mini-Check / Self-Test
- **Question:** If $\theta = 10.0$, what is the gradient of $D_{JS}$ with respect to $\theta$?
- **Answer:** Exactly $0.0$, because $D_{JS} = \log(2)$ is a flat constant for all non-overlapping distributions.

---

## Pillar 2: Kantorovich-Rubinstein Duality
<a id="p2-kantorovich-duality"></a>

### 👶 ELI5 Intuition
Instead of surveying every single truck driver and route to find the cheapest way to ship goods across the country (the primal problem), you hire an independent logistics auditor who sets fair prices at every warehouse (the dual problem). If the auditor's price differences are capped by actual transportation costs (1-Lipschitz), the total money collected matches the cheapest shipping plan.

### 🔢 Concrete Micro-Numbers
Let $P_r$ put mass at $x = 2.0$ and $P_g$ put mass at $y = 5.0$.
A 1-Lipschitz function $f(x)$ must satisfy $|f(2) - f(5)| \le |2 - 5| = 3.0$.
To maximize $\mathbb{E}_{P_r}[f(x)] - \mathbb{E}_{P_g}[f(y)] = f(2) - f(5)$:
The maximum possible difference is achieved when $f(2) - f(5) = 3.0$ (e.g. $f(x) = -x$).
Then $f(2) - f(5) = -2 - (-5) = 3.0$, which equals the exact Wasserstein distance!

### 📐 Formal Math
By the Kantorovich-Rubinstein theorem:
$$W_1(P_r, P_g) = \sup_{||f||_L \le 1} \left( \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{y \sim P_g}[f(y)] \right)$$
where the supremum is taken over all functions with Lipschitz constant:
$$\|f\|_L = \sup_{x \ne y} \frac{|f(x) - f(y)|}{||x - y||} \le 1$$

### 💻 Runnable Code Snippet
```python
import numpy as np

x_real = 2.0
x_fake = 5.0
# Optimal 1-Lipschitz witness function f(x) = -x
f_real = -x_real
f_fake = -x_fake
w1_dual = f_real - f_fake
assert np.isclose(w1_dual, 3.0)
print(f"Pillar 2 Clean: Dual supremum evaluated to exact distance {w1_dual}")
```

### 🎯 Diagnostic Mini-Check / Self-Test
- **Question:** What happens to the dual objective if the 1-Lipschitz constraint is removed?
- **Answer:** The objective diverges to $+\infty$, because a function with unbounded slope could award $+10^{9}$ to real samples and $-10^{9}$ to fake samples.

---

## Pillar 3: Lipschitz Continuity and Gradient Norms
<a id="p3-lipschitz-continuity"></a>

### 👶 ELI5 Intuition
Imagine driving a car on a road where the speed limit is 1 meter per second. No matter how aggressively you steer or tap the gas pedal, you can never cover more than 10 meters in 10 seconds. Lipschitz continuity is a speed limit on a mathematical function: its output can never jump or spike abruptly.

### 🔢 Concrete Micro-Numbers
Let $f(x) = W x$ for scalar $x$.
Difference: $|f(x_1) - f(x_2)| = |W (x_1 - x_2)| = |W| \cdot |x_1 - x_2|$.
To ensure $|f(x_1) - f(x_2)| \le 1.0 \cdot |x_1 - x_2|$, we must require:
$$|W| \le 1.0$$
If $W = 1.5$, then for $x_1 = 0, x_2 = 2$: $|f(0) - f(2)| = 3.0 > 2.0$, violating 1-Lipschitz continuity!

### 📐 Formal Math
For a differentiable function $f: \mathbb{R}^D \to \mathbb{R}$, $f$ is $K$-Lipschitz continuous if and only if its gradient norm is bounded by $K$ everywhere:
$$||\nabla_x f(x)||_2 \le K, \quad \forall x \in \mathbb{R}^D$$
In deep networks $f(x) = W_L \sigma(W_{L-1} \dots \sigma(W_1 x))$, if activation $\sigma$ is 1-Lipschitz (like ReLU):
$$\|f\|_L \le \prod_{l=1}^L \|W_l\|_2$$

### 💻 Runnable Code Snippet
```python
import torch

W = torch.tensor([[0.8]], requires_grad=True)
x1 = torch.tensor([[0.0]])
x2 = torch.tensor([[2.0]])
diff_out = torch.abs(torch.matmul(x1, W) - torch.matmul(x2, W))
diff_in = torch.abs(x1 - x2)
lipschitz_ratio = diff_out / diff_in
assert lipschitz_ratio.item() <= 1.0
print(f"Pillar 3 Clean: Function with W={W.item()} is 1-Lipschitz (ratio={lipschitz_ratio.item():.2f})")
```

### 🎯 Diagnostic Mini-Check / Self-Test
- **Question:** Is the LeakyReLU activation function 1-Lipschitz continuous?
- **Answer:** Yes, because its maximum slope is $\max(1.0, \alpha) = 1.0$ for $\alpha \in [0, 1]$.

---

## Pillar 4: Hard Weight Clipping Mechanics
<a id="p4-hard-weight-clipping"></a>

### 👶 ELI5 Intuition
How do you guarantee that a deep neural network obeys a speed limit without redesigning every gear? The simplest and crudest way is to install a physical iron bracket on the gas pedal: clamp every single weight in the network so it can never exceed $[-0.01, 0.01]$.

### 🔢 Concrete Micro-Numbers
Suppose a Critic weight tensor has value $w = 0.035$.
With clipping threshold $c = 0.01$:
$$w_{\text{clipped}} = \text{clamp}(w, -0.01, 0.01) = 0.01$$
If another weight has value $w = -0.050$:
$$w_{\text{clipped}} = \text{clamp}(w, -0.01, 0.01) = -0.01$$
All weights are strictly forced into the compact interval $[-0.01, 0.01]$.

### 📐 Formal Math
Let network parameters $w \in \mathcal{W} = [-c, c]^P$. The operator:
$$\mathcal{P}_{\mathcal{W}}(w) = \max(-c, \min(c, w))$$
projects parameters onto the compact hypercube $\mathcal{W}$. Because the parameter space is compact, the function space $\{f_w : w \in \mathcal{W}\}$ is bounded and therefore $K$-Lipschitz for some constant $K(c)$.

### 💻 Runnable Code Snippet
```python
import torch

weights = torch.tensor([-0.05, 0.005, 0.03])
c = 0.01
clipped = torch.clamp(weights, -c, c)
assert torch.all(clipped >= -c)
assert torch.all(clipped <= c)
assert torch.isclose(clipped[0], torch.tensor(-0.01))
assert torch.isclose(clipped[2], torch.tensor(0.01))
print(f"Pillar 4 Clean: Weights clipped to range [{-c}, {c}]: {clipped.tolist()}")
```

### 🎯 Diagnostic Mini-Check / Self-Test
- **Question:** Why does weight clipping enforce $K$-Lipschitz continuity rather than strictly $1$-Lipschitz continuity?
- **Answer:** The product of weight norms depends on the network depth $L$ and width, yielding $K \le (c \sqrt{D})^L$, which bounds the Lipschitz constant to some finite $K$ rather than exactly $1.0$.
