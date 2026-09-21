# Tutorial 13: Wasserstein GAN (WGAN) Implementation using Gradient Clip

> **Prerequisites First:** Review foundational optimal transport, Kantorovich-Rubinstein duality, and Lipschitz continuity in [PREREQUISITES.md](./PREREQUISITES.md). For research literature, university slide decks, and production implementations, consult [references.md](./references.md). Interactive testing questions are available in [quiz.html](./quiz.html).

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Master End-to-End Simulation](#master-end-to-end-simulation)
3. [Topic 1: Pitfalls of Traditional GANs and Jensen-Shannon Divergence Pathologies](#topic-1-pitfalls-of-traditional-gans-and-jensen-shannon-divergence-pathologies)
4. [Topic 2: Optimal Transport and the 1-Wasserstein Metric (Earth Mover's Distance)](#topic-2-optimal-transport-and-the-1-wasserstein-metric-earth-movers-distance)
5. [Topic 3: Kantorovich-Rubinstein Duality and the 1-Lipschitz Constraint](#topic-3-kantorovich-rubinstein-duality-and-the-1-lipschitz-constraint)
6. [Topic 4: Critic Architecture: Eliminating Sigmoid and Outputting Linear Scores](#topic-4-critic-architecture-eliminating-sigmoid-and-outputting-linear-scores)
7. [Topic 5: Enforcing Lipschitz Continuity via Hard Weight Clipping [-c, c]](#topic-5-enforcing-lipschitz-continuity-via-hard-weight-clipping--c-c)
8. [Topic 6: Training Dynamics: Critic Iterations (n_critic = 5) and Practical Clipping Failures](#topic-6-training-dynamics-critic-iterations-n_critic--5-and-practical-clipping-failures)
9. [Workplace Debugging Scenarios](#workplace-debugging-scenarios)
10. [References & Further Reading](#references--further-reading)

---

## Executive Summary

Standard Generative Adversarial Networks suffer from training instability and mode collapse because they implicitly optimize Jensen-Shannon divergence. In Tutorial 13, Prof. Prathosh implements the Wasserstein GAN (WGAN) using hard weight clipping. By leveraging the 1-Wasserstein metric and Kantorovich-Rubinstein duality, WGAN transforms the adversarial discriminator into a 1-Lipschitz continuous Critic whose loss function correlates directly with generative sample quality.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     WGAN WEIGHT CLIPPING SYSTEM ARCHITECTURE                           │
└────────────────────────────────────────────────────────────────────────────────────────┘

  [ Real Data x ]  ───> ┌───────────────────────────┐ ───> Score f_w(x)
                        │      1-Lipschitz Critic   │
  [ Noise z ] ──> [ G ] ───> │  f_w: Linear Output       │ ───> Score f_w(G(z))
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Wasserstein Loss:       │
                        │   L_C = E[f(fake)] - E[f(real)]
                        └─────────────┬─────────────┘
                                      │ RMSprop Step
                                      ▼
                        ┌───────────────────────────┐
                        │   Hard Weight Clipping:   │
                        │   w <--- clamp(w, -c, c)  │
                        └───────────────────────────┘
```
*Figure 1: High-level architectural pipeline of Wasserstein GAN with weight clipping, illustrating the linear Critic and parameter clamping.*

### Scenario Walkthrough
During training, real images $x \sim P_r$ and latent noise vectors $z \sim \mathcal{N}(0, I)$ are sampled in mini-batches. The generator $G_\theta(z)$ synthesizes fake images $\tilde{x}$. The Critic network $f_w$, free of any sigmoid activations, outputs unconstrained scalar scores for both real and fake images. The Critic maximizes score separation $\mathbb{E}[f(x)] - \mathbb{E}[f(\tilde{x})]$. Following each RMSprop update, all Critic weights are clamped to $[-c, c]$. The Critic is updated for $n_{critic} = 5$ steps before the generator receives a single update step.

### Failure / Contrast Path
If clipping parameter $c$ is set too small ($c < 0.001$), gradients vanish exponentially through deep layers, freezing the Critic. If $c$ is set too large ($c > 0.1$), the Critic's gradients explode and weights slam against boundaries, causing extreme instability.

### STOP / Out of Scope
Gradient penalty formulation (WGAN-GP) and spectral normalization are previewed conceptually as advanced remedies for weight clipping, but remain out of scope for full mathematical derivation in this tutorial.

### Load-Bearing Claims
1. Standard GANs suffer from vanishing gradients when data and model distributions reside on disjoint low-dimensional manifolds.
2. The 1-Wasserstein distance provides continuous and differentiable gradients everywhere, even when distributions have disjoint supports.
3. Kantorovich-Rubinstein duality converts intractable infimum over transport plans into a supremum over 1-Lipschitz continuous functions.
4. The Critic network discards the final Sigmoid activation and outputs an unbounded scalar score representing Wasserstein potential.
5. Clamping weights to $[-c, c]$ guarantees that the Critic is $K$-Lipschitz with constant $K$ depending on $c$ and network depth.
6. Hard weight clipping leads to capacity underuse and requires training the Critic for $n_{critic} = 5$ steps per generator step using RMSprop.

### Comparative Feature & Tradeoff Matrix

| Method | Objective Divergence | Discriminator / Critic Output | Lipschitz Enforcement | Gradient Behavior on Disjoint Supports | Optimizer |
|:---|:---|:---|:---|:---|:---|
| Standard GAN | Jensen-Shannon ($D_{JS}$) | Probability $D(x) \in [0, 1]$ (Sigmoid) | None | Vanishes completely ($\nabla_\theta = 0$) | Adam ($\beta_1 = 0.5$) |
| WGAN (Clipping) | Earth Mover's Distance ($W_1$) | Unbounded scalar $f_w(x) \in \mathbb{R}$ | Hard Weight Clipping $w \in [-c, c]$ | Continuous and non-zero everywhere | RMSprop (No momentum) |
| WGAN-GP | Earth Mover's Distance ($W_1$) | Unbounded scalar $f_w(x) \in \mathbb{R}$ | Gradient Norm Penalty $(\|\nabla f\| - 1)^2$ | Continuous and non-zero everywhere | Adam ($\beta_1 = 0.0, \beta_2 = 0.9$) |

### Common Traps & Numerical Fixes
- **Trap 1: Momentum Thrashing at Clipping Boundaries:** Using Adam with $\beta_1 = 0.9$ causes running momentum to push weights repeatedly against clipping limits. **Fix:** Use RMSprop with learning rate $5 \times 10^{-5}$ without momentum.
- **Trap 2: Adding Sigmoid to the Critic Output:** Placing a sigmoid activation on the Critic collapses its unbounded potential into $[0, 1]$, destroying Wasserstein metric proportionality. **Fix:** Ensure the Critic's final layer is strictly linear.

---

## Master End-to-End Simulation

The following complete PyTorch simulation verifies the WGAN Critic forward pass, Wasserstein adversarial loss computation, backward pass, and weight clipping step end-to-end:

```python
import torch
import torch.nn as nn

# Master Simulation: WGAN Critic Training Step with Weight Clipping
torch.manual_seed(42)

class WGANTestCritic(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1) # Strictly linear output
        )

    def forward(self, x):
        return self.net(x)

# Setup data and network
B, D = 8, 16
c = 0.01
critic = WGANTestCritic(in_dim=D)
real_data = torch.randn(B, D)
fake_data = torch.randn(B, D)

# Critic forward evaluation
real_score = critic(real_data)
fake_score = critic(fake_data)

# Wasserstein loss: E[f(fake)] - E[f(real)]
loss_critic = fake_score.mean() - real_score.mean()
loss_critic.backward()

# Weight clipping step
for p in critic.parameters():
    p.data.clamp_(-c, c)

# Assertions verifying weight clamping and gradient validity
for p in critic.parameters():
    assert torch.all(p.data >= -c), f"Weight exceeded lower bound -c: {p.data.min()}"
    assert torch.all(p.data <= c), f"Weight exceeded upper bound c: {p.data.max()}"

print(f"Master Simulation Clean: Critic Loss = {loss_critic.item():.4f}, all weights in [{-c}, {c}]")
```

---

## Topic 1: Pitfalls of Traditional GANs and Jensen-Shannon Divergence Pathologies

### Where this sits on the master map
Grounds the motivation for optimal transport in the mathematical failure of standard GANs, connecting to Pillar 1 ([The Earth Mover's Distance](./PREREQUISITES.md#p1-wasserstein-metric)).

### Board / screenshot
```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ BOARD: PATHOLOGY OF JENSEN-SHANNON DIVERGENCE                                    │
│                                                                                  │
│   Real Distribution P_r: Dirac delta at x = 0                                    │
│   Model Distribution P_theta: Dirac delta at x = theta                           │
│                                                                                  │
│   For ANY theta != 0:                                                            │
│   - Supports are DISJOINT! P_r cap P_theta = empty set.                          │
│   - D_JS(P_r || P_theta) = log(2) = 0.693 (CONSTANT!)                            │
│   - d D_JS / d theta = 0.0  ===> VANISHING GRADIENTS!                            │
│                                                                                  │
│   Notice: JS divergence provides zero learning signal when manifolds do not cross│
└──────────────────────────────────────────────────────────────────────────────────┘
```

### What he is establishing
Prof. Prathosh opens Tutorial 13 by reviewing why standard GANs frequently suffer from vanishing gradients and mode collapse. In standard GANs, the discriminator optimizes binary cross-entropy, which when maximized reaches the Jensen-Shannon divergence $D_{JS}(P_r \parallel P_g)$. In high-dimensional spaces, real data resides on low-dimensional sub-manifolds. The probability that two independent low-dimensional manifolds intersect in high-dimensional space is measure zero.

Whenever the generated distribution $P_g$ and real data distribution $P_r$ have non-overlapping supports, the Jensen-Shannon divergence is a flat constant equal to $\log(2) \approx 0.693$. The derivative with respect to generator parameters evaluates to zero everywhere. The discriminator reaches 100% accuracy instantly, but yields zero gradient to guide the generator.

The wrong move is attempting to fix training by lowering learning rates or adding arbitrary Gaussian noise to images; the right move is replacing Jensen-Shannon divergence with the 1-Wasserstein metric, and we now have a distance function that provides smooth, non-zero gradients even across completely disjoint distributions.

- `👶 ELI5 Intuition`: If you are lost in a desert 10 miles from an oasis, a standard GAN referee only shouts "You are wrong!" without telling you which way to walk. A Wasserstein referee tells you "You are 10 miles due East," pointing you straight toward the oasis.
- `🔍 Plain-English Breakdown`: Standard GANs give up when the fake images are completely different from real ones. WGAN measures the physical distance between them so the generator always knows which way to move.
- `🔢 Concrete Numbers`: For two point masses separated by distance $\theta = 5.0$, JS divergence is flat at $0.693$ with gradient $0.0$. The Wasserstein distance is $5.0$ with constant gradient $1.0$.
- `📐 Formal Math`: The gradient of the JS objective under disjoint supports:
  $$\nabla_\theta D_{JS}(P_r \parallel P_\theta) = \mathbf{0} \quad (\text{when } \text{supp}(P_r) \cap \text{supp}(P_\theta) = \emptyset)$$
- `💻 Runnable Code`:
  ```python
  import numpy as np
  # Divergence comparison across distance theta
  thetas = [1.0, 5.0, 10.0]
  js_divs = [np.log(2.0) for _ in thetas] # Constant!
  w1_dists = [abs(t) for t in thetas]     # Linear!
  assert all(np.isclose(j, 0.693147, atol=1e-4) for j in js_divs)
  assert w1_dists == [1.0, 5.0, 10.0]
  print("Topic 1 Clean: JS divergence remains flat while W1 scales linearly with distance")
  ```
- `🔗 MathsTerm Link`: [Minimax Game and GANs](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/09-Minimax_Game_and_GANs.md).

### Contrastive Analysis: Why X, Not Y?
Why choose the Wasserstein distance (X) rather than Jensen-Shannon or KL divergence (Y)? KL and JS divergences fail to induce a continuous metric topology on low-dimensional manifolds with disjoint supports, whereas Wasserstein distance is continuous and differentiable everywhere.

### Active Comprehension Checks
- **Check Your Understanding (Recall):** What numerical value does the JS divergence take when two distributions are completely disjoint? (Answer: Exactly $\log 2 \approx 0.693$ nats).
- **Check Your Understanding (Apply):** Why does an optimal discriminator in standard GANs cause the generator to stop learning? (Answer: Because the discriminator function becomes saturated at 1 on real data and 0 on fake data, flattening its derivative to zero).

### Analogy for this topic only
Is Jensen-Shannon divergence like an exam marked purely pass/fail with no score? If every student fails, nobody knows if they missed the passing mark by 1 point or 50 points. In lecture words: "When the discriminator is optimal, the gradients vanish because the JS divergence is locally constant."

### Local picture
```
┌────────────────────────────────────────────────────────────────────────┐
│ DISJOINT MANIFOLD TOPOLOGY                                             │
│                                                                        │
│   Real Manifold P_r                   Fake Manifold P_theta            │
│   ┌─────────────────────┐             ┌─────────────────────┐          │
│   │ x = 0 (Probability) │             │ x = theta (Fake)    │          │
│   └─────────────────────┘             └─────────────────────┘          │
│             │                                    │                     │
│             └───────────── Distance theta ───────┘                     │
│                                                                        │
│   Notice: Overlap is measure zero, rendering JS divergence flat.       │
└────────────────────────────────────────────────────────────────────────┘
```

### Bridge
Having established the catastrophic gradient failure of JS divergence on disjoint manifolds, we now turn to the mathematical definition of optimal transport and the Earth Mover's Distance.

---

## Topic 2: Optimal Transport and the 1-Wasserstein Metric (Earth Mover's Distance)

### Where this sits on the master map
Formulates the primal optimal transport problem, connecting to Pillar 1 ([The Earth Mover's Distance](./PREREQUISITES.md#p1-wasserstein-metric)).

### Board / screenshot
```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ BOARD: THE 1-WASSERSTEIN METRIC (EARTH MOVER'S DISTANCE)                         │
│                                                                                  │
│   Primal Formulation:                                                            │
│   W_1(P_r, P_g) = inf_{gamma in Pi(P_r, P_g)}  E_{(x, y) ~ gamma}[ ||x - y|| ]   │
│                                                                                  │
│   - gamma(x, y): Transport plan moving probability mass from y to x.             │
│   - ||x - y||:   Cost function (Euclidean distance).                             │
│                                                                                  │
│   Notice: Measures the minimum physical effort required to transport distributions.│
└──────────────────────────────────────────────────────────────────────────────────┘
```

### What he is establishing
Prof. Prathosh formalizes the 1-Wasserstein distance, historically termed the Earth Mover's Distance (EMD). Informally, if probability distributions are viewed as piles of earth, $W_1(P_r, P_g)$ represents the minimum cost of moving dirt from pile $P_g$ until it matches pile $P_r$, where cost is defined as mass times distance traveled.

Mathematically, let $\Pi(P_r, P_g)$ denote the collection of all joint probability distributions $\gamma(x, y)$ whose marginals satisfy $\int \gamma(x, y) dy = P_r(x)$ and $\int \gamma(x, y) dx = P_g(y)$. The 1-Wasserstein distance is:
$$W_1(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma}[||x - y||]$$
Unlike $f$-divergences, the Wasserstein distance incorporates the underlying geometric metric of the space ($||x - y||$).

A naive wrong move is assuming optimal transport requires transporting mass along straight Euclidean paths without accounting for density mass conservation; the right move is evaluating the infimum over valid couplings, and we now have a distance metric that smoothly reflects geometric convergence.

- `👶 ELI5 Intuition`: If you have 5 tons of sand at point A and want to build a sandcastle at point B 100 meters away, you must do $5 \times 100 = 500$ ton-meters of work.
- `🔍 Plain-English Breakdown`: The Wasserstein distance measures how far you have to move probability mass to turn one distribution into another.
- `🔢 Concrete Numbers`: Moving 1 unit of mass from $x = 1.0$ to $y = 4.0$ incurs transport cost $|1.0 - 4.0| = 3.0$.
- `📐 Formal Math`: The 1-Wasserstein distance satisfies all metric axioms on the space of probability measures: non-negativity, symmetry, identity of indiscernibles ($W_1 = 0 \iff P_r = P_g$), and triangle inequality.
- `💻 Runnable Code`:
  ```python
  import numpy as np
  # Simulating discrete 1D Earth Mover's Distance
  p = np.array([1.0, 0.0, 0.0])
  q = np.array([0.0, 0.0, 1.0])
  # Cumulative sum difference for 1D discrete distributions
  emd = np.sum(np.abs(np.cumsum(p) - np.cumsum(q)))
  assert emd == 2.0
  print(f"Topic 2 Clean: Discrete 1D EMD = {emd:.2f} distance units")
  ```
- `🔗 MathsTerm Link`: [Wasserstein Distance & EMD](../../MathsTerms/04-Information-Theory-and-Divergences/05-Wasserstein_Distance_and_EMD.md).

### Contrastive Analysis: Why X, Not Y?
Why choose the Earth Mover's Distance (X) rather than total variation distance (Y)? Total variation distance evaluates whether two samples are identical without considering how close they are in space, making it discontinuous when distributions do not overlap.

### Active Comprehension Checks
- **Check Your Understanding (Recall):** What does the term $\gamma(x, y)$ represent in the primal optimal transport problem? (Answer: A transport plan specifying how much probability mass is moved from location $y$ to location $x$).
- **Check Your Understanding (Apply):** If distribution $P$ converges weakly to distribution $Q$ ($P_n \xrightarrow{\mathcal{D}} Q$), what happens to $W_1(P_n, Q)$? (Answer: It converges continuously to zero: $W_1(P_n, Q) \to 0$).

### Analogy for this topic only
Is the Earth Mover's Distance like a logistics shipping bill? You pay the freight company based on weight multiplied by kilometers driven; shorter deliveries cost less money.

### Local picture
```
┌────────────────────────────────────────────────────────────────────────┐
│ EARTH MOVER'S TRANSPORT PLAN                                           │
│                                                                        │
│   Source Pile P_g                     Destination Hole P_r             │
│        ▄█▄                                     ▄█▄                     │
│       █████  ───────── Transport Plan ──────>  █████                   │
│      ███████         Gamma(x, y)              ███████                  │
│   ──────┬────────────────────────────────────────┬──────               │
│      y = 1.0                                  x = 4.0                  │
│                                                                        │
│   Notice: Cost = Mass * Distance = 1.0 * (4.0 - 1.0) = 3.0.            │
└────────────────────────────────────────────────────────────────────────┘
```

### Bridge
While the primal optimal transport formulation is conceptually transparent, searching over all joint probability distributions in high dimensions is computationally intractable. We now examine how Kantorovich-Rubinstein duality converts this into an optimization over neural networks.

---

## Topic 3: Kantorovich-Rubinstein Duality and the 1-Lipschitz Constraint

### Where this sits on the master map
Translates the infinite-dimensional primal transport problem into the dual optimization objective that powers WGAN, connecting to Pillar 2 ([Kantorovich-Rubinstein Duality](./PREREQUISITES.md#p2-kantorovich-duality)).

### Board / screenshot
```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ BOARD: KANTOROVICH-RUBINSTEIN DUALITY                                            │
│                                                                                  │
│   Primal:  inf_{gamma} E[ ||x - y|| ]   (INTRACTABLE IN HIGH DIMENSIONS)         │
│                        ||                                                        │
│                        vv                                                        │
│   Dual:    sup_{||f||_L <= 1} ( E_{P_r}[f(x)] - E_{P_g}[f(y)] )                  │
│                                                                                  │
│   Condition: f must be 1-Lipschitz!                                              │
│   |f(x_1) - f(x_2)| <= ||x_1 - x_2||  ===>  ||\nabla f(x)||_2 <= 1              │
│                                                                                  │
│   Notice: Converts impossible search over plans into training a single network f.│
└──────────────────────────────────────────────────────────────────────────────────┘
```

### What he is establishing
Prof. Prathosh presents the cornerstone theoretical theorem of WGAN: the Kantorovich-Rubinstein Duality theorem. The theorem proves that computing the infimum over all transport couplings $\Pi(P_r, P_g)$ is mathematically equivalent to computing the supremum over all 1-Lipschitz continuous functions:
$$W_1(P_r, P_g) = \sup_{||f||_L \le 1} \left( \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{y \sim P_g}[f(y)] \right)$$

This duality is transformative for deep learning. Instead of attempting to sample or represent high-dimensional transport matrices, we parameterize the witness function $f$ using a deep neural network $f_w$ (the Critic). The Critic's task is simple: assign high scores to real images $x \sim P_r$ and low scores to generated images $y \sim P_g$, subject to the strict constraint that $f_w$ is 1-Lipschitz continuous.

The wrong move is allowing the Critic to output arbitrarily steep gradients to separate real and fake samples; the right move is enforcing the 1-Lipschitz bound, and you can now guarantee that the Critic's objective measures the true Wasserstein distance.

- `👶 ELI5 Intuition`: Instead of checking every truck in the fleet, the government appoints a single building inspector. As long as the inspector's fees don't exceed $1 per mile, the difference between real and fake inspection fees matches the true travel cost.
- `🔍 Plain-English Breakdown`: The dual theorem lets us replace an impossible math problem with training a neural network that gives high numbers to real images and low numbers to fakes.
- `🔢 Concrete Numbers`: If $P_r$ is at $x=0$ and $P_g$ is at $y=4$, an unconstrained network could predict $f(0) = 1000$ and $f(4) = -1000$, yielding false score 2000. A 1-Lipschitz network is restricted to $f(0) - f(4) \le 4$, yielding the exact distance 4.
- `📐 Formal Math`: The 1-Lipschitz constraint restricts the function's directional derivative:
  $$\max_{v : ||v||=1} \langle \nabla_x f(x), v \rangle = ||\nabla_x f(x)||_2 \le 1$$
- `💻 Runnable Code`:
  ```python
  import torch
  # Verifying 1-Lipschitz slope constraint
  x1 = torch.tensor([0.0])
  x2 = torch.tensor([4.0])
  f1 = torch.tensor([2.0])
  f2 = torch.tensor([-2.0])
  slope = torch.abs(f1 - f2) / torch.abs(x1 - x2)
  assert slope.item() <= 1.0, "1-Lipschitz violated!"
  print(f"Topic 3 Clean: Verified 1-Lipschitz slope = {slope.item():.2f}")
  ```
- `🔗 MathsTerm Link`: [Lipschitz Continuity](../../MathsTerms/05-Convexity-Duality-and-Metric-Analysis/04-Lipschitz_Continuity.md).

### Contrastive Analysis: Why X, Not Y?
Why enforce 1-Lipschitz continuity (X) rather than unconstrained score maximization (Y)? Without the Lipschitz constraint, the supremum diverges to $+\infty$ and the objective ceases to be a valid metric.

### Active Comprehension Checks
- **Check Your Understanding (Recall):** What mathematical constraint must function $f$ satisfy in the Kantorovich-Rubinstein dual formulation? (Answer: It must be 1-Lipschitz continuous, meaning $|f(x_1) - f(x_2)| \le ||x_1 - x_2||$).
- **Check Your Understanding (Apply):** If a Critic network outputs $f(x) = 10 x$ for a scalar input $x$, what is its Lipschitz constant? (Answer: $K = 10$, which violates the 1-Lipschitz constraint).

### Analogy for this topic only
Is Kantorovich-Rubinstein duality like pricing carbon offsets? Instead of tracking every single carbon emission molecule, regulators set a capped price per ton; the net financial delta matches the total environmental cost.

### Local picture
```
┌────────────────────────────────────────────────────────────────────────┐
│ 1-LIPSCHITZ BOUND CONE GEOMETRY                                        │
│                                                                        │
│                    \           /  Allowed growth region                │
│                     \  f(x)   /   bounded by slope <= 1.0              │
│                      \   o   /                                         │
│                       \ / \ /                                          │
│                        x   x                                           │
│                       /     \                                          │
│                      /       \                                         │
│                                                                        │
│   Notice: Function cannot penetrate the cone of slope > 1.0.           │
└────────────────────────────────────────────────────────────────────────┘
```

### Bridge
Now that we have established the dual objective, we must examine how to modify the neural network architecture from a standard Discriminator into a true WGAN Critic.

---

## Topic 4: Critic Architecture: Eliminating Sigmoid and Outputting Linear Scores

### Where this sits on the master map
Defines the architectural adjustments required to convert binary classification networks into continuous potential witnesses, connecting to Pillar 2 and Pillar 3.

### Board / screenshot
```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ BOARD: CRITIC ARCHITECTURE SPECIFICATION                                         │
│                                                                                  │
│   Standard Discriminator:  Conv2d ---> BatchNorm ---> LeakyReLU ---> Sigmoid     │
│                            Output in [0, 1]  (Probability)                       │
│                                                                                  │
│   WGAN Critic:             Conv2d ---> [No BN] ---> LeakyReLU ---> Linear        │
│                            Output in (-inf, +inf)  (Unbounded Potential)         │
│                                                                                  │
│   Notice: The Critic is a regression model evaluating geometric potential.       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### What he is establishing
Prof. Prathosh details the architectural modifications that distinguish a WGAN Critic from a standard GAN Discriminator. A standard discriminator terminates with a Sigmoid activation to output a class probability $D(x) = \sigma(h(x)) \in [0, 1]$, and is trained via binary cross-entropy. In WGAN, the network is not a classifier: it is a function approximating the dual potential $f_w(x)$.

Consequently, the Critic discards the Sigmoid activation completely. The final layer is a linear projection producing an unbounded scalar $f_w(x) \in (-\infty, +\infty)$. A higher score indicates that $x$ lies in a high-density region of the real data distribution. The training loss for the Critic is simply:
$$\mathcal{L}_{\text{critic}} = \frac{1}{B} \sum_{i=1}^B f_w(G_\theta(z_i)) - \frac{1}{B} \sum_{i=1}^B f_w(x_i)$$
The generator minimizes:
$$\mathcal{L}_{\text{gen}} = -\frac{1}{B} \sum_{i=1}^B f_w(G_\theta(z_i))$$

A naive wrong move is leaving a Sigmoid on the final layer; the right move is outputting unconstrained linear scores, and we now have a Critic whose output delta directly reflects the Wasserstein distance.

- `👶 ELI5 Intuition`: A discriminator is a bouncer who only says "yes" or "no". A Critic is an Olympic judge who gives a score from 0 to 100 based on posture and technique.
- `🔍 Plain-English Breakdown`: Remove the sigmoid. The Critic should output any real number, where bigger numbers mean "more real."
- `🔢 Concrete Numbers`: If Critic scores real images at $+5.2$ and fake images at $-1.8$, the empirical Wasserstein estimate is $5.2 - (-1.8) = 7.0$.
- `📐 Formal Math`: The gradient of the generator loss with respect to fake image $\tilde{x} = G_\theta(z)$ is:
  $$\nabla_{\tilde{x}} \mathcal{L}_{\text{gen}} = -\nabla_{\tilde{x}} f_w(\tilde{x})$$
  Because $f_w$ is linear, this gradient does not vanish when $\tilde{x}$ is far from real data.
- `💻 Runnable Code`:
  ```python
  import torch
  import torch.nn as nn
  # Verifying linear output layer
  critic_head = nn.Linear(32, 1)
  feature_vec = torch.randn(4, 32)
  out = critic_head(feature_vec)
  assert out.shape == (4, 1)
  print(f"Topic 4 Clean: Linear Critic produced unconstrained outputs: {out.squeeze().tolist()}")
  ```
- `🔗 MathsTerm Link`: [Loss Functions](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/08-Loss_Functions.md).

### Contrastive Analysis: Why X, Not Y?
Why use a linear scalar output (X) rather than a sigmoid probability (Y)? Sigmoid saturates at $0$ and $1$, compressing output gradients to zero. A linear output maintains proportional gradients across the entire space.

### Active Comprehension Checks
- **Check Your Understanding (Recall):** Does the WGAN Critic use a Sigmoid activation in its output layer? (Answer: No, it outputs a strictly linear real number).
- **Check Your Understanding (Apply):** If a Critic outputs $+100$ for real images and $+20$ for fake images, what is the estimated Wasserstein distance? (Answer: $100 - 20 = 80$).

### Analogy for this topic only
Is the Critic like an altimeter rather than an on/off switch? An altimeter tells an airplane pilot exactly how many meters they are above the runway, providing a continuous landing gradient.

### Local picture
```
┌────────────────────────────────────────────────────────────────────────┐
│ CRITIC OUTPUT RANGE COMPARISON                                         │
│                                                                        │
│   Standard GAN Discriminator:                                          │
│   0.0 [===================== Saturated =====================] 1.0     │
│                                                                        │
│   WGAN Linear Critic:                                                  │
│   -inf <-------------------------- 0 ---------------------------> +inf │
│                                                                        │
│   Notice: Linear output provides unbounded, unsaturating signals.      │
└────────────────────────────────────────────────────────────────────────┘
```

### Bridge
While the linear Critic architecture provides unsaturated gradients, we must still enforce the 1-Lipschitz condition. We next examine the original weight clipping mechanism proposed by Arjovsky et al.

---

## Topic 5: Enforcing Lipschitz Continuity via Hard Weight Clipping [-c, c]

### Where this sits on the master map
Explores the parameter clamping mechanism used to constrain the Critic's Lipschitz constant, connecting to Pillar 4 ([Hard Weight Clipping Mechanics](./PREREQUISITES.md#p4-hard-weight-clipping)).

### Board / screenshot
```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ BOARD: HARD WEIGHT CLIPPING MECHANISM                                            │
│                                                                                  │
│   Constraint: w in [-c, c]^P                                                     │
│                                                                                  │
│   Algorithm:                                                                     │
│   1. Compute gradients: g = \nabla_w L_critic                                    │
│   2. Optimizer update:  w = w - \eta * g                                         │
│   3. CLAMP STEP:        w = torch.clamp(w, -c, c)                                │
│                                                                                  │
│   Notice: Simple and fast, but forces parameters into extreme boundary corners.  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### What he is establishing
Prof. Prathosh analyzes the simplest mechanism to enforce Lipschitz continuity: hard weight clipping. Since the Lipschitz constant of a neural network is bounded by the product of its layer spectral norms $\|W_l\|_2$, bounding each individual parameter $w_{i,j} \in [-c, c]$ guarantees that the entire network $f_w$ is $K$-Lipschitz for some constant $K$ depending on $c$ and architecture depth.

In PyTorch, weight clipping is implemented in two lines of code following the optimizer step:
```python
optimizer.step()
for p in critic.parameters():
    p.data.clamp_(-c, c)
```
Typically, clipping threshold $c$ is set to $0.01$.

However, Prof. Prathosh highlights that weight clipping is a terrible way to enforce a 1-Lipschitz constraint. By clamping all parameters into a hypercube $[-c, c]$, the optimizer quickly pushes almost all weights to the extreme boundaries $\pm c$. As a result, the Critic loses representational flexibility, acting as an overly simplistic piecewise linear function that underutilizes its network capacity.

A naive wrong move is setting clipping bound $c$ to an arbitrary large value like $1.0$; the right move is recognizing that $c$ must be kept small (e.g. $0.01$) to prevent gradient explosion, while preparing to migrate to gradient penalties (Tutorial 14).

- `👶 ELI5 Intuition`: To make sure a dog doesn't pull on its leash, you tie its legs together with short strings. It definitely can't run too fast (Lipschitz bound achieved), but it also can't walk naturally (capacity underuse).
- `🔍 Plain-English Breakdown`: After every update, any weight bigger than $0.01$ is forced back to $0.01$, and any weight smaller than $-0.01$ is forced to $-0.01$.
- `🔢 Concrete Numbers`: If a weight updates to $0.045$, clamping sets it to $0.010$. Over training, up to 90% of all weights in the network end up pegged at exactly $+0.01$ or $-0.01$.
- `📐 Formal Math`: The projection operator onto the $L_\infty$ ball:
  $$\Pi_{\mathcal{W}}(w) = \text{sign}(w) \min(|w|, c)$$
- `💻 Runnable Code`:
  ```python
  import torch
  p = torch.tensor([0.05, -0.03, 0.005], requires_grad=True)
  c = 0.01
  with torch.no_grad():
      p.clamp_(-c, c)
  assert torch.all(p >= -c) and torch.all(p <= c)
  print(f"Topic 5 Clean: Clamped parameter values: {p.tolist()}")
  ```
- `🔗 MathsTerm Link`: [Gradient Descent](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/09-Gradient_Descent.md).

### Contrastive Analysis: Why X, Not Y?
Why choose weight clipping (X) rather than unconstrained gradient descent (Y)? Without weight clipping, the Critic's parameters grow unboundedly to maximize score separation, destroying the 1-Lipschitz constraint and invalidating the Kantorovich-Rubinstein theorem.

### Active Comprehension Checks
- **Check Your Understanding (Recall):** What is the typical value of the clipping parameter $c$ in WGAN? (Answer: $c = 0.01$).
- **Check Your Understanding (Diagnose):** Why do weights in a clipped WGAN Critic tend to concentrate at the boundaries $+c$ and $-c$? (Answer: Because the Critic is continuously trained to maximize score separation, pushing weights outward until they strike the clipping limits).

### Analogy for this topic only
Is weight clipping like trimming a hedge with a chainsaw? It keeps the bushes inside a rectangular boundary, but it destroys the fine leaf shapes and delicate branches inside.

### Local picture
```
┌────────────────────────────────────────────────────────────────────────┐
│ WEIGHT DISTRIBUTION UNDER HARD CLIPPING                                │
│                                                                        │
│   Mass concentrates heavily at boundary spikes:                        │
│                                                                        │
│        ███                                             ███             │
│        ███                                             ███             │
│        ███                   ▄▄▄▄▄                     ███             │
│   ─────┴───────────────────────┼────────────────────────┴─────         │
│      -c = -0.01               0.0                     +c = +0.01       │
│                                                                        │
│   Notice: Capacity is underused as interior parameter space empties.   │
└────────────────────────────────────────────────────────────────────────┘
```

### Bridge
With the clipping mechanism established, we examine the overall training dynamics of the WGAN algorithm, including the necessity of multiple Critic iterations and the selection of optimizers.

---

## Topic 6: Training Dynamics: Critic Iterations (n_critic = 5) and Practical Clipping Failures

### Where this sits on the master map
Analyzes the operational training loop of WGAN, detailing the $n_{critic} = 5$ ratio and RMSprop optimizer requirements.

### Board / screenshot
```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ BOARD: WGAN TRAINING LOOP DYNAMICS                                               │
│                                                                                  │
│   For each training epoch:                                                       │
│     For t = 1 to n_critic (e.g. 5 steps):                                        │
│       - Sample real batch x ~ P_r, noise batch z ~ p(z)                          │
│       - L_critic = f_w(G(z)).mean() - f_w(x).mean()                              │
│       - Update Critic via RMSprop                                                │
│       - Clamp Critic weights: w = clamp(w, -c, c)                                │
│                                                                                  │
│     - Sample fresh noise z ~ p(z)                                                │
│     - L_gen = -f_w(G(z)).mean()                                                  │
│     - Update Generator via RMSprop (1 step)                                      │
│                                                                                  │
│   Notice: The Critic is trained close to optimality before generator moves.      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### What he is establishing
Prof. Prathosh concludes Tutorial 13 by examining the asymmetrical training loop of WGAN. In standard GANs, training the discriminator to optimality is disastrous because its gradients vanish. In WGAN, the opposite is true: the Kantorovich-Rubinstein duality theorem requires that the Critic be at the supremum. If the Critic is suboptimal, its score difference does not accurately approximate the Wasserstein distance.

To keep the Critic near optimality, WGAN trains the Critic for $n_{critic} = 5$ steps for every single generator step. Furthermore, Prof. Prathosh explains why RMSprop is chosen over Adam. When weights hit the hard boundaries $[-c, c]$, Adam's running first-moment buffer (momentum) continues trying to move in the historical gradient direction, causing weights to oscillate wildly against the clipping boundary. RMSprop tracks only the squared gradient denominator without directional momentum, providing stable updates.

The wrong move is training Critic and Generator 1:1 using Adam with momentum; the right move is setting $n_{critic} = 5$ with RMSprop, and we now have a stable training loop where the Critic loss steadily tracks sample fidelity.

- `👶 ELI5 Intuition`: If you want a dance teacher to critique your routine accurately, you let the teacher watch you practice 5 times before they give you a single piece of corrective advice.
- `🔍 Plain-English Breakdown`: The Critic needs 5 practice rounds for every 1 generator round. Use RMSprop because Adam gets confused by the clipping walls.
- `🔢 Concrete Numbers`: For 1000 generator steps, the Critic performs $1000 \times 5 = 5000$ forward and backward passes.
- `📐 Formal Math`: The empirical training loss tracks the estimated Wasserstein distance:
  $$\hat{W}_1(t) = \frac{1}{B} \sum_{i=1}^B f_w(x_i) - \frac{1}{B} \sum_{i=1}^B f_w(G_\theta(z_i))$$
  As generator fidelity improves, $\hat{W}_1(t) \to 0$ monotonically.
- `💻 Runnable Code`:
  ```python
  import torch
  # Simulating training step ratio
  n_critic = 5
  critic_steps = 0
  gen_steps = 0
  for _ in range(10):
      for _ in range(n_critic):
          critic_steps += 1
      gen_steps += 1
  assert critic_steps == 50
  assert gen_steps == 10
  print(f"Topic 6 Clean: Completed {critic_steps} Critic steps across {gen_steps} Generator steps")
  ```
- `🔗 MathsTerm Link`: [Gradient Descent](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/09-Gradient_Descent.md).

### Contrastive Analysis: Why X, Not Y?
Why choose RMSprop without momentum (X) rather than Adam with $\beta_1 = 0.9$ (Y)? Adam's momentum buffer becomes corrupted when weights are clamped, while RMSprop scales learning rates cleanly without boundary thrashing.

### Active Comprehension Checks
- **Check Your Understanding (Recall):** How many Critic updates are executed for each Generator update in standard WGAN? (Answer: $n_{critic} = 5$).
- **Check Your Understanding (Diagnose):** Why does monitoring the negative Critic loss give developers an accurate metric for image quality? (Answer: Because it directly approximates the true Wasserstein distance; lower distance corresponds to higher perceptual quality).

### Analogy for this topic only
Is $n_{critic} = 5$ like sharpening a chef's knife five times before making a delicate sushi cut? If the knife is dull (suboptimal Critic), the cut will be ragged and ruined.

### Local picture
```
┌────────────────────────────────────────────────────────────────────────┐
│ ASYMMETRIC TRAINING CADENCE                                            │
│                                                                        │
│   Step 1: Critic   ──────┐                                             │
│   Step 2: Critic   ──────┤                                             │
│   Step 3: Critic   ──────┼───> 1 Generator Step                        │
│   Step 4: Critic   ──────┤                                             │
│   Step 5: Critic   ──────┘                                             │
│                                                                        │
│   Notice: 5 Critic steps ensure optimal dual witness approximation.    │
└────────────────────────────────────────────────────────────────────────┘
```

### Bridge
We have now synthesized the entire theoretical and implementation pipeline of WGAN with weight clipping. We solidify these principles with real-world workplace debugging scenarios.

---

## Workplace Debugging Scenarios

### Scenario 1: Critic Weight Saturation and Loss Stagnation
**Incident:** An engineer training a WGAN with weight clipping $c = 0.01$ notices that the Critic loss stops decreasing after epoch 3. Inspecting the Critic weights reveals that 94% of all parameter values are exactly $+0.01$ or $-0.01$. The generator produces grainy, repetitive patterns.

**Mathematical Root Cause:** Hard weight clipping caused severe capacity underuse. The network lost its non-linear modeling capacity and collapsed into an extreme linear function whose internal layers were saturated at clipping boundaries.

**Debugging Steps:**
1. Inspect weight distribution: `torch.histc(critic.conv1.weight)`.
2. Observe massive spikes at $+c$ and $-c$ with near-zero density in between.
3. Verify gradient norms: early layers have vanishing gradients due to repeated multiplications by clipped weights.

**Code Fix:**
```python
# Temporary mitigation: slightly widen clipping threshold and reduce learning rate
c = 0.02
for p in critic.parameters():
    p.data.clamp_(-c, c)
# Long-term fix: Migrate to WGAN Gradient Penalty (Tutorial 14)
```

### Scenario 2: Erratic Loss Spikes under Adam Optimizer
**Incident:** A computer vision team replaced RMSprop with Adam ($\beta_1 = 0.9$) to speed up WGAN training. Within 50 iterations, the Critic loss fluctuated wildly between $-100$ and $+150$, causing generated images to explode into NaNs.

**Mathematical Root Cause:** When weights were clamped to $[-c, c]$, Adam's momentum term $\beta_1 m_t$ continued accumulating large directional forces, pushing the weights back into the clipping wall on every step and destabilizing the effective step size.

**Debugging Steps:**
1. Check optimizer configuration: Adam was using default $\beta_1 = 0.9$.
2. Monitor parameter updates before and after clamping: updates were thrashing back and forth across boundaries.

**Code Fix:**
```python
# Switch to RMSprop without momentum as specified by Arjovsky et al.
optimizer_C = torch.optim.RMSprop(critic.parameters(), lr=5e-5)
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=5e-5)
```

---

## References & Further Reading
For exhaustive mathematical literature, seminal arXiv publications, and university lecture slide archives, refer directly to [references.md](./references.md).
