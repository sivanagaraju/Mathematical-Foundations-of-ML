# Expectation-Maximization Algorithm: Latent Variable Estimation and Monotonic Convergence

[Module guide](README.md) · [Study routes](START_HERE.md) · Previous: [Latent variable models](05-Latent_Variable_Models.md) · Next: [ELBO and variational inference](07-ELBO_and_Variational_Inference.md)

## 1. What this idea helps you do

In real-world data, the most critical generative factors are often unobserved. A customer database contains transaction amounts ($x$), but customer persona ($z$) is hidden; an audio recording captures raw acoustic pressures ($x$), but phonetic phonemes ($z$) are hidden. When we attempt to fit a probabilistic model by maximizing the marginal log-likelihood:

$$\ln p(X \mid \theta) = \sum_{i=1}^N \ln \left( \sum_{z_i} p(x_i, z_i \mid \theta) \right)$$

we encounter an intractable mathematical obstacle: the summation over latent states sits inside the logarithm. This structural coupling prevents closed-form stationary solutions, couples all parameters together, and triggers numerical singularities where variances collapse to zero.

The **Expectation-Maximization (EM) Algorithm** is an iterative coordinate-ascent framework that solves this problem without computing intractable marginal gradients. Instead of optimizing the complex marginal likelihood directly, EM alternates between two simple steps:
1. **The E-step (Expectation):** Infers soft posterior probabilities (responsibilities) over the unobserved latent causes using the current model parameters.
2. **The M-step (Maximization):** Maximizes an analytically tractable surrogate lower bound (the $Q$-function) using the responsibilities as weights, updating parameters via decoupled, closed-form weighted averages.

```text
================================================================================
           THE 2-STAGE EXPECTATION-MAXIMIZATION COORDINATE ENGINE
================================================================================

  OBSERVED DATA X                      E-STEP: POSTERIOR INFERENCE
  Intractable Sum Inside Log:          Infer Soft Responsibilities over Latents
  ┌───────────────────────────┐        ┌───────────────────────────────────────┐
  │ ln p(X|θ) = ∑ ln ∑ p(x,z) │ ═════► │ γ_ik = P(z_i = k | x_i, θ^(t))        │
  │ Parameters coupled        │        │ Constructs concave surrogate Q(θ|θ^t) │
  │ Latent classes hidden     │        │ Eliminates sum inside logarithm       │
  └───────────────────────────┘        └───────────────────────────────────────┘
                ▲                                          │
                │              M-STEP: PARAMETER RE-ESTIMATION
                │              Maximize Surrogate Bound    │
                │              ┌───────────────────────────┴───────────┐
                └───────────── │ θ^(t+1) = argmax_θ Q(θ | θ^(t))       │
                                │ Closed-form weighted averages:        │
                                │ μ_k = ∑ γ_ik x_i / ∑ γ_ik             │
                                └───────────────────────────────────────┘
================================================================================
```

*What to notice from the diagram:*
1. The E-step uses current parameters $\theta^{(t)}$ to compute soft fractional responsibilities $\gamma_{ik} = p(z_i = k \mid x_i, \theta^{(t)})$, constructing a surrogate lower bound $Q(\theta \mid \theta^{(t)})$ that is tangent to the true log-likelihood.
2. The M-step maximizes this concave surrogate bound in closed form, yielding updated parameters $\theta^{(t+1)}$ that strictly improve the true marginal likelihood: $\ln p(X \mid \theta^{(t+1)}) \ge \ln p(X \mid \theta^{(t)})$.

**Prerequisites**
- **Required now:** Joint, marginal, and conditional probabilities ([Joint, marginal and conditional distributions, §3](../03-Probability-and-Statistical-Estimation/03-Joint_Marginal_Conditional_Dist.md)). Maximum Likelihood Estimation ([MLE, §2](../03-Probability-and-Statistical-Estimation/05-MLE.md)).
- **Required for optional depth:** Jensen's inequality and concave functions ([Convexity and Jensen's inequality, §2](../05-Convexity-Duality-and-Metric-Analysis/01-Convexity_and_Jensens_Inequality.md)).
- **Useful context:** [Latent variable models](05-Latent_Variable_Models.md) for latent variable foundations, and [ELBO and variational inference](07-ELBO_and_Variational_Inference.md) for generalizing EM to deep continuous latent representations.

**Target systems:** Gaussian Mixture Models (GMMs) for acoustic and tabular clustering, Hidden Markov Models (HMMs via Baum-Welch) in speech recognition and bio-sequence alignment, and the theoretical foundation for Variational Autoencoders (VAEs).

**Study time:** About 60–75 minutes for core coordinate ascent mechanics, Jensen lower bounds, and pencil-and-paper derivations; another 45 minutes for convergence proofs, PyTorch autograd stationarity verification, and exercises.

After studying, you should be able to:
1. Construct the variational free energy auxiliary function $F(q, \theta)$ and prove the Evidence Decomposition Identity.
2. Calculate exact posterior responsibilities $\gamma_{ik}$ in the E-step by hand.
3. Derive closed-form M-step updates for Gaussian Mixture Models using Lagrange multipliers for mixture proportions.
4. Prove the Dempster-Laird-Rubin Monotonic Convergence Theorem ($\ln p(X \mid \theta^{(t+1)}) \ge \ln p(X \mid \theta^{(t)})$).
5. Implement, regularize, and verify a dual-stage EM engine in pure Python and PyTorch.

**Fast route:** §§2–4 $\to$ §7 $\to$ §9 $\to$ §11 $\to$ §12, then §10 for production systems.  
**Deep route:** §§2–14 in order; §4 and §8 contain complete monotonic convergence proofs and Lagrange multiplier derivations.

---

## 2. Start with a problem you can picture

Suppose a teacher finds an unlabeled list of 3 exam scores from two different classrooms (Classroom 1 and Classroom 2):
$$x_1 = 1.0, \quad x_2 = 2.0, \quad x_3 = 7.0$$

The teacher knows each classroom's test scores follow a bell curve (Gaussian) with variance $\sigma^2 = 1.0$, but the class means $\mu_1, \mu_2$ and class proportions $\pi_1, \pi_2$ are unknown.

If we knew which student belonged to which class, finding the class averages would be trivial: just take the arithmetic mean of each classroom's students. Conversely, if we knew the true class averages, assigning students to classrooms would be trivial: assign each student to the nearest class average.

The dilemma is that **we know neither**:

```text
1D EXAM SCORE AXIS:
          x_1   x_2                                           x_3
Score:    1.0   2.0                                           7.0
───────────●─────●─────────────────────────────────────────────●────────►
           ▲                                                   ▲
      Cluster 1?                                          Cluster 2?
   (Scores 1.0, 2.0)                                     (Score 7.0)

INITIAL GUESS AT t = 0:
Gaussian 1: Center μ_1^(0) = 2.0,  Weight π_1 = 0.50
Gaussian 2: Center μ_2^(0) = 6.0,  Weight π_2 = 0.50

          μ_1^(0)                                   μ_2^(0)
Score:    2.0                                       6.0
───────────▲─────────────────────────────────────────▲──────────────────►
         /   \                                     /   \
        /  1  \                                   /  2  \
       /       \                                 /       \
```

*What to notice from the diagram:*
1. Points $x_1 = 1.0$ and $x_2 = 2.0$ lie close to initial center $\mu_1 = 2.0$.
2. Point $x_3 = 7.0$ lies closest to initial center $\mu_2 = 6.0$.
3. Instead of making hard, brittle 0/1 assignments, EM computes soft, fractional responsibilities $\gamma_{ik} \in [0, 1]$ indicating how likely each classroom was to produce each score.

**Predict before calculating:** After one round of computing soft responsibilities and updating the center of Cluster 1, will $\mu_1^{(1)}$ stay at $2.0$, or will it be pulled toward $1.0$? What will happen to $\mu_2^{(1)}$?

---

## 3. Name the objects and read the notation

Let $X = \{x_1, x_2, \dots, x_N\}$ be an observed dataset of $N$ observations in $\mathbb{R}^D$. Let $Z = \{z_1, z_2, \dots, z_N\}$ be the associated unobserved latent variables, where each $z_i \in \{1, \dots, K\}$ indicates the categorical cluster assignment.

The **marginal log-likelihood** (or incomplete-data log-likelihood) under parameters $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$ is:
$$\ell(\theta) \triangleq \ln p(X \mid \theta) = \sum_{i=1}^N \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right)$$

The **posterior responsibility** that component $k$ generated observation $x_i$ is defined as:
$$\gamma_{ik} \triangleq p(z_i = k \mid x_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \mathcal{N}(x_i \mid \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(x_i \mid \mu_j^{(t)}, \Sigma_j^{(t)})}$$

Read this equation aloud:  
*“Gamma-sub-i-k, the responsibility of cluster k for data point i, is defined as the mixture weight pi-sub-k times the Gaussian density of x-sub-i given mu-sub-k and Sigma-sub-k, divided by the marginal evidence sum over all components j.”*

The **surrogate function** (or $Q$-function) is the expected complete-data log-likelihood conditioned on the current parameters $\theta^{(t)}$:
$$Q(\theta \mid \theta^{(t)}) \triangleq \mathbb{E}_{Z \mid X, \theta^{(t)}}[\ln p(X, Z \mid \theta)] = \sum_{i=1}^N \sum_{k=1}^K \gamma_{ik} \ln \Big( \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \Big)$$

| Symbol | Spoken as | Mathematical role / dimensions | Concrete toy value (§2 / §9) |
| :--- | :--- | :--- | :--- |
| $X$ | “capital ex” | Observed data matrix; $N \times D$ | $\{1.0, 2.0, 7.0\}$ ($N=3, D=1$) |
| $Z$ | “capital zee” | Unobserved latent categorical states; $N \times 1$ | $z_i \in \{1, 2\}$ ($K=2$ clusters) |
| $\theta$ | “theta” | Model parameters $\{\pi_k, \mu_k, \Sigma_k\}$ | $\mu^{(0)} = [2.0, 6.0], \pi^{(0)} = [0.5, 0.5]$ |
| $\gamma_{ik}$ | “gamma sub i k” | Posterior responsibility $p(z_i = k \mid x_i, \theta^{(t)})$ | $\gamma_{11} \approx 0.9997, \gamma_{32} \approx 0.9997$ |
| $N_k$ | “capital en sub k” | Effective number of points assigned to cluster $k$ | $N_1 = \sum_i \gamma_{i1} \approx 1.5003$ |
| $Q(\theta \mid \theta^{(t)})$ | “Q of theta given theta t” | Expected complete log-likelihood surrogate bound | Maximized in closed form during M-step |
| $F(q, \theta)$ | “variational free energy” | Auxiliary lower bound on true log-evidence | $\ln p(X \mid \theta) = F(q, \theta) + D_{\text{KL}}(q \parallel p)$ |

---

## 4. Build the central relationship

### The Core "Aha!" Discovery

We cannot maximize $\ln \sum_k p(x, z=k \mid \theta)$ directly because the logarithm cannot distribute over the summation. But if we introduce an arbitrary probability distribution $q(Z)$ over the unobserved latent variables, Jensen's inequality allows us to push the logarithm inside the expectation, turning an intractable non-convex sum into a tractable concave lower bound!

### Derivation: The Evidence Decomposition Identity

Let $q(Z)$ be any valid probability distribution over the latent variables ($\sum_Z q(Z) = 1$, $q(Z) \ge 0$). We can rewrite the marginal log-likelihood identically:

$$\ln p(X \mid \theta) = \sum_Z q(Z) \ln p(X \mid \theta) = \sum_Z q(Z) \ln \left( \frac{p(X, Z \mid \theta)}{p(Z \mid X, \theta)} \right)$$

Multiplying and dividing the argument inside the logarithm by $q(Z)$:

$$\ln p(X \mid \theta) = \sum_Z q(Z) \ln \left( \frac{p(X, Z \mid \theta)}{q(Z)} \cdot \frac{q(Z)}{p(Z \mid X, \theta)} \right)$$

Using the logarithmic identity $\ln(a \cdot b) = \ln a + \ln b$:

$$\ln p(X \mid \theta) = \sum_Z q(Z) \ln \left( \frac{p(X, Z \mid \theta)}{q(Z)} \right) + \sum_Z q(Z) \ln \left( \frac{q(Z)}{p(Z \mid X, \theta)} \right)$$

We define the two fundamental components:
1. **The Variational Free Energy (Auxiliary Function):**
   $$F(q, \theta) \triangleq \sum_Z q(Z) \ln \left( \frac{p(X, Z \mid \theta)}{q(Z)} \right) = \mathbb{E}_q[\ln p(X, Z \mid \theta)] + \mathcal{H}(q)$$
2. **The Kullback-Leibler Divergence:**
   $$D_{\text{KL}}\big( q(Z) \parallel p(Z \mid X, \theta) \big) \triangleq \sum_Z q(Z) \ln \left( \frac{q(Z)}{p(Z \mid X, \theta)} \right)$$

This yields the exact **Evidence Decomposition Identity**:
$$\boxed{\ln p(X \mid \theta) = F(q, \theta) + D_{\text{KL}}\big( q(Z) \parallel p(Z \mid X, \theta) \big)}$$

```text
================================================================================
           SURROGATE FUNCTION Q(θ | θ^t) TANGENT TO MARGINAL EVIDENCE
================================================================================

  Log-Likelihood ▲                     True Log-Likelihood ln p(X | θ)
                 │                                .-------.
                 │                             .-'         '-.
                 │                          .-'               '-.
                 │                       .-'                     '-.
                 │                    .-'                           '-.
                 │                 .-'                                 '-.
                 │              .-'       Surrogate Bound Q(θ | θ^t)      '-.
                 │           .-'         .----.                              '-.
                 │        .-'         .-'      '-.
                 │     .-'         .-'            '-.
                 │  .-'         .-'                  '-.  (M-Step Peak!)
                 └─┴───────────┴────────────────────────┴────────────────────► θ
                               θ^(t)                    θ^(t+1)
================================================================================
```

*What to notice from the diagram:*
1. By Gibbs' inequality, $D_{\text{KL}}(q \parallel p) \ge 0$, which proves that $F(q, \theta) \le \ln p(X \mid \theta)$ is a strict lower bound.
2. In the **E-step**, setting $q(Z) = p(Z \mid X, \theta^{(t)})$ drives $D_{\text{KL}} = 0$, making the lower bound touch the true log-likelihood surface at $\theta^{(t)}$ with zero gap.
3. In the **M-step**, maximizing the surrogate bound $Q(\theta \mid \theta^{(t)})$ with respect to $\theta$ produces $\theta^{(t+1)}$, pulling the true marginal likelihood upward!

---

## 5. Why choose this tool for this problem?

| Characteristic | Expectation-Maximization (EM) | Direct Gradient Ascent on $\ln p(X)$ | K-Means Clustering (Hard EM) | Full Bayesian MCMC |
| :--- | :--- | :--- | :--- | :--- |
| **Optimization Type** | Coordinate ascent on lower bound | First-order gradient descent (SGD/Adam) | Coordinate descent on squared error | Stochastic Markov chain sampling |
| **Step Type** | **Closed-form analytical updates** | Step-size dependent iterative updates | Closed-form hard cluster updates | Stochastic sample collection |
| **Convergence Guarantee** | **Monotonic likelihood improvement** | May diverge or oscillate without tuning | Monotonic decrease of inertia | Asymptotically exact posterior |
| **Handling Latents** | Soft fractional responsibilities $\gamma_{ik} \in [0, 1]$ | No explicit posterior computation | Hard 0/1 assignments ($z_i \in \{0, 1\}$) | Draws posterior samples $z^{(s)} \sim p(z \mid x)$ |
| **Hyperparameters** | **Zero learning rate tuning required** | Requires learning rate, momentum, schedule | Zero learning rate tuning | Requires burn-in, step size, thinning |

### Concrete Counterexample: Likelihood Singularity in Direct Gradient Ascent

Suppose an engineer trains a 2-component GMM using unconstrained gradient ascent directly on $\ln p(X \mid \theta)$:
1. A Gaussian component accidentally centers on a single isolated data point $x_n$ (so $\mu_k = x_n$).
2. The optimizer computes the gradient with respect to $\sigma_k$:
   $$\frac{\partial \ln p(x_n)}{\partial \sigma_k} = -\frac{1}{\sigma_k} + \frac{(x_n - \mu_k)^2}{\sigma_k^3} = -\frac{1}{\sigma_k} < 0$$
3. To maximize likelihood, the optimizer drives $\sigma_k \to 0$. As $\sigma_k \to 0$, the Gaussian density explodes:
   $$\mathcal{N}(x_n \mid x_n, \sigma_k^2) = \frac{1}{\sigma_k \sqrt{2\pi}} \to +\infty$$
4. The marginal log-likelihood diverges to $+\infty$, causing floating-point overflow followed by `NaN` gradients.
5. In EM, because parameter updates are decoupled into closed-form weighted averages, we can enforce an explicit **variance floor** $\sigma_k^2 \leftarrow \max(\sigma_k^2, \epsilon_{\text{floor}})$ or add a conjugate Inverse-Gamma prior (MAP EM) to mathematically eliminate singularities.

---

## 6. Strengthen the intuition and mark its limits

```text
================================================================================
          THE CLASSROOM SURVEYOR: PHYSICAL INTUITION OF THE EM CYCLE
================================================================================
 STEP 0: Unlabeled test scores on the table: [1.0, 2.0, 7.0]
         Initial Classroom Averages: Class 1 = 2.0,  Class 2 = 6.0
 
 STEP 1 (E-STEP): GUESS CLASSROOMS BASED ON CURRENT AVERAGES
         • Student 1.0 is closest to 2.0 ──► 99.97% Class 1,  0.03% Class 2
         • Student 2.0 is at 2.0         ──► 99.97% Class 1,  0.03% Class 2
         • Student 7.0 is closest to 6.0 ──►  0.03% Class 1, 99.97% Class 2
 
 STEP 2 (M-STEP): MOVE CLASSROOM AVERAGES TO MATCH STUDENT GUESSES
         • New Class 1 Average = (0.9997*1.0 + 0.9997*2.0) / (0.9997+0.9997) = 1.50
         • New Class 2 Average = (0.0003*1.0 + 0.0003*2.0 + 0.9997*7.0) / 1.0 = 7.00
 
 STEP 3: REPEAT UNTIL CLASSROOM AVERAGES STOP MOVING!
================================================================================
```

### Mechanical Mapping: Intuition to Mathematics and Implementation

| Physical Intuition / Metaphor | Mathematical Operation | Software / Hardware Implementation | Failure Mode / Boundary Condition |
| :--- | :--- | :--- | :--- |
| **Guessing student's classroom** | Posterior responsibility $\gamma_{ik} = p(z_i = k \mid x_i)$ | Softmax over log-densities in E-step | Point equidistant between clusters gets $\gamma_{ik} = 0.5$, high uncertainty |
| **Re-computing class average** | Closed-form centroid $\mu_k = \frac{1}{N_k}\sum_i \gamma_{ik} x_i$ | Vectorized reduction along sample axis | If cluster receives zero points ($N_k = 0$), division by zero occurs |
| **Classroom size proportion** | Mixture weight $\pi_k = N_k / N$ | Sum of responsibilities normalized by $N$ | Must satisfy unit simplex constraint $\sum \pi_k = 1$ |
| **Surrogate stepping stone** | Auxiliary bound $Q(\theta \mid \theta^{(t)})$ | Expected complete-data log-likelihood | Touches true likelihood with zero slope difference |

### Where this analogy stops working

1. **Local Optima and Saddle Points:** In the classroom analogy, moving averages feels so intuitive that one might assume EM always finds the single "true" grouping. In reality, the marginal log-likelihood surface is highly non-convex. EM guarantees **monotonic improvement**, but it frequently gets trapped in sub-optimal local maxima or saddle points depending heavily on initial guesses.
2. **Infinite Density Singularities:** Real classrooms cannot have zero students or zero variance. But in a mathematical Gaussian mixture, if a cluster centers on exactly one data point and variance is unconstrained, its density approaches $+\infty$. The algorithm can "cheat" by collapsing one cluster to zero width rather than discovering true groupings.

---

## 7. Terms worth keeping straight

### Core Terminology Reference Table

| Term | Pronunciation | Plain-English Meaning | Formal Definition & Conditions |
| :--- | :--- | :--- | :--- |
| **Incomplete-Data Likelihood** | “in-kum-PLEET DAY-tuh” | Observed marginal log-likelihood with sum inside logarithm | $\ln p(X \mid \theta) = \sum_{i=1}^N \ln \sum_{k=1}^K \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$. Intractable directly. |
| **Complete-Data Likelihood** | “kum-PLEET DAY-tuh” | Fictitious joint log-likelihood assuming latent labels known | $\ln p(X, Z \mid \theta) = \sum_{i=1}^N \sum_{k=1}^K z_{ik} \ln [\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)]$. Linear in $z$. |
| **Responsibility ($\gamma_{ik}$)** | “ree-spon-sih-BIL-ih-tee” | Posterior probability that cluster $k$ generated point $x_i$ | $\gamma_{ik} = p(z_i = k \mid x_i, \theta^{(t)}) = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$. |
| **E-step (Expectation)** | “EE-step” | Computes expected complete log-likelihood given current parameters | $Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z \mid X, \theta^{(t)}} [\ln p(X, Z \mid \theta)]$. Forms tight lower bound. |
| **M-step (Maximization)** | “EM-step” | Updates parameters to maximize the auxiliary $Q$-function | $\theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})$. Closed-form in exponential families. |
| **Soft EM** | “soft ee-em” | Probabilistic assignment of data points to clusters | Fractional responsibilities $\gamma_{ik} \in (0, 1)$ weighted across all components. |
| **Hard EM** | “hard ee-em” | Winner-take-all assignment to the single most probable cluster | $\gamma_{ik} \in \{0, 1\}$ with $\gamma_{ik} = 1$ iff $k = \arg\max_j \gamma_{ij}$ (reduces to K-Means). |
| **Variance Collapse** | “VAIR-ee-uns kuh-LAPS” | Singularity where a Gaussian centers on one point with $\sigma^2 \to 0$ | Density $\mathcal{N}(x_n \mid x_n, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \to +\infty$; objective diverges. |

### Confused Pairs Distinction Breakdown

1. **Incomplete-Data vs. Complete-Data Log-Likelihood**:
   - *Core Distinction:* Incomplete-data likelihood contains an intractable sum inside the logarithm; complete-data likelihood has the sum outside the logarithm.
   - *Common Confusion:* Trying to directly optimize incomplete likelihood with gradient descent without recognizing the non-convex multi-modal landscape.
   - *Rule of Thumb:* If the latent variable is unknown, optimize via the complete-data auxiliary lower bound (EM).
- **Incomplete-Data Log-Likelihood $\ln p(X \mid \theta)$:** The real-world objective. Contains an intractable sum inside the log: $\sum_i \ln \sum_k p(x_i, z_i = k)$.
- **Complete-Data Log-Likelihood $\ln p(X, Z \mid \theta)$:** The hypothetical objective if latent assignments $Z$ were known. The logarithm acts directly on exponential family terms without an intervening sum, making maximization trivial.

### 2. Soft EM (Standard GMM) vs. Hard EM (K-Means)
- **Soft EM:** Computes fractional posterior responsibilities $\gamma_{ik} \in [0, 1]$. Every data point contributes fractionally to every cluster.
- **Hard EM:** Forces responsibilities to binary indicator values: $\gamma_{ik} \in \{0, 1\}$ by assigning each point strictly to its $\arg\max$. K-Means is the exact zero-variance limit ($\sigma^2 \to 0$) of Soft EM.

### 3. Monotonic Likelihood Increase vs. Parameter Convergence
- **Likelihood Monotonicity:** $\ln p(X \mid \theta^{(t+1)}) \ge \ln p(X \mid \theta^{(t)})$ holds at **every single step**. The objective value never decreases.
- **Parameter Convergence:** The parameters $\theta^{(t)}$ converge to a stationary critical point ($\nabla_\theta \ln p = 0$), but this critical point is not guaranteed to be the global maximum.

### 4. Jensen's Lower Bound vs. Variational ELBO
- **Jensen's Lower Bound in EM:** Evaluated with $q(Z) = p(Z \mid X, \theta^{(t)})$, which is exact and tractable for discrete latent models (GMMs, HMMs).
- **Variational ELBO in Deep Learning:** Used when $p(Z \mid X, \theta)$ is continuous and intractable (as in VAEs), requiring an amortized neural network $q_\phi(z \mid x)$ to approximate the posterior.

---

## 8. Work through the mathematics and its conditions

### Theorem 1: Dempster-Laird-Rubin Monotonic Convergence Theorem (1977)

**Statement:** Let $\{\theta^{(t)}\}_{t=0}^\infty$ be a sequence of parameter estimates generated by the EM algorithm, where $\theta^{(t+1)} = \arg\max_\theta Q(\theta \mid \theta^{(t)})$. Then the marginal data log-likelihood is monotonically non-decreasing at every iteration:
$$\ln p(X \mid \theta^{(t+1)}) \ge \ln p(X \mid \theta^{(t)})$$

**Proof:**
1. From conditional probability, express the marginal likelihood as:
   $$p(X \mid \theta) = \frac{p(X, Z \mid \theta)}{p(Z \mid X, \theta)}$$
   Taking natural logarithms on both sides:
   $$\ln p(X \mid \theta) = \ln p(X, Z \mid \theta) - \ln p(Z \mid X, \theta)$$

2. Take the mathematical expectation of both sides under the posterior distribution of latent variables conditioned on the current parameters $\theta^{(t)}$, namely $q(Z) = p(Z \mid X, \theta^{(t)})$:
   $$\mathbb{E}_q[\ln p(X \mid \theta)] = \mathbb{E}_q[\ln p(X, Z \mid \theta)] - \mathbb{E}_q[\ln p(Z \mid X, \theta)]$$

3. Because $\ln p(X \mid \theta)$ does not depend on latent variables $Z$, its expectation under $q(Z)$ is simply itself:
   $$\ln p(X \mid \theta) = Q(\theta \mid \theta^{(t)}) - H(\theta \mid \theta^{(t)})$$
   where:
   $$Q(\theta \mid \theta^{(t)}) \triangleq \sum_Z p(Z \mid X, \theta^{(t)}) \ln p(X, Z \mid \theta)$$
   $$H(\theta \mid \theta^{(t)}) \triangleq \sum_Z p(Z \mid X, \theta^{(t)}) \ln p(Z \mid X, \theta)$$

4. Write the change in true log-likelihood between candidate $\theta$ and $\theta^{(t)}$:
   $$\ln p(X \mid \theta) - \ln p(X \mid \theta^{(t)}) = \Big( Q(\theta \mid \theta^{(t)}) - Q(\theta^{(t)} \mid \theta^{(t)}) \Big) - \Big( H(\theta \mid \theta^{(t)}) - H(\theta^{(t)} \mid \theta^{(t)}) \Big)$$

5. Analyze the difference in the entropy term $H$:
   $$H(\theta \mid \theta^{(t)}) - H(\theta^{(t)} \mid \theta^{(t)}) = \sum_Z p(Z \mid X, \theta^{(t)}) \ln \left( \frac{p(Z \mid X, \theta)}{p(Z \mid X, \theta^{(t)})} \right) = - D_{\text{KL}}\Big( p(Z \mid X, \theta^{(t)}) \parallel p(Z \mid X, \theta) \Big)$$

6. By Gibbs' inequality, the KL divergence is strictly non-negative:
   $$D_{\text{KL}}\Big( p(Z \mid X, \theta^{(t)}) \parallel p(Z \mid X, \theta) \Big) \ge 0 \implies -\Big( H(\theta \mid \theta^{(t)}) - H(\theta^{(t)} \mid \theta^{(t)}) \Big) \ge 0$$

7. Substituting this back into Step 4 yields the fundamental lower bound inequality:
   $$\boxed{\ln p(X \mid \theta) - \ln p(X \mid \theta^{(t)}) \ge Q(\theta \mid \theta^{(t)}) - Q(\theta^{(t)} \mid \theta^{(t)})}$$

8. In the M-step, $\theta^{(t+1)}$ is chosen to maximize $Q(\theta \mid \theta^{(t)})$, so $Q(\theta^{(t+1)} \mid \theta^{(t)}) \ge Q(\theta^{(t)} \mid \theta^{(t)})$. Consequently:
   $$\ln p(X \mid \theta^{(t+1)}) - \ln p(X \mid \theta^{(t)}) \ge 0 \implies \boxed{\ln p(X \mid \theta^{(t+1)}) \ge \ln p(X \mid \theta^{(t)})}$$
   The marginal log-likelihood is monotonically non-decreasing at every iteration. $\blacksquare$

---

### Theorem 2: Monotone Sequence Convergence to a Stationary Limit

**Statement:** If covariance matrices are bounded by a variance floor $\lambda_{\min}(\Sigma_k) \ge \epsilon_{\text{floor}} > 0$, the sequence of log-likelihood values $\{\ell_t\}_{t=0}^\infty$ where $\ell_t = \ln p(X \mid \theta^{(t)})$ converges to a stationary limit value $\ell^* < \infty$, and limit points $\theta^*$ satisfy first-order stationarity $\nabla_\theta \ln p(X \mid \theta^*) = 0$.

**Proof:**
1. By Theorem 1, $\{\ell_t\}$ is monotonically non-decreasing: $\ell_0 \le \ell_1 \le \dots \le \ell_t \le \ell_{t+1}$.
2. With $\lambda_{\min}(\Sigma_k) \ge \epsilon_{\text{floor}} > 0$, the Gaussian density is bounded above by $M = (2\pi)^{-D/2} \epsilon_{\text{floor}}^{-D/2} < \infty$.
3. The dataset log-likelihood is bounded above: $\ell(\theta) \le N \ln M \equiv L_{\max} < \infty$.
4. By the Monotone Convergence Theorem for real numbers, any bounded monotonic sequence converges to its supremum: $\lim_{t \to \infty} \ell_t = \sup_{t \ge 0} \ell_t = \ell^* \le L_{\max}$.
5. Because $\theta^*$ maximizes $Q(\theta \mid \theta^*)$, $\left. \nabla_\theta Q(\theta \mid \theta^*) \right|_{\theta^*} = 0$. Since $\nabla_\theta H(\theta \mid \theta^*)|_{\theta^*} = 0$ (attaining its minimum of 0 at $\theta^*$), differentiating $\ln p = Q - H$ gives $\boxed{\nabla_\theta \ln p(X \mid \theta^*) = 0}$. $\blacksquare$

---

### M-Step Derivation: Lagrange Multipliers for Mixture Proportions

The mixture weights must satisfy $\sum_{k=1}^K \pi_k = 1$ and $\pi_k \ge 0$.  
We formulate the Lagrangian:

$$\mathcal{L}_{\text{Lag}}(\pi, \lambda) = \sum_{i=1}^N \sum_{k=1}^K \gamma_{ik} \ln \pi_k + \lambda \left( 1 - \sum_{k=1}^K \pi_k \right)$$

Taking the partial derivative with respect to $\pi_k$ and setting to zero:

$$\frac{\partial \mathcal{L}_{\text{Lag}}}{\partial \pi_k} = \sum_{i=1}^N \frac{\gamma_{ik}}{\pi_k} - \lambda = 0 \implies \pi_k = \frac{1}{\lambda} \sum_{i=1}^N \gamma_{ik} = \frac{N_k}{\lambda}$$

Summing both sides over all $K$ components:

$$\sum_{k=1}^K \pi_k = \frac{1}{\lambda} \sum_{k=1}^K N_k = 1 \implies \lambda = \sum_{k=1}^K \sum_{i=1}^N \gamma_{ik} = \sum_{i=1}^N 1 = N$$

Substituting $\lambda = N$ back gives the exact closed-form update:

$$\boxed{\pi_k^* = \frac{N_k}{N}}$$

### Hardware and Computational Realities: Numerical Underflow, Log-Sum-Exp, and GPU Memory Bandwidth

In production implementations, calculating responsibilities $\gamma_{ik}$ in raw probability space leads to catastrophic floating-point underflow. When dimension $D$ is large (e.g., $D > 100$), the Mahalanobis distance $(x - \mu)^\top \Sigma^{-1} (x - \mu)$ can easily exceed $1000$, causing $\exp(-500) = 0.0$ in IEEE-754 `float32`.

1. **The Log-Sum-Exp Trick:**
   Responsibilities are strictly evaluated in log-space:
   $$\ln \gamma_{ik} = \ln \pi_k + \ln \mathcal{N}(x_i \mid \mu_k, \Sigma_k) - \operatorname{LSE}_j \left( \ln \pi_j + \ln \mathcal{N}(x_i \mid \mu_j, \Sigma_j) \right)$$
   where $\operatorname{LSE}(a_1, \dots, a_K) = m + \ln \sum_j \exp(a_j - m)$ with $m = \max_j a_j$. This guarantees that the largest exponent is exactly $0$, completely eliminating underflow and NaN values.

2. **GPU Memory Bandwidth vs. Compute Bound:**
   Computing pairwise distances between $N$ data vectors and $K$ cluster centers requires an $N \times K \times D$ tensor. For $N = 10^6, K = 1024, D = 128$ in float32, this tensor consumes $512\text{ GB}$ of memory—far exceeding GPU VRAM. Modern GPU implementations tile the computation along the $N$ dimension and fuse the distance evaluation, log-sum-exp reduction, and responsibility accumulation into a single Triton/CUDA kernel, eliminating intermediate global memory traffic.

3. **Covariance Regularization:**
   To prevent matrix inversion failure during Cholesky decomposition on GPUs, a small jitter constant $\epsilon I$ (e.g. $\epsilon = 10^{-6}$) is added to the diagonal: $\tilde{\Sigma}_k = \Sigma_k + \epsilon I$.

---

## 9. Calculate it by hand

### Worked Example: 1 Iteration of EM on 3 Data Points ($N=3, K=2$)

Using the exact toy numbers from Section 2:
- Observed data points: $x_1 = 1.0, \quad x_2 = 2.0, \quad x_3 = 7.0$ ($N=3$).
- Known cluster variances: $\sigma_1^2 = 1.0, \quad \sigma_2^2 = 1.0$.
- Initial parameters at $t=0$:
  $$\mu_1^{(0)} = 2.0, \quad \mu_2^{(0)} = 6.0, \quad \pi_1^{(0)} = 0.50, \quad \pi_2^{(0)} = 0.50$$

Recall 1D Gaussian density: $\mathcal{N}(x \mid \mu, 1.0) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2}\right)$, with $\frac{1}{\sqrt{2\pi}} \approx 0.398942$.

---

#### Step 1: E-Step — Evaluate Component Densities

1. **For point $x_1 = 1.0$:**
   - Under Cluster 1 ($\mu_1 = 2.0$): $(1.0 - 2.0)^2 = 1.0 \implies \mathcal{N}_1(1.0) = 0.398942 \times e^{-0.5} \approx \mathbf{0.241971}$
   - Under Cluster 2 ($\mu_2 = 6.0$): $(1.0 - 6.0)^2 = 25.0 \implies \mathcal{N}_2(1.0) = 0.398942 \times e^{-12.5} \approx \mathbf{0.0000015}$
2. **For point $x_2 = 2.0$:**
   - Under Cluster 1 ($\mu_1 = 2.0$): $(2.0 - 2.0)^2 = 0.0 \implies \mathcal{N}_1(2.0) = 0.398942 \times e^0 = \mathbf{0.398942}$
   - Under Cluster 2 ($\mu_2 = 6.0$): $(2.0 - 6.0)^2 = 16.0 \implies \mathcal{N}_2(2.0) = 0.398942 \times e^{-8.0} \approx \mathbf{0.000134}$
3. **For point $x_3 = 7.0$:**
   - Under Cluster 1 ($\mu_1 = 2.0$): $(7.0 - 2.0)^2 = 25.0 \implies \mathcal{N}_1(7.0) = 0.398942 \times e^{-12.5} \approx \mathbf{0.0000015}$
   - Under Cluster 2 ($\mu_2 = 6.0$): $(7.0 - 6.0)^2 = 1.0 \implies \mathcal{N}_2(7.0) = 0.398942 \times e^{-0.5} \approx \mathbf{0.241971}$

---

#### Step 2: E-Step — Compute Posterior Responsibilities $\gamma_{ik}$

With $\pi_1 = \pi_2 = 0.50$, the prior weights cancel out: $\gamma_{ik} = \frac{\mathcal{N}_k(x_i)}{\mathcal{N}_1(x_i) + \mathcal{N}_2(x_i)}$.

1. **For $x_1 = 1.0$:**
   $$\gamma_{11} = \frac{0.241971}{0.241971 + 0.0000015} \approx \mathbf{0.999994}, \quad \gamma_{12} = 1 - \gamma_{11} \approx \mathbf{0.000006}$$
2. **For $x_2 = 2.0$:**
   $$\gamma_{21} = \frac{0.398942}{0.398942 + 0.000134} \approx \mathbf{0.999664}, \quad \gamma_{22} = 1 - \gamma_{21} \approx \mathbf{0.000336}$$
3. **For $x_3 = 7.0$:**
   $$\gamma_{31} = \frac{0.0000015}{0.0000015 + 0.241971} \approx \mathbf{0.000006}, \quad \gamma_{32} = 1 - \gamma_{31} \approx \mathbf{0.999994}$$

---

#### Step 3: M-Step — Effective Cluster Counts $N_k$

$$N_1 = \gamma_{11} + \gamma_{21} + \gamma_{31} = 0.999994 + 0.999664 + 0.000006 = \mathbf{1.999664}$$
$$N_2 = \gamma_{12} + \gamma_{22} + \gamma_{32} = 0.000006 + 0.000336 + 0.999994 = \mathbf{1.000336}$$
$$\text{Total: } N_1 + N_2 = 1.999664 + 1.000336 = \mathbf{3.000000} = N$$

---

#### Step 4: M-Step — Update Mixture Weights and Means

1. **Updated Mixture Weights:**
   $$\pi_1^{(1)} = \frac{N_1}{N} = \frac{1.999664}{3} \approx \mathbf{0.666555} \quad (\approx 2/3)$$
   $$\pi_2^{(1)} = \frac{N_2}{N} = \frac{1.000336}{3} \approx \mathbf{0.333445} \quad (\approx 1/3)$$
2. **Updated Cluster Centroids:**
   $$\mu_1^{(1)} = \frac{\gamma_{11} x_1 + \gamma_{21} x_2 + \gamma_{31} x_3}{N_1} = \frac{0.999994(1.0) + 0.999664(2.0) + 0.000006(7.0)}{1.999664}$$
   $$\mu_1^{(1)} = \frac{0.999994 + 1.999328 + 0.000042}{1.999664} = \frac{2.999364}{1.999664} \approx \mathbf{1.499931} \quad (\approx 1.5000)$$

   $$\mu_2^{(1)} = \frac{\gamma_{12} x_1 + \gamma_{22} x_2 + \gamma_{32} x_3}{N_2} = \frac{0.000006(1.0) + 0.000336(2.0) + 0.999994(7.0)}{1.000336}$$
   $$\mu_2^{(1)} = \frac{0.000006 + 0.000672 + 6.999958}{1.000336} = \frac{7.000636}{1.000336} \approx \mathbf{6.998284} \quad (\approx 7.0000)$$

---

#### Step 5: Verify Monotonic Likelihood Increase

1. **Initial Log-Likelihood $\ell(\theta^{(0)})$:**
   - $p(x_1) = 0.5(0.241971) + 0.5(0.0000015) \approx 0.120986 \implies \ln p(x_1) \approx -2.1121$
   - $p(x_2) = 0.5(0.398942) + 0.5(0.000134) \approx 0.199538 \implies \ln p(x_2) \approx -1.6118$
   - $p(x_3) = 0.5(0.0000015) + 0.5(0.241971) \approx 0.120986 \implies \ln p(x_3) \approx -2.1121$
   $$\ell(\theta^{(0)}) = -2.1121 - 1.6118 - 2.1121 = \mathbf{-5.8360\text{ nats}}$$
2. **Updated Log-Likelihood $\ell(\theta^{(1)})$:**
   - At $\mu_1 = 1.50, \mu_2 = 7.00, \pi_1 = 2/3, \pi_2 = 1/3$:
   - $x_1 = 1.0$: $\mathcal{N}_1(1.0) = 0.398942 e^{-0.125} \approx 0.352065 \implies p(x_1) \approx \frac{2}{3}(0.352065) \approx 0.234710 \implies \ln p(x_1) \approx -1.4494$
   - $x_2 = 2.0$: $\mathcal{N}_1(2.0) = 0.398942 e^{-0.125} \approx 0.352065 \implies p(x_2) \approx \frac{2}{3}(0.352065) \approx 0.234710 \implies \ln p(x_2) \approx -1.4494$
   - $x_3 = 7.0$: $\mathcal{N}_2(7.0) = 0.398942 e^0 = 0.398942 \implies p(x_3) \approx \frac{1}{3}(0.398942) \approx 0.132981 \implies \ln p(x_3) \approx -2.0176$
   $$\ell(\theta^{(1)}) = -1.4494 - 1.4494 - 2.0176 = \mathbf{-4.9164\text{ nats}}$$
3. **Ascent Delta:**
   $$\Delta \ell = \ell(\theta^{(1)}) - \ell(\theta^{(0)}) = -4.9164 - (-5.8360) = \mathbf{+0.9196\text{ nats}} > 0$$
   The true marginal log-likelihood improved by nearly $1.0$ nat in a single iteration!

### Second Case: Singular Boundary Evaluation and Variance Collapse

To observe the primary failure mode of unconstrained maximum likelihood in mixture models, examine what occurs when a component mean coincides with a single data point ($x_n = \mu_k$) as variance collapses ($\sigma_k^2 \to 0$):

1. **Singular Density Evaluation:**
   Suppose component 1 centers on $x_1 = 1.0$ with $\mu_1 = 1.0$. The likelihood contribution of $x_1$ under component 1 is:
   $$\mathcal{N}(x_1 = 1.0 \mid \mu_1 = 1.0, \sigma_1^2) = \frac{1}{\sqrt{2\pi \sigma_1^2}} \exp\left(-\frac{(1.0 - 1.0)^2}{2\sigma_1^2}\right) = \frac{1}{\sqrt{2\pi \sigma_1^2}}$$
   - If $\sigma_1^2 = 1.0$: $\mathcal{N} = \frac{1}{\sqrt{2\pi}} \approx 0.3989$
   - If $\sigma_1^2 = 10^{-4}$: $\mathcal{N} = \frac{1}{\sqrt{2\pi \times 10^{-4}}} = \frac{100}{\sqrt{2\pi}} \approx 39.894$
   - If $\sigma_1^2 = 10^{-8}$: $\mathcal{N} = \frac{1}{\sqrt{2\pi \times 10^{-8}}} = \frac{10^4}{\sqrt{2\pi}} \approx 3989.4$
   - As $\sigma_1^2 \to 0$: $\lim_{\sigma_1^2 \to 0} \mathcal{N}(x_1 \mid \mu_1, \sigma_1^2) = +\infty$

2. **Marginal Log-Likelihood Explosion:**
   Because the total marginal log-likelihood is:
   $$\ell(\theta) = \sum_{i=1}^N \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i \mid \mu_k, \sigma_k^2) \right)$$
   the single term for $i=1$ contains $\ln(\pi_1 \cdot \infty) = +\infty$. Even if the remaining $N-1$ points have finite likelihood under other components, the total objective $\ell(\theta) \to +\infty$.

3. **Mathematical Consequence:**
   The maximum likelihood objective for continuous mixture models is **unbounded above**. The global maximum is not a valid density estimator, but a pathological singularity. This proves why numerical implementations must enforce a variance lower bound ($\sigma_k^2 \ge \sigma_{\min}^2$) or place an inverse-Gamma / Wishart prior over covariance matrices (MAP/Bayesian estimation).

---

## 10. Connect the concept to an actual system

```text
================================================================================
                    EM ACROSS PRODUCTION GENERATIVE SYSTEMS
================================================================================
 1. SPEECH RECOGNITION (HMM Baum-Welch)       2. TOKEN CLUSTERING IN VQ-VAE
 Forward-Backward E-Step + Transition M-Step  Soft GMM Codebook Initialization
 ┌────────────────────────────────────────┐   ┌────────────────────────────────┐
 │ E-Step: Forward-backward lattice       │   │ Clusters 100M continuous audio │
 │ M-Step: Closed-form transition counts  │   │ or visual vectors into 1024    │
 │ Decodes phonemes from noisy waveforms  │   │ discrete codebook tokens       │
 └────────────────────────────────────────┘   └────────────────────────────────┘
================================================================================
```

*What to notice from the diagram:*
1. The Baum-Welch algorithm is the exact dynamic-programming implementation of EM for sequential Hidden Markov Models.
2. In discrete tokenizers (such as AudioCraft and VQ-VAE), GMM EM is used to initialize codebooks cleanly, preventing dead codes.

| Mathematical Object | Role in Toy Example | Real Production System Counterpart | Hardware / Scale Approximation |
| :--- | :--- | :--- | :--- |
| **Latent State $z_i$** | Classroom index $z_i \in \{1, 2\}$ | Phonetic state in HMM or discrete acoustic codebook index | Scaled to $K = 1,024$ to $4,096$ codebook entries |
| **Responsibility $\gamma_{ik}$** | $3 \times 2$ matrix of fractional weights | Forward-backward trellis probabilities $\alpha_t(j) \beta_t(j) / P(O)$ | Computed in log-space via `torch.logsumexp` to avoid floating-point underflow |
| **Surrogate $Q(\theta \mid \theta^{(t)})$** | Scalar expected complete log-likelihood | Fused GPU kernel accumulating expected sufficient statistics | Fused reduction across batch dimensions on GPU Tensor Cores |
| **Centroid Update $\mu_k^*$** | Closed-form weighted average of exam scores | Codebook vector update or Gaussian emission mean update | Enforces variance floor $\Sigma_k + \epsilon I$ to eliminate likelihood singularities |

We have mapped the architectural connections. Next, we verify these formulations with executable Python and PyTorch scripts.

---

## 11. Verify the idea with a small experiment

We implement the **Dual-Stage Code Architecture**:
- **Stage 1 (Pure Python):** Standard library implementation with zero external dependencies, computing 1 iteration of GMM EM on $X = [1.0, 2.0, 7.0]$ matching Section 9 exactly, and verifying monotonic likelihood improvement with assertions.
- **Stage 2 (Production PyTorch):** Vectorized PyTorch implementation using `torch.logsumexp`, verifying surrogate gradient stationarity ($\nabla_\mu Q = 0$) and testing monotonic ascent.

```python
"""
Expectation-Maximization (EM) Dual-Stage Verification Suite
==========================================================
Part A: Pure Python standard library (built-in math only, zero dependencies).
Part B: PyTorch industrial verification suite with log-sum-exp stabilization,
        autograd surrogate stationarity checks, and monotonic ascent verification.
"""

import math
import sys

# Ensure UTF-8 output on all consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 80)
print("PART A: PURE PYTHON STDLIB GMM EM VERIFICATION")
print("=" * 80)

# ─── 1. Pure Python 1D Gaussian Density Function ───
def pure_gaussian_pdf(x, mu, sigma2):
    return (1.0 / math.sqrt(2.0 * math.pi * sigma2)) * math.exp(-0.5 * ((x - mu) ** 2) / sigma2)

# Dataset from Section 2 and Section 9
X = [1.0, 2.0, 7.0]
N = len(X)
K = 2

# Initial parameters at t = 0
mu = [2.0, 6.0]
pi = [0.5, 0.5]
var = [1.0, 1.0]

def compute_log_likelihood(data, weights, means, variances):
    total_ll = 0.0
    for x_val in data:
        p_x = sum(weights[k] * pure_gaussian_pdf(x_val, means[k], variances[k]) for k in range(K))
        total_ll += math.log(p_x)
    return total_ll

ll_initial = compute_log_likelihood(X, pi, mu, var)
print(f"1. Initial Parameters: mu = {mu}, pi = {pi}")
print(f"   * Initial Log-Likelihood: {ll_initial:.4f} nats (Expected: ~ -5.8360)")
assert math.isclose(ll_initial, -5.8360, abs_tol=1e-3)

# ─── 2. E-Step: Posterior Responsibilities gamma_ik ───
gamma = []
for x_val in X:
    densities = [pi[k] * pure_gaussian_pdf(x_val, mu[k], var[k]) for k in range(K)]
    total_density = sum(densities)
    gamma.append([d / total_density for d in densities])

print("\n2. E-Step Responsibilities (gamma_ik):")
for i, g_row in enumerate(gamma):
    print(f"   * Point x_{i+1} = {X[i]:.1f}: gamma_{i+1},1 = {g_row[0]:.6f}, gamma_{i+1},2 = {g_row[1]:.6f}")

assert math.isclose(gamma[0][0], 0.999994, abs_tol=1e-4)
assert math.isclose(gamma[1][0], 0.999664, abs_tol=1e-3)
assert math.isclose(gamma[2][1], 0.999994, abs_tol=1e-4)

# ─── 3. M-Step: Closed-Form Updates ───
Nk = [sum(gamma[i][k] for i in range(N)) for k in range(K)]
pi_next = [Nk[k] / N for k in range(K)]
mu_next = [sum(gamma[i][k] * X[i] for i in range(N)) / Nk[k] for k in range(K)]

print("\n3. M-Step Updated Parameters:")
print(f"   * Effective Counts Nk: [{Nk[0]:.4f}, {Nk[1]:.4f}]")
print(f"   * Updated Weights pi:  [{pi_next[0]:.4f}, {pi_next[1]:.4f}] (Expected: [0.6666, 0.3334])")
print(f"   * Updated Means mu:    [{mu_next[0]:.4f}, {mu_next[1]:.4f}] (Expected: [1.5000, 7.0000])")

assert math.isclose(pi_next[0], 2.0 / 3.0, abs_tol=1e-3)
assert math.isclose(mu_next[0], 1.5000, abs_tol=1e-3)
assert math.isclose(mu_next[1], 7.0000, abs_tol=5e-3)

# ─── 4. Monotonic Likelihood Increase ───
ll_next = compute_log_likelihood(X, pi_next, mu_next, var)
delta_ll = ll_next - ll_initial
print("\n4. Monotonic Likelihood Verification:")
print(f"   * Updated Log-Likelihood: {ll_next:.4f} nats (Expected: ~ -4.9164)")
print(f"   * Ascent Delta:           {delta_ll:+.4f} nats (Expected: ~ +0.9196)")

assert delta_ll > 0.0, "Likelihood failed to increase monotonically!"
assert math.isclose(delta_ll, 0.9196, abs_tol=1e-2)
print("Part A Pure Python Suite: ALL CHECKS PASSED [OK]")

print("\n" + "=" * 80)
print("PART B: PYTORCH INDUSTRIAL AUTOGRAD & STATIONARITY SUITE")
print("=" * 80)

import torch

X_torch = torch.tensor([1.0, 2.0, 7.0], dtype=torch.float64)
N_torch = len(X_torch)
K_torch = 2

pi_torch = torch.tensor([0.5, 0.5], dtype=torch.float64)
mu_torch = torch.tensor([2.0, 6.0], dtype=torch.float64)
var_torch = torch.tensor([1.0, 1.0], dtype=torch.float64)

def torch_gaussian_pdf(x_tensor, mu_val, var_val):
    inv_denom = 1.0 / torch.sqrt(2.0 * torch.pi * var_val)
    exponent = -0.5 * ((x_tensor - mu_val) ** 2) / var_val
    return inv_denom * torch.exp(exponent)

def torch_marginal_ll(x_data, weights, means, variances):
    densities = torch.stack(
        [weights[k] * torch_gaussian_pdf(x_data, means[k], variances[k]) for k in range(K_torch)],
        dim=1
    )
    return torch.sum(torch.log(densities.sum(dim=1)))

ll_init_torch = torch_marginal_ll(X_torch, pi_torch, mu_torch, var_torch)
print(f"1. Initial PyTorch Log-Likelihood: {ll_init_torch.item():.4f} nats")

# E-step
densities_tensor = torch.stack(
    [pi_torch[k] * torch_gaussian_pdf(X_torch, mu_torch[k], var_torch[k]) for k in range(K_torch)],
    dim=1
)
gamma_torch = densities_tensor / densities_tensor.sum(dim=1, keepdim=True)

# M-step
Nk_torch = gamma_torch.sum(dim=0)
pi_next_torch = Nk_torch / N_torch
mu_next_torch = (gamma_torch * X_torch.unsqueeze(1)).sum(dim=0) / Nk_torch

print(f"2. Analytical M-Step Centroids: mu_next = {mu_next_torch.tolist()}")

# Verify Stationarity via Autograd: dQ/dmu evaluated at mu_next must be ZERO
mu_candidate = mu_next_torch.clone().detach().requires_grad_(True)
Q_surrogate = torch.tensor(0.0, dtype=torch.float64)

for i in range(N_torch):
    for k in range(K_torch):
        ln_gauss = (
            -0.5 * torch.log(2.0 * torch.pi * var_torch[k])
            - 0.5 * ((X_torch[i] - mu_candidate[k]) ** 2) / var_torch[k]
        )
        Q_surrogate = Q_surrogate + gamma_torch[i, k] * (torch.log(pi_next_torch[k]) + ln_gauss)

Q_surrogate.backward()
surrogate_grad = mu_candidate.grad

print(f"3. Autograd Stationarity Verification:")
print(f"   * Surrogate Gradient at mu*: {surrogate_grad.tolist()}")
assert torch.allclose(surrogate_grad, torch.zeros_like(surrogate_grad), atol=1e-8),     f"Stationarity failed! Grad was {surrogate_grad}"
print("   * Stationarity Confirmed: Analytical M-Step matches exact peak (grad = 0) [OK]")

# Monotonicity check
ll_next_torch = torch_marginal_ll(X_torch, pi_next_torch, mu_next_torch, var_torch)
delta_torch = ll_next_torch - ll_init_torch
print(f"4. Monotonic Convergence Check:")
print(f"   * Initial LL: {ll_init_torch.item():.4f}, Next LL: {ll_next_torch.item():.4f}")
print(f"   * Ascent Delta: {delta_torch.item():+.4f} nats")
assert delta_torch.item() > 0.0, "Monotonic ascent failed in PyTorch!"

print("\n" + "=" * 80)
print("ALL EXPECTATION-MAXIMIZATION VERIFICATION TESTS PASSED SUCCESSFULLY! [OK]")
print("=" * 80)
```

*Expected output:*
```text
================================================================================
PART A: PURE PYTHON STDLIB GMM EM VERIFICATION
================================================================================
1. Initial Parameters: mu = [2.0, 6.0], pi = [0.5, 0.5]
   * Initial Log-Likelihood: -5.8360 nats (Expected: ~ -5.8360)

2. E-Step Responsibilities (gamma_ik):
   * Point x_1 = 1.0: gamma_1,1 = 0.999994, gamma_1,2 = 0.000006
   * Point x_2 = 2.0: gamma_2,1 = 0.999664, gamma_2,2 = 0.000336
   * Point x_3 = 7.0: gamma_3,1 = 0.000006, gamma_3,2 = 0.999994

3. M-Step Updated Parameters:
   * Effective Counts Nk: [1.9997, 1.0003]
   * Updated Weights pi:  [0.6666, 0.3334] (Expected: [0.6666, 0.3334])
   * Updated Means mu:    [1.4999, 6.9983] (Expected: [1.5000, 7.0000])

4. Monotonic Likelihood Verification:
   * Updated Log-Likelihood: -4.9164 nats (Expected: ~ -4.9164)
   * Ascent Delta:           +0.9196 nats (Expected: ~ +0.9196)
Part A Pure Python Suite: ALL CHECKS PASSED [OK]

================================================================================
PART B: PYTORCH INDUSTRIAL AUTOGRAD & STATIONARITY SUITE
================================================================================
1. Initial PyTorch Log-Likelihood: -5.8360 nats
2. Analytical M-Step Centroids: mu_next = [1.4999313264426544, 6.998284050518779]
3. Autograd Stationarity Verification:
   * Surrogate Gradient at mu*: [0.0, 0.0]
   * Stationarity Confirmed: Analytical M-Step matches exact peak (grad = 0) [OK]
4. Monotonic Convergence Check:
   * Initial LL: -5.8360, Next LL: -4.9164
   * Ascent Delta: +0.9196 nats

================================================================================
ALL EXPECTATION-MAXIMIZATION VERIFICATION TESTS PASSED SUCCESSFULLY! [OK]
================================================================================
```

---

## 12. Practise, compare, and debug

Attempt all five diagnostic exercises before inspecting the separated solutions.

1. **Recognize.** An engineer attempts to train a Gaussian Mixture Model using unconstrained gradient ascent directly on the incomplete data log-likelihood $\ln p(X \mid \theta)$. After 20 steps, the loss abruptly outputs `NaN`. What mathematical phenomenon occurred, and how does the EM formulation avoid it?
2. **Calculate.** In a 2-component Gaussian mixture model with $\pi_1 = \pi_2 = 0.50$, cluster parameters are $\mu_1 = 0.0, \sigma_1^2 = 1.0$ and $\mu_2 = 4.0, \sigma_2^2 = 1.0$. At observation $x_1 = 2.0$, calculate the responsibilities $\gamma_{11}$ and $\gamma_{12}$. If a second point $x_2 = 1.0$ is added with responsibilities $\gamma_{21} = 0.95$ and $\gamma_{22} = 0.05$, compute the updated mean $\mu_1^{(1)}$.
3. **Contrast.** Contrast K-Means clustering (Hard EM) with Gaussian Mixture Models (Soft EM). Specifically, what does K-Means assume about cluster covariances, and what happens to GMM responsibilities when the cluster variances approach zero ($\sigma_k^2 \to 0$)?
4. **Transfer.** Suppose all latent variables $Z$ in a dataset were completely observed. Show that the EM algorithm reduces to standard Maximum Likelihood Estimation in a single iteration without needing further cycles.
5. **Debug.** During GMM training, an implementation crashes in the M-step with:
   `ZeroDivisionError: division by zero` when updating $\mu_k = \frac{1}{N_k} \sum_i \gamma_{ik} x_i$.
   Diagnose the underlying data condition and prescribe the standard production safeguard.

---

### Separated Diagnostic Solutions

<details>
<summary>Click to view solution for Exercise 1</summary>

**Diagnosis:** The engineer encountered a **likelihood singularity**. When one Gaussian component drifts onto a single training sample ($x_n$), the gradient optimizer shrinks the component's variance $\sigma_k \to 0$, causing the density $\mathcal{N}(x_n \mid \mu_k, \sigma_k^2) \to +\infty$. The log-likelihood explodes to $+\infty$, producing numerical overflow followed by `NaN` in floating-point registers.

**Resolution:** In EM, engineers enforce a **variance floor** $\sigma_k^2 = \max(\sigma_k^2, \epsilon_{\text{floor}})$ or regularize covariance matrices via a diagonal ridge $\Sigma_k \leftarrow \Sigma_k + \epsilon I$.
</details>

<details>
<summary>Click to view solution for Exercise 2</summary>

**Calculation:**
1. At $x_1 = 2.0$, evaluate component densities:
   $$\mathcal{N}(2.0 \mid 0.0, 1.0) = \frac{1}{\sqrt{2\pi}} e^{-(2.0-0.0)^2 / 2} = \frac{1}{\sqrt{2\pi}} e^{-2.0} \approx 0.05399$$
   $$\mathcal{N}(2.0 \mid 4.0, 1.0) = \frac{1}{\sqrt{2\pi}} e^{-(2.0-4.0)^2 / 2} = \frac{1}{\sqrt{2\pi}} e^{-2.0} \approx 0.05399$$
   Because $x_1 = 2.0$ is equidistant between the two centers and $\pi_1 = \pi_2$:
   $$\gamma_{11} = \frac{0.5(0.05399)}{0.5(0.05399) + 0.5(0.05399)} = 0.5000, \qquad \gamma_{12} = 0.5000$$

2. Adding $x_2 = 1.0$ with $\gamma_{21} = 0.95$:
   Effective cluster weight:
   $$N_1 = \gamma_{11} + \gamma_{21} = 0.50 + 0.95 = 1.45$$
   Updated mean $\mu_1^{(1)}$:
   $$\mu_1^{(1)} = \frac{\gamma_{11} x_1 + \gamma_{21} x_2}{N_1} = \frac{0.50(2.0) + 0.95(1.0)}{1.45} = \frac{1.00 + 0.95}{1.45} = \frac{1.95}{1.45} \approx \mathbf{1.3448}$$
   The center shifts toward $x_2 = 1.0$, which carries 95% responsibility.
</details>

<details>
<summary>Click to view solution for Exercise 3</summary>

**Contrast:**
- **K-Means Covariance Assumption:** K-Means assumes isotropic, spherical clusters with identical variances ($\Sigma_k = \sigma^2 I$) across all components.
- **Limit Behavior as $\sigma^2 \to 0$:** In a GMM with shared spherical variance $\sigma^2 I$, the responsibility equation contains:
  $$\gamma_{ik} = \frac{\pi_k \exp(-\|x_i - \mu_k\|^2 / 2\sigma^2)}{\sum_j \pi_j \exp(-\|x_i - \mu_j\|^2 / 2\sigma^2)}$$
  As $\sigma^2 \to 0^+$, the softmax behaves as a hard argmin operator:
  $$\lim_{\sigma^2 \to 0^+} \gamma_{ik} = \begin{cases} 1 & \text{if } k = \arg\min_j \|x_i - \mu_j\|^2 \\ 0 & \text{otherwise} \end{cases}$$
  K-Means is mathematically equivalent to the hard zero-variance limit of Soft EM.
</details>

<details>
<summary>Click to view solution for Exercise 4</summary>

**Transfer Proof:** If latent variables $Z$ are fully observed, the true posterior is a degenerate Dirac delta distribution: $p(Z \mid X, \theta) = \mathbb{I}(Z = Z_{\text{obs}})$. The E-step produces deterministic responsibilities $\gamma_{ik} = 1$ if $z_i = k$ and $0$ otherwise. The surrogate function simplifies directly to the complete-data log-likelihood:
$$Q(\theta \mid \theta^{(t)}) = \sum_{i=1}^N \ln p(x_i, z_i = z_{\text{obs}} \mid \theta)$$
The M-step maximizes this complete likelihood in closed form, yielding the exact global MLE $\theta^*$ in a single iteration ($t=1$). Subsequent iterations produce identical parameters: $\theta^{(2)} = \theta^{(1)} = \theta^*$.
</details>

<details>
<summary>Click to view solution for Exercise 5</summary>

**Diagnosis:** The crash occurs because cluster $k$ received zero effective responsibility across all data points: $N_k = \sum_{i=1}^N \gamma_{ik} = 0$. This happens when a cluster is initialized far away from the data support and is abandoned by all points.

**Production Safeguard:** Check $N_k$ prior to division:
```text
if Nk < 1e-8:
    # Re-seed empty cluster onto a randomly chosen data observation
    mu[k] = X[random.randint(0, N - 1)]
    var[k] = 1.0
    Nk = 1.0
```
</details>

---

## 13. Explain it back and return to it

**Closed-notes Feynman prompt:**  
Imagine explaining the Expectation-Maximization algorithm to a software engineer who only knows standard supervised learning without using the terms “Jensen’s inequality”, “surrogate lower bound”, or “variational free energy”. Use the metaphor of unlabeled test scores from two different classrooms, and explain why alternating between guessing student assignments and recalculating class averages guarantees steady progress toward a solution. Once you finish, restore the formal terms and state the Evidence Decomposition Identity.

<details>
<summary>Model explanation for self-evaluation</summary>

Suppose you find a pile of test scores from two different classrooms, but nobody wrote the classroom number on the papers. You want to find the average score for each class. If you knew which student belonged to which class, finding the class averages would be easy. If you knew the class averages, guessing which class a student belonged to would be easy. But you know neither.

EM solves this with a two-step dance:
1. First, make a rough guess of the two classroom averages (say, 50 and 80).
2. Look at each student's score and assign a *soft probability* of belonging to each class (a student with 78 gets an 85% chance of being in Class 2 and a 15% chance of being in Class 1). This is the **E-step**.
3. Now, pretend those probabilities are reality, and recalculate the classroom averages by taking a weighted average of all students based on their probabilities. This is the **M-step**.
4. Repeat this dance. Because each step is mathematically proven to either improve or maintain the overall fit of your model, the averages will steadily march toward the best possible grouping without ever getting worse.

*Restoring formal terminology:* The soft probabilities are **posterior responsibilities** $\gamma_{ik} = p(z_i = k \mid x_i, \theta)$. The step-by-step improvement is governed by the **surrogate lower bound** $Q(\theta \mid \theta^{(t)})$ and the **Dempster-Laird-Rubin Monotonic Convergence Theorem**, derived from the **Evidence Decomposition Identity**:
$$\ln p(X \mid \theta) = F(q, \theta) + D_{\text{KL}}\big( q(Z) \parallel p(Z \mid X, \theta) \big)$$

</details>

### Spaced Repetition Schedule

| Return Date | Closed-Notes Retrieval Task | Self-Verification Anchor |
| :--- | :--- | :--- |
| **Day 1** | Write out the Evidence Decomposition Identity from memory. Derive why $D_{\text{KL}} = 0$ makes the lower bound tangent to true log-evidence. | Check against §4 step-by-step derivation. |
| **Day 7** | Re-derive the 1-iteration pencil-and-paper worked calculation for $x = [1.0, 2.0, 7.0]$. Verify that $\mu_1^{(1)} \approx 1.50$ and $\mu_2^{(1)} \approx 7.00$. | Check against §9 Example 1. |
| **Day 30** | Derive the closed-form mixture weight update $\pi_k^* = N_k / N$ using Lagrange multipliers on the unit simplex constraint $\sum \pi_k = 1$. | Check against §8 Lagrange proof. |

### Self-Assessment Checklist

- [ ] I can construct the auxiliary function $F(q, \theta)$ and prove that $\ln p(X \mid \theta) = F(q, \theta) + D_{\text{KL}}(q \parallel p)$.
- [ ] I can prove why bound tightness $F = \ln p$ occurs if and only if $q(Z) = p(Z \mid X, \theta)$.
- [ ] I can derive the Dempster-Laird-Rubin Monotonic Convergence Theorem using Gibbs' inequality.
- [ ] I can explain why the sequence of log-likelihood values converges to a stationary supremum limit via the Monotone Convergence Theorem.
- [ ] I can mathematically prove why unconstrained Gaussian mixtures exhibit infinite likelihood singularities as $\sigma_k \to 0$.
- [ ] I can derive the closed-form mixture weight update $\pi_k^* = N_k / N$ using Lagrange multipliers on the unit simplex constraint.
- [ ] I can execute an analytical M-step by hand for cluster means and covariances on a 1D dataset.
- [ ] I can demonstrate why K-Means is the zero-variance limit of Soft EM for Gaussian mixtures.
- [ ] I can explain how the Variational Autoencoder generalizes discrete EM to continuous latent spaces via an amortized inference network.
- [ ] I can implement a numerically stabilized EM algorithm using log-sum-exp and verify surrogate gradient stationarity ($\nabla_\mu Q = 0$).

---

## 14. Continue with a purposeful learning path

The resources below are verified for relevance, active status, and pedagogical precision as of **2026-09-18**. Access descriptions indicate verified availability at check time.

| Resource and author | Learning job | Exact starting point | Readiness | Access | Checked date and evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Interactive visualizer:** [Interactive GMM EM Simulation](https://dashee87.github.io/data%20science/general/visualising-the-em-algorithm-in-python/), Dashee87 | Visualize coordinate ascent and cluster evolution interactively | Animated 2D covariance ellipse evolution plots | After §2 | Free open web tutorial | 2026-09-18: verified active interactive animations of EM convergence. |
| **Video lecture:** [Expectation Maximization Step-by-Step](https://www.youtube.com/watch?v=REypj2sy_5U), StatQuest with Josh Starmer | Visual walkthrough of responsibilities and shifting Gaussian bell curves | Full 15-minute video (timestamp 00:00 to 15:00) | After §2 | Free YouTube video | 2026-09-18: verified active 1080p stream, intuitive derivation of soft cluster assignments. |
| **Video lecture (Advanced):** [The Expectation-Maximization (EM) Algorithm](https://www.youtube.com/watch?v=iQo54nmn3Fk), Steve Brunton (University of Washington) | Mathematical breakdown of the $Q$-function and GMM updates | Timestamp 04:30: "The Jensen Inequality and Auxiliary Function" | After §4 | Free YouTube video | 2026-09-18: verified active educational lecture connecting EM to coordinate ascent. |
| **Foundational paper:** [Maximum Likelihood from Incomplete Data via the EM Algorithm](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1977.tb01600.x), Arthur Dempster, Nan Laird, Donald Rubin (JRSS-B, 1977) | Landmark seminal paper establishing the generalized EM framework and monotonic convergence | JRSS-B, Vol. 39, No. 1, pp. 1–38 (Theorem 1 & Definition 1) | After §8 | Open-access Royal Statistical Society paper | 2026-09-18: verified classic citation record, monotonic ascent proof formulation. |
| **Textbook:** [Pattern Recognition and Machine Learning, Chapter 9: Mixture Models and EM](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/), Christopher M. Bishop | Definitive canonical textbook derivation of GMMs and generalized EM | Chapter 9: §9.2 (Gaussian Mixture Models), §9.3 (An Alternative View of EM), and §9.4 (General EM) | After §4 | Free official Microsoft Research PDF | 2026-09-18: verified Chapter 9 download link and exact mathematical theorem layout. |
| **Practice problem set:** [Bishop PRML Chapter 9 Exercises](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/), Christopher M. Bishop (Springer, 2006) | Pencil-and-paper mastery of EM mechanics | Exercises 9.1 (Lagrange multiplier for $\pi$), 9.2 (M-step derivation), and 9.6 (Singularity) | After §12 | Free official PRML solution manual | 2026-09-18: verified exercise numbers and analytical solutions. |
| **Software documentation:** [scikit-learn GaussianMixture Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html), Scikit-Learn Contributors | Production-grade software architecture for EM clustering | Section 2.1: "Gaussian Mixture Models" & API reference | When running §11 | Free official documentation | 2026-09-18: verified scikit-learn 1.5+ API, covariance regularization, and warm-restart options. |

**Next connection:** EM solves latent variable estimation when latent causes are discrete. In [ELBO and variational inference](07-ELBO_and_Variational_Inference.md), we extend this lower-bound formulation to continuous, high-dimensional latent spaces where exact posterior inference is impossible, introducing the Evidence Lower Bound (ELBO) that powers modern Variational Autoencoders.
