# Prerequisites — warm-up before Lec 08 (distribution estimation)

> **Do this first** if “estimate a distribution,” “conditional,” or the symbols $D\sim P$ still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on [Lec 07](../08-Lec07-IID-Assumption/PREREQUISITES.md) (IID + given $D$ estimate $P$).  
> Still a **warm-up** — but every formula below is decoded in plain English.

---

## ⚡ 3-Minute Executive Fast-Track Card

```
              THE DISTRIBUTION ESTIMATION ECOSYSTEM
              
   [ Finite Sample Dataset ]           [ True Underlying Law ]
      D = {(x_i, y_i)}_{i=1}^N  ──────►         P(X, Y)
               │                                   │
               │ (Fit / Parameterize)              ├──► Joint:       P(X, Y)
               ▼                                   ├──► Conditional: P(Y | X) (Classification/Regression)
   [ Estimated Model P_hat ]  ────────┘            ├──► Marginal:    P(Y) (Prevalence)
               │                                   └──► Data Lik:    P(X | Y) (Appearance)
               └──► Generative Goal: Sample new z_new ~ P_hat
```

### 3 Core Mental Shifts
1. **Estimate vs Sample:** Estimating $\hat{P}$ means recovering the probability density or mathematical law from data. Sampling means building an algorithm that draws brand new random points from $\hat{P}$.
2. **Fixed Conditioner Rule:** In any conditional expression $P(A \mid B=b)$, the variable after the vertical bar is clamped/fixed to value $b$.
3. **Density is NOT Probability:** A density $p(x)$ is a rate (probability per unit volume) and can exceed $1.0$. True probabilities only emerge when you integrate $p(x)$ over an interval.

### 3-Question Instant Readiness Gate
1. *In $P(Y \mid X=x)$, which variable is conditioned and fixed?*  
   <details><summary>Reveal Answer</summary><b>Variable $X$ is fixed at value $x$.</b> $Y$ remains the random variable whose conditional distribution is evaluated.</details>
2. *If $p(x) = 2.5$ at some point $x$, is probability theory violated?*  
   <details><summary>Reveal Answer</summary><b>No.</b> $p(x)$ is a probability density, not a probability mass. Densities can be arbitrarily large; only integrals of densities over sets cannot exceed 1.</details>
3. *What does empirical risk minimization estimate in standard supervised classification?*  
   <details><summary>Reveal Answer</summary><b>The conditional probability $P(Y \mid X)$ (discriminative decision rule).</b></details>

---

```
  After this warm-up you can say:

  "D ~_iid P means the dataset is drawn independently from the same unknown law P."
  "Estimate P means recover that law (or a useful piece) from the finite sample D."
  "P(Y|X=x) = law of the label when the image is fixed at x."
  "P(Y) is prevalence; P(X|Y=y) is how data look when the label is fixed."
  "A moment (like E[X]) is a summary number computed from P, not the whole P."
  "In every conditional, the conditioner is fixed."
```

**Warm-up → lecture boxes**

```
  §1  Read D ~_iid P symbol-by-symbol   ──► Topics 1–2
  §2  Estimate P (core job formula)      ──► Topic 3
  §3  Joint / margin / conditional math  ──► Topics 4–5,7
  §4  Moments E[·] vs full P             ──► Topic 4
  §5  Which formula for which question   ──► Topic 5
  §6  Disc vs gen                        ──► Topic 3
  §7  Supervised packaging               ──► Topic 6
  §8  Conditioner fixed + P vs density   ──► Topic 7
```

---

## Math Terminology Rosetta Stone

Before diving into the foundational pillars, use this reference table to decode mathematical shorthand and notation into plain English, spoken phonetics, and software implementations.

| Symbol / Notation | Spoken English (Phonetics) | Mathematical Concept | Plain-English Intuition | Dedicated MathsTerm Link |
| :--- | :--- | :--- | :--- | :--- |
| $\mathcal{P} = \{P_\theta : \theta \in \Theta\}$ | **CAL-ih-GRAF-ik PEE** | Statistical Model Family | Candidate hypothesis set of probability distributions parameterized by weight tensor $\theta$ | [Functions & Derivatives](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/01-Functions_Derivatives_and_Rules.md) |
| $\hat{P} \approx P$ | **PEE-HAT APPROX-ih-MATE-ly PEE** | Distribution Estimation | Using finite empirical samples to recover the unknown data-generating distribution | [Random Variables & Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/01-Random_Variables_and_Distributions.md) |
| $\mathbb{E}_{X \sim P}[g(X)]$ | **EX-pek-TAY-shun OF GEE OF EKS** | Expected Value (Statistical Moment) | Probability-weighted average summary statistic of function $g(X)$ under distribution $P$ | [Random Variables & Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/01-Random_Variables_and_Distributions.md) |
| $P(Y \mid X)$ | **PEE OF WYE GIV-un EKS** | Discriminative Conditional Model | Probability of target label given observed input features; ignores marginal $P(X)$ | [Joint, Marginal & Conditional Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/03-Joint_Marginal_Conditional_Dist.md) |
| $P(X, Y)$ | **PEE OF EKS COMMA WYE** | Generative Joint Model | Full joint distribution of features and labels; models how data is generated | [Joint, Marginal & Conditional Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/03-Joint_Marginal_Conditional_Dist.md) |

---

## 🌉 Curriculum & Sibling Course Prerequisite Bridges

Before diving into the foundational pillars, review these key concepts from sibling course series and standalone mathematical foundations:

| Assumed Concept | Primary Series Foundation | MathsTerms Deep-Dive | 1-Sentence Intuition Refresher |
| :--- | :--- | :--- | :--- |
| **Function Approximation Shift** | [Lec 01: Function Approximation](../../Mathematical-foundation-ml/02-Lec01-Overview-Function-Approximation/NOTES.md) | [Functions & Derivatives](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/01-Functions_Derivatives_and_Rules.md) | Moving from deterministic function fitting to statistical probability distribution estimation. |
| **IID Sampling** | [Lec 07: IID Assumption](../../Mathematical-foundation-ml/08-Lec07-IID-Assumption/NOTES.md) | [Probability Basics & Axioms](../../MathsTerms/03-Probability-and-Statistical-Estimation/00-Probability_Basics_and_Axioms.md) | Dataset $D$ consists of independent and identically distributed realizations from $P_{data}$. |
| **Probability Densities** | [Lec 09: Density Function](../../Mathematical-foundation-ml/10-Lec09-Density-Function/NOTES.md) | [Random Variables & Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/01-Random_Variables_and_Distributions.md) | Continuous density functions represent the mathematical parameterization of estimated distributions. |

---

## 1. Reading $D\sim_{\mathrm{iid}} P$ (symbol by symbol)

<a id="p1-dataset"></a>

### 👶 Physical Analogy & Intuition
Imagine scooping a jar of pebbles from an ancient riverbed. The jar in your hands is the dataset $\mathcal{D}$. The geological process of the entire river that weathered and deposited millions of rocks over millennia is the hidden probability law $P$. The symbol $\sim_{\text{iid}}$ tells you that every pebble in your jar was swept into place independently by that exact same river current.

### 🔍 Plain-English Breakdown
Every later machine learning formula builds on reading this setup line without hesitation:
$$D=\{(x_i,y_i)\}_{i=1}^{n} \quad\sim_{\mathrm{iid}}\quad P_{X,Y} \quad\text{(unknown)}$$

| Symbol | Read as | Plain English |
|--------|---------|----------------|
| $D$ | “dee” / dataset | The finite list of samples you hold |
| $\{(x_i,y_i)\}_{i=1}^{n}$ | pairs $i=1$ to $n$ | Point $i$ has feature vector $x_i$ and label $y_i$ |
| $x_i\in\mathbb{R}^{d}$ | “x-sub-i in R-dee” | $d$ real numbers (stacked image / features) |
| $y_i\in\mathbb{R}^{k}$ or discrete | “y-sub-i” | Continuous vector label **or** category tag $\{0,1,\ldots\}$ |
| $\sim$ | “is sampled from” | The tilde: drawn according to a law |
| $\mathrm{iid}$ | “eye-eye-dee” | **I**ndependent and **i**dentically **d**istributed |
| $P_{X,Y}$ | “joint of X and Y” | The unknown probability law of the pair $(X,Y)$ |

If $Z_1,\ldots,Z_n$ are the data points, **iid** means the joint probability factors as a pure product:
$$P(Z_1\in A_1,\ldots,Z_n\in A_n) = P(Z_1\in A_1)\cdots P(Z_n\in A_n)$$

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Suppose we hold $N=3$ samples with $d=2$ features and a binary label $y \in \{0, 1\}$:
- Point 1: $(\mathbf{x}_1, y_1) = ([1.2, 0.4], 1)$
- Point 2: $(\mathbf{x}_2, y_2) = ([0.5, 1.1], 0)$
- Point 3: $(\mathbf{x}_3, y_3) = ([2.0, 1.8], 1)$
- If each sample has evaluated individual likelihood under candidate model $P_\theta$:
  $$p(\mathbf{x}_1, y_1) = 0.10, \quad p(\mathbf{x}_2, y_2) = 0.10, \quad p(\mathbf{x}_3, y_3) = 0.10$$
- Total joint likelihood under the IID product rule:
  $$p(D) = 0.10 \times 0.10 \times 0.10 = 0.10^3 = 0.0010$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# Verify joint likelihood product factorization
individual_likelihoods = np.array([0.10, 0.10, 0.10])
total_joint_likelihood = np.prod(individual_likelihoods)

assert np.isclose(total_joint_likelihood, 0.0010), "Joint likelihood must equal 0.001"
assert len(individual_likelihoods) == 3
print(f"Joint likelihood across 3 IID samples: {total_joint_likelihood:.6f}")
```

### 🩺 Diagnostic Mini-Check
Does the mathematical statement $D \sim_{\text{iid}} P_{X,Y}$ imply that we already have access to the mathematical formula of $P_{X,Y}$?
<details><summary>Reveal Answer</summary>
<b>No.</b> $P_{X,Y}$ is the <i>unknown</i> true nature distribution that generated the data. We only possess the finite collection of samples in $D$.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $(\mathcal{X} \times \mathcal{Y}, \mathcal{B}, P)$ be a Borel probability space. The dataset $\mathcal{D}_n = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ is a realization of coordinate projection mappings on the product measure space:
$$(\Omega^n, \mathcal{F}^{\otimes n}, \mathbb{P}^n) \to (\mathcal{X} \times \mathcal{Y})^n, \quad P^{\otimes n} = \bigotimes_{i=1}^n P$$
The joint distribution function satisfies $F_n(\mathbf{z}_1, \dots, \mathbf{z}_n) = \prod_{i=1}^n F(\mathbf{z}_i)$ where $\mathbf{z}_i = (\mathbf{x}_i, y_i)$.
</details>

---

## 2. The core job formula: estimate $P$

<a id="p2-estimate-p"></a>

### 👶 Physical Analogy & Intuition
Imagine listening to a symphony played behind a thick soundproof wall where only muffled fragments leak through. Your task is to reconstruct the entire orchestral score from those muffled snippets. In machine learning, the muffle is our finite sample $D$; reconstructing the full musical symphony is "estimating $P$." If you additionally build a robotic orchestra that plays brand-new symphonies in that exact style, you have solved "generative sampling."

### 🔍 Plain-English Breakdown
This is the single overarching problem formulation of modern statistical machine learning:
$$\boxed{\; \text{Given } D\sim_{\mathrm{iid}} P \text{ (unknown)}, \quad \text{estimate } P \;}$$

| Piece | Meaning |
|-------|---------|
| Given $D$ | You start with samples only |
| $P$ unknown | The law is not observed directly |
| Estimate $P$ | Build $\hat{P}$ (or a useful piece of $P$) that explains $D$ |

**Second objective (generative modeling):**
$$\text{also learn to sample new } z\sim \hat{P}$$
Estimating a probability function mathematically is distinct from having a numerical procedure that draws random samples.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Suppose a coin has true unknown success probability $p^* = 0.60$.
- We observe $N=10$ draws: $7$ heads ($1$) and $3$ tails ($0$).
- Empirical estimate:
  $$\hat{p} = \frac{1}{N}\sum_{i=1}^{10} z_i = \frac{7}{10} = 0.70$$
- Estimation error:
  $$|\hat{p} - p^*| = |0.70 - 0.60| = 0.10$$
- Total law probability verification: $\hat{P}(H) + \hat{P}(T) = 0.70 + 0.30 = 1.00$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# Estimate unknown Bernoulli parameter p from finite sample
true_p = 0.60
observed_draws = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
p_hat = np.mean(observed_draws)

assert np.isclose(p_hat, 0.70), "Empirical p_hat must equal 0.70"
assert np.isclose(abs(p_hat - true_p), 0.10), "Estimation error must equal 0.10"
print(f"True p: {true_p}, Estimated p_hat: {p_hat:.2f}")
```

### 🩺 Diagnostic Mini-Check
Does "estimating $P$" mean simply memorizing the training dataset $D$?
<details><summary>Reveal Answer</summary>
<b>No!</b> Memorization corresponds to the discrete empirical measure $\hat{P}_N = \frac{1}{N}\sum \delta_{z_i}$, which assigns zero probability to all unseen points. Estimating $P$ requires learning a structured hypothesis rule $\hat{P}_\theta$ that generalizes to novel inputs.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $\mathcal{P} = \{P_\theta : \theta \in \Theta\}$ be a statistical model family. Distribution estimation finds $\hat{\theta} \in \Theta$ minimizing a statistical divergence (e.g., Kullback-Leibler divergence):
$$\hat{\theta} = \arg\min_{\theta \in \Theta} D_{\mathrm{KL}}(P \| P_\theta) = \arg\max_{\theta \in \Theta} \mathbb{E}_{Z \sim P}[\log p_\theta(Z)]$$
By the Law of Large Numbers, the empirical surrogate $\hat{\theta}_N = \arg\max_\theta \frac{1}{N}\sum_{i=1}^N \log p_\theta(z_i)$ converges to $\theta^*$ as $N \to \infty$.
</details>

---

## 3. Joint, margin, conditional — formulas decoded

<a id="p3-jcm"></a>

### 👶 Physical Analogy & Intuition
Imagine a 2D contingency spreadsheet tracking patients in a clinic. The cells in the middle of the table show the **joint** probability of having both high blood sugar and diabetic retinopathy. Summing across a row gives the **marginal** probability of high blood sugar regardless of eye health. If you zoom in and inspect ONLY the high blood sugar row and re-normalize the numbers to sum to 1.0, you obtain the **conditional** distribution of eye health given high sugar.

### 🔍 Plain-English Breakdown
Most machine learning tasks depend on distinguishing these three operations:

- **Joint:** Both events occur simultaneously: $P_{X,Y}(A,B) = P(X \in A, Y \in B)$.
- **Marginals:** Sum or integrate out one variable to see the other alone:
  $$P_X(x) = \sum_y P(X=x, Y=y), \qquad P_Y(y) = \sum_x P(X=x, Y=y)$$
- **Conditional:** Fix the conditioning variable at a specific observed value:
  $$P(Y=y \mid X=x) = \frac{P(X=x, Y=y)}{P(X=x)} \quad \text{when } P(X=x) > 0$$

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Consider this joint probability table:

```
           Y=0    Y=1    row sum = P(X=·)
  X=a      0.1    0.2      0.3
  X=b      0.3    0.4      0.7
  col      0.4    0.6      1.0  = P(Y=·)
```

- Joint probability: $P(X=a, Y=1) = 0.20$.
- Marginal probability: $P(Y=1) = 0.20 + 0.40 = 0.60$.
- Conditional probability:
  $$P(Y=1 \mid X=a) = \frac{P(X=a, Y=1)}{P(X=a)} = \frac{0.20}{0.30} = \frac{2}{3} \approx 0.6667$$
- Note that $P(Y=1 \mid X=a) = 0.6667 \neq P(Y=1) = 0.6000$ because $X$ and $Y$ are dependent!

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# Construct joint table
joint_matrix = np.array([[0.10, 0.20],
                         [0.30, 0.40]])

# Marginal calculations
p_x = joint_matrix.sum(axis=1)  # [0.30, 0.70]
p_y = joint_matrix.sum(axis=0)  # [0.40, 0.60]

# Conditional calculation: P(Y=1 | X=a)
p_y1_given_xa = joint_matrix[0, 1] / p_x[0]

assert np.isclose(p_x[0], 0.30), "P(X=a) must equal 0.30"
assert np.isclose(p_y[1], 0.60), "P(Y=1) must equal 0.60"
assert np.isclose(p_y1_given_xa, 2.0 / 3.0), "Conditional P(Y=1|X=a) must equal 2/3"
print(f"P(Y=1) = {p_y[1]:.2f}, but P(Y=1 | X=a) = {p_y1_given_xa:.4f}")
```

### 🩺 Diagnostic Mini-Check
If $P(X=a) = 0.40$ and $P(X=a, Y=1) = 0.10$, what is $P(Y=1 \mid X=a)$?
<details><summary>Reveal Answer</summary>
$$P(Y=1 \mid X=a) = \frac{P(X=a, Y=1)}{P(X=a)} = \frac{0.10}{0.40} = 0.25$$
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space, and let $X, Y$ be random variables. The conditional distribution $P_{Y \mid X}(\cdot \mid x)$ is the regular conditional probability measure satisfying:
$$P_{X,Y}(A \times B) = \int_A P_{Y \mid X}(B \mid x) \, P_X(dx) \quad \forall A \in \mathcal{B}(\mathcal{X}), \; B \in \mathcal{B}(\mathcal{Y})$$
When densities exist with respect to Lebesgue measure $\lambda$:
$$p(y \mid x) = \frac{p(x, y)}{p(x)} = \frac{p(x, y)}{\int_{\mathcal{Y}} p(x, y') \, dy'}$$
</details>

---

## 4. Moments vs full distribution — formulas

<a id="p4-moments"></a>

### 👶 Physical Analogy & Intuition
Imagine summarizing an entire human face using just one number: the average distance between their eyes. Two completely different people might have the exact same eye distance, but totally different jawlines, noses, and expressions. A statistical moment (like mean or variance) is just a single summary measurement computed from $P$; the full distribution $P$ contains the complete, uncompressed geometry.

### 🔍 Plain-English Breakdown
Sometimes machine learning only needs a **function of $P$**, rather than estimating all of $P$:

- **Expectation (First Moment):** The probability-weighted average value:
  $$\mathbb{E}[Z] = \sum_z z\,P(Z=z) \quad\text{(discrete)}, \qquad \mathbb{E}[Z] = \int z\,p(z)\,dz \quad\text{(continuous)}$$
- **Second Moment & Variance:** $\mathbb{E}[Z^2]$ measures squared spread, with $\text{Var}(Z) = \mathbb{E}[Z^2] - (\mathbb{E}[Z])^2$.

| Job | Formula-level target |
|-----|----------------------|
| Full estimation | recover the complete function $P$ (or density $p$) |
| Moment estimation | recover summary scalar $\mathbb{E}[g(Z)]$ (e.g., $g(z)=z$) |

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let $Z \in \{10, 20, 30\}$ with probabilities $[0.20, 0.50, 0.30]$:
- Probability sum check: $0.20 + 0.50 + 0.30 = 1.00$.
- Expected value (1st moment):
  $$\mathbb{E}[Z] = 10(0.20) + 20(0.50) + 30(0.30) = 2.0 + 10.0 + 9.0 = 21.0$$
- Second raw moment:
  $$\mathbb{E}[Z^2] = 10^2(0.20) + 20^2(0.50) + 30^2(0.30) = 100(0.20) + 400(0.50) + 900(0.30) = 20 + 200 + 270 = 490.0$$
- Variance calculation:
  $$\text{Var}(Z) = \mathbb{E}[Z^2] - (\mathbb{E}[Z])^2 = 490.0 - (21.0)^2 = 490.0 - 441.0 = 49.0$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

z_values = np.array([10.0, 20.0, 30.0])
probs = np.array([0.20, 0.50, 0.30])

mean_z = np.sum(z_values * probs)
second_moment = np.sum((z_values ** 2) * probs)
variance_z = second_moment - (mean_z ** 2)

assert np.isclose(mean_z, 21.0), "Expected value must equal 21.0"
assert np.isclose(second_moment, 490.0), "Second moment must equal 490.0"
assert np.isclose(variance_z, 49.0), "Variance must equal 49.0"
print(f"Mean: {mean_z:.1f}, Variance: {variance_z:.1f}")
```

### 🩺 Diagnostic Mini-Check
If two probability distributions share the exact same mean and variance, must their distributions be identical?
<details><summary>Reveal Answer</summary>
<b>No.</b> Higher-order moments (skewness, kurtosis) and tail behaviors can differ dramatically. A bimodal distribution and a unimodal Gaussian can have identical means and variances while having completely different shapes.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

For a random variable $Z$ with probability measure $P$, the $k$-th raw moment is the Lebesgue integral:
$$\mu_k' = \mathbb{E}[Z^k] = \int_{\mathbb{R}} z^k \, dP(z)$$
The characteristic function uniquely determines the distribution:
$$\phi_Z(t) = \mathbb{E}[e^{itZ}] = \sum_{k=0}^\infty \frac{(it)^k}{k!} \mu_k'$$
Under Carleman's condition ($\sum_{k=1}^\infty (\mu_{2k}')^{-1/(2k)} = \infty$), the sequence of moments uniquely specifies the probability distribution.
</details>

---

## 5. Match each question to a formula

<a id="p5-match-question"></a>

### 👶 Physical Analogy & Intuition
Imagine walking into a massive library archive of legal cases. If you ask "How many total fraud cases exist in this country?", you want the overall category count ($P(Y)$). If you bring an encrypted hard drive and ask "Is this specific drive evidence of fraud?", you want the conditional verdict ($P(Y \mid X=x)$). If you ask "What kinds of hard drives do fraudsters buy?", you want the class-conditional feature distribution ($P(X \mid Y=1)$). One archive, three distinct questions, three distinct formulas.

### 🔍 Plain-English Breakdown
Matching real-world engineering questions to their rigorous probabilistic targets is the core skill of statistical modeling:

| Real question | Formula to estimate | English |
|---------------|---------------------|---------|
| Is **this** image diseased? | $P(Y\mid X=x)$ | label law given fixed image $x$ |
| Where is the tumor box? | $P(Y\mid X=x)$ with continuous $Y$ | continuous label geometry |
| How common is disease? | $P(Y=1)$ | label marginal; **no image** |
| How do pixels look (ignore labels)? | $P_X$ or $p(x)$ | data marginal |
| How do **diseased** images look? | $P(X\mid Y=1)$ | data law with disease fixed |
| Fill missing pixels | $P(Y\mid X=x)$ where $Y$ = missing pixels | self-supervised inpainting |

- **Classification:** $Y$ is discrete.
- **Regression:** $Y \in \mathbb{R}^k$ is continuous.
Both share the exact same mathematical goal: estimating the conditional distribution $P(Y \mid X)$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Suppose in a cohort of $N=1000$ clinic patients:
- Overall disease prevalence (marginal): $P(Y=1) = \frac{50}{1000} = 0.050$ ($5\%$).
- For a patient with an abnormal biomarker score $x^*$:
  $$P(Y=1 \mid X=x^*) = \frac{P(X=x^*, Y=1)}{P(X=x^*)} = \frac{0.040}{0.050} = 0.800 \quad (80\%)$$
- Notice the difference: $P(Y=1) = 0.050$, whereas $P(Y=1 \mid X=x^*) = 0.800$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# Prevalence vs conditional diagnostic probability
marginal_prevalence = 50.0 / 1000.0  # 0.05
p_joint_abnormal_and_disease = 0.040
p_marginal_abnormal = 0.050

p_conditional_diagnosis = p_joint_abnormal_and_disease / p_marginal_abnormal

assert np.isclose(marginal_prevalence, 0.050)
assert np.isclose(p_conditional_diagnosis, 0.800)
print(f"General prevalence P(Y=1): {marginal_prevalence:.1%}, Conditional P(Y=1|X=x*): {p_conditional_diagnosis:.1%}")
```

### 🩺 Diagnostic Mini-Check
If a doctor asks: "What do chest CT scans look like in patients with pneumonia?", what formal probability distribution should be modeled?
<details><summary>Reveal Answer</summary>
<b>$P(X \mid Y=\text{pneumonia})$</b> (the class-conditional data distribution where label $Y$ is fixed to pneumonia).
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Under Bayes' Theorem, these probability objects are related via:
$$P(Y=y \mid \mathbf{X}=\mathbf{x}) = \frac{p(\mathbf{x} \mid Y=y) P(Y=y)}{p(\mathbf{x})} = \frac{p(\mathbf{x} \mid Y=y) P(Y=y)}{\sum_{y'} p(\mathbf{x} \mid Y=y') P(Y=y')}$$
The Bayes Optimal Classifier minimizes risk under 0-1 loss by choosing:
$$h^*(\mathbf{x}) = \arg\max_{y \in \mathcal{Y}} P(Y=y \mid \mathbf{X}=\mathbf{x})$$
</details>

---

## 6. Discriminative vs generative (formulas of intent)

<a id="p6-disc-gen"></a>

### 👶 Physical Analogy & Intuition
Think of an art appraiser vs a master painter. The appraiser only needs to distinguish genuine paintings from fakes—they evaluate $P(\text{Authentic} \mid \text{Painting})$. That is **discriminative**. The master painter, however, knows the entire structure of pigments, light, and canvas so deeply that they can create brand-new original masterpieces from scratch. That is **generative**.

### 🔍 Plain-English Breakdown
The difference lies in what the model parameterizes and whether it can draw new points:

| Style | Intent formula |
|-------|----------------|
| **Discriminative** (this lecture’s use) | Produce $\hat{P}$ or $\widehat{P}(Y\mid X)$ for decisions; **not** focused on drawing new $z$ |
| **Generative** | Also provide a way to draw $z_{\mathrm{new}}\sim \hat{P}$ |

Having a mathematical formula for $\hat{P}_X$ or $\hat{P}_{X,Y}$ does not automatically provide an algorithm to draw random samples. Sampling is an additional operational capability.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Consider discrete binary feature $X \in \{0, 1\}$ and label $Y \in \{0, 1\}$:
- Joint distribution matrix $P(X, Y)$:
  $$P(0, 0) = 0.30, \quad P(0, 1) = 0.10, \quad P(1, 0) = 0.20, \quad P(1, 1) = 0.40$$
  Sum: $0.30 + 0.10 + 0.20 + 0.40 = 1.00$.
- Discriminative target for $X=1$:
  $$P(Y=1 \mid X=1) = \frac{P(1, 1)}{P(X=1)} = \frac{0.40}{0.20 + 0.40} = \frac{0.40}{0.60} = \frac{2}{3} \approx 0.6667$$
- A discriminative classifier only needs to store the conditional threshold $\frac{2}{3}$; a generative model retains the full joint table and can sample synthetic pairs $(x, y)$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# Verify joint vs conditional calculation
p_joint = np.array([[0.30, 0.10],
                    [0.20, 0.40]])

# Marginal P(X=1)
p_x1 = np.sum(p_joint[1, :])  # 0.60
# Discriminative conditional P(Y=1 | X=1)
p_y1_given_x1 = p_joint[1, 1] / p_x1  # 0.40 / 0.60

assert np.isclose(np.sum(p_joint), 1.0)
assert np.isclose(p_y1_given_x1, 2.0 / 3.0)
print(f"Discriminative conditional P(Y=1|X=1): {p_y1_given_x1:.4f}")
```

### 🩺 Diagnostic Mini-Check
If a neural network achieves 99% accuracy predicting whether an image is a cat or dog, can it necessarily generate synthetic images of cats?
<details><summary>Reveal Answer</summary>
<b>No.</b> A discriminative network models only the decision boundary $P(Y \mid X)$. It has no generative representation of $P(X)$ and cannot synthesize new images.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

- **Discriminative approach:** Directly models the conditional likelihood $p_\theta(y \mid \mathbf{x})$, optimizing the conditional log-likelihood $\sum_{i=1}^n \log p_\theta(y_i \mid \mathbf{x}_i)$.
- **Generative approach:** Models the joint density $p_\theta(\mathbf{x}, y) = p_\theta(\mathbf{x} \mid y)p_\theta(y)$, requiring integration over high-dimensional input spaces $\mathcal{X}$. According to Vapnik's principle, one should solve the targeted estimation problem directly rather than solving the more general joint modeling problem as an intermediate step.
</details>

---

## 7. Supervised / unsupervised (no new probability objects)

<a id="p7-sup-unsup"></a>

### 👶 Physical Analogy & Intuition
Imagine a deck of standard playing cards. If you ask a student to guess the card's suit based on its number, you call the game "supervised." If you ask the student to sort the deck into piles based on visual patterns without giving them rules, you call the game "unsupervised." The deck of cards and the random chance of drawing any specific card are identical in both cases. The division between supervised and unsupervised is simply how we assign roles to the variables.

### 🔍 Plain-English Breakdown
In probability theory, there is no separate "supervised probability" or "unsupervised probability":

| Name | Packaging |
|------|-----------|
| Supervised | $D$ includes labels $y_i$; algorithms explicitly predict them |
| Unsupervised | $D=\{x_i\}$ only; algorithms model data structure |

Both setups study random variables defined on sample space $\Omega$ with probability measure $P$. Inpainting (masking pixels in an image to predict them from visible pixels) demonstrates that targets $y$ can always be carved directly out of inputs $x$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Consider an observation with $d=2$ continuous features and $k=1$ binary label:
- Feature vector: $\mathbf{x} = [0.85, -1.20] \in \mathbb{R}^2$.
- Target label: $y = [1.0] \in \mathbb{R}^1$.
- Combined unsupervised vector:
  $$\mathbf{z} = [\mathbf{x}; y] = [0.85, -1.20, 1.0] \in \mathbb{R}^{2 + 1} = \mathbb{R}^3$$
- The joint distribution $P(\mathbf{X}, Y)$ over pair $(\mathbf{x}, y)$ is mathematically equivalent to the distribution $P(\mathbf{Z})$ over single 3-dimensional vector $\mathbf{z}$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# Unify feature and label into single random vector
features = np.array([0.85, -1.20])
label = np.array([1.0])

z_vector = np.concatenate([features, label])
assert z_vector.shape == (3,), "Combined vector dimension must be 3"
assert np.allclose(z_vector[:2], features) and z_vector[2] == label[0]
print(f"Combined single random vector z: {z_vector}")
```

### 🩺 Diagnostic Mini-Check
Why does self-supervised masked autoencoding blur the line between supervised and unsupervised learning?
<details><summary>Reveal Answer</summary>
Because it creates an explicit supervised regression target ($y = \text{masked pixels}$) from unlabelled data ($x = \text{visible pixels}$), using standard supervised loss functions on what was originally an unsupervised dataset.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $\mathbf{Z} = (\mathbf{X}, Y) \in \mathcal{Z}$ with Borel probability space $(\mathcal{Z}, \mathcal{B}(\mathcal{Z}), P_{\mathbf{Z}})$. Supervised learning projects onto the conditional probability distribution $P_{Y \mid \mathbf{X}}$ via disintegrations:
$$P_{\mathbf{Z}}(d\mathbf{x}, dy) = P_{\mathbf{X}}(d\mathbf{x}) P_{Y \mid \mathbf{X}}(dy \mid \mathbf{x})$$
Unsupervised learning estimates the projection $P_{\mathbf{X}}$. Because both derive from the same measure $P_{\mathbf{Z}}$, no new axiomatic foundations are introduced.
</details>

---

## 8. Conditioner fixed; $P$ vs density (notation)

<a id="p8-conditional-fixed"></a>

### 👶 Physical Analogy & Intuition
Imagine reading a thermometer in your kitchen. If you say "the temperature given that it is December in Chicago," you have clamped the geographic and temporal conditions before reading the temperature dial. Furthermore, a probability density value $p(x) = 4.0$ is like a speedometer needle reading 80 mph. 80 mph is a **speed** (rate), not a **distance traveled**. To find actual distance (probability), you must multiply that speed by the duration of time (interval width).

### 🔍 Plain-English Breakdown
### Conditional evaluation (how to read the board)
When written as $P(X \mid Y)$, it means:
$$\text{evaluate at } x \text{ with } Y \text{ fixed at } y \quad=\quad P(X\in\cdot \mid Y=y)$$

- $x$: evaluation point (dummy variable where the density is read).
- $y$: fixed value of the conditioner (locked constant).

### Distribution vs density
- **Probability Measure / Distribution $P$:** Assigns probabilities $\in [0, 1]$ to whole sets of events.
- **Probability Density $p(x)$:** A continuous derivative/rate. $p(x)$ can be greater than $1.0$! True probabilities come only from **integrals** of $p(x)$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Consider a continuous uniform random variable $X$ over the narrow interval $[0.0, 0.2]$:
- Density function height:
  $$p(x) = \frac{1}{b - a} = \frac{1}{0.2 - 0.0} = \frac{1}{0.2} = 5.0 \quad \text{for } x \in [0.0, 0.2]$$
- Notice: $p(x) = 5.0 \gg 1.0$! (Density is a rate, not a probability).
- Actual probability of falling in sub-interval $[0.05, 0.15]$:
  $$P(X \in [0.05, 0.15]) = \int_{0.05}^{0.15} 5.0 \, dx = 5.0 \times (0.15 - 0.05) = 5.0 \times 0.10 = 0.50 \le 1.0$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

# Uniform distribution on narrow interval [0.0, 0.2]
a, b = 0.0, 0.2
density = 1.0 / (b - a)  # 5.0

# Calculate probability of sub-interval [0.05, 0.15]
sub_a, sub_b = 0.05, 0.15
prob = density * (sub_b - sub_a)

assert density == 5.0, "Density can exceed 1.0"
assert np.isclose(prob, 0.50), "Probability must equal 0.50"
print(f"Density height p(x) = {density}, Integrated probability = {prob}")
```

### 🩺 Diagnostic Mini-Check
If an algorithm outputs $p(x) = 3.2$ at a test point $x$, is the model broken?
<details><summary>Reveal Answer</summary>
<b>No!</b> For continuous random variables, $p(x)$ represents probability density, which can legally take any non-negative value in $[0, \infty)$. Only the integrated area under the curve must equal 1.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $P$ be absolutely continuous with respect to Lebesgue measure $\lambda$ on $\mathbb{R}^d$ ($P \ll \lambda$). By the Radon-Nikodym theorem, there exists a non-negative measurable function $p = \frac{dP}{d\lambda}$ such that:
$$P(B) = \int_B p(\mathbf{x}) \, d\lambda(\mathbf{x}) \quad \forall B \in \mathcal{B}(\mathbb{R}^d)$$
While $P(B) \in [0, 1]$ for all measurable sets $B$, the density $p(\mathbf{x})$ satisfies only $p(\mathbf{x}) \ge 0$ and $\int_{\mathbb{R}^d} p(\mathbf{x}) \, d\mathbf{x} = 1$, allowing $p(\mathbf{x}) > 1$ locally.
</details>

---

### Paper check

1. Expand $D\sim_{\mathrm{iid}} P_{X,Y}$ into a full English sentence.  
2. Write the product formula for $n$ independent points.  
3. From the micro table, compute $P(Y=1\mid X=a)$.  
4. Write $\mathbb{E}[Z]$ for a discrete $Z$ and say what it means.  
5. Match: prevalence → $?$; diseased appearance → $?$.  
6. In $P(X\mid Y=y)$, what is fixed?  
7. Estimate $P$ vs sample from $P$ — one sentence each.

---

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).  
Prior: [Lec 07](../08-Lec07-IID-Assumption/NOTES.md).

---

## 🗝️ Mathematical Foundations & MathsTerms Bridge

> [!TIP]
> **Foundational Knowledge Base:** This module directly relies upon formal mathematical constructs systematically defined and verified in our central [`MathsTerms`](../../MathsTerms) repository. For visual dependency graphs and multi-track learning roadmaps, consult the [Grand Unified Concept Map](../../MathsTerms/CONCEPT_MAP.md).

| Mathematical Concept | Dedicated Guide | Role & Significance in This Lecture |
| :--- | :--- | :--- |
| **Likelihood & Log-Likelihood** | [Likelihood & Log-Likelihood](../../MathsTerms/03-Probability-and-Statistical-Estimation/04-Likelihood_and_Log_Likelihood.md) | Parametric likelihood functions $L(\theta; \mathcal{D})$ as measures of parameter goodness-of-fit |
| **Maximum Likelihood Estimation (MLE)** | [Maximum Likelihood Estimation (MLE)](../../MathsTerms/03-Probability-and-Statistical-Estimation/05-MLE.md) | Foundational principles of fitting parametric distribution families $\arg\max_\theta \ell(\theta)$ |
| **Common Probability Distributions** | [Common Probability Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/02-Common_Probability_Distributions.md) | Gaussian, Bernoulli, Poisson, and categorical parametric models |

---
