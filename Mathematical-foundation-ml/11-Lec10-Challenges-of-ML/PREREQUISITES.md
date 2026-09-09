# Prerequisites — warm-up before Lec 10 (challenge with ML)

> **Do this first** if “parametric model,” “argmin,” or “distance between densities” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on [Lec 09](../10-Lec09-Density-Function/PREREQUISITES.md) (density $p$; estimate $p$ not only $P$).  
> You asked for stronger basics — every idea below is decoded with formulas and micro numbers.

```
  After this warm-up you can say:

  "ML here means: estimate an unknown density p (or a surrogate) from finite IID samples D."
  "A model is a density estimate from a parametric family p_θ."
  "Parameters θ pick one member of that family (e.g. mean and variance)."
  "Training = find θ that minimizes a distance d(p, p_θ) (or a sample stand-in)."
  "arg min returns the best θ, not the distance value itself."
  "Different (family, distance, optimizer) triples make different algorithms."
  "ERM (next lecture) is the same recipe in a deterministic dialect."
```

**Warm-up → lecture boxes**

```
  §1  Density + estimate p (reload)     ──► Topics 1–2
  §2  IID samples / dataset D           ──► Topics 1–2
  §3  Parametric family + parameters    ──► Topics 2–3
  §4  Model vs algorithm                ──► Topic 3
  §5  Distance / divergence intuition   ──► Topic 4
  §6  Optimization + argmin             ──► Topic 5
  §7  Gradient descent (one picture)    ──► Topics 5–6
  §8  Recipe + surrogates + ERM preview  ──► Topics 5–7
```

---

## Math Terminology Rosetta Stone

Before diving into the foundational pillars, use this reference table to decode mathematical shorthand and notation into plain English, spoken phonetics, and software implementations.

| Symbol / Notation | Spoken English (Phonetics) | Mathematical Concept | Plain-English Intuition | Dedicated MathsTerm Link |
| :--- | :--- | :--- | :--- | :--- |
| $\mathcal{P} = \{p_\theta : \theta \in \Theta\}$ | **CAL-ih-GRAF-ik PEE** | Parametric Hypothesis Class | The family of candidate probability density models indexed by parameter vector $\theta$ | [Functions & Derivatives](../../MathsTerms/03-Multivariate-Calculus-and-Optimization/01-Functions_Derivatives_and_Rules.md) |
| $D(p, q)$ | **DEE OF PEE COMMA KYOO** | Statistical Divergence / Distance | Measure of statistical dissimilarity between true density $p$ and candidate model $q$ | [KL Divergence](../../MathsTerms/05-Information-Theory-and-Divergences/02-KL_Divergence.md) |
| $\theta^* = \arg\min_\theta D(p, p_\theta)$ | **ARG-MIN OVER THAY-tuh OF DEE** | Optimal Parameter Vector | The specific weight configuration that minimizes the discrepancy between true and modeled densities | [Functions & Derivatives](../../MathsTerms/03-Multivariate-Calculus-and-Optimization/01-Functions_Derivatives_and_Rules.md) |
| $\nabla_\theta \mathcal{L}(\theta)$ | **NAB-luh THAY-tuh OF EL** | Loss Gradient Vector | Direction of steepest error increase in parameter space used by optimizers to update weights | [Derivatives, Gradients & Jacobians](../../MathsTerms/03-Multivariate-Calculus-and-Optimization/02-Derivatives_Gradients_and_Jacobians.md) |
| $\theta^{(t+1)} = \theta^{(t)} - \eta \nabla \mathcal{L}$ | **THAY-tuh AT TEE PLUS ONE** | Gradient Descent Update Step | Moving weights downhill against the loss gradient scaled by learning rate $\eta$ | [Gradient Descent](../../MathsTerms/03-Multivariate-Calculus-and-Optimization/09-Gradient_Descent.md) |

---

## 1. Density and “estimate $p$” (reload)

<a id="p1-density-estimate"></a>

### Purpose for the video

Lec 09 installed density $p$. This lecture’s **challenge** is that $p$ is **unknown** and we must recover it from data.

### Density (one-line)

$$
p_X:\text{range}(X)\to\mathbb{R}_+,\qquad
P_X(x)=\int_{-\infty}^{x}p_X(t)\,dt,\qquad
\int p_X=1
$$

- Height $p(x)$ is **not** a probability.  
- Area $\int_A p$ **is** a probability.  
- Course writes small $p$ for density, capital / two-line $P$ for distribution.

### Estimate vs estimator (words)

| Word | Meaning |
|------|---------|
| **Estimate** | the concrete function you output after seeing $D$ (e.g. a fitted Gaussian) |
| **Estimator** | the **procedure / rule** that turns $D$ into that estimate |

He calls the fitted object a **model**. In estimation theory, that model is an **estimator** of the unknown density (always imperfect once $D$ is finite).

“Estimate $p$” means: produce a function $\hat p$ (from data) that stands in for the unknown true density.

### Why $p(x)$, $p(y\mid x)$, $p(x,y)$ feel “the same”

Each target is still “estimate a non-negative function of range points whose integrals are probabilities.”  
Only **which** function (marginal / conditional / joint) changes with the ML use case.

### Analogy — purpose

Unknown mountain height profile (density). You only get random GPS samples on the mountain. Reconstruct the profile.

---

## 2. IID samples and the dataset $D$

<a id="p2-iid-dataset"></a>

### Purpose for the video

The only information we get about $p$ is a finite list of draws.

### Dataset

$$
D=\{x^{(1)},x^{(2)},\ldots,x^{(n)}\}
\quad\text{(or pairs }(x,y)\text{ when labels exist)}
$$

**IID** = independent and identically distributed: each point is drawn from the **same** unknown law, and draws do not depend on each other (idealized assumption from earlier lectures).

### Important honesty (he will say this)

Strictly, you sample from a **distribution**, not from a density curve. Saying “IID from density $p$” is a common shorthand / leap of faith.

### Micro

Toss a coin with unknown bias $p(\text{heads})=\theta^\star$.  
$D =$ sequence of $n$ tosses. Goal: recover something about $\theta^\star$ (or the full law).

---

## 3. Parametric family and parameters $\theta$

<a id="p3-parametric"></a>

### Purpose for the video

Without any assumption, “all functions range→$\mathbb{R}_+$ with integral 1” is too big. We **restrict** the search.

### Why restrict? (the logical “why parametric”)

The set of *all* valid densities is **infinite-dimensional** (you could wiggle the curve almost anywhere and renormalize).  
Finite data $D$ only gives finite information. A **parametric family** collapses the search to a **finite list of free numbers** $\theta\in\mathbb{R}^k$ (often $k$ small; modern nets make $k$ huge but still finite). That is what makes optimization well-posed as “search in $\theta$-space.”

### Parametric family

A **parametric family** is a collection of functions all sharing the **same shape**, differing only by a finite list of numbers $\theta$ (the **parameters**):

$$
\{p_\theta : \theta\in\Theta\}
$$

Example — 1D Gaussian family:

$$
p_\theta(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big),\qquad
\theta=(\mu,\sigma)
$$

| Object | Meaning |
|--------|---------|
| Family | “all Gaussians” (shape fixed: bell curve) |
| $\theta$ | which bell: center $\mu$, width $\sigma$ |
| One fixed $\theta$ | one concrete density (one **model instance**) |
| $\dim(\theta)$ | how many free numbers you will optimize |

**Linear / affine form** (he uses as a simple parametric **object** with parameters $w_1,w_2$ on a vector $x\in\mathbb{R}^d$): parameters pick a line/hyperplane.

**Important clarity for densities:** a raw line $w_1^\top x+w_2$ is **not** automatically a probability density (need non-negativity + integral 1). In the lecture it is mainly a **toy parametric shape** / stand-in for “restrict to a family with finite $\theta$.” For *probability densities*, cleaner first examples are **Gaussian / exponential / Bernoulli**. He also names linear regression, logistic regression, NNs, GMMs, transformers as $p_\theta$ **choices** in the *broad* ML recipe (the object being fit may be a density, a conditional density, or a related surrogate — see §9).

### Micro numbers

True data from Normal$(0,1)$.  
You wrongly assume “all mass on a spike at 5” (bad family) → no amount of training fixes the family.  
You correctly assume Gaussians and only need to find $(\mu,\sigma)$ → training can help.

### Analogy — purpose

Clothing rack: “all T-shirts” = family. Size + color = $\theta$. Buying “size M blue” = fixing parameters.  
Searching “every possible garment in the universe” = unrestricted densities.

---

## 4. Model vs algorithm

<a id="p4-model-vs-algo"></a>

### Purpose for the video

He draws a hard line between choosing the **family** and searching **inside** the family.

| Word | Job |
|------|-----|
| **Model / model choice** | Pick the family $p_\theta$ (Gaussian? linear? neural net?) — a **leap of faith** |
| **Algorithm** | Given that family, find a good $\theta$ from data (and later: which distance, which optimizer) |

If the true density is quadratic-shaped and you forced a “line-shaped” family, the algorithm only answers “which line?” — it cannot invent a parabola for you.

### Slogan you will hear

**All models are wrong. Some are useful.**  

Why “always wrong” (intuition, not a full theorem):

1. **Wrong family** → structural bias (line when truth is quadratic).  
2. **Finite samples** → even a correct family usually yields $\theta^\star\neq\theta_{\text{true}}$ (noise).  
3. So the fitted $p_{\theta^\star}$ is almost never *exactly* nature’s $p$ — yet it can still be **useful** under a score.

---

## 5. Distance and divergence (why not always a “metric”)

<a id="p5-distance"></a>

### Purpose for the video

After fixing a family, infinitely many $\theta$ remain. You need a score: how far is $p_\theta$ from true $p$?

### Distance-like score

$$
d(p, p_\theta)\;\ge\;0
$$

- Larger $d$ ⇒ poorer match.  
- Ideal: $d(p,p)=0$ when $p_\theta=p$.  

He writes **“distance metric” in quotes** because real ML scores (e.g. **KL divergence**) often fail classical metric axioms:

| Metric axiom | Classical distance | Many ML divergences |
|--------------|--------------------|---------------------|
| Non-negative | yes | usually yes |
| Symmetric $d(a,b)=d(b,a)$ | required | often **no** (KL is asymmetric) |
| Triangle inequality | required | often **no** |

So think **score / divergence / discrepancy**, not always “true metric.”

### KL divergence (name he will drop — one-line only)

**Kullback–Leibler (KL) divergence** is a famous asymmetric score between distributions. Rough English:

> How surprised are you, on average, if nature generates data from true $p$ but you believed $p_\theta$?

He uses “KL-style” distances as the $d$ slot when sketching linear regression and deep nets. You do **not** need the full formula today — only: it is one concrete choice of $d$, and it is **not** a classical metric (asymmetric).

### Circular worry (he raises it)

$d$ seems to need the unknown $p$ as an input.  
Later lectures: **statistics** (law of large numbers, etc.) let you estimate the needed pieces from **samples only**.

**How that will feel in practice (forward pointer):** you rarely plug the true $p$ into a computer. You minimize a **sample-based stand-in** — an average score on $D$ that approximates the population distance $d(p,p_\theta)$. That is the bridge from “distance to truth” to “loss on data,” and it is exactly why ERM (next class) can match this recipe.

### Micro

Two candidate Gaussians $p_{\theta_1}$, $p_{\theta_2}$. If $d(p,p_{\theta_1}) < d(p,p_{\theta_2})$, prefer $\theta_1$.  
With only samples: prefer the $\theta$ that makes $D$ look more typical under $p_\theta$ (details later).

---

## 6. Optimization and $\mathrm{arg\,min}$

<a id="p6-argmin"></a>

### Purpose for the video

**Training** is an optimization problem.

### Minimize

$$
\theta^\star \in \mathrm{arg\,min}_{\theta\in\Theta}\; d(p, p_\theta)
$$

| Symbol | Meaning |
|--------|---------|
| $\min_\theta d(\ldots)$ | the **smallest distance value** (a number) |
| $\mathrm{arg\,min}_\theta d(\ldots)$ | the **parameter $\theta$** that achieves that minimum (the minimizer) |

He stresses: ML wants the **minimizer** $\theta^\star$, not only the min value.

### Micro

$f(\theta)=(\theta-3)^2$.  
$\min f = 0$.  
$\mathrm{arg\,min}\,f = 3$. Training returns $3$, not $0$.

### Convex vs non-convex (one sentence)

If the landscape has one bowl, many algorithms find the bottom reliably.  
If it has many valleys (**non-convex** — common in ML distances + neural nets), local methods may settle in a valley that is not the global best.

---

## 7. First-order gradient descent (picture only)

<a id="p7-gd"></a>

### Purpose for the video

He names **first-order gradient descent** as the modern go-to optimizer (no deep proof yet).

### Idea

To minimize a smooth $J(\theta)$:

$$
\theta \leftarrow \theta - \eta\,\nabla_\theta J(\theta)
$$

- $\nabla J$ = slope / steepest-ascent direction  
- minus sign = go downhill  
- $\eta>0$ = step size (learning rate)

“First-order” = uses first derivatives only (not Hessian / second-order methods).  
$\theta$ may live in $\mathbb{R}^k$ with $k=5$ or $k=10^9$ — same update idea; the space is just higher-dimensional.

```
  J(θ)
    │   ╲
    │    ╲___
    │        ╲___● start
    │            ╲
    │             ● after steps
    └──────────────── θ
```

Deep nets train by repeating this idea with **backpropagation** (chain rule for gradients) — named later as jargon, not derived today.

---

## 8. The three-step recipe (skeleton) + micro walkthrough

<a id="p8-recipe"></a>

### Purpose for the video

This is the **entire architecture** of the lecture. Memorize the skeleton.

```
  Given D ~ unknown p
       │
       ├─ (1) MODEL     choose family p_θ
       ├─ (2) DISTANCE  choose/compute d(p, p_θ)
       └─ (3) TRAIN     θ* = arg min_θ d(p, p_θ)
```

| Step | Human name | Math |
|------|------------|------|
| 1 | Model choice | pick $p_\theta$ family |
| 2 | Loss / distance | pick $d$ (later: sample stand-in) |
| 3 | Training | optimize $\theta$ → get $p_{\theta^\star}$ |

**Different algorithms** ≈ different fills of the three slots  
(e.g. linear + KL-style + closed form → linear regression story; NN + KL-style + GD/backprop → deep net story — he previews these as jargon).

### End-to-end micro (coin)

Unknown coin bias. True law: Bernoulli with heads probability $\theta_{\text{true}}$ (unknown).

| Step | What you do |
|------|-------------|
| Data | $D=$ sequence of $n$ tosses (IID) |
| (1) Model | family: Bernoulli($\theta$), $\theta\in(0,1)$ — already parametric |
| (2) Distance | any score that prefers $\theta$ making $D$ likely (later: KL / likelihood-style) |
| (3) Train | e.g. $\theta^\star=$ fraction of heads in $D$ (a common minimizer for natural scores) |

Output model: “coin with bias $\theta^\star$.”  
Still wrong almost surely if $n$ finite and $\theta_{\text{true}}$ irrational relative to the sample — **but useful** for predicting the next toss.

### Surrogate of a distribution

He often says: estimate the distribution **or some surrogates** of it.

| Target | Example meaning |
|--------|-----------------|
| Full joint density | $p(x,y)$ — richest |
| **Surrogate** | a piece you actually need for the task: $p(y\mid x)$, $p(x)$, a mean function, a decision boundary, … |

You do not always need every detail of $p$ — only enough of it for prediction / generation / decisions. That is why “estimate $p$ or a surrogate” is the honest problem statement.

### Supervised vs unsupervised (one line for Topic 7)

| Setting | What $D$ looks like |
|---------|---------------------|
| **Supervised** | pairs $(x,y)$ (inputs + labels) |
| **Unsupervised** | only $x$ (no labels) — density of $x$, clusters, … |

Deterministic “learn $f:x\to y$” stories struggle to state unsupervised cleanly; density estimation covers both.

### Function approximation ∥ density recipe

| | Function approximation (Lec 01 spirit) | This lecture (density) |
|--|----------------------------------------|-------------------------|
| Unknown object | map $f$ | density $p$ |
| Data | samples of $(x,y)$ | IID samples from $p$ (or pairs) |
| Restrict | parametric $f_\theta$ | parametric $p_\theta$ |
| Score | loss / risk | distance $d(p,p_\theta)$ |
| Fit | optimize $\theta$ | train $\theta^\star=\mathrm{arg\,min}\,d$ |

Same **assume form → score → optimize** skeleton. Different object.

### Empirical risk minimization (ERM) — words only (proved next class)

| Term | Plain meaning |
|------|----------------|
| **Loss** $\ell$ | how bad a prediction/model is on one example |
| **Risk** | *expected* loss under the true law of the data |
| **Empirical risk** | *average* loss on the finite sample $D$ |
| **ERM** | pick $\theta$ that **minimizes empirical risk** on $D$ |

He claims (next lecture): this ERM story is **the same** as the density-distance-train recipe once you connect the loss/distance carefully. Today you only need the vocabulary so the promise is not empty jargon.

### What this lecture does *not* finish

- How to compute $d$ without knowing $p$ (stats later)  
- Proof that **ERM** equals this recipe (next class)  
- Full linear models chapter (starts next)

<a id="p9-surrogate-erm"></a>

### Paper check

1. What is completely unknown in the central ML problem? What is not unknown?  
2. Why use a parametric family instead of “all densities”?  
3. Write $p_\theta$ for a 1D Gaussian; identify $\theta$ and $\dim(\theta)$.  
4. Model vs algorithm: if family is wrong, can training fix it?  
5. Why “metric” in quotes? What is a sample stand-in for $d$?  
6. $\mathrm{arg\,min}$ of $(\theta-2)^2$ is ___ ; min value is ___.  
7. Walk the coin micro through the three recipe steps.  
8. What is a *surrogate* of a distribution? Give one example.  
9. In one sentence each: risk vs empirical risk (ERM).  
10. What does “training” mean in one sentence?

---

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).  
Prior: [Lec 09 Density](../10-Lec09-Density-Function/NOTES.md).


---

## 🗝️ Mathematical Foundations & MathsTerms Bridge

> [!TIP]
> **Foundational Knowledge Base:** This module directly relies upon formal mathematical constructs systematically defined and verified in our central [`MathsTerms`](../../MathsTerms) repository. For visual dependency graphs and multi-track learning roadmaps, consult the [Grand Unified Concept Map](../../MathsTerms/CONCEPT_MAP.md).

| Mathematical Concept | Dedicated Guide | Role & Significance in This Lecture |
| :--- | :--- | :--- |
| **Vector Norms & Inner Products** | [Vector Norms & Inner Products](../../MathsTerms/02-Linear-Algebra-Geometry-and-Tensors/02-Vector_Norms_and_Inner_Products.md) | Euclidean ($L_2$) and Manhattan ($L_1$) distance collapse in high dimensions |
| **Vectors & Matrices** | [Vectors & Matrices](../../MathsTerms/02-Linear-Algebra-Geometry-and-Tensors/01-Vectors_and_Matrices.md) | Geometry of high-dimensional unit hyperspheres and distance concentration |
| **Loss Functions & Regularization** | [Loss Functions & Regularization](../../MathsTerms/03-Multivariate-Calculus-and-Optimization/08-Loss_Functions.md) | Loss design, norm penalties ($L_1, L_2$), and combatting overfitting |
| **Optimization Algorithms** | [Optimization Algorithms](../../MathsTerms/03-Multivariate-Calculus-and-Optimization/09-Gradient_Descent.md) | Ill-conditioned curvature and optimization challenges in high dimensions |

---

## 🌉 Curriculum & Sibling Course Prerequisite Bridges

| Prerequisite Concept | Source Module / Resource | Target Application in Lecture 10 | Verification Check |
| :--- | :--- | :--- | :--- |
| **Overview & Function Approximation** | [Lecture 01: Overview Function Approximation](../../Mathematical-foundation-ml/02-Lec01-Overview-Function-Approximation/NOTES.md) | Universal Function Approximation (UFA) and choosing flexible model families | Differentiate lookup tables from parametric models |
| **IID Assumption & Sample Factorization** | [Lecture 07: IID Assumption](../../Mathematical-foundation-ml/08-Lec07-IID-Assumption/NOTES.md) | Formulating the empirical dataset $\mathcal{D} = \{x_i\}_{i=1}^N$ as i.i.d. draws from true $p$ | State why joint sample probability factors as a product |
| **Continuous Densities & Normalization** | [Lecture 09: Density Function](../../Mathematical-foundation-ml/10-Lec09-Density-Function/NOTES.md) | Continuous density models $p_\theta(x)$ and probability integral definitions | Verify $\int p_\theta(x) dx = 1$ |
| **Information Entropy** | [Lecture 11: Entropy](../../Mathematical-foundation-ml/12-Lec11-Entropy/NOTES.md) | Measuring average surprisal and information content in data-generating distribution | Compute $H(p) = -\sum p(x)\log p(x)$ |
| **Kullback-Leibler Divergence** | [Lecture 12: KL-Divergence](../../Mathematical-foundation-ml/13-Lec12-KL-Divergence/NOTES.md) | Formulating statistical divergence $D_{KL}(p \parallel p_\theta)$ as the canonical distance in Step 2 | Check asymmetric properties of KL divergence |
