# Lec 10 — Challenge With ML

**Video:** [Lec 10 Challenge With ML](https://www.youtube.com/watch?v=767MLwniPKE) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 09 — Density Function](../10-Lec09-Density-Function/NOTES.md)

---

## Table of Contents

1. [Topic 1 — Central problem: estimate unknown p; model = estimator](#topic-1-central-problem-estimate-unknown-p-model--estimator-0011–0426) (00:11–04:26)
2. [Topic 2 — Dataset D, FA history, assume parametric form](#topic-2-dataset-d-fa-history-assume-parametric-form-0426–0752) (04:26–07:52)
3. [Topic 3 — Model choice p_θ; UFA; model vs algorithm](#topic-3-model-choice-pθ-ufa-model-vs-algorithm-0752–1400) (07:52–14:00)
4. [Topic 4 — Distance “metric” d(p, p_θ); begging the question](#topic-4-distance-metric-dp-pθ-begging-the-question-1400–2048) (14:00–20:48)
5. [Topic 5 — Training = argmin; full three-step recipe](#topic-5-training--argmin-full-three-step-recipe-2048–2538) (20:48–25:38)
6. [Topic 6 — Scale, AGI aside, recipe → linreg/DNN/SVM](#topic-6-scale-agi-aside-recipe--linregdnnsvm-2538–3114) (25:38–31:14)
7. [Topic 7 — ERM equivalence next; formal recap](#topic-7-erm-equivalence-next-formal-recap-3114–3531) (31:14–35:31)
8. [External references](#external-references)
9. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: estimate an unknown density $p$ from finite samples when $p$ itself is completely unknown. Method: three steps — choose family $p_\theta$, score with $d(p,p_\theta)$, train by $\mathrm{arg\,min}$. Fork: named algorithms are different fills of those slots; next lecture shows risk minimization is the same system.

**Worldview arc:** from “density estimation under total ignorance” **to** “ML = (model, distance, optimizer); scale changes, the math problem does not.” Also bridges **probabilistic** density estimation to the **deterministic** FA / risk-minimization dialect.

### System context

```
  ╔══════════════════════════════════╗
  ║ Outside                          ║
  ║ · Lec 08–09: estimate P / p      ║
  ║ · Next: ERM ≡ this recipe        ║
  ║ · Then: linear model family      ║
  ╚══════════════╤═══════════════════╝
                 │ this lecture
                 ▼
        ┌────────────────────┐
        │ 3-step ML recipe   │
        └────────────────────┘
```

### Main blueprint

```
  D = IID samples from unknown p
         │
         ▼
  ┌──────────────────────────────────┐
  │ CENTRAL PROBLEM                  │
  │ estimate p (or surrogate)        │
  │ p completely unknown             │
  │ only info = samples              │
  │ model := density estimator       │
  │ “all models wrong; some useful”  │
  └────────────────┬─────────────────┘
                   │
                   ▼
  ┌──────────────────────────────────┐
  │ RECIPE                           │
  │ (1) MODEL   choose p_θ family    │
  │ (2) DISTANCE d(p, p_θ) ≥ 0       │
  │     (“metric” often a divergence)│
  │ (3) TRAIN   θ* = argmin_θ d      │
  └────────────────┬─────────────────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
      p_θ slot   d slot   optimizer
         │         │         │
         └─────────┴─────────┘
                   │ different fills
                   ▼
         linreg / DNN / SVM / …
                   │
         ┌ · · · · ┴ · · · · · · · · ┐
         │ STOP: how to compute d    │
         │ without p; ERM proof next │
         └ · · · · · · · · · · · · · ┘
```

### Scenario walkthrough

1. Given $D$ from unknown $p$ (images, text, …).  
2. Admit: $p$ unknown; only samples.  
3. Choose family $p_\theta$ (e.g. neural net — UFA motivation).  
4. Define score $d(p,p_\theta)$ (larger = worse).  
5. Train: $\theta^\star=\mathrm{arg\,min}\,d$ (gradient descent at scale).  
6. Recognize: same skeleton at 200 params or 8 billion.  
7. Next: show ERM is the same story under another name.

### Failure / contrast path

```
  Demand the “one true model”              ──X──► forget all estimators wrong
  Fix a linear family when truth is curved ──X──► algorithm cannot save you
  Treat d as needing full p with no stats  ──X──► circular stall
  Confuse min value with argmin θ          ──X──► wrong training target
```

### STOP / out of scope

How to compute $d$ from samples alone (statistics later); full ERM equivalence proof (next class); catalog of linear models (starts next); deep backprop derivation; true AGI theory.

### Load-bearing claims (closed-book)

- Central ML problem: estimate unknown $p$ (or a **surrogate**) from IID samples.  
- Model = density estimator; all models wrong, some useful (bias + finite $n$).  
- Recipe: (1) $p_\theta$ (2) $d(p,p_\theta)$ (3) $\theta^\star=\mathrm{arg\,min}\,d$ = training.  
- Model choice ≠ algorithm; algorithm only picks $\theta$ inside the family.  
- “Metric” often a divergence; in practice a **sample stand-in** for $d$ (stats later).  
- Modern go-to: neural $p_\theta$ + first-order gradient descent; scale can be billions of parameters.  
- Same math problem across scale; next = ERM ≡ this recipe.

**Speaker / course:** NPTEL IISc · MFML · Lec 10.

---

## Topic 1: Central problem — estimate unknown $p$; model = estimator (00:11–04:26)

### Where this sits on the master map

**PROBLEM** box — name the challenge before any recipe. Warm-up: [density / estimate](./PREREQUISITES.md#p1-density-estimate), [IID dataset](./PREREQUISITES.md#p2-iid-dataset).

### Board / screenshot

![Central ML problem](./screenshots/composites/ch01-topic-01-central-problem-panel1of1.png)

**Figure — ~00:11–04:20:** boxed central problem; $p$ unknown; model as estimator; “all models wrong…”

### What he is establishing

First, notation comfort from Lec 09: whether he writes $p(x)$, $p(x,y)$, $p(y\mid x)$, or $p(y)$, the **mathematical job is the same** — estimate a non-negative function whose integrals are probabilities. Only the **interpretation** (marginal, joint, conditional) changes with the target you care about. Warm-up: [same job](./PREREQUISITES.md#p1-density-estimate).

He boxes the **central problem of machine learning**: estimate the underlying density / distribution **(or a surrogate** — e.g. only $p(y\mid x)$ when you only need labels). The words “machine” and “learning” are partly nomenclature — “machine” is just a compute unit for now; the real object is the **mathematical** problem.

**Main challenge:** the true $p_X$ is **completely unknown**. Zero structural information. Still we want an estimate. That sounds herculean: if you already knew the shape of $p$, estimation would be easier. Here you know nothing a priori.

**Not totally blind:** you have **samples** drawn from the unknown law. So the central question becomes:

> Given samples from an underlying distribution, can you learn that distribution?

If that problem were fully well-defined and easy, you would not need **hundreds of algorithms** — one sufficient method would do.

**New nomenclature — model:** every algorithm’s estimate of the underlying distribution is called a **model**. A model is an **estimator** of the distribution (the *rule* that turns $D$ into a fitted $\hat p$; the fitted curve is the estimate). Large language models, in this worldview, are estimators of a distribution over text (and related objects).

From estimation theory: **all estimators are wrong by definition** — therefore **all models are wrong**. But some estimators are useful — therefore **some models are useful**. Closed slogan:

> **All models are wrong. Some are useful.**

Why always wrong? Because they are density **estimators**, not the true density handed down from nature: wrong family (bias), finite $n$ (noise), or both. There is no single “correct” density estimator; you compare estimators using metrics. The standing challenge remains: unknown $p$, only samples.

If you still think ML is “pick the one true formula,” this topic already kills that. Instead: estimation under ignorance, scored later by metrics.

You can now name the central problem and the model slogan. Still missing: the concrete dataset object and the historical method that forces assumptions.

### Analogy for this topic only

A blurry photo of a landscape (true density). You get random pixel peeks (samples). Any reconstruction is a “model.”

**If every reconstruction is imperfect, what makes one useful?** A later score that says which estimate helps decisions — not “perfect truth.”

In lecture words: estimate unknown $p$ from samples; model = estimator; all models wrong, some useful.

### Local picture

```
  unknown p  ──?──►  want estimate
       │
       │ only path
       ▼
  samples D
       │
       ▼
  model = estimator of p
  (always imperfect; sometimes useful)
```

**Notice:** $p(x)$, $p(y\mid x)$, … are the **same estimation job** with different targets.

### Bridge

What concrete data object do we start from, and what historical method from function approximation do we copy?

---

## Topic 2: Dataset $D$, FA history, assume parametric form (04:26–07:52)

### Where this sits on the master map

**SETUP** — data + historical method that forces assumptions. Warm-up: [IID](./PREREQUISITES.md#p2-iid-dataset), [parametric](./PREREQUISITES.md#p3-parametric).

### Board / screenshot

![Dataset and parametric assumption](./screenshots/composites/ch02-topic-02-setup-fa-path-panel1of1.png)

**Figure — ~04:26–07:50:** $D$ IID from $p_X$; goal estimate density; assume $p_\theta$.

### What he is establishing

Central question restated: given samples from an underlying distribution, how do we estimate it? He will give a **broad framework**, then many examples later in the course.

**Starting object:** dataset $D$ whose points are **IID** draws from an unknown underlying density $p_X$. He flags a **misnomer**: strictly, you cannot sample from a density function; you sample from a **distribution**. Writing “IID from density $p$” is a leap of faith he accepts for notation.

**Goal:** estimate the underlying **density**.

**Historical parallel (function approximation, Lec 01):** you had samples $(x,y)$ from two sets and wanted an unknown map. Historical method: you are totally blind without a stick — so you **assume a form** on the unknown function, then **solve an optimization problem**. Same spirit here. Side-by-side: [FA ∥ density recipe](./PREREQUISITES.md#p9-surrogate-erm).

**Mathematical move for density:**

1. We estimate a function range$(X)\to\mathbb{R}_+$ (definition of density).  
2. Assume it comes from a **parametric family** $p_\theta$.

So: assume a **parametric functional form** on $p_X$, denoted $p_\theta$.

**Why not stay unrestricted?** The set of all densities is effectively **infinite-dimensional**. Finite $D$ cannot pin down an arbitrary curve. A parametric family collapses the problem to a finite list of free numbers $\theta$ so optimization has a place to search.

If you skip the assumption and demand pure blindness, you have no handle for algorithms. Instead: restrict the universe of densities to a parameterized family, then optimize inside it.

You can now write the starting point ($D$, IID, goal estimate $p$) and the first leap ($p_\theta$). Still missing: concrete parametric examples and the model-vs-algorithm split.

### Analogy for this topic only

You must guess a secret recipe tasting only a few spoonfuls. Historical cooks start by assuming a **recipe family** before tuning one dial.

**Without any family assumption, what are you optimizing over?** An infinite wild set of possible recipes — no algorithm handle.

In lecture words: $D$ IID (from density, loosely); goal estimate $p$; assume parametric $p_\theta$.

### Local picture

```
  D = {x¹,…,xⁿ}  ~  unknown p   (IID; density wording ≈ leap)
         │
         ▼
  Goal: estimate p : range → R₊
         │
         │ FA history: assume form + optimize
         ▼
  Assume p ∈ { p_θ }   ← parametric family
```

**Notice:** “sample from density” is informal; “sample from distribution” is strict.

### Bridge

What does “parametric” mean with examples, and why is choosing the family a leap of faith?

---

## Topic 3: Model choice $p_\theta$; UFA; model vs algorithm (07:52–14:00)

### Where this sits on the master map

**MODEL** — recipe step 1. Warm-up: [parametric](./PREREQUISITES.md#p3-parametric), [model vs algorithm](./PREREQUISITES.md#p4-model-vs-algo).

### Board / screenshot

![Parametric model choice](./screenshots/composites/ch03-topic-03-model-p-theta-panel1of1.png)

**Figure — ~07:52–14:00:** parametric examples; model menu; UFA; model vs algorithm; recipe step 1.

### What he is establishing

A **parametric function** is a function object controlled by a **few parameters**. Classic picture: a line.

**Example — affine / linear form** on a vector $x\in\mathbb{R}^d$: something like $w_1^\top x + w_2$ with $\theta=\{w_1,w_2\}$. Changing slope and intercept gives different lines; in higher $d$, a **hyperplane**.

**Do not over-read this as “the density *is* a line.”** A raw affine formula is not automatically a valid density (need $p\ge 0$ and $\int p=1$). In the lecture the line is a **toy parametric shape** to show finite $\theta$; for probability densities the clean example is next — Gaussians. In the *broad* recipe, $p_\theta$ may also stand for related parametric objects (regression maps, conditional densities, …) that fill the same “model slot.”

**Example — Gaussian density** as the assumed form. Standard named densities exist: Gaussian, exponential; discrete Bernoulli; etc. You may adopt them as ready-made $p_\theta$ choices. For a Gaussian, $\theta$ includes mean $\mu\in\mathbb{R}^d$ and covariance structure $\Sigma$ (he describes sigma related to $d\times d$).

**Model choice** = choose which $p_\theta$ family. The course will show about **ten** examples. Modern names on the menu:

- linear regression, logistic regression  
- neural network / deep net  
- kernel machine  
- GMM (Gaussian mixture)  
- transformer  

All of these, in his framing, are **choices of $p_\theta$** (fills of the model slot).

Once you fix the family, **degrees of freedom are already lost**. You make a **leap of faith** that the true density is a member of that family. You will **not** know whether that is true.

**Universal function approximation (UFA):** some families (mixture densities; neural networks) can approximate **any** function to arbitrary closeness if parameters are chosen well. That is why neural nets are today’s go-to $p_\theta$: UFA capability — a flexible family so the leap of faith is less brutal than “must be a line.”

**Trap:** if you pick a **linear** model family but the true density shape is **quadratic**, you are stuck with a very bad model. “All models are wrong” — this one is *more* wrong. The **algorithm will not rescue the family**. The algorithm only answers: *given that reality is a line, which line?* There are **infinitely many** lines (vary intercept and slope) — still a hard search, but *inside* a bad box.

**Recipe step 1 (named):** assume parametric form $p_\theta$ on $p_X$ — **model choice**.

If you mix “I used Adam optimizer” with “I chose ResNet,” remember: optimizer is not the family. Instead: family first; search inside second.

You can now state recipe step 1 and the model≠algorithm trap. Still missing: how to score which $\theta$ inside the family.

### Analogy for this topic only

You swear the mystery curve is “a straight road.” Your GPS (algorithm) can only pick *which* straight road.

**If the truth is a hairpin mountain path, what does better GPS software still fail to offer?** Any non-straight road — family was wrong.

In lecture words: $p_\theta$ with parameters; leap of faith; UFA; model ≠ algorithm; step 1 = model choice.

### Local picture

```
  Recipe step 1: MODEL CHOICE
       │
       ▼
  pick family {p_θ}
       │
       ├─ linear / hyperplane form
       ├─ Gaussian (μ, Σ)
       ├─ NN / transformer / GMM / …
       │
       ▼
  leap of faith: true p ∈ family
       │
       ├─ UFA families (mixtures, NNs) ≈ can approximate anything
       └─ bad family (line vs quadratic) → stuck; algo only picks θ
```

**Notice:** infinite members **inside** a family still leave a hard search problem for step 2–3.

### Bridge

Given infinitely many members of the family, how do we score which $\theta$ is better?

---

## Topic 4: Distance “metric” $d(p,p_\theta)$; begging the question (14:00–20:48)

### Where this sits on the master map

**METRIC** — recipe step 2. Warm-up: [distance / divergence](./PREREQUISITES.md#p5-distance).

### Board / screenshot

![Distance between densities](./screenshots/composites/ch04-topic-04-distance-metric-panel1of1.png)

**Figure — ~14:00–20:45:** $d(p_X,p_\theta)$; parameters; quotes on “metric”; begging the question.

### What he is establishing

After model choice, he still has not named a full “algorithm.” Next logical step: **define/compute a distance** between the true density $p_X$ and the model $p_\theta$.

**Why?** Fix “normals” as the family. There are **infinitely many** normals (vary mean and variance). You need a notion of **how good or bad** a particular normal is.

**Parameters:** the $\theta$ values are called **parameters**. Fixing $\theta$ yields one concrete model instance. Mathematically, if you had a distance between densities, **larger distance ⇒ poorer model**.

**Notation:** he wanted capital $D$ for distance but $D$ is already the dataset — so **small $d$** for the distance.

**Quotes on “distance metric”:** in mathematics a metric needs properties such as **symmetry** and the **triangle inequality**. Many scores used in ML **do not** have those properties. They are **divergences** / distance-like discrepancies — hence the quotes. Not metrics in the strict sense.

**Formal shape:**

$$
d : (p_X, p_\theta) \mapsto \mathbb{R}_+
$$

Non-negative (cannot be a negative “distance”). Takes two distributions/densities as input.

Model choice is “easy” (pick a function class). Defining $d$ is also a design choice. **What differentiates ML algorithms:**

1. choice of $p_\theta$  
2. choice of $d$  
3. (preview) choice of **optimization algorithm** for step 3  

**Begging the question:** $d$ is written with $p_X$ as an input, but $p_X$ is unknown. Aren’t we stuck at step one? His answer: **no** — there are **statistical methods** (law of large numbers and other probability tools) that let you compute what you need without full knowledge of $p$. Those tools are not “useless math”; the course will use them later. For now: assume $d$ can be computed; proceed.

**What that will look like in code (conceptually):** you minimize a **sample-based stand-in** for $d$ — typically an average score on $D$ that approximates the population distance. That stand-in is the bridge to “loss on data” / empirical risk (next class). Warm-up: [distance + stand-in](./PREREQUISITES.md#p5-distance).

If you freeze because “I don’t know $p$ so I can’t write $d$,” hold the circularity — statistics will break it. Instead: design $d$, then estimate it from samples later.

You can now state recipe step 2 and why “metric” is in quotes. Still missing: turning the score into a training objective.

### Analogy for this topic only

Two candidate weather maps vs the true climate. You need a scoring rule for “how wrong is this map?”

**If you never observe the full climate field, what still lets you rank maps?** Long-run station averages standing in for pieces of the score — the stats escape hatch.

In lecture words: $d(p,p_\theta)\ge 0$; not always a true metric; circularity resolved by stats later.

### Local picture

```
  family fixed (e.g. all Gaussians)
       │
       ▼
  infinite θ  →  need score
       │
       ▼
  d(p, p_θ) ≥ 0     larger = worse
  (“metric” ≈ divergence; may lack symmetry / triangle)
       │
       │ problem: d seems to need unknown p
       ▼
  stats / LLN later → compute from samples
```

**Notice:** algorithm identity = $(p_\theta,\, d,\, \text{optimizer})$ triple, not one magic name.

### Bridge

Once we can score models, what is the third step — and what does the ML world call it?

---

## Topic 5: Training = $\mathrm{arg\,min}$; full three-step recipe (20:48–25:38)

### Where this sits on the master map

**TRAIN** — recipe step 3 + whole recipe. Warm-up: [argmin](./PREREQUISITES.md#p6-argmin), [recipe](./PREREQUISITES.md#p8-recipe).

### Board / screenshot

![Training as argmin](./screenshots/composites/ch05-topic-05-training-argmin-panel1of1.png)

**Figure — ~20:48–25:35:** $\theta^\star=\mathrm{arg\,min}\,d$; training; full recipe; NN + GD.

### What he is establishing

**Step 3:** solve an optimization problem. Seek $\theta^\star$ (optimal parameters) such that $d(p_X,p_\theta)$ is **minimized**:

$$
\theta^\star \in \mathrm{arg\,min}_{\theta}\, d(p_X, p_\theta)
$$

In ML language, this minimization is **training** — training the model.

Course framing: the content of ML is not new magic; the goal is one **broad framework** that can contain many textbook stories.

**Problem reduces to:** given $D$, solve that optimization.

**$\mathrm{arg\,min}$ carefully:** if you minimize a function of $\theta$, $\mathrm{arg\,min}$ is the **point in $\theta$-space** where the minimum is achieved — the **minimizer**. It is **not** the minimum **value** of the function. ML wants the parameters, not only “how small the distance got.”

**Full recipe (walkthrough he wants memorized):**

1. Input: dataset $D$ of samples from an unknown distribution; estimate the distribution or a **surrogate** of it (not always the full joint — e.g. only $p(y\mid x)$ if labels are the job; [surrogates](./PREREQUISITES.md#p9-surrogate-erm)).  
2. Assume a **parametric form** $p_\theta$ of the underlying object.  
3. Define and compute a **distance** between true and model (in practice: a computable stand-in from $D$).  
4. Find parameters by solving the optimization that seeks the **minimizer** of that distance.  

> “That’s ML for you.”

Micro you can replay with a coin: [end-to-end recipe](./PREREQUISITES.md#p8-recipe).

**Course questions going forward:**

- Which $p_\theta$?  
- Which $d$?  
- How to optimize? (Often **non-convex** — classical convex solvers do not apply cleanly.)

**Today’s industrial defaults (he states):** go-to $p_\theta$ = **neural network**; go-to optimizer = **first-order gradient descent**. Even paths people market as AGI-scale systems still rest on simple first-order GD, not necessarily fancy second-order / conjugate methods from classical optimization courses.

If training means “run library fit,” decode it as $\mathrm{arg\,min}_\theta d$. Instead: own the three steps.

You can now write the full recipe and define training. Still missing: scale, named algorithms as slot-fills, and the ERM bridge.

### Analogy for this topic only

You have a radio dial (parameters) and a static meter (distance). Training = twist until static is minimized.

**Do you report the final dial position or the static number as “the trained model”?** The dial position — that is $\mathrm{arg\,min}$, not the min value.

In lecture words: training = minimize $d$; $\mathrm{arg\,min}$ = minimizer; full recipe; NN + first-order GD.

### Local picture

```
  (1) choose p_θ
  (2) choose d(p, p_θ)
  (3) θ* = argmin_θ d(p, p_θ)   ← “training”

  argmin = which θ
  min    = how small d became
```

**Notice:** non-convex $d$ makes step 3 hard even when (1) and (2) look clean on paper.

### Bridge

How big is $\theta$ in modern systems, and how do famous algorithm names sit inside this recipe?

---

## Topic 6: Scale, AGI aside, recipe → linreg / DNN / SVM (25:38–31:14)

### Where this sits on the master map

**SCALE + INSTANCES** — same recipe at extreme scale; name-drops as fills of the slots. Warm-up: [GD](./PREREQUISITES.md#p7-gd), [recipe](./PREREQUISITES.md#p8-recipe).

### Board / screenshot

![Scale and recipe instances](./screenshots/composites/ch06-topic-06-scale-agi-instances-panel1of1.png)

**Figure — ~25:38–31:10:** billion parameters; anecdote; AGI framing; linreg/DNN/SVM; density vs distribution Q.

### What he is establishing

Optimization will use **numerical first-order gradient descent**. Modern models (e.g. GPT-class) have **billions** of parameters — optimization in a **billion-dimensional** space. The course will start with a handful of parameters; appreciate the scale jump.

**Anecdote (ISI economist):** a colleague called a model with **eight parameters** “very large” and wanted fewer. In modern ML, people casually discuss **8 billion** parameters. Linear models on $100$-d data might have roughly $200$ parameters (weights + intercepts). Today’s models: billions.  

**Satisfaction:** no matter how scale and fashion change, the **underlying mathematical problem remains the same** — the recipe above.

**AGI-scale story (aspirational, not a proof):** imagine $p$ as amalgamation of human knowledge; sample space as the universe of documents, text, images produced by humanity; throw massive data + compute at estimating that $p$ and AGI-like behavior emerges. **Why** that works is still open. Is it the only path to intelligence? Probably not: a child needs far less data and energy. Human learning can also be cast as distribution estimation (act → feedback → update), but brains burn orders of magnitude less power than data-center models — maybe a parallel model of intelligence exists. (He marks this as another day’s discussion.)

**Recipe instances (jargon allowed; do not panic):**

| $p_\theta$ | Distance (named) | Optimizer | Named method |
|------------|------------------|-----------|--------------|
| linear model | KL-style divergence | deterministic / closed form | **linear regression** (his sketch) |
| compositions of sigmoids (NN) | KL-style | numerical GD + chain rule (**backprop**) | **deep neural network** |
| kernel formulation | (as set) | dual optimization methods | **SVM** |

**KL** here is just one concrete asymmetric score in the $d$ slot ([one-line KL](./PREREQUISITES.md#p5-distance)) — not a full derivation. Read the table as: **same three slots, different fills → different logos.**

Basic idea unchanged: choose $p_\theta$, choose distance, solve optimization — **that’s ML**.

**Student Q — begging the question again:** how compute $d$ without $p$? Deferred: **statistics / probability theory** will help next.

**Student Q — density vs distribution:** does switching from estimating distributions to densities change the scale of the problem? **No.** When both are defined, they have a **one-to-one** relationship. We work with densities for **mathematical convenience** (e.g. Gaussian density is easy; Gaussian CDF needs special functions / tables).

If brand names intimidate you, map them back to the three slots. Instead: recipe first, logos second.

You can now see scale and brand names as cosmetics on one recipe. Still missing: the deterministic ERM dialect and formal recap.

### Analogy for this topic only

Three restaurants all serve dinner. One uses a grill, one a multi-layer oven, one a dual steamer.

**What is shared if menus look totally different?** The job structure: choose cooking setup → score quality → adjust until score is best.

In lecture words: billions of params; same math; recipe fills become linreg/DNN/SVM; density for convenience.

### Local picture

```
  same recipe
     │
     ├─ 8 parameters (old “large”)
     ├─ ~200 (linear, 100-d)
     └─ ~8×10⁹ (modern nets)
     │
     ▼
  still: model → d → argmin

  density ↔ distribution (when both exist): 1–1
  prefer density for ease of computation
```

**Notice:** AGI talk is worldview, not a new formula in the recipe.

### Bridge

How does this probabilistic recipe connect to the deterministic “risk minimization” story many textbooks start with?

---

## Topic 7: ERM equivalence next; formal recap (31:14–35:31)

### Where this sits on the master map

**BRIDGE + RECAP** — close the loop; promise ERM ≡ density recipe. Warm-up: [recipe](./PREREQUISITES.md#p8-recipe), [ERM / surrogates](./PREREQUISITES.md#p9-surrogate-erm).

### Board / screenshot

![ERM next and recap](./screenshots/composites/ch07-topic-07-erm-next-recap-panel1of1.png)

**Figure — ~31:14–35:30:** deterministic FA detour; ERM ≡ distribution estimation; class recap.

### What he is establishing

**Detour:** Lec 01 offered a **deterministic / non-probabilistic** view of ML — learn a mapping from $x$ to $y$ by assuming a form and optimizing. In that view it is awkward to talk about **unsupervised** learning cleanly (only $x$, no labels $y$). Density estimation covers both supervised pairs and unsupervised $x$-only data. Still, the skeleton is familiar: assume form + optimize.

**Two worldviews on $(x,y)$:**

- Probabilistic: ranges of random variables $X,Y$.  
- Deterministic: points in sets $\mathbb{R}^d,\mathbb{R}^k$; learn a function between sets.

The deterministic path leads to **risk minimization** / function-learning frameworks (standard ML 101). Vocabulary (proved next class, not today):

| Term | Meaning |
|------|---------|
| Loss | how bad one example is under a model |
| Risk | expected loss under the true data law |
| Empirical risk | average loss on finite $D$ |
| **ERM** | choose $\theta$ that minimizes empirical risk |

**Claim for next class:** **empirical risk minimization (ERM)** is **exactly the same** as the distribution / density estimation + parameter-fitting recipe of this lecture (once the loss is tied to the distance/stand-in carefully). He wants equivalence fixed so textbook paths do not feel like different subjects.

**Textbook landscape:** Christopher Bishop’s treatment is often more non-probabilistic; Kevin Murphy’s *Probabilistic Machine Learning* is probabilistic by title and design. Both exist in the literature; equivalence means you are doing the **same thing** when you switch books.

**End-of-class formal statement of the ML problem:**

- Given samples drawn **IID** from an unknown underlying distribution  
- Estimate that distribution **or surrogates** of it  
- Challenge: the distribution is unknown  

**General recipe (recap):**

1. Assume a **parametric form**  
2. Define/compute a **distance** between true and model  
3. **Solve an optimization problem** for the parameters  

Next class: start from risk minimization, show equivalence, then begin the **linear family of models**.

If tomorrow’s ERM notation looks foreign, map it onto today’s three steps. Instead: one architecture, two dialects.

You can now recite the formal ML problem and recipe closed-book, and know what next class will prove.

### Analogy for this topic only

Two maps of the same city: street names vs GPS coordinates.

**Before the equivalence proof, what mistake do students make when books switch dialects?** Treating them as different subjects instead of two labels for one system.

In lecture words: ERM ≡ this recipe (next); recap problem + three steps; then linear models.

### Local picture

```
  Today:
    IID samples → estimate p / surrogate
    recipe: p_θ → d → argmin

  Next:
    risk minimization / ERM  ≡  today’s recipe
    then: linear family of models

  Textbooks:
    Bishop (more non-prob) · Murphy (probabilistic)
    same activity after equivalence
```

**Notice:** end recap is the closed-book checklist for the whole lecture.

### Bridge

Next: ERM equivalence, then linear models — still the same three-step system.

---

## External references

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| [StatQuest — Maximum Likelihood, clearly explained](https://www.youtube.com/watch?v=XepXtl9YKwc) | Topics 1–5 | Estimation + scoring models from data |
| [3Blue1Brown — Gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Topics 5–6 | First-order GD picture for training |
| [StatQuest — Neural Networks / backprop playlist entry](https://www.youtube.com/watch?v=CqOfi41LfDw) | Topic 6 | Chain-rule training jargon in context |
| [Lec 09 package — Density](../10-Lec09-Density-Function/NOTES.md) | Topics 1–2, 6 | $p$ vs $P$; why densities |
| [Lec 08 package — Distribution estimation](../09-Lec08-Distribution-Estimation/NOTES.md) | Topics 1–2 | Given $D$ estimate $P$ (rewritten as $p$) |
| [Murphy — Probabilistic ML (book site)](https://probml.github.io/pml-book/) | Topic 7 | Probabilistic textbook pole he names |
| [Bishop — PRML (author page / book info)](https://www.microsoft.com/en-us/research/people/cmbishop/) | Topic 7 | Classic more “pattern recognition” / non-prob flavored pole |

---

## Sources

- Video: [Lec 10 Challenge With ML](https://www.youtube.com/watch?v=767MLwniPKE)  
- Channel: NPTEL — Indian Institute of Science, Bengaluru  
- Skill: `youtube-lecture-tutor`  
- Captions cleaned via timed transcript / claim sheets  
