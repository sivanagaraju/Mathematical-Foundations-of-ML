# Lec 03 — f-Divergence and Examples

**Video:** [Lec 03 f-Divergence and Examples](https://www.youtube.com/watch?v=LR9UQXY_IU8) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 02 problem formulation](../15-Lec02-Generative-Models-Problem-Formulation/NOTES.md)  
**Course:** Mathematical Foundations of **Generative AI** (~43 min)  
**Speaker:** NPTEL IISc · KL, reverse KL, Jensen–Shannon, total variation

---

## Table of Contents

1. [Topic 1 — Two jobs: estimate and sample](#topic-1-two-jobs-estimate-and-sample-0002–0316) (00:02–03:16)
2. [Topic 2 — Implicit neural samplers; any law](#topic-2-implicit-neural-samplers-any-law-0316–0822) (03:16–08:22)
3. [Topic 3 — Recipe: model, divergence, train](#topic-3-recipe-model-divergence-train-0822–1328) (08:22–13:28)
4. [Topic 4 — The hole: D without either density](#topic-4-the-hole-d-without-either-density-1328–1522) (13:28–15:22)
5. [Topic 5 — Primitive generator Z → G_θ](#topic-5-primitive-generator-z--g_θ-1522–2034) (15:22–20:34)
6. [Topic 6 — Samples are not the law](#topic-6-samples-are-not-the-law-2034–2320) (20:34–23:20)
7. [Topic 7 — Infinite support and multimodal Z](#topic-7-infinite-support-and-multimodal-z-2320–2700) (23:20–27:00)
8. [Topic 8 — VDM family; f-divergence definition](#topic-8-vdm-family-f-divergence-definition-2700–3141) (27:00–31:41)
9. [Topic 9 — Properties and the named family](#topic-9-properties-and-the-named-family-3141–3618) (31:41–36:18)
10. [Topic 10 — Mode-cover vs junk; variational next](#topic-10-mode-cover-vs-junk-variational-next-3618–4230) (36:18–42:30)
11. [External references](#external-references)
12. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

This lecture restates generative modeling as two jobs: estimate an unknown density $p_x$ from $n$ IID samples, **and** learn to draw new points. It installs a recipe — guess a parametric **model** $p_\theta$, pick a **divergence** $D(p_x\|p_\theta)$, **train** by minimizing $D$ — then shows the hole: we know neither density. The primitive machine is noise $Z\sim\mathcal{N}(0,I)$ through a net $G_\theta$. The first algorithm family is **variational divergence minimization (VDM)** (GANs live here). It starts by defining **$f$-divergence**, which is **not** a metric.

**Worldview arc:** from “estimate $p_x$” **to** “choose an $f$ so $G_\theta$ can match $p_x$ and then sample.”

### System context

```
  ╔══════════════════════════════════════╗
  ║ Outside: architectures (tutorials)   ║
  ║ Outside: variational estimate of D   ║
  ║          (next lecture)              ║
  ╚════════════════╤═════════════════════╝
                   │ this lecture (~43 min)
                   ▼
        ┌──────────────────────────┐
        │ Recipe + f-divergence    │
        │ family and behaviors     │
        └──────────────────────────┘
```

### Main blueprint

```
  ╔════ GOAL ════╗
  ║ n IID ~ p_x  ║
  ║ (p_x unknown)║
  ║ 1) estimate  ║
  ║ 2) sample    ║
  ╚══════╤═══════╝
         │
         ▼
  ┌─ MODEL p_θ ─────────────┐
  │ mixture (explicit)  or  │
  │ neural net (implicit)   │
  └──────────┬──────────────┘
             │ need a score
             ▼
  ┌─ D(p_x ∥ p_θ) ──────────┐     ──X──  not a metric
  │ f-div: ∫ p_θ f(p_x/p_θ) │           (no triangle)
  │ f convex, lsc, f(1)=0   │
  └──────────┬──────────────┘
             │ we have only samples
             ▼
  ┌─ OBSTACLE ──────────────┐
  │ know neither density    │
  │ have: data cloud +      │
  │       G_θ(Z) cloud      │
  └──────────┬──────────────┘
             │ primitive machine
             ▼
  Z ~ N(0,I) ──► G_θ ──► x̂ ~ p_θ
             │
             ▼
  train: θ* = argmin D    then sample = push Z through G_θ*
             │
     ┌───────┴────────┐
     ▼                ▼
  forward KL       reverse KL
  f(u)=u log u     (exercise: log u)
  punishes miss    punishes junk
     └───────┬────────┘
             ▼
        JSD / TV / χ² / Hellinger
             │
  ┌ · · · · ·┴ · · · · · ┐
  │ STOP: how to compute │
  │ D from samples only  │
  │ → variational next   │
  └ · · · · · · · · · · ┘
```

### Scenario walkthrough

```
  MNIST pile D  (samples of p_x, not p_x)
       │
       ▼
  draw Z ~ N(0,I)  →  G_θ  →  fake digits  (samples of p_θ)
       │
       ▼
  want D(p_x ∥ p_θ) small
       │  pick f
       ▼
  u log u   →  KL   →  cover every digit, maybe ugly hybrids
  reverse   →  clean 3s, drop the 8s
  JSD       →  pressure both sins
       │
       STOP: still cannot evaluate the integral (no p_x, no p_θ)
```

### Failure / contrast path

```
  only estimate p, never sample     ──X──►  not generative
  call D_f a metric                 ──X──►  no triangle
  treat p(x) as P({x})              ──X──►  density height ≠ probability
  start Z on a short interval       ──X──►  cannot grow support
  train only with forward KL        ──X──►  junk barely punished
```

### STOP / out of scope

- How to **estimate** $D_f$ from two sample clouds (variational calculus — next class).
- Architectures (CNN / transformer) except in passing.
- Proofs of $D_f\ge 0$ and $=0$ iff $p_x=p_\theta$ (homework).

### Load-bearing claims

- Generative modeling = estimate $p_x$ **and** sample.
- Model $p_\theta$ is a parametric surrogate (mixture or net).
- We usually have **samples**, not the densities.
- Primitive sampler: $Z\sim\mathcal{N}(0,I)$ through $G_\theta$.
- $D_f(p_x\|p_\theta)=\int p_\theta\,f(p_x/p_\theta)\,dx$ with $f$ convex, lsc, $f(1)=0$ — **not** a metric.
- KL is $f(u)=u\log u$; reverse KL is a *different* $f$; JSD mixes their behaviors.
- Forward KL punishes missing modes; reverse KL punishes junk.

**Speaker / course:** NPTEL IISc, Mathematical Foundations of Generative AI — Lecture 03.

---

## Topic 1: Two jobs: estimate and sample (00:02–03:16)

### Where this sits on the master map

This is the **GOAL** box. Last lecture left ML as “estimate an unknown law.” This course adds a second verb: **sample**. Warm-up: [IID cloud](./PREREQUISITES.md#p2-iid), [density vs samples](./PREREQUISITES.md#p1-density).

### Board / screenshot

![Generative modeling: D ~ iid p_x unknown; estimate and sample; p_θ as NN model](./screenshots/composites/ch01-topic-01-two-jobs-estimate-sample-panel1of1.png)

**Figure — ~00:29–02:56:** “Generative Modeling.” Given $D=\{x_1,\ldots,x_n\}\sim_{\mathrm{iid}}p_x$ (unknown). Goal: estimate $p_x$ **and** learn to sample from it. Principle (i): assume a parametric family $p_\theta$, represented using deep nets (the **model**).

### What he is establishing

Machine learning, in his courses, is one problem: you are given samples from an unknown law $p_x$, and you must estimate that law. Generative modeling does **not** replace that problem. It **adds a burden**.

The $n$ points are IID draws. The tilde means “sampled from.” He will use **sample** as a verb (“sample from $p$”) and as a noun (“you are given samples”). Context decides.

Where do the numbers come from? A random experiment is repeated. A random variable maps outcomes to the range. What you write in the notebook are **range points**. The **distribution** of those points is the **pushforward** of the original probability through that function.

Sampling, as a process, is repeating that experiment — equivalently, asking the random variable for another range point. After training, a generative model must be able to do *that*, not only name $p_x$.

The whole course asks two questions of the same pile $D$: how do you estimate the law, and how do you learn to draw more points from its range?

Treating the list $D$ itself as the distribution is the wrong move — that is table lookup, not a law. The right move is: $D$ is evidence about an unknown pushforward $p_x$, and you must also learn to draw new range points.

You can now state both jobs. What is still open: which machines actually *sample*, and which only estimate.

### Analogy for this topic only

You inherited a box of unlabeled photos from one camera. **Can you describe the camera’s taste, and can you take a new photo that belongs in the box?** Describing the taste is estimation. Clicking the shutter again is sampling. This course insists on both.

In lecture words: box = $D$, camera law = $p_x$, new click = sample.

### Local picture

```
  random experiment  →  ω ∈ Ω
         │  RV
         ▼
      x = X(ω)     ←  what we write down
         │
         ▼
   n IID copies = dataset D     (p_x unknown)

  JOB 1: estimate p_x
  JOB 2: draw new x's  (repeat the experiment)
```

Notice: the distribution is a pushforward, not “the list $D$ itself.”

### Bridge

We have two jobs, but most estimators from the previous course only did job 1. The next box sorts samplers from non-samplers and says the target law need not be a marginal.

---

## Topic 2: Implicit neural samplers; any law (03:16–08:22)

### Where this sits on the master map

**SAMPLER CLASS.** The leftover from Topic 1 is “who can actually draw new $x$?” He also widens $p_x$ to joints and conditionals. Warm-up: [IID](./PREREQUISITES.md#p2-iid).

### Board / screenshot

![Goal board still up: parametric p_θ via deep nets](./screenshots/composites/ch02-topic-02-implicit-neural-any-law-panel1of1.png)

**Figure — ~03:43–07:51:** the *same* goal tablet is still up (estimate + sample; $p_\theta$ via nets). Last tile begins line (ii) “Define & estimate a divergence.” Spoken overlay — not new handwriting — is the payload: previous estimators vs samplers; GMM scales badly; LLMs are conditional; architectures stay out of this lecture.

### What he is establishing

The previous course already estimated laws: non-Bayesian **KL minimization**, which is **maximum likelihood**, plus Bayesian methods, plus nonparametric nearest neighbors, Parzen windows, kernels. Most of those machines **do not sample**. A nearest-neighbor estimator is not a sampler. A Bayes classifier estimates a law and still does not hand you a new $x$.

This course graduates from estimate-without-sample to estimate-**with**-sample. You cannot sample without estimating *something* — implicitly or explicitly — but the **focus** is the draw. Most methods here will be **implicit** distribution estimators.

Classical **mixture models / GMMs** *are* samplers: they write $p_\theta$ and they can draw. They **do not scale** to high dimension (curse of dimensionality). So the course turns to **neural samplers**: universal approximators as the engine, implicit $p_\theta$, sampling as the product.

The unknown law need not be a **marginal**. It can be a joint, a marginal, or a **conditional**. Every chatbot you type at is a **conditional sampler**: the prompt is $Y$, the completion is drawn from $P(X\mid Y)$. He still writes $p_x$ for brevity. When an algorithm needs an extra conditioning wire (adversarial nets, for example), he will say so. Do not walk out thinking “only marginals.”

He will **not** teach architectures. The same algorithm is meant to sit on a CNN, an RNN, or a transformer. Diffusion and autoregressive models get a one-line architectural aside; tutorials may show more.

You can now say what a neural sampler is and that $p_x$ is a stand-in for any of joint / margin / conditional. Still missing: the three-line training recipe.

### Analogy for this topic only

A weather model that only **scores** tomorrow’s rain is job 1. A model that can **emit** a new weather map is job 2.

**Can last course’s nearest-neighbor scorer click the shutter and hand you a new map?** No. That is why he graduates to emitters. The maps can be “weather given tonight’s radar” (conditional), not only “weather” (marginal).

In lecture words: scorer = estimator, emitter = sampler, radar = $Y$.

### Local picture

```
  last course                         this course
  estimate p                     →    estimate AND sample
  kNN, Bayes clf  ──X──► sample
  GMM             ══════► sample     but curse of dimension
                                      │
                                      ▼
                               neural implicit sampler
                                      │
                    p_x may be  joint / marginal / P(X|Y)
                    GPT = conditional sampler
```

Notice: “implicit” means we may never write a formula for $p_\theta$ — only a way to draw from it.

### Bridge

We know the *kind* of machine. We still lack the **score** that tells $p_\theta$ it is close to $p_x$, and the name of the optimization that moves $\theta$.

---

## Topic 3: Recipe: model, divergence, train (08:22–13:28)

### Where this sits on the master map

**RECIPE.** Three boxes: assume a family, define $D$, minimize. Warm-up: [divergence vs metric](./PREREQUISITES.md#p6-div-vs-metric).

### Board / screenshot

![Assume parametric family p_θ; model via deep nets](./screenshots/composites/ch03-topic-03-recipe-model-div-train-panel1of1.png)

**Figure — ~08:53–12:56:** the three-line recipe fills in. (i) Assume a parametric family $p_\theta$ (deep nets = the **model**). (ii) Define and *estimate* a divergence (he still writes “distance metric” here). (iii) Solve an optimization over $\theta$ to minimize that $D$. Spoken: that solve is “training”; for composite nets it is backprop.

### What he is establishing

Assume the unknown density lives in a **parametric family** $p_\theta$. Classically that family is an **explicit** density — a **mixture** is a linear combination of simpler densities. Or $p_\theta$ is whatever a neural net induces. Starting point: the unknown belongs to *some* family you can move with parameters.

That surrogate **is the model**. When he says “model” in this course, he means $p_\theta$, not a GitHub repo.

The tablet’s second line is “define **and estimate**” a divergence. That second verb is the hole of Topic 4: you will not plug two closed-form hills into an integral. You will have to *estimate* $D$ later.

Then you need a number that says how far $p_\theta$ sits from $p_x$. That number is a **divergence** (he also says distance, then later takes the word back). Same job as a ruler between points, now between two hills.

He assumes you can tell a **distribution function** from a **density function**, then announces he will mix the English. All data in the course are assumed to have densities. Definitions will be written on **densities** for easier integrals; they extend to distributions. Notation: small $p$ = density; script / double-struck $P$ = distribution.

Then **optimize** $\theta$ to make $D$ small. “Training a model” is a fashionable name for that solve. Because $p_\theta$ is usually a **composite** net, the solve uses gradients and the chain rule — **backpropagation**. He assumes you already own that.

Course in a nutshell: samples → pick a model family → pick a $D$ → solve. The rest is different families, different $D$s, different solvers.

Calling that solve anything except an optimization problem (and, for nets, backprop) is decoration. The wrong move is to invent a new “training metaphysics.” The right move is: $D$ is a scalar, $\theta$ moves to shrink it.

You can now recite the three-step recipe. Still missing: you do not actually *have* either hill.

### Analogy for this topic only

You want a cover band to sound like the original album. **Model** = this year’s lineup. **Divergence** = a critic’s score of “how far from the album.” **Train** = rehearse until the score drops.

**How does the critic score a band she cannot hear as a density?** That question is still open — you only named the critic. Inventing a vibe instead of a number $D$ is the wrong move.

In lecture words: lineup = $p_\theta$, album = $p_x$, critic = $D$, rehearse = backprop.

### Local picture

```
  unknown p_x
       │  assume
       ▼
  family { p_θ }     ←  MODEL  (mixture or net)
       │  score
       ▼
  D(p_x ∥ p_θ)
       │  minimize θ
       ▼
  “training”  =  optimization
  (nets → chain rule → backprop)
```

Notice: small $p$ vs script $P$ is the only notational seatbelt he offers.

### Bridge

The recipe needs $D(p_x\|p_\theta)$. Both arguments are invisible. The next box is that hole — and why statistics will have to plug it.

---

## Topic 4: The hole: $D$ without either density (13:28–15:22)

### Where this sits on the master map

**OBSTACLE.** The recipe is blocked because $p_x$ and $p_\theta$ are implicit. Warm-up: [sample mean](./PREREQUISITES.md#p3-sample-mean).

### Board / screenshot

![Recipe still on the tablet while he names the missing densities](./screenshots/composites/ch04-topic-04-hole-d-without-densities-panel1of1.png)

**Figure — ~13:40–15:09:** full three-step list is now on the tablet — (i) family $p_\theta$, (ii) define and estimate $D$, (iii) minimize over $\theta$. Last tile starts the **Example**: $Z\in\mathbb{R}^k$, $Z\sim\mathcal{N}(0,I)$, $\hat x=g_\theta(z)$ has a different law. Spoken: we know neither density; 30–40% of the course is estimating $D$ from samples.

### What he is establishing

The real issue is not “pick an $f$.” It is that **neither** $p_x$ **nor** $p_\theta$ is in hand. They are implicit. How to estimate a divergence in that blindness is **thirty to forty percent of the course**.

What you *have* is a pile of draws from the unknown law. The central question becomes: can a divergence written as an integral of densities be replaced by something a **sample estimator** can touch?

You already know one such replacement. $\mathbb{E}[X]$ is an integral against $p$. The **sample mean** estimates it. The theme of the algorithms is: rewrite $D$ so it looks like expectations, then replace those expectations by averages over the points you actually possess. Then optimize $\theta$.

You can now name the hole. You cannot yet name the rewrite — that is later lectures. Next he shows a *machine* that at least produces the second cloud of points.

### Analogy for this topic only

Two towns on a map with the roads erased. **How far are they?** You cannot read a mileage table (the densities). You can only send scouts (samples) and average their reports. The sample mean is the first scout trick you already trust.

In lecture words: mileage table = $D$ on densities, scouts = samples, average = statistical estimator.

### Local picture

```
  WANT:   D(p_x ∥ p_θ) = some integral of densities
  HAVE:   x1..xn  ~ p_x          (no formula for p_x)
          ???     ~ p_θ          (not even this cloud yet)

  ESCAPE: write D as an expectation
          replace E[·] by a sample average
          (baby case:  E[X]  ≈  x̄)
```

Notice: this lecture *poses* the escape. It does not carry it out. Variational calculus is next time.

### Bridge

We need a second cloud — samples from $p_\theta$. The primitive generator $Z\to G_\theta$ is how that cloud is manufactured.

---

## Topic 5: Primitive generator $Z\to G_\theta$ (15:22–20:34)

### Where this sits on the master map

**GENERATOR.** Leftover: we need samples from $p_\theta$ and a knob $\theta$ that can, in theory, match $p_x$. Warm-up: [pushforward](./PREREQUISITES.md#p7-push).

### Board / screenshot

![Z~N(0,I) through g_θ to x̂; θ*=argmin D; inference = sample](./screenshots/composites/ch05-topic-05-generator-z-gtheta-panel1of1.png)

**Figure — ~15:52–20:04:** $Z\in\mathbb{R}^k$, $Z\sim\mathcal{N}(0,I)$. $g_\theta:Z\to\mathcal{X}$, $\hat x=g_\theta(z)$ has a different law. Trapezoid net: $Z\to g_\theta\to\hat x\sim p_\theta$. $\theta^*=\arg\min D(p_x\|p_\theta)$, $D\ge 0$, $D=0$ iff $p_x=p_\theta$. Inference (pink): a $Z$ through $g_{\theta^*}$ is a sample from $p_x$.

### What he is establishing

Start from a random variable $Z$ whose law you **know**, typically $Z\sim\mathcal{N}(0,I)$ on $\mathbb{R}^k$. Think of a $k$-dimensional bell, not a 1-D cartoon.

Push $Z$ through a deterministic map $G_\theta$. The output $\hat x=G_\theta(Z)$ is a new RV. Its law depends on $G_\theta$ (Jacobian / transform facts from earlier). Let $G_\theta$ be a neural net. Universal approximation says: by moving $\theta$ you can, in theory, realize any target output law.

Why a Gaussian? **Infinite support.** A universal approximator can then send mass wherever the target lives. (Topic 7 will qualify this.)

Write $D(p_x\|p_\theta)$. Do not ask yet how to compute it — you do not know $p_x$. *If* you could minimize $D$ over $\theta$, a UFA $G$ makes $p_{\theta^*}=p_x$ a live possibility. For that to work, $D$ should be $\ge 0$ and $0$ only when the two hills match.

Then the same machine is a **sampler**. After training, draw $Z\sim\mathcal{N}(0,I)$, push through $G_{\theta^*}$, and treat the output as a draw from $p_x$. Standard notation for the rest of the course: $p_x$ unknown, $p_\theta$ model.

You can now draw the $Z\to G_\theta\to\hat x$ cartoon and say why it samples *after* $D$ is driven to zero. You still cannot evaluate $D$, and you must not confuse the net’s **outputs** with a formula for $p_\theta$.

### Analogy for this topic only

A pasta extruder with Gaussian dough. **Which die $G_\theta$ makes noodles that match the restaurant’s house shape?** If you had a score $D$ that is zero only for a perfect match, twisting the die until $D=0$ would give you both the recipe and an endless noodle machine.

In lecture words: dough = $Z$, die = $G_\theta$, house shape = $p_x$, score = $D$.

### Local picture

```
  Z ~ N(0, I_k)     known, infinite support
         │
         ▼
      G_θ  (neural net, knobs θ)
         │
         ▼
      x̂ ~ p_θ      (unknown formula)

  θ* = argmin_θ  D(p_x ∥ p_θ)
  want D ≥ 0 and  D=0  ⇔  p_x = p_θ

  AFTER:  Z  →  G_θ*  →  sample from ≈ p_x
```

Notice: $G_\theta$ gives **samples from** $p_\theta$, not a printout of $p_\theta$.

### Bridge

The cartoon looks as if we “have $p_\theta$.” We do not. The next box drives that wedge in with MNIST.

---

## Topic 6: Samples are not the law (20:34–23:20)

### Where this sits on the master map

**WHAT WE HAVE.** Leftover from the cartoon: it *looks* like $p_\theta$ is known. Warm-up: [density vs samples](./PREREQUISITES.md#p1-density).

### Board / screenshot

![Same generator cartoon; he distinguishes samples from p_θ](./screenshots/composites/ch06-topic-06-samples-not-law-panel1of1.png)

**Figure — ~20:50–23:02:** after $\theta^*=\arg\min D$, “$p_x$ is *implicitly* estimated by $g_\theta$ and one can sample from $p_x$ using $g_\theta$.” Then the numbered **questions**: (i) how to compute $D$ without knowing $p_x$ and $p_\theta$; (ii) which $D$; (iii) how to choose $g_\theta$ (hence $p_\theta$); (iv) how to solve the min.

### What he is establishing

The course’s first hard question is: compute $D$ **without knowing** $p_x$ and $p_\theta$. Even in the $G_\theta$ picture you do **not** have $p_\theta$. The network emits **samples** from $p_\theta$. Knowing a distribution and holding samples from it are different acts.

**MNIST** is the concrete wedge. Everyone “has MNIST.” That is a pile of images drawn from some unknown digit-law. Nobody hands you $p_{\mathrm{MNIST}}$.

If you had defined $p_\theta$ as a **GMM**, you would know the formula. Here you do not. Treating the fake digits as “I now know $p_\theta$” is the wrong move. What you *do* have: the dataset (samples from $p_x$) and the generator outputs (samples from $p_\theta$). First question, sharpened: from **two clouds**, can you estimate $D$?

The tablet then lists three more questions, which become the course’s table of contents: **which** $D$ (the choice changes behavior), **which** $G_\theta$ (architecture), **how** to minimize $D$ (several solvers). He will answer all four in several algorithmic dialects.

You can now list the four questions and refuse the confusion “I ran $G_\theta$, so I know $p_\theta$.” Next, a student asks whether the Gaussian input is mandatory.

### Analogy for this topic only

A jar of cookies vs the bakery’s recipe card. **You have the jar. Do you have the recipe?** MNIST is the jar. $G_\theta$’s outputs are a second jar the bakery just baked. $D$ wants both recipe cards. You hold two jars.

In lecture words: jar = samples, recipe = density.

### Local picture

```
  HAVE                         DO NOT HAVE
  dataset D  ~ p_x             the function p_x
  G_θ(z) for many z  ~ p_θ     the function p_θ
                                (unless p_θ is a GMM you wrote)

  Q1  D from two clouds?
  Q2  which D?
  Q3  which G_θ?
  Q4  how to minimize D?
```

Notice: we are “not completely blind.” Two clouds is already a lot.

### Bridge

A student objects: maybe a **uniform** $Z$ is enough. That question forces a support argument — the next box.

---

## Topic 7: Infinite support and multimodal $Z$ (23:20–27:00)

### Where this sits on the master map

**SUPPORT.** Leftover: why $\mathcal{N}(0,I)$ rather than $\mathrm{Unif}[0,1]^k$? Warm-up: [pushforward](./PREREQUISITES.md#p7-push).

### Board / screenshot

![Discussion board; support / bijectivity argued in speech](./screenshots/composites/ch07-topic-07-support-multimodal-z-panel1of1.png)

**Figure — ~23:44–26:38:** tablet is mostly the ongoing generator notes. Spoken: nets onto/bijective and continuous; bounded input support cannot grow; disjoint input supports stay disjoint; their paper — unimodal $Z$ is not optimal for multimodal data.

### What he is establishing

The question: must the input have **infinite support**, or is a **uniform** start enough?

Neural nets are (or can be made) **bijective**, at least **onto**, and **continuous**. A continuous onto map cannot manufacture mass **outside** a bounded input support. One-dimensional cartoon: $Z$ uniform on a short interval, target law living on a *longer* interval — you cannot get there. “Scale the uniform — to how much? That’s Gaussian.”

He cites their own work on the **best input law**. **Manifold hypothesis** sits in the background. If $Z$ has **disjoint** supports and $G$ is Lipschitz / bijective, the output **still** has disjoint supports.

If the *data* are multimodal (almost disjoint blobs), is a **unimodal** $Z$ the best start? Their answer: **no**. A Gaussian is a *good representative* because you can shrink support, but it is **arbitrary**. What matters more is **unimodal vs multimodal**. What matters less is the exact support, **as long as it is infinite** — a UFA can choose where to put mass. That is the usual defense of a Gaussian start.

Real images are not cleanly disjoint-support objects; theory uses easier assumptions so the math moves. Even for continuous multimodal data, unimodal $Z$ was not optimal on their metrics (including faster optimization).

You can now say why a short uniform is unsafe, and that infinite support is the cheap insurance — not a uniqueness theorem. The lecture then leaves Q&A and names the first algorithm family.

### Analogy for this topic only

A cookie cutter fed dough only from a **saucer** cannot stamp a cookie larger than the saucer. **How big should the dough sheet be?** Infinite (Gaussian) dough can be trimmed. Two separate dough blobs stay two cookies. If the bakery sells two shapes, starting with one blob of dough is convenient, not always best.

In lecture words: saucer = bounded support, two blobs = multimodal $Z$.

### Local picture

```
  Z uniform on a short interval
           │  continuous onto G
           ▼
        output still short     ──X──►  cannot grow support

  Z with two disjoint islands
           │  Lipschitz / bijective G
           ▼
        output still two islands

  data multimodal, Z unimodal Gaussian
           │
           ▼
        works, not optimal  (their paper)
```

Notice: infinite support is insurance; **modality** of $Z$ is the sharper design choice.

### Bridge

The machine and its input law are on the table. The first algorithm family needs a **concrete** $D$. That $D$ is $f$-divergence, and he will refuse to call it a metric.

---

## Topic 8: VDM family; $f$-divergence definition (27:00–31:41)

### Where this sits on the master map

**F-DIV.** First named family: variational divergence minimization. Warm-ups: [convex](./PREREQUISITES.md#p4-convex), [divergence vs metric](./PREREQUISITES.md#p6-div-vs-metric).

### Board / screenshot

![VDM title; D_f = ∫ p_θ f(p_x/p_θ) dx; f:ℝ₊→ℝ convex lsc f(1)=0](./screenshots/composites/ch08-topic-08-vdm-fdiv-definition-panel1of1.png)

**Figure — ~27:31–31:13:** “Variational Divergence Minimization.” $f$-divergence: $D_f(p_x\|p_\theta)=\int_{\mathcal{X}} p_\theta(x)\,f\bigl(p_x(x)/p_\theta(x)\bigr)\,dx$. $f:\mathbb{R}_+\to\mathbb{R}$, convex, left semi-continuous, $f(1)=0$. $\mathcal{X}$ = space both densities are supported on.

### What he is establishing

First algorithmic class: **variational divergence minimizers (VDM)**. He picked an order so one family teaches the next; any order is possible. **Generative adversarial networks** are a **member** of this family. Later: **variational autoencoders**, then **diffusion**.

The family’s first move is to pick a $D$. The pick is **$f$-divergence**. He almost says “distance metric,” then stops. The *earlier* recipe board still says “distance metric.” Here he takes the word back: $D_f$ does **not** obey the **triangle inequality**, so it is **not** a metric. Call it a **divergence measure**. The wrong move is to import triangle proofs from $\mathbb{R}^n$. The right move is $\ge 0$ and $=0$ iff the hills match.

Definition, on **densities** of continuous RVs (an equivalent distribution-function form exists):

$$
D_f(p_x\|p_\theta)=\int_{\mathcal{X}} p_\theta(x)\,f\!\left(\frac{p_x(x)}{p_\theta(x)}\right)\,dx.
$$

$f$ is **convex**, **left semi-continuous**, and **$f(1)=0$**. Any such $f$ is a legal generator, so there is an **infinite family** of $f$-divergences.

A density at a point is a **positive real**, not a number in $[0,1]$. Density is not a probability measure. The ratio $p_x/p_\theta$ is therefore in $\mathbb{R}_+$, which is why $f$ is defined on the positive reals. $f$ itself may return any real.

$\mathcal{X}$ is the space on which **both** densities are supported.

You can now write $D_f$ and list the three demands on $f$. Homework properties and the famous special cases are the next box.

### Analogy for this topic only

A family of kitchen scales, each built from a different **spring** $f$. **Which springs are allowed?** Only convex sagging springs that read zero when both pans hold the same weight (ratio $1$). The scale still will not obey the triangle law of a surveyor’s chain. Do not sell it as a ruler.

In lecture words: spring = $f$, scale reading = $D_f$, ruler = metric.

### Local picture

```
  p_x(x) , p_θ(x)   = two heights  ≥ 0
  u = p_x(x) / p_θ(x)  ∈ ℝ₊

  D_f = ∫  p_θ(x) · f(u(x))  dx

  LEGAL f:
    convex · left semi-continuous · f(1)=0

  each legal f  →  one divergence
  infinitely many f  →  a family

  NOT a metric: no triangle
```

Notice: the integral is weighted by **$p_\theta$**, not by $p_x$. That is why swapping the two hills is a different $f$, not a swap of labels.

### Bridge

We have a family. We need the two properties that make $D_f$ usable as a training score, and the named members (KL, reverse KL, JSD, TV).

---

## Topic 9: Properties and the named family (31:41–36:18)

### Where this sits on the master map

**EXAMPLES.** Leftover: which $f$, and why $D_f$ is safe to drive to zero. Warm-up: [convex](./PREREQUISITES.md#p4-convex).

### Board / screenshot

![D_f≥0; =0 iff p_x=p_θ; f(u)=u log u → KL; JSD and TV formulas](./screenshots/composites/ch09-topic-09-properties-named-family-panel1of1.png)

**Figure — ~32:10–35:49:** (i) $D_f\ge 0$ for any allowed $f$. (ii) $D_f(p_x\|p_\theta)=0$ iff $p_x=p_\theta$. Convexity inequality written. Example $f(u)=u\log u$ plugs in to $D_{\mathrm{KL}}=\int p_x\log(p_x/p_\theta)$. Forward KL $\neq$ reverse KL. $f(u)=\tfrac12\bigl(u\log u-(u+1)\log\frac{u+1}{2}\bigr)$ is **JS**. $f(u)=\tfrac12|u-1|$ is **TV**.

### What he is establishing

Homework, both directions: for any allowed $f$, $D_f\ge 0$; and $D_f=0$ **if and only if** the two densities are exactly the same. The second property is the one generative models buy. Drive $D_f$ to zero and you have matched the hills.

He recalls convexity (the board’s weighted form) and assumes you know it.

**KL** is the first named child: $f(u)=u\log u$ ($u$ is a dummy). Plug into the integral and the $p_\theta$ cancels against the ratio to leave

$$
D_{\mathrm{KL}}(p_x\|p_\theta)=\int p_x(x)\log\frac{p_x(x)}{p_\theta(x)}\,dx.
$$

Minimizing that KL is the **maximum likelihood** estimator from the previous course. So MLE is already an $f$-divergence method.

**Reverse KL** is assigned as an exercise: try $f(u)=\log u$ and check that you recover $\int p_\theta\log(p_\theta/p_x)$. Carry the **sign** carefully — many textbooks write $f(u)=-\log u$ for the same child. He stresses the slogan: KL is **not symmetric**. Calling the swap “reverse KL” is a **misnomer**. They are two different $f$s, two different divergences.

Treating $D_{\mathrm{KL}}(p_x\|p_\theta)$ as interchangeable with $D_{\mathrm{KL}}(p_\theta\|p_x)$ is the wrong move — you would train a *different* generator. The right move is to pick an $f$ on purpose.

That matters for samples. Training $G_\theta$ with forward KL is **not** the same job as training it with reverse KL.

**Jensen–Shannon** uses the $f$ on the board (half of $u\log u$ minus $(u+1)\log((u+1)/2)$). **Total variation** uses $\tfrac12|u-1|$. He also names **chi-square** and **Hellinger**. All are $f$-divergences with a particular spring.

You can now plug $u\log u$ by hand and refuse to treat KL as symmetric. Interpretations of the two KLs — miss a mode vs invent junk — are the last box.

### Analogy for this topic only

Two one-way roads between the same two towns. **Does “A to B” equal “B to A”?** No. Both roads are legal $f$-highways. $u\log u$ is the A-to-B road (KL, also MLE). The other $f$ is B-to-A. JSD is a road that listens to both directions. TV is yet another spring.

In lecture words: A-to-B = forward KL, B-to-A = reverse KL.

### Local picture

```
  f(u) = u log u
    ∫ p_θ · (p_x/p_θ) log(p_x/p_θ) dx
    = ∫ p_x log(p_x/p_θ) dx   =  KL(p_x ∥ p_θ)
    minimizer = MLE

  f(u) = log u     (his exercise)
    should recover reverse KL
    (watch the sign: tables often use −log u)

  KL(p_x ∥ p_θ)  ≠  KL(p_θ ∥ p_x)

  also:  JSD ,  TV = ½|u−1| ,  χ² ,  Hellinger
```

Notice: property (ii) is why this family can *train* a generator at all.

### Bridge

Different $f$s are not cosmetic. The last box shows the two sins of a generator and which $f$ fines which sin.

---

## Topic 10: Mode-cover vs junk; variational next (36:18–42:30)

### Where this sits on the master map

**INTERPRET / STOP.** Leftover: *why* pick one $f$ over another. Warm-ups: [likelihood = height](./PREREQUISITES.md#p5-likelihood), [miss vs junk](./PREREQUISITES.md#p8-modes).

### Board / screenshot

![KL; x̂ with p_x↑ p_θ↓ and x̂̂ with p_x↓ p_θ↑; then sampler + estimate D from samples](./screenshots/composites/ch10-topic-10-modes-jsd-variational-next-panel1of1.png)

**Figure — ~36:53–41:51:** KL integral. Consider $\hat x$ where $p_x$ is high and $p_\theta$ is low; then $\hat{\hat x}$ where $p_x$ is low and $p_\theta$ is high. Bottom: the $G_\theta$ cartoon again; objective = minimize $D_f$ **without knowing either density**, given samples from both (dataset $D$ and $G_\theta(Z)$).

### What he is establishing

How should you **choose** $D$? After the 2014 GAN paper, a zoo of letter-GANs appeared. Later people saw a **unified** $f$-divergence story — mostly an **afterthought**. Each variant was really a request for a different **property** of the trained model.

He fixes the word **likelihood** for this hour: evaluate the **density at a point**. Not the probability of that point.

**Case A.** A location $\hat x$ where $p_x$ is **tall** and $p_\theta$ is **short**. The KL integrand is **large**. We *want* that: the model is missing a place the data actually lives.

**Case B.** A location $\hat{\hat x}$ where $p_x$ is **short** and $p_\theta$ is **tall**. KL is **small**. We do **not** want that silence: the model is inventing **junk** and KL barely fines it.

In slogan form: failure A is “not generating what it should” (missing a mode). Failure B is “generating junk.” Forward KL punishes A, not B.

**Reverse KL** inverts the ratio. It **discourages junk** and does **not** force the model to cover every data mode.

A divergence that **averages** both pressures is attractive. That is his reading of **Jensen–Shannon**. TV, chi-square, Hellinger each have their own personality — homework: go read the papers.

The setup is now a sampler **if** $p_\theta=p_x$. The **objective** for the next class: an algorithm that minimizes $D_f$ **without knowing either density**, using the dataset as samples from $p_x$ and $G_\theta(Z)$ as samples from $p_\theta$. That needs **variational calculus**.

Refresh before next time: probability, **central limit theorem**, **convergence**, **Markov and Chebyshev** inequalities.

You can now tell the two sins apart and say why JSD is a political compromise. You cannot yet compute $D_f$ from the two clouds.

### Analogy for this topic only

A cover band. **Missing the hit everyone came for** (tall data hill, short model hill) vs **playing a song nobody asked for** (short data hill, tall model hill). Forward KL yells at the first sin. Reverse KL yells at the second. JSD yells at both, more quietly. **Which setlist fine do you want?** The $f$ you pick *is* that choice.

In lecture words: miss the hit = mode drop, unrequested song = junk.

### Local picture

```
  point x̂:   p_x HIGH,  p_θ LOW     →  KL LARGE     (good: fines missed mode)
  point x̂̂:  p_x LOW,   p_θ HIGH    →  KL SMALL     (bad: junk almost free)

  forward KL   : cover every mode; junk cheap
  reverse KL   : no junk; may drop a mode
  JSD          : average of the two pressures

  NEXT: estimate D_f from
        cloud D          ~ p_x
        cloud G_θ(Z)     ~ p_θ
        without the hills   →  variational calculus
```

Notice: “likelihood” in this story is a **height**, not a dataset product.

### Bridge

$f$-divergence is installed, named, and interpreted. The missing algorithm is a **sample-based** stand-in for the integral — next lecture, with CLT and tail inequalities warmed up.

---

## External references

Two or three companions **per topic**, listed **only here** (not under each topic). Mix of **video** and **blog/notes**. No Wikipedia.

| Resource | Type | Matches lecture… | Why it helps |
|----------|------|------------------|--------------|
| [Lec 02 — Generative models: problem formulation](../15-Lec02-Generative-Models-Problem-Formulation/NOTES.md) | notes | Topic 1 · GOAL | Same course: estimate $p_x$ from data. |
| [StatQuest — The main ideas of fitting a line](https://www.youtube.com/watch?v=PaFPbb66DxQ) | video | Topic 1 · samples → unknown rule | Baby “cloud of points, hidden rule.” |
| [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | video | Topic 2 · UFA / nets | Why a net can stand in for $G_\theta$. |
| [Lilian Weng — From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) | blog | Topic 2 · implicit sampler | Implicit $p_\theta$ + sampling before VDM. |
| [Goodfellow — NIPS 2016 GAN tutorial (arXiv)](https://arxiv.org/abs/1701.00160) | notes | Topic 2 · why generate | Estimate vs sample, how GANs sit among models. |
| [StatQuest — MLE, step by step](https://www.youtube.com/watch?v=XepXtl9YKwc) | video | Topic 3 · KL min = MLE | The previous-course special case he cites. |
| [Penn State STAT 415 — MLE](https://online.stat.psu.edu/stat415/lesson/1/1.2) | notes | Topic 3 · family + score | Written “pick a family, maximize a score.” |
| [StatQuest — Backpropagation, main ideas](https://www.youtube.com/watch?v=IN2XmBhILt4) | video | Topic 3 · train = chain rule | How a composite $G_\theta$ actually moves. |
| [Seeing Theory — Frequentist inference / LLN](https://seeing-theory.brown.edu/frequentist-inference/index.html) | demo | Topic 4 · sample mean | Averages stand in for expectations. |
| [3Blue1Brown — Essence of calculus (chain rule)](https://www.youtube.com/watch?v=YG15m2VwSjA) | video | Topic 4 · why backprop exists | Composite maps, derivatives, the hole is still $D$. |
| [Distill — Why momentum works](https://distill.pub/2017/momentum/) | blog | Topic 4 · training is opt | Visual “solve an optimization problem.” |
| [StatQuest — The normal distribution](https://www.youtube.com/watch?v=rzFX5NWojp0) | video | Topic 5 · $Z\sim N(0,I)$ | Infinite-support start, $k$-D bell. |
| [Tutorial 9 — Jacobian / transforms](../23-Tutorial09-Review-Basic-Probability-3/NOTES.md) | notes | Topic 5 · $G(Z)$ new law | Same series: maps of RVs. |
| [UAT explained (deep-mind.org)](https://www.deep-mind.org/2023/03/26/the-universal-approximation-theorem/) | blog | Topic 5 · UFA slogan | Why twisting $\theta$ *might* hit any output law. |
| [MNIST (TensorFlow Datasets)](https://www.tensorflow.org/datasets/catalog/mnist) | demo | Topic 6 · samples ≠ $p$ | The pile he names: images, not a density. |
| [Rohit Bandaru — ML divergences](https://rohitbandaru.github.io/notes/ml-divergences/) | notes | Topic 6 · which $D$ | Short written menu of $D$s (asymmetric, not a metric). |
| [Chris Olah — Nets, manifolds, topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) | blog | Topic 7 · support / folds | Why disjoint supports and folds survive $G$. |
| [TDS — Universal approximation (intuition)](https://towardsdatascience.com/understand-universal-approximation-theorem-with-code-774dcef55731/) | blog | Topic 7 · UFA limits | What “can approximate” does *not* say about support. |
| [Nowozin et al. — f-GAN (arXiv)](https://arxiv.org/abs/1606.00709) | paper | Topic 8 · $D_f$ family | The unified “every $f$ is a GAN / VDM” paper. |
| [Goodfellow NIPS 2016 tutorial (PDF notes)](https://arxiv.org/pdf/1701.00160) | notes | Topic 8 · GAN ∈ VDM | Same chronology he sketches. |
| [StatQuest — KL divergence](https://www.youtube.com/watch?v=SxGYPqCgJWM) | video | Topic 9 · $f(u)=u\log u$ | Forward KL as an $f$-child. |
| [ritvikmath — The KL divergence](https://www.youtube.com/watch?v=q0AkK8aYbLY) | video | Topic 9 · KL algebra | Second pass on $\int p\log(p/q)$. |
| [Grosse — KL reading (CSC321)](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/readings/L05%20KL%20Divergence.pdf) | notes | Topic 9 · two KLs | Written mode-cover vs mode-seek. |
| [NannyML — Jensen–Shannon distance](https://www.youtube.com/watch?v=YBjfT9hIUus) | video | Topic 9–10 · JSD | Average-of-two-KLs intuition he wants. |
| [Medium — Understand and use JSD](https://medium.com/data-science/how-to-understand-and-use-jensen-shannon-divergence-b10e11b03fd6) | blog | Topic 10 · JSD vs KL | Why JSD is the “both sins” compromise. |
| [Jon Vet — Forward and reverse KL](https://www.jonvet.com/blog/forward-and-reverse-kl-divergence) | blog | Topic 10 · two sins | Same height-story, modern RL language. |
| [Huszár — How to train your generative models](https://www.inference.vc/how-to-train-your-generative-models-why-generative-adversarial-networks-work-so-well-2/) | blog | Topic 10 · junk vs miss | Why forward KL is a bad *sample-quality* score. |
| [Seeing Theory — distributions / CLT](https://seeing-theory.brown.edu/probability-distributions/index.html) | demo | Topic 10 · homework | CLT he asked you to refresh. |
| [Tutorial 8 — Markov / Chebyshev](../22-Tutorial08-Review-Basic-Probability-2/NOTES.md) | notes | Topic 10 · inequalities | Same series’ tail inequalities. |
| [Dibya Ghosh — KL for ML](https://dibyaghosh.com/blog/probability/kldivergence/) | blog | Topic 10 · KL picture | Extra diagrams if Grosse is too terse. |

**How to use:** after Topic 5, UAT + Tutorial 9. After Topic 6, stare at MNIST as a *pile*. After Topic 9, StatQuest or ritvikmath + Grosse. After Topic 10, Huszár or JSD video — then stop; variational estimation is the next lecture.

---

## Sources

- Video: [Lec 03 f-Divergence and Examples](https://www.youtube.com/watch?v=LR9UQXY_IU8) · NPTEL IISc
- Description: KL Divergence, Reverse KL and JSD
- Auto-captions in `raw/captions.en.timed.txt` (cleaned; reverse-KL $f$ left as his exercise)
- Boards transcribed from `screenshots/composites/`
- **Code audit:** no code on the tablet. No invented Python. Math in `$` / `$$` only.
