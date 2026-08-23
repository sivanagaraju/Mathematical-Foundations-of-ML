# W1_L4 — Variational divergence minimization

> **Video:** [W1_L4: Variational divergence minimization](https://www.youtube.com/watch?v=nfZQYopzv20) · **~26 min**  
> **Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md) · **Quiz:** [quiz.html](./quiz.html)

**Do PREREQUISITES before this article.**  
**Previous:** [W1_L2 problem setting](../01-W1-L2-Introduction-Problem-Setting/NOTES.md)  
**Course:** IIT Madras BS · Mathematical Foundations of Generative AI · Prof. Prathosh  
**Same math, different recording:** NPTEL [Lec 03 f-divergence](../../Mathematical-Foundation-for-GenerativeAI/25-Lec03-f-Divergence-Examples/NOTES.md) is longer; this sitting **stops before** the variational bound.

The title says VDM. **What is delivered:** recap + **$f$-divergence** (KL, JS, TV). **What is next:** a generic algorithm that minimizes **any** $f$-div, then GAN.

| When the lecture hits… | Warm-up |
|------------------------|---------|
| Unknown $p_x$; dataset | [p1-px](./PREREQUISITES.md#p1-px) |
| Estimate **and** sample | [p2-two-jobs](./PREREQUISITES.md#p2-two-jobs) |
| Model $=p_\theta$ net | [p3-model](./PREREQUISITES.md#p3-model) |
| Divergence as a score | [p4-div](./PREREQUISITES.md#p4-div) |
| $z$ through $G_\theta$ | [p5-push](./PREREQUISITES.md#p5-push) |
| Samples $\neq$ density | [p6-samples](./PREREQUISITES.md#p6-samples) |
| Convex $f$, $f(1)=0$ | [p7-f](./PREREQUISITES.md#p7-f) |
| KL not symmetric | [p8-kl](./PREREQUISITES.md#p8-kl) |

---

## Table of Contents

1. [Topic 1 — Recap: estimate, sample, 3-step recipe](#topic-1--recap-estimate-sample-3-step-recipe-0011–0312) (00:11–03:12)
2. [Topic 2 — Push-forward: samples, not the law](#topic-2--push-forward-samples-not-the-law-0312–0639) (03:12–06:39)
3. [Topic 3 — Four questions; VDM is the same recipe](#topic-3--four-questions-vdm-is-the-same-recipe-0639–0911) (06:39–09:11)
4. [Topic 4 — $f$-divergence definition](#topic-4--f-divergence-definition-0911–1651) (09:11–16:51)
5. [Topic 5 — $D_f\ge 0$ and zero iff equal](#topic-5--d_f-ge-0-and-zero-iff-equal-1651–1853) (16:51–18:53)
6. [Topic 6 — KL is $f(u)=u\log u$](#topic-6--kl-is-fuu-log-u-1853–2155) (18:53–21:55)
7. [Topic 7 — Forward KL $\neq$ reverse KL](#topic-7--forward-kl-neq-reverse-kl-2155–2322) (21:55–23:22)
8. [Topic 8 — JS and TV; different $f$, different properties](#topic-8--js-and-tv-different-f-different-properties-2322–2528) (23:22–25:28)
9. [Topic 9 — Next: any-$f$ algorithm, then GAN](#topic-9--next-any-f-algorithm-then-gan-2528–2608) (25:28–26:08)
10. [External references](#external-references)
11. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

You have IID files from an unknown law $p_x$. The job is two verbs: **estimate** that law **and sample** new files. The method is a three-step recipe — guess a net $p_\theta$, score it with a divergence, minimize over $\theta$ — plus a push-forward machine $z\to G_\theta(\hat x)$. This sitting **names** the score **$f$-divergence**. The variational algorithm and GAN are **next class**.

**Worldview arc:** from “unnamed $D$ on a push-forward $G_\theta$” **to** “a family $D_f$ whose children are KL / JS / TV.”

**Code this hour:** none. Chalk only. Do not invent a training loop.

### System context

```
  ╔══════════════════════════════════════════╗
  ║ W1_L2: estimate p_x AND sample           ║
  ║ W2_L5: VDM algorithm + parent of GANs    ║
  ║ NPTEL Lec 03/04: longer twin recording   ║
  ╚══════════════════╤═══════════════════════╝
                     │ this sitting (~26 min)
                     ▼
          name the yardstick: f-divergence
          (not yet the variational bound)
```

### Method card — the approach (hold this)

```
  HOLD     data D = {x_i} ~ iid p_x (unknown)
           two jobs: estimate p_x  AND  sample

  RECIPE   (i)   p_θ  = parametric family, usually a DNN  (“the model”)
           (ii)  pick a divergence D(p_x || p_θ)
           (iii) θ* = argmin_θ D

  MACHINE  z ~ N(0,I)  -->  G_θ(z)  -->  x̂ ~ p_θ
           net emits SAMPLES, not the density p_θ

  NAME D   D_f(p_x || p_θ) = ∫ p_θ(x) f( p_x(x)/p_θ(x) ) dx
           f: R+ → R, convex, left-semicontinuous, f(1)=0
           D_f ≥ 0;   D_f = 0  iff  p_x = p_θ

  CHILDREN f(u)=u log u              → KL
           board JS spring           → Jensen–Shannon  (GAN uses a version, later)
           f(u)=½|u−1|               → total variation
           KL is NOT symmetric: forward ≠ reverse

  STOP     generic algorithm that minimizes ANY f-div from two clouds
           then one f → GAN
           (that IS the variational method — next sitting)
```

**Walk the sitting (one screen).** (1) Recap two jobs + three-step recipe. (2) Build $z\to G_\theta$: outputs are **samples**, not a density. (3) Four holes, then the slogan: VDM **is** this recipe. (4) Write $D_f$; $f$ convex, left-semicontinuous, $f(1)=0$; the ratio is a **scalar**. (5) $D_f\ge 0$ and $=0$ iff the laws match — so driving $D_f\to 0$ matches $p_\theta$ to $p_x$. (6) $f(u)=u\log u$ cancels to KL. (7) Forward KL $\neq$ reverse. (8) JS and TV springs; different $f$ $\Rightarrow$ different generative-model properties. (9) STOP: generic any-$f$ minimizer, then GAN.

### Main blueprint

```
  ╔════ JOB ════╗
  ║ estimate p_x║
  ║ AND sample  ║
  ╚════╤════════╝
       │ 3-step recipe
       ▼
  p_θ as a DNN (MODEL)
       │
       ▼
  z ~ N(0,I) --> G_θ --> x̂ ~ p_θ     samples, NOT the density
       │
       ▼
  ┌── NAME THE SCORE ─────────────┐
  │ D_f = ∫ p_θ f(p_x / p_θ) dx   │
  │ f convex, lsc, f(1)=0         │
  │ ≥0 and 0 iff p_x = p_θ        │
  └──────────────┬────────────────┘
                 │ pick f
        ┌────────┼────────┐
        ▼        ▼        ▼
       KL       JS        TV
   (u log u)  (board)  (½|u−1|)
        │
        └── KL not symmetric
                 │
      ┌ · · · · ·┴ · · · · · ┐
      │ STOP: any-f algorithm │
      │ and GAN  — next class │
      └ · · · · · · · · · · ┘
```

### Scenario walkthrough (two clouds, one integral)

```
  real cloud:  MNIST files     ~ p_x     (density unknown)
  fake cloud:  z through G_θ   ~ p_θ     (density unknown)

  cannot plug two formulas into D_f
  CAN write D_f once you name f

  pick f(u)=u log u  →  algebra → KL = ∫ p_x log(p_x/p_θ)
  still cannot evaluate it from the clouds   ← next sitting’s hole
```

1. Recap the two jobs and the three-step recipe.  
2. Build the pasta-extruder $G_\theta$. Outputs are noodles (samples), not a recipe card (density).  
3. Ask the four questions. Announce: VDM **is** this recipe.  
4. Write $D_f$. Check $f$’s three conditions.  
5. Read off $\ge 0$ and “zero iff match.”  
6. Cancel $p_\theta$ for KL.  
7. Refuse to treat KL as a two-way street.  
8. Add JS and TV springs.  
9. Stop: the **algorithm** is next.

### Failure / contrast path

```
  treat G_θ(z) as a printout of p_θ(x)     ──X──► samples ≠ density
  plug p_x into D_f as if you had a formula ──X──► you only have D
  use KL as if D(p||q)=D(q||p)              ──X──► forward ≠ reverse
  write the GAN loss from this sitting      ──X──► promised next
```

### STOP / out of scope

- Generic **push-forward minimizer of any $f$-div** (the actual variational method).  
- **GAN** value function (a version of JS, “as we move on”).  
- Conjugate $f^*$, critic $T$, $\min_\theta\max_w$ saddle.  
- Mode-cover vs junk (not spoken here).  
- Proof that $D_f\ge 0$ (stated, not proved).

### Load-bearing claims (closed-book)

- Two jobs: estimate $p_x$ **and** sample. Model $=p_\theta$ as a net.  
- $z$ through $G_\theta$ yields **samples from** $p_\theta$, not $p_\theta$ itself.  
- VDM **is no different** from this push-forward recipe.  
- $D_f=\int p_\theta\,f(p_x/p_\theta)$; $f$ convex, left-semicontinuous, $f(1)=0$.  
- $D_f\ge 0$; $=0$ iff $p_x=p_\theta$.  
- $f(u)=u\log u$ $\Rightarrow$ KL after $p_\theta$ cancels.  
- Forward KL $\neq$ reverse KL.  
- Board JS and TV; different $f$ $\Rightarrow$ different generative-model properties.

**Speaker / course.** IIT Madras BS · BSDA5002 · Prof. Prathosh A. P. · Week 1 Lecture 4. Next in-list: [W2_L5](https://www.youtube.com/watch?v=stZC0Zk5KYo) completes the algorithm.

---

## Topic 1: Recap: estimate, sample, 3-step recipe (00:11–03:12)

### Where this sits on the master map

**RECAP.** Continuity from last sitting’s formal definition, then the unnamed 3-step recipe. GANs are previewed as a **member of VDM**, not derived. Warm-up: [unknown $p_x$](./PREREQUISITES.md#p1-px), [two jobs](./PREREQUISITES.md#p2-two-jobs), [model](./PREREQUISITES.md#p3-model).

### Board / screenshot

![Topic 1 board — generative modeling recipe](./screenshots/composites/ch01-topic-01-recap-estimate-sample-recipe-panel1of1.png)

“Generative Modeling. Given $D=\{x_1,\ldots,x_n\}\sim\mathrm{iid}\,p_x$ (unknown). Goal: Estimate $p_x$ **and** learn to sample from it.” General principle: (i) assume a parametric family $p_\theta$, represented using DNNs — **(Model)**. (ii) Define and estimate a divergence (distance) metric between $p_\theta$ and $p_x$. (iii) Solve an optimization over the parameters of $p_\theta$ to minimize that metric.

### What he is establishing

This session studies a class of generative models based on **variational divergence minimization**. Famous **GANs** are a **member of this family**. Before details, a recap so there is continuity.

Given: a dataset of $n$ samples drawn **IID** from an underlying **unknown** distribution $p_x$. Goal: **estimate** $p_x$ **and learn to sample from it**. Estimating $p_x$ may be **implicit or explicit**, depending on the model. A **primary** requirement is still to **sample**.

The general principle has **three steps**. First: assume a **parametric family** on $p_x$, denoted $p_\theta$. In modern models $p_\theta$ is a **deep neural network**. That object is what this course calls the **model**. Second: **define and estimate** a divergence between $p_x$ and $p_\theta$. Third: **optimize** $\theta$ to minimize that divergence so $p_\theta$ **approaches** $p_x$.

The wrong move is to think “estimate” is enough — a classifier does not emit a new $x$. The right move: keep both verbs, and keep the three-step skeleton. The divergence is still **unnamed**. You can now write the recipe on one card. What is still missing: a machine that emits $\hat x$, and a *named* $D$.

### Analogy for this topic only

A radio with a tuning dial (the family $p_\theta$), a “how far from the real station” meter (the unnamed $D$), and a hand that turns the dial until the meter drops (the $\arg\min$). **Does owning the radio mean you can play a new song?** Only if the meter and the dial actually get used — estimate without sample is a radio you never turn on.

In lecture words: dial $=\theta$, radio $=p_\theta$ net, meter $=D$, play $=$ sample.

### Local picture

```
  D = {x1,…,xn} ~ iid p_x   (UNKNOWN)

  Goal:  estimate p_x   AND   sample

  (i)   p_θ  via a DNN     ← “the MODEL”
  (ii)  D(p_x || p_θ)      ← still unnamed
  (iii) min_θ of that D    so p_θ approaches p_x
```

Notice: “model” in this course **means** $p_\theta$ as a net, not a PyTorch file.

### Bridge

The recipe still has no **machine** that actually emits $\hat x$. The leftover is a concrete example: start from noise you can sample, push it through a net.

---

## Topic 2: Push-forward: samples, not the law (03:12–06:39)

### Where this sits on the master map

**PUSH-FORWARD.** The running example of the recipe. Warm-up: [push-forward](./PREREQUISITES.md#p5-push), [samples $\neq$ density](./PREREQUISITES.md#p6-samples), [divergence](./PREREQUISITES.md#p4-div).

### Board / screenshot

![Topic 2 board — z through G_θ; θ* = argmin D](./screenshots/composites/ch02-topic-02-push-forward-gtheta-samples-not-law-panel1of1.png)

Example: RV $z\in\mathbb{R}^k$ with a known law, typically $z\sim\mathcal{N}(0,I)$. $G_\theta:Z\to\mathcal{X}$. $\hat x=G_\theta(z)$ has a **different** law, depending on $G_\theta$. If $G_\theta$ is a net, that law is $p_\theta(\hat x)$. Cartoon: $z\sim\mathcal{N}(0,I)$ into a trapezoid $G_\theta(z)$ out $\hat x\sim p_\theta$. $\theta^*=\arg\min_\theta D(p_x\|p_\theta)$. $D\ge 0$, $D=0$ iff $p_x=p_\theta$.

### What he is establishing

**Push-forward** methods: start from an **arbitrary** RV you **know how to sample**. Typical choice: standard normal. Represent $p_\theta$ as the law of the **output** of a **deterministic** $G_\theta$ mapping $Z$-space to data space $\mathcal{X}$.

Elementary probability: a random variable pushed through a deterministic function is another random variable, here $\hat x$. Its law **depends on** $G_\theta$, and is not the law of $z$.

Take $G_\theta$ to be a **deep net**. The law the net **imposes** is $p_\theta$. Load-bearing: the net’s output gives **samples from $p_\theta$**, **not** the density $p_\theta$ itself.

Then optimize over the **parameters of $G_\theta$** to minimize a distributional divergence between $p_x$ and that imposed law. Board: $\theta^*=\arg\min_\theta D(p_x\|p_\theta)$. If $D$ is well defined and the opt is solved, $p_{\theta^*}$ is close to $p_x$, because $D$ is **non-negative** and **zero iff** the two laws match.

Work a tiny case of the same idea. Let $Z$ be a fair die $\{1,2,3,4,5,6\}$. If $g(z)=0$ on even faces and $1$ on odd, $\hat X$ is a fair bit ($3/6$ each). Change $g$ to “$1$ only if $z\ge 5$”: now $P(\hat X=1)=2/6$. You changed the output **law by changing $g$**, without writing a density on paper.

The wrong move is to treat a forward pass as “now I know $p_\theta(x)$ at every $x$.” You know 128 fake images, not a formula. You can now draw the trapezoid and write $\theta^*=\arg\min D$. What is still missing: how to **compute** $D$ from those 128 images plus the real pile.

### Analogy for this topic only

A pasta extruder. Flour is easy random stuff. The **die** shapes it. Change the die, change the pasta-law. **Do you hold a physics formula for “probability of this noodle,” or a pile of noodles?** A pile. That is “samples, not the density.”

In lecture words: flour $=z$, die $=G_\theta$, noodles $=\hat x\sim p_\theta$, $\arg\min$ $=$ grinding the die until $D$ is small.

### Local picture

```
  z ~ N(0, I)  -->  G_θ(z)  -->  x̂ ~ p_θ
                         │
                         └── law of x̂ DEPENDS on G_θ
                             output = SAMPLES from p_θ
                                      not p_θ itself

  θ* = argmin_θ D(p_x || p_θ)
  D ≥ 0,   D = 0  iff  p_x = p_θ
```

Notice: $D$ is still unnamed. The two properties preview Topic 5.

### Bridge

If $\theta^*$ is found, $z$ through $G_{\theta^*}$ should sample from near $p_x$. The leftover is: you still **cannot compute** $D$, because you know **neither density**. Four questions from last sitting return.

---

## Topic 3: Four questions; VDM is the same recipe (06:39–09:11)

### Where this sits on the master map

**FOUR QUESTIONS / VDM $=$ SAME RECIPE.** Completes the trained sampler, restates the holes, then the slogan: VDM is **no different** from push-forward. Warm-up: [two jobs](./PREREQUISITES.md#p2-two-jobs), [samples](./PREREQUISITES.md#p6-samples).

### Board / screenshot

![Topic 3 board — trained sampler; questions iii–iv](./screenshots/composites/ch03-topic-03-four-questions-vdm-is-same-recipe-panel1of1.png)

Rebuilt panel (dropped an empty “Variational Di…” title frame). After the opt, $p_x$ is **implicitly estimated** by $G_{\theta^*}(z)$; one can sample from $p_x$ using $G_{\theta^*}(z)$. Cartoon: $z\sim\mathcal{N}(0,I)$ through $G_{\theta^*}$ to $\hat x\sim p_{\theta^*}$ (close to $p_x$). Questions on the tablet: (iii) how to choose $G_\theta(z)$, in turn $p_\theta$? (iv) how to solve $\min D$?

### What he is establishing

After training, $z\sim\mathcal{N}(0,I)$ through $G_{\theta^*}$ yields samples from $p_{\theta^*}$, which is close to $p_x$ **by construction** — hence (near) samples from $p_x$. That package **is** the general push-forward principle.

Then the four questions one should ask.

**Q1.** How do we **compute** divergences **without knowing either** $p_\theta$ or $p_x$? What we have: samples from $p_x$ (the dataset). Samples from $p_\theta$ are **easy by design**: sample $z$, pass through $G$; the outputs **are** samples from $p_\theta$. Trap: we **neither** know $p_x$ **nor** $p_\theta$ as density functions — only clouds.

**Q2.** What should the **choice of divergence** be?  
**Q3.** How to choose **$G_\theta$** — choosing $G_\theta$ **is** choosing $p_\theta$.  
**Q4.** How do we **solve** the minimization?

Load-bearing slogan: **variational divergence minimization is no different from** the push-forward method just seen. Then the title card. Do **not** start the $D_f$ integral yet. You can now list the four questions. What is still missing: a named family of $D$ (next box) and an algorithm that uses only the two clouds (not this sitting).

### Analogy for this topic only

Two photo dumps, no lighting equation. **Can you score “how different are these two worlds” without the physics formula?** That is Q1. Q2 is which scoring rule. Q3 is which camera body. Q4 is how you turn the knobs. VDM, today, is **not a new job** — it is this same shop under a new sign.

In lecture words: photo dumps $=$ two sample clouds, physics formula $=$ density, new sign $=$ VDM.

### Local picture

```
  trained:  z ~ N(0,I) --> G_θ* --> x_new ≈ sample from p_x
            (p_x implicitly estimated by G_θ*)

  Q1  compute D without densities?   (two clouds only)
  Q2  which D?
  Q3  which G_θ  (= which p_θ)?
  Q4  how to minimize D?

  slogan:  VDM = this push-forward recipe
```

Notice: Q1 is why a **variational** bound will be needed later. It is **not** derived in these 26 minutes.

### Bridge

The sign on the door is VDM. The leftover is to **name** the divergence family that the rest of the family (including GANs) will sit inside.

---

## Topic 4: $f$-divergence definition (09:11–16:51)

### Where this sits on the master map

**F-DIV.** First named family. Warm-up: [convex $f$](./PREREQUISITES.md#p7-f).

### Board / screenshot

![Topic 4 board — D_f integral; f convex, lsc, f(1)=0](./screenshots/composites/ch04-topic-04-f-divergence-definition-panel1of1.png)

Title: **Variational Divergence Minimization.** “Define divergence metrics between distributions.” Then $f$-divergence: given densities $p_x$ and $p_\theta$,

$$
D_f(p_x\|p_\theta)=\int_{\mathcal{X}} p_\theta(x)\,f\!\left(\frac{p_x(x)}{p_\theta(x)}\right)\,dx.
$$

$f(u):\mathbb{R}^+\to\mathbb{R}$, **convex**, **left-semi continuous**, $f(1)=0$. $\mathcal{X}$: space on which $p_x$ and $p_\theta$ are supported (typically $\mathbb{R}^d$).

### What he is establishing

VDM starts by defining a **distributional divergence**. The class is **$f$-divergence**, written $D_f(p_x\|p_\theta)$ with the **densities** in the parentheses.

Working assumption: **continuous** RVs with **well-defined densities**. Equivalent definitions exist when densities fail or the RVs are discrete; most models in this course are the continuous case, so he uses this form.

$f(u)$ is an **arbitrary convex** function $\mathbb{R}^+\to\mathbb{R}$, **left-semicontinuous**, with **$f(1)=0$**. $u$ is a dummy (board writes $\omega$). Any such $f$ yields **one** member of the family.

$\mathcal{X}$ is the common **support**, typically $\mathbb{R}^d$ because that is where the data live.

The integrand is well-defined even when $x$ is a **$d$-vector**. A density at a point is a **scalar** (non-negative; he tightens to **positive**). The **ratio** of two densities at the same point is a **positive real** — exactly the domain of $f$.

The wrong move is to feed the **vector** $x$ into $f$. $f$ eats the **ratio of heights**, one number. You can now write $D_f$ and the three conditions on $f$. What is still missing: why driving this integral to zero matches the laws (next), and how to evaluate it from clouds (later sitting).

### Analogy for this topic only

Walk a city of addresses. At each address, two buildings have heights. The **ratio of those two heights** is one number. A spring sits at rest when the heights match. **Does the spring care about the street’s GPS coordinates, or about that one ratio?** The ratio. Walk every address, add the payments, get the city-wide bill.

In lecture words: address $=x\in\mathbb{R}^d$, heights $=$ densities, spring $=f$, bill $=D_f$.

### Local picture

```
  D_f(p_x || p_θ) = ∫_X  p_θ(x) · f( p_x(x)/p_θ(x) ) dx

  f: R+ → R,  convex,  left-semicontinuous,  f(1)=0
  X typically R^d

  x may be a vector;  p(x) is a SCALAR;  the ratio is a SCALAR
```

Notice: $f(1)=0$ is on the board (ASR said “z”). Left-**semi**continuous, not merely continuous.

### Bridge

A named integral is not yet a reason to train. The leftover is the two properties that make driving $D_f$ to zero the same as matching the laws.

---

## Topic 5: $D_f\ge 0$ and zero iff equal (16:51–18:53)

### Where this sits on the master map

**PROPERTIES.** Why $D_f$ is a usable training score. Warm-up: [divergence](./PREREQUISITES.md#p4-div).

### Board / screenshot

![Topic 5 board — D_f ≥ 0; =0 iff p_x = p_θ](./screenshots/composites/ch05-topic-05-f-div-nonneg-zero-iff-equal-panel1of1.png)

Properties of $f$-divergence: (i) $D_f\ge 0$ for **any** choice of $f$. (ii) $D_f(p_x\|p_\theta)=0$ **iff** $p_x=p_\theta$. Definition of $D_f$ and conditions on $f$ remain on the tablet.

### What he is establishing

Property (i): $D_f\ge 0$ **for any** admissible $f$. We **desire** that in a divergence (a distance between laws should not go negative). For $f$-div it holds **by construction**. He does **not** prove it here.

Property (ii): $D_f=0$ **if and only if** the two distributions are **exactly equal**.

Both properties will be used while **constructing generative models**. The goal is to drive the divergence between true and model to **zero**. With these two properties, $D_f\to 0$ **forces** $p_\theta$ to match $p_x$. Therefore $f$-divergence is a usable training divergence.

The wrong move is to minimize a score that can be negative or that can hit zero for mismatched laws. Then “small $D$” would not mean “matched.” You can now state both properties. What is still missing: a named $f$ so $D_f$ is not an empty integral.

### Analogy for this topic only

A bathroom scale under two flour bowls that **cannot read below zero**, and that reads **exactly zero only when the bowls are the same pile**. **If you turn knobs until the scale hits zero, have you matched the piles?** Yes — that is what (i) and (ii) buy you. He does not show the inner mechanism of the scale today.

In lecture words: scale $=D_f$, piles $=p_x$ and $p_\theta$, knobs $=\theta$.

### Local picture

```
  (i)  D_f  ≥  0     for ANY admissible f     (by construction)
  (ii) D_f(p_x || p_θ) = 0   iff   p_x = p_θ

  train by driving D_f → 0  ⇒  the laws match
```

Notice: no Jensen proof. Stated properties only.

### Bridge

One integral, many children. The leftover is to **plug in** a particular $f$ and see a famous name: KL.

---

## Topic 6: KL is $f(u)=u\log u$ (18:53–21:55)

### Where this sits on the master map

**EXAMPLE (a).** First named child. Warm-up: [KL](./PREREQUISITES.md#p8-kl).

### Board / screenshot

![Topic 6 board — f(u)=u log u; p_θ cancels to D_KL](./screenshots/composites/ch06-topic-06-kl-is-f-of-u-log-u-panel1of1.png)

“Examples of $f$-divergence. (a) $f(u)=u\log u$: KL-divergence.” Algebra: $D_f=\int p_\theta f(p_x/p_\theta)\,dx=\int p_\theta\cdot(p_x/p_\theta)\log(p_x/p_\theta)\,dx=\int p_x\log(p_x/p_\theta)\,dx=D_{\mathrm{KL}}$.

### What he is establishing

Example (a): $f(u)=u\log u$. The corresponding $f$-divergence **is** Kullback–Leibler. Immediate from the definition: substitute $u=p_x/p_\theta$, and **$p_\theta$ cancels**, leaving $\int p_x\log(p_x/p_\theta)\,dx=D_{\mathrm{KL}}$.

Therefore **KL is a special case of $f$-divergence**. The family is **larger** than this one $f$ — different purposes will want different members. End of slice: KL is **not known to be symmetric** (names wait one box).

The wrong move is to think $f$-div **is** KL. KL is **one spring**.

### Analogy for this topic only

One spring recipe: “height ratio times log of itself.” Plug it into the city-wide bill. The fake-pile height sitting in front of the spring **cancels** the same height in the denominator, and you are left with the usual KL textbook line. **Is every $f$-divergence KL?** No — this is one recipe.

In lecture words: spring $=u\log u$, cancel $=p_\theta$ in front vs $p_\theta$ in the ratio, leftover $=\int p_x\log(p_x/p_\theta)$.

### Local picture

```
  f(u) = u log u

  ∫ p_θ · (p_x/p_θ) log(p_x/p_θ) dx
      ↑ cancel
  = ∫ p_x log(p_x/p_θ) dx  =  D_KL(p_x || p_θ)

  KL is ONE child of the family
```

Notice: ASR “ulb library / kale” is **KL**. Trust the tablet algebra.

### Bridge

If KL were a two-way street, one $f$ might be enough. The leftover is that swapping the arguments is a **different** number — which is why a **family** of $f$ is worth having.

---

## Topic 7: Forward KL $\neq$ reverse KL (21:55–23:22)

### Where this sits on the master map

**KL NOT SYMMETRIC.** Justifies looking at more than one $f$. Warm-up: [KL](./PREREQUISITES.md#p8-kl).

### Board / screenshot

![Topic 7 board — D_KL(p_x||p_θ) ≠ D_KL(p_θ||p_x)](./screenshots/composites/ch07-topic-07-kl-not-symmetric-forward-reverse-panel1of1.png)

$D_{\mathrm{KL}}(p_x\|p_\theta)\neq D_{\mathrm{KL}}(p_\theta\|p_x)$. Under the left: **forward KL**. Under the right: **reverse KL**. The KL algebra from Topic 6 is still on the upper half.

### What he is establishing

KL is **not symmetric**. The KL between $p_x$ and $p_\theta$ is **not** the KL between $p_\theta$ and $p_x$. Literature names: $D_{\mathrm{KL}}(p_x\|p_\theta)$ is **forward KL**; $D_{\mathrm{KL}}(p_\theta\|p_x)$ is **reverse KL**.

Asymmetry has **consequences** (not enumerated this sitting). That is a reason one might **not** want KL as the training divergence. If one rejects KL, one may use a **different** metric, with **properties of its own**. This **justifies a large family**: different $f$ $\Rightarrow$ different divergences with different properties.

He does **not** do the mode-cover vs junk cartoon here. Do not import it.

### Analogy for this topic only

A one-way street. Driving A→B is not driving B→A; the odometer (KL) reads different numbers. **If your training score secretly depends on which pile you call “true,” is one KL enough?** No — that is why other springs exist.

In lecture words: A→B $=$ forward $D_{\mathrm{KL}}(p_x\|p_\theta)$, B→A $=$ reverse.

### Local picture

```
  D_KL(p_x || p_θ)  ≠  D_KL(p_θ || p_x)
       forward              reverse

  not a two-way street  →  look at other f’s
```

Notice: consequences named, not listed. Mode-covering stays in other lectures.

### Bridge

Need at least two more named $f$’s so the family is not a slogan. JS and TV are on the next tablet lines.

---

## Topic 8: JS and TV; different $f$, different properties (23:22–25:28)

### Where this sits on the master map

**EXAMPLES (ii)–(iii).** Two more children; the load-bearing summary. Warm-up: [KL / family](./PREREQUISITES.md#p8-kl).

### Board / screenshot

![Topic 8 board — JS generator and TV](./screenshots/composites/ch08-topic-08-js-tv-different-f-different-properties-panel1of1.png)

Still the KL inequality, then (ii) $f(u)=\tfrac12\big(u\log u-(u+1)\log\frac{u+1}{2}\big)$ : **JS-divergence**. (iii) $f(u)=\tfrac12|u-1|$ : **total variation distance**.

### What he is establishing

Second named generator: that $f$ makes the corresponding $f$-divergence **Jensen–Shannon**. **A version of** JS is what the famous **GAN** uses — “we will see that as we move on.” Not derived today.

Third: $f(u)=\tfrac12|u-1|$ is **total variation**.

All $f$’s looked at (KL’s $u\log u$, JS, TV) obey the standing hypotheses: $\mathbb{R}^+\to\mathbb{R}$, convex, left-semicontinuous, $f(1)=0$. “One can easily verify that.”

Summary: choosing a particular **$f$** gives an $f$-divergence with **certain properties**. **Different** $f$ $\Rightarrow$ **different** metrics $\Rightarrow$ **different properties in the way the generative model is built**.

ASR said “half $u\log u$ minus $u+2$.” **Ignore it.** Use the **board** JS formula. Do not expand JS into “average of two KLs to the midpoint” unless he writes it (he does not). You can now name three children. What is still missing: the training algorithm that uses them.

### Analogy for this topic only

Three spring recipes on the same city integral. KL is a one-way odometer. JS is a compromise road (GAN will use a **version** later). TV is a cruder “how much mass sits on mismatched addresses.” **If you change the spring, do you still train the same generative model?** No — that is the summary line.

In lecture words: springs $=$ the three $f$’s, city integral $=D_f$, later shop $=$ GAN.

### Local picture

```
  (a) f(u)=u log u
        → KL
  (ii) f(u)= ½ ( u log u − (u+1) log((u+1)/2) )
        → JS-divergence     (GAN uses a VERSION, later)
  (iii) f(u)= ½ |u−1|
        → total variation

  all three:  R+→R, convex, lsc, f(1)=0
  different f  →  different properties of the gen. model
```

Notice: GAN is a **pointer**, not a derivation.

### Bridge

The family is installed. The leftover promised in the last 40 seconds: a **generic algorithm** that uses this family as a training score, then one $f$ that **becomes GAN**.

---

## Topic 9: Next: any-$f$ algorithm, then GAN (25:28–26:08)

### Where this sits on the master map

**STOP.** Title said VDM; today named the $f$-family. The **algorithm** is next. Warm-up: [two jobs](./PREREQUISITES.md#p2-two-jobs) — you still cannot compute $D_f$ from clouds.

### Board / screenshot

![Topic 9 — JS/TV catalog still on screen, then IITM end card](./screenshots/composites/ch09-topic-09-next-generic-algorithm-then-gan-panel1of1.png)

Rebuilt two-panel (the auto 2×2 tripled the same JS/TV card). Left: the catalog from Topic 8, unchanged. Right: IIT Madras production card (`study.iitm.ac.in/ds`). Teaching content is **spoken** STOP, not a new formula.

### What he is establishing

Next: given a particular $f$-divergence, a **generic algorithm** to construct a generative model by the **push-forward** method already seen, **minimizing any $f$-divergence**. After that: fix a **particular** $f$ and see how that construction **becomes the famous GAN algorithm**. “That is the idea.” End card.

**Coverage STOP.** Those two objects are **not delivered this sitting**. Do not write $f^*$, a critic $T$, or $J_{\mathrm{GAN}}$ as if they were on this board.

You can now name $D_f$ and three children. You cannot yet train from two clouds.

### Analogy for this topic only

The spring catalog is on the wall. **Have we built the machine that reads the springs from two photo dumps?** Not today. Tomorrow: a generic reader for any spring, then the GAN spring.

In lecture words: catalog $=\{$KL, JS, TV$\}$, reader $=$ next sitting’s algorithm, GAN spring $=$ a version of JS.

### Local picture

```
  TODAY     named D_f and three f’s
  NEXT      generic push-forward minimizer of ANY f-div
            then one f → GAN

  not today:  f*, critic T, min_θ max_w, GAN loss
```

Notice: last frames reuse the JS/TV card. Teaching content is **spoken** STOP, not a new formula.

### Bridge

The $f$-family is the yardstick. The leftover — computing it from two clouds, then specializing to GAN — is [W2_L5](https://www.youtube.com/watch?v=stZC0Zk5KYo) (and NPTEL Lec 04 in the other recording).

---

## External references

Two layers, **both kept**. All companions live **here**, not under the topics. Mix of **video**, **notes**, and **original papers**. No Wikipedia.

The URL is playlist index 4. The video **is W1_L4**; the **algorithm** named in the title is next.

### Start here

**If last sitting is foggy (Topics 1–3).** This course’s [W1_L2](../01-W1-L2-Introduction-Problem-Setting/NOTES.md). NPTEL twin: [Lec 02 problem formulation](../../Mathematical-Foundation-for-GenerativeAI/15-Lec02-Generative-Models-Problem-Formulation/NOTES.md).

**If $D_f$ / KL / JS will not stay apart (Topics 4–8).** Nowozin, Cseke, Tomioka — [$f$-GAN (arXiv:1606.00709)](https://arxiv.org/abs/1606.00709) tables $f$, $f^*$, and last activations (the **next** sitting’s objects; today’s $f$ is the left column). Stanford CS236 [GAN / $f$-GAN notes](https://deepgenerativemodels.github.io/notes/gan/). Lilian Weng — [From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) (KL vs JS menu).

**If you want the algorithm this title promised (Topic 9).** Same instructor, next lecture: [W2_L5 Generative modelling via VDM](https://www.youtube.com/watch?v=stZC0Zk5KYo). NPTEL longer twin: [Lec 04 VDM](../../Mathematical-Foundation-for-GenerativeAI/27-Lec04-Variational-Divergence-Minimization/NOTES.md) (conjugate, critic, saddle).

### Full topic map — 2–3 companions each

| Topic | Resource | Type | Why it helps |
|------|----------|------|--------------|
| **1** recipe | [W1_L2 — problem setting](../01-W1-L2-Introduction-Problem-Setting/NOTES.md) | notes | Same course: estimate **and** sample. |
| **1** | [NPTEL Lec 02](../../Mathematical-Foundation-for-GenerativeAI/15-Lec02-Generative-Models-Problem-Formulation/NOTES.md) | notes | Twin recording of the 3-step recipe. |
| **1** | [Stanford CS231N 2025 Lec 14](https://www.youtube.com/watch?v=Edr4uZFh4EE) | video | Latest Stanford vision hour: why generators exist. |
| **2** push-forward | [Weng — From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) | blog | $z$ through $G$ as an implicit sampler. |
| **2** | [3Blue1Brown — Neural nets](https://www.youtube.com/watch?v=aircAruvnKk) | video | A net as a function $z\mapsto\hat x$. |
| **2** | [Serrano — Friendly introduction to GANs](https://www.youtube.com/watch?v=8L11aMN5KY8) | video | Visual $G(z)$ samples; he will later distrust the Hollywood reading. |
| **3** four questions | [NPTEL Lec 03 — the hole](../../Mathematical-Foundation-for-GenerativeAI/25-Lec03-f-Divergence-Examples/NOTES.md) | notes | Same instructor: $D$ without either density. |
| **3** | [CS236 2023 Lec 9 — GANs](https://www.youtube.com/watch?v=3Zv-gokhLu8) | video | Why you only ever have two minibatches. |
| **3** | [Weng — two clouds](https://lilianweng.github.io/posts/2017-08-20-gan/) | blog | Written “samples from both, densities from neither.” |
| **4** $D_f$ definition | [Nowozin et al. — $f$-GAN](https://arxiv.org/abs/1606.00709) | paper | Original VDM / $f$-div generator table. |
| **4** | [Stanford CS236 notes](https://deepgenerativemodels.github.io/notes/gan/) | notes | $D_f=\int q f(p/q)$ in course handwriting. |
| **4** | [Boyd EE364A Lec 4 — convex / conjugate](https://www.youtube.com/watch?v=lEN2xvTTr0E) | video | Convex $f$ (conjugate is **next** sitting). |
| **5** properties | [Weng — divergences](https://lilianweng.github.io/posts/2017-08-20-gan/) | blog | Why $\ge 0$ and $=0$ iff match matter for training. |
| **5** | [Mutual Information — Jensen’s inequality](https://www.youtube.com/watch?v=u0_X2hX6DWE) | video | Why a convex $f$ makes KL-style scores non-negative (he states, does not prove). |
| **5** | [CS236 notes — $f$-div](https://deepgenerativemodels.github.io/notes/gan/) | notes | Same two properties in writing. |
| **6** KL algebra | [3Blue1Brown — Cross entropy](https://www.youtube.com/watch?v=ErfnhcEV1O8) | video | $\int p\log(p/q)$ as surprise difference. |
| **6** | [StatQuest — KL](https://www.youtube.com/watch?v=SxGYPqCgJWM) | video | Short KL definition. |
| **6** | [Nowozin Table 1 — $f(u)=u\log u$](https://arxiv.org/abs/1606.00709) | paper | KL as one row of the $f$ table. |
| **7** forward/reverse | [Weng — forward vs reverse KL](https://lilianweng.github.io/posts/2017-08-20-gan/) | blog | Why swapping arguments is a different object. |
| **7** | [CS236 Lec 10 — $f$-GANs](https://www.youtube.com/watch?v=M3Fkvu78ZXc) | video | Ermon: original GAN is a JS **variant**. |
| **7** | [NPTEL Lec 03](../../Mathematical-Foundation-for-GenerativeAI/25-Lec03-f-Divergence-Examples/NOTES.md) | notes | Twin: KL not symmetric. |
| **8** JS / TV | [Is this the best way to compare distributions? (JS)](https://www.youtube.com/watch?v=3-mSufD_zq0) | video | Symmetric / finite JS vs one-way KL. He does **not** expand JS to two KLs today. |
| **8** | [Arize — JS divergence](https://arize.com/blog-course/jensen-shannon-divergence/) | blog | Written “symmetric cousin of KL.” |
| **8** | [Nowozin Table 1 — JS / TV rows](https://arxiv.org/abs/1606.00709) | paper | Board JS spring and $\tfrac12\|u-1\|$ in the original table. |
| **9** STOP / next | [W2_L5 — VDM continued](https://www.youtube.com/watch?v=stZC0Zk5KYo) | video | **This** course’s next hour: the algorithm. |
| **9** | [NPTEL Lec 04 VDM](../../Mathematical-Foundation-for-GenerativeAI/27-Lec04-Variational-Divergence-Minimization/NOTES.md) | notes | Conjugate, two-E bound, saddle — **not** today. |
| **9** | [MIT 6.S191 2026 L4](https://www.youtube.com/watch?v=R8V8CbuxryI) | video | Latest MIT generative-modeling hour (GAN later). |

**How to use.** Topics 1–3: W1_L2 then this recap. After Topic 4, Nowozin **left** column ($f$) only — $f^*$ is next. After Topic 6, do the cancel on paper. After Topic 8, stop. Topic 9: open W2_L5 or NPTEL Lec 04. **No invented Python.**

---

## Sources

- Video: [W1_L4 Variational divergence minimization](https://www.youtube.com/watch?v=nfZQYopzv20) · IIT Madras BS · Prof. Prathosh · playlist [PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu](https://www.youtube.com/playlist?list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu) index 4
- Auto-captions in `raw/captions.en.timed.txt` (cleaned: $p_x$, $p_\theta$, $G_\theta$, KL, JS, TV, left-semicontinuous)
- Boards transcribed from `screenshots/composites/` (9 unique paths). Topic 3 rebuilt without the empty title frame. Topic 9 is a two-panel: JS/TV catalog + end card (spoken STOP).
- **Code audit:** no training-loop code on the tablet. These notes add **no invented Python**. Math in `$` / `$$` only. JS formula from the **board**, not ASR. Variational bound / GAN **not delivered**.
