# Tutorial 10 — Review of Machine Learning 1

**Video:** [Tutorial 10 : Review of Machine Learning 1](https://www.youtube.com/watch?v=wjSKM1xFoSU) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Tutorial 9](../23-Tutorial09-Review-Basic-Probability-3/NOTES.md)  
**Course:** Mathematical Foundations of **Generative AI** (~47.5 min)  
**Speaker:** Chandan Jayaram (NPTEL IISc) · Numerical MLE and EM (two-exponential mixture)

---

## Table of Contents

1. [Topic 1 — Why this recap exists](#topic-1-why-this-recap-exists-0001–0127) (00:01–01:27)
2. [Topic 2 — Sign-censored Normal](#topic-2-sign-censored-normal-0127–0422) (01:27–04:22)
3. [Topic 3 — Standardize and write the likelihood](#topic-3-standardize-and-write-the-likelihood-0422–0853) (04:22–08:53)
4. [Topic 4 — Bernoulli reduction and MLE of p](#topic-4-bernoulli-reduction-and-mle-of-p-0853–1401) (08:53–14:01)
5. [Topic 5 — Invert Φ and read the sign of μ̂](#topic-5-invert-φ-and-read-the-sign-of-μ̂-1401–1831) (14:01–18:31)
6. [Topic 6 — Two-exponential mixture and latent Z](#topic-6-two-exponential-mixture-and-latent-z-1831–2252) (18:31–22:52)
7. [Topic 7 — Complete-data density and log-likelihood](#topic-7-complete-data-density-and-log-likelihood-2252–2839) (22:52–28:39)
8. [Topic 8 — E-step responsibilities by Bayes](#topic-8-e-step-responsibilities-by-bayes-2839–3419) (28:39–34:19)
9. [Topic 9 — The Q-function](#topic-9-the-q-function-3419–3917) (34:19–39:17)
10. [Topic 10 — M-step closed forms, iterate, close](#topic-10-m-step-closed-forms-iterate-close-3917–4733) (39:17–47:33)
11. [External references](#external-references)
12. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

You sit two incomplete notebooks. The first is a Normal that only wrote plus or minus; the second is a mixture that wrote waiting times but not which kitchen cooked them. For each, write the likelihood of *what was actually recorded*, take the log, differentiate, and set the derivative to zero. The sign-censored Normal reduces to a coin whose bias is a standard-Normal area $\Phi$; invert $\Phi$ to recover the mean. When a hidden switch $Z$ makes the observed density a sum, the log will not split — invent $Z$, take its posterior, maximize the **Q-function**, and loop. That second machine is a two-exponential mixture whose M-step has closed updates for $\pi,\beta_1,\beta_2$.

**Worldview arc:** from first-course MLE / EM as slogans (or `loss.backward()` folklore) **to** likelihood constructed on recorded data, reduced, inverted, or completed by a latent coin.

**Hour at a glance (whole video).** The first half is a Normal $\mathcal{N}(\mu,1)$ whose *values* were never written — only plus or minus, and $m$ of the $n$ draws are minus. You cannot average the missing $x_i$. After standardizing, a minus has probability $\Phi(-\mu)$ and a plus has $\Phi(\mu)$, so the likelihood of the notebook is those areas multiplied. Differentiating $\Phi$ by hand is ugly, so rename $p=\Phi(-\mu)$: the same product is a Bernoulli likelihood (negative = success) and the MLE is $\hat p=m/n$. Then invert: $\hat\mu=-\Phi^{-1}(m/n)=\Phi^{-1}((n-m)/n)$. Many minuses force a negative mean; that is a picture check, not extra theory. MAP is named and left as homework.

The second half is a two-exponential mixture: flip a coin $\pi$, then draw $\mathrm{Exp}(\beta_1)$ or $\mathrm{Exp}(\beta_2)$. The observed density is a *sum*, so $\log f$ has no closed maximizer — that is why EM exists. Invent a latent coin $Z$ so the complete-data density *selects* one kitchen by using $z$ as an exponent. The E-step replaces each missing $z_i$ by its posterior $\gamma_i=P(Z_i=1\mid x_i,\theta^{\mathrm{old}})$ (Bayes). Those $\gamma_i$ go into the **Q-function**, the expected complete log-likelihood. The M-step differentiates $Q$ and gets closed updates: $\pi$ is the mean of the $\gamma$'s, each $\beta$ is a $\gamma$-weighted exponential MLE. Then loop. Handwritten backprop is promised for the next tutorial block, not this one.

### System context

```
  ╔════════════════════════════════════════════╗
  ║ Outside: first-level ML + probability      ║
  ║ Outside: MAP derivation (homework)         ║
  ║ Outside: NN numerical backprop (next)      ║
  ╚════════════════════╤═══════════════════════╝
                       │ this tutorial (~47 min)
                       ▼
          ┌────────────────────────────┐
          │ Two numerical recoveries   │
          │   MLE on signs · EM on mix │
          └────────────────────────────┘
```

### Main blueprint

```
  ╔════════ GOAL ════════╗
  ║ Surface MLE + EM for ║
  ║ later generative     ║
  ║ models               ║
  ╚══════════╤═══════════╝
             │
     ┌───────┴────────┐
     ▼                ▼
 ┌─ PROBLEM 1 ──┐  ┌─ PROBLEM 2 ──────────┐
 │ X ~ N(μ, 1)  │  │ mix of two Exp(β)    │
 │ n IID, signs │  │ θ = (π, β1, β2)      │
 │ m of n are − │  │ latent coin Z        │
 └──────┬───────┘  └──────────┬───────────┘
        │ standardize         │ complete pairs (x,z)
        ▼                     ▼
 ┌─ L(μ) ─────────────┐  ┌─ complete density ──┐
 │ [Φ(-μ)]^m          │  │ [π β1 e^{-β1 x}]^z  │
 │ [Φ(μ)]^{n-m}       │  │ [(1-π)β2 e^{-…}]^{1-z}
 └──────┬─────────────┘  └──────────┬──────────┘
        │ p := Φ(-μ)                │ log, expand
        ▼                           ▼
 ┌─ Bernoulli ────────┐  ┌─ E-STEP ────────────┐
 │ L(p)=p^m(1-p)^{n-m}│  │ γ = P(Z=1 | x, θold)│
 │ ℓ' = 0 ⇒ p̂ = m/n   │  │      (Bayes)        │
 └──────┬─────────────┘  └──────────┬──────────┘
        │ invert Φ                  │ replace z by γ
        ▼                           ▼
 ┌─ μ̂ = −Φ⁻¹(m/n) ────┐  ┌─ Q(θ | θold) ──────┐
 │   = Φ⁻¹((n-m)/n)   │  │ E[complete log-ℓ]   │
 │ many − ⇒ μ̂ < 0     │  └──────────┬──────────┘
 │ MAP = homework     │             │ ∂Q/∂θ = 0
 └────────────────────┘             ▼
                           ┌─ M-STEP ────────────┐
                           │ π_new = mean(γ)     │
                           │ β1 = Σγ / Σγ x      │
                           │ β2 = Σ(1-γ)/Σ(1-γ)x │
                           │ then loop           │
                           └──────────┬──────────┘
                                      │
                           ┌ · · · · ·┴ · · · · · ┐
                           │ STOP: MAP not done   │
                           │ STOP: NN numerical   │
                           │ next section         │
                           └ · · · · · · · · · · ┘
```

### Scenario walkthrough

Walk this **one** exam through the blueprint above. Each step answers “so what?” for the next box.

**Story:** a numerical recap exam. Page 1 is ten Normal draws whose *values* were never written — only plus or minus, and three of the ten are minus. Page 2 is waiting times from two unlabeled exponential kitchens.

1. **Why recap MLE and EM on paper?** Later generative models will need the same two machines sitting on the surface, not sunk inside `loss.backward()`. That is the GOAL box.

2. **Why not average the ten missing numbers?** They were never recorded. The data *are* the signs. That is PROBLEM 1.

3. **How do you score a mean from signs only?** Standardize $\mathcal{N}(\mu,1)$. A minus has probability $\Phi(-\mu)$; a plus has $\Phi(\mu)$. The likelihood of the notebook is those areas multiplied. That is $L(\mu)$.

4. **Why rename $p=\Phi(-\mu)$?** Differentiating $\Phi$ by hand is ugly. The same product is a Bernoulli likelihood (negative = success), so $\hat p=m/n=3/10$. That is REDUCE + MAX.

5. **How do you get back to $\mu$?** Invert: $\hat\mu=-\Phi^{-1}(m/n)=\Phi^{-1}((n-m)/n)$. Three minuses out of ten is a plus-fraction $0.7$, so $\hat\mu>0$. Many minuses would have forced $\hat\mu<0$. That is INVERT + CHECK.

6. **Why is page 2 a different machine?** Every wait $x$ is visible, but the *kitchen* is not. The observed density is a **sum** of two exponentials, so $\log f$ will not split. That is PROBLEM 2.

7. **What do you invent?** A coin $Z$: heads $\Rightarrow\mathrm{Exp}(\beta_1)$, tails $\Rightarrow\mathrm{Exp}(\beta_2)$. Using $z$ as an exponent *selects* one kitchen. That is the complete-data log-likelihood.

8. **You still do not know $Z$.** Replace each $z_i$ by its posterior $\gamma_i=P(Z_i=1\mid x_i,\theta^{\mathrm{old}})$ (Bayes). That is the E-step.

9. **What do you maximize?** The **Q-function**: expected complete log-likelihood, $z_i$ swapped for $\gamma_i$. Differentiate $Q$. The M-step is closed: $\pi$ is the mean of the $\gamma$'s; each $\beta$ is a $\gamma$-weighted exponential MLE. Then loop.

```
  page 1:  + + − + − + + − + +     (m = 3 minuses)
       │  cannot use x̄ — the x_i were never written
       ▼
  each minus costs Φ(-μ); each plus costs Φ(μ)
       │  rename p = Φ(-μ)
       ▼
  p̂ = 3/10  →  μ̂ = −Φ⁻¹(0.3) = Φ⁻¹(0.7)   (positive mean; fewer minuses)

  page 2: waiting times, two kitchens, no label
       │  log of a sum will not split
       ▼
  invent Z → complete log-ℓ → γ_i by Bayes → Q → closed β, π → repeat
```

### Failure / contrast path

```
  average the missing numbers          ──X──►  those x_i were never written
  maximize the mixture by one
    brute derivative                   ──X──►  log of a *sum* has no closed
                                               MLE; that is why EM exists
  treat autograd as understanding      ──X──►  next hour wants numerical
                                               backprop, not loss.backward()
```

### STOP / out of scope

- **MAP** is named as a recap you should already own; no prior, no posterior mode.
- No Gaussian mixture, no general EM proof (monotonicity of the observed likelihood).
- Numerical **backpropagation** is expected background and is the *next* tutorial block, not this one.

### Load-bearing claims (closed-book)

- Build the likelihood from **what was recorded**, not from hidden $x_i$.
- After standardizing $\mathcal{N}(\mu,1)$, a minus has probability $\Phi(-\mu)$ and a plus has $\Phi(\mu)$.
- Signs reduce to a Bernoulli with $p=\Phi(-\mu)$; $\hat p=m/n$; invert to $\hat\mu=-\Phi^{-1}(m/n)=\Phi^{-1}((n-m)/n)$.
- Many negatives force $\hat\mu<0$ — a picture check, not extra theory.
- A two-exponential mixture is a coin $\pi$ then $\mathrm{Exp}(\beta_1)$ or $\mathrm{Exp}(\beta_2)$.
- Complete-data density uses $z$ as an exponent that *selects* one kitchen.
- E-step: $\gamma_i=P(Z_i=1\mid x_i,\theta^{\mathrm{old}})$ by Bayes.
- Q replaces each $z_i$ by $\gamma_i$; the M-step is a weighted exponential MLE, then loop.

**Speaker / course:** Chandan Jayaram, NPTEL IISc, Mathematical Foundations of Generative AI — Tutorial 10.

---

## Topic 1: Why this recap exists (00:01–01:27)

### Where this sits on the master map

This is the **GOAL / SERIES** box. The hour is not a first introduction to machine learning. It is a *numerical* recap of a first-level course, put on the table because later generative models will need the same two machines — MLE and EM — sitting on the surface, not sunk. If “likelihood scores a parameter” is shaky, start at [likelihood](./PREREQUISITES.md#p1-likelihood).

### Board / screenshot

![Talking-head open: numerical recap, not autograd folklore](./screenshots/composites/ch01-topic-01-tutorial-mission-panel1of1.png)

**Figure — ~00:14–01:18:** no tablet yet. He is setting the contract: this bunch of tutorials formulates and solves problems *numerically and mathematically*. The expected toolkit is MLE, EM, and (for later) backpropagation done by hand — not `loss.backward()`.

### What he is establishing

The series now shifts from the probability recap (Tutorials 7–9) into first-course **machine learning** done with a pen. The stated aim is modest and strict: understand how we *formulate* a problem and how we *solve* it, with numbers and derivatives, not with a library call.

Picture a two-line worksheet he would accept later today: “$n=10$ signs, $m=3$ minuses, write $L(\mu)$, find $\hat\mu$.” Or: “two exponential kitchens, write $\gamma_i$, update $\beta_1$.” Those are *numerical* recap problems. A screenshot of `loss.backward()` returning a tensor is not.

Most of what follows is labeled a **prerequisite** for the generative-AI lectures. That word is doing real work. He is not teaching a first course from zero. He is hauling three objects back onto the desk: **maximum likelihood estimation (MLE)**, the **expectation–maximization (EM)** algorithm, and — as background the viewer is assumed to own — **backpropagation** done numerically.

The last item is a warning, not a lesson. Watching a tensor library run `loss.backward()` is not the understanding he wants. A viewer who can recite “the graph does reverse mode” but cannot differentiate a two-layer chain on paper is under-equipped. This particular hour will not do that chain. It will do MLE and EM. The neural-network numericals are promised for the next section of tutorials.

So the contract is: you already took probability and a first ML course; this session makes the two estimation machines executable again. What is still missing after one minute is the first concrete notebook — a Normal whose values were never written down.

### Analogy for this topic only

A kitchen exam says: write the sauce recipe, then cook it. Someone sets a jar labeled `sauce.backward()` on the table. **What should the examiner mark — the jar, or the handwritten steps?**

The jar may taste fine. The exam still fails, because the point was the recipe. Reciting “the graph does reverse mode” is the same miss.

Here the “recipe” is MLE and EM on paper. The jar is autograd. This hour opens the recipe book, not the jar.

In lecture words: recap = first-course ML, numerical; `loss.backward()` is the jar he refuses.

### Local picture

```
  first-level ML course          this bunch of tutorials
  (already taken)                (numerical recap)
         │                              │
         ├─ MLE                  ══►    Problem 1 (signs)
         ├─ EM                   ══►    Problem 2 (mixture)
         └─ backprop (by hand)   · ·    next section (NNs)
                                      not this video
```

Notice: three expected tools, two used today. The third is a forward pointer, not a missing derivation.

### Bridge

We have a contract — write likelihoods and differentiate them — but no data set yet. The first notebook is cruel on purpose: $n$ Normal draws, and the page only recorded **plus or minus**.

---

## Topic 2: Sign-censored Normal (01:27–04:22)

### Where this sits on the master map

This fills **PROBLEM 1**. The goal box asked for a numerical MLE. The obstacle is not “estimate a Normal mean” in the usual way. The obstacle is that the usual numbers were never written. Warm-up: [IID](./PREREQUISITES.md#p2-iid) and [likelihood](./PREREQUISITES.md#p1-likelihood).

### Board / screenshot

![Problem statement: X ~ N(μ,1), n IID, only signs, m negatives, MLE of μ](./screenshots/composites/ch02-topic-02-sign-censored-normal-panel1of1.png)

**Figure — ~01:48–04:04:** tablet fills in four beats. (1) Let $X\sim\mathcal{N}(\mu,1)$. (2) $n$ IID realizations. (3) While taking observations we *only* noted whether the value is positive. (4) Suppose $m$ of $n$ have negative values; find the MLE of $\mu$ based on these observations.

### What he is establishing

A scalar random variable $X$ is drawn from a **Normal** with unknown mean $\mu$ and **known variance 1**. That last word matters. There is only one parameter. We are not estimating a width.

We are given $n$ **IID** copies — same machine, independent tickets. In a kinder problem the page would show $1.2,\,-0.4,\,0.7,\ldots$ and the MLE of $\mu$ would be their average $\bar x$. That kinder problem is not this problem.

While the observations were taken, the actual values were **not recorded**. The notebook has only a sign: positive or not. Out of the $n$ tickets, $m$ are negative. Those two integers $(n,m)$ are the entire data set. Inventing ten fake weights and averaging them is the wrong move: you would be solving a different exam.

The ask is the **MLE of $\mu$** based on the signs. You must treat the recorded signs as the data, and score $\mu$ for how well it explains *those* signs.

A micro picture: $n=10$, $m=3$. The page looks like

```
  + + − + − + + − + +
```

Ten bells were sampled. Three landed left of zero. $\mu$ is still a real number we have not estimated. The next topic turns “landed left of zero” into $\Phi(-\mu)$.

You can now state the formal problem: model $\mathcal{N}(\mu,1)$, data $=$ counts of signs, target $=$ $\hat\mu_{\mathrm{MLE}}$. What is still missing is a likelihood written only in terms of $\mu$ and $(n,m)$.

### Analogy for this topic only

Ten sheep jump a fence at position $0$. You are not allowed to weigh them. You only tick “left of fence” or “right of fence.” Three ticks are left.

**Where is the flock’s typical location $\mu$?** Memory of the ten missing weights cannot answer — those weights were never written. Sliding the hill until three-left / seven-right is the least surprising pattern is the right move. Inventing ten fake weights and averaging them solves a different exam.

In lecture words: flock center = $\mu$, ticks = $m$ and $n-m$, missing weights = the unrecorded $x_i$.

### Local picture

```
  TRUE draw                 WHAT WAS WRITTEN
  x1 =  1.2                 +
  x2 = -0.4                 −
  x3 =  0.7                 +
  ...                       ...
  xn = -1.1                 −     (this is one of the m minuses)

  usual MLE would use x̄     ──X──►  x̄ is not on the page
  this MLE must use signs   ══════►  (n, m) only
```

Notice: the model is still a Normal on $\mathbb{R}$. The *likelihood* will live on $\{+,-\}^n$.

### Bridge

We know the hidden machine and the recorded alphabet. We do not yet have $P(\text{minus})$ as a function of $\mu$. That probability is an area under a shifted bell — the next box.

---

## Topic 3: Standardize and write the likelihood (04:22–08:53)

### Where this sits on the master map

This is the **LIKELIHOOD** box on Problem 1. The open problem from Topic 2 is “score $\mu$ using only signs.” The move is: convert a sign into a **standard-Normal area** $\Phi$, then multiply. Warm-up: [$\Phi$](./PREREQUISITES.md#p3-phi).

### Board / screenshot

![Standardize to Z, Φ(-μ) and Φ(μ), then L(μ) = Φ(-μ)^m Φ(μ)^{n-m}](./screenshots/composites/ch03-topic-03-standardize-likelihood-panel1of1.png)

**Figure — ~04:48–08:27:** solution starts $P[X<0]=P[(X-\mu)/1 < -\mu]$. Then $Z\sim\mathcal{N}(0,1)$, so that probability is $\Phi(-\mu)$. Then $P[X>0]=1-\Phi(-\mu)=\Phi(\mu)$ “using normality.” After counting $m$ negatives and $n-m$ positives, the likelihood of the observed data is $L(\mu)=(P[X<0])^m(P[X>0])^{n-m}=(\Phi(-\mu))^m(\Phi(\mu))^{n-m}$.

### What he is establishing

A minus occurs when $X<0$. Subtract the mean and divide by the known standard deviation $1$:

$$
P(X<0)=P\bigl(X-\mu<-\mu\bigr)=P(Z<-\mu)
$$

where $Z=(X-\mu)/1\sim\mathcal{N}(0,1)$. The left-hand area of the *standard* bell up to $-\mu$ is the CDF value $\Phi(-\mu)$. So

$$
P(\text{negative})=\Phi(-\mu).
$$

A plus is the complementary event. Because a standard Normal is symmetric about zero, $1-\Phi(-\mu)=\Phi(\mu)$:

$$
P(\text{positive})=\Phi(\mu).
$$

He says “using inverse” while writing that identity. The clean reading is: we are using the known inverse-symmetry of $\Phi$, not inverting $\Phi$ yet. Inversion arrives in Topic 5.

Each recorded sign is now a Bernoulli-like trial with success probabilities that *depend on $\mu$*. There are $m$ independent minuses and $n-m$ independent pluses, so the likelihood of the notebook is the product

$$
L(\mu)=\bigl[\Phi(-\mu)\bigr]^m\bigl[\Phi(\mu)\bigr]^{n-m}.
$$

In words: every minus contributes a factor $\Phi(-\mu)$; every plus contributes $\Phi(\mu)$. That is the likelihood of the *observed* data. Multiplying the Normal densities of the missing $x_i$ would be the wrong move — those numbers are not in the notebook.

He pauses to say this assumes a rigorous probability course and a first-level ML course. The objects $\Phi$ and “likelihood of IID signs” are supposed to be already in muscle memory. This hour is the numerical assembly.

A micro case: $n=2$, $m=1$ (one minus, one plus). $L(\mu)=\Phi(-\mu)\,\Phi(\mu)$. At $\mu=0$ this is $(1/2)^2=1/4$. At a huge positive $\mu$, $\Phi(-\mu)$ is tiny and $L$ collapses — one unexplained minus kills a large positive mean.

You can now write $L(\mu)$ from $(n,m)$ alone. What is still awkward is differentiating $\Phi(-\mu)$ with respect to $\mu$. The next box removes that pain by renaming $p=\Phi(-\mu)$.

### Analogy for this topic only

Each sheep is a coin flip whose *bias* depends on where you parked the hill. Park the hill far right ($\mu$ large positive) and almost every flip is “right of fence.” Three left-ticks then make a terrible score.

**What number should we write down as “how well this parking explains the ten ticks”?** Reciting the ten missing weights does not answer. The score is “bias-for-left, multiplied $m$ times, times bias-for-right, multiplied $n-m$ times.” That product *is* $L(\mu)$.

In lecture words: left-bias = $\Phi(-\mu)$, right-bias = $\Phi(\mu)$, score = $L(\mu)$.

### Local picture

```
  N(μ,1) density, threshold at 0

       μ < 0                         μ > 0
  ----▲------|----             ----|------▲----
       μ     0                      0     μ
     big left area               small left area
     Φ(-μ) large                 Φ(-μ) small

  notebook:  m copies of [left]   and   (n-m) copies of [right]
  L(μ) = [left area]^m  ×  [right area]^{n-m}
```

Notice: the Normal *density* of a hidden $x_i$ never appears. Only areas.

### Bridge

$L(\mu)$ is a product of $\Phi$ values — ugly to differentiate. If we give the left-area a single name $p$, the same product becomes a coin likelihood we already know how to maximize.

---

## Topic 4: Bernoulli reduction and MLE of $p$ (08:53–14:01)

### Where this sits on the master map

This is **REDUCE + MAX**. The likelihood exists; the leftover problem is “maximize a function of $\Phi$.” The method is: take the log, rename $p=\Phi(-\mu)$, recognize a **Bernoulli**, differentiate. Warm-ups: [log](./PREREQUISITES.md#p4-log), [Bernoulli](./PREREQUISITES.md#p5-bernoulli).

### Board / screenshot

![Log of L, Bernoulli with negative = success, derivative set to zero](./screenshots/composites/ch04-topic-04-bernoulli-mle-p-panel1of1.png)

**Figure — ~11:21–13:47:** $L(p)=p^m(1-p)^{n-m}$. Words: this is exactly the Bernoulli model; **negative is treated as success**; $m$ successes in $n$ trials. Then $\ell(p)=m\log p+(n-m)\log(1-p)$ and the critical-point equation $m/p-(n-m)/(1-p)=0$.

### What he is establishing

First take the log. He writes $\ell(\mu)=\log L(\mu)$ (same letter $L$ on the board for both; we keep $\ell$ for the log):

$$
\ell(\mu)=m\log\Phi(-\mu)+(n-m)\log\Phi(\mu).
$$

That is already nicer, but $\Phi$ is still nested. For simplicity he substitutes

$$
p:=P(X<0)=\Phi(-\mu),\qquad 1-p=P(X>0)=\Phi(\mu).
$$

The *likelihood* (he stresses: not the log) becomes the elementary product

$$
L(p)=p^m(1-p)^{n-m}.
$$

“Does this ring a bell?” It should. That is exactly the likelihood of a **Bernoulli** model in which a **negative sign is called a success**. You have seen $m$ successes in $n$ independent trials.

Differentiating $\Phi(-\mu)$ directly is the painful move. The cheap move is two-step: find the MLE $\hat p$ first, then convert back to $\hat\mu$. Do not fight $\Phi$ until $p$ is known.

Log of the coin:

$$
\ell(p)=m\log p+(n-m)\log(1-p).
$$

Differentiate with respect to $p$ and set the derivative to zero — “the standard idea” he will repeat at the end of the hour:

$$
\frac{d\ell}{dp}=\frac{m}{p}-\frac{n-m}{1-p}=0.
$$

Clear the fractions: $m(1-p)=(n-m)p$, hence $p=m/n$. So

$$
\hat p=\frac{m}{n}.
$$

He calls this “straightforward known to us.” The MLE of the probability of a negative observation *is* the fraction of negative observations. No $\Phi$ was harmed.

Micro numbers: $n=10$, $m=3$ gives $\hat p=0.3$. That $0.3$ is an estimate of $\Phi(-\mu)$, not of $\mu$ itself. The inversion is the next box.

You can now maximize the reduced problem. What is still missing is the map from $\hat p$ back through $\Phi^{-1}$ to a number on the $\mu$-axis — and a check that the sign of that number matches the notebook.

### Analogy for this topic only

You counted 3 left-ticks out of 10. Forget the hill for a minute. Pretend you only have a coin labeled “left.” The coin’s unknown bias is $p$.

**What single number $\hat p$ best explains 3 left and 7 right?** Memory of a “fair coin” does not answer — fairness was never assumed. The handful itself says $\hat p=0.3$. Differentiating the hill-area $\Phi(-\mu)$ directly is the same work with worse handwriting.

In lecture words: success = minus, $\hat p=m/n$, hill comes back after the coin.

### Local picture

```
  RECIPE (this hour, both problems)
    1. write L(parameter)
    2. take log
    3. d/d(parameter) = 0
    4. solve

  HERE the parameter is temporarily p, not μ

  signs:  − − − + + + + + + +     (m=3, n=10)
  coin:   S S S F F F F F F F
  p̂ = 3/10

  SEE → SYMBOLS
    fraction of minuses  →  p̂ = m/n
    left-area of the bell →  p = Φ(-μ)
```

Notice: calling minus a “success” is a naming choice. It must stay consistent when we invert.

### Bridge

We own $\hat p=m/n=\Phi(-\hat\mu)$. That is an area, not a mean. We still need the unique $z$ whose left-area is $m/n$, then flip the sign to get $\hat\mu$.

---

## Topic 5: Invert $\Phi$ and read the sign of $\hat\mu$ (14:01–18:31)

### Where this sits on the master map

This is **INVERT + CHECK**, the last box of Problem 1. The leftover is “$p$ is not $\mu$.” The tools are $\Phi^{-1}$ and the identity $-\Phi^{-1}(q)=\Phi^{-1}(1-q)$. Warm-up: [$\Phi$](./PREREQUISITES.md#p3-phi).

### Board / screenshot

![p̂ = Φ(-μ̂), invert, box μ̂ = Φ^{-1}((n-m)/n), symmetry −Φ^{-1}(q)=Φ^{-1}(1-q)](./screenshots/composites/ch05-topic-05-invert-phi-intuition-panel1of1.png)

**Figure — ~14:28–18:07:** $m/n=\Phi(-\hat\mu)$. Red sidebar: $\Phi(a)=b\Rightarrow a=\Phi^{-1}(b)$, with $a=-\hat\mu$, $b=m/n$. Then $-\hat\mu=\Phi^{-1}(m/n)$ so $\hat\mu=-\Phi^{-1}(m/n)$. Using $-\Phi^{-1}(q)=\Phi^{-1}(1-q)$ he rewrites and **boxes** $\hat\mu=\Phi^{-1}((n-m)/n)$.

### What he is establishing

Start from the definition we substituted: $p=\Phi(-\mu)$. Plug in the hats:

$$
\hat p=\Phi(-\hat\mu)\qquad\text{i.e.}\qquad \frac{m}{n}=\Phi(-\hat\mu).
$$

$\Phi$ is strictly increasing, so it has an inverse. If $\Phi(a)=b$ then $a=\Phi^{-1}(b)$. Here $a=-\hat\mu$ and $b=m/n$, therefore

$$
-\hat\mu=\Phi^{-1}\!\left(\frac{m}{n}\right)\qquad\Rightarrow\qquad\hat\mu=-\Phi^{-1}\!\left(\frac{m}{n}\right).
$$

A standard-Normal identity rewrites the minus in front of $\Phi^{-1}$. For any $q\in(0,1)$,

$$
-\Phi^{-1}(q)=\Phi^{-1}(1-q).
$$

Take $q=m/n$. Then

$$
\hat\mu=\Phi^{-1}\!\left(1-\frac{m}{n}\right)=\Phi^{-1}\!\left(\frac{n-m}{n}\right).
$$

That boxed line is the MLE. $(n-m)/n$ is the **fraction of pluses**. The estimated mean is the standard-Normal quantile of the plus-fraction.

Now the sanity check, in his words. If there are **many negative observations**, $m/n$ is large, so $\Phi(-\hat\mu)$ is large, so $-\hat\mu$ is positive, so $\hat\mu$ is **negative**. That “makes total sense”: a Normal that keeps landing left of zero should have a negative mean. The algebra and the picture agree.

He restates the **standard MLE method** one more time: whichever parameter you want, write the likelihood, take the log, differentiate, equate to zero. That slogan will govern Problem 2 as well — except the missing labels will force an extra expectation before the derivative.

Finally he flags a sibling estimator. You should also be comfortable with **maximum a posteriori (MAP)** estimation. He does **not** derive it. It is left as a viewer recap. This package follows him: MAP is a named homework, not a second formula on this page.

Micro numbers: $n=10$, $m=3$. Plus-fraction $=0.7$. A standard table (or `norm.ppf(0.7)`) gives $\Phi^{-1}(0.7)\approx +0.52$. Few minuses, positive mean. Walk it the other way: $m/n=0.3=\Phi(-\hat\mu)$ so $-\hat\mu=\Phi^{-1}(0.3)\approx -0.52$, hence the same $\hat\mu\approx +0.52$. If instead $m=8$, $\hat\mu=\Phi^{-1}(0.2)\approx -0.84$. Majority minuses, negative mean.

MAP stays homework: you would multiply $L(\mu)$ by a prior $P(\mu)$ and maximize that product. He does not pick a prior or take that derivative.

You can now compute $\hat\mu$ from a $\Phi^{-1}$ table. What this box cannot do is handle data whose *source* is itself a coin-flip between two machines. That is Problem 2.

### Analogy for this topic only

You estimated that 30% of sheep land left of the fence. Which hill-position $\mu$ would put 30% of a unit-width hill left of that fence? Slide the hill until the left-area matches $0.3$, then read the hill’s center. That slide *is* $\Phi^{-1}$.

If you report a positive $\mu$ after watching eight of ten sheep go left, you slid the hill the wrong way. The check is the picture, not extra algebra.

In lecture words: left-area = $m/n$, hill center = $\hat\mu=-\Phi^{-1}(m/n)$.

### Local picture

```
  m/n = 0.3 = Φ(-μ̂)
                    area 0.3
               |████--------
              -μ̂          0     (standard bell)
               μ̂ = −(that z) > 0

  SAME, rewritten
    plus-fraction = 0.7
    μ̂ = Φ⁻¹(0.7) > 0

  CHECK
    many minuses (m/n > 1/2)  ⇒  μ̂ < 0
    many pluses  (m/n < 1/2)  ⇒  μ̂ > 0
```

Notice: $\Phi^{-1}((n-m)/n)$ automatically has the correct sign. You do not add a sign by hand.

### Bridge

Problem 1 is closed except for the MAP homework. The next notebook is incomplete in a different way: every $x$ is fully visible, but the **machine that produced it** is not.

---

## Topic 6: Two-exponential mixture and latent $Z$ (18:31–22:52)

### Where this sits on the master map

This opens **PROBLEM 2 / LATENT**. MLE-by-hand hit a wall: the observed density is a *sum*, and $\log(\text{sum})$ has no closed maximizer. The new object is a **mixture** plus a hidden coin $Z$. Warm-up: [mixture](./PREREQUISITES.md#p6-mixture), [latent $Z$](./PREREQUISITES.md#p7-latent).

### Board / screenshot

![Mixture of two exponentials, θ={π,β1,β2}, constraints, binary Z, EM](./screenshots/composites/ch06-topic-06-two-exp-mixture-latent-panel1of1.png)

**Figure — ~20:04–22:35:** mixture density $f(x;\theta)=\pi\beta_1 e^{-\beta_1 x}+(1-\pi)\beta_2 e^{-\beta_2 x}$, $\theta=\{\pi,\beta_1,\beta_2\}$ with $0\le\pi\le 1$, $\beta_1>0$, $\beta_2>0$. “Use EM algorithm. Use a scalar binary $Z$ as a latent variable.” Solution starts $Z=1$ if $x$ comes from component 1.

### What he is establishing

The density of a positive random variable $X$ is a **mixture of two exponential densities**. For an observation $x$ parameterized by $\theta$,

$$
f(x;\theta)=\pi\,\beta_1 e^{-\beta_1 x}+(1-\pi)\,\beta_2 e^{-\beta_2 x}.
$$

The parameter is the triple $\theta=(\pi,\beta_1,\beta_2)$.

Read the pieces. $\pi$ is the probability of choosing **component 1**; $1-\pi$ is the probability of choosing component 2. Each component is exponential with its own **rate** $\beta_j>0$ (density $\beta_j e^{-\beta_j x}$ on $x>0$). He requires $0\le\pi\le 1$ (he also says “between zero and one”; “we can use equal to here, that’s fine”) and both rates strictly positive.

The job is not a one-shot MLE of that sum. The job is: **use the EM algorithm**, and specifically use a **binary latent** $Z$. Picture a coin. Heads: draw from exponential 1. Tails: draw from exponential 2. $Z=1$ if this $x$ came from component 1, $Z=0$ if from component 2. That $Z$ is $\mathrm{Bernoulli}(\pi)$.

He tells you the whole walk in one breath: **E-step, Q-function, and M-step** — “just go through the whole EM algorithm once.” Some treatments skip the name Q; he will use it as an intermediary.

Why EM rather than $\partial/\partial\theta\log f(x;\theta)=0$ on the mixture itself? Because $f$ is a **sum**. The log of a sum does not split, and the three-parameter critical point is ugly. Completing the data with $Z$ turns the sum into a *product of selected terms*, whose log *does* split. That algebra is the next box.

Micro numbers: $\pi=0.4$, $\beta_1=2$, $\beta_2=0.5$, one wait $x=1$. Mixture height $=0.4\cdot 2e^{-2}+0.6\cdot 0.5e^{-0.5}\approx 0.291$. You see $1$. You do not see the coin.

You can now write the observed density, name $\theta$, and state the latent coin. What is still missing is the joint density of the pair $(x,z)$ as if the coin had been written down.

### Analogy for this topic only

Two kitchens, one dining room. A waiter flips a coin ($\pi$), then fetches a plate from kitchen 1 (fast cook, large $\beta$) or kitchen 2 (slow cook, small $\beta$). You taste the plate ($x$). You never see the coin.

**Which coin bias and which two cooking speeds best explain tonight’s plates?** If you only write the blended menu $f(x)$, the derivative is a mess. If you pretend you saw the coin, each plate belongs to one kitchen and the derivatives split. EM is the honest version of that pretense.

In lecture words: coin = $\pi$, kitchens = $\mathrm{Exp}(\beta_1),\mathrm{Exp}(\beta_2)$, missing flip = $Z$.

### Local picture

```
            θ = (π, β1, β2)
                    │
                    ▼
              flip Z ~ Bern(π)
               /           \
            Z=1             Z=0
             │               │
        Exp(β1)          Exp(β2)
             │               │
             └────── x ──────┘
                    │
                    ▼
         you observe x, not Z

  constraints:  0 ≤ π ≤ 1 ,  β1 > 0 ,  β2 > 0
```

Notice: $x$ is complete as a *number* and incomplete as a *story*. The missing piece is the switch.

### Bridge

We have a coin-then-machine story and an observed mixture. To differentiate, we need the likelihood we *would* have written if every $z_i$ had been inked next to $x_i$.

---

## Topic 7: Complete-data density and log-likelihood (22:52–28:39)

### Where this sits on the master map

This is **COMPLETE LL**. Problem 2 has a latent; the leftover is “write a likelihood that *uses* $Z$.” The trick is an exponent that selects one component. Warm-up: [latent / complete data](./PREREQUISITES.md#p7-latent), [log](./PREREQUISITES.md#p4-log).

### Board / screenshot

![Z ~ Bernoulli(π); product of complete densities; z as exponent; log expanded](./screenshots/composites/ch07-topic-07-complete-data-ll-panel1of1.png)

**Figure — ~23:24–28:08:** $Z\sim\mathrm{Bernoulli}(\pi)$. Complete-data density $f(x,z;\theta)=\prod_i f(x_i,z_i;\theta)$, then the product $\prod_i\bigl(\pi\beta_1 e^{-\beta_1 x_i}\bigr)^{z_i}\bigl((1-\pi)\beta_2 e^{-\beta_2 x_i}\bigr)^{1-z_i}$. After the log: a sum of $z_i\log(\cdots)$ plus $(1-z_i)\log(\cdots)$, beginning to expand into $z_i(\log\pi+\cdots)$.

### What he is establishing

We have observations $x_1,\ldots,x_n$. For each $x_i$ there is a hidden $z_i$. The **complete data** is the list of pairs $(x_1,z_1),\ldots,(x_n,z_n)$.

Those pairs are IID, so the joint complete-data density is a product

$$
f(x,z;\theta)=\prod_{i=1}^n f(x_i,z_i;\theta).
$$

One pair has a compact form that looks magical until you plug in $z_i\in\{0,1\}$:

$$
f(x_i,z_i;\theta)=\bigl[\pi\beta_1 e^{-\beta_1 x_i}\bigr]^{z_i}\bigl[(1-\pi)\beta_2 e^{-\beta_2 x_i}\bigr]^{1-z_i}.
$$

If $z_i=1$, the second factor is raised to $0$ and becomes $1$; you keep only component 1. If $z_i=0$, the first factor becomes $1$; you keep only component 2. The exponent is a **selector**, not a mysterious power.

Now take the log of the product. Logs turn products into sums and pull exponents down:

$$
\ell(\theta)=\sum_{i=1}^n\Bigl\{z_i\log\bigl(\pi\beta_1 e^{-\beta_1 x_i}\bigr)+(1-z_i)\log\bigl((1-\pi)\beta_2 e^{-\beta_2 x_i}\bigr)\Bigr\}.
$$

Expand each log (product $\to$ sum, $\log e^{-\beta x}=-\beta x$):

$$
\ell(\theta)=\sum_{i=1}^n\Bigl\{z_i\bigl(\log\pi+\log\beta_1-\beta_1 x_i\bigr)+(1-z_i)\bigl(\log(1-\pi)+\log\beta_2-\beta_2 x_i\bigr)\Bigr\}.
$$

He assumes the log rules are comfortable. They are the entire reason complete data helps: every parameter now sits inside a *sum of simple terms*, ready to differentiate — **if** we knew the $z_i$.

We do not know them. That is the only remaining obstruction, and it is exactly what the E-step removes by replacing $z_i$ with an expectation.

Micro check: $z_i=1$, $x_i=1$, $\pi=0.4$, $\beta_1=2$. The $i$-th summand is $\log 0.4+\log 2-2\cdot 1$, and the component-2 chunk is multiplied by $0$ and vanishes.

You can now write complete-data $\ell(\theta)$ as a sum linear in each $z_i$. What you cannot do is evaluate it on the real notebook. The next box computes $\mathbb{E}[z_i\mid x_i]$.

### Analogy for this topic only

Each plate in the dining room now has a sticky note that *would* say “kitchen 1” or “kitchen 2.” With the notes, tonight’s score is: for every kitchen-1 plate add $\log\pi+\log\beta_1-\beta_1 x$, and for every kitchen-2 plate add the other triple. Adding is easy.

**What is tonight’s score if the notes were real — plate $x=1$ stamped kitchen 1, $\pi=0.4$, $\beta_1=2$?** Reciting the blended menu $f(x)$ does not answer; that menu hid the stamp. The complete-data answer is $\log 0.4+\log 2-2$. The notes are imaginary. EM will later pencil a *fractional* note ($0.7$ kitchen 1) and use the same addition.

In lecture words: sticky note = $z_i$, score with notes = complete log-likelihood.

### Local picture

```
  z = 1                         z = 0
  [π β1 e^{-β1 x}]^1            [π β1 e^{-β1 x}]^0 = 1
  [(1-π) β2 e^{-β2 x}]^0 = 1    [(1-π) β2 e^{-β2 x}]^1
  → keep kitchen 1              → keep kitchen 2

  n IID pairs
    (x1,z1) (x2,z2) ... (xn,zn)
         │
         ▼
    product of n selectors
         │  log
         ▼
    sum_i  z_i A_i + (1-z_i) B_i
    A_i = log π + log β1 − β1 x_i
    B_i = log(1-π) + log β2 − β2 x_i
```

Notice: $\ell$ is *linear* in each $z_i$. That is why taking $\mathbb{E}[z_i]$ later is a one-line swap.

### Bridge

The complete log-likelihood is ready, but every $z_i$ is blank. The E-step fills those blanks with posterior probabilities given the current $\theta$.

---

## Topic 8: E-step responsibilities by Bayes (28:39–34:19)

### Where this sits on the master map

This is the **E-STEP** box. The leftover from complete-data $\ell$ is “we do not know $Z$.” The move is Bayes, using a frozen $\theta^{\mathrm{old}}$. Warm-ups: [EM loop / Bayes line](./PREREQUISITES.md#p8-em), Tutorial 9’s conditionals if needed.

### Board / screenshot

![γ_i = E[Z_i | x_i; θ_old] = P(Z_i=1 | x_i) by Bayes; numerator π_old β1_old e^{-β1_old x}](./screenshots/composites/ch08-topic-08-estep-bayes-panel1of1.png)

**Figure — ~29:11–33:48:** after the expanded $(1-z_i)(\log(1-\pi)+\log\beta_2-\beta_2 x_i)$, he starts the E-step. $\gamma_i=\mathbb{E}[Z_i\mid x_i;\theta^{\mathrm{old}}]=P[Z_i=1\mid x_i;\theta^{\mathrm{old}}]$, written as the Bayes fraction $f(x_i\mid Z_i=1;\theta^{\mathrm{old}})P(Z_i=1;\theta^{\mathrm{old}})/f(x_i;\theta^{\mathrm{old}})$. Numerator $=\pi^{\mathrm{old}}\beta_1^{\mathrm{old}}e^{-\beta_1^{\mathrm{old}}x_i}$. $\theta^{\mathrm{old}}=\{\pi^{\mathrm{old}},\beta_1^{\mathrm{old}},\beta_2^{\mathrm{old}}\}$.

### What he is establishing

We do not know $z_i$. In the **E-step** we compute the **expected value of $Z_i$ given $X_i$**, using the **current** parameter estimate.

He introduces a second copy of the parameter, $\theta^{\mathrm{old}}$, so that “old” and “updated” do not share a letter. $\theta^{\mathrm{old}}=(\pi^{\mathrm{old}},\beta_1^{\mathrm{old}},\beta_2^{\mathrm{old}})$. The equations get cluttered; that is the price of honesty.

Call the mixing coefficient (he later just says “the coefficient”; the board letter looks like $\gamma$ / $\mathcal{D}$)

$$
\gamma_i:=\mathbb{E}[Z_i\mid x_i;\,\theta^{\mathrm{old}}].
$$

$Z_i$ is $0$ or $1$, so an expectation *is* a probability:

$$
\gamma_i=P(Z_i=1\mid x_i;\,\theta^{\mathrm{old}}).
$$

Bayes turns that into ingredients we already have:

$$
P(Z_i=1\mid x_i;\,\theta^{\mathrm{old}})=\frac{f(x_i\mid Z_i=1;\,\theta^{\mathrm{old}})\,P(Z_i=1;\,\theta^{\mathrm{old}})}{f(x_i;\,\theta^{\mathrm{old}})}.
$$

The prior $P(Z_i=1;\,\theta^{\mathrm{old}})$ is $\pi^{\mathrm{old}}$. The component density $f(x_i\mid Z_i=1)$ is exponential with rate $\beta_1^{\mathrm{old}}$. The denominator is the **mixture** density at $x_i$ under $\theta^{\mathrm{old}}$. Using this week’s half-updated $\beta$ inside the same fraction is the wrong move — the E-step freezes one menu. So

$$
\gamma_i=\frac{\pi^{\mathrm{old}}\,\beta_1^{\mathrm{old}}\,e^{-\beta_1^{\mathrm{old}}x_i}}{\pi^{\mathrm{old}}\beta_1^{\mathrm{old}}e^{-\beta_1^{\mathrm{old}}x_i}+(1-\pi^{\mathrm{old}})\beta_2^{\mathrm{old}}e^{-\beta_2^{\mathrm{old}}x_i}}.
$$

Auto-captions said “1 minus beta 2 old” in the denominator. The board and the mixture definition fix it: the second term is $(1-\pi^{\mathrm{old}})$ times the second exponential, not “$1-\beta_2$.”

The other responsibility is automatic:

$$
P(Z_i=0\mid x_i;\,\theta^{\mathrm{old}})=1-\gamma_i=\mathbb{E}[1-Z_i\mid x_i;\,\theta^{\mathrm{old}}].
$$

That **completes the E-step**. We have not touched $\pi,\beta_1,\beta_2$ as variables to optimize. We have only *scored* each point under the frozen old parameters.

Micro numbers: $x=1$, $\pi^{\mathrm{old}}=0.4$, $\beta_1^{\mathrm{old}}=2$, $\beta_2^{\mathrm{old}}=0.5$. Numerator $0.4\cdot 2e^{-2}\approx 0.108$. Second term $0.6\cdot 0.5e^{-0.5}\approx 0.182$. $\gamma\approx 0.108/0.290\approx 0.37$. A middling wait is a bit more “slow kitchen” than “fast kitchen.”

You can now compute a responsibility vector $(\gamma_1,\ldots,\gamma_n)$. What is still missing is the function of $\theta$ those $\gamma_i$ will be plugged into — the Q-function.

### Analogy for this topic only

Each plate gets a pencil note: “37% kitchen 1, 63% kitchen 2,” computed from last week’s estimated coin and cooking speeds. You do not flip a new coin.

**Given this plate $x=1$ and last week’s cards $(\pi=0.4,\beta_1=2,\beta_2=0.5)$, how believable is kitchen 1?** Reciting $\pi=0.4$ is the prior, not the answer. Bayes gives $\gamma\approx 0.37$. Using *this* week’s half-updated $\beta$ inside the same note mixes two recipes. The E-step is one frozen menu.

In lecture words: pencil note = $\gamma_i$, last week’s cards = $\theta^{\mathrm{old}}$, Bayes = the fraction on the board.

### Local picture

```
  Bayes, one plate x

           prior π_old          ×   fast density at x
   γ = ────────────────────────────────────────────────
        (that same product)  +  (1-π_old)×slow density

  SEE (x=1, π=0.4, β1=2, β2=0.5)
    num   = 0.4 · 2 e^{-2}     ≈ 0.108
    den   = 0.108 + 0.6·0.5 e^{-0.5} ≈ 0.290
    γ     ≈ 0.37
    1-γ   ≈ 0.63

  E-step done: every i has a (γ_i, 1-γ_i)
```

Notice: $\gamma_i$ depends on $\theta^{\mathrm{old}}$ and $x_i$ only. It does not depend on the *new* $\pi,\beta$ we are about to solve for.

### Bridge

Responsibilities are numbers. The complete log-likelihood is still a formula in the unknown $z_i$. The Q-function is what you get when you take the expectation of that formula — i.e. when every $z_i$ becomes $\gamma_i$.

---

## Topic 9: The Q-function (34:19–39:17)

### Where this sits on the master map

This is the **Q** box, the intermediary some books never name. The leftover after the E-step is “$\gamma_i$ is not yet plugged into a function we can maximize.” Warm-up: [EM loop](./PREREQUISITES.md#p8-em).

### Board / screenshot

![Q(θ|θ_old)=E[ℓ]; A_i and B_i; E[Z_i]A_i=γ_i A_i; Q as sum of γ A + (1-γ) B](./screenshots/composites/ch09-topic-09-q-function-panel1of1.png)

**Figure — ~34:47–38:49:** “Now the Q-function is” $Q(\theta\mid\theta^{\mathrm{old}})=\mathbb{E}_{Z\mid X;\,\theta^{\mathrm{old}}}[\ell]$. He writes the sum inside the expectation, marks $B_i$ on the second parenthesis, then in gold: $\mathbb{E}[Z_i]A_i=\gamma_i A_i$ and $\mathbb{E}[1-Z_i]B_i=(1-\gamma_i)B_i$. Final line: $Q=\sum_i\{\gamma_i(\log\pi+\log\beta_1-\beta_1 x_i)+(1-\gamma_i)(\log(1-\pi)+\log\beta_2-\beta_2 x_i)\}$.

### What he is establishing

Some treatments never say **Q-function**. They still perform this intermediary step. The name is not a standard everyone uses; the algebra is.

Leaving the $z_i$ as unknowns and trying to maximize $\ell$ anyway is the wrong move — those $z_i$ are blank. Replacing each $z_i$ by the number $\gamma_i$ we just computed is the right move. One plate with $\gamma=0.37$ and $x=1$ then contributes $0.37(\log\pi+\log\beta_1-\beta_1)+0.63(\log(1-\pi)+\log\beta_2-\beta_2)$ to the surrogate we will maximize.

He defines

$$
Q(\theta\mid\theta^{\mathrm{old}})=\mathbb{E}_{Z\mid X;\,\theta^{\mathrm{old}}}\bigl[\log f(X,Z;\theta)\bigr].
$$

In words: take the complete-data log-likelihood (a function of the unknown $Z$ and of the *free* parameter $\theta$) and average it under the posterior of $Z$ given $X$ and the *frozen* $\theta^{\mathrm{old}}$.

The inner log is the sum we already expanded. Nickname the two parentheses $A_i$ and $B_i$:

$$
A_i=\log\pi+\log\beta_1-\beta_1 x_i,\qquad B_i=\log(1-\pi)+\log\beta_2-\beta_2 x_i.
$$

Expectation of a sum is the sum of expectations. $A_i$ and $B_i$ do not depend on $Z$, so they factor out:

$$
\mathbb{E}[Z_i A_i]=\mathbb{E}[Z_i]\,A_i=\gamma_i A_i,
$$
$$
\mathbb{E}[(1-Z_i)B_i]=\mathbb{E}[1-Z_i]\,B_i=(1-\gamma_i)B_i.
$$

Therefore the whole Q-function is the complete log-likelihood with every $z_i$ replaced by $\gamma_i$:

$$
Q(\theta\mid\theta^{\mathrm{old}})=\sum_{i=1}^n\Bigl\{\gamma_i\bigl(\log\pi+\log\beta_1-\beta_1 x_i\bigr)+(1-\gamma_i)\bigl(\log(1-\pi)+\log\beta_2-\beta_2 x_i\bigr)\Bigr\}.
$$

E-step plus this substitution *is* “forming Q.” Next we **maximize** $Q$ in the three free parameters. When you differentiate with respect to $\pi$, only the $\log\pi$ and $\log(1-\pi)$ terms survive. When you differentiate with respect to $\beta_1$, only $\gamma_i(\log\beta_1-\beta_1 x_i)$ survives. Same for $\beta_2$. Independence of the unused terms is the whole simplification.

You can now write $Q$ as an ordinary function of $\pi,\beta_1,\beta_2$ with numerical weights $\gamma_i$. What is still missing is setting the three derivatives to zero and solving — the M-step.

### Analogy for this topic only

The imaginary sticky notes are now fractional: plate $i$ counts as a fraction of a kitchen-1 plate and the rest as kitchen 2.

**What is tonight’s expected score for the plate $x=1$ with a 37% kitchen-1 note?** Reciting the complete-data formula with a blank $z$ does not answer. Count it as $0.37$ of a kitchen-1 plate and $0.63$ of a kitchen-2 plate, then add. Some chefs never call that expected score “Q.” They still cook with it.

In lecture words: expected score = $Q$, fractions = $\gamma_i$, next move = maximize $Q$.

### Local picture

```
  complete ℓ = Σ  z_i A_i + (1-z_i) B_i     (z unknown)

  take E[ · | X, θ_old ]

  Q = Σ  γ_i A_i + (1-γ_i) B_i              (γ known)

  SEE (one i, γ=0.37, x=1)
    contributes  0.37 (log π + log β1 − β1)
               + 0.63 (log(1-π) + log β2 − β2)

  WHICH TERMS SEE WHICH PARAMETER
    π     ←  γ log π + (1-γ) log(1-π)
    β1    ←  γ (log β1 − β1 x)
    β2    ←  (1-γ) (log β2 − β2 x)
```

Notice: $Q$ is *not* the observed log-likelihood $\log f(x;\theta)$. It is a surrogate that is easy to maximize and that, at $\theta=\theta^{\mathrm{old}}$, touches the observed log-likelihood (a fact he does not prove; he just uses Q).

### Bridge

$Q$ is an explicit function of three parameters with the $\gamma_i$ held fixed. Three derivatives, three closed forms, then a loop — that is the last box.

---

## Topic 10: M-step closed forms, iterate, close (39:17–47:33)

### Where this sits on the master map

This is **M-STEP / LOOP**, then the course’s STOP. The leftover is “maximize $Q$.” Each parameter uses only its own terms. Warm-up: [EM loop](./PREREQUISITES.md#p8-em).

### Board / screenshot

![∂Q/∂π = 0 with S=Σγ; β1_new = Σγ / Σγx; β2_new = Σ(1-γ)/Σ(1-γ)x](./screenshots/composites/ch10-topic-10-mstep-iterate-recap-panel1of1.png)

**Figure — ~40:01–46:28:** $Q$ rewritten with $\gamma_i$. Equating $\partial Q/\partial\pi$ to zero: $\frac1\pi\sum\gamma_i=\frac1{1-\pi}\sum(1-\gamma_i)$, and $S=\sum\gamma_i$ implies $\sum(1-\gamma_i)=n-S$. Then $Q_{\beta_1}=\sum\gamma_i(\log\beta_1-\beta_1 x_i)$, derivative $\sum\gamma_i(1/\beta_1-x_i)=0$. Closed forms $\beta_1^{\mathrm{new}}=\sum\gamma_i\big/\sum\gamma_i x_i$ and $\beta_2^{\mathrm{new}}=\sum(1-\gamma_i)\big/\sum(1-\gamma_i)x_i$.

### What he is establishing

**M-step for $\pi$.** Keep only the $\pi$-terms of $Q$:

$$
Q_\pi=\sum_{i=1}^n\bigl\{\gamma_i\log\pi+(1-\gamma_i)\log(1-\pi)\bigr\}.
$$

This is the same shape as the Bernoulli log-likelihood from Problem 1, with *soft* counts $\sum\gamma_i$ and $\sum(1-\gamma_i)$ instead of $m$ and $n-m$. Differentiate and set to zero:

$$
\frac{1}{\pi}\sum_{i=1}^n\gamma_i=\frac{1}{1-\pi}\sum_{i=1}^n(1-\gamma_i).
$$

Let $S=\sum_i\gamma_i$. Then $\sum_i(1-\gamma_i)=n-S$. The critical-point line is $S/\pi=(n-S)/(1-\pi)$. Cross-multiply: $S(1-\pi)=\pi(n-S)$, so $S=\pi n$, hence

$$
\pi^{\mathrm{new}}=\frac{S}{n}=\frac{1}{n}\sum_{i=1}^n\gamma_i.
$$

In words: the updated coin bias is the **average responsibility** for component 1.

**M-step for $\beta_1$.** Only the first kitchen’s terms appear:

$$
Q_{\beta_1}=\sum_{i=1}^n\gamma_i\bigl(\log\beta_1-\beta_1 x_i\bigr).
$$

Derivative:

$$
\sum_{i=1}^n\gamma_i\Bigl(\frac{1}{\beta_1}-x_i\Bigr)=0\qquad\Rightarrow\qquad\frac{1}{\beta_1}\sum_i\gamma_i=\sum_i\gamma_i x_i.
$$

Hence

$$
\beta_1^{\mathrm{new}}=\frac{\sum_{i=1}^n\gamma_i}{\sum_{i=1}^n\gamma_i x_i}.
$$

That is the MLE of an exponential **rate** on a *weighted* sample: total weight over weight-times-$x$. (An exponential mean would have been the weighted average of the $x_i$; the rate is the reciprocal of that mean.)

**M-step for $\beta_2$.** The same argument with weights $1-\gamma_i$:

$$
\beta_2^{\mathrm{new}}=\frac{\sum_{i=1}^n(1-\gamma_i)}{\sum_{i=1}^n(1-\gamma_i)x_i}.
$$

Auto-captions mumbled the division; the tablet writes both fractions clearly.

Those three lines are one **M-step**. Then you **loop**: the new $\theta$ becomes $\theta^{\mathrm{old}}$, you recompute every $\gamma_i$, you maximize $Q$ again, and you stop when the parameters settle.

He ends the section on purpose. The sole purpose of the hour was to **reiterate** first-course MLE and EM so they sit on the surface for **generative models**. Next section of tutorials: **neural-network numerical problems**. The standard idea, said one last time: write the likelihood, differentiate with respect to the parameter, equate to zero. Be comfortable with mixture densities and with ordinary differentiation — “tricks of the trade.” Then bye.

Micro numbers, two points $x=(1,4)$, $\gamma=(0.8,0.2)$:

$$
\pi^{\mathrm{new}}=\frac{0.8+0.2}{2}=0.5,
\quad
\beta_1^{\mathrm{new}}=\frac{1}{0.8\cdot 1+0.2\cdot 4}=\frac{1}{1.6}=0.625,
\quad
\beta_2^{\mathrm{new}}=\frac{0.2+0.8}{0.2\cdot 1+0.8\cdot 4}=\frac{1}{3.4}\approx 0.294.
$$

Component 1, which claimed the short wait, gets a larger rate (faster clock). Component 2 sits on the long wait and gets a smaller rate. That is the M-step behaving.

You can now run one full EM cycle on a two-exponential mixture by hand. Still open, as he left them: MAP (homework), a general EM convergence proof, and the neural-network numericals promised next.

### Analogy for this topic only

After every plate has a fractional membership, each kitchen updates as if it had cooked a *fractional number of plates*.

**Two plates, waits $1$ and $4$, notes $0.8$ and $0.2$ kitchen 1. What is kitchen 1’s new rate?** Reciting last week’s $\beta_1$ does not answer. Claimed plates $=1$, claimed wait $=1.6$, new rate $=0.625$. The coin’s new bias is the average note, $0.5$. Then throw away last week’s notes and write new ones. Stopping after one rewrite leaves the kitchens mid-argument.

In lecture words: claimed plates = $\sum\gamma$, claimed wait = $\sum\gamma x$, new rate = their ratio; loop = EM.

### Local picture

```
           θ_old
             │
             ▼
        E-step: γ_i
             │
             ▼
        M-step
          π_new   = (γ1+…+γn) / n
          β1_new  = (γ1+…+γn) / (γ1 x1 + … + γn xn)
          β2_new  = ((1-γ1)+…) / ((1-γ1)x1 + …)
             │
             └── become θ_old ──►  repeat

  SEE  x=(1,4)  γ=(0.8,0.2)
    π  = 1/2
    β1 = 1 / 1.6 = 0.625     (fast; owns the short wait)
    β2 = 1 / 3.4 ≈ 0.29      (slow; owns the long wait)
```

Notice: if every $\gamma_i\in\{0,1\}$, these formulas collapse to ordinary per-kitchen MLEs. Soft $\gamma$ is the only new ingredient.

### Bridge

Both numerical machines are on the desk: invert $\Phi$ for censored Normals, and loop E–Q–M for unlabeled mixtures. The next tutorials pick up the third promised tool — backpropagation on paper — and leave MAP as the recap you do yourself.

---

## External references

Two layers, **both kept**.

1. **Start here** — the newer high-signal companions (famous teachers, mapped to this lecture’s hard boxes).
2. **Full topic map** — the previous per-topic list (2–3 companions each) **plus** any new entries already woven above. Use a group when one box still feels thin.

### Start here — high-signal companions

Only a few **widely used** companions — the ones people actually finish. Not a pile of random blogs. Use them after the matching topic, with this tutorial still closed.

**If “likelihood” still sounds like “probability” (Topics 1–4).** Josh Starmer’s [StatQuest — Probability vs Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4) is the short, famous fix: $L$ scores a *parameter* for data you already saw.

**If the MLE recipe is rusty (Topics 1, 4).** The same channel’s [StatQuest — Maximum Likelihood, clearly explained](https://www.youtube.com/watch?v=XepXtl9YKwc) is the classroom standard: write $L$, take the log, differentiate, set it to zero. That is exactly the coin $\hat p=m/n$ he reduces to.

**If $\Phi$ is still a table, not an area (Topics 3, 5).** [Khan Academy — Introduction to the Normal](https://www.youtube.com/watch?v=hgtMWR3TFnY) plus Brown’s [Seeing Theory — probability distributions](https://seeing-theory.brown.edu/probability-distributions/index.html) beat a dozen “$z$-score cheat sheet” posts. Drag the bell until the left area is $m/n$; that slide *is* $\Phi^{-1}$.

**If the E-step fraction $\gamma_i$ will not sit still (Topic 8).** StatQuest’s [Bayes’ Theorem](https://www.youtube.com/watch?v=9wCnvr7Xw4E) is the popular English for $P(Z\mid x)=P(x\mid Z)P(Z)/P(x)$. That fraction *is* the responsibility.

**If the two kitchens still feel like slogans (Topics 6, 10).** [StatQuest — MLE for the Exponential](https://www.youtube.com/watch?v=p3T-_LMrvBc) recovers $\hat\beta=1/\bar x$ for one clock — the hard-label limit of the M-step. The M-step just *weights* that formula by $\gamma$.

**If “complete data” is still a phrase (Topics 7–9).** Do and Batzoglou’s Nature Biotech primer [What is the EM algorithm?](https://www.nature.com/articles/nbt1406) is the short famous reason we invent pairs $(x,z)$ at all, and why Q is an *expected* complete log-likelihood.

**How to use.** $\Phi$ fog → Khan or Seeing Theory *before* Topic 3. Coin MLE → StatQuest MLE *before* Topic 4. Soft labels → StatQuest Bayes *before* Topic 8. Do not open ten tabs. One famous teacher per stuck idea.

---

### Full topic map — previous list plus new entries

Two or three companions **per topic**, listed **only here** (not under each topic). Mix of **video** and **blog/notes**. Wikipedia omitted. Watch/read after that map box; they do not replace the boards.

| Resource | Type | Matches lecture… | Why it helps |
|----------|------|------------------|--------------|
| [StatQuest — Probability vs Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4) | video | Topic 1 · GOAL | $L(\theta\mid\text{data})$ is a score for $\theta$, not $P(\theta)$. |
| [Steve Brunton — MLE with examples](https://www.youtube.com/watch?v=rCdxlN6Ph14) | video | Topic 1 · numerical MLE | Write $L$, maximize; names the MAP sibling he left as homework. |
| [MAP estimation, clearly explained](https://machinelearningplus.com/statistics/maximum-a-posteriori-map-estimation-clearly-explained/) | blog | Topic 1 · MAP pointer | One-page MLE vs MAP: prior $\times$ likelihood. |
| [StatQuest — MLE for the Normal](https://www.youtube.com/watch?v=Dn6b9fCIUpM) | video | Topic 2 · $X\sim\mathcal{N}(\mu,1)$ | Ordinary $\hat\mu=\bar x$, so you feel why missing $x_i$ block it. |
| [Penn State STAT 415 — Maximum Likelihood](https://online.stat.psu.edu/stat415/lesson/1/1.2) | notes | Topic 2 · model vs data | Written MLE: parameter in the model, data in $L$. |
| [Stanford CS109 — MLE lecture notes (PDF)](https://web.stanford.edu/class/archive/cs/cs109/cs109.1202/lectureNotes/LN21_parameters_mle.pdf) | notes | Topic 2 · Bernoulli setup | Starts MLE from a coin — the same reduction Topic 4 will use. |
| [Khan Academy — Introduction to the Normal](https://www.youtube.com/watch?v=hgtMWR3TFnY) | video | Topic 3 · standardize | Bell, mean shift, area language before $\Phi(-\mu)$. |
| [Khan Academy — Standard Normal table (area below)](https://www.youtube.com/watch?v=Fo4kitkFB3I) | video | Topic 3 · $\Phi$ | $z$-score then left-area $=P(Z<-\mu)$. |
| [Seeing Theory — probability distributions](https://seeing-theory.brown.edu/probability-distributions/index.html) | demo | Topic 3 · CDF slider | Drag a Normal; orange CDF is the left area. |
| [StatQuest — MLE, step by step](https://www.youtube.com/watch?v=XepXtl9YKwc) | video | Topic 4 · REDUCE + MAX | Log, differentiate, set 0 — the coin $m/n$ case. |
| [Penn State STAT 415 — MLE (Bernoulli)](https://online.stat.psu.edu/stat415/lesson/1/1.2) | notes | Topic 4 · $\hat p=m/n$ | $L(p)=p^m(1-p)^{n-m}$ written out. |
| [CMU 10-315 — MLE notes (PDF)](https://www.cs.cmu.edu/~10315/notes/10315_S24_Notes_MLE.pdf) | notes | Topic 4 · Bernoulli algebra | Same derivative $m/p-(n-m)/(1-p)=0$. |
| [3-Minute Data Science — Normal PDF, CDF, PPF](https://www.youtube.com/watch?v=3VYupIsbLlY) | video | Topic 5 · $\Phi^{-1}$ | PPF is $\Phi^{-1}$: area $m/n$ back to a location. |
| [Radford Mathematics — Inverse Normal](https://www.youtube.com/watch?v=Xlro6m4ssmc) | video | Topic 5 · invert | Given a left-area, find the $z$ — exactly $-\hat\mu=\Phi^{-1}(m/n)$. |
| [$\Phi^{-1}$ calculator (probabilitycourse)](https://www.probabilitycourse.com/calculator/phi_inverse.php) | demo | Topic 5 · check $\hat\mu$ | Type $0.7$ and read $\Phi^{-1}((n-m)/n)$. |
| [Victor Lavrenko — EM: how it works](https://www.youtube.com/watch?v=REypj2sy_5U) | video | Topic 6 · mixture + latent | Soft clustering = coin then a source; $Z$ never observed. |
| [StatQuest — MLE for the Exponential](https://www.youtube.com/watch?v=p3T-_LMrvBc) | video | Topic 6 · rate $\beta$ | $\hat\beta=1/\bar x$ for one exponential — the hard-label limit of Topic 10. |
| [Statlect — Exponential MLE](https://www.statlect.com/fundamentals-of-statistics/exponential-distribution-maximum-likelihood) | notes | Topic 6 · $\beta=n/\sum x$ | Written derivation of the rate MLE he will weight by $\gamma_i$. |
| [Do & Batzoglou — What is the EM algorithm? (Nature Biotech)](https://www.nature.com/articles/nbt1406) | primer | Topic 7 · complete vs incomplete | Why we invent pairs $(x,z)$ at all. |
| [Allauzen — Mixtures and the EM algorithm](https://allauzen.github.io/articles/MixturesAndEM.html) | blog | Topic 7 · complete log-ℓ | Bishop-style $z$ as a selector inside the product. |
| [Stephens — Introduction to EM](https://stephens999.github.io/fiveMinuteStats/intro_to_em.html) | notes | Topic 7 · two-component mix | Same $\theta=(\pi,\ldots)$ shape before E/M. |
| [StatQuest — Bayes’ Theorem](https://www.youtube.com/watch?v=9wCnvr7Xw4E) | video | Topic 8 · E-step | $P(Z\mid x)=P(x\mid Z)P(Z)/P(x)$ — the $\gamma_i$ fraction. |
| [3Blue1Brown — Bayes, geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM) | video | Topic 8 · Bayes picture | Prior $\times$ likelihood / evidence, visually. |
| [3Blue1Brown — Bayes lesson (text)](https://www.3blue1brown.com/lessons/bayes-theorem/) | blog | Topic 8 · same idea, written | Read if you prefer still diagrams to the video. |
| [Stats with Brian — EM clearly explained](https://www.youtube.com/watch?v=3zbAsgCf1Sw) | video | Topic 9 · Q | Missing-data E-step as an expected complete log-ℓ. |
| [Do & Batzoglou — What is the EM algorithm?](https://www.nature.com/articles/nbt1406) | primer | Topic 9 · Q name | Q as $\mathbb{E}[\text{complete }\log\ell]$ — the name some books skip. |
| [Allauzen — Mixtures and EM](https://allauzen.github.io/articles/MixturesAndEM.html) | blog | Topic 9 · $z\to\gamma$ | Replace indicators by responsibilities inside $\ell$. |
| [Stephens — Introduction to EM](https://stephens999.github.io/fiveMinuteStats/intro_to_em.html) | notes | Topic 10 · M-step + loop | Closed-form weighted updates, then iterate. |
| [StatQuest — Exponential MLE](https://www.youtube.com/watch?v=p3T-_LMrvBc) | video | Topic 10 · $\beta=\sum\gamma/\sum\gamma x$ | Hard-label $\hat\beta=n/\sum x$; EM just weights by $\gamma$. |
| [Stats with Brian — EM clearly explained](https://www.youtube.com/watch?v=3zbAsgCf1Sw) | video | Topic 10 · iterate | One E/M cycle is not the algorithm; the loop is. |

**How to use:** after Topic 5, one invert-normal video + the $\Phi^{-1}$ calculator. After Topic 8, StatQuest Bayes (or 3Blue1Brown) before re-reading $\gamma_i$. After Topic 10, Stephens plus StatQuest exponential MLE — then check that $\pi^{\mathrm{new}}$ and $\beta^{\mathrm{new}}$ match the tablet.

---


## Sources

- Video: [Tutorial 10 : Review of Machine Learning 1](https://www.youtube.com/watch?v=wjSKM1xFoSU) · NPTEL — Indian Institute of Science, Bengaluru
- Speaker overlay: Chandan Jayaram
- Auto-captions in `raw/captions.en.timed.txt` (cleaned in these notes; denominator “$1-\beta_2$” restored to $(1-\pi)\beta_2$)
- Boards transcribed from `screenshots/composites/`
- **Code audit:** the tablet is handwritten math only. No Python / autograd / training loop appears. These notes add **no invented code blocks** (math_technical; `$` / `$$` only).
