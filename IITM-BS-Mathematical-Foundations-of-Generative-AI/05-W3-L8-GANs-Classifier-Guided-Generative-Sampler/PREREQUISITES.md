# PREREQUISITES — warm-up before the module

> **Do this first.** Then open [NOTES.md](./NOTES.md) at the Executive Summary map.
> One screen per idea. No prior GAN or calculus background assumed — every idea below
> is built from a small concrete scene.

```
  After this warm-up you can say in plain words:

  "A distribution is a law that produces samples; a dataset is a pile of draws."
  "A classifier draws a boundary; its score says how much it believes a sample is real."
  "Training a classifier = maximizing the log-likelihood of the answers being right."
  "A GAN is a min–max game hunting a saddle point — and saddles are slippery."
  "A ⇒ B does not mean B ⇒ A — the whole trap of this lecture lives here."
```

---

## 1. Two clouds of samples — the real pile and the fake pile

<a id="p1-two-clouds"></a>

A **distribution** is an invisible law that produces samples. A **dataset** is a pile
of actual draws from that law — a fingerprint of the law, not the law itself.

Imagine two workshops. One prints genuine banknotes (the **real law** $p_x$). The other
is a photocopier with knobs (the **model law** $p_\theta$). You never see either
workshop's blueprint. You only hold two **piles of notes**: one real pile (the dataset
$D$) and one printed pile (what the photocopier produces when you feed it blank paper
$z$ — usually random noise).

```
   real pile D  (drawn iid from p_x):   ₹ ₹ ₹ ₹ ₹
   fake pile    (G_θ(z) for noise z):   ¤ ¤ ¤ ¤ ¤
   we NEVER see p_x or p_θ themselves — only piles
```

- Micro example: 1000 real photos of cats (pile one) vs 1000 images a generator just
  printed (pile two). The laws are invisible; the piles are all you have.
- In the lecture's language: real pile $= p_x$ samples, printed pile $= p_\theta$ samples,
  knobs $= \theta$.

---

## 2. A divergence — a score for "how different two laws are"

<a id="p2-divergence"></a>

A **divergence** $D(p_x \| p_\theta)$ is a ruler that reads a single number: how far
apart the two laws are. Zero means identical; bigger means more different.

You would love to turn the knobs $\theta$ until that number hits 0. The catch from
earlier modules: the formula for $D$ needs the **density formulas** of both laws, and
you have neither — only the two piles. So the course first builds a **lower bound**
on $D$ that needs only samples (the variational trick), and trains against that.

```
   want:  D(p_x ‖ p_θ)  →  0
   have:  only samples  →  compute a LOWER BOUND instead
   train: push the bound down by moving θ
```

---

## 3. Binary classifiers and decision boundaries

<a id="p3-binary-classifier"></a>

A **binary classifier** is a machine that answers one question: "which pile did this
item come from?" It looks at one item and outputs a **score between 0 and 1** —
read it as "how much I believe this is real": near 1 = real pile, near 0 = fake pile.

On a map of samples, a classifier draws a **decision boundary**: everything on one
side is called real, everything on the other side is called fake.

```
        boundary line (classifier)
              │
    real  × × │ ×      ← classifier says "1" on this side
          × × │ 
   fake   · · │ · ·    ← classifier says "0" on this side
              │ ·
```

- Micro example: an email spam filter. "1" = inbox, "0" = spam; the boundary is its rule.
- In the lecture's language: the classifier is $D_w$, a neural network with knobs $w$;
  $D_w(x) \approx 1$ means "likely from $p_x$".

---

## 4. Likelihood and log-likelihood — scoring a rule by its track record

<a id="p4-likelihood"></a>

A classifier's quality is judged on items whose truth you know. If the item is real
and the classifier scored it 0.9, good; if it scored 0.1, bad. The **likelihood** of
a classifier is the product of the scores it gave to the correct answers across the
pile. Multiplying hundreds of small numbers underflows computers, so we use
**log-likelihood**: logs turn products into sums, and the best rule is unchanged.

```
   rule A on 4 real notes: scores 0.9, 0.8, 0.95, 0.9   → product 0.616
   rule B on same notes:   scores 0.5, 0.4, 0.3, 0.6    → product 0.036
   log(product A) ≈ −0.48      log(product B) ≈ −3.32
   A is the better rule — log-likelihood is HIGHER (less negative)
```

"Maximize the likelihood of the data coming from the real distribution" means exactly:
tune $w$ so the classifier keeps calling real items real. The lecture writes this as
an **expected** log-likelihood — an average over the whole pile, not one item.

---

## 5. Expectation = long-run average (and why samples can stand in)

<a id="p5-expectation-average"></a>

An **expectation** $\mathbb{E}_{p_x}[h(x)]$ is the average value of $h$ over *every*
possible draw from $p_x$, weighted by likelihood. You cannot enumerate the whole law,
but the **law of large numbers** says: average $h$ over a big iid pile and you land
near the true expectation.

```
   goal: average h over ALL of p_x        (invisible law)
   do:   average h over n drawn samples   (visible pile)
   (1/n) Σ h(x_i)  →  E[h]   as n grows
```

One condition the course repeats: the pile must come from **the same law** you are
averaging over. The GAN objective will have one expectation over $p_x$ (real pile)
and one over $p_\theta$ (fake pile) — each gets averaged over its own pile.

---

## 6. Min–max games and saddle points

<a id="p6-minmax-saddle"></a>

A **min–max** problem is a tug-of-war over one scoreboard $J$ shared by two players
with opposite goals: one player tunes knobs $w$ to **push $J$ up**, the other tunes
knobs $\theta$ to **push $J$ down**.

$$
\min_{\theta}\; \max_{w}\; J(\theta, w)
$$

A **saddle point** is a resting configuration where neither player can improve alone:
at that $\theta$, no $w$ can push $J$ higher; at that $w$, no $\theta$ can push $J$
lower. Picture a horse saddle: it curves **up** along the spine (the maximizer's
direction) and **down** across the flanks (the minimizer's direction) — the saddle
point is the flat middle.

```
        up-curve (w's win)       down-curve (θ's win)
           \      /                  ···
            \    /      the saddle:  —•—   max in one dir, min in the other
             \__/                    
```

Saddles are slippery: two players taking turns can circle around the middle forever
without settling — remember this when the lecture says "notoriously unstable".

---

## 7. Lower bounds — when you can't touch the real number

<a id="p7-lower-bound"></a>

Suppose you cannot compute some quantity $D$ directly (the true divergence), but you
can compute a **lower bound** $J \le D$ that *does* use only what you have (samples).
Two rules make bounds useful:

1. Push the bound **up** (better $w$) → it squeezes toward the truth from below.
2. Push the bound **down** (better $\theta$) → the thing above it has less room and
   tends to shrink too.

```
   truth D  =  ════════  (unreachable)
   bound J  =  ────────  (computable; max over w lifts it, min over θ drops it)
   J ≤ D    — work on J, aim at D
```

The earlier module built exactly such a bound for the $f$-divergence. This module
shows the same bound wearing a classifier costume.

---

## 8. One-way logic — the trap this lecture is built on

<a id="p8-one-way-logic"></a>

"$A$ implies $B$" never means "$B$ implies $A$." Rain implies wet streets; wet streets
do not imply rain (someone may have washed them).

```
   p_θ = p_x   ──►  classifier fails          (guaranteed)
   classifier fails  ──►  p_θ = p_x   ✗ NOT guaranteed
```

- Micro example: "If the photocopier is perfect, the expert can't tell the piles
  apart" (true). "The expert couldn't tell them apart, so the copier is perfect"
  (false — maybe the expert had a bad day).

This single asymmetry is the reason a GAN cannot be trained with a **fixed**
classifier — the lecture's central trap (Topic 4) sits exactly on the wet-street
mistake.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).
Quiz later: [quiz.html](./quiz.html) — Part A covers this file.
