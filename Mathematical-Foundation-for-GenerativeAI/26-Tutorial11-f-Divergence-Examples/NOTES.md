# Tutorial 11 — f-Divergence and Examples

**Video:** [Tutorial 11 – f-Divergence and Examples](https://www.youtube.com/watch?v=GjxuVZeMSfE) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Lecture this complements:** [Lec 03 f-Divergence](../25-Lec03-f-Divergence-Examples/NOTES.md)  
**Course:** Mathematical Foundations of **Generative AI** (~48 min)  
**Speaker:** tutorial TA (complements Prof. Pratush) · proofs, KL / reverse KL / TV / JSD, why KL is not a metric

---

## Table of Contents

1. [Topic 1 — Redefine f-divergence](#topic-1-redefine-f-divergence-0001–0334) (00:01–03:34)
2. [Topic 2 — Likelihood ratio and f(1)=0](#topic-2-likelihood-ratio-and-f10-0334–0647) (03:34–06:47)
3. [Topic 3 — Jensen: D_f ≥ 0](#topic-3-jensen-d_f--0-0647–1013) (06:47–10:13)
4. [Topic 4 — Zero iff p = q](#topic-4-zero-iff-p--q-1013–1541) (10:13–15:41)
5. [Topic 5 — Child: KL and MLE](#topic-5-child-kl-and-mle-1541–1859) (15:41–18:59)
6. [Topic 6 — Child: reverse KL](#topic-6-child-reverse-kl-1859–2053) (18:59–20:53)
7. [Topic 7 — Child: total variation](#topic-7-child-total-variation-2053–2257) (20:53–22:57)
8. [Topic 8 — Child: JSD and the GAN zoo](#topic-8-child-jsd-and-the-gan-zoo-2257–3010) (22:57–30:10)
9. [Topic 9 — Four axioms; symmetry fails](#topic-9-four-axioms-symmetry-fails-3010–4123) (30:10–41:23)
10. [Topic 10 — Triangle fails; TV homework](#topic-10-triangle-fails-tv-homework-4123–4809) (41:23–48:09)
11. [External references](#external-references)
12. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

This tutorial **proves** what Lecture 03 assigned as homework. It rewrites $f$-divergence as $\mathbb{E}_Q[f(R)]$ with $R=p/q$, which is legal once $P$ is absolutely continuous with respect to $Q$ — not because we assume $P=Q$. Jensen plus $f(1)=0$ gives $D_f\ge 0$; if $f$ is strictly convex at $1$, then $D_f=0$ if and only if $p=q$. Four springs recover KL ($u\log u$), reverse KL ($f=-\log u$), total variation, and Jensen–Shannon. Two Bernoulli coins then kill the slogan “KL is a metric”: symmetry fails ($0.368\neq 0.511$) and the triangle fails ($1.758>0.879$). Total variation’s four tickets are homework; there is no code on this tablet.

**Worldview arc:** from “$f$-divergence slogan” **to** “a Jensen proof plus a numerical takedown of KL-as-distance.”

**Hour at a glance (whole video).** He writes $D_f(P\|Q)=\int q\,f(p/q)\,dx$ again, then spends the first third on why that ratio is even allowed. The red line is $P\ll Q$ (absolute continuity): whenever $Q$ gives a set mass zero, $P$ must too. That licenses $R=p/q$. It is **not** the sentence “assume $P=Q$” — identity already forces $R\equiv 1$ and $D_f=0$. Because $f$ is convex, Jensen says $\mathbb{E}_Q[f(R)]\ge f(\mathbb{E}[R])$. The inner expectation is $1$ (the $q$ cancels, $\int p=1$), and $f(1)=0$, so $D_f\ge 0$. If $f$ is strictly convex at $1$, equality needs $R$ constant $Q$-almost surely, hence $R=1$, hence $p=q$. Those are the two homework properties.

Then he swaps springs. $f(u)=u\log u$ is KL (and min-KL is MLE). This TA writes $f(u)=-\log u$ for reverse KL — Lecture 03 left $\log u$ as an exercise; watch the sign. $\tfrac12|u-1|$ is total variation. JSD expands to the average of two KLs to the midpoint $M=(P+Q)/2$. Last third: is KL a metric? Four tickets. Non-negativity and identity already hold. Symmetry dies on $P=(0.9,0.1)$, $Q=(0.5,0.5)$: $0.368\neq 0.511$. Triangle dies on $p=(0.1,0.9)$, fair $q$, $r=(0.9,0.1)$: $1.758>0.879$. TV’s four tickets are your homework. Chalkboard arithmetic only — do not invent Python.

### System context

```
  ╔══════════════════════════════════╗
  ║ Lec 03: defined D_f, named kids  ║
  ║ Lec 03 HW: prove ≥0 and =0 iff   ║
  ╚══════════════╤═══════════════════╝
                 │ this tutorial (~48 min)
                 ▼
        ┌────────────────────┐
        │ Proofs + numbers   │
        └────────────────────┘
```

### Main blueprint

```
  D_f(P∥Q) = ∫ q f(p/q) dx = E_Q[ f(R) ]
  R = p/q
  f convex, lsc, f(1)=0
         │ Jensen
         ▼
  E[f(R)] ≥ f(E[R]) = f(1) = 0     ← NONNEG
         │ strict convexity at 1
         ▼
  equality iff R constant Q-a.s.
  E[R]=1 ⇒ R=1 ⇒ p=q a.e.         ← IDENTITY
         │
         ├─ f(u)=u log u      →  KL(P∥Q)     (= MLE when minimized)
         ├─ f(u)= −log u      →  KL(Q∥P)     reverse
         ├─ f(u)=½|u−1|       →  TV = ½∫|p−q|
         └─ JSD f             →  ½ KL(P∥M)+½ KL(Q∥M)
                                 M=(P+Q)/2
         │
         ▼
  Is KL a metric?  4 tickets
    (1) ≥0          YES
    (2) =0 iff =    YES
    (3) symmetry    NO   0.368 ≠ 0.511
    (4) triangle    NO   1.758 > 0.879
         │
  ┌ · · · ┴ · · · ┐
  │ HW: is TV a    │
  │     metric?    │
  └ · · · · · · · ┘
```

### Scenario walkthrough

Walk this **one** story through the blueprint above. Each step answers “so what?” for the next box.

**Story:** two coins, $P=(0.9,0.1)$ and $Q=(0.5,0.5)$, plus a third $r=(0.9,0.1)$ opposite a $p=(0.1,0.9)$. You will prove $D_f\ge 0$, $=0$ only when the hills match, and that **KL is not a metric**.

1. **What is $D_f$?** $\int q\,f(p/q)\,dx$. Weight $q$, spring $f$, reading $p/q$. That is the DEF.

2. **May you even form $p/q$?** Only if $P\ll Q$ (absolute continuity): $Q$'s blank streets cannot hide $P$'s cathedral. That is **not** “assume $P=Q$” — identity already forces $D_f=0$. That is the LICENSE.

3. **Why $f(1)=0$?** Matched hills give $R=1$. The spring must read zero there, or “drive $D$ to zero” is impossible.

4. **Why $D_f\ge 0$?** $D_f=\mathbb{E}_Q[f(R)]\ge f(\mathbb{E}[R])$ (Jensen). $\mathbb{E}[R]=1$, so the right side is $f(1)=0$. That is NONNEG.

5. **When is it zero?** If $f$ is strictly convex at $1$, equality needs $R$ constant $Q$-a.s., hence $R=1$, hence $p=q$. That is IDENTITY.

6. **Which springs?** $u\log u$ → KL (also MLE). This TA writes $f=-\log u$ → reverse KL. $\tfrac12|u-1|$ → TV. JSD = average of two KLs to the midpoint. That is the children.

7. **Is KL a metric?** Ticket (3) symmetry: $0.368\neq 0.511$ on the two coins. Ticket (4) triangle: $1.758>0.879$ from $p$ to $r$ via fair. Both fail. TV’s four tickets are homework. No Python — chalkboard only.

```
  two coins  P=(0.9,0.1)  Q=(0.5,0.5)
  KL(P∥Q)≈0.368    KL(Q∥P)≈0.511     ← not symmetric
  add r=(0.9,0.1) opposite of p=(0.1,0.9)
  KL(p,r)≈1.758  >  0.368+0.511=0.879   ← not a triangle
```

### Failure / contrast path

```
  call KL a distance          ──X──►  fails symmetry and triangle
  skip f(1)=0                 ──X──►  Jensen does not hit 0
  assume P=Q to define p/q    ──X──►  that is identity, not the license
  use +log u for reverse KL   ──X──►  this TA writes −log u
  invent Python               ──X──►  no code on this tablet
```

### STOP / out of scope

- Variational estimation of $D$ from samples (next lectures).
- Full proof that TV *is* a metric (homework).
- What a GAN is (promised later).

### Load-bearing claims (closed-book)

- $P\ll Q$ licenses $R=p/q$. $P=Q$ is stronger and already forces $D_f=0$ — do not swap those jobs.
- $D_f=\mathbb{E}_Q[f(p/q)]$.
- Jensen + $\mathbb{E}[R]=1$ + $f(1)=0$ ⇒ $D_f\ge 0$.
- Strict convexity at $1$ ⇒ $D_f=0$ if and only if $p=q$.
- $u\log u$ → KL; $-\log u$ → reverse KL; $\tfrac12|u-1|$ → TV; JSD = average of two KLs to the midpoint.
- KL fails symmetry ($0.368\neq 0.511$) and triangle ($1.758>0.879$).
- Chalkboard numbers only. No invented Python.

---

## Topic 1: Redefine f-divergence (00:01–03:34)

### Where this sits on the master map

**DEF.** Lec 03 stated the formula; this hour writes it again as $D_f(P\|Q)$ with densities $p,q$. Warm-up: [law vs height](./PREREQUISITES.md#p1-law).

### Board / screenshot

![D_f(P∥Q)=∫ q f(p/q) dx; f:ℝ₊→ℝ](./screenshots/composites/ch01-topic-01-redefine-fdiv-panel1of1.png)

**Figure — ~00:26–03:14:** “Given 2 probability distribution functions $(P,Q)$ with corresponding density functions $(p,q)$.” Then $D_f(P\|Q)=\int_{\mathcal{X}} q(x)\,f(p(x)/q(x))\,dx$. $f:\mathbb{R}_+\to\mathbb{R}$.

### What he is establishing

The tutorial’s job list: redefine $f$-divergence, **prove** the properties, work examples, and prove **KL is not a distance metric**. Complementary to Prof. Pratush’s Lec 03 — that lecture named the family; this hour does the algebra.

Capital $P,Q$ are laws; small $p,q$ are their densities. The integral is weighted by **$q$**, and $f$ eats the **ratio** $p/q$. Same convention as Lec 03 if you identify $Q$ with the model $p_\theta$. Swapping the names is a *different* integral — that is already a hint that this $D$ will not be a metric.

$f$ maps **non-negative reals including $0$** into $\mathbb{R}$, is **convex**, **left/lower semi-continuous**, and **$f(1)=0$**. Convexity is what will let Jensen fire. The axiom $f(1)=0$ is the zero of the ruler: two identical hills must score zero. Lower semi-continuity is the technical glue that keeps $f$ from jumping down at $0$; he lists it and moves on.

A micro pair you can hold: let $p$ be height $2$ on $[0,0.5]$ and $q$ height $1$ on $[0,1]$. At $x=0.25$ the ratio is $2$; at $x=0.75$ the first hill is $0$ and the second is $1$, so the ratio is $0$. The integral averages $f$ of those readings using $q$ as the visiting schedule. The trap is to treat $p(x)$ as a probability in $[0,1]$ — heights can be $2$. The right move is: $P,Q$ are laws, $p,q$ are heights, $f$ eats the ratio.

You can now write the integral from the job list. Why $f(1)=0$ is not decoration, and why you are even allowed to form $p/q$, are still open.

### Analogy for this topic only

Two hills. At each spot you read “how many times taller is the first hill?” and feed that number to a **spring**. Average the spring’s reading using the *second* hill as the visiting schedule. That average is the score.

In lecture words: visiting schedule = $q$, spring = $f$, reading = $D_f$.

### Local picture

```
  P, Q   laws
  p, q   heights
  R(x) = p(x)/q(x)
  D_f(P∥Q) = ∫ q(x) f(R(x)) dx
  f: ℝ₊ → ℝ, convex, lsc, f(1)=0
```

Notice: the weight is $q$, not $p$. Swapping $P$ and $Q$ is a different integral.

### Bridge

The formula is up. We still do not know why $f(1)$ must be zero — that is what makes Jensen spit out $D_f\ge 0$.

---

## Topic 2: Likelihood ratio and $f(1)=0$ (03:34–06:47)

### Where this sits on the master map

**RATIO.** Leftover: why that extra axiom. Warm-up: [ratio](./PREREQUISITES.md#p2-ratio), [convex](./PREREQUISITES.md#p3-convex).

### Board / screenshot

![f(1)=0; assume P absolutely continuous wrt Q; R=p/q](./screenshots/composites/ch02-topic-02-ratio-f-of-one-panel1of1.png)

**Figure — ~03:54–06:25:** red line — “Assume $P$ is **absolutely continuous wrt $Q$** so that $p(x)/q(x)$ is well defined whenever needed.” Then **likelihood ratio** $R(x)=p(x)/q(x)$. $f$ is convex so Jensen applies. Spoken: Wikipedia is fine for the *proof* of Jensen; Tutorial 8 only defined it. (ASR “ration” on the tablet is **ratio**.)

### What he is establishing

The red line is the license to divide. He does **not** assume $P=Q$. He assumes $P$ is **absolutely continuous with respect to $Q$** ($P\ll Q$): whenever $Q$ gives a set mass zero, $P$ must too. Then $p/q$ is a legal height wherever the integral needs it. $P=Q$ is much stronger — that would already force $R\equiv 1$ and $D_f=0$. Mixing those two sentences makes the hour look circular.

Write $R(x)=p(x)/q(x)$ and call it the **likelihood ratio**. Micro: heights $p=2$, $q=1$ give $R=2$; matched heights give $R=1$. Without a name for that ratio, Jensen has nothing to eat. $f$ still eats a *number* at each $x$, not the whole function at once.

Because $f$ is convex, **Jensen’s inequality** is legal. He will not reprove Jensen. Tutorial 8 stated it; Markov and Chebyshev were also only defined, not proved. If you need a proof of Jensen or of convexity, he points you to a standard page rather than derailing the hour.

The second job of this box is the axiom $f(1)=0$. If $f(1)$ were $7$, two identical hills would score $7$, and “drive $D$ to zero” would be impossible. When the hills match, $R=1$, so the spring must read zero there. That is the zero of the ruler, not calligraphy.

You can now form $R$ under $P\ll Q$ and you know why the spring is calibrated at $1$. The next move treats $D_f$ as an **expectation of $f(R)$ under $Q$**.

### Analogy for this topic only

Two maps of one town. You want to read “how many times taller is map A at this church?”

**May you ask that on a street map B left blank?** No — if A hid a cathedral there, the ratio blows up. Absolute continuity is the rule “B’s blank streets cannot hide A’s cathedral.” That is *not* the same as “the two maps are identical.”

**What should the city-wide average read when the maps *are* identical?** If the spring reads $7$ even when every church matches, identical maps never score zero. The spring must read zero at $1$.

In lecture words: $P\ll Q$ licenses $R=p/q$; identical maps ⇒ $R=1$ ⇒ $f(1)$ must be $0$.

### Local picture

```
  P ≪ Q     (absolute continuity — license to divide)
  R(x) = p(x)/q(x)
  if p = q  then  R ≡ 1
  we will need f(1) = 0
  so that “same hills” scores 0

  P = Q  is stronger than  P ≪ Q
  (identity, not the license)
```

Notice: $f(1)=0$ is the zero of the ruler. $P\ll Q$ is the license to form the reading. Do not swap those jobs.

### Bridge

Rewrite $D_f$ as $\mathbb{E}_Q[f(R)]$ and feed it to Jensen.

---

## Topic 3: Jensen: $D_f\ge 0$ (06:47–10:13)

### Where this sits on the master map

**NONNEG.** The first homework from Lec 03. Warm-ups: [Jensen](./PREREQUISITES.md#p4-jensen), [expectation](./PREREQUISITES.md#p5-expect).

### Board / screenshot

![E_Q[f(R)] ≥ f(E_Q[R]) = f(1) = 0](./screenshots/composites/ch03-topic-03-jensen-nonneg-panel1of1.png)

**Figure — ~07:05–09:52:** the red $P\ll Q$ line is still on the tablet. Then $D_f=\mathbb{E}_Q[f(R)]\ge f(\mathbb{E}_Q[R])$. Then $\mathbb{E}_Q[R]=\int q\cdot(p/q)=\int p=1$, so the right side is $f(1)=0$.

### What he is establishing

The same red license is still in force: $P\ll Q$, so $R=p/q$ is a random number $Q$ is allowed to average. With that, $D_f$ **is** an expectation under $Q$:

$$
D_f(P\|Q)=\mathbb{E}_Q\bigl[f(R)\bigr],\qquad R=p/q.
$$

Jensen, $f$ convex:

$$
\mathbb{E}_Q[f(R)]\ge f\bigl(\mathbb{E}_Q[R]\bigr).
$$

Evaluate the inside. This is the only cancellation in the proof:

$$
\mathbb{E}_Q[R]=\int q(x)\frac{p(x)}{q(x)}\,dx=\int p(x)\,dx=1
$$

because $p$ is a density. Therefore

$$
D_f\ge f(1)=0.
$$

That is the first property. The trap is to skip $f(1)=0$ and stop at “$D_f\ge\text{some junk}$,” which is useless as a training score. The other trap is to skip $P\ll Q$ — then the cancellation is illegal wherever $q=0$ and $p>0$.

A micro check you can do without the video: if $p=q$, then $R\equiv 1$, so $\mathbb{E}_Q[f(R)]=f(1)=0$. The inequality is tight on matched hills. Whether it can be tight on *unmatched* hills is the next box.

You can now reproduce the three-line proof. When equality holds is still open.

### Analogy for this topic only

City-wide, the spring’s average reading is at least the spring’s reading at the **average ratio**.

**What is that average ratio?** Reciting “some number between the hills” does not answer. Total mass of $p$ is $1$, so $\mathbb{E}_Q[R]=1$. The spring reads $0$ at $1$. The city-wide average cannot go negative.

In lecture words: average ratio $=1$, $f(1)=0$, $D_f\ge 0$.

### Local picture

```
  D_f = E_Q[ f(R) ]
        ≥ f( E_Q[R] )     Jensen, f convex
        = f(1)            because ∫p = 1
        = 0               axiom on f
```

Notice: the $q$ in the integral **cancels** in $\mathbb{E}[R]$. That cancellation is the whole trick.

### Bridge

$\ge 0$ is half a ruler. We still need “$=0$ only when the hills match.”

---

## Topic 4: Zero iff $p=q$ (10:13–15:41)

### Where this sits on the master map

**IDENTITY.** Second homework. Warm-up: [strict convexity](./PREREQUISITES.md#p3-convex), [almost surely](./PREREQUISITES.md#p8-as).

### Board / screenshot

![Strict convexity at 1; R constant Q-a.s.; then c=1 so p=q](./screenshots/composites/ch04-topic-04-zero-iff-equal-panel1of1.png)

**Figure — ~10:43–15:09:** if $f$ is strictly convex at $u=1$, Jensen equality holds only when $R$ is constant $Q$-a.s. Combined with $\mathbb{E}[R]=1$, that constant is $1$, hence $p=q$ a.e.

### What he is establishing

Ask when $D_f=0$. Jensen has a $\ge$. Equality, for $f$ **strictly convex at $1$**, holds only when the random variable $R$ is **constant $Q$-almost surely**. “Almost surely” is not a weasel word: $Q$ is allowed to ignore a set of $q$-mass zero. That is the only equality Jensen will give you.

If $R\equiv c$ $Q$-a.s., then $\mathbb{E}_Q[R]=c$. But we already computed $\mathbb{E}_Q[R]=1$. So $c=1$, hence $p/q=1$ $Q$-a.s., hence $p=q$ almost everywhere.

Therefore $D_f(P,Q)=0$ **only when** $P=Q$ (under that strict-convexity extra). Together with non-negativity, these are the two standing conditions Lec 03 wanted.

If $f$ is merely convex, not strictly, equality can be cheaper. TV’s $f(u)=\tfrac12|u-1|$ is a V: a whole segment of the chord *is* the graph. Jensen can be equal without $R$ being a single number. That is why TV needs its own “$=0$ iff” argument later, as homework.

A concrete picture: $R$ jumps between $0.5$ and $1.5$ on two halves of the $q$-mass. Average ratio is still $1$, so $f(\mathbb{E}[R])=f(1)=0$, but $\mathbb{E}[f(R)]$ sits strictly above $0$ for a curved $f$. The hills “look balanced on average” and still score positive. The trap is “$D_f=0$ whenever the hills look similar.” Equality only when $R\equiv 1$ $Q$-a.s.

You can now close both homework properties for every *strictly* convex $f$. Examples by changing $f$ start next.

### Analogy for this topic only

The spring reads zero city-wide only if it reads zero on **almost every street**.

**Can two different maps still score zero?** Only if they disagree on streets $q$ never visits. A strictly curved spring reads zero only at $R=1$. So the maps must agree wherever $q$ bothers to go.

In lecture words: $R$ constant a.s. ⇒ $R=1$ ⇒ $p=q$.

### Local picture

```
  want D_f = 0
  ⇒ E[f(R)] = f(E[R])     (Jensen equality)
  ⇒ R = c   Q-a.s.        (strict convexity at 1)
  ⇒ c = E[R] = 1
  ⇒ p = q   a.e.
```

Notice: “almost surely” is not a weasel word. It is the only equality Jensen will give you.

### Bridge

The ruler works. Now swap springs and watch KL, reverse KL, TV, and JSD fall out.

---

## Topic 5: Child: KL and MLE (15:41–18:59)

### Where this sits on the master map

**KL.** First named child. Warm-up: [ratio](./PREREQUISITES.md#p2-ratio).

### Board / screenshot

![f(u)=u log u plugs into KL = ∫ p log(p/q)](./screenshots/composites/ch05-topic-05-kl-mle-panel1of1.png)

**Figure — ~16:02–18:40:** recall $D_f=\int q\,f(p/q)\,dx$. Choose $f(u)=u\log u$. After canceling $q$ you get $\int p\log(p/q)\,dx=D_{\mathrm{KL}}(P\|Q)$. Spoken: this is also min-KL = MLE.

### What he is establishing

Vary $f$ inside the same integral; you get a **class** of divergences. Same kitchen scale, new spring. He calls them “divergence metrics” in passing — later he will take the word *metric* back from KL.

First spring: $f(u)=u\log u$ ($u$ dummy). Plug in and watch $q$ cancel:

$$
\int q\cdot\frac{p}{q}\log\frac{p}{q}\,dx=\int p\log\frac{p}{q}\,dx=D_{\mathrm{KL}}(P\|Q).
$$

The remaining weight is **$p$**. Forward KL cares where the *first* law is large. That is why the two directions will later read different numbers on the same pair of coins.

He asks you to remember: **maximum likelihood = minimum KL**. That should have been the first proof in a first ML course. Lec 03 said the same. Minimizing $\mathrm{KL}(p_x\|p_\theta)$ is the same job as maximizing $\mathbb{E}_{p_x}[\log p_\theta]$ — the extra $\mathbb{E}_{p_x}[\log p_x]$ does not depend on $\theta$. Do not treat KL as a new animal. It is $f$-divergence with this one spring, and MLE is its minimizer. You can now recover KL from the master integral in two lines. Still missing: the spring that keeps the $q$-weight instead of canceling it.

### Analogy for this topic only

Same kitchen scale, first spring installed.

**What does the scale read if you only change the spring to $u\log u$?** Reciting “some new divergence” does not answer. The $q$ cancels and you are left with the familiar KL. Minimizing that reading *is* maximizing a likelihood. Inventing a second theory of MLE is the wrong move.

In lecture words: $u\log u$ → KL → MLE.

### Local picture

```
  f(u) = u log u
  q · (p/q) log(p/q)  =  p log(p/q)
  ∫ that  =  KL(P∥Q)
  argmin_θ KL(p_x ∥ p_θ)  =  MLE
```

Notice: $q$ cancels. The remaining weight is **$p$** — where the *first* law is large.

### Bridge

A different spring will put the weight on $q$ instead. That is reverse KL.

---

## Topic 6: Child: reverse KL (18:59–20:53)

### Where this sits on the master map

**RKL.** Second child. This TA writes **$f(u)=-\log u$** (Lec 03 left $\log u$ as an exercise — watch the sign).

### Board / screenshot

![f(u)=−log u → ∫ q log(q/p) = KL(Q∥P)](./screenshots/composites/ch06-topic-06-reverse-kl-panel1of1.png)

**Figure — ~19:10–20:41:** $f(u)=-\log u$. Integral becomes $\int q\log(q/p)\,dx=D_{\mathrm{KL}}(Q\|P)$, “famously reverse KL.”

### What he is establishing

Second spring: **minus** log. Use $\log(a/b)=\log a-\log b$ once:

$$
\int q\bigl(-\log(p/q)\bigr)\,dx=\int q\log\frac{q}{p}\,dx=D_{\mathrm{KL}}(Q\|P).
$$

That is reverse KL. The weight stayed on **$q$**. Forward KL asked “how surprised is $Q$ about what $P$ actually does.” Reverse asks “how much mass does $P$ put where $Q$ lives.” Same two towns, opposite commute.

The choice of $f$ *is* the choice of divergence. Lec 03 assigned $f=\log u$ as homework; the algebra that matches this child is $-\log u$. This tutorial writes the minus. Using $+\log u$ without checking the sign is the trap — you would get the *negative* of reverse KL, which is not a legal $f$-divergence reading ($f$ would fail $f(1)=0$ in the convex-cup sense he needs, and the score would go the wrong way).

Micro: the same two coins $P=(0.9,0.1)$ and $Q=(0.5,0.5)$ will later score $0.368$ one way and $0.511$ the other. Those two numbers are this spring and the last spring, not a rounding error.

You can now install $-\log u$ and get reverse KL. Still missing: the spring that is allowed to be called a distance.

### Analogy for this topic only

Turn the spring over.

**Does the scale still care about the same streets?** No. KL weighted the streets where **$p$ is tall**. This spring weights streets where **$q$ is tall**. Same two towns, opposite commute. Reciting “just swap the names on KL” hides that you installed a *new* $f$.

In lecture words: $-\log u$ → reverse KL.

### Local picture

```
  f(u) = −log u
  q · (−log(p/q)) = q log(q/p)
  ∫ that = KL(Q∥P)   “reverse KL”
```

Notice: not a swap of labels on the same $f$. A **new** $f$.

### Bridge

A third spring uses an absolute value and is allowed to be called a **distance** — after we check axioms.

---

## Topic 7: Child: total variation (20:53–22:57)

### Where this sits on the master map

**TV.** Third child. Warm-up: [metric](./PREREQUISITES.md#p6-metric).

### Board / screenshot

![f(u)=½|u−1| → ½∫|p−q| = TV](./screenshots/composites/ch07-topic-07-total-variation-panel1of1.png)

**Figure — ~21:07–22:42:** $f(u)=\tfrac12|u-1|$. After $q$ cancels inside the absolute value you get $\tfrac12\int|p-q|\,dx$. He says **distance** and promises a justification.

### What he is establishing

$$
f(u)=\frac12|u-1|\qquad\Rightarrow\qquad D_f=\frac12\int|p-q|\,dx.
$$

Watch the algebra, because he briefly writes the wrong denominator. Start from $q\cdot\bigl|(p/q)-1\bigr|$. Factor $1/q$ inside the absolute value:

$$
q\cdot\Bigl|\frac{p-q}{q}\Bigr|=|p-q|.
$$

The $q$ cancels. What remains is half the $L^1$ gap between the two heights. He first left a stray $q$ in a denominator, then fixed it. Trust the last line on the tablet.

He **uses the word distance** for TV and will justify it after the KL takedown. Two hints already: $|p-q|=|q-p|$ (symmetry is free), and the V is convex so $D_f\ge 0$ is free. The V is *not* strictly convex, so the “$=0$ iff $p=q$” ticket is *not* free — that is why TV is homework. Calling TV a distance *before* those four checks is the slogan; the work is Topic 10.

Micro: two uniforms, height $2$ on $[0,0.5]$ and height $2$ on $[0.5,1]$, have $L^1$ gap $2$ and TV $1$ — as far as two laws can get. Same pair swapped is still TV $1$. KL on that pair is infinite in both directions (disjoint support). Different springs, different readings.

You can now recover TV from $f=\tfrac12|u-1|$. Still missing: the combination spring Lec 03 called JSD.

### Analogy for this topic only

Hold a ruler between two hills and add up the **vertical gap**.

**Is that gap the same if you swap the hills?** Yes — $|p-q|=|q-p|$. Half of the total gap is TV. Unlike KL, this spring cannot tell which hill you called $p$. That is why he dares the word “distance.”

In lecture words: $\tfrac12|u-1|$ → TV.

### Local picture

```
  f(u) = ½ |u − 1|
  q · |(p/q) − 1| = |p − q|
  D_f = ½ ∫ |p − q| dx   =  TV(P,Q)
```

Notice: the V-shape is convex but not strictly convex. The “=0 iff $p=q$” proof for TV is not the Jensen-strictness argument.

### Bridge

The spring Lec 03 called “an average of both KLs” needs a longer algebra — JSD.

---

## Topic 8: Child: JSD and the GAN zoo (22:57–30:10)

### Where this sits on the master map

**JSD.** The combination spring. Complements Lec 03’s two-sins story.

### Board / screenshot

![JSD f; m=(p+q)/2; ½KL(P,M)+½KL(Q,M); alphabet GANs](./screenshots/composites/ch08-topic-08-jsd-gan-zoo-panel1of1.png)

**Figure — ~23:35–29:31:** $f(u)=\tfrac12 u\log(2u/(u+1))+\tfrac12\log(2/(u+1))$. After $q$ goes inside: $\tfrac12\mathrm{KL}(P\|M)+\tfrac12\mathrm{KL}(Q\|M)$ with $m=(p+q)/2$. Spoken: A-GAN / B-GAN — change $f$, change the GAN.

### What he is establishing

The JSD spring (written here as two half-logs)

$$
f(u)=\frac12 u\log\frac{2u}{u+1}+\frac12\log\frac{2}{u+1}
$$

expands, after substituting $u=p/q$ and canceling $q$ in each term, to

$$
D_{\mathrm{JS}}(P,Q)=\frac12 D_{\mathrm{KL}}(P\|M)+\frac12 D_{\mathrm{KL}}(Q\|M),\qquad M=\frac{P+Q}{2}.
$$

The first half-log becomes $p\log\bigl(p/m\bigr)$ with $m=(p+q)/2$. The second becomes $q\log\bigl(q/m\bigr)$. Those are exactly the two KL integrals against the midpoint. He circles $m$ on the tablet so you see the mixture is not a new $f$ — it is the *argument* of the two KLs.

That is the combination Lec 03 wanted: forward KL cares where $P$ is large; reverse cares where $Q$ puts junk; JSD asks both, via the midpoint $M$. JSD is **symmetric** in $P$ and $Q$ (swap the names, $M$ stays put). That is one reason people like it as a “distance-ish” score — though this hour does not check the four tickets on JSD.

The lecture joke: A-GAN, B-GAN, every alphabet. The serious point: **change $f$, get a different divergence, get a different GAN class**. What a GAN *is* waits for a later week. No code this hour.

You can now expand the JSD spring to the midpoint formula. Still missing: the advertised takedown — KL fails two of the four metric axioms, with numbers.

### Analogy for this topic only

Two critics, one who fines missed hits and one who fines unrequested songs, average their scores after both listen to a **mix tape** $M$ of the album and the cover.

**Why not just pick one critic?** Lec 03 already showed each critic is blind in one eye. The mix tape is the compromise. Changing the spring (A-GAN, B-GAN) is the same idea with a different $f$.

In lecture words: $M=(P+Q)/2$, JSD = mean of two KLs to $M$.

### Local picture

```
  f(u) = ½ u log(2u/(u+1)) + ½ log(2/(u+1))
                 │  plug u=p/q, multiply by q
                 ▼
  ½ ∫ p log(p / m) + ½ ∫ q log(q / m)
  m = (p+q)/2
                 │
                 ▼
  ½ KL(P∥M) + ½ KL(Q∥M)  =  JSD

  change f  →  new D  →  new letter-GAN
```

Notice: JSD is **symmetric** in $P$ and $Q$. That is one reason people like it as a “distance-ish” score.

### Bridge

We have four children. Now the advertised takedown: KL fails two of the four metric axioms, with numbers.

---

## Topic 9: Four axioms; symmetry fails (30:10–41:23)

### Where this sits on the master map

**NOT METRIC.** Leftover slogan from Lec 03. Warm-ups: [four tickets](./PREREQUISITES.md#p6-metric), [Bernoulli](./PREREQUISITES.md#p7-bern).

### Board / screenshot

![Why D_KL is not a metric: 4 properties; non-negativity holds; symmetry heading](./screenshots/composites/ch09-topic-09-metric-axioms-symmetry-panel1of2.png)

**Figure — ~31:07–35:05:** “Why is $D_{\mathrm{KL}}$ not a distance metric.” A distance must satisfy **4** properties. (1) Non-negativity — KL has it. (3) Symmetry heading appears at the end of this panel.

![Bernoulli 0.368 vs 0.511; surprise slogans](./screenshots/composites/ch09-topic-09-metric-axioms-symmetry-panel2of2.png)

**Figure — ~36:24–40:22:** $P=(0.9,0.1)$, $Q=(0.5,0.5)$. $D_{\mathrm{KL}}(P\|Q)=0.9\log(0.9/0.5)+0.1\log(0.1/0.5)\approx 0.368$, $D_{\mathrm{KL}}(Q\|P)\approx 0.511$. Then $D_{\mathrm{KL}}(P\|Q)=\mathbb{E}_P[\log(P/Q)]$: “regions where $P$ is large matter a lot; regions where $P$ is small hardly matter.” Green slogans: forward = how surprised is the model about samples that actually occur; reverse = how much probability the model puts where the data is not.

### What he is establishing

Any metric $d(P,Q)$ needs **four** tickets. KL already has two from the $f$-divergence theorems:

1. **Non-negativity.** $D_{\mathrm{KL}}\ge 0$.
2. **Identity of indiscernibles.** $D_{\mathrm{KL}}=0$ iff $P=Q$.

3. **Symmetry.** $d(P,Q)=d(Q,P)$. KL **fails**. Forward KL and reverse KL are different children.

Bernoulli check. Let $P=(0.9,0.1)$ and $Q=(0.5,0.5)$:

$$
D_{\mathrm{KL}}(P\|Q)=0.9\log\frac{0.9}{0.5}+0.1\log\frac{0.1}{0.5}\approx 0.368,
$$
$$
D_{\mathrm{KL}}(Q\|P)\approx 0.511.
$$

Not equal. That is enough to kill “KL is a metric.”

Interpretation (abusing notation $P$ for both law and height): $D_{\mathrm{KL}}(P\|Q)=\mathbb{E}_P[\log(P/Q)]$ **cares where $P$ is large** — how surprised is the model about what actually occurs. Regions where $P$ is small hardly matter. Reverse KL asks how much mass the model puts **where the data is not**. If you want to **sample**, you want both. That is why JSD exists. Symmetry failing is not a bookkeeping accident; the two directions *mean* different jobs.

Calling $0.368$ and $0.511$ “the same distance, rounding error” is the trap. They are different jobs. You can now fail KL on ticket (3) with a two-line calculator. Ticket (4) still stands.

### Analogy for this topic only

Two coins, 90/10 vs 50/50. **Is “how weird is the fair coin to the 90/10 person” the same number as the reverse?** Compute it: $0.368$ vs $0.511$. A ruler cannot read two different lengths for the same pair of points.

In lecture words: $P=(0.9,0.1)$, $Q=(0.5,0.5)$.

### Local picture

```
  METRIC tickets          KL
  (1) ≥ 0                 yes
  (2) =0 iff same         yes
  (3) symmetric           NO
  (4) triangle            (next)

  P = (0.9, 0.1)     Q = (0.5, 0.5)
  KL(P∥Q) ≈ 0.368
  KL(Q∥P) ≈ 0.511
```

Notice: he first mumbled $P$ as $(0.1,\ldots)$ then wrote the $0.9\log(0.9/0.5)$ line. The **formula** is the source of truth.

### Bridge

Three tickets checked. The fourth — triangle — gets its own Bernoulli triple.

---

## Topic 10: Triangle fails; TV homework (41:23–48:09)

### Where this sits on the master map

**TRIANGLE / STOP.** Last axiom, last exercise. Warm-up: [metric](./PREREQUISITES.md#p6-metric).

### Board / screenshot

![d(P,R)≤d(P,Q)+d(Q,R); numbers 1.758 > 0.368+0.511](./screenshots/composites/ch10-topic-10-triangle-tv-homework-panel1of1.png)

**Figure — ~41:58–47:24:** (4) Triangle: $d(P,R)\le d(P,Q)+d(Q,R)$. “Going directly from P to R can never be longer than going through Q.” $p=(0.1,0.9)$, $q=(0.5,0.5)$, $r=(0.9,0.1)$. $D_{\mathrm{KL}}(P,R)\approx 1.758$, $D_{\mathrm{KL}}(P,Q)\approx 0.368$, $D_{\mathrm{KL}}(Q,R)\approx 0.511$, and $1.758\ge$ the sum.

### What he is establishing

**Triangle inequality:** the direct road $P\to R$ is never longer than $P\to Q\to R$.

Counter-example, three Bernoullis:

$$
p=(0.1,0.9),\quad q=(0.5,0.5),\quad r=(0.9,0.1).
$$

$$
D_{\mathrm{KL}}(p,r)\approx 1.758,\quad D_{\mathrm{KL}}(p,q)\approx 0.368,\quad D_{\mathrm{KL}}(q,r)\approx 0.511.
$$

Sum of the two legs $\approx 0.879$. Direct $1.758>0.879$. Triangle **fails**. (He first said $0.879$ for the second leg, then corrected to $0.511$; the **sum** is $0.879$.)

So KL has no honest “straight line shorter than a detour” picture. That is the second reason it is not a metric.

**Exercise:** is **total variation** a valid metric? The name already hints. He wants you to walk all four axioms and convince yourself **yes**. Treating the name as a proof is the wrong move; the four tickets are the proof.

Recap of the hour: proved $f$-div properties; by changing $f$ got KL, reverse KL, TV, JSD; showed KL is not a distance. Complementary to the lecture. No code, no GAN construction yet.

You can now kill KL as a metric with two numerical counter-examples. Still missing: the TV write-up (your homework) and the algorithms that *use* these $D$s.

### Analogy for this topic only

Three towns on a map: a 10/90 coin, a fair coin, a 90/10 coin.

**Is the highway from 10/90 to 90/10 allowed to be longer than going via fair?** A real mileage table says no. KL’s table says $1.758>0.879$. Throw that table out as a map.

In lecture words: $p=(0.1,0.9)$, $r=(0.9,0.1)$, fair coin in the middle.

### Local picture

```
        Q (0.5, 0.5)
       / \
 0.368/   \0.511
     /     \
    P       R
 (0.1,0.9) (0.9,0.1)
        \   /
         \ /
        1.758

  1.758  >  0.368 + 0.511 = 0.879
  triangle FAILS

  HW: check TV on all four tickets
```

Notice: $p$ and $r$ are swaps of each other; the fair coin sits in the middle. KL still thinks the direct swap is *longer* than going via fair. A metric cannot do that.

### Bridge

Proofs and numbers are on the desk. Next weeks: the algorithms that *use* these $D$s (and what a GAN actually is).

---

## External references

Two layers, **both kept**.

1. **Start here** — the newer high-signal companions (famous teachers, mapped to this lecture’s hard boxes).
2. **Full topic map** — the previous per-topic list (2–3 companions each) **plus** any new entries already woven above. Use a group when one box still feels thin.

### Start here — high-signal companions

Only a few **widely used** companions — the ones people actually finish. Not a pile of random blogs. Use them after the matching topic, with this tutorial still closed. Chalkboard only — no Colab.

**If the parent lecture is still foggy (Topics 1, 8).** Start from this course’s [Lec 03 — f-Divergence](../25-Lec03-f-Divergence-Examples/NOTES.md). Same integral, same children; this hour is the algebra. Lilian Weng’s [From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) is the field’s usual written menu of those scores.

**If $P\ll Q$ is still “assume $P=Q$” (Topic 2).** Cofiber’s [Radon–Nikodym derivative](https://www.youtube.com/watch?v=sPcXyZB1bkM) is the standard visual for “a density $p/q$ exists only after absolute continuity.” The red line licenses the *division*; identity is a later theorem.

**If Jensen or “$=0$ only when $p=q$” will not close (Topics 3–4).** Taboga’s [Statlect — Jensen](https://www.statlect.com/fundamentals-of-probability/Jensen-inequality) is the written three-line form on the tablet. Francis Bach’s [Revisiting Jensen](https://francisbach.com/jensen-inequality/) is the equality case: strictly convex ⇒ constant almost surely.

**If min-KL still does not sound like MLE (Topic 5).** Josh Starmer’s [StatQuest — Maximum Likelihood, clearly explained](https://www.youtube.com/watch?v=XepXtl9YKwc) is the popular version of the “first ML proof” he says you should already own.

**If the two directions will not stay apart (Topics 6, 9).** Roger Grosse’s [CSC321 KL notes](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/readings/L05%20KL%20Divergence.pdf) and Hiroaki Hayashi’s [forward vs reverse KL](https://hiroakih.me/kl-divergence.html) sliders are the two pages people actually finish. Drag $q$ until $0.368$ and $0.511$ stop looking like a rounding error.

**If TV is only a V-shaped $f$ (Topic 7).** Todd Kemp’s [Total variation](https://www.youtube.com/watch?v=2Lpg7AITvnU) is the classroom video for $\mathrm{TV}=\tfrac12\int|p-q|$ — the identity after $q$ cancels.

**If JSD is only a name (Topic 8).** [NannyML — Jensen–Shannon distance](https://www.youtube.com/watch?v=YBjfT9hIUus) is the well-known video for “average of two KLs to the midpoint.”

**If the four tickets or the triangle still blur (Topics 9–10).** Dr. Bevin Maultsby’s [Metric spaces](https://www.youtube.com/watch?v=oz-LycQIL8g) is the same four axioms he lists. Brown’s [Seeing Theory — probability distributions](https://seeing-theory.brown.edu/probability-distributions/index.html) lets you drag two-point masses before you recompute $0.368$, $0.511$, and $1.758$ by hand.

**How to use.** License $P\ll Q$ is taught in these notes (Topic 2) — do not replace it with “assume $P=Q$.” Two sins → Grosse or Hiroaki *after* Topic 6. Numbers → Seeing Theory, then redo the three KLs yourself before the TV homework. One famous teacher per stuck idea.

---

### Full topic map — previous list plus new entries

Two or three companions **per topic**, listed **only here** (not under each topic). Mix of **video** and **blog/notes**. No Wikipedia (he mentioned wiki for Jensen; we point to teaching pages instead).

| Resource | Type | Matches lecture… | Why it helps |
|----------|------|------------------|--------------|
| [Lec 03 — f-Divergence](../25-Lec03-f-Divergence-Examples/NOTES.md) | notes | Topic 1 · DEF | The lecture this tutorial proves. Same $D_f=\int q\,f(p/q)$. |
| [Nowozin — f-GAN talk](https://www.youtube.com/watch?v=bJNQkPldWZg) | video | Topic 1 · family | Same integral, every $f$ is a different child. |
| [Lilian Weng — From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) | blog | Topic 1 · $D$ as a score | Written menu of divergences before the proofs. |
| [Cofiber — Radon–Nikodym derivative](https://www.youtube.com/watch?v=sPcXyZB1bkM) | video | Topic 2 · $P\ll Q$ | Why a density $p/q$ exists only after absolute continuity. |
| [StatQuest — Probability is not Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4) | video | Topic 2 · ratio of heights | $R=p/q$ as two heights at one $x$, not two areas. |
| [Bright Side of Mathematics — Radon–Nikodym](https://www.youtube.com/watch?v=JalFzLvYQY0) | video | Topic 2 · license to divide | $P\ll Q$ vs “the two laws are equal.” |
| [ritvikmath — Jensen’s inequality](https://www.youtube.com/watch?v=LOwj7UxQwJ0) | video | Topic 3 · convex cup | Chord-above-graph picture he refused to reprove. |
| [Statlect — Jensen’s inequality](https://www.statlect.com/fundamentals-of-probability/Jensen-inequality) | notes | Topic 3 · $\mathbb{E}[f]\ge f(\mathbb{E})$ | Written three-line form matching the tablet. |
| [Tutorial 8 — Jensen stated](../22-Tutorial08-Review-Basic-Probability-2/NOTES.md) | notes | Topic 3 · series link | Where this course first wrote Jensen (no proof). |
| [Francis Bach — Revisiting Jensen](https://francisbach.com/jensen-inequality/) | blog | Topic 4 · equality case | Strictly convex ⇒ equality iff $X$ is a.s. constant. |
| [Djalil Chafaï — About Jensen](https://djalil.chafai.net/blog/2013/11/17/about-the-jensen-inequality/) | blog | Topic 4 · strict convexity | Equality cases when the cup is not a V. |
| [Adian Liusie — Intuitively understanding KL](https://www.youtube.com/watch?v=SxGYPqCgJWM) | video | Topic 5 · $u\log u$ | Forward KL as extra surprise. |
| [ritvikmath — The KL divergence](https://www.youtube.com/watch?v=q0AkK8aYbLY) | video | Topic 5 · algebra | Second pass on $\int p\log(p/q)$. |
| [Jake Tae — MLE and KL](https://jaketae.github.io/study/kl-mle/) | blog | Topic 5 · MLE = min KL | The “first ML proof” written out. |
| [StatQuest — MLE, clearly explained](https://www.youtube.com/watch?v=XepXtl9YKwc) | video | Topic 5 · MLE | The maximizer he says *is* min-KL. |
| [Hiroaki Hayashi — Forward vs reverse KL](https://hiroakih.me/kl-divergence.html) | demo | Topic 6 · two commutes | Drag $q$; watch $\mathrm{KL}(p\|q)$ and $\mathrm{KL}(q\|p)$ disagree. |
| [Tuan Anh Le — Reverse vs forward KL](https://www.tuananhle.co.uk/notes/reverse-forward-kl.html) | notes | Topic 6 · $-\log u$ | Mode-seek vs mass-cover after the sign check. |
| [Grosse — KL reading (CSC321)](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/readings/L05%20KL%20Divergence.pdf) | notes | Topic 6 · two directions | Same height-story in a short PDF. |
| [Todd Kemp — Total variation](https://www.youtube.com/watch?v=2Lpg7AITvnU) | video | Topic 7 · half the $L^1$ gap | TV as $\frac12\int|p-q|$, the identity he wrote. |
| [Djalil Chafaï — Total variation distance](https://djalil.chafai.net/blog/2010/10/14/back-to-basics-total-variation-distance/) | blog | Topic 7 · TV axioms | Discrete TV properties before the homework. |
| [Rohit Bandaru — ML divergences](https://rohitbandaru.github.io/notes/ml-divergences/) | notes | Topic 7 · TV in the menu | TV next to KL / JSD, not a new animal. |
| [NannyML — Jensen–Shannon distance](https://www.youtube.com/watch?v=YBjfT9hIUus) | video | Topic 8 · JSD | Average-of-two-KLs picture. |
| [Josh Lospinoso — Jensen–Shannon](https://lospino.so/statistics/jensen-shannon-divergence/) | notes | Topic 8 · $M=(P+Q)/2$ | Midpoint formula + a two-bin worked number. |
| [Nowozin et al. — f-GAN](https://arxiv.org/abs/1606.00709) | paper | Topic 8 · change $f$ | Alphabet-GAN unification he joked about. |
| [Dr. Bevin Maultsby — Metric spaces](https://www.youtube.com/watch?v=oz-LycQIL8g) | video | Topic 9 · four tickets | Non-negativity, identity, symmetry, triangle — same list. |
| [Seeing Theory — probability distributions](https://seeing-theory.brown.edu/probability-distributions/index.html) | demo | Topic 9 · Bernoulli piles | Two-point masses you can drag before computing $0.368$. |
| [Huszár — How to train generative models](https://www.inference.vc/how-to-train-your-generative-models-why-generative-adversarial-networks-work-so-well-2/) | blog | Topic 9 · two sins | Why sampling wants both directions, not one KL. |
| [Oxford Mathematics — Metric spaces](https://www.youtube.com/watch?v=yvaFeNLZ9s8) | video | Topic 10 · triangle | “Direct road never longer than a detour.” |
| [Dibya Ghosh — KL for ML](https://dibyaghosh.com/blog/probability/kldivergence/) | blog | Topic 10 · numbers | Extra diagrams for the Bernoulli takedown. |
| [Lec 03 Topic 10 — two sins](../25-Lec03-f-Divergence-Examples/NOTES.md#topic-10-mode-cover-vs-junk-variational-next-3618–4230) | notes | Topic 10 · recap | Same story, lecture voice, then TV homework. |

**How to use:** Topics 2–4 with Cofiber + Tutorial 8 + Francis Bach (license, Jensen, equality). After Topic 5, Jake Tae or StatQuest MLE. After Topic 6, Hiroaki’s sliders. After Topic 8, NannyML or Lospinoso. After Topic 10, recompute the three Bernoulli KLs yourself before the TV homework. No Colab — the tablet has no code.

---


## Sources

- Video: [Tutorial 11 – f-Divergence and Examples](https://www.youtube.com/watch?v=GjxuVZeMSfE) · NPTEL IISc
- Complements Prof. Pratush, Lec 03
- Captions: `raw/captions.en.timed.txt`
- Boards: `screenshots/composites/`
- **Code audit:** despite the prompt, **no code / Colab / Python appears**. These notes add **no invented code**. Math in `$` / `$$` only. Numerical KL values are chalkboard arithmetic, not a program.
