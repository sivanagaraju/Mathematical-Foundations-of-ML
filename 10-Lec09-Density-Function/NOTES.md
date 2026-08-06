# Lec 09 — Density Function

**Video:** [Lec 09 Density Function](https://www.youtube.com/watch?v=_QrezNPmxDk) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 08 — Distribution Estimation](../09-Lec08-Distribution-Estimation/NOTES.md)

---

## Table of Contents

1. [Topic 1 — Density notation and CDF integral definition](#topic-1-density-notation-and-cdf-integral-definition-0005–0129) (00:05–01:29)
2. [Topic 2 — Trap: p(x) is not a probability](#topic-2-trap-px-is-not-a-probability-0129–0258) (01:29–02:58)
3. [Topic 3 — Uniform[0,½] density=2; likelihood](#topic-3-uniform01½-density2-likelihood-0258–0400) (02:58–04:00)
4. [Topic 4 — Why densities; continuous vs discrete PMF](#topic-4-why-densities-continuous-vs-discrete-pmf-0400–0526) (04:00–05:26)
5. [Topic 5 — Estimate density; P vs p; multi-d; next](#topic-5-estimate-density-p-vs-p-multi-d-next-0526–0805) (05:26–08:05)
6. [External references](#external-references)
7. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: install the **density function** as the working object for continuous ML. Method: define $p$ via the CDF integral, kill the trap that $p(x)$ is a probability, contrast with discrete PMF, then restate estimation as **density** estimation. Fork: heights can exceed 1; only **integrals** of $p$ are probabilities.

**Worldview arc:** from “estimate distribution $P$” **to** “estimate density $p$ (and conditionals $p(y\mid x)$) for the rest of the course.”

### System context

```
  ╔══════════════════════════════════╗
  ║ Outside                          ║
  ║ · Lec 08: given D, estimate P    ║
  ║ · challenges of ML (next)        ║
  ╚══════════════╤═══════════════════╝
                 │ this lecture
                 ▼
        ┌────────────────────┐
        │ density p          │
        │ height ≠ prob      │
        └────────────────────┘
```

### Main blueprint

```
  Continuous RV X
       │
       ▼
  p_X : range → R_+     (non-negative; can be >1)
       │
       ▼
  P_X(x) = ∫_{-∞}^{x} p_X(t) dt
       │
       ├─ p(x0)  →  height / "likelihood"  ≠ probability
       └─ ∫_A p  →  probability of region A
       │
       ▼
  Discrete twin: PMF  (mass at point = probability)
       │
       ▼
  Course: given D, estimate p  (not only P)
  Notation: one-line p = density; two-line P = distribution
  Next: challenges of ML density estimation
```

### Scenario walkthrough

1. Continuous data (most of our images/features).  
2. Introduce $p_X$ with $P_X(x)=\int_{-\infty}^{x}p$.  
3. Reject “$p(x)$ is probability.”  
4. See Uniform$[0,1/2]$ with height 2.  
5. Call height **likelihood**; keep probability for integrals.  
6. Switch problem statement to density estimation.  
7. Next: why estimating $p$ is hard.

### Failure / contrast path

```
  Treat Gaussian height as probability     ──X──► interview trap
  Ignore Uniform[0,½] height=2             ──X──► miss >1 counterexample
  Demand density for every abstract RV     ──X──► overclaim
  Forget multi-d integral is volume        ──X──► wrong dimension
```

### STOP / out of scope

Full theory of when densities exist (Radon–Nikodym), MLE theory, algorithm list, “challenges of ML” (next lecture).

### Load-bearing claims (closed-book)

- $p_X$ maps range → $\mathbb{R}_+$; not restricted to $[0,1]$ (rate / height, not a fraction).  
- $P_X(x)=\int_{-\infty}^{x}p_X$; locally $P([x,x+dx])\approx p(x)\,dx$.  
- $p(x)$ at a point ≠ probability; $\int_A p$ is; $P(X=x_0)=0$ even if $p(x_0)>0$.  
- Uniform$[0,1/2]$ has $p\equiv 2$ on support (height $=1/L$).  
- Density-at-point = likelihood (name).  
- Discrete PMF: point mass is probability.  
- Course estimates densities going forward; one-line $p$ vs two-line $P$; $p(y\mid x)=p(x,y)/p(x)$.

**Speaker / course:** NPTEL IISc · MFML · Lec 09.

---

## Topic 1: Density notation and CDF integral definition (00:05–01:29)

### Where this sits on the master map

**DEFINE $p$** box — new object for continuous RVs. Warm-up: [CDF](./PREREQUISITES.md#p1-continuous-cdf), [density def](./PREREQUISITES.md#p2-density-def).

### Board / screenshot

![Density definition](./screenshots/composites/ch01-topic-01-define-density-panel1of1.png)

**Figure — ~00:05–01:25:** small $p$; range → $\mathbb{R}_+$; CDF as running integral of density.

### What he is establishing

Given a **continuous** random variable with distribution function $P_X$, the **density function** is written with a **small** $p$:

$$
p_X(x)
$$

(The $x$ marks the underlying RV.) Notation choice: small $p$ for density.

**What kind of function is it?** A map from the **range** of the random variable into the **non-negative reals** $\mathbb{R}_+$:

$$
p_X:\text{range}(X)\to\mathbb{R}_+
$$

Density functions are simply **non-negative** functions. They are **not** bounded between 0 and 1 — he will stress properties next.

**Definition link to the distribution / CDF:**

$$
P_X(x)=\int_{-\infty}^{x} p_X(t)\,dt
$$

| Piece | Meaning |
|-------|---------|
| Left side $P_X(x)$ | CDF value = $P(X\le x)\in[0,1]$ |
| Right side | area under $p_X$ from $-\infty$ to $x$ |
| Equality | **definition** of density (when it exists) |

A valid density on the whole line also satisfies total mass one:

$$
\int_{-\infty}^{\infty} p_X(t)\,dt=1
$$

**What the height means operationally** (thin strip): over a tiny width $dx$,

$$
P\big(x\le X\le x+dx\big)\;\approx\; p_X(x)\,dx
$$

So $p_X(x)$ is **probability per unit of $x$** — a rate — and only the product (area) is a probability. That is why the codomain is $\mathbb{R}_+$, not $[0,1]$: rates are not fractions of a whole.

**Micro (already with Uniform$[0,1]$ height 1):**  
$P_X(0.3)=\int_0^{0.3}1\,dt=0.3$ — the CDF grows as accumulated area.  
Same idea locally: on $[0.3,0.3+0.01]$, probability $\approx 1\cdot 0.01=0.01$.

If you only remember “a curve called density” without the integral, you cannot connect to probability. Instead: density is the object whose running integral recovers $P_X$; heights alone are rates.

We now have the definition. Still missing: the trap of reading $p(x)$ as a probability.

### Analogy for this topic only

A water tank fills with a flow rate that changes over time.

- Minute 1 rate high; minute 2 rate low  
- Height of the rate curve at minute 3 is **not** “liters of water.”  
- **Area** under the rate from minute 0 to 3 **is** liters added.  

**What recovers “how full is the tank so far”?** The running integral of rate — same role as $P_X(x)=\int_{-\infty}^{x}p$.

In lecture words: $P_X(x)=\int_{-\infty}^{x}p$; density = non-negative function on the range.

### Local picture

```
  X continuous RV
       │
       ▼
  p_X(x) ≥ 0     (height; can be >1)
       │  ∫_{-∞}^{∞} p = 1
       ▼
  P_X(x) = ∫_{-∞}^{x} p_X(t) dt   ← definition link
```

**Notice:** codomain is $\mathbb{R}_+$, not $[0,1]$.

### Bridge

What goes wrong if someone treats the height $p(x)$ itself as a probability?

---

## Topic 2: Trap — $p(x)$ is not a probability (01:29–02:58)

### Where this sits on the master map

**TRAP ≠P** — critical kill shot. Warm-up: [height ≠ prob](./PREREQUISITES.md#p3-height-not-prob).

### Board / screenshot

![Density at a point is not probability](./screenshots/composites/ch02-topic-02-trap-not-prob-panel1of1.png)

**Figure — ~01:29–02:55:** evaluate $p$ at a point; integrate over range for probability; pushforward.

### What he is establishing

Evaluate the density at a particular point. You get a **positive / non-negative number**. That does **not** correspond to probability. Trivial mistake many people make: treating density evaluation as a valid probability measure.

Student pushback in class: “Come again — can you evaluate?” Yes — density is a function; you **can** plug in $x$ (e.g. a **Gaussian** density at a point). You may get a nonzero height — “it will not be zero.” Two facts sit next to each other:

1. In a continuous model, $P(X=x_0)=0$ for an exact singleton (probability of a point).  
2. Even so, $p(x_0)$ can be a large positive height — and that height is **still not** a probability.

These are consistent: the singleton $\{x_0\}$ has **zero width**, so

$$
P(X=x_0)=\int_{\{x_0\}} p(t)\,dt=0
$$

even when $p(x_0)$ is huge. Fact 1 is about integrating over zero width; fact 2 is about not renaming the height a probability.

His sharper operational point: **the height is not “the probability.”**

What **is** probability? **Integrate** the density over a range:

$$
P(a\le X\le b)=\int_a^b p(t)\,dt
$$

Integrating over a small range corresponds to a chunk of the distribution function, and the distribution is a valid probability measure — the **pushforward** of the original $P$ on $\Omega$ through the map $X$. Evaluating density at a single point is **not** that measure.

| Operation | Valid probability? |
|-----------|--------------------|
| $p(x_0)$ | **No** (height / later called likelihood) |
| $\int_a^b p(t)\,dt$ | **Yes** |

If you report $p(x_0)$ as “the chance $X=x_0$,” you are wrong even when the number is between 0 and 1. Instead: integrate over regions.

We now have the trap. Still missing: a density that is **greater than 1** so the “looks like probability” excuse dies.

### Analogy for this topic only

A speedometer shows 90.

- Is that “90% of the trip done”? No.  
- Is speed a rate you can integrate over time to get distance? Yes.  

**Can you evaluate the speedometer at a moment?** Yes. **Is that reading a fraction of a whole?** No.

In lecture words: evaluate density freely; only integrals give probabilities.

### Local picture

```
  p(x0) = 0.4     ← height / number ≥0   ≠ probability

  ∫_{[a,b]} p     ← area                  = P(X∈[a,b])

  Distribution P_X = pushforward measure  = valid probabilities on sets
```

**Notice:** “probability of a continuous point is zero” and “height is not probability” are related but both worth keeping.

### Bridge

Is there a density that is obviously bigger than 1 so nobody can call the height a probability?

---

## Topic 3: Uniform$[0,1/2]$ density $=2$; likelihood (02:58–04:00)

### Where this sits on the master map

**EXAMPLE + LIKELIHOOD**. Warm-up: [uniform micro](./PREREQUISITES.md#p4-uniform-half), [likelihood](./PREREQUISITES.md#p5-likelihood).

### Board / screenshot

![Uniform density equals 2](./screenshots/composites/ch03-topic-03-uniform-likelihood-panel1of1.png)

**Figure — ~02:58–04:00:** Gaussian trap; Uniform$[0,1/2]$ density 2; likelihood name.

### What he is establishing

Interview trap: people look at a **Gaussian** density, see a height less than 1 (so it “looks bounded like a probability”), and interpret the height as probability. Counterexample: **Uniform** random variable on $[0,1/2]$.

**Why height equals 2:** uniform on an interval of length $L=1/2$ is a flat rectangle of **area 1**, so

$$
\text{height}=\frac{1}{L}=\frac{1}{1/2}=2
$$

$$
p(x)=\begin{cases} 2 & 0\le x\le 1/2 \\ 0 & \text{else} \end{cases}
$$

Same family: Uniform$[0,1]$ has height $1$; Uniform$[0,1/10]$ has height $10$. Shorter support → taller density — total mass $1$ squeezed into less width.

Check normalization: $\int_0^{1/2}2\,dx=1$. At every point in the **support** (where $p>0$), evaluation gives **2**, always **more than 1**. Probability measures are bounded in $[0,1]$. So you **cannot** interpret that evaluation as probability. Integrating still works:

$$
P\!\left(0\le X\le \tfrac14\right)=\int_0^{1/4}2\,dx=\tfrac12
$$

(half the support → half the probability, because the density is flat).

**Nomenclature:** the value of the density at a particular point is called a **likelihood**. He will use that name later. For now:

| Name | Object |
|------|--------|
| Likelihood (here) | $p(x)$ at a point (can be $>1$) |
| Probability | measure of a set, e.g. $\int_A p$ (always in $[0,1]$) |

If your only counterexample is “Gaussian looks like probs,” the uniform kills it. Instead: heights can exceed 1; areas give probabilities.

We now have the counterexample and the likelihood name. Still missing: why algorithms care, and discrete vs continuous.

### Analogy for this topic only

Paint a short tall rectangle on graph paper: width half a unit, height two units.

- Area of the paint blotch = 1 (a whole unit of mass).  
- Height of the blotch = 2 (taller than “one whole”).  

**Is height allowed to exceed one whole unit?** For densities, yes. For probabilities, no — so height cannot be probability.

In lecture words: Uniform$[0,1/2]$ density equals 2; evaluation = likelihood, not probability.

### Local picture

```
  Uniform[0, 1/2]:
       p
       2 |████████
         |████████
       0 +-------- 1/2 ----► x

  p(x)=2 on support  →  not a probability
  ∫_0^{1/2} p = 1    →  valid
  height p(x)        →  called likelihood
```

**Notice:** any density with a peak above 1 (many real densities) shows the same point.

### Bridge

Why switch the course to densities at all, and how do discrete RVs differ?

---

## Topic 4: Why densities; continuous vs discrete PMF (04:00–05:26)

### Where this sits on the master map

**WHY + PMF**. Warm-up: [PMF](./PREREQUISITES.md#p6-pmf), [estimate density](./PREREQUISITES.md#p8-estimate-density).

### Board / screenshot

![Why densities and PMF](./screenshots/composites/ch04-topic-04-why-pmf-panel1of1.png)

**Figure — ~04:00–05:25:** algorithms on densities; not all RVs have densities; PMF vs continuous; estimate density.

### What he is establishing

Why define density? **Most algorithms operate on density functions, not distribution functions** — for **mathematical ease**, no other deep reason.

**Not all random variables have densities.** There exist continuous RVs where densities are not even defined. The definition above is for the continuous case **when** a density exists.

**Discrete case:** evaluating a “density” at a point **is** a valid probability — but it is **not** called a density. It is the **probability mass function (PMF)**. By definition, $P(X=x)$ for discrete $X$ is a probability. That is **not** true for continuous densities.

Most of our data can be interpreted as continuous RVs → work with densities. Therefore **change the problem statement**:

- Before: given $D$, estimate the **distribution** function.  
- Now: given $D$, estimate the **density** function.

Assumption: data can be modeled with RVs for which densities are **well defined** — fair for most well-behaved practical RVs.

If you use continuous density thinking on discrete mass problems (or the reverse), you mis-handle probabilities. Instead: PMF for discrete points; density for continuous regions.

We now have the course motivation. Still missing: notation $P$ vs $p$, multi-d integrals, and the stop.

### Analogy for this topic only

Two bookshelves:

- Continuous shelf: “thickness of books per centimeter” (density) — thickness at a knife-edge is not a count of books.  
- Discrete shelf: “this exact ISBN has 3 copies” (mass) — that count **is** the probability weight after normalizing.  

In lecture words: algorithms prefer densities; PMF for discrete; estimate $p$ given $D$.

### Local picture

```
  Continuous:  p(x) height     → integrate for P
  Discrete:    p_X(x)=P(X=x)   → already probability (PMF)

  Problem shift:
    given D  estimate  P   →   given D  estimate  p
  (assume density exists for our data)
```

**Notice:** “not every abstract RV has a density” ≠ “real data never have densities.”

### Bridge

How will notation mark density vs distribution for the rest of the course?

---

## Topic 5: Estimate density; $P$ vs $p$; multi-d; next (05:26–08:05)

### Where this sits on the master map

**COURSE SHIFT + STOP**. Warm-up: [estimate density](./PREREQUISITES.md#p8-estimate-density), [multi-d](./PREREQUISITES.md#p7-multid).

### Board / screenshot

![Course uses densities](./screenshots/composites/ch05-topic-05-course-shift-panel1of1.png)

**Figure — ~05:26–08:00:** conditional densities; one-line vs two-line notation; multi-d integral; challenges next.

### What he is establishing

From now on work with density functions. Instead of estimating conditional **distributions**, estimate conditional **densities** — and the related joint / marginal densities:

$$
p(x,y),\qquad
p(x)=\int p(x,y)\,dy,\qquad
p(y\mid x)=\frac{p(x,y)}{p(x)}\quad(p(x)>0)
$$

(same structural jobs as Lec 08, written with $p$; the conditional still needs a **marginal** density in the denominator — itself an integral of the joint).

**Notation change (critical):**

```
  two-line / script:   P   or   𝒫     →  distribution function
  one-line:            p              →  density function
```

| Writing | Meaning |
|---------|---------|
| $P$ / two-line / script $\mathcal{P}$ | distribution function |
| $p$ / one line | density function |

Rest of the course: **densities**, not distribution functions.

Student Q about real-world distributions without nice closed formulas: he clarifies — he is **not** claiming every mathematical RV has a density. He **is** assuming practical data can be represented with RVs that **have** well-defined densities (“fair assumption for well-behaved” variables). Abstract counterexamples exist; engineering datasets are assumed density-friendly.

Recap definition: density from range of $X$ to $\mathbb{R}_+$ (non-negative reals); running integral recovers the distribution at a point. That integral is **not** always 1D: if the RV is $d$-dimensional (images!),

$$
P_X(x)=\int_{(-\infty,x_1]\times\cdots\times(-\infty,x_d]} p_X(t)\,dt
$$

— an integral over $\mathbb{R}^{d}$, “usual multi-dimensional integrals,” not a single scalar $\int$.

Foundation for ML problems is laid: mostly **density estimation** — conditional, marginal, joint densities given samples. **How** to do that is nontrivial — **challenges with ML** next lecture.

If you keep writing only CDFs while the course writes $p(y\mid x)$, you will misread algorithms. Instead: one-line $p$ = density; multi-d integrals for vectors; next = why estimation is hard.

We now have the course contract (densities forward). Still missing: the practical challenges of estimating a free-form $p$ from finite samples — next lecture.

### Analogy for this topic only

A map legend:

- Thick contour lines = CDF / distribution story  
- Color heatmap = density heights  

**Which legend will the rest of the course use?** The heatmap. Integrals of color over a region still give probability mass; a single pixel’s color intensity is not “the probability of that GPS point.”

In lecture words: estimate densities; one-line $p$ vs two-line $P$; multi-d integrals; challenges next.

### Local picture

```
  Notation:
    P  /  𝒫   →  distribution (two lines / script)
    p         →  density (one line)

  Problem:
    given D  estimate  p(x), p(y|x), p(x,y), …

  Multi-d:
    X ∈ R^d  →  ∫ over R^d  (volume integral)

  Next lecture: challenges of ML (density estimation)
```

**Notice:** foundation is complete for stating ML as density estimation from samples.

### Bridge

What makes estimating a free-form density from finite samples hard? Next lecture.

---

## External references

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| [3Blue1Brown — But what is a neural network? (not density but continuous inputs)](https://www.youtube.com/watch?v=aircAruvnKk) | Topic 4–5 | Continuous high-d inputs mindset |
| [StatQuest — Histograms, Probability & Density](https://www.youtube.com/watch?v=qBigTkBLU6g) | Topics 1–3 | Density vs probability visually |
| [Khan Academy — Probability density functions](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-continuous/v/probability-density-functions) | Topics 1–2 | CDF–PDF integral link |
| [Lec 08 package (estimate P)](../09-Lec08-Distribution-Estimation/NOTES.md) | Topics 4–5 | Problem statement this lecture rewrites |
| [Seeing Theory — Continuous distributions](https://seeingtheory.brown.edu/probability-distributions/index.html) | Topics 1–3 | Interactive continuous vs discrete |
| [3Blue1Brown — Binomial → Normal (density intuition)](https://www.youtube.com/watch?v=8idr1WZ1A7Q) | Topic 3 | Continuous height curves |

---

## Sources

- Video: [Lec 09 Density Function](https://www.youtube.com/watch?v=_QrezNPmxDk)  
- Channel: NPTEL — Indian Institute of Science, Bengaluru  
- Skill: `youtube-lecture-tutor`  
- Captions cleaned via timed transcript / claim sheets  
