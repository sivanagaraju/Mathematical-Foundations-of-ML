# Prerequisites — warm-up before Lec 09 (density functions)

> **Do this first** if “density,” “CDF,” or “$p(x)$ is a probability” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on [Lec 08](../09-Lec08-Distribution-Estimation/PREREQUISITES.md) (estimate $P$; densities teased).  
> Every formula below is decoded in plain English.

```
  After this warm-up you can say:

  "A density p is a non-negative height (rate) on the range of a continuous RV."
  "The CDF is the running integral of the density: P(X≤x)=∫_{-∞}^x p(t) dt."
  "Thin strip: P([x,x+dx]) ≈ p(x) dx — only area is probability."
  "p(x) at one point is NOT a probability; ∫ over a region IS."
  "Uniform on [0,1/2] has height 2 (=1/L) — already >1, so height cannot be probability."
  "Discrete twin: PMF masses at points ARE probabilities."
  "Course shift: estimate densities p, not only distributions P."
```

**Warm-up → lecture boxes**

```
  §1  Continuous RV + range of P/CDF     ──► Topic 1
  §2  Density definition (integral link) ──► Topic 1
  §3  Height ≠ probability               ──► Topics 2–3
  §4  Uniform[0,½] worked micro          ──► Topic 3
  §5  Likelihood vs probability          ──► Topic 3
  §6  PMF (discrete) vs density          ──► Topic 4
  §7  Multi-d integrals                  ──► Topic 5
  §8  Course: estimate p                 ──► Topics 4–5
```

---

## 1. Continuous random variables and the CDF (reload)

<a id="p1-continuous-cdf"></a>

### Purpose for the video

Densities are for **continuous** RVs. Start from the CDF you already met.

### Random variable and range

$X:\Omega\to\mathbb{R}$ (or $\mathbb{R}^{d}$) is a deterministic map. Data $x$ is a **range point**, not a probability.

### CDF / distribution function

$$
P_X(x)=P(X\le x)=P\big(X^{-1}((-\infty,x])\big)
$$

| Symbol | Meaning |
|--------|---------|
| $P_X(x)$ | probability that $X$ lands at most $x$ |
| Always | a number in $[0,1]$ (a true probability of a set) |

For continuous models, $P(X=x)=0$ for each single point $x$ — probability lives on **intervals / regions**, not isolated points.

### Analogy — purpose

A continuous thermometer: chance of reading **exactly** $36.512938\ldots^\circ$C is zero; chance of “between 36 and 37” is positive.

---

## 2. Density definition — the integral formula

<a id="p2-density-def"></a>

### Purpose for the video

This is the definition he writes on the board.

### Notation

$$
p_X(x)
\quad\text{or simply}\quad
p(x)
$$

- **Small** $p$ = density  
- Subscript $X$ (optional) = which RV  

### Domain and codomain

$$
p_X:\{\text{range of }X\}\to\mathbb{R}_+
$$

| Fact | Meaning |
|------|---------|
| Non-negative | $p_X(x)\ge 0$ for all $x$ |
| **Not** forced into $[0,1]$ | heights can be $2$, $10$, $100$ |
| Output is a **height**, not a probability | see §3 |

### Link to the CDF (definition)

$$
\boxed{P_X(x)=\int_{-\infty}^{x} p_X(t)\,dt}
$$

| Piece of the formula | Meaning |
|----------------------|---------|
| $P_X(x)$ | CDF: chance $X\le x$ (a probability in $[0,1]$) |
| $\int_{-\infty}^{x}$ | “add up all area under the curve to the left of $x$” |
| $p_X(t)\,dt$ | thin strip of height $p_X(t)$ and width $dt$ |
| whole equality | **definition** of density (when it exists): CDF = running integral of $p$ |

**Whole-space normalization** (valid density must put total mass 1):

$$
\int_{-\infty}^{\infty} p_X(t)\,dt = 1
$$

(or $\int_{\mathbb{R}^{d}} p=1$ in higher dimension). That is why the CDF satisfies $\lim_{x\to\infty}P_X(x)=1$ and $\lim_{x\to-\infty}P_X(x)=0$ when a density exists.

**Conversely (when density exists and is nice):** $p_X$ is the derivative of the CDF:

$$
p_X(x)=\frac{d}{dx}P_X(x)
$$

(The lecture stresses the integral form.)

### Probability of a region

For an interval $[a,b]$:

$$
P(a\le X\le b)=\int_a^b p_X(t)\,dt
=P_X(b)-P_X(a)
$$

English: **area under $p$ between $a$ and $b$** is the probability of landing there.

### Thin strip — what “density” actually means

Over a tiny width $dx$ around a point $x$:

$$
P\big(x\le X\le x+dx\big)\;\approx\; p_X(x)\,dx
$$

| Piece | Meaning |
|-------|---------|
| $p_X(x)$ | height = **probability per unit of $x$** (a rate / density) |
| $dx$ | width of a thin bin |
| $p_X(x)\,dx$ | **area** of that thin bin ≈ probability of landing in it |

So:

- Probability is always an **area** (or volume in multi-d), never a bare height.  
- Height $p$ has **units of “1 / unit of $x$”** — e.g. “per centimeter,” not “percent.” That is why $p(x)$ is allowed to be larger than 1: if the whole mass-1 is squeezed into a short interval, average height must exceed 1 (see §4).  
- Evaluating $p(x_0)$ alone drops the $dx$ — you threw away the width that turns a rate into a probability.

### Two continuous facts (do not mix them)

| Fact | Statement |
|------|-----------|
| A | For a continuous model, $P(X=x_0)=0$ for each single exact value $x_0$ |
| B | Even so, $p(x_0)$ may be **positive** (or $>1$) — that height is **still not** $P(X=x_0)$ |

**Why A is automatic from the integral:** the “region” $\{x_0\}$ has zero width, so

$$
P(X=x_0)=\int_{\{x_0\}} p_X(t)\,dt=0
$$

even when $p_X(x_0)$ is large. Fact A is about **probability of a singleton**. Fact B is about **not confusing height with probability**. Both appear in the lecture dialogue.

### Analogy — purpose

Density = height of a mountain profile.  
Probability = **land area** under a stretch of the mountain — not the elevation at one GPS pin.

---

## 3. The critical trap: $p(x)$ is not a probability

<a id="p3-height-not-prob"></a>

### Purpose for the video

This is the main kill shot of the lecture.

### Two different operations

| Operation | Result | Is it a probability? |
|-----------|--------|----------------------|
| Evaluate $p(x_0)$ at one point | a number $\ge 0$ (height) | **No** |
| Integrate $p$ over a set $A$ | $\int_A p$ | **Yes** (when density exists) |

Even if $p(x_0)=0.3$, that is **not** “30% chance that $X=x_0$.”

### Why people get trapped

Gaussian densities often take values less than 1 near the mode, so the height “looks like” a probability. That is coincidence, not a rule — Gaussians with small variance have peaks **above** 1 too (same $1/L$-style squeeze).

### Logic chain (closed-book)

```
  p(x0)  is a rate (height)
     │
     │  × width of a set A
     ▼
  ∫_A p   is a probability ∈ [0,1]
```

If someone reports $p(x_0)$ as “the chance of $x_0$,” they skipped the “× width / integrate” step.

### Analogy — purpose

Speedometer reads $120$ km/h. That is not “120% of a trip completed.” Height of a rate is not a fraction of a whole unless integrated over time.

---

## 4. Worked micro: Uniform on $[0,1/2]$ has height $2$

<a id="p4-uniform-half"></a>

### Purpose for the video

His interview counterexample.

### Where does the number **2** come from?

“Uniform on an interval of length $L$” means the density is a **flat rectangle** whose **area is 1**.

$$
\text{height}\times\text{width}=1
\quad\Rightarrow\quad
\text{height}=\frac{1}{L}
$$

Here the interval is $[0,1/2]$, so $L=1/2$ and height $=1/(1/2)=2$:

$$
p(x)=\begin{cases}
2 & 0\le x\le 1/2\\
0 & \text{otherwise}
\end{cases}
$$

**Check normalization:**

$$
\int_0^{1/2} 2\,dx = 2\cdot\frac12 = 1
$$

Same recipe for any flat uniform on length $L$: height $=1/L$.  
- Uniform$[0,1]$ → height $1$  
- Uniform$[0,1/2]$ → height $2$  
- Uniform$[0,1/10]$ → height $10$  

Shorter support → taller density. That is the thin-strip idea from §2 in numbers: total mass 1 packed into small width forces large height.

### Why this kills “height = probability”

At every $x$ in the support, $p(x)=2>1$.  
But every **probability** must lie in $[0,1]$.  
So the number $2$ **cannot** be a probability — yet it is a perfectly valid density height.

Integrals still give legal probabilities. Example:

$$
P\!\left(0\le X\le \tfrac14\right)=\int_0^{1/4}2\,dx=\tfrac12
$$

(half the support length → half the mass, because the density is flat).

### Support (word worth knowing)

The **support** is where $p(x)>0$ (here $[0,1/2]$). Outside the support, $p=0$ — still a legal density value, just “no mass there.”

### Analogy — purpose

A short tall rectangle of height 2 and width $1/2$ has area 1. Height exceeds 1; area does not.

---

## 5. Likelihood vs probability (name only)

<a id="p5-likelihood"></a>

### Purpose for the video

He names density-at-a-point as **likelihood** (used later in the course).

| Word | Rough role here |
|------|-----------------|
| **Probability** | size of a **set** of outcomes / range region (in $[0,1]$) |
| **Likelihood** | value of the **density** $p(x)$ at a point (can be $>1$) |

Do not swap the words casually. Later algorithms maximize likelihood — they maximize a density height, not a probability of a singleton continuous point.

---

## 6. Discrete twin: PMF

<a id="p6-pmf"></a>

### Purpose for the video

Discrete RVs use a different object.

### Probability mass function

For discrete $X$ taking values in a countable set:

$$
p_X(x)=P(X=x)
$$

| Fact | Continuous density | Discrete PMF |
|------|--------------------|--------------|
| Name | density $p$ | mass function (PMF) |
| Evaluate at a point | **not** a probability | **is** a probability |
| Recover law | integrate over regions | sum masses |

Lecture: for continuous data we use densities; for discrete, point masses are already probabilities.

### Micro

Fair die: $P(X=6)=1/6$ — that **is** a probability of the singleton $\{6\}$.

---

## 7. Vector RVs: multi-dimensional integrals

<a id="p7-multid"></a>

### Purpose for the video

When $X\in\mathbb{R}^{d}$ (images, $d$ large):

$$
P_X(x)=\int_{(-\infty,x_1]\times\cdots\times(-\infty,x_d]} p_X(t)\,dt
$$

or “running integral over $\mathbb{R}^{d}$,” not a single 1D integral. Notation $\int p(x)\,dx$ often **abuses** dimension — $dx$ means volume element in $\mathbb{R}^{d}$.

---

## 8. Course shift: estimate the density

<a id="p8-estimate-density"></a>

### Purpose for the video

Lec 08: given $D$, estimate the **distribution**.  
Lec 09: assume continuous well-behaved data → given $D$, estimate the **density** $p$. Then joints / marginals / conditionals become density objects:

| Object | Density form |
|--------|----------------|
| Joint | $p(x,y)$ |
| Marginal of $x$ | $p(x)=\displaystyle\int p(x,y)\,dy$ (integrate out $y$) |
| Conditional | $p(y\mid x)=\dfrac{p(x,y)}{p(x)}$ when $p(x)>0$ |

same skeleton as before, now written with densities. The conditional formula needs the **marginal** density in the denominator — still an integral of a joint density, not a bare height read as a probability.

### Notation he will use (visual)

```
  Distribution / CDF-style:   P   or   𝒫   (he says "two lines")
  Density:                    p         (he says "one line")
```

| Writing | Meaning |
|---------|---------|
| One-line $p$ | density height function |
| Two-line / script $\mathcal{P}$ or capital $P$ | distribution / CDF-style measure |

Rest of course: densities for mathematical ease (algorithms prefer them).  
Assumption: practical data can be modeled with RVs that **have** densities (not: every abstract RV has a density).

### Pushforward (one-line bridge to earlier lectures)

The distribution $P_X$ is still the **pushforward** of the original measure $P$ on $\Omega$ through the map $X$. Density is a convenient **description** of that pushforward on the range when it exists — not a replacement of $\Omega$.

### Paper check

1. Write $P_X(x)=\int_{-\infty}^{x}p(t)\,dt$ and $\int p=1$ in English.  
2. Write the thin-strip formula $P([x,x+dx])\approx p(x)\,dx$ and say what $p$ is a rate of.  
3. Derive: why is Uniform$[0,1/2]$ density equal to 2? What about Uniform$[0,1/10]$?  
4. Is $p(0.3)=0.8$ a probability? Is $P(X=0.3)=0$ in a continuous model? Why both can be true?  
5. Compute $P(0\le X\le 1/4)$ for Uniform$[0,1/2]$.  
6. PMF vs density: which is probability when evaluated at a point?  
7. Write $p(y\mid x)$ and the marginal $p(x)$ in terms of the joint density.  
8. What does the course estimate from now: $P$ or $p$?

---

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).  
Prior: [Lec 08](../09-Lec08-Distribution-Estimation/NOTES.md).
