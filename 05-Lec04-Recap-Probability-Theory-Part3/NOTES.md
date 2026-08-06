# Lec 04 — Recap of Probability Theory (Part 3)

**Video:** [Lec 04 Recap of Probability Theory - 1, Part 3](https://www.youtube.com/watch?v=0R6Agp4tqSU) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 03 — random variables](../04-Lec03-Recap-Probability-Theory-Part2/NOTES.md)

---

## Table of Contents

1. [Topic 1 — Borel sets and the pushforward measure](#topic-1-borel-sets-and-the-pushforward-measure-0000–0322) (00:00–03:22)
2. [Topic 2 — CDF via inverse images](#topic-2-cdf-via-inverse-images-0322–0920) (03:22–09:20)
3. [Topic 3 — Measurable RVs; estimating P is the course job](#topic-3-measurable-rvs-estimating-p-is-the-course-job-0920–1112) (09:20–11:12)
4. [Topic 4 — New triplet on the range; products in R^d](#topic-4-new-triplet-on-the-range-products-in-rd-1112–1410) (11:12–14:10)
5. [Topic 5 — Vector-valued RVs and product geometry](#topic-5-vector-valued-rvs-and-product-geometry-1410–1840) (14:10–18:40)
6. [Topic 6 — CDF is a measure; density is not](#topic-6-cdf-is-a-measure-density-is-not-1840–2210) (18:40–22:10)
7. [Topic 7 — Multiple RVs, two X-ray experiments, review](#topic-7-multiple-rvs-two-x-ray-experiments-review-2210–2906) (22:10–29:06)
8. [External references](#external-references)
9. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: move $P$ from abstract $\Omega$ onto the range of $X$ where vectors live. Method: $P_X$ sizes Borel regions by inverse images under $X$ (pushforward / CDF). Fork: density height is not probability; image and disease can be two interacting RVs.

**Worldview arc:** from $(\Omega,F,P)+X$ **to** working with $(R^d,B,P_X)$ and multi-RV conditionals next.

### System context

```
  ╔══════════════════════════════════╗
  ║ Outside                          ║
  ║ · Lec 02–03: RE, Ω, F, P, X      ║
  ║ · full formal probability course ║
  ║ · ML as distribution learning    ║
  ╚══════════════╤═══════════════════╝
                 │ this lecture builds
                 ▼
        ┌────────────────────┐
        │ pushforward P_X    │
        │ (distribution/CDF) │
        └────────────────────┘
```

### Main blueprint

```
  (Ω, F, P)  ──X──►  (R or R^d, Borel B, P_X)

  Scalar case:
    P_X(x) = P( X^{-1}((-∞, x]) )
           = P({ω : X(ω) ≤ x})

  Vector case:
    region = (-∞,x1] × … × (-∞,xd]
    P_X(x) = P( X^{-1}(region) )

  Rules of thumb:
    · CDF / distribution = valid measure
    · density ≠ probability at a point
    · ML often works only in pushforward space

  Multi-RV preview:
    image RE  ×  disease RE  → joints / conditionals
```

### Scenario walkthrough (X-ray)

1. Abstract process lives in $\Omega$; sensor/RV gives vector $x\in\mathbb{R}^{d}$.  
2. Ask “probability mass for values ≤ some threshold region.”  
3. Pull that region back to $\Omega$, apply $P$ → $P_X$.  
4. Disease label may be another coordinate/RV; conditionals = set interactions.  
5. Engineering job: estimate $P_X$ (or $P$) from samples in the range.

### Failure / contrast path

```
  Density at a point "is probability"  ──X──► wrong
  Stay only on Ω with no CDF on R^d   ──X──► cannot code on vectors
  Forget inverse-image definition of CDF ──X──► handwavy "distribution"
```

### STOP / out of scope

Full construction of densities, formal σ-algebra proofs, and the complete multi-RV toolkit are homework / next class. ML as distribution learning is announced, not finished.

### Load-bearing claims (closed-book)

- Borel-style regions include $(-\infty,x]$ (products in higher $d$).  
- Distribution / CDF / pushforward $P_X$ is $P$ of inverse images under $X$.  
- Course core: estimate that measure from range samples.  
- After sensors, work in range triplet, not raw $\Omega$ every day.  
- Vector RV uses Cartesian half-space products.  
- Density at a point is not a probability.  
- Image vs disease can be two REs/RVs with interactions.  
- Next: conditionals + ML as distribution learning.

**Speaker / course:** NPTEL IISc · MFML · Lec 04 Part 3.

---

## Topic 1: Borel sets and the pushforward measure (00:00–03:22)

### Where this sits on the master map

**PUSHFORWARD** box — answers the cliffhanger from Lec 03: what happens to $P$ under $X$? Warm-up: [sets](./PREREQUISITES.md#p0-sets), [sample space](./PREREQUISITES.md#p0-sample-space), [half-lines / cumulative](./PREREQUISITES.md#p2-half-lines), [triplet $(\Omega,F,P)$](./PREREQUISITES.md#p4-measure), [bridge formula](./PREREQUISITES.md#p8-why).

### Board / screenshot

![Borel sets and pushforward](./screenshots/composites/ch01-topic-01-pushforward-panel1of1.png)

**Figure — ~00:00–03:20:** subsets of the form $(-\infty,x]$; mapping of measures under $X$; names distribution / pushforward.

### What he is establishing

We had a probability measure $P$ on events in $\Omega$. Under a random variable, we do not score arbitrary wild subsets of $\mathbb{R}$. We use special **measurable** subsets — in particular, for range $\mathbb{R}$, sets shaped like $(-\infty, x]$, and analogous product regions in $\mathbb{R}^{d}$. These sit in the **Borel σ-algebra** (spoken “Borel”; ASR often writes “boral”).

$X$ maps outcomes $\omega$ to numbers, so it also maps subsets of $\Omega$ to subsets of $\mathbb{R}$. Because we already assigned $P$ on $F$, there must be an **equivalent sizing** on the image side. That object is called the **distribution function**, **probability distribution function**, **cumulative distribution function (CDF)**, and also the **pushforward measure**.

Picture: start with a measurable-space triplet, define a function on it, and a **new measure emerges**. Course notation: a script $P$ with subscript $X$ (written here $P_X$) means the distribution under random variable $X$.

Wrong move: invent a new probability on $\mathbb{R}$ with no link to $P$ on $\Omega$. Right move: push $P$ forward through $X$.

You can now name Borel half-lines, pushforward, and $P_X$, and keep the original triplet $(\Omega,F,P)$ in view. Still missing: the explicit formula for $P_X(x)$.

### Analogy for this topic only

You have weights on city blocks (measure on the city). A map projects the city onto a number line of “distance from downtown.”

**How do you score the interval “distance ≤ 3 km”?** Collect all city blocks that land in that interval, add their original weights. That is a pushforward score on the line.

In lecture words: city = $\Omega$, map = $X$, line scores = distribution $P_X$.

### Local picture

```
  (Ω, F, P)  --X-->  (R, Borel, P_X)

  Borel-ish generators on R:  (-∞, x]
  Name: distribution / CDF / pushforward measure
```

**Notice:** new measure is not free invention — it is $P$ transported by $X$.

### Bridge

Write the 1D formula carefully using inverse images.

---

## Topic 2: CDF via inverse images (03:22–09:20)

### Where this sits on the master map

**CDF DEF** box — the core formal procedure. Warm-up: [inverse image + coin table](./PREREQUISITES.md#p1-inverse-image), [cumulative half-lines](./PREREQUISITES.md#p2-half-lines).

### Board / screenshot

![CDF definition board](./screenshots/composites/ch02-topic-02-cdf-definition-panel1of1.png)

**Figure — ~03:22–09:20:** $P_X(x)$ via inverse image of $(-\infty,x]$; endpoint correction (closed at $x$).

### What he is establishing

Work in **one dimension** first for clarity, then generalize. Evaluate the distribution $P_X$ at a real number $x$. Probability measures are always assigned to **sets**, so this evaluation must secretly refer to a set.

**Procedure:**

1. Form the Borel-style set $S = (-\infty, x]$ in $\mathbb{R}$ (open toward $-\infty$, **closed at $x$**).  
2. Form the inverse image $A = X^{-1}(S) = \{\omega\in\Omega : X(\omega)\in S\} = \{\omega : X(\omega)\le x\}$.  
3. Set $P_X(x) := P(A)$.

In symbols:

$$
P_X(x) = P\big(X^{-1}((-\infty, x])\big) = P(\{\omega : X(\omega) \le x\})
$$

In words: the distribution at $x$ is the original probability of the event “the random variable is at most $x$.” That is why people also say **cumulative** distribution: it piles probability for all values up through $x$.

**Micro numbers (fair coin).** Let $\Omega=\{H,T\}$, $P(\{H\})=P(\{T\})=\tfrac12$, $X(H)=1$, $X(T)=0$.

| $x$ | region $(-\infty,x]$ | preimage | $P_X(x)$ |
|-----|----------------------|----------|----------|
| $-1$ | no values of $X$ | $\emptyset$ | $0$ |
| $0$ | includes $0$ only | $\{T\}$ | $1/2$ |
| $0.5$ | includes $0$ only | $\{T\}$ | $1/2$ |
| $1$ | includes $0$ and $1$ | $\{H,T\}$ | $1$ |

Classroom detail: students correct the endpoint openness; the standard CDF uses closed at $x$. (There is a short cultural aside on human feedback and confirmation bias — not load-bearing math.)

Wrong move: say “$P_X(0.5)$ is just a free number called probability.” Right move: always name the set $(-\infty,0.5]$ and its preimage first — then read $1/2$ off $P$.

You can now compute the CDF definition as inverse image + $P$, including a full numeric table. Still missing: why such inverse images always exist for a “legal” RV, and what engineers estimate in practice.

### Analogy for this topic only

A coin shows head or tail. You label head as one and tail as zero. Someone asks for the distribution score at one half.

**Which faces land at most one half?** Only tails. That preimage’s old probability is the distribution value.

In lecture words: pull the half-line back, then use $P$.

### Local picture

```
  R:   (-∞ .......... x]
              │
         X^{-1}
              ▼
  Ω:   {ω : X(ω) ≤ x}  --P-->  P_X(x)

  Fair coin micro: X(H)=1, X(T)=0, P fair
  P_X(-1)=0 · P_X(0)=1/2 · P_X(0.5)=1/2 · P_X(1)=1
```

**Notice:** $P_X(x)$ is always a set probability, never a free scalar with no set behind it. The table above is the same idea as the establishing micro numbers.

### Bridge

How do we know inverse images of Borel sets exist? And who “assigns” $P$ in the real world?

---

## Topic 3: Measurable RVs; estimating $P$ is the course job (09:20–11:12)

### Where this sits on the master map

**ESTIMATION** box — ties theory to the MFML engineering goal. Warm-up: [measure sizes sets](./PREREQUISITES.md#p4-measure).

### Board / screenshot

![Estimation of measure](./screenshots/composites/ch03-topic-03-estimation-goal-panel1of1.png)

**Figure — ~09:20–11:12:** measurability of RVs; nature/labeler assigns measure; engineers estimate from samples.

### What he is establishing

Student worry: for an interval on $\mathbb{R}$, does a preimage in $\Omega$ exist as an allowed event? By definition, **random variables are set up so that inverse images of Borel sets are events** (measurability). That is part of what “being a random variable” means in this framework.

Separate question: how is the probability measure **assigned or estimated** in practice? That is essentially the **central course question**: given elements from the **range** of random variables, **estimate the underlying probability measure / distribution**.

Concrete scene: a radiologist looks at a film and clicks “diseased” with hard-to-state intuition. Nature/labeler “assigns” the true measure that way. Your folder of labeled vectors is what you get. Engineers do not invent that truth from pure physics; they **estimate** from samples.

Wrong move: demand a first-principles formula for every $P$ before learning is allowed. Right move: accept that samples carry information about an unknown measure.

You can now separate measurability of $X$ from the estimation goal of the course. Still missing: the daily working triplet on the range side.

### Analogy for this topic only

A doctor labels films “diseased / not” with hard-to-state intuition. Your dataset is the visible samples.

**Do you need the doctor’s full inner monologue to start learning?** No — you estimate the labeling measure from many labeled samples.

In lecture words: nature/labeler assigns; you estimate from range samples.

### Local picture

```
  True P (nature / labeler)
        │
        │ samples of X(ω)
        ▼
  estimate P or P_X   ← course core job
```

**Notice:** “how is $P$ assigned?” is not a bug — it is the learning problem.

### Bridge

After $X$, what triplet do we actually compute with?

---

## Topic 4: New triplet on the range; products in $\mathbb{R}^{d}$ (11:12–14:10)

### Where this sits on the master map

**RANGE TRIPLET** box — rename the workspace of ML practice. Warm-up: [Cartesian product](./PREREQUISITES.md#p6-cartesian).

### Board / screenshot

![Range triplet and products](./screenshots/composites/ch04-topic-04-range-triplet-panel1of1.png)

**Figure — ~11:12–14:10:** Borel products in higher $d$; pushforward triplet; work in measured vector space.

### What he is establishing

What do Borel generators look like in $\mathbb{R}^{2}$ or $\mathbb{R}^{d}$? **Cartesian products** of one-dimensional generators (half-lines). That is why joint distributions and vector-valued random variables appear.

From the original triplet we obtain a **new triplet** on the range: roughly $(\mathbb{R}, B, P_X)$ (or $\mathbb{R}^{d}$). What made the transformation possible? The random variable — again stressed as a **function**, a pushforward map from one measure space to another, **not** a “variable.”

Practical course stance: once sensors give vectors, the instructor will mostly talk about the **pushforward space** — range, Borel structure, distribution — and less about raw $\Omega,F,P$ in every sentence. Assume a random variable exists. That is why RVs were introduced with so much care: not randomly, but to justify living where data live.

Wrong move: pretend $\Omega$ was deleted. Right move: $\Omega$ is backstage while code runs on $R^{d}$ with $P_X$.

You can now state the range triplet and why ML conversation shifts there. Still missing: explicit geometry of vector CDFs and the density warning.

### Analogy for this topic only

You stop discussing “which abstract hospital story occurred” and start discussing “which numeric film vector you hold,” while still knowing the film came through a camera map.

**Is that abandoning probability?** No — you moved to the pushforward coordinates where code runs.

In lecture words: work with $R$, Borel, $P_X$ after measurement.

### Local picture

```
  (Ω, F, P)  --X-->  (R^d, B, P_X)

  Daily ML talk after sensors:
    vectors + distributions on R^d
```

**Notice:** $\Omega$ is not deleted; it is backstage.

### Bridge

Spell out vector-valued RVs and product half-spaces.

---

## Topic 5: Vector-valued RVs and product geometry (14:10–18:40)

### Where this sits on the master map

**VECTOR RV** box — $d$-dimensional data distributions. Warm-up: [vectors $\mathbb{R}^{d}$](./PREREQUISITES.md#p0-vectors), [Cartesian product](./PREREQUISITES.md#p6-cartesian).

### Board / screenshot

![Vector RV geometry](./screenshots/composites/ch05-topic-05-vector-rv-panel1of1.png)

**Figure — ~14:10–18:40:** range $\mathbb{R}^{d}$; product regions $(-\infty,x_1]\times(-\infty,x_2]$; arguments of $P_X$ are range points.

### What he is establishing

In general the range is $\mathbb{R}^{d}$. People say **vector-valued random variable** — slightly misnamed: the honest phrase is “a random variable whose range consists of vectors in $\mathbb{R}^{d}$.”

For $d=2$, the basic regions are products $(-\infty,x_1]\times(-\infty,x_2]$. Geometry: on the plane, that is everything **left of** $x_1$ and **below** $x_2$ (the south-west infinite rectangle toward $(x_1,x_2)$). Extend the same idea to $d$ dimensions.

**Always:** arguments of a distribution function are **elements of the range** of the RV. If the range is $\mathbb{R}^{d}$, the input to $P_X$ is a vector in $\mathbb{R}^{d}$. Notation overload: capital $X$ for the RV (or bold vector) versus small $x$ for a point in the range — read carefully as the course goes on.

Wrong move: treat the corner $(x_1,x_2)$ as if $P_X$ only cares about that single point. Right move: the corner **indexes** a whole south-west region whose preimage is measured.

You can now draw the 2D product region and state what $P_X$ takes as input. Still missing: why densities are not probabilities, and multi-RV modeling of labels.

### Analogy for this topic only

Exam A score and exam B score. Someone asks for the distribution at the pair (70, 80).

**Is that only “exactly 70 and exactly 80”?** No — it sizes everyone who scored at most 70 on A **and** at most 80 on B. That joint region is a product of two half-lines on the score plane.

In lecture words: products of $(-\infty,x_i]$ for vector RVs.

### Local picture

```
  R^2:
       x2
        |     (x1,x2)
        |____·
        |////|   shaded = (-∞,x1] × (-∞,x2]
        |////|
        +----------- x1

  P_X(x1,x2) = P( X^{-1}(that shaded region) )
```

**Notice:** input to $P_X$ is the corner point $(x_1,x_2)$, but the measure is of a whole region.

### Bridge

Is “the value of a density at $x$” the same kind of object as $P_X(x)$?

---

## Topic 6: CDF is a measure; density is not (18:40–22:10)

### Where this sits on the master map

**CDF VS PDF** box — critical trap for beginners. Warm-up: [density vs measure](./PREREQUISITES.md#p5-density-vs-measure).

### Board / screenshot

![CDF versus density](./screenshots/composites/ch06-topic-06-cdf-vs-density-panel1of1.png)

**Figure — ~18:40–22:10:** distribution as measure; density not a probability; inverse image of product region.

### What he is establishing

The **distribution function is a valid measure**. The **density function is not**. If someone says “evaluate the probability density at a point and that number is the probability,” they are **wrong**. Densities will be defined carefully later; today the warning is enough.

Why is the CDF a measure? Because it is really evaluating the original probability measure $P$ on an inverse-image event. For a vector argument $x\in\mathbb{R}^{d}$,

$$
P_X(x) = P\big(X^{-1}((-\infty,x_1]\times\cdots\times(-\infty,x_d])\big)
$$

In words: probability of the sample-space event that is the preimage of that product region. That event is a subset of $\Omega$ and can even be a singleton in some models. In the X-ray story it can correspond to something like the probability mass tied to a diseased case — but whenever the lecturer says “probability,” he means the **measure**, not slang.

He notes this is not a full probability course: refresh formal ST0-style material; he will rush remaining tools.

Wrong move: treat $p(x)$ density height as $P(\{X=x\})$ in continuous models. Right move: only set measures (CDFs of regions) are probabilities.

You can now refuse density-as-probability and write the product inverse-image formula. Still open: multiple interacting RVs and the review list.

### Analogy for this topic only

A weather map shows “rain intensity” (height) and also “chance of rain in this county” (area-like score).

**Is the intensity at the city center the chance it rains in the county?** No. Height ≠ regional chance.

In lecture words: density height ≠ probability measure.

### Local picture

```
  OK:   P_X(region) = P(preimage)   ← measure
  BAD:  "density(x) = probability"  ← not a measure evaluation

  Vector:
  P_X(x) = P(X^{-1}(∏ (-∞, x_i]))
```

**Notice:** continuous point probabilities are a different conversation; do not smuggle them via density evaluation.

### Bridge

Labels and images as two random variables — and what to review before next class.

---

## Topic 7: Multiple RVs, two X-ray experiments, review (22:10–29:06)

### Where this sits on the master map

**MULTI-RV + STOP** — interactions, modeling choices, homework. Warm-up: [two experiments / joints](./PREREQUISITES.md#p7-two-experiments), [sample space design choice](./PREREQUISITES.md#p0-sample-space).

### Board / screenshot

![Multiple RVs and review](./screenshots/composites/ch07-topic-07-multi-rv-review-panel1of1.png)

**Figure — ~22:10–29:00:** joints; image vs disease experiments; vector Gaussians; discriminative/generative preview; review list.

### What he is establishing

There need not be only one random variable. **Joint distributions** describe pairs (or many) RVs. Multiple RVs mean multiple underlying sample spaces in the strict function-on-Ω view — each RV is defined on a sample space. “Interaction” is not vague poetry: it is the same kind of structure set theory gives (intersections, unions, and so on) transported through pushforward measures.

**X-ray modeling choice:**  
- Process of being imaged can be one random experiment / sample space.  
- Process of having disease or not can be another.  
They may be correlated, but can still be two experiments. Labeling connects them.  
Alternatively, “get disease and get imaged” can be modeled as **one** RE — the designer chooses.  

A **vector-valued** RV can be seen as one map $\Omega\to\mathbb{R}^{d}$ **or** as $d$ correlated scalar RVs. That is why multivariate Gaussians have correlation matrices: either one $d$-dimensional RV or $d$ interacting scalars.

Wrong move: insist there is only one “correct” RE packaging for image and label. Right move: designer chooses one RE or two; both can support joints and conditionals.

Worldview payoff: **discriminative and generative** models start to look unified — the label is simply **one coordinate** of a vector-valued RV. Conditionals arise because set operations (intersections) are available.

**Homework / review list before next class:**

- Multiple random variables  
- Conditional distributions — always ask which **events** on the sample space you are conditioning  
- Relate everything back to sample spaces and sets  
- Discrete vs continuous random variables  
- What distributions are  

Next class: quick run through key multi-RV ideas, then **formulate machine learning as distribution learning** and reconnect to function approximation. This class covered sample spaces, probability measures, distribution functions, and the pushforward story.

You can now model image and label as interacting RVs and list the review tasks. Still missing: the formal ML problem statement as distribution estimation (next lecture).

### Analogy for this topic only

Two coins in two rooms can be biased the same way (correlated factories) even if tossed separately.

**Question:** are those two coins “one experiment” or “two”? Either answer can be made rigorous — outcomes in a product space vs a pair of sample spaces.

Same data, two modeling languages — like vector RV vs several scalars.

In lecture words: designer chooses RE; interactions = set-like structure on RVs.

### Local picture

```
  View A:  RE_image → X_image ∈ R^d
           RE_disease → Y ∈ {0,1}
           joints / conditionals between them

  View B:  one RE → (X,Y) as one vector RV

  Review: multi-RV · conditionals · discrete/continuous · back to Ω sets
  Next:   ML = distribution learning (+ link to FA)
```

**Notice:** generative vs discriminative unification is a preview, not a finished theory chapter.

### Bridge

Refresh the listed probability tools; next sessions cast learning as estimating $P_X$ / conditionals and reconnect FA.

---

## External references

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| [StatQuest — The normal distribution, clearly explained](https://www.youtube.com/watch?v=rzFX5NWojp0) | Topics 5–6 continuous distributions | Intuition before density-vs-measure trap |
| [StatQuest — Histograms, Probability & Density](https://www.youtube.com/watch?v=qBigTkBLU6g) | Topic 6 density ≠ probability | Directly supports the height-vs-area distinction |
| [MIT 6.041SC — Continuous RVs / CDF ideas (playlist)](https://www.youtube.com/playlist?list=PLUl4u3cNGP60A3XMwZ5sep719_nh95qYp) | Topics 2,5–6 | University CDF and multi-RV material |
| [Seeing Theory — Probability Distributions](https://seeingtheory.brown.edu/probability-distributions/index.html) | Topics 2,5 | Interactive CDF-style thinking |
| [Lec 03 package (this series)](../04-Lec03-Recap-Probability-Theory-Part2/NOTES.md) | Topic 1 setup | RV bridge this lecture pushes forward |
| [StatQuest.org](https://statquest.org/) | Topics 3,7 estimation language | Short posts on sampling and estimation |

---

## Sources

- Video: [Lec 04 Recap of Probability Theory - 1, Part 3](https://www.youtube.com/watch?v=0R6Agp4tqSU)  
- Channel: NPTEL — Indian Institute of Science, Bengaluru  
- Skill: `youtube-lecture-tutor`  
- Captions cleaned in TRANSCRIPT.md / claim sheets  
