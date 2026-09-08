# Prerequisites — warm-up before Lec 04 (distributions / pushforward)

> **Do this first** if probability notation feels shaky. Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.
> Goal: unlock every map word in this video (inverse image, CDF, Borel half-lines, pushforward, density trap, joints).
> Still a **warm-up**, not a full probability course — but detailed enough if you do not know the basics.
> Prior packages: [Lec 02 Part 1](../03-Lec02-Recap-Probability-Theory-Part1/PREREQUISITES.md) · [Lec 03 Part 2](../04-Lec03-Recap-Probability-Theory-Part2/PREREQUISITES.md)

```
  After this warm-up you can say:

  "Sets hold outcomes; events are the subsets we size."
  "A function maps each input to one output; a preimage pulls a set of outputs back."
  "(-∞, x] means everything up to and including x (cumulative)."
  "(Ω, F, P) is the abstract probability triple; X turns outcomes into numbers."
  "P_X(x) sizes the event {outcomes whose number is ≤ x}."
  "Density height is not probability; set size is."
  "2D '≤ both scores' regions are products of half-lines."
  "Image and disease can be two experiments that interact."
```

**How to use:** work top to bottom. Redo every **Micro** on paper. Optional 2-minute check is at the end of idea 8.

---

## 1. Sets, experiments, and sample space

<a id="p0-sets"></a>
<a id="p0-sample-space"></a>

### Sets and subsets

A **set** is a bag of objects. Members are **in**; everything else is **out**.

```
  Die faces:     Ω = {1, 2, 3, 4, 5, 6}
  Even faces:    A = {2, 4, 6}     ← a subset of Ω
  Empty bag:     ∅ = { }          ← no members
```

**Subset:** every member of the smaller bag is already in the bigger bag ($A \subset \Omega$).

**Micro:** Is $\{1,7\}$ a subset of the die $\Omega$? **No** — $7$ is not a face.

Why it matters: probability scores **subsets of outcomes**, not free-floating adjectives.

### Random experiment and sample space

A **random experiment (RE)** is a procedure we treat as able to produce different outcomes (toss a coin, image a patient, open an email).

The **sample space** $\Omega$ is the set of **all possible outcomes** you admit for that experiment.

```
  Coin toss once:     Ω = {H, T}
  Roll one die:       Ω = {1,2,3,4,5,6}
  Abstract clinic:    Ω = { process stories … }   (may not be numbers!)
```

Rules of thumb for this course:

- $\Omega$ can be **non-numeric** (head/tail, clinic process).
- **You choose** the experiment: “one toss” vs “five tosses” are different REs.
- After $\Omega$ is fixed, most of the work is **sets + measures**.

**Micro scene — two modeling choices for email spam:**

- One RE whose outcomes are pairs (email text, spam bit).
- Two REs: writing/sending email, and labeling spam.

Both can be valid. The lecture uses the same freedom for X-ray **image** vs **disease**.

---

## 2. Functions and inverse images (the video’s main tool)

<a id="p0-functions"></a>
<a id="p1-inverse-image"></a>

### Function = one input → one output

$$
f : A \to B

$$

assigns **exactly one** element of $B$ to each element of $A$.

```
  OK:     H → 1 ,  T → 0
  NOT OK: H → 1 sometimes and 0 other times  (not one function)
```


| Word                 | Meaning                                                  |
| ---------------------- | ---------------------------------------------------------- |
| **Domain**           | input set$A$                                             |
| **Range / codomain** | output world$B$ (often $\mathbb{R}$ or $\mathbb{R}^{d}$) |

### Inverse image (preimage)

If $X:A\to B$ and $S\subset B$:

$$
X^{-1}(S) = \{ a\in A : X(a)\in S \}

$$

In words: **all inputs whose outputs land inside $S$.**

```
  Domain A                 Range B
  --------                 -------
   a1 ──X──►  3
   a2 ──X──►  7
   a3 ──X──►  3

  If S = {3}, then X^{-1}(S) = {a1, a3}
```

### Worked coin table (memorize this pattern)

Let $\Omega=\{H,T\}$, $X(H)=1$, $X(T)=0$, fair coin $P(\{H\})=P(\{T\})=\tfrac12$.


| Region$S$       | Preimage$X^{-1}(S)$ | $P\big(X^{-1}(S)\big)$ |
| ----------------- | --------------------- | ------------------------ |
| $(-\infty,-1]$  | $\emptyset$         | $0$                    |
| $(-\infty,0]$   | $\{T\}$             | $1/2$                  |
| $(-\infty,0.5]$ | $\{T\}$             | $1/2$                  |
| $(-\infty,1]$   | $\{H,T\}$           | $1$                    |

**That table is the skeleton of a CDF.**
The lecture formula is exactly: size the preimage under $P$.

Common confusion:

- **Forward:** $\omega \mapsto X(\omega)$ (outcome → number)
- **Backward:** region $S \mapsto X^{-1}(S)$ (region → event on $\Omega$)

This video lives mostly on the **backward** step.

---

## 3. Half-lines, open/closed ends, and “cumulative”

<a id="p2-half-lines"></a>

### Inequalities as sets


| English         | Set of reals   |
| ----------------- | ---------------- |
| numbers ≤$x$   | $(-\infty, x]$ |
| numbers <$x$    | $(-\infty, x)$ |
| numbers ≥$x$   | $[x, \infty)$  |
| only exactly$x$ | $\{x\}$        |

Bracket tips:

- `]` or `[` → **include** endpoint (**closed**)
- `)` or `(` → **exclude** endpoint (**open**)

Standard **CDF** uses $(-\infty, x]$ — **closed at $x$** (includes $X\le x$).

```
  number line:  ... ========]x
                 (-∞, x]
```

### Cumulative

**Cumulative** = “everything up to here,” not “only this exact value.”

- Cumulative: “What fraction scored **at most** 70?”
- Point: “What fraction scored **exactly** 70?”

Half-lines generate the **Borel** “nice” regions on $\mathbb{R}$ used with real-valued RVs.
For this class, “Borel” ≈ *standard nice regions built from intervals/half-lines (and products in higher dimensions)*. You do not need full σ-algebra theory to follow the video.

---

## 4. Vectors $\mathbb{R}^{d}$ and the random variable map

<a id="p0-vectors"></a>


### Real line and vectors

**Reals** $\mathbb{R}$: the usual number line.
A **vector** in $\mathbb{R}^{d}$ is an ordered list of $d$ reals.

```
  d=1:  3.2
  d=2:  (3.2, -1.0)      point in the plane
  d large: stacked image pixels → one long list in R^d
```

An X-ray after stacking is **one point** in a huge $\mathbb{R}^{d}$.

### Random variable

$$
X : \Omega \to \mathbb{R}
\quad\text{or}\quad
X : \Omega \to \mathbb{R}^{d}

$$

```
  Ω  --X-->  R or R^d
```


| Piece                | Random?   | Why                              |
| ---------------------- | ----------- | ---------------------------------- |
| Which$\omega$ occurs | uncertain | experiment                       |
| The rule$X$          | **no**    | same$\omega$ → same $X(\omega)$ |

Better mental name than “random variable”: **outcome-to-number map**.

**Vector-valued RV (plain English):** $X$ is a function whose outputs are vectors in $\mathbb{R}^{d}$.

**Measurable (course voice):** for half-lines / product boxes, the preimage must be an **event** we may size with $P$. That is what “legal random variable” means here.

```
  abstract outcome ω ∈ Ω
        │
        │ X
        ▼
  numeric data x = X(ω) ∈ R^d
```

---

## 5. The probability triplet $(\Omega, F, P)$ and estimation

<a id="p4-measure"></a>

### Triplet


| Piece               | Symbol   | Plain job                                                     |
| --------------------- | ---------- | --------------------------------------------------------------- |
| Sample space        | $\Omega$ | all admitted outcomes                                         |
| Events              | $F$      | subsets of$\Omega$ we are allowed to size                     |
| Probability measure | $P$      | $P:F\to[0,1]$ with rules like $P(\Omega)=1$, $P(\emptyset)=0$ |

```
  Abstract side                 Numeric side
  -------------                 ------------
  (Ω, F, P)                     x ∈ R^d  (data)
       \                         /
        \____ X connects them ___/
```

### $P$ sizes **sets**

```
  P(A) = size of event A ⊂ Ω
```

**Micro (fair die):**

- $A=\{2,4,6\}$ (even) → $P(A)=\tfrac12$
- $B=\{6\}$ → $P(B)=\tfrac16$
- $P(\Omega)=1$

$P$ is a **sizing rule for events**, not primarily “a height at a point.”

### Who assigns $P$? Who estimates it?

In the video:

- **Nature / a labeler** (doctor intuition, spam judgment) is treated as assigning the “true” measure.
- **You as engineer** observe samples from the range of $X$ and must **estimate** that measure / distribution.

That estimation job is the heart of the course’s ML story.

---

## 6. Density vs measure (the trap this video attacks)

<a id="p5-density-vs-measure"></a>

Imagine a smooth hill over the number line (a “density curve”).


| Object                        | Picture                                | Probability of an event? |
| ------------------------------- | ---------------------------------------- | -------------------------- |
| **Density height** at one $x$ | how tall the hill is                   | **No** (in general)      |
| **Area** over an interval     | size of a region                       | **Yes-like** (measure)   |
| **CDF / distribution**        | sizes cumulative regions via preimages | **Yes — measure-like**  |

```
  height at one point  ≠  probability of an event
  area over a set      ≈  probability of that set
```

### Why students get this wrong

In coin/die problems, $P(X=1)$ looks like “probability at a value.”In continuous models, copying that blindly is dangerous. The safe object is always:

> probability of a **set / region** (half-line or product box).

Lecture hard line: “plug $x$ into the density and call it probability” is **wrong**.

### Name bank (same family in this lecture)

- distribution function
- probability distribution function
- cumulative distribution function (**CDF**)
- **pushforward measure** ($P$ moved through $X$)

---

## 7. Cartesian products, joints, and two experiments

<a id="p6-cartesian"></a>
<a id="p7-two-experiments"></a>

### Cartesian product

$$
A\times B = \{ (u,v) : u\in A,\ v\in B \}

$$

**2D CDF-style region:** $(-\infty,x_1]\times(-\infty,x_2]$

```
       x2
        |     (x1,x2)
        |____·
        |////|   shaded = left of x1 AND below x2
        |////|
        +----------- x1
```

The corner $(x_1,x_2)$ **names** the region; probability is for the **whole shaded box**.

In $d$ dimensions:

$$
(-\infty,x_1]\times\cdots\times(-\infty,x_d]

$$

### Two experiments (image vs disease)

```
  RE_image   →  image vector
  RE_disease →  label 0/1
```

as **two** experiments (two $\Omega$s / two RVs), possibly correlated
**or**

```
  one RE → pair (image, label)
```

as **one** experiment with a vector outcome.

**You choose.** That is modeling, not physics law.

### Joint / conditional / correlation (names)


| Word            | Plain sense                                                 |
| ----------------- | ------------------------------------------------------------- |
| **Joint**       | sizes involving two (or more) RVs together                  |
| **Conditional** | re-size after restricting to an event (“given disease…”) |
| **Correlation** | how two numeric RVs tend to move together                   |

Preview only — full multi-RV formulas come later.

### Unification sketch

Think of a big vector:

```
  ( pixel features… , disease label )
```

Then “discriminative vs generative” can start to look like different questions about pieces of the **same** vector-valued RV (preview in the lecture, not finished theory).

---

## 8. Why this lecture exists (bridge formula + paper check)

<a id="p8-why"></a>

### Story so far


| Lecture           | Installs                                                       |
| ------------------- | ---------------------------------------------------------------- |
| Lec 02            | RE →$\Omega$ → events $F$ → measure $P$                     |
| Lec 03            | RV$X:\Omega\to\mathbb{R}^{d}$ so data sit in range of $X$      |
| **Lec 04 (this)** | Move$P$ onto the range: distribution / CDF / pushforward $P_X$ |

### The formulas to own before NOTES

**Scalar RV:**

$$
P_X(x) = P\big(X^{-1}((-\infty,x])\big) = P(\{\omega : X(\omega)\le x\})

$$

**Vector RV** at $x=(x_1,\ldots,x_d)$:

$$
P_X(x) = P\Big(X^{-1}\big((-\infty,x_1]\times\cdots\times(-\infty,x_d]\big)\Big)

$$

In words:

1. Draw a cumulative region on $\mathbb{R}$ / $\mathbb{R}^{d}$.
2. Pull it back to an event on $\Omega$ with $X^{-1}$.
3. Apply original $P$.
4. Call the result the distribution (pushforward) of $X$.

```
  (Ω, F, P)  --X-->  (R^d, Borel regions, P_X)

  After sensors, daily talk often stays on the right
  (vectors + P_X), with Ω backstage.
```

### Fair-coin check (same as idea 2, closed form)

$X(H)=1$, $X(T)=0$, fair:

$$
P_X(-1)=0,\quad P_X(0)=\tfrac12,\quad P_X(0.5)=\tfrac12,\quad P_X(1)=1

$$

### Optional paper check (2 minutes)

Without looking back:

1. Write $X^{-1}((-\infty,0])$ for $H\mapsto 1$, $T\mapsto 0$.
2. If fair, what is $P_X(0)$?
3. Shade $(-\infty,2]\times(-\infty,3]$ in the plane.
4. One sentence: why density height is not a probability.
5. Name the three pieces of the probability triplet.

If those five are easy, go to NOTES.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).
Quiz later: [quiz.html](./quiz.html) Part A = this file.
Prior video package: [Lec 03 Part 2](../04-Lec03-Recap-Probability-Theory-Part2/NOTES.md).
