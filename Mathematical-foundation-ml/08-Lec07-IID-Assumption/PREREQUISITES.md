# Prerequisites — warm-up before Lec 07 (IID assumption)

> **Do this first** if “IID,” “independent,” or “identically distributed” still blur together.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on [Lec 06](../07-Lec06-XRay-Sample-From-Distribution/PREREQUISITES.md) (dataset $\sim P_{X,Y}$).  
> Still a **warm-up** — detailed enough if you do not know the basics.  
> **Goal:** unlock every map word (identical, independent, product joint, sampling, train=test distribution, given D estimate P).

```
  After this warm-up you can say:

  "IID = Independently and Identically Distributed (two separate claims)."
  "Identically distributed means the same P for every data point (conditions not shifting)."
  "Statistically independent events: P(A∩B)=P(A)P(B) — product, not sum."
  "IID independence is across observations (patients/images), not across pixels inside one image."
  "Sampling = many trials of the random experiment; tilde means drawn from."
  "Train and test are assumed to share the same underlying distribution."
  "All ML here: given a dataset D, estimate the unknown distribution P."
```

**Spell the acronym once (do not skip):**

| Letter chunk | Meaning |
|--------------|---------|
| **I**ndependently | across **data points** (product joint of the $n$ rows) |
| **I**dentically | same distribution $P$ for every row |
| **D**istributed | each row is a draw from a probability law |

Those are **two** assumptions glued into one word. You can break one without the other (same $P$ but dependent rows; independent rows from drifting machines).

**How to use:** top to bottom; every **Micro** on paper; read **Purpose for the video**.

**Warm-up → lecture boxes**

```
  §1  Same distribution (identical)     ──► Topic 1
  §2  Event independence (product)      ──► Topic 2
  §3  Points vs dimensions (critical)   ──► Topic 3
  §4  Two views of a dataset            ──► Topic 4
  §5  Sampling + tilde                  ──► Topic 4
  §6  Same P for train & test           ──► Topic 5
  §7  Supervised names are packaging    ──► Topic 6
  §8  Given D, estimate P               ──► Topics 6–7
```

---

## 1. Identically distributed (same $P$ for every row)

<a id="p1-identical"></a>

### Purpose for the video

The first half of **IID** is **I**dentically **D**istributed: every data point is treated as coming from the **same** underlying distribution.

### Plain idea

If $X_1$ and $X_2$ are random variables with **the same** distribution $P$, you can think:

- “I looked at the same experiment’s measurement rule twice,” or  
- “I have two different maps that happen to share the same pushforward law.”

For a dataset of $n$ points: **identical** means all $n$ share one $P$ — the **experimental conditions are not changing** mid-dataset (same hospital protocol story, not mixed mystery sources unless you *design* them as one).

### Micro

Coin always fair → every toss identically distributed.  
Coin that becomes biased after toss 50 → **not** identically distributed across the 100 tosses.

### Analogy — purpose

A factory scale calibrated the same way all week.  
**What fails if Tuesday’s scale is broken?** Tuesday’s weights are not from the same measurement law as Monday’s — identical distribution broke.

**Why the video needs this:** Topic 1 defines “identical”; Topic 2 shows lung X-ray vs brain MRI as a break.

---

## 2. Statistical independence of events (product rule)

<a id="p2-indep-events"></a>

### Purpose for the video

The second half of **IID** is **I**ndependent. Start with **events**, then lift to RVs.

### Definition (events)

Events $A,B$ are **statistically independent** when

$$
P(A\cap B)=P(A)\,P(B)
$$

**Product**, not sum. (Sum would be a different rule for other situations.)

### Definition (random variables)

RVs are statistically independent when their **joint** factors as the **product of marginals**:

$$
p(x,y)=p(x)\,p(y)
$$

For **$n$ data points** under the multi-point independence story:

$$
p(z_1,z_2,\ldots,z_n)=p(z_1)\,p(z_2)\cdots p(z_n)
$$

This is a **technical** term. It is related to everyday “independence,” but do not replace the formula with English vibes alone.

### Product, not sum (classroom trap)

In class he asks: intersection probability equals **sum** or **product**?  
Answer: **product**. Sum is the wrong operator for independence.

```
  WRONG instinct:  P(A and B)  ≈  P(A) + P(B)
  RIGHT definition: P(A∩B) = P(A) P(B)   when independent
```

### Micro (die)

$A=\{\text{even}\}$, $B=\{\text{prime faces}\}=\{2,3,5\}$ on a fair die.  
Check whether $P(A\cap B)=P(A)P(B)$ — independence is a **property to verify**, not a feeling.

### Analogy — purpose

Two light switches that do **not** share wiring: knowing one is on does not change the chance the other is on.  
If they share a breaker, independence fails.

**Why the video needs this:** Topic 2 defines independence so the joint of $n$ data points can factor as a product.

---

## 3. Critical: independence of **points**, not of **pixels**

<a id="p3-points-vs-dims"></a>

### Purpose for the video

This is the **#1 confusion** in the lecture.  
IID independence is **not** “every coordinate inside one vector is independent of every other coordinate.”

### Two axes (see the grid)

Think of the dataset as an $n\times d$ table:

```
           feat 1   feat 2  …  feat d
  point 1    •        •          •      ← one image / one patient
  point 2    •        •          •
   …
  point n    •        •          •

  IID independence  =  across ROWS (point 1 ⊥ point 2 ⊥ …)
  NOT claimed by IID = across COLUMNS inside one row (pixels may depend)
```

If you flatten every cell into $n\cdot d$ separate scalars, independence in IID still groups by the **$n$ row-blocks** (vector RVs), not cell-by-cell across all columns.

**IID says:** data point $i$ is independent of data point $j$.  
**IID does not say:** pixel 3 is independent of pixel 4 inside the same image.

Some algorithms *add* a feature-independence assumption (columns independent). That is **extra**, not part of vanilla IID. Real images almost always have dependent pixels.

### Analogy — purpose

**Patients vs organs:**  
“Independence across patients” ≠ “your left lung intensity is independent of your right lung intensity in the same scan.”

**Why the video needs this:** Topic 3 exists almost entirely for this kill shot.

**Trap:** “IID means features are independent.” **False** (unless you add that assumption separately).

---

## 4. Two views of the same dataset

<a id="p4-two-views"></a>

### Purpose for the video

The same list of $n$ points can be told two stories.

| View | Story | Need full “IID” words? |
|------|--------|-------------------------|
| **A** | $n$ **realizations** of **one** RV | “Identically distributed” is **redundant** (only one RV). Multi-RV independence language is not the natural packaging. |
| **B** | **One** realization of **$n$** RVs that share the same $P$ and are independent | **Yes** — identical **and** independent (this is where “IID” earns both letters) |

Both describe the usual dataset picture; textbooks often write the multi-RV / product form (View B).

**Logic check:** independence is a relation **between** random variables. With only one RV and many trials, you do not say “the RV is independent of itself.” You just say “$n$ trials.”

### Analogy — purpose

**One die rolled 100 times** vs **100 dice rolled once each with the same fairness.**  
Same spreadsheet of numbers; slightly different English packaging. The multi-die story needs “all fair and independent.” The one-die story does not need to chant “identically distributed” — there is only one die.

**Why the video needs this:** Topic 4 makes the two viewpoints explicit and corrects when independence language applies.

---

## 5. Sampling and the tilde $\sim$

<a id="p5-sampling"></a>

### Purpose for the video

Notation $D\sim P$ is not decoration.

### Meanings

| Phrase | Meaning |
|--------|---------|
| **Sampling** | Run the random experiment **multiple times** (multiple trials) |
| **Tilde $\sim$** | “sampled / drawn from” |
| $D=\{z_1,\ldots,z_n\}\sim P$ | $n$ outcomes produced under law $P$ |

Coin tossed $n$ times → $n$ observations from the coin’s distribution.  
That distribution is what later algorithms try to **estimate**.

### Analogy — purpose

Pulling $n$ tickets from a lottery drum is sampling.  
The drum’s mix is $P$. Estimating $P$ from the tickets is the learning problem.

**Why the video needs this:** Topics 4–7 rest on “dataset sampled from unknown $P$.”

---

## 6. Train / test and the same distribution

<a id="p6-train-test"></a>

### Purpose for the video

Once you build an algorithm assuming data $\sim P$, **new** points you care about should also come from **that same** $P$ (by construction of the theory).

### Plain idea

- **Training data:** points used to **estimate / fit** the model (to learn about $P$).  
- **Test data:** **new** held-out points used to check whether predictions still work.  

Because algorithms are built under “everything $\sim P$,” test points should also come from **that same** $P$. If they come from a **different** experiment (other hospital, other modality), the identical-distribution assumption **breaks** and failure is **expected**, not surprising.

This is **not** the same claim as “test points must equal train points pixelwise.” Same **law**, new **draws**.

### Design choice

Are “faces” and “buildings” one RE or two?  
You **choose** the packaging. Train on faces only → do not expect magic on buildings. Merge both into one dataset deliberately → one bigger experiment (then both live under one assumed $P$).

Student worry (preview): “If test must match train’s hospital, do we overfit that hospital?” Fair question — the lecture **defers** the full generalization answer to later in the course.

### Analogy — purpose

Study only **coastal weather**, then “test” on **desert weather** — you violated same-$P$, not the calculator.

**Why the video needs this:** Topic 5; hospital generalization is flagged as deferred.

---

## 7. Supervised / unsupervised / self-supervised as names

<a id="p7-supervised-names"></a>

### Purpose for the video

He refuses to start ML from a sacred trichotomy. Names are **packaging**.

| Name | Rough packaging |
|------|------------------|
| **Supervised** | pairs $(x,y)$ — “label” exists |
| **Unsupervised** | only $x$ (still a distribution to estimate) |
| **Self-supervised** | invent $y$ from parts of $x$ (e.g. mask pixels → inpainting) |

Mathematically you can always fold $y$ into a bigger vector and talk about **one** RV.  
Why split $X$ and $Y$? History + intuition: sensors vs human judgment; or convenience for conditionals $P(Y\mid X)$.

### Analogy — purpose

A form with two columns vs one long row of fields.  
Same information can be stored either way; names change how people talk, not the existence of a distribution.

**Why the video needs this:** Topic 6.

---

## 8. All machine learning: given $D$, estimate $P$

<a id="p8-given-d-estimate-p"></a>

### Purpose for the video

The slogan that closes the probability arc:

$$
\text{Given dataset } D \text{ sampled from unknown } P,\quad \text{estimate } P
$$

(or estimate pieces: joints, conditionals, margins) and optionally **learn to sample** (generative).

### Flavors (preview)

| Estimate… | Name people use |
|-----------|-----------------|
| $P(Y\mid X)$, $Y$ discrete | **classification** (a **classifier**) |
| $P(Y\mid X)$, $Y\in\mathbb{R}^{k}$ | **regression** (a **regressor**) |
| $P(X\mid Y)$ | conditional data models |
| $P_{X,Y}$ (+ learn to sample) | **generative** modeling |

**Discriminative vs generative (one line each):**

- **Discriminative (as he uses it):** focus on estimating the pieces you need for decisions (often conditionals like $P(Y\mid X)$).  
- **Generative:** estimate the underlying law **and** learn to **sample** new points as if from that law.

### Labels can be synthetic

Class tags are often **human divisions**, not physics (spam definitions change; man/woman tags are social, not Newtonian). Nature can be “simple” (few parameters to fly an aircraft) while semantics need huge models. Models can even fit **random** labels — so “good label” is not a pure math object. Engineers still solve the problem they are given.

### Notation preview (course habit)

| Style | Used for |
|-------|----------|
| Lowercase $x_1,\ldots,x_d$ | scalar coordinates / dimensions of a vector |
| Uppercase $X$ | vector-valued random variable (default in this course) |

### Analogy — purpose

You are handed a bag of tickets from an **unknown** drum mix.  
**Job:** figure out the mix (and maybe draw new tickets like the drum would).  
That is “given $D$, estimate $P$.”

**Why the video needs this:** Topics 6–7 formal definition; next lectures start algorithms.

---

### Paper check (before NOTES)

1. Expand **IID** letter by letter in your own words.  
2. State “identically distributed” in one sentence.  
3. Write the event independence formula — product or sum?  
4. Does IID require independent pixels? Yes/no + why. Sketch the $n\times d$ grid.  
5. Name the two views of $n$ data points. When is “identical” redundant?  
6. What does sampling mean operationally? What does $\sim$ mean?  
7. Why might train-on-faces / test-on-buildings fail?  
8. One-line ML problem? Discriminative vs generative in one sentence each?

---

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).  
Prior: [Lec 06](../07-Lec06-XRay-Sample-From-Distribution/NOTES.md).
