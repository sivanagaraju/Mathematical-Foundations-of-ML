# Prerequisites — warm-up before Lec 02 (Generative Models: Problem Formulation)

> **Do this first** if “random variable,” “R^d,” “distribution,” or “sample from a model” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Course: NPTEL **Mathematical Foundations of Generative AI** · follows [Lec 01 Introduction](../14-Lec01-MFGAI-Introduction/NOTES.md).  
> **Beginner deep warm-up:** definition · worked micro · analogy · notice · mini-check for each idea.

```
  After this warm-up you can say:

  "Nature is modeled by a probability triplet (Ω, F, P); we rarely see Ω."
  "A random variable X maps abstract outcomes to numbers/vectors in R^d."
  "An image, word, or speech chunk is a high-dimensional vector (data in R^d)."
  "Data points live in the range of X; the law on R^d is written p_X or p_x."
  "If I knew p_X fully, I could answer uncertainty questions about the system."
  "ML is given a dataset D of vectors; the core job is estimate unknown p_x."
  "Generative modeling also requires learning to sample (simulate new outcomes)."
  "Training assumes a model p_θ and minimizes a divergence d(p_x, p_θ)."
```

**Warm-up → lecture boxes**

```
  §1  Probability triplet (light)           ──► Topics 1, 4–6
  §2  Random variable X: Ω → R^d            ──► Topics 1, 4
  §3  Vectors & dimension R^d               ──► Topics 2–3
  §4  Range of a function / of X            ──► Topic 4
  §5  Distribution p_X (working object)     ──► Topics 1, 5, 7
  §6  Dataset D; realizations; ~ p_x        ──► Topics 6–7
  §7  Sampling vs estimating                ──► Topic 8
  §8  Model p_θ + divergence + training     ──► Topics 9–10
```

---

## 1. Probability triplet (light reload)

<a id="p1-triplet"></a>

### Purpose for the video

Lec 01 built $(\Omega,\mathcal{F},P)$. Lec 02 keeps saying: we **do not have access** to that triplet in practice.

### Definitions

| Object | Plain meaning |
|--------|----------------|
| **Random experiment (RE)** | A procedure with uncertain outcome (coin toss, camera click, die roll) |
| **Sample space $\Omega$** | Set of all possible outcomes |
| **Event space $\mathcal{F}$** | Allowed subsets of $\Omega$ we may assign probability to |
| **Probability measure $P$** | Function that sizes events with $P(\Omega)=1$, non-negative, additive on disjoint unions |

### Worked micro

Coin: $\Omega=\{H,T\}$, $P(\{H\})=P(\{T\})=\tfrac12$.  
You never “see” a pure abstract $\Omega$ when you train an image model — you see **pixel vectors**.

### Analogy — hidden stage play

The play’s full script of possible endings is $\Omega$. Allowed plot questions are events. $P$ rates how likely each plot twist is. The audience (you) only sees **what the camera recorded** later — that camera is the random variable.

### Second analogy — sealed lottery machine

Balls inside the machine are outcomes in $\Omega$. You never open the machine; you only see a digital readout when a ball is drawn. Training an image model is like studying thousands of readouts without inventorying every possible ball.

### ASCII — what you cannot touch vs what you work with

```
  HIDDEN (nature)              ACCESSIBLE (practice)
  ─────────────────            ─────────────────────
  RE → Ω, F, P        ──X──►   usually not given as objects
         │
         │  X (sensor / RV)
         ▼
  measurements in R^d          ← dataset lives here
  law p_X on R^d               ← what we estimate
```

### Notice

- $P$ lives on **events** (subsets of $\Omega$), not on “floats” until we introduce $X$.  
- Generative modeling’s dream of “new images” is secretly: re-run the RE without the real stage.

### Mini-check

1. Name the three pieces of the probability triplet.  
2. Why does “we do not have access to $\Omega$” matter for ML?

---

## 2. Random variable $X:\Omega\to\mathbb{R}^d$

<a id="p2-rv"></a>

### Purpose for the video

The course’s central bridge: **function** that turns abstract outcomes into **measurable vectors**.

### Definitions

| Idea | Meaning |
|------|---------|
| **Function** $f:A\to B$ | Each $a\in A$ maps to exactly one $f(a)\in B$ |
| **Random variable** $X$ | Function $X:\Omega\to\mathbb{R}^d$ (here $d$ can be huge) |
| **Domain** | $\Omega$ (abstract outcomes) |
| **Codomain / range space** | $\mathbb{R}^d$ (numbers we store) |

### Worked micro

Die face $\omega\in\{1,\ldots,6\}$. Let $X(\omega)=\omega$ (scalar, $d=1$).  
Photo RE: $\omega$ is “what was in front of the camera.” $X(\omega)$ is a $100\times200$ image stacked into a vector of length $20{,}000$.

### Analogy — barcode scanner

Items on a shelf are abstract outcomes. The scanner prints a **numeric barcode string**. You stock the warehouse by barcodes, not by metaphysical “item essences.” $X$ is the scanner.

### Notice

- Randomness is *which* $\omega$ occurs; $X$ only **labels** it with numbers.  
- Once $X$ is fixed, we can talk about the **law of $X$** on $\mathbb{R}^d$.

### Mini-check

1. Domain of $X$? Codomain?  
2. Is “a random number with no domain” a correct definition of RV?

---

## 3. Vectors and $\mathbb{R}^d$ (Euclidean space)

<a id="p3-rd"></a>

### Purpose for the video

He writes **$\mathbb{R}^d$** constantly (“R D” in speech). You must picture **one point with $d$ coordinates**.

### Definitions

| Symbol | Meaning |
|--------|---------|
| $\mathbb{R}$ | real line |
| $\mathbb{R}^2$ | plane (pairs $(x,y)$) |
| $\mathbb{R}^d$ | $d$-tuples $(x_1,\ldots,x_d)$; $d$ can be $20{,}000$ |
| **Vector / point** | one member of $\mathbb{R}^d$ |

### Worked micro — image stacking

Image $100\times 200$ pixels, each a number $0$–$255$.

- Grid has $100\cdot 200=20{,}000$ numbers.  
- Stack row-by-row into one long list $\Rightarrow$ **one vector in $\mathbb{R}^{20000}$**.  
- Two different photos = two different points in that huge space.

**Stacking sketch**

```
  row1: p1 p2 … p200
  row2: …
  …
  row100: …
       │  concatenate
       ▼
  [p1, p2, …, p20000]  ∈  R^{20000}
```

### Analogy — flat-pack furniture list

A wardrobe has many parts. You list every screw count and panel size in one long **bill of materials**. That list is the vector. The assembled wardrobe is the “image” you look at; the list is what the math sees.

### Second analogy — GPS coordinates

$(2,3)$ is a point in $\mathbb{R}^2$. Imagine $20{,}000$ axes instead of two — still “one address,” just absurdly high-dimensional.

### Third analogy — spreadsheet row

Each photo is one row of a giant spreadsheet with $20{,}000$ columns (pixel 1 … pixel 20000). Training is learning the statistics of those rows.

### Notice

- Dimensionality $d$ depends on **how you represent** data (resolution, vocab size, window length).  
- Algorithms in this course are mostly **data-type agnostic**: they see vectors, not “images” specially.  
- Order of stacking is a convention — what matters is **one fixed map** from grid → long vector.

### Mini-check

1. $50\times 50$ grayscale image → dimension of the vector?  
2. Why is $\mathbb{R}^3$ still “just vectors” like $\mathbb{R}^{20000}$?  
3. If you reverse the stacking order every time, does $\mathbb{R}^d$ still make sense as a consistent space?

---

## 4. Range of a function (and of $X$)

<a id="p4-range"></a>

### Purpose for the video

Slogan: **every data point is a member of the range space of the random variable.**

### Definitions

| Idea | Meaning |
|------|---------|
| **Range (image) of $f$** | set of all values $f$ actually hits: $\{f(a):a\in A\}$ |
| **Range space of $X$** | the $\mathbb{R}^d$ side (or the set of attainable measurements) |
| **Data point $x$** | one realized measurement $x=X(\omega)$ for some (unknown) $\omega$ |

### Worked micro

$X$ maps coin outcomes to $\{0,1\}$. Range of $X$ is $\{0,1\}$. Seeing data `1,0,1,1` means you only saw range members — never “pure $\omega$ labels.”

### Analogy — passport photos

People (abstract $\Omega$) walk into a booth. The machine outputs passport photos ($\mathbb{R}^d$ pixels). Your dataset is a pile of photos = pile of **range values**. You never store the soul of the person.

### Notice

- Saying “$x\in\mathbb{R}^d$ is data” **implies** (in this worldview) an underlying $\Omega$ and $X$ you may never write down.  
- That implication is how probability re-enters machine learning.

### Mini-check

1. If every data vector lives in the range of $X$, what must exist underneath?  
2. Do we need to name $\Omega$ explicitly to store a dataset?

---

## 5. Probability distribution $p_X$ (the working object)

<a id="p5-px"></a>

### Purpose for the video

Because $(\Omega,\mathcal{F},P)$ is inaccessible, we work with a **probability distribution function** on the range side — written $p_X$ or $p_x$ in the lecture.

### Definitions (course spirit)

| Object | Role |
|--------|------|
| $P$ on $\Omega$ | true abstract measure (inaccessible) |
| $p_X$ (or $p_x$) on $\mathbb{R}^d$ | law of the measurements; what data “come from” |
| **CDF / density / mass** | concrete representations of that law (details in tutorials) |

Lecture claim (philosophical): **if you know the probability measure / distribution fully, you can answer every uncertainty question** about the system (coin sequences, image likelihoods, etc.).

### Worked micro

Fair coin: knowing $P(H)=\tfrac12$ lets you compute $P(\text{HHTH})$ by independence.  
Unknown coin: you only have flips; you must **estimate** the law from flips.

### Analogy — weather climate file

If you own the complete climate probability model for a city, you can answer “chance of 10 rainy days in a row.” If you only have 30 days of logs, you estimate that climate file from logs — that estimate is the ML job.

### Notice

- Estimating $p_x$ is treated as **equivalent** (for the course’s purposes) to understanding the underlying uncertainty once we work on the range side.  
- Notation: $p_x$ means the unknown true distribution of data vectors $x$.

### Mini-check

1. Why shift the goal from “estimate $P$ on $\Omega$” to “estimate $p_x$ on $\mathbb{R}^d$”?  
2. What questions become answerable if $p_x$ is known?

---

## 6. Dataset, realizations, and $x_i\sim p_x$

<a id="p6-dataset"></a>

### Purpose for the video

Starting point of ML: **given data**. Colloquial: “data is the new oil.”

### Definitions

| Symbol | Meaning |
|--------|---------|
| **Dataset** $D=\{x_1,\ldots,x_n\}$ | $n$ observed vectors in $\mathbb{R}^d$ |
| **Realization** | one concrete outcome of $X$ (one sample) |
| $x_i\sim p_x$ (lecture shorthand) | each $x_i$ is drawn according to unknown law $p_x$ |
| **IID** (when assumed) | independent and identically distributed draws |

### Worked micro

$n=3$ images, each reshaped to length $d$.  
$D=\{x_1,x_2,x_3\}\subset\mathbb{R}^d$.  
Story: three times someone ran the “take a photo” experiment; $X$ turned each outcome into pixels.

### Analogy — oil barrels

Crude oil is raw material for fuel. **Data barrels** $x_1,\ldots,x_n$ are raw material for learning $p_x$. Empty talk about “AI” without $D$ is a refinery with no oil.

### Second analogy — repeated lab trials

Same protocol, $n$ runs, $n$ measurement vectors. The protocol is the RE; measurements are realizations of $X$.

### Notice

- Seeing $D$ **implicitly assumes** some $\Omega$ and $P$ underneath — even if never named.  
- Notation $D\sim p_x$ is shorthand for “these points are samples from $p_x$.”

### Mini-check

1. Write $D$ in set notation.  
2. What does $x_i\sim p_x$ claim about how data were generated?

---

## 7. Sampling vs estimating

<a id="p7-sample"></a>

### Purpose for the video

**Estimate** $p_x$ is the ML core. **Sample** from it is the extra GenAI demand.

### Definitions

| Verb | Meaning in this course |
|------|-------------------------|
| **Estimate $p_x$** | recover a usable description of the unknown law from $D$ |
| **Sample** | produce new $x$ as if you re-ran the random experiment / drew from the law |
| **Simulate the RE** | run a synthetic coin toss, die roll, or “new image” without real $\Omega$ |

### Worked micro

Coin: estimate $P(H)$ from 100 flips → estimation.  
Then generate 100 synthetic flips with a computer → sampling.  
Image GenAI: fit a model of face photos, then draw a **new** face vector never in $D$.

### Analogy — weather generator

Estimating climate from historical days ≠ generating a **plausible synthetic year**. Generative AI wants both: understand the climate **and** invent new days.

### Notice

- Classification might stop at estimating $p(y\mid x)$ without “new X-rays from scratch.”  
- Generative modeling explicitly wants **sampling**.

### Mini-check

1. Can you sample without estimating anything? (practical GenAI view in lecture: need estimated law)  
2. Translate “LLM generates text” into sample/estimate language.

---

## 8. Model $p_\theta$, divergence, and training

<a id="p8-recipe"></a>

### Purpose for the video

General engineering recipe for “fit unknown $p_x$.”

### Definitions

| Piece | Meaning |
|-------|---------|
| **Model / parametric family** $p_\theta$ | assumed form for the unknown law; $\theta$ = knobs (weights, slope, …) |
| **Divergence** $d(p_x,p_\theta)$ | score of how far two distributions are (KL, JS, $f$-divergences, …) |
| **Training** | find $\theta^\star=\arg\min_\theta d(p_x,p_\theta)$ |

### Worked micro — curve fitting analogy (lecture)

Points in the plane. Assume “line” $\Rightarrow$ estimate slope & intercept.  
Assume “circle” $\Rightarrow$ estimate center & radius.  
Same pattern for distributions: assume a family $p_\theta$, then fit $\theta$.

### Worked micro — numbers

If $d(p_x,p_{\theta_1})=0.8$ and $d(p_x,p_{\theta_2})=0.2$, training prefers $\theta_2$ (smaller divergence).

### Recipe as three boxes (memorize)

```
  (1) MODEL     assume family p_θ     (θ = knobs)
  (2) SCORE     pick divergence d(p_x, p_θ)
  (3) TRAIN     θ* = argmin_θ d(p_x, p_θ)
       │
       └─► (GenAI) SAMPLE from p_θ*
```

### Analogy — thermostat vs outdoor climate

Outdoor climate is $p_x$ (unknown). Thermostat settings are $\theta$ defining a comfort model $p_\theta$. Divergence asks how far comfort is from climate. Training twists knobs to minimize that gap.

### Second analogy — high-school curve fit

Scatter of points on paper. Assume “line” → fit slope and intercept. Assume “circle” → fit center and radius. Same spirit for distributions: assume a **shape family** $p_\theta$, then fit $\theta$ by a distance score between laws.

### Notice

- Today $p_\theta$ is often a **neural network** (universal approximation intuition: enough parameters can approximate rich functions).  
- You usually **cannot plug true $p_x$ into $d$** — algorithms must work from **samples** $D$ (open issue for later lectures).  
- After fitting, **sample from $p_{\theta^\star}$** to generate new data.  
- Named scores you will meet: KL, Jensen–Shannon, $f$-divergences.

### Mini-check

1. What is a “model” in “large language model” per this lecture?  
2. Write the training optimization in symbols.  
3. Why is “just minimize $d$ with the true $p_x$ written on the board” not available in real training?

---

### Paper check (end-to-end)

1. $X$ maps _____ → _____.  
2. A $100\times200$ image is a vector in $\mathbb{R}^{_____}$.  
3. Dataset $D$ is a set of _____ from unknown _____.  
4. ML core goal: estimate _____.  
5. GenAI adds: learn to _____.  
6. Recipe: assume _____; minimize _____ between $p_x$ and $p_\theta$.

**Peek:** (1) $\Omega\to\mathbb{R}^d$ (2) $20000$ (3) vectors / $p_x$ (4) $p_x$ (5) sample (6) $p_\theta$ / divergence $d$.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file · Part B = NOTES.
