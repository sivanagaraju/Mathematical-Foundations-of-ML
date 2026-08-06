# Lec 08 — Distribution Estimation

**Video:** [Lec 08 Distribution Estimation](https://www.youtube.com/watch?v=aYb8KG9JYsg) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 07 — IID Assumption](../08-Lec07-IID-Assumption/NOTES.md)

---

## Table of Contents

1. [Topic 1 — Context, dataset raw material, range points](#topic-1-context-dataset-raw-material-range-points-0000–0429) (00:00–04:29)
2. [Topic 2 — Worldview and IID recap](#topic-2-worldview-and-iid-recap-0429–0849) (04:29–08:49)
3. [Topic 3 — Foundational problem: estimate P and/or sample](#topic-3-foundational-problem-estimate-p-andor-sample-0849–1157) (08:49–11:57)
4. [Topic 4 — What to estimate: moments, P(Y|X), class/reg](#topic-4-what-to-estimate-moments-pyx-classreg-1157–1642) (11:57–16:42)
5. [Topic 5 — Use cases P(Y), P(X), P(X|Y); inpainting](#topic-5-use-cases-py-px-pxy-inpainting-1642–2101) (16:42–21:01)
6. [Topic 6 — Supervised vs unsupervised packaging](#topic-6-supervised-vs-unsupervised-packaging-2101–2446) (21:01–24:46)
7. [Topic 7 — Notation, P(Y)≠P(X|Y), densities next](#topic-7-notation-py-pxy-densities-next-2446–2846) (24:46–28:46)
8. [External references](#external-references)
9. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: state the **foundational machine learning problem** as distribution estimation from samples. Method: start from $D\sim_{\mathrm{iid}}$ unknown $P$, choose a target (full $P$, moment, joint, margin, or conditional), and name algorithms by what they estimate. Fork: **estimating** $P$ is not the same as **sampling** from $P$; this course emphasizes estimation.

**Worldview arc:** from probability + IID dataset story **to** a catalog of estimation targets + packaging names; densities next.

### System context

```
  ╔══════════════════════════════════╗
  ║ Outside                          ║
  ║ · Lec 02–07: Ω, RV, joints, IID  ║
  ║ · densities (next)               ║
  ║ · generative sampling (later)    ║
  ╚══════════════╤═══════════════════╝
                 │ this lecture
                 ▼
        ┌────────────────────┐
        │ Given D, estimate P│
        └────────────────────┘
```

### Main blueprint

```
  D ~_iid unknown P
       │
       ▼
  Estimate P  (+ optional: sample from P)
       │
       ├─ full law
       ├─ moments / expectations
       └─ P(Y|X) · P(X) · P(Y) · P(X|Y) · P(X,Y)
            │
            ├─ Y discrete → classification
            └─ Y continuous → regression
       │
       ▼
  Names: supervised / unsupervised (packaging)
  Disc: estimate without sampling focus
  Gen: learn to sample
  Next: density functions
```

### Scenario walkthrough

1. Hold $n$ labeled X-rays as $D\sim_{\mathrm{iid}} P$.  
2. Disease given image → estimate $P(Y\mid X)$.  
3. Prevalence → estimate $P(Y)$.  
4. Pixel appearance → $P(X)$; diseased appearance → $P(X\mid Y)$.  
5. Inpainting → $P(Y\mid X)$ with $Y$ carved from pixels of $X$.  
6. Call it supervised if you use a label column; still the same math job.  
7. Next: densities as the working objects of continuous estimation.

### Failure / contrast path

```
  Confuse P(Y) with P(X|Y)                 ──X──► wrong population question
  Think estimating P_X automatically samples ──X──► miss generative skill
  Treat sup/unsup as deep ontology           ──X──► miss packaging
  Leave conditioner random in P(x|y)         ──X──► broken conditional
```

### STOP / out of scope

Full ERM derivation (teased), full density theory (next), generative sampler course (later), exhaustive algorithm list.

### Load-bearing claims (closed-book)

- $D\sim_{\mathrm{iid}}$ unknown $P$; data = range points.  
- Foundational problem: **estimate $P$** (and optionally sample).  
- Discriminative ≈ estimate without sampling focus; generative ≈ sample.  
- Targets: full $P$, moments, joints/margins/conditionals.  
- $P(Y\mid X)$ for class/reg; $P(Y)$ prevalence; $P(X\mid Y)$ data given label.  
- Supervised/unsupervised = packaging.  
- Conditioner fixed; densities next.

**Speaker / course:** NPTEL IISc · MFML · Lec 08.

---

## Topic 1: Context, dataset raw material, range points (00:00–04:29)

### Where this sits on the master map

**SETUP** box — open every later estimation target from the same raw material $D$. Warm-up: [dataset](./PREREQUISITES.md#p1-dataset).

### Board / screenshot

![Dataset setup](./screenshots/composites/ch01-topic-01-setup-panel1of1.png)

**Figure — ~00:00–04:20:** course context; ERM teaser; $D=\{(x_i,y_i)\}$; features/labels as range points.

### What he is establishing

Welcome back. We continue from setting the context for the **core problem of machine learning** and the foundations of probability (RVs, distributions, conditionals). Today: formulate that foundational problem, and mention **empirical risk minimization (ERM)** as one specialized way to realize distribution-estimation problems — paving the way for algorithms later.

Recall the given — write it as a formula and read every piece:

$$
D=\{(x_i,y_i)\}_{i=1}^{n}
\sim_{\mathrm{iid}}
p_{X,Y}
\quad\text{(unknown)}
$$

| Piece | Meaning |
|-------|---------|
| $D$ | finite dataset you hold |
| $(x_i,y_i)$ | $i$-th feature vector and label |
| $x_i\in\mathbb{R}^{d}$ | $d$-dimensional range point (stacked image / features) |
| $y_i\in\mathbb{R}^{k}$ or discrete | continuous target or category tags |
| $\sim_{\mathrm{iid}}$ | sampled independently from the same law |
| $p_{X,Y}$ unknown | joint law of $(X,Y)$ is **not** given; only $D$ is |

The label **may not exist** (then $D=\{x_i\}$ and the law is $p_X$). Nomenclature (varies by literature): $x$ = input / features / data; $y$ = label space / output. For this course they are **elements of the range** of underlying random variables defined on **one** sample space. Both $X$ and $Y$ are RVs when labels exist.

The raw material of ML is this dataset — “**data is the new oil**.” Algorithms are only as good as these points.

If you start algorithms without this opening scene, later math floats. Instead: oil = $D$; unknown drum = $P$.

We now have the setup. Still missing: the fixed worldview + precise IID meaning again.

### Analogy for this topic only

A refinery needs crude oil before any process.

- Tanker 1: 100 barrels  
- Tanker 2: 500 barrels  
- No tanker: the plant sits idle  

**What is ML’s crude oil here?** The finite sample $D$, not a free gift of $P$. If you wait for nature to hand you the true distribution instead of samples, you never start the plant.

In lecture words: dataset of range points iid from unknown $p_{xy}$.

### Local picture

```
  D = {(x1,y1),…,(xn,yn)}  ~_iid  p_{X,Y}  (unknown)
       │
       │  ~_iid  =  same law + product joint over i
       ▼
  xi ∈ R^d     yi ∈ R^k or {0,1,…}
  X,Y : Ω → ranges   (one sample space)

  data = oil / raw material
  ERM = one specialized path (teased; algorithms later)
```

**Notice:** labels optional; worldview still holds for $x$-only data.

### Bridge

How do the two dataset views and mathematical IID sit under every later problem?

---

## Topic 2: Worldview and IID recap (04:29–08:49)

### Where this sits on the master map

**WORLDVIEW + IID**. Warm-up: [dataset](./PREREQUISITES.md#p1-dataset); Lec 07 IID.

### Board / screenshot

![Worldview and IID](./screenshots/composites/ch02-topic-02-worldview-iid-panel1of1.png)

**Figure — ~04:29–08:40:** two views; pushforward unknown; identity/independence as math; design choice.

### What he is establishing

Two views again: $n$ realizations of one RV, or one realization of $n$ statistically independent identically distributed RVs. When he says “dataset,” hear range points from one underlying RV story (or $n$ iid RVs), sample space with measure, **pushforward** to a distribution we **do not know**. That is the **worldview** for every problem.

Student Q on “identical.” Identity and independence are **mathematical constructs**, not English.

**Identity:** if you view the $n$ points as $n$ RVs $Z_1,\ldots,Z_n$, they share the **same** distribution — same functional form of the underlying measure / same $P$.

**Independence:** the joint factors as a **product of marginals** (product, not sum). For densities of the $n$ points:

$$
p(z_1,z_2,\ldots,z_n)=p(z_1)\,p(z_2)\cdots p(z_n)
$$

That is **all** the math claims. Rough workplace metaphors (similar scanners, independent patients) are only weak translations.

Rough intuition: one image independent of another; similar device/population for X-rays — but the jump from math to intuition is **weak**. You *can* package X-rays and faces as one distribution (valid design choice). Algorithms **use** the iid properties. Designer question: can this dataset be **fairly** represented as iid? If not, do not use algorithms that need it. Non-iid methods exist.

Starting point restated: $D$ sampled iid from unknown $P$; range points of a RV with pushed-forward measure.

If you treat iid as casual English, you will misuse algorithms. Instead: math definitions + design honesty.

We now have the worldview locked. Still missing: the one-line problem statement.

### Analogy for this topic only

Two legal contracts in a loan file: “identical distribution” and “independent.”

- Clause 1: same interest law for every borrower row  
- Clause 2: rows do not secretly couple  

**Are these poetry or clauses?** Clauses — product joint and same law. Workplace metaphors (“similar scanners”) are rough translations; the binding text is the math.

In lecture words: math identity/independence; design whether iid is fair.

### Local picture

```
  Worldview:
    Ω + P  --X,Y-->  unknown distribution
    D = range samples (iid)

  Identity  = same law for each of n RVs
  Independence = joint product of marginals
  Design: is iid fair for this D?
```

**Notice:** packaging X-ray+faces as one $P$ is allowed mathematically; usefulness is separate.

### Bridge

What exactly is the foundational problem we solve with this $D$?

---

## Topic 3: Foundational problem — estimate $P$ and/or sample (08:49–11:57)

### Where this sits on the master map

**CORE PROBLEM**. Warm-up: [estimate P](./PREREQUISITES.md#p2-estimate-p), [disc vs gen](./PREREQUISITES.md#p6-disc-gen).

### Board / screenshot

![Estimate P and sample](./screenshots/composites/ch03-topic-03-core-problem-panel1of1.png)

**Figure — ~08:49–11:50:** estimate distribution; sample; discriminative vs generative; this course focus.

### What he is establishing

The **foundational / fundamental problem of machine learning**, as formulas:

$$
\boxed{\text{Given } D\sim_{\mathrm{iid}} P \text{ (unknown), estimate } P}
$$

and, as a second objective,

$$
\text{learn to sample } z_{\mathrm{new}}\sim \hat{P}
$$

Concrete scene: 10,000 chest X-rays with disease bits arrive as $D$. You do not receive $P$ on a USB stick. You must recover (pieces of) the unknown law that could have produced those pairs — and maybe later draw new synthetic pairs.

Methods that estimate the underlying distribution but **do not bother about sampling** are typically **discriminative** models. Methods **designed to sample** are **generative** models (so-called genAI). Slight variations exist; intuition later.

All algorithms in **this course** mostly concern **estimation** (first part). Generative sampling is for the **next course**. Question in the room: which distributions do we estimate? Next.

If you only memorize model names without “estimate $P$,” you miss the spine. Instead: samples in → estimated law out (+ optional sampler).

We now have the core problem. Still missing: full $P$ vs moments vs which conditional.

### Analogy for this topic only

You hear music through a wall.

- Task A: rebuild the score from what you heard → **estimate**  
- Task B: build a piano that improvises new songs in that style → **sample**  

**Can you do B without any idea of the score’s law?** Not really. This course is mostly score recovery; the generative course builds the piano.

In lecture words: given $D$, estimate $P$; generative adds sampling.

### Local picture

```
  Given D ~_iid P unknown

  Job A:  produce hat{P}  (estimate)     ← this course (mostly)
  Job B:  draw z_new ~ hat{P} (sample) ← generative / next course

  Discriminative: Job A without Job B focus
  Generative: designed for Job B (usually with estimation too)
```

**Notice:** writing down $\hat{P}_{X,Y}$ is not yet a machine that draws new images.

### Bridge

Do we always need the entire distribution, or is a mean enough — and which conditional for disease?

---

## Topic 4: What to estimate — moments, $P(Y\mid X)$, class/reg (11:57–16:42)

### Where this sits on the master map

**TARGETS**. Warm-up: [moments](./PREREQUISITES.md#p4-moments), [match question](./PREREQUISITES.md#p5-match-question).

### Board / screenshot

![Estimation targets](./screenshots/composites/ch04-topic-04-targets-panel1of1.png)

**Figure — ~11:57–16:30:** moments; $P(Y\mid X)$; classification/regression; joints/margins.

### What he is establishing

People categorize algorithms by **which distribution** you estimate. Sometimes you estimate not the full distribution but a **function** of it — **moments**.

**Moment / expectation idea (discrete):**

$$
\mathbb{E}[Z]=\sum_z z\,P(Z=z)
$$

English: probability-weighted average of the values $Z$ can take. The **second moment** $\mathbb{E}[Z^2]$ uses $z^2$ instead of $z$. Moment-generating functions are another “function of $P$” family (name-drop). Given $D$, estimating a first/second moment is **lighter** than recovering entire $P$. Or recover the **entire** distribution.

**Conditional target** (main example):

$$
P(Y\mid X=x)
\quad\text{or discrete}\quad
P(Y=y\mid X=x)=\frac{P(X=x,Y=y)}{P(X=x)}
$$

Distributions are measures; interpret as **likelihoods**. So $P(Y\mid X=x)$ = likelihood of labels when the data point is fixed at $x$. Disease classification: given image $x$, category $y$. **Bounding-box regression:** still the same conditional shape with continuous

$$
Y=(\text{center},\,\text{height},\,\text{width})\in\mathbb{R}^{3}
$$

(or four numbers for a box — he uses the three-number story). You estimate $P(Y\mid X=x)$: given this image, the law of box coordinates.

Problems estimating $P(Y\mid X)$ are often called **classification** when $Y$ is discrete and **regression** when $Y\in\mathbb{R}^{k}$. Boundaries are thin; he is not a big fan of hard walls. You can also estimate marginals/joints:

$$
P_X,\qquad P_Y,\qquad P_{X,Y}
$$

If you always estimate the wrong object, the use case fails. Instead: pick the target matching the ask.

We now have the target catalog starting. Still missing: concrete disease use cases and $P(Y)$ vs $P(X\mid Y)$.

### Analogy for this topic only

A photo + sticky note.

- Photo A, note “disease”  
- Photo B, note “healthy”  
- Photo C, box around a spot (three numbers)  

**What do you want from the archive?** “What note given this photo?” → $P(Y\mid X)$. “Only average brightness of all photos?” → a **moment**, not the full law. Wrong move: always demand the entire joint when a mean answers the business question.

In lecture words: full $P$ or moments; $P(Y\mid X)$ for class/reg.

### Local picture

```
  Given D ~ P_{X,Y}

  Estimate full P
     or  E[Z] = Σ z P(Z=z)     (moment)
     or  P(Y=y|X=x)=P(X=x,Y=y)/P(X=x)
            → class if Y discrete
            → reg if Y = (center,h,w) ∈ R^3  etc.
     or  P(X), P(Y), P(X,Y)
```

**Notice:** box regression is still $P(Y\mid X)$ with vector $Y$, not a different probability universe.

### Bridge

Walk the 10k-image disease archive: which $P$ for which question?

---

## Topic 5: Use cases $P(Y)$, $P(X)$, $P(X\mid Y)$; inpainting (16:42–21:01)

### Where this sits on the master map

**USE CASES**. Warm-up: [match question](./PREREQUISITES.md#p5-match-question).

### Board / screenshot

![Disease use cases](./screenshots/composites/ch05-topic-05-use-cases-panel1of1.png)

**Figure — ~16:42–21:00:** generative note; P(Y|X), P(Y), P(X), P(X|Y); inpainting.

### What he is establishing

Why estimate $P(Y\mid X)$? Given image, want label. Why estimate $P_X$, $P_Y$, $P_{X,Y}$? Often generative interest — you want to **sample**. But estimating margins/joints is **not** the same as generative modeling: generative algorithms **implicitly sample**. You can still estimate margins/joints without sampling.

Concrete: 10,000 images, binary disease labels $Y\in\{0,1\}$. Write the four targets as formulas:

| Use case | Formula | Read carefully |
|----------|---------|----------------|
| Disease for **this** image $x$ | $P(Y\mid X=x)$ | $X$ fixed at this scan |
| Prevalence in population | $P(Y=1)$ | **no** $X$ in the formula |
| Pixel appearance (ignore disease) | $P_X$ or $p(x)$ | law of images alone |
| Pixels **given** disease | $P(X\mid Y=1)$ | $Y$ fixed at disease; law of $X$ |

Practitioner skill: cast the real-world problem into “which formula?” Labels are pseudo: **inpainting** (masked pixels) still fits

$$
P(Y\mid X=x)
$$

where now $Y$ is the **missing coordinates** of the same image and $x$ is the observed (unmasked) part.

If you answer prevalence with $P(X\mid Y)$, you mixed questions. Instead: match carefully.

We now can map questions to targets. Still missing: supervised/unsupervised names.

### Analogy for this topic only

Hospital archive, four questions, four doors:

1. This scan → disease?  
2. How common is disease?  
3. What do scans look like?  
4. What do **diseased** scans look like?  

**Can one door answer all four?** No — different $P$ objects.

In lecture words: $P(Y\mid X)$, $P(Y)$, $P(X)$, $P(X\mid Y)$; inpainting as conditional.

### Local picture

```
  10k (image, disease bit)

  P(Y|X)  disease | image
  P(Y)    prevalence (no image)
  P(X)    pixel law (no disease)
  P(X|Y)  pixels | disease fixed

  Inpainting: Y = missing pixels of X → still P(Y|X)
```

**Notice:** generative sampling is extra beyond estimating the joint.

### Bridge

Why do people still say supervised vs unsupervised if labels are just another RV?

---

## Topic 6: Supervised vs unsupervised packaging (21:01–24:46)

### Where this sits on the master map

**NOMENCLATURE**. Warm-up: [sup/unsup](./PREREQUISITES.md#p7-sup-unsup).

### Board / screenshot

![Supervised unsupervised](./screenshots/composites/ch06-topic-06-nomenclature-panel1of1.png)

**Figure — ~21:01–24:40:** supervised/unsupervised definitions; math indifference; course problem restated.

### What he is establishing

**Supervised** algorithms use labels; **unsupervised** do not. Conventional story: labels from humans (clinician cost); unsupervised cheaper. Mathematician: **don’t care** — two RVs on same $\Omega$. If a $d$-vector is $d$ scalars on one space, nothing stops adding a human-assigned label as another RV. He is uncomfortable with rigid sup/unsup ontology but teaches the words so graduates know the lingo.

Course problem restated: given samples sampled iid from unknown distribution, **estimate** that distribution (joint, margin, conditional, or function/moment). Call it supervised if one dimension is named “label.”

Playful note: “sample” as **noun** (a data point) vs **verb** (to draw). Same root, two roles.

If you think unsupervised is a different universe of math, you miss unity. Instead: packaging over the same estimation problem.

We now have names demystified. Still missing: class/reg recap, $P(Y)$ vs $P(X\mid Y)$, density teaser.

### Analogy for this topic only

Calling a form field “label” vs “feature 17.”

- Form with sticky “disease” column → people say supervised  
- Form with only pixel columns → people say unsupervised  
- Form where you hide three pixels and predict them → self-supervised packaging  

**Does renaming change $(\Omega,F,P)$ and the RVs?** No — only the story you tell coworkers. The math job remains estimate a law from samples.

In lecture words: supervised/unsupervised = packaging; estimate $P$ either way.

### Local picture

```
  Supervised:   use y column
  Unsupervised: no y column
  Math: same Ω, RVs, estimate P

  Course: D ~_iid unknown P → estimate P
          (joints / cond / margins / moments)
```

**Notice:** he will still mix “supervised” algorithms in the course without caring philosophically.

### Bridge

Nail $P(Y)$ vs $P(X\mid Y)$ and conditional notation; densities next.

---

## Topic 7: Notation, $P(Y)\neq P(X\mid Y)$, densities next (24:46–28:46)

### Where this sits on the master map

**NOTATION + STOP**. Warm-up: [conditional fixed](./PREREQUISITES.md#p8-conditional-fixed).

### Board / screenshot

![Conditional notation and densities next](./screenshots/composites/ch07-topic-07-notation-stop-panel1of1.png)

**Figure — ~24:46–28:46:** class/reg; P(Y) vs P(X|Y); fixed conditioner; density functions next.

### What he is establishing

If estimating the conditional of the label given data: $Y$ **discrete** → **classification**; $Y$ continuous / $\mathbb{R}^{k}$ → **regression**.

Student confuses $P(Y)$ with something image-related. Write them side by side:

$$
P(Y=1)
\quad\text{vs}\quad
P(X\in A\mid Y=1)
$$

- $P(Y=1)$: chance of disease in the population — **no image** appears.  
- $P(X\in A\mid Y=1)$: chance the image lands in region $A$ **given** disease — only diseased cases shape this law.  

They are **not** the same. ($P(X\mid Y)$ is also **not** $P(Y\mid X)$ — different variable is fixed.)

**How to read conditional notation on the board.** He writes $P(X\mid Y)$ but means:

$$
\text{evaluate at } x \text{ with } Y \text{ fixed at } y
$$

| Symbol | Role |
|--------|------|
| $x$ | evaluation point (where you read the distribution of $X$) |
| $y$ | **fixed** value of the conditioner |
| “$X$ given $Y$” | names the two RVs |

The conditioned variable is always fixed; change $y$ → a **different** conditional function. That is why conditionals are “functions of the conditioner.”

**Notation for the course ahead.** Script $\mathcal{P}$ often denotes **distribution functions** (set-sizing measures / CDFs). ML practice mostly works with **density functions** $p(x)$ (continuous heights). Densities are introduced next lecture. Remember: a density **height** at a point is not itself the probability of that point; probabilities come from integrating the density over regions.

If you leave conditioner free, conditionals collapse. Instead: fix $y$, evaluate in $x$; next, densities.

We now close the formulation lecture. Still missing next: density theory for continuous estimation.

### Analogy for this topic only

Two different hospital stats:

- **How many patients have disease?** → $P(Y)$  
- **What do diseased X-rays look like?** → $P(X\mid Y=\text{disease})$  

**Same archive, different questions.** Mixing them is the classic trap.

In lecture words: $P(Y)\neq P(X\mid Y)$; conditioner fixed; densities next.

### Local picture

```
  Classification: Y discrete, estimate P(Y|X)
  Regression:     Y in R^k, estimate P(Y|X)

  P(Y)     = population label law (no X)
  P(X|Y=y) = data law with Y fixed at y

  Notation: evaluate at x | Y := y
  Next: probability density functions
```

**Notice:** density height is still not event probability — that care continues next time.

### Bridge

Introduce PDFs so continuous estimation has the right objects.

---

## External references

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| [StatQuest — Maximum Likelihood](https://www.youtube.com/watch?v=XepXtl9YKwc) | Topics 3–4 | Fit a distribution to samples |
| [3Blue1Brown — Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) | Topics 5,7 | Conditioning intuition |
| [Google ML Crash Course — Framing](https://developers.google.com/machine-learning/crash-course/framing/ml-problem-framing) | Topics 5–6 | Cast real problems as ML targets |
| [Lec 07 package (IID + estimate P)](../08-Lec07-IID-Assumption/NOTES.md) | Topics 1–3 | Immediate prior |
| [Seeing Theory — Conditional Probability](https://seeingtheory.brown.edu/basic-probability/index.html#section2) | Topic 7 | Fix conditioner visually |
| [StatQuest — The Main Ideas of Fitting a Line](https://www.youtube.com/watch?v=PaFPbb66DxQ) | Topic 4 | Regression as estimation energy |

---

## Sources

- Video: [Lec 08 Distribution Estimation](https://www.youtube.com/watch?v=aYb8KG9JYsg)  
- Channel: NPTEL — Indian Institute of Science, Bengaluru  
- Skill: `youtube-lecture-tutor`  
- Captions cleaned via timed transcript / claim sheets  
