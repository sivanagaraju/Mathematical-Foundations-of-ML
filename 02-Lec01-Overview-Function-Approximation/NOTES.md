# Lec 01 — Overview of function approximation

> **Video:** [Lec 01 Overview of Function Approximation](https://www.youtube.com/watch?v=G2h7nD_Stxg)
> **~48 min** · chalk board · NPTEL / IISc · Prof. Prathosh A. P.
> **Warm-up:** [PREREQUISITES.md](./PREREQUISITES.md) · **Quiz:** [quiz.html](./quiz.html) · **Transcript:** [TRANSCRIPT.md](./TRANSCRIPT.md)

**How to read this:** follow each topic top to bottom like a story. The middle block (“What he is establishing”) is the real lesson — written so you can understand **without** rewatching. Screenshots are optional board check. Analogies are short confirmations, not a second course.


| Stuck on…          | Open                                            |
| --------------------- | ------------------------------------------------- |
| sets / domain-range | [p1-sets](./PREREQUISITES.md#p1-sets)           |
| function            | [p2-functions](./PREREQUISITES.md#p2-functions) |
| data pairs$D$       | [p3-data](./PREREQUISITES.md#p3-data)           |
| image → vector     | [p4-vectors](./PREREQUISITES.md#p4-vectors)     |
| class / model       | [p5-models](./PREREQUISITES.md#p5-models)       |
| ill-posed           | [p6-illposed](./PREREQUISITES.md#p6-illposed)   |

---

## Table of Contents

1. [Executive Summary](#executive-summary--what-this-lecture-does)
2. [Topic 1: Why this course exists — math as a precision language (00:00–06:30)](#topic-1-why-this-course-exists--math-as-a-precision-language-00000630)
3. [Topic 2: The core problem — given $D$, find $f$ (06:30–12:00)](#topic-2-the-core-problem--given-d-find-f-06301200)
4. [Topic 3: Why you cannot just memorize — ill-posed / overfitting (12:00–15:00)](#topic-3-why-you-cannot-just-memorize--ill-posed--overfitting-12001500)
5. [Topic 4: Guess, model, algorithm — how learning works (15:00–20:00)](#topic-4-guess-model-algorithm--how-learning-works-15002000)
6. [Topic 5: Why probability; X-rays as vectors in $\mathbb{R}^d$ (20:00–30:00)](#topic-5-why-probability-x-rays-as-vectors-in-mathbbrd-20003000)
7. [Topic 6: Why semantic concepts are hard — light vs disease, speech physics (30:00–42:00)](#topic-6-why-semantic-concepts-are-hard--light-vs-disease-speech-physics-30004200)
8. [Topic 7: Statistical methods and what to review next (42:00–47:49)](#topic-7-statistical-methods-and-what-to-review-next-42004749)
9. [External references](#external-references)
10. [Sources](#sources)

---

## Executive Summary — what this lecture does

### In one breath

This lecture answers: **What is machine learning, in math language, before we touch fancy algorithms?**

Almost every science problem is: there is an unknown rule $f$ from inputs to outputs; we only see some examples $D$; we want a useful estimate of $f$ so we can predict on **new** inputs. That job is called **function approximation**.

Because English words like “the model learns” are fuzzy, the course uses **mathematics as a precision language**. Course 1 = foundations (especially **probability**). Course 2 = modern algorithms. Goal: you should read ML papers like a novel, not like an “internet data scientist” who only clicks tools.

How learning is described on the board (this is the sentence to remember):

> **Given a dataset $D$, find the unknown function $f$.**
> We start with an **initial guess** for $f$ — that estimate is called a **model**.
> We then run a **procedure** that improves the guess using $D$ — that procedure is an **algorithm** (e.g. error backpropagation for neural nets).

If $f$ is totally unknown, many curves fit the same points → problem is **blind / ill-posed**. Memorizing training points fails on new $x$ (**overfitting** preview). So we guess a **class** of shapes (ellipse, parabola, neural nets…), then refine with data. *All models are wrong, but some are useful.*

**Two viewpoints (the lecture’s big arc):**


| Deterministic (~first half)   | Probabilistic (course engine)                     |
| ------------------------------- | --------------------------------------------------- |
| $X,Y$ sets; unknown map $f$   | $X,Y$ as **random variables**                     |
| Given$D$, estimate $f$        | Estimate**distributions** (parallels to FA)       |
| Model + algorithm on data     | Repeated observations +$P$ language               |
| Fine for simple physical laws | Needed for**semantic** labels + generative models |

**Semantic gap:** light/pressure are physics; disease/gender/phones are human constructs — complicated, non-trivial, no short $F{=}ma$. That is why probability is required.

### Architecture map

```
  COURSE MISSION
  Math = precision language · foundations · not only tool use
           │
           ▼
  DETERMINISTIC SETUP
  Given D → estimate f
  model = guess · algorithm = refine with D
           │
           ├─ OK: physical maps (Newton-like, few parameters)
           │
           └─ BREAKS: semantic gap (light ↛ disease/gender)
           │
           ▼
  PROBABILISTIC ENGINE
  RVs · distribution estimation · one-to-one parallels
  review: experiment, Ω, RV, P, distribution, density
```

**Own after one read:** FA sentence; deterministic vs probabilistic; semantic gap; vector stacking; review list + FA↔distribution bridge.

---

## Topic 1: Why this course exists — math as a precision language (00:00–06:30)

### Where this sits on the master map

Top box: **COURSE MISSION**. No formulas yet — only why math, and who this course is for. If “function” is already shaky: [functions](./PREREQUISITES.md#p2-functions).

### Board / screenshot

![Mission 1](./screenshots/composites/ch01-course-mission-math-language-panel1of2.png)

![Mission 2](./screenshots/composites/ch01-course-mission-math-language-panel2of2.png)

**On the board:** two-course series; probabilistic foundations; chalk-and-talk; “internet data scientist” vs engine builder.

### What he is establishing

Machines “thinking” raises ethics — this course sets that aside and focuses on **algorithms that make machines learn**.

This is **course one of two**. Here you get foundations (especially probabilistic). The next course covers state-of-the-art generative / ML models. Without the first-course tools (conditional expectation, laws of large numbers, transforms of random variables, …) you cannot honestly build systems like GPT or Gemini — even if those ideas felt pointless in a pure probability class.

**Main goal of this course (and the series):** separate you from an **“internet data scientist”** who only uses tools. Driving a car does not require knowing the internal combustion engine. **Building** a better engine does. By the end of both courses you should pick up SOTA ML papers and read them without fear — “like a novel.”

**Why math is better than normal language.** English (or any colloquial language) is fragile: one sentence, many readings. Engineers need **precision**. Mathematics is not magic; it is a **language designed for precision** so you can implement and check claims.

Teaching style: **board work**, almost no slides (LLMs can already make slide decks; live board shows mistakes and fixes). The course will look at ML algorithms mostly from a **probabilistic standpoint**.

**You can now:** state the course goal and why math, not English, is the working language.
**Still missing:** the actual technical problem of ML — next topic.

### Analogy for this topic only

App user vs engine designer: both can “use a car.” Only one can redesign the engine. This course is for engine designers. Math is the blueprint language, not a decoration.

### Local picture

```
  English: "the model learned"  ──many meanings──►  argue forever
  Math equations                ──one reading──►    implement / prove

  Course 1: foundations (probability lens)
       │
       ▼
  Course 2: modern algorithms / generative models
```

Notice: probability tools are not optional trivia — they are how modern generative systems are built.

### Bridge

Mission is clear. What is the single technical problem almost every science field already met?

---

## Topic 2: The core problem — given $D$, find $f$ (06:30–12:00)

### Where this sits on the master map

**PROBLEM** box. Warm-up: [sets](./PREREQUISITES.md#p1-sets), [functions](./PREREQUISITES.md#p2-functions), [data](./PREREQUISITES.md#p3-data).

### Board / screenshot

![Given D find f 1](./screenshots/composites/ch02-function-approx-formal-panel1of2.png)

![Given D find f 2](./screenshots/composites/ch02-function-approx-formal-panel2of2.png)

**On the board:** domain $\mathcal{X}$, range $\mathcal{Y}$, pairs, $D$, unknown $f$, slogan **given $D$, find $f$**; planet example (time → position).

### What he is establishing

Most problems in science and engineering can be seen as **function approximation**.

**Setup (plain English first).** There is a bag of allowed inputs (the **domain** $\mathcal{X}$) and a bag of possible outputs (the **range** $\mathcal{Y}$). There is a fixed rule $f$ that, for every allowed input, returns **exactly one** output:

$$
f : \mathcal{X} \to \mathcal{Y}.

$$

We do **not** know $f$. What we get is a finite list of observations — **data**:

$$
D = \{(x_1,y_1),\ (x_2,y_2),\ \ldots,\ (x_n,y_n)\}.

$$

Each pair means: when the input was $x_i$, we saw output $y_i$.

**The problem, written the way he writes it (~11:06):**

$$
\text{Given } D,\ \text{find } f.

$$

More carefully: find a **useful estimate** of $f$, because the true $f$ is never handed to us. Why care? **Prediction** — for a new $x$ not in $D$, we still want $y$.

**Classic example — planets.** Observe position at several times. Time plays the role of $x$; position plays the role of $y$. Predict positions at times you did **not** observe. Kepler, Gauss, and others lived this problem long before the phrase “machine learning.”

**Tiny numbers so it sticks:**

```
  time t   position (toy)
    1         ≈ 2
    2         ≈ 5
    3         ≈ 10
    4         ???     ← why “find f” is not “store the table”
```

**You can now:** write domain, range, $D$, $f$, and the one-line problem “given $D$, find $f$.”
**Still missing:** can we solve this when we know **nothing** about the shape of $f$?

### Analogy for this topic only

Three night sightings of a planet, then “where on night 10?” The true path is $f$; the notebook of sightings is $D$. Approximating $f$ means answering nights you never watched — not only re-reading the notebook.

### Local picture

```
  domain X                      range Y
     x  ─────── f (unknown) ──────►  y = f(x)

  We only see finite arrows:
     x1→y1 , x2→y2 , … , xn→yn     =  data D

  Goal: recover a useful f so NEW x* still gets y*
```

Notice: the entire ML problem, in first principles, is already on that picture.

### Bridge

Someone might say: “Just set the function equal to the training points.” That sounds clever — and it is the wrong idea.

---

## Topic 3: Why you cannot just memorize — ill-posed / overfitting (12:00–15:00)

### Where this sits on the master map

**OBSTACLE** box. Warm-up: [data](./PREREQUISITES.md#p3-data), [ill-posed](./PREREQUISITES.md#p6-illposed).

### Board / screenshot

![Ill-posed 1](./screenshots/composites/ch03-ill-posed-overfitting-panel1of2.png)

![Ill-posed 2](./screenshots/composites/ch03-ill-posed-overfitting-panel2of2.png)

**On the board:** blindness to $f$; table lookup fails for new $x$; need to start from a structured guess; “optimize error” needs precise definitions later.

### What he is establishing

With **no information about $f$**, only samples, the recovery problem is **ill-posed** or “blind.” The true time→planet map is unknown; you only have observations.

A student suggestion: define the function to equal $y_i$ at every observed $x_i$ and ignore everything else. That is a **table of training points**. It fails the purpose of estimating $f$: we need outputs for **$x$ we have not observed**. He flags this early as the seed of **overfitting**.

```
  same three training hits:

       y
       │        *
       │    *
       │ *
       └──────── x → new x
         many curves fit the dots and DISAGREE later
```

People also say “optimize the error.” Without a precise notion of error and optimization, the slogan is empty — this course exists to make those notions precise. Philosophically, when you are blind you still must **start from somewhere**: a guess about the structure of $f$.

**You can now:** explain why pure memorization is not “finding $f$,” and why the problem is blind without structure.
**Still missing:** the historical/practical strategy — guess a class, then refine.

### Analogy for this topic only

Memorizing every exam question you have already seen does not answer a **new** question. That is table lookup dressed as a function.

### Local picture

```
  BAD:  f_table(x_i)=y_i for training only  →  silent on new x
  NEED: a rule that still answers beyond D
```

Notice: formal overfitting theory comes later; the warning is already clear.

### Bridge

If table lookup is out, what did science actually do for centuries?

---

## Topic 4: Guess, model, algorithm — how learning works (15:00–20:00)

### Where this sits on the master map

**METHOD** box — the most important vocabulary block of the lecture (~11–20 min philosophy). Warm-up: [models](./PREREQUISITES.md#p5-models).

### Board / screenshot

![Model algorithm 1](./screenshots/composites/ch04-guess-class-models-data-panel1of2.png)

![Model algorithm 2](./screenshots/composites/ch04-guess-class-models-data-panel2of2.png)

**On the board:** guess class (parabola / ellipse); infinitely many members; model = estimate of $f$; “all models are wrong…”; refine using $D$; algorithms depend on model choice; probabilistic viewpoint introduced.

### What he is establishing

This is the heart of how he wants you to **speak** about machine learning.

#### Step 1 — Start with a guess on the form of $f$

Philosophically you must start somewhere. Example: assume the unknown curve is a **parabola**, or (Kepler) that planetary paths are **ellipses**.

Guessing “ellipse” does **not** finish the job. There are **infinitely many** ellipses. Remaining question: **which** ellipse fits the observations?

#### Step 2 — That guess / estimate is a **model**

Every estimate you make for $f$ is called a **model** (~16:19). Famous line of the modeling community:

> **All models are wrong, but some are useful.**

Meaning: you never access the true generating $f$; every estimate is approximate. Usefulness is about prediction / decision value, not “perfect truth.”

#### Step 3 — Refine the guess using data $D$

The observation set $D$ is what people call **data**. The process of improving the initial guess using $D$ is what “learning” means here. Quality of data will later determine quality of the estimate — “data is the new oil” is not empty hype in this framing.

#### Step 4 — The refinement procedure is an **algorithm**

There are many ways to refine. The **algorithm** is the concrete procedure that updates the model using $D$.

- If your model family is **neural nets**, one famous refinement algorithm is **error backpropagation**.
- If your model family is **linear** or **quadratic**, the refinement algorithm changes.

So: **model choice and algorithm choice are linked.** (~19:05–19:31)

#### The clean sentence (model + algorithm + $D$ + $f$)

> In machine learning, the problem is to find an unknown function $f$ given a dataset $D$.
> To do this we start with an initial guess for $f$ (a **model**) and use a specific procedure (an **algorithm**) to refine that guess using the data $D$.

#### Probabilistic viewpoint begins (~19:50)

He already said the course takes a **probabilistic standpoint**. Why? Much of classical “function learning” talk is incomplete for modern ML. In particular, a probabilistic framework is **broad enough** to cover both **discriminative** and **generative** formulations (terms defined later). Generative models especially sit naturally in probability language. That thread continues in Topic 5–7.

```
  unknown true f
        │
        │  observe D
        ▼
  choose a CLASS of candidates
  (ellipses / lines / neural nets / …)
        │
        │  ALGORITHM uses D to pick / refine
        ▼
  MODEL  f̂   (estimate — wrong but hopefully useful)
        │
        ▼
  predict on new x
```

**You can now:** say the given-$D$-find-$f$ sentence with the words **model** and **algorithm** used correctly; explain “all models are wrong…”.
**Still missing:** a concrete high-dimensional input (X-ray as a vector) and why semantics force probability.

### Analogy for this topic only

IKEA catalog = **class** of tables. Measuring your room and picking one SKU = **algorithm + data**. The table in your house = **model**. It is not “the true cosmic table,” but it can still be useful. Same spirit as “all models are wrong, but some are useful.”

### Local picture

```
  GIVEN D                 FIND f (estimate)
       │                        ▲
       │                        │
       └──── algorithm ─────────┘
              refines
              the model

  model  = current estimate of f
  algorithm = how you update that estimate using D
```

Notice: without separating model vs algorithm, “learning” stays fuzzy English again.

### Bridge

Time to make inputs concrete: an X-ray is not a single number — it is a long list of numbers (a vector). And labels like disease are not physics dials.

---

## Topic 5: Why probability; X-rays as vectors in $\mathbb{R}^d$ (20:00–30:00)

### Where this sits on the master map

Probabilistic course lens + first real high-dimensional example. Warm-up: [vectors](./PREREQUISITES.md#p4-vectors).

### Board / screenshot

![X-ray vector 1](./screenshots/composites/ch05-probabilistic-xray-vectors-panel1of3.png)

![X-ray vector 2](./screenshots/composites/ch05-probabilistic-xray-vectors-panel2of3.png)

![X-ray vector 3](./screenshots/composites/ch05-probabilistic-xray-vectors-panel3of3.png)

**On the board:** probabilistic vs only deterministic view; generative/discriminative breadth; X-ray setup; stacking image columns into a vector in $\mathbb{R}^d$; binary disease label; radiologist as the real-world $f$.

### What he is establishing

#### Why a probabilistic viewpoint?

Not every task needs it, but **some things will not fit** if you refuse probability. Generative models are naturally probabilistic. More broadly, the probabilistic framework can **encompass both discriminative and generative** formulations — it is the more general language for modern ML. That is why this foundations course is probabilistic.

#### Concrete problem: disease from X-rays

- Each input $x_i$ is an **X-ray image** (e.g. lungs).
- Each output $y_i \in \{0,1\}$ is disease vs not (assignment of 0/1 is conventional).
- Unknown $f$ maps image → label. In the real world that map is performed by a **radiologist**.
- Given thousands of labeled films (data $D$), estimate $f$ so a **new** film still gets a label.

#### How is an X-ray treated mathematically? (~21:58–26:15)

**Step A — digital image = matrix of numbers.**
Each pixel holds a brightness (classically something like 0–255). The picture is a table with $p$ rows and $q$ columns.

**Step B — stacking (vectorization) (~24:02–24:36).**
Take column 1 top→bottom, then column 2, then column 3, … and paste them into **one long column**. That long list is a **vector**.

```
  PICTURE (2 rows × 3 columns)          STACK COLUMNS

       c1   c2   c3                      take c1, then c2, then c3
     ┌────┬────┬────┐
  r1 │ 10 │ 30 │ 50 │                   10
     ├────┼────┼────┤                   20   ← finished col1
  r2 │ 20 │ 40 │ 60 │                   30
     └────┴────┴────┘                   40   ← finished col2
                                        50
                                        60   ← finished col3

  Result: one vector with 6 numbers

        x = (10, 20, 30, 40, 50, 60)
```

**Step C — dimensionality (~24:40–25:19).**
If the image has $p$ rows and $q$ columns, the vector has

$$
d = p \times q

$$

entries. He writes this as a point in **$\mathbb{R}^{pq}$** (same as $\mathbb{R}^d$ with $d=pq$).

#### Plain English: vector, Euclidean space, $\mathbb{R}^{pq}$


| Word                                    | Super simple meaning                                                                                            |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Vector**                              | A**shopping list of numbers** that describes one example (one image).                                           |
| **Euclidean space**                     | The normal flat geometry world where “distance” is the usual idea (graph paper, not a weird curved universe). |
| **$\mathbb{R}$**                        | All ordinary real numbers (…, −1, 0, 0.5, 2, …).                                                             |
| **$\mathbb{R}^2$**                      | Points need**2** numbers (a plane).                                                                             |
| **$\mathbb{R}^3$**                      | Points need**3** numbers (ordinary space).                                                                      |
| **$\mathbb{R}^d$ or $\mathbb{R}^{pq}$** | Points need**$d$** numbers; for an image $d = p\times q$.                                                       |

**How you specify Euclidean space:** just name the dimension.
“Work in $\mathbb{R}^{100}$” means every point is a list of 100 real numbers.

```
  d = 1     ℝ¹     a number line        •----•----•
  d = 2     ℝ²     a flat plane          y
                                         │  • (3,2)
                                         └──── x
  d = 3     ℝ³     room-like space       (you can still picture)
  d = 100   ℝ¹⁰⁰   cannot draw           but STILL "one point = one list"
  d = p·q   ℝ^{pq} image with p×q pixels treated as ONE point
```

**Why bother?** (~25:47–26:15) Once images, audio, brain signals, … are **points in multi-dimensional space**, the same math/stats toolbox can look for relationships: nearness, averages, maps $f:\mathbb{R}^{d}\to\{0,1\}$, later probability on those spaces.

A **scalar** input is only $d=1$. The formal problem never required $x$ to be one number.

**You can now:** draw stacking; say $x\in\mathbb{R}^{pq}$; explain Euclidean space without fear.
**Still missing:** why “disease / gender /ah/” are hard even after vectorization — the semantic gap.

### Analogy for this topic only

A photo is a spreadsheet of brightness. Stacking is reading the spreadsheet **column by column into one shopping list**. That list is the vector. $\mathbb{R}^{pq}$ is the huge map where each whole photo is a single pin.

**Question:** after stacking, is the computer “looking at a picture” or “holding one long list of numbers”?
Holding one long list — and that is enough to run math on it.

### Local picture

```
  X-ray matrix (pixels)  --stack columns-->  vector x ∈ ℝ^{pq}
                                                    │
                                                    │  f  (radiologist in real life)
                                                    ▼
                                               y ∈ {0,1}

  many labeled films = D
  learn f̂ so a NEW film (new point in ℝ^{pq}) still gets a label
```

Notice: speech later is also “a vector” (pressure samples over time) — same idea, different $d$.

### Bridge

A camera measures light. Why should that equal “diseased” or “female” or the abstract sound /ah/? That mismatch is the deep reason for statistics.

---

## Topic 6: Why semantic concepts are hard — light vs disease, speech physics (30:00–42:00)

### Where this sits on the master map

**WHY PROBABILITY / STATISTICS** — longest conceptual block. Warm-up: [vectors](./PREREQUISITES.md#p4-vectors).

### Board / screenshot

![Semantic 1](./screenshots/composites/ch06-abstract-labels-vs-physics-panel1of3.png)

![Semantic 2](./screenshots/composites/ch06-abstract-labels-vs-physics-panel2of3.png)

![Semantic 3](./screenshots/composites/ch06-abstract-labels-vs-physics-panel3of3.png)

**On the board:** reflectance/light vs disease/gender; microphone pressure vs phone; speech as waveguides (physics era); statistical speech recognition ~1980s; random-label experiment; coin physics hard; need repeated observations.

### What he is establishing

#### Deterministic viewpoint (recap of the first half)

In the **deterministic** framing (~6:43–18:19 spirit):

1. Domain $\mathcal{X}$ (inputs), range $\mathcal{Y}$ (outputs).
2. Assume unknown $f:\mathcal{X}\to\mathcal{Y}$ exists.
3. Given observations $D$, **estimate** $f$.
4. Strategy: **initial guess** (class) → that estimate is a **model** → **algorithm** refines the model using $D$.

That is the “ideal hard map” story: recover a function between two sets.

#### Why the deterministic viewpoint falls short (~29–41 min)

It works when inputs and outputs are **measurable physical quantities** with short laws — Newton’s $F=ma$ style systems with **few** parameters (aircraft example).

It struggles when targets are **abstract semantic / human constructs**:


| Raw measurable data           | Human label (semantic) |
| ------------------------------- | ------------------------ |
| Light reflectance on a sensor | disease? gender?       |
| Air pressure at a mic         | phone /ah/? intent?    |
| Pixel grid                    | digit class?           |

**Semantic gap (~29:30–31:30):** there is no built-in physics rule that turns a pattern of light into the *idea* “gender” or “diseased.” Those labels are **synthetic human categories**, layered on top of physics. Constructs are **inherently complicated and non-trivial** — so you cannot expect a small deterministic formula.

```
  DETERMINISTIC IDEAL                    SEMANTIC REALITY
  ───────────────────                    ────────────────
  x  ── short law ──► y                  light  ── ??? ──► "gender"
  (force → accel)                        pressure ── ??? ──► "/ah/"

  few parameters                         may need ~10^8 parameters
  first principles                       no clean first principles
```

#### What creates a model’s performance? (from this lecture’s logic)

Performance is **not** magic. Roughly it comes from:

1. **Choice of class / model family** (what shapes are allowed).
2. **Data $D$** (quality and amount — “data is the new oil”).
3. **Algorithm** that fits the model to $D$.
4. Whether the **task is physical or semantic** (semantic maps need more data/capacity).
5. Whether you only **fit training labels** or truly **generalize** (random-label warning below).

A flexible model can look “good” on training while learning nonsense — see next point.

#### Can machines learn random labels?

Yes, in the sense of **fitting**. On handwritten digits, if you **scramble labels randomly** and train, a rich model can still match the training table. That is **not** success: it shows capacity can memorize **any** labeling. True overfitting is when the model locks onto training quirks (even pure noise) and fails the real concept / new data. The lecture uses this as a caution: **fitting ≠ understanding**.

#### How history shaped modern ML (speech story)

**Old path (physics / waveguides):** model vocal tract as tubes; invert the system from pressure signal to constrictions → phone. Deterministic first-principles ambition.

**Modern path (~1980s stats → today):** treat signals as **vectors**, collect huge labeled sets, learn statistically. Physicists joked this was **“ignorance modeling.”** He accepts the joke as the truth of practice: if you do not know the full mechanism, **repeated observations + statistics** still build systems that work (speech, images, …).

#### “Ignorance modeling” as a useful term


| Phrase         | Meaning                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| Joke           | “You don’t know physics, so you just fit data.”                                                   |
| Useful reading | When the map is unknown/complex,**model the uncertainty with statistics** instead of fake certainty. |
| Coin           | Full physics would be deterministic; we use probability because we are**not** omniscient.            |
| Semantic ML    | Same: cannot write light→gender from first principles → probability/statistics.                    |

#### Microphone / waveguides (one more time, concrete)

```
  air pressure over time  →  numbers  →  vector
        │
        ├─ OLD: invert waveguide physics  →  phone
        └─ NEW: many (vector, label) pairs → statistical map → phone
```

#### Preview: random variables (full treatment next lectures)

He will treat inputs/outputs in a probabilistic frame as **random variables**. Important seed: an RV is a **deterministic function** from outcomes to numbers — the name is misleading; randomness sits in the experiment, not in the function object.

**You can now:** contrast deterministic vs probabilistic *motivation*; define the semantic gap; explain random labels, history, ignorance modeling.
**Still missing:** formal slogan + full review list + distribution-estimation parallel (Topic 7).

### Analogy for this topic only

A kitchen scale measures **mass**. “Is this gift thoughtful?” is not a scale reading.
Disease / gender / /ah/ are like “thoughtful?” — human meanings stuck on top of physical numbers.
So you do not invent F=ma for thoughtfulness; you collect many labeled examples and learn a statistical pattern.

**Question:** after you fit a model to random digit labels, did the machine “understand digits”?
No — it only matched the scrambled table. That is the overfitting warning in disguise.

### Local picture

```
  DETERMINISTIC PATH (good for physics)
    X,Y fixed sets · f unknown · estimate f from D
    works when y is a physical dial

  SEMANTIC GAP (breaks short laws)
    sensor numbers  ↛  human labels
    constructs complicated → huge models / lots of data

  HISTORICAL FIX
    physics inversion  ──►  statistical "ignorance modeling"
                              (modern ML default)
```

Notice: the course is not anti-physics; it is honest about **when** physics does not close the label map.

### Bridge

State the statistical slogan cleanly, list probability objects to review, and open the bridge: **function approximation ↔ distribution estimation**.

---

## Topic 7: Statistical methods, review list, and the bridge to distributions (42:00–47:49)

### Where this sits on the master map

**Payoff + course trajectory.** Warm-up: [sets](./PREREQUISITES.md#p1-sets), [functions](./PREREQUISITES.md#p2-functions). For $\Omega$/RV language after this video, see also package `03-Lec02-…`.

### Board / screenshot

![Stats 1](./screenshots/composites/ch07-stats-probability-review-panel1of2.png)

![Stats 2](./screenshots/composites/ch07-stats-probability-review-panel2of2.png)

**On the board:** complex $f$ → repeated observations → statistics; $X,Y$ as RVs; function approx ↔ distribution estimation; review list; rigor / measure-theoretic foundations foreshadowed.

### What he is establishing

#### Why the professor requires probability theory

**Long story short:** when $f$ is too complex to model from the **physics of the problem**, **resort to statistical methods** — repeated observations and estimates. Probability is the precise language of that method.

Not optional decoration: semantic ML + generative models live here. Deterministic “find a hard $f$” remains the intuition; probability is the **working engine**.

```
  f too complex for first principles
           │
           ▼
  repeated labeled observations
           │
           ▼
  statistical estimation
           │
           ▼
  probability theory (precise words for that)
```

#### How probability changes function approximation (~43:45–46:48)


| Deterministic language                  | Probabilistic language                                                                                                                 |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathcal{X},\mathcal{Y}$ as fixed sets | Data as**random variables** (support of distributions)                                                                                 |
| Find / estimate map$f$                  | Estimate**distributions** (e.g. structure behind $(X,Y)$, often thought as $P(Y\mid X)$ / joint structure — full formalization later) |
| Error minimization on predictions       | Counterparts like**likelihood**-style thinking (parallel, not identical formulas yet)                                                  |
| Ideal hard link                         | Explicit**uncertainty** and path to **generative** modeling                                                                            |

He stresses a **one-to-one parallel**: steps you take in the deterministic story have counterparts in the probabilistic story. He will often give **deterministic equivalents** for intuition, but the formal course logic is probabilistic because it is more robust for modern ML.

#### Deep dive ~46:30–47:50 (end of lecture)

**1. Function mapping → distribution estimation (~46:32–46:48)**
“Given $D$, find $f$” will be rephrased as estimating probabilistic structure of the data. That shift is what lets systems handle uncertainty and, later, **generate** new samples (heart of generative modeling).

**2. One-to-one parallels (~46:53–47:05)**
Do not throw away the first half of the lecture. Every deterministic move (guess a class, fit with data, minimize error) has a rigorous cousin in probability (choose a probabilistic model family, fit with data, maximize likelihood / related criteria — details in later lectures).

**3. Need for rigor / measure-theoretic foundations (~47:15–47:50)**
High-dimensional data (images as points in $\mathbb{R}^{pq}$) need careful definitions of what events you can assign probability to. He points toward foundations: sample spaces, measures, and (in the broader series language) objects like **measure spaces** / **sigma-algebras** (including Borel-type constructions for continuous spaces). You do not need full measure theory tonight — but you **must** review the basic probability vocabulary so next classes can build without restarting.

#### Homework: review these before next class (Plain English + course meaning)


| Concept                      | Plain English                                                                           | In this course                                                                            |
| ------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Random experiment**        | A process whose outcome you treat as uncertain (coin, full X-ray visit+label pipeline). | Starting point of probability; you*choose* how to define it.                              |
| **Sample space $\Omega$**    | The bag of**all possible outcomes** of that experiment.                                 | Set of outcomes; may be non-numeric.                                                      |
| **Random variable**          | A**rule that turns each outcome into a number** (or vector of numbers).                 | **Deterministic function** $\Omega\to\mathbb{R}$ (or $\mathbb{R}^d$). Name is misleading. |
| **Probability measure $P$**  | A size rule on events with total mass 1.                                                | Assigns numbers in$[0,1]$ to allowed subsets (events).                                    |
| **Probability distribution** | How probability is spread over values of an RV.                                         | Describes the law of the RV (discrete or continuous).                                     |
| **Density function**         | For continuous RVs, a curve whose area gives probability.                               | **Does not exist for every RV** — he warns you not to assume densities always.           |

```
  random experiment
        │
        ▼
  Ω = all outcomes
        │
        │  RV = deterministic map
        ▼
  numbers / vectors (data we store)
        │
        │  P / distribution (/ density if it exists)
        ▼
  probabilistic model of the world
```

#### Course approach to model building (summary of whole lecture)

1. State ML as **function approximation** (universal first-principles goal: recover useful $f$ from $D$).
2. Use **models** (guesses) + **algorithms** (refinement) + **data** (oil).
3. Admit **semantic** tasks break short deterministic physics.
4. Shift to **probabilistic** foundations for the rest of the series.
5. Keep deterministic pictures as intuition; demand probability for rigor and SOTA.

#### “Data is the new oil” / internet data scientist

**Data is the new oil:** refinement quality tracks data quality/quantity once the class of models is fixed.
**Internet data scientist:** tool user without foundations. This course wants the opposite: foundations so you can **build and read** models, not only call APIs.

**You can now:** explain why probability is required; state the FA ↔ distribution parallel; recite and roughly define the review list.
**Still missing:** full constructions of $\Omega$, $P$, RVs — next lectures (start with Lec 02 package in this series).

### Analogy for this topic only

Deterministic view: “Draw the exact road from city $X$ to city $Y$.”
Probabilistic view: “Learn which roads people usually take, with traffic uncertainty — and maybe generate new trips.”

Same travel problem; second view handles mess and generation. That is why generative ML sits in probability land.

**Question:** if you only memorize last week’s trips, are you ready for a new destination?
No — same as overfitting / table lookup in the deterministic half.

### Local picture

```
  TODAY
    deterministic FA:  given D, estimate f
    semantic gap → need statistics
    probabilistic FA:  X,Y as RVs · estimate distributions
    one-to-one parallels · course = probabilistic + det. equivalents

  REVIEW BEFORE NEXT CLASS
    experiment · Ω · RV · P · distribution · density (when exists)

  FORWARD
    rigor for high-d data · generative models · measure foundations
```

Notice: Lec 01 ends by installing the *problem* and the *bridge* — not by finishing estimation theory.

### Bridge

Next: probability recap lectures. Bring the review list cold. Keep both sentences in your head:

1. **Given $D$, estimate $f$** (model + algorithm).
2. **That story will be retold as distribution estimation** under probability.

---

## Quick self-check (answers you should now own)


| Question                  | Short answer from this lecture                                                    |
| --------------------------- | ----------------------------------------------------------------------------------- |
| Main goal of this course? | Foundations (esp. probabilistic) so you can**build/read** ML, not only use tools. |
| Why math = language?      | English is fuzzy; math is for**precision**.                                       |
| Why board not slides?     | Live mistakes/thinking; slides are LLM-easy.                                      |
| Internet data scientist?  | Tool user without foundations — course fights that.                              |
| Core problem?             | **Given $D$, estimate $f$.** (universal FA goal)                                  |
| Model?                    | Estimate/guess of$f$.                                                             |
| Algorithm?                | Procedure that refines the model using$D$.                                        |
| What creates performance? | Class + data quality + algorithm + task difficulty; not magic.                    |
| “Useful” model?         | Not truth; good enough for prediction.                                            |
| Overfit (true sense)?     | Fits training (even noise/random labels) but fails real structure/new data.       |
| Deterministic view?       | Hard map$f:X\to Y$ from $D$.                                                      |
| Probabilistic view?       | RVs +**distribution estimation**; handles uncertainty/generation.                 |
| How prob changes FA?      | Same goal, retold as laws of data; one-to-one parallels.                          |
| Semantic gap?             | Physical measurements ≠ human meanings.                                          |
| History → modern ML?     | Physics inversion → statistical “ignorance modeling.”                          |
| Data is new oil?          | $D$ quality drives how well you refine the model.                                 |
| Image mathematically?     | Matrix of pixels → stack → vector in$\mathbb{R}^{pq}$.                          |
| Review list?              | Experiment,$\Omega$, RV, $P$, distribution, density (not always).                 |
| After 46:30?              | FA ↔ distribution estimation · parallels · rigor for high-d$P$.                |

---

## External references


| Resource                                                                                                                             | Matches…                  | Why it helps                        |
| -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ------------------------------------- |
| [Supervised Learning as Function Approximation](https://www.adventuresinwhy.com/post/supervised-learning-as-function-approximation/) | Topics 2–4                | Same “given$D$, find $f$” framing |
| [Caltech ML — Lec 1 “The Learning Problem”](https://www.youtube.com/watch?v=mbyG85GZ0PI)                                          | Topics 1–4                | Parallel chalk lecture              |
| [3Blue1Brown — Vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs)                                                                | Topic 5                    | Visual$\mathbb{R}^d$                |
| [Seeing Theory](https://seeingtheory.brown.edu/)                                                                                     | Topic 7 homework list      | Interactive probability prep        |
| [Box (1976) PDF](https://www-sop.inria.fr/members/Ian.Jermyn/philosophy/writings/Boxonmaths.pdf)                                     | “All models are wrong…” | Primary source of the slogan        |
| [`03-Lec02-Recap-Probability-Theory-Part1`](../03-Lec02-Recap-Probability-Theory-Part1/)                                             | After this video           | Next NPTEL probability foundations  |

---

## Sources

- Video: https://www.youtube.com/watch?v=G2h7nD_Stxg
- Playlist index 2: https://www.youtube.com/playlist?list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu
- Local: [PREREQUISITES](./PREREQUISITES.md) · [TRANSCRIPT](./TRANSCRIPT.md) · [screenshots/composites](./screenshots/composites/)
- Skill: `youtube-lecture-tutor` · content_type `math_technical`
