# Lec 01 — Overview of Function Approximation

**Video:** [Lec 01 Overview of Function Approximation](https://www.youtube.com/watch?v=G2h7nD_Stxg) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)

---

## Table of Contents

1. [Topic 1 — Course mission: builders and math as precision](#topic-1-course-mission-builders-and-math-as-precision-0000–0630) (00:00–06:30)
2. [Topic 2 — Function approximation as the core problem](#topic-2-function-approximation-as-the-core-problem-0630–1204) (06:30–12:04)
3. [Topic 3 — Blind problem and the table-lookup trap](#topic-3-blind-problem-and-the-table-lookup-trap-1204–1507) (12:04–15:07)
4. [Topic 4 — Models, algorithms, and slogans](#topic-4-models-algorithms-and-slogans-1507–1954) (15:07–19:54)
5. [Topic 5 — Vectors, X-rays, stacking into high dimension](#topic-5-vectors-x-rays-stacking-into-high-dimension-1954–2850) (19:54–28:50)
6. [Topic 6 — Semantic gap: sensors vs abstract labels](#topic-6-semantic-gap-sensors-vs-abstract-labels-2850–3309) (28:50–33:09)
7. [Topic 7 — Physics-first vs statistical methods](#topic-7-physics-first-vs-statistical-methods-3309–4217) (33:09–42:17)
8. [Topic 8 — Why probability: FA becomes distribution estimation](#topic-8-why-probability-fa-becomes-distribution-estimation-4217–4749) (42:17–47:49)
9. [External references](#external-references)
10. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: from a finite table of pairs, estimate unknown $f$ and answer new inputs. Method: guess a **model family**, refine with data by an **algorithm** (do not only memorize the table). Fork: physics often fails for abstract labels, so the course uses **statistical / probabilistic** modeling.

**Worldview arc:** deterministic **function approximation (FA)** → model+algorithm → probability ($X,Y$ as RV supports; FA ↔ distribution estimation).

### System context

```
  ╔══════════════════════════════════════════╗
  ║ Outside this lecture                     ║
  ║ · full probability course (refresh list) ║
  ║ · SOTA algorithms (course 2)             ║
  ║ · RVs / densities built next sessions    ║
  ╚══════════════════╤═══════════════════════╝
                     │ this video installs
                     ▼
            ┌─────────────────┐
            │ MFML Lec 01 FA  │
            └─────────────────┘
```

### Main blueprint

```
  ╔════════════ GOAL ════════════╗
  ║ Builders; math = precision   ║
  ║ Read papers without freeze   ║
  ╚══════════════╤═══════════════╝
                 │ needs a precise job
                 ▼
  ┌──────────────────────────────┐
  │ PROBLEM: function approx.    │
  │ domain X · range Y · data D  │
  │ ∃ unknown f : X→Y            │
  │ given D, estimate f          │
  └──────────────┬───────────────┘
                 │ obstacle
                 ▼
  ┌──────────────────────────────┐
  │ OBSTACLE: blind / ill-posed  │
  │ trap: table-only at x_i      │
  │ (overfitting seed)           │
  └──────────────┬───────────────┘
                 │ method
                 ▼
  ┌──────────────────────────────┐
  │ METHOD: model + algorithm    │
  │ (mini) family = guess class  │
  │ (mini) algorithm = refine D  │
  │ slogans: models wrong/useful;│
  │          data is oil         │
  └──────────────┬───────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  ┌────────────┐   ┌──────────────┐
  │ REPRESENT  │   │ WHY HARD     │
  │ stack image│   │ sensor ≠     │
  │ → x in R^d │   │ abstract y   │
  └──────┬─────┘   └──────┬───────┘
         └────────┬───────┘
                  ▼
         ┌────────────────┐
         │ FORK            │
         │ physics-first   │
         └────────┬───────┘
               ──X──► often blocked
                  │
                  ▼
         ┌────────────────┐
         │ STATS PATH      │
         │ many repeats    │
         │ "ignorance mdl" │
         └────────┬───────┘
                  ▼
         ┌────────────────┐
         │ PROBABILITY     │
         │ X,Y as RV supp. │
         │ FA ↔ distrib.   │
         └────────┬───────┘
                  │
         ┌ · · · ·┴ · · · · ┐
         │ STOP: build RVs, │
         │ measures next    │
         └ · · · · · · · · ·┘
```

### Scenario walkthrough

1. Planet nights fill $D$ → ask night 10 → **PROBLEM**.  
2. Recite only nights 1–3 → fails → **OBSTACLE**.  
3. Guess ellipses; pick which with data → **METHOD**.  
4. Stack X-ray to $x\in\mathbb{R}^{d}$, label $0/1$ → **REPRESENT**.  
5. Ask why reflectance equals “disease” → **WHY HARD**.  
6. Count coin tosses instead of fluid dynamics → **STATS** → **PROBABILITY** door.

### Failure / contrast path

```
  F=ma style links          abstract human labels
         │                           │
         ▼                           ▼
  deterministic OK           physics invert ──X──►
                                         │
                                         ▼
                              annotate + statistics
```

### STOP / out of scope

No full probability axioms, no trained neural net, no finished random-variable theory. Next sessions connect FA to sample spaces, measures, and distributions.

### Load-bearing claims (closed-book)

- FA: given finite pairs $D$, estimate unknown $f:X\to Y$.  
- Blind without structure; table lookup fails for new $x$.  
- **Model** = estimate/family; **algorithm** = refine with data.  
- All models wrong, some useful; data quality limits the estimate.  
- Images stack to vectors in $\mathbb{R}^{d}$; classification is still FA.  
- Sensor readings ≠ semantic labels; physics-first often blocked.  
- Statistical methods: repeated observations when $f$ is too complex.  
- Probabilistic view: $X,Y$ as RV supports; FA ↔ distribution estimation.

**Speaker / course:** NPTEL — Indian Institute of Science, Bengaluru · Mathematical Foundations of Machine Learning · Lec 01.

---

## Topic 1: Course mission — builders and math as precision (00:00–06:30)

### Where this sits on the master map

**GOAL** box: who the course is for and what language it uses, before any formal FA. Refresh [sets](./PREREQUISITES.md#p1-sets) and [functions](./PREREQUISITES.md#p2-functions) if symbols feel cold.

### Board / screenshot

![Early lecture framing](./screenshots/composites/ch01-topic-01-course-mission-panel1of1.png)

**Figure — ~00:30–06:00:** instructor frames the two-course series and board-work style; substance later writes FA symbols, so treat this panel as mission + language contract.

### What he is establishing

Welcome to **Mathematical Foundations of Machine Learning**. Machines “thinking” raises ethics; this course sets ethics aside and studies **algorithms** that make learning systems work. It is a **two-course series**: this course lays **probabilistic foundations** for generative and classical ML; a later course covers state-of-the-art algorithms.

Concrete scene: two students read the same paper full of “conditional expectation” and “convergence of random variables.” One freezes and closes the tab. The other follows the symbols because the course made them load-bearing for systems like GPT-scale models. Everyday English is fragile — words like “learning” and “thinking” can mean a hundred things. If you implement systems, ambiguity breaks products. So the instructor treats **mathematics as a language designed for precision** — not magic, the right tool when a sentence must have one meaning.

That is also why probability topics that felt pointless (conditional expectation, laws of large numbers, transforms of random variables) reappear as load-bearing. Without them you cannot honestly explain systems on the scale of large language models. The social goal is sharper: leave the role of **“internet data scientist”** (everyone-and-their-grandmother tooling) and become a **builder**. Car analogy: you can drive without knowing the internal combustion engine; you cannot design a faster engine without the inner workings. After both courses, the target is reading state-of-the-art literature with the ease of a novel.

Style: mostly **board work**, including mistakes and corrections. Substance flag for the whole course: ML algorithms will be treated primarily from a **probabilistic standpoint** (explained in Topics 5–8).

Picture two people at the same chat-model demo. One clicks and ships a demo; the other opens the paper, reads the theorems, and changes the method when it fails. This course is built for the second person.

You can now state the course contract: precision language, builder mindset, probabilistic foundations first. Still missing is the scientific job for the next forty minutes — function approximation.

### Analogy for this topic only

You drive a car by habit for years. Someone asks: **how do you redesign the engine so it starts in winter and uses less fuel?** Habit alone cannot answer.

The course is not anti-driving. It is anti-**only**-driving when your goal is to build. English slogans about “AI thinking” are dashboard lights. Equations are workshop drawings.

In lecture words: driver ≈ internet data scientist; drawings ≈ math as precision language.

### Local picture

```
  Course 1 (this)              Course 2 (later)
  probabilistic foundations →  SOTA algorithms

  Driver path          Builder path
  call fit()           own theorems / models
  English slogans      math as language
```

**Notice:** GOAL is social and linguistic; FA is not defined yet.

### Bridge

A precise language needs a precise **job**. Most of science, engineering, and ML share one: **function approximation**.

---

## Topic 2: Function approximation as the core problem (06:30–12:04)

### Where this sits on the master map

**PROBLEM** box. Everything later is machinery for this job. Warm-up: [domain/range](./PREREQUISITES.md#p4-domain-range), [pairs as data](./PREREQUISITES.md#p3-pairs-data).

### Board / screenshot

![Domain range and data pairs](./screenshots/composites/ch02-topic-02-function-approximation-panel1of1.png)

**Figure — ~06:30–12:00:** board introduces domain/range and observation pairs for classical FA (planet-style setup).

### What he is establishing

Most problems in science and engineering can be cast as **function approximation**. The lineage is old: Kepler, Gauss, and others hunted unknown rules from observations. Stock story: record planet positions at several times; want positions at times you **did not** observe.

Formally, name two sets. **Domain** $\mathcal{X}$ holds allowed inputs (the lecturer avoids overusing “input” because later these behave like random variables). **Range** $\mathcal{Y}$ holds allowed outputs. You are handed **pairs**

$$
D = \{(x_1,y_1),\ldots,(x_n,y_n)\}
$$

In words: for each observed $x_i$ you also saw $y_i$.

Assume there **exists** a function

$$
f : \mathcal{X} \to \mathcal{Y}
$$

In words: a mapping that, given a domain element, returns the corresponding range element. The pairs in $D$ are **evaluations** of that mapping. The crisis: **we do not know $f$**. Slogan:

**Given $D$, find $f$.**

More carefully: estimate the underlying function well enough to predict at new domain points. For planets, time lives in $\mathcal{X}$ and position in $\mathcal{Y}$.

If you only store the $n$ pairs, you have a notebook, not a function estimate. That is the wrong complete answer when people confuse memory with science. The right problem statement forces honesty: existence of $f$ is an assumption; knowledge of $f$ is not; $D$ is evidence, not the map.

You can now write the formal FA problem with domain, range, $D$, and unknown $f$. Still missing is how to attack the problem when you know nothing about $f$ except $D$.

### Analogy for this topic only

Three sky watches:

- night 1 → position A  
- night 2 → B  
- night 3 → C  

**Where is the planet on night 10?** You never watched night 10.

There is a real path. You only have three sightings. Function approximation means invent a path that fits A, B, C *and* answers night 10. Reciting only 1–3 stores sightings; it does not approximate the path.

In lecture words: path = $f$, sightings = $D$, night 10 = new $x$.

### Local picture

```
  Domain X        unknown f         Range Y
  x1  ──────────────────────────►   y1
  x2  ──────────────────────────►   y2
  x_new ── ? ───────────────────►   ???

  D = {(x1,y1),(x2,y2),...} finite
```

**Notice:** $D$ is finite; the domain may be far larger.

### Bridge

Stating “given $D$ find $f$” is easy. Solving it with **no information about $f$ except $D$** is not. Next: blindness and a tempting wrong fix.

---

## Topic 3: Blind problem and the table-lookup trap (12:04–15:07)

### Where this sits on the master map

**OBSTACLE** box. Ties to [estimate vs truth](./PREREQUISITES.md#p5-estimate).

### Board / screenshot

![Blind estimation discussion](./screenshots/composites/ch03-topic-03-blind-table-lookup-panel1of1.png)

**Figure — ~12:04–15:00:** board tension — data present, rule absent; conversation moves toward “start with a guess.”

### What he is establishing

Can we solve FA as stated? In a strong sense it is **ill-posed** or **blind**: the map is completely unknown while you only hold evaluations. Question: **how do you estimate a function when you are totally blind except for samples?**

A wrong answer appears immediately: define the function to equal $y_i$ at each observed $x_i$ and ignore the rest. Perfect fit on the notebook; useless for science. We want outputs at **$x$ we have not observed**. A function that only recites the training table fails that purpose.

That bad idea will later be formalized as **overfitting**. Seed now: perfect memory of $D$ is not a good approximation of $f$.

Classroom pressure often jumps to “optimize the error.” Premature: error and optimization are not defined yet. That gap — folk intuition without precise objects — is what the course fills. At the philosophical level available today:

**Start with some guess on $f$.**

Not a random number — a structured assumption about what kind of rule might exist. Jumping to minimize error without defining error is the other wrong reflex.

You can now reject table-only “solutions” and state: structured guess first. Still missing names for *guess class* versus *refinement procedure*.

### Analogy for this topic only

Three planet nights in a notebook. A friend “solves” it:

- night 1 → A, night 2 → B, night 3 → C  
- otherwise: undefined  

**Where on night 10?** The notebook shrugs. Better first move: guess a shape of path, then use the three nights to choose among shapes.

In lecture words: table-only = overfitting seed; need unobserved $x$; start with a guess on $f$.

### Local picture

```
  x: 1  2  3     table f: f(1)=A,f(2)=B,f(3)=C
  y: A  B  C              f(10)= ???  purpose fails
```

**Notice:** fitting $D$ perfectly can still fail the scientific job.

### Bridge

A guess needs a name and a refinement story: **model**, **data**, **algorithm**, and the slogans that stick.

---

## Topic 4: Models, algorithms, and slogans (15:07–19:54)

### Where this sits on the master map

**METHOD** box. Model and algorithm are **co-defined** — both must stick.

### Board / screenshot

![Guess and refine ideas](./screenshots/composites/ch04-topic-04-model-algorithm-panel1of1.png)

**Figure — ~15:07–19:50:** ellipse/guess family and refinement with data; slogans enter speech here even if chalk is dense.

### What he is establishing

A first guess might be: “$f$ is a parabola.” Historically, Kepler-type thinking said planetary paths are **ellipses**. Naming the shape is progress and incomplete. There are **infinitely many** ellipses. After you assume “ellipse,” data must choose **which** one.

Every estimate you produce for $f$ is a **model**. You have no portal to the true generator — only estimates. Proverb:

**All models are wrong, but some are useful.**

Wrong because they are not inaccessible true $f$. Useful because they answer new $x$ well enough. Workflow: **start with an initial guess on $f$, then refine using $D$.** Set $D$ is **data**. Slogan **“data is the new oil”**: quality of $D$ limits how good the estimate can become.

Three driving questions for the whole course:

1. **What guess** (model family)?  
2. **How refine** with data?  
3. **How know** the refinement is good?

Every refinement procedure is an **algorithm**. Every initial guess class is a **model / model family**: linear, quadratic, kernels, neural nets, decision trees. Pair a family with a refinement algorithm → machine-learning method. Example: **error back-propagation** refines the **neural network** family. That combined process is **learning**.

Flag reopened: the course insists on a **probabilistic viewpoint**. Much of ML can be narrated deterministically as “guess $f$ and fit $D$.” **Generative models** sit more naturally in probability; the framework is broad enough for discriminative and generative stories (terms later).

You can now separate family, algorithm, and shipped model, and recite both slogans with meaning. Still open: how real $x$ look as images, and why probability is forced.

### Analogy for this topic only

You declare: “the planet follows an ellipse” — a **family**, not one path.

**Which ellipse should we use on night 10?** Naming the family does not answer. Nights choose among ellipses; the selection procedure is the algorithm; the chosen ellipse is the model.

In lecture words: family = model class; chosen ellipse = model; nights = $D$; selection = algorithm.

### Local picture

```
  family H={ellipses} ──algorithm A(D)──► one model hat f

  Three questions: (1) family? (2) refine how? (3) is it good?
```

**Notice:** model ≠ algorithm; learning needs both.

### Bridge

The method still looks abstract if every $x$ is a scalar time. Real ML often has $x$ as an **image**. Next: stacking an X-ray into a vector.

---

## Topic 5: Vectors, X-rays, stacking into high dimension (19:54–28:50)

### Where this sits on the master map

**REPRESENT** box. Need [vectors](./PREREQUISITES.md#p6-vectors) and [high dimension](./PREREQUISITES.md#p7-high-dim).

### Board / screenshot

![X-ray and vector setup](./screenshots/composites/ch05-topic-05-vectors-xray-panel1of2.png)

![Stacking and high-d points](./screenshots/composites/ch05-topic-05-vectors-xray-panel2of2.png)

**Figure — ~19:54–28:50 (two panels):** images as matrices, column stacking toward $\mathbb{R}^{PQ}$, labels $0/1$, radiologist as target $f$.

### What he is establishing

Why leave pure determinism? Generative models fit best under a probabilistic view; probability can host both discriminative and generative ML later. Concrete pressure: medicine.

You are given **X-ray images** and labels for disease. Still FA: domain = images; range = labels. Domain elements need not be scalars — they may be **vectors**. A **vector-valued function** takes a vector in and returns a vector out; scalars are special vectors.

Digital X-ray: **matrix** of **pixels**. 8-bit pixels in $0,\ldots,255$. Size $P\times Q$ → $P\cdot Q$ numbers.

**Stacking procedure:** take column 1, then column 2, …, concatenate into one long list of length $PQ$. That vector is a point in $\mathbb{R}^{PQ}$. Euclidean space extends the Cartesian plane: 2 numbers in 2D, $n$ in $n$D. Humans cannot visualize $PQ$ axes; the algebra does not care. Going forward, each $x_i\in\mathbb{R}^{d}$ (real-valued default).

Labels can be binary $y_i\in\{0,1\}$. Then $f$ maps a high-dimensional point to a bit. Real-world $f$ is approximately a **radiologist**. Given ~1000 labeled X-rays, estimate a function that **mimics** that expert. Casual “universal function approximation” slogan: many human maps might be learnable as FA. Classification of images is still FA with vector inputs and discrete outputs.

Treating an X-ray as “not math” because it is a picture is the wrong move; stacking turns it into a point in $\mathbb{R}^{d}$.

You can now convert a toy matrix into a vector, state $x\in\mathbb{R}^{d}$, and write disease as FA. Still missing: **why** brightness should mean “disease.”

### Analogy for this topic only

Radiologist archive:

- image 1 → disease  
- image 2 → no disease  
- image 3 → disease  

Stack brightness numbers into one ID badge per image — a point in a huge space. **New patient badge: what label?** Reciting the old archive does not answer.

In lecture words: badge = $x\in\mathbb{R}^{d}$, label = $y$, doctor = $f$, archive = $D$.

### Local picture

```
  2×3 micro image:          stack cols → length 6
    10 20 30                  (10,40,20,50,30,60) ∈ R^6
    40 50 60

  General: P×Q → x ∈ R^{PQ} → f(x) ∈ {0,1}
```

**Notice:** $d=PQ$ is “how many numbers,” not a mystical axis.

### Bridge

We can write $f(x)=y$ for X-rays, but **why** should radiation equal “disease”? Semantic gap next.

---

## Topic 6: Semantic gap — sensors vs abstract labels (28:50–33:09)

### Where this sits on the master map

**WHY HARD** box: FA on abstract labels is not like FA on Newtonian variables.

### Board / screenshot

![Semantic gap discussion](./screenshots/composites/ch06-topic-06-semantic-gap-panel1of1.png)

**Figure — ~28:50–33:00:** argument shifts from “we can stack an image” to “why labels mean anything.”

### What he is establishing

**Force** and **acceleration** are measurable and linked by elementary physics (linear dependence in the law the class recalls). Velocity and acceleration are similarly grounded. The X-ray problem is subtler. Vector $x$ measures radiation absorbed/reflected at a sensor. **Disease** is an abstract clinical idea. Why should reflectance equal that idea?

Same crack: gender from photos (synthetic construct; light ≠ gender); speech intent while a microphone records **pressure gradients**. Flying an aircraft can lean on Newtonian mechanics with few parameters; gender-from-image may need ~**100 million** parameters because human constructs are complicated.

Experiment spirit: **MNIST** digits with **random** labels can still be fit by a high-capacity model. Capacity ≠ semantics.

When both sides are clean physical measurables, a **deterministic worldview** may suffice. When targets are abstract and poorly tied to sensors by known laws, that worldview is not enough. Pretending disease is “just $F=ma$ with more algebra” is the wrong comfort.

You can now explain why sensor→label maps are hard even when FA notation looks identical to the planet problem. History’s failed comfort — pure physics inversion — is next.

### Analogy for this topic only

**Notebook A:** mass, force, acceleration — short known story.  
**Notebook B:** microphone voltages + label “speaker is angry.”

**Can you write a five-parameter law for B the way Newton writes A?** You cannot — and that is the point.

In lecture words: measurable physics links vs semantic targets; deterministic worldview limited for the second.

### Local picture

```
  Easy: force ──linear──► accel
  Hard: radiation ──???──► disease / gender / intent
  Sensor = numbers · Label = meanings · Gap = why equal?
```

**Notice:** the gap explains data-hungry statistics; it does not claim ML “cannot work.”

### Bridge

History tried physics models of speech and images. Industry then won with statistical methods. That fork is Topic 7.

---

## Topic 7: Physics-first vs statistical methods (33:09–42:17)

### Where this sits on the master map

**FORK** box. Warm-up: [chance from repeats](./PREREQUISITES.md#p8-repeats).

### Board / screenshot

![Physics path and speech history](./screenshots/composites/ch07-topic-07-physics-vs-stats-panel1of2.png)

![Statistical methods and complexity of f](./screenshots/composites/ch07-topic-07-physics-vs-stats-panel2of2.png)

**Figure — ~33:09–42:00 (two panels):** coin/speech physics attempts; late-1980s statistical digit recognition; “ignorance modeling”; board line that $f$ is complex from physics.

### What he is establishing

Predict a **coin toss**. Even with many past outcomes and measurements (mass, atmospheric pressure, toss force), writing a reliable physical $f$ for the next face is brutal.

**Speech recognition** is FA: inputs = pressure-gradient vectors; outputs = abstract **phones**. Same phone “ah” from two speakers → different vectors. Classical approach: model production with **physics** — waveguides (pipes of different diameters, flute-like constrictions), boundary conditions, vocal-fold geometry. Recognition = **invert** the physical system from the measured signal. Decades of effort.

Images: detect edges, lines, gradients; build hierarchical objects toward scene understanding.

Then **late 1980s** industry digit speech recognition succeeded with **statistical methods**: do not start from full physics; make **repeated observations** and deduce statistical regularities. Coin method: toss 10,000 times, count heads, answer chance questions without fluid dynamics.

Speech scientists pejoratively called this **“ignorance modeling.”** Engineers: fine — **it works**. Parallel to modern deep learning: we often engineer systems better than we philosophically understand them.

Bottom line: for “thinking” problems, $f$ is complex from physics; you cannot estimate it from first principles the way you estimate aircraft dynamics from Newton. Light reflectance ↔ gender is not a clean handwritten law. Practical path: many pictures, **annotate**, use statistics.

You can now retell the fork: physics inversion vs repeated-observation statistics. Still need the **language** of that path — probability.

### Analogy for this topic only

**Team Physics** simulates air, thumb, bounce for years.  
**Team Count** flips 10,000 times and answers “about 60% heads.”

**What is the chance the next toss is heads?** Team Physics may still be debugging turbulence. Team Count has a number.

Speech history is Team Physics with waveguides; 1980s digits is Team Count. “Ignorance modeling” is the insult while products shipped.

In lecture words: statistical methods = repeated observations; ignorance modeling = pejorative name for success.

### Local picture

```
  Physics-first: model gen → invert  ──X──► hard for complex f
  Statistical:   many (x,y) → count/estimate → works

  Coin: 10000 tosses, 6012 heads → ~0.6012
```

**Notice:** same FA problem; different attack when physics cannot write $f$.

### Bridge

Winning path is repeated observation and estimation. Precise language: **probability theory**. Last topic rewrites FA and assigns the review list.

---

## Topic 8: Why probability — FA becomes distribution estimation (42:17–47:49)

### Where this sits on the master map

**PROBABILITY** box and lecture **STOP**. Preview only. Warm-up: [repeats](./PREREQUISITES.md#p8-repeats).

### Board / screenshot

![Probability motivation and review list](./screenshots/composites/ch08-topic-08-probability-view-panel1of1.png)

**Figure — ~42:17–47:49:** complexity of $f$ → statistical methods → probability; spoken review list (sample space, RV, measure, distribution, density).

### What he is establishing

Because $f$ is hard to model from physics, **resort to statistical methods**: repeated observations and estimates. **That is why this course needs probability theory.** Plain gloss: probability is about making repeated observations and learning estimates. Micro case: 10,000 coin tosses with 6,012 heads is already a numerical estimate without solving air dynamics — the same spirit that will become precise with measures.

Deterministic FA one last time: two sets, mapping, observations, **given $D$ find $f$.** Under a **probabilistic standpoint**, domain and range are no longer mere fixed sets — they are tied to **random variables** (more precisely, **support sets** of RVs).

A first-course shock: a **random variable** is neither “random” nor a “variable” in the street sense. It is a **deterministic function** from a **sample space** to real numbers. Together with measure ideas (Borel σ-algebra, measure spaces) it supports a **probability measure**.

**Mandatory review list** before next lecture:

- sample space  
- random experiment  
- random variable  
- probability measure  
- probability distribution function  
- density functions (**densities may not exist for every random variable**)

Next session connects FA to random variables and distribution functions. Punchline preview: function approximation **translates into distribution (or density) estimation**, with parallels to the deterministic worldview. Staying purely deterministic because “FA already made sense” is the wrong close for this course. The course stays **probabilistic as the base language**, with **deterministic equivalences** along the way.

You can now restate FA in both worldviews and list the probability ideas to refresh. Still missing is the actual construction of measures and distributions — next lectures’ work.

### Analogy for this topic only

Deterministic FA: notebook of fixed pairs and hidden $f$.  
Probabilistic FA: a **lottery machine** (random experiment); recorded numbers are random variables; the pattern is a distribution.

**You only saw 500 lottery draws. What is the pattern of the whole machine?** That is distribution estimation — twin of “what is $f$?”

In lecture words: supports of RVs replace bare sets; FA ↔ distribution estimation.

### Local picture

```
  Deterministic FA              Probabilistic FA
  sets X,Y · f · D              RV supports · distributions
  estimate f                    estimate distribution
           \                    /
            └── parallels (next lectures)

  Review: sample space · RE · RV · measure · CDF · density?
```

**Notice:** this video stops at the doorway of probability.

### Bridge

You now own the architecture: FA → blind obstacle → model/algorithm → vectors → semantic gap → statistical fork → probability. The next lecture builds probabilistic objects on that foundation.

---

## External references

Companions for **this** video’s boxes (videos + notes/blogs; not a wiki dump).

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | Topics 4–5 model families; $x$ as list of numbers | Visual vector→label intuition under the FA framing |
| [StatQuest — Fitting a line to data](https://www.youtube.com/watch?v=PaFPbb66DxQ) | Topics 2–4 given $D$ estimate $f$ | Concrete scalar FA before high-d X-rays |
| [StatQuest — Machine Learning Fundamentals (playlist start)](https://www.youtube.com/watch?v=Gv9_4yMHFhI) | Topic 4 model vs procedure intuition | Friendly language for guess/refine without skipping rigor later |
| [Josh Starmer / StatQuest — The main ideas of fitting a line](https://statquest.org/) | Topics 2–4 | Blog-style notes paired with short videos |
| [MIT 6.041SC — Probability intro (first lectures)](https://www.youtube.com/playlist?list=PLUl4u3cNGP60A3XMwZ5sep719_nh95qYp) | Topic 8 review list | University backup before Lec 02 objects |
| [MNIST database (LeCun et al.)](https://yann.lecun.com/exdb/mnist/) | Topic 6 random-label thought experiment | Canonical digit images for the assignment spirit |

---

## Sources

- Video: [Lec 01 Overview of Function Approximation](https://www.youtube.com/watch?v=G2h7nD_Stxg)  
- Channel: NPTEL — Indian Institute of Science, Bengaluru  
- Skill: `youtube-lecture-tutor`  
- Captions cleaned in TRANSCRIPT.md / claim sheets  
