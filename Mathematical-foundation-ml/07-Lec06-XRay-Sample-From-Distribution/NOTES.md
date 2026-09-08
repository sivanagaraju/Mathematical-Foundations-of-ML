# Lec 06 — Understanding a Chest X-Ray as a Sample from a Distribution

**Video:** [Lec 06 Understanding a Chest X-Ray as Sample from Distribution](https://www.youtube.com/watch?v=bdcvsSNAHIk) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 05 — Joints / conditionals / margins](../06-Lec05-Recap-Probability-Theory-Part2/NOTES.md)

---

## Table of Contents

1. [Topic 1 — Ground the X-ray in probability language](#topic-1-ground-the-x-ray-in-probability-language-0000–0101) (00:00–01:01)
2. [Topic 2 — Image grid stacked into a vector in R^d](#topic-2-image-grid-stacked-into-a-vector-in-rd-0101–0511) (01:01–05:11)
3. [Topic 3 — Sample space, realizations, physics to statistics](#topic-3-sample-space-realizations-physics-to-statistics-0511–0805) (05:11–08:05)
4. [Topic 4 — Trap: data values are not probabilities](#topic-4-trap-data-values-are-not-probabilities-0805–1037) (08:05–10:37)
5. [Topic 5 — Distribution specifies the sample space; course mission](#topic-5-distribution-specifies-the-sample-space-course-mission-1037–1600) (10:37–16:00)
6. [Topic 6 — Labels as a second RV; joints; ranges of X and Y](#topic-6-labels-as-a-second-rv-joints-ranges-of-x-and-y-1600–2406) (16:00–24:06)
7. [Topic 7 — Dataset, feature/label space, i.i.d. cliffhanger](#topic-7-dataset-featurelabel-space-iid-cliffhanger-2406–2650) (24:06–26:50)
8. [External references](#external-references)
9. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: turn the clinic X-ray into “a sample from a distribution,” not a free-floating file. Method: stack the image to a feature vector, put the label on the **same** experiment, and treat the joint of image and label as what we estimate. Fork: pixel **values are not probabilities** — chance lives on events and regions, not on intensity ink.

**Worldview arc:** from abstract multi-RV tools (Lec 05) **to** grounded X-ray + label as RVs; course identity = **estimate distributions** from data (i.i.d. next; densities later).

### System context

```
  ╔════════════════════════════════════╗
  ║ Outside                            ║
  ║ · Lec 05: joints / cond / margins  ║
  ║ · densities (later)                ║
  ║ · i.i.d. definition (next)         ║
  ╚══════════════╤═════════════════════╝
                 │ this lecture grounds
                 ▼
        ┌────────────────────┐
        │ X-ray x ∈ R^d      │
        │ label Y on same Ω  │
        │ estimate P_{X,Y}   │
        └────────────────────┘
```

### Main blueprint

```
  RE: image patient + assign label
           │
           ▼
  X: Ω → R^d     (image: P×Q grid ──stack──► vector, d=PQ)
  Y: Ω → R^k or discrete set   (label / box / category)
           │
           ▼
  data x,y = range points   ← NOT probabilities
           │
           ▼
  P_X / P_{X,Y}  sizes likelihood of observation-regions
           │
           ▼
  KEY: distribution completely specifies sample space
           │
           ▼
  Course mission: estimate distributions from observations
           │
           ▼
  Dataset {(x_i,y_i)} ~_iid P_{X,Y}   (iid next)
```

### Scenario walkthrough

1. Patient X-ray is a $P\times Q$ pixel grid on the computer.  
2. Stack rows → one vector $x\in\mathbb{R}^{d}$ with $d=PQ$.  
3. That vector is a **range point** of RV $X$, not a probability.  
4. Disease label / tumor box is RV $Y$ on the **same** $\Omega$.  
5. Joint $P_{X,Y}$ is the generative story of image+label.  
6. A dataset is $n$ pairs treated as samples from that joint.  
7. Next: what i.i.d. means; densities still ahead.

### Failure / contrast path

```
  Treat pixel intensity as P(event)     ──X──► density / measure trap
  Forget shared Ω for label             ──X──► no joint story
  Think "image is RV" without range     ──X──► category error
  Skip distribution estimation mission  ──X──► lose course identity
```

### STOP / out of scope

Density definitions (promised, not delivered), full i.i.d. theory (next lecture), formal definition of “machine learning” (later), full supervised/unsupervised unification (preview only).

### Load-bearing claims (closed-book)

- Measurements on the computer = elements of the **range** of an RV (not “the PNG is the function”).  
- Image $P\times Q$ stacks to $x\in\mathbb{R}^{d}$, $d=PQ$ (**view**, not a new physics map).  
- **Data values ≠ probabilities** (even after 0–255 → $[0,1]$ normalize).  
- Distribution of $X$ (or joint of $X,Y$) completely specifies the sample space operationally.  
- Course = estimate distributions from observations; engineer packages the RE.  
- Label = second RV $Y$ on same $\Omega$; joint of $(X,Y)$; $Y\in\mathbb{R}^{k}$ or discrete tags.  
- Dataset $\{(x_i,y_i)\}\sim_{\mathrm{iid}} P_{X,Y}$ (tilde = sampled from; i.i.d. next).

**Speaker / course:** NPTEL IISc · MFML · Lec 06.

---

## Topic 1: Ground the X-ray in probability language (00:00–01:01)

### Where this sits on the master map

**GROUND** box — open the clinic example with the tools from Lec 05. Warm-up: [triplet + RV](./PREREQUISITES.md#p1-triplet-rv).

### Board / screenshot

![Grounding X-ray in probability](./screenshots/composites/ch01-topic-01-ground-xray-panel1of1.png)

**Figure — ~00:00–01:00:** welcome back; ground the example; imaging + label as RV story; measurements as range elements.

### What he is establishing

Welcome back. The plan: **ground the X-ray example** in the notations and languages already built, and only later move to **density functions**.

From a probabilistic perspective, the entire process of **imaging and assigning a label** can be seen through random variables. Once you measure, the actual numbers sitting on the computer are **elements from the range space** of that random variable. There always exists an underlying **probability distribution function / probability measure** imposed by the random experiment — that measure is still there even when all you stare at is a PNG file.

If the file is treated as free-floating with no experiment and no measure behind it, the probability story never starts. Instead: measurement = range point; RE brings $P$.

We now have the X-ray story plugged into the abstract stack. Still missing: how a 2D image becomes a vector in $\mathbb{R}^{d}$.

### Analogy for this topic only

A hospital prints a scan onto a disk.

- Folder name on the drive  
- Matrix of intensities inside the file  
- Doctor later adds a disease tag  

**Is the file “just storage” or a measurement from a process?** If you only see storage, you miss the experiment. Right move: treat the numbers as range points of a measurement map under a clinic process. Wrong move: pretend probability is optional metadata.

In lecture words: imaging + label via RVs; data = range elements; underlying $P$ from the RE.

### Local picture

```
  RE: image + label
       │
       ▼
  X (and later Y) : Ω → numbers
       │
       ▼
  computer file = range point(s)
  + underlying distribution / measure
```

**Notice:** densities are promised later; this lecture’s first job is grounding, not density theory.

### Bridge

How does a rectangular pixel grid become the $d$-dimensional range point the theory wants?

---

## Topic 2: Image grid stacked into a vector in $\mathbb{R}^{d}$ (01:01–05:11)

### Where this sits on the master map

**IMAGE→VECTOR** box — the concrete data type for the rest of the course. Warm-up: [stacking](./PREREQUISITES.md#p2-stack-image).

### Board / screenshot

![Image stacking to R^d](./screenshots/composites/ch02-topic-02-image-vector-panel1of1.png)

**Figure — ~01:01–05:10:** $P\times Q$ image; pixels; stack rows; $d=PQ$; data point in range of RV.

### What he is establishing

Take an image of dimensions $P\times Q$ ($P,Q$ arbitrary). On a computer (OpenCV or similar), that image is a **matrix** — a 2D grid of $PQ$ numbers. Each entry is a **pixel**.

For the course, view that matrix as a vector by **stacking all rows** one after another into one long list. The result lives in $\mathbb{R}^{PQ}$. Set **$d = PQ$**. In general, any data point is a **$d$-dimensional vector**.

In the probabilistic viewpoint, that $\mathbb{R}^{d}$ is the **range space** of the underlying random variable. Every image you see — every data point — is an **element of that range space**.

A student objects: “But it is a rectangle — how is it a vector?” Another phrasing: “Are we mapping it?” Answer: you are not inventing a mysterious second physical map; you are **viewing** the grid as a $PQ$-dimensional vector by stacking. Same $PQ$ numbers, different bookkeeping. Other data types (text was already previewed; more later) will also become $d$-vectors. Start concrete with images.

**Take-home slogan:** every data point is an element in the range space of a random variable. That is the viewpoint to keep.

If stacking is treated as optional decoration, images never enter the RV formalism. Instead: stack → $x\in\mathbb{R}^{d}$ → range point of $X$.

We now have images as range vectors. Still missing: what $\Omega$ is for this experiment, and how realizations work.

### Analogy for this topic only

A contact sheet of photo cells laid out in rows and columns.

- As a sheet: a rectangle of cells.  
- As a memory buffer: one long list made by reading row after row.  

**Can both descriptions be true of the same photo?** Yes — same cells, two bookkeeping shapes. If you refuse the long list, the later “feature vector” language has nowhere to live.

Wrong move: demand that nature “is” only a rectangle. Right move: stack when the theory needs a single list of coordinates.

In lecture words: $d=PQ$; data point ∈ range of the RV.

### Local picture

```
  P×Q grid of pixels
       │ stack rows
       ▼
  x = (… PQ numbers …) ∈ R^d ,  d = P·Q

  R^d = range space of RV X
  every image x is a point in that range
```

**Notice:** $d$ is the dimension of the **representation**, not a magic “amount of randomness.”

### Bridge

If $x$ lives in the range of $X$, what is the underlying sample space, and what are repeated measurements?

---

## Topic 3: Sample space, realizations, physics to statistics (05:11–08:05)

### Where this sits on the master map

**Ω + REALIZATIONS** box. Warm-up: [realizations](./PREREQUISITES.md#p3-realizations), [triplet](./PREREQUISITES.md#p1-triplet-rv).

### Board / screenshot

![Sample space and realizations](./screenshots/composites/ch03-topic-03-omega-realizations-panel1of1.png)

**Figure — ~05:11–08:00:** event/sample space questions; RE is user-conceptualized; single-pixel camera; 100 realizations; physics unknown → stats.

### What he is establishing

A student asks about the **event space / sample space** behind the image RV. Answer: it is **open for interpretation**. The random experiment is something the **user conceptualizes**. There is some experiment happening; outcomes can be thought of as observing these images.

Intuitively, once the image is a vector in the range of $X$, the modeling story is about **how likely particular pixel values** are under that experiment. For ease, view the vector RV as **$d$ scalar-valued RVs** (equivalent packaging from Lec 05).

Thought experiment: a **single-pixel camera**. One scalar measurement. Photograph 100 people in the class → **100 realizations** = 100 points from the range of the RV. The distribution/measure answers: when you run this experiment, what is the likelihood of observing this particular value?

Why all this jugglery? Because we **do not know the physics** of the full problem. The whole point is **repeated experiments and statistics** to model.

He corrects loose language: it is the **underlying probability measure / distribution function** that tells the likelihood of obtaining a particular pixel value under the RE — not a sloppy claim that “the RV itself measures likelihood” without $P$.

If you demand a unique “true $\Omega$ written by nature,” you freeze. Instead: engineer packages a RE, collects realizations, models with $P$ and $X$.

We now have realizations and the stats motive. Still missing: a brutal trap about what the numbers themselves are.

### Analogy for this topic only

You cannot write down every atom of a camera sensor’s physics.

- Day 1: photo of student A → one pixel number  
- Day 2: student B → another number  
- … 100 days → 100 numbers  

**What did you learn without full physics?** The pattern of values the experiment tends to produce — statistics over realizations.

Wrong move: wait forever for a perfect physical $\Omega$ list. Right move: package the experiment, measure repeatedly, model the distribution of readings.

In lecture words: user-conceptualized RE; realizations from the range; no physics → repeated experiments + stats.

### Local picture

```
  User packages RE (imaging story)
       │
       ▼
  X: Ω → R  (single pixel) or R^d
       │
       ▼
  100 runs → 100 range points (realizations)
       │
       ▼
  distribution / measure → likelihood of values
  motive: physics unknown → statistics
```

**Notice:** vector RV ≡ $d$ scalar RVs is still available as a dual view.

### Bridge

Students often think normalizing pixels to $[0,1]$ “makes them probabilities.” That trap needs an explicit kill shot.

---

## Topic 4: Trap — data values are not probabilities (08:05–10:37)

### Where this sits on the master map

**DATA≠P** box — critical correction. Warm-up: [data ≠ P](./PREREQUISITES.md#p4-data-not-p), [preimages](./PREREQUISITES.md#p5-preimages).

### Board / screenshot

![Data values are not probabilities](./screenshots/composites/ch04-topic-04-data-not-prob-panel1of1.png)

**Figure — ~08:05–10:30:** normalization confusion; range elements ≠ P; P on events; inverse images.

### What he is establishing

A student pushes on pixel ranges (e.g. 0–255) and normalizing to $[0,1]$: “without normalizing, how is it a probability?”

The correction is sharp: **the values of the data vector are not probabilities.** Every data point is an element of the **range space** of the RV. That element is a **real number / real vector**. The actual numerical value you see has **nothing to do with probability**.

Work the trap with numbers: intensity $173$ on a $0$–$255$ scale is a coordinate. Divide by $255$ to get $\approx 0.68$ in $[0,1]$ — still a coordinate. Neither $173$ nor $0.68$ is $P(\text{event})$. The value the pixel takes “doesn’t matter” for *being* a probability: raw or normalized, it is still range data.

Range space of a (real-valued / vector) RV is reals / $\mathbb{R}^{d}$. Values can be anything in that space; they need not lie in $[0,1]$.

Where does probability live? There is an underlying sample space and a **probability measure defined on events**. Through the RV you form **inverse images** of regions in the range; those inverse images are events on which $P$ is defined. So data values can be wild reals; probability still sizes **sets**, not “the ink of intensity 173.”

If intensity $0.7$ is read as “70% chance,” the model is already broken. Instead: $0.7$ is a coordinate; chance is $P$ of the preimage of a region around such images.

We now know data ≠ $P$. Still missing: why estimating the **distribution** is the whole game.

### Analogy for this topic only

A bathroom scale prints a weight after you step on it.

- Is that printed weight “the chance you exist”? No.  
- Is it a measured reading? Yes.  
- **When does chance appear?** When you ask how often people land in a weight band under a model of who steps on the scale.

Wrong move: squeeze the display into the unit interval and call the display a probability. Right move: keep measurement and measure as different objects.

In lecture words: data ∈ range($X$); not a probability; $P$ on events via preimages.

### Local picture

```
  x = (pixel values…) ∈ R^d     ← NOT a probability

  S ⊂ R^d  (region of possible images)
  E = X^{-1}(S) ⊂ Ω             ← event
  P(E)                          ← probability

  normalize to [0,1]  ≠  make it P
```

**Notice:** continuous case will talk about subsets/regions, not casual “$P($exact float$)$.”

### Bridge

If probability lives on events/regions, what does the **distribution function** buy you about the experiment itself?

---

## Topic 5: Distribution specifies the sample space; course mission (10:37–16:00)

### Where this sits on the master map

**DIST→Ω + MISSION** — key slogan + course identity. Warm-up: [distribution specifies Ω](./PREREQUISITES.md#p6-dist-specifies).

### Board / screenshot

![Distribution specifies sample space](./screenshots/composites/ch05-topic-05-dist-specifies-omega-panel1of1.png)

**Figure — ~10:37–16:00:** likelihood under $X$; subsets/preimages; slogan on the board; course = estimate distributions; engineer designs RE.

### What he is establishing

The distribution of a random variable $X$ is about the **likelihood of observations under $X$**. Carefully: for continuous RVs it is not exact to treat every single range point as an event casually — consider a **subset** of the range; that subset has an inverse image in the sample space; that event has an underlying measure.

In the X-ray story, the event can be read as **obtaining that image** (those pixel observations). The distribution ultimately tells the **likelihood of obtaining this particular image** under the random variable.

**Key observation (write it down):** the distribution function **completely specifies the underlying sample space**. If you know the distribution, you know everything you are supposed to know about the underlying experiment for this modeling purpose.

Traceback: RE happening; $P$ on the sample space; RV maps outcomes to reals; that **translates** $P$ into distribution functions on the data side. $P$ interprets as likelihood of events; the distribution interprets as likelihood of obtaining observations/images (he prefers “likelihood” carefully rather than overusing “probability” for every phrase).

This is the statement the course will justify by building algorithms: the **entire course is about estimating distribution functions** of different forms given observations from random variables, because that tells everything needed about the underlying experiment.

Engineer’s responsibility: design the scenario so the problem **fits this narrative** — identify the RE / sample space packaging, get data accordingly, design the problem, then apply an algorithm.

If you only memorize formulas without this mission, later algorithms float. Instead: estimate $P_X$ / $P_{X,Y}$ from data because that is knowing the experiment.

We now have the course mission. Still missing: labels — the second half of the X-ray problem.

### Analogy for this topic only

Imagine forecasting daily high temperatures for a city.

- You never list every air-molecule microstate.  
- **What can you still answer if you truly know the distribution of highs?** Every practical question about what temperatures this place tends to produce.

Wrong move: insist on writing the infinite microstate list before doing any science. Right move: treat the distribution as the operational specification of the experiment for data questions.

In lecture words: distribution completely specifies the sample space; course = estimate distributions from observations.

### Local picture

```
  RE + P on Ω
       --X-->  distribution on range regions
                 │
                 ▼
  "likelihood of obtaining this image"

  If distribution known → experiment known (for our purposes)

  Course: observations → estimate distributions
  Engineer: package RE + data to fit this story
```

**Notice:** he still says “image as d-dimensional RV” loosely — next topic tightens that speech.

### Bridge

The original problem also had **labels**. How do labels enter without breaking the shared experiment?

---

## Topic 6: Labels as a second RV; joints; ranges of $X$ and $Y$ (16:00–24:06)

### Where this sits on the master map

**LABEL + JOINT** box — complete the X-ray story. Warm-up: [label as RV](./PREREQUISITES.md#p7-label-rv).

### Board / screenshot

![Labels and joints](./screenshots/composites/ch06-topic-06-labels-joint-panel1of2.png)

![X in R^d and Y in R^k](./screenshots/composites/ch06-topic-06-labels-joint-panel2of2.png)

**Figure — ~16:00–24:00:** loose “image is RV” meaning; label as second RV on same $\Omega$; $X\in\mathbb{R}^{d}$, $Y\in\mathbb{R}^{k}$; tumor box $k=3$; categories; joint.

### What he is establishing

When he says “an image can be seen as a $d$-dimensional random variable,” he means carefully: the image is an **element of the range** of a function $X:\Omega\to\mathbb{R}^{d}$. He admits the short sentence is a **wrong sentence** if taken literally — the matrix is not the function — but he will still say it, and books say “observations from a random variable,” assuming you decode: data = range points of $X$.

The full problem also had **labels**: X-ray taken and a label assigned. Model that with **joint distributions**. The label is **another random variable on the same sample space**. Calling it a “label” is semantic nomenclature; mathematically it is just another map on $\Omega$. The entire experiment = image acquisition + label assignment = two RVs. The slogan “distribution completely specifies the sample space” remains true in this two-RV packaging: the object of interest becomes the **joint**.

He previews a generalist view: he does not usually start from “supervised vs unsupervised” slogans; **everything is a special case of distribution estimation** (full argument later — even supervised can be seen through that lens; he flags it as nomenclature and postpones the detailed unification).

Ranges: most general, $X\in\mathbb{R}^{d}$ and $Y\in\mathbb{R}^{k}$ (student question: does $Y$ also map into $\mathbb{R}^{d}$? Not necessarily — $Y$ has its own codomain dimension $k$). Binary labels: $k=1$ with values in a subset such as $\{0,1\}$, or discrete. Tumor localization: put a **box** around the tumor; a rectangle can be represented by **center, height, width** — often three numbers, so $k=3$ (sometimes people count four coordinates for a box; he uses the three-number story). Five categories: $Y$ discrete $\{0,1,2,3,4\}$ rather than a continuum $\mathbb{R}^{k}$; discrete codes are often pure **categorization** without intrinsic numeric meaning — the integers are tags, not measurements of size.

He repeats because it is the key latch: once this RV packaging is clear, everything else is algorithmic extension. Two RVs ⇒ there is a **joint distribution**.

If labels live in a second universe with no shared $\Omega$, joints are fake. Instead: same experiment, two maps, joint $P_{X,Y}$.

We now have image+label as a joint pair. Still missing: how a finite **dataset** is written in standard notation.

### Analogy for this topic only

A photo finish with a judge’s call:

- The photo of the finish line  
- A tag: win / foul / redo  
- Both come from the **same race story**  

**Where is the joint?** In races that produce photo **and** call together. Gluing a photo from one stadium to a random tag from another stadium is not a joint experiment.

Wrong move: train on images from hospital A and labels from hospital B with no shared patients. Right move: pairs from one experiment story.

In lecture words: label = second RV on same $\Omega$; joint of image and label.

### Local picture

```
  Same Ω
    X: Ω → R^d     (stacked image)
    Y: Ω → R^k or {0,1,…,C-1}

  Examples for Y:
    binary disease     → k=1, {0,1}
    5 classes          → discrete tags
    tumor box          → (center, h, w) ≈ k=3

  Joint distribution of (X,Y)
```

**Notice:** “label” is a name; the math is “another RV.”

### Bridge

How do practitioners write a finite list of pairs and connect it to $P_{X,Y}$?

---

## Topic 7: Dataset, feature/label space, i.i.d. cliffhanger (24:06–26:50)

### Where this sits on the master map

**DATASET + STOP**. Warm-up: [dataset sampling](./PREREQUISITES.md#p8-dataset).

### Board / screenshot

![Dataset and i.i.d. notation](./screenshots/composites/ch07-topic-07-dataset-iid-panel1of1.png)

**Figure — ~24:06–26:50:** joint nomenclature; feature/label space; $n$ tuples; sampled i.i.d. from $p_{XY}$; i.i.d. next.

### What he is establishing

Because there are two RVs, there is a **joint distribution**. He has **not** fully defined “machine learning” yet — that comes later.

You start with a **data set**: $n$ images, each with a label. Names:

- $x$ lives in the **data space / feature space** (the stacked image world $\mathbb{R}^{d}$)  
- $y$ lives in the **label space** (he also says **action space** — returns later; think “what is attached / decided on the $y$-side”)

The dataset is a set of tuples $(x_i,y_i)$ for $i=1,\ldots,n$. Conventional notation: this is **sampled** (a **tilde** $\sim$ on the page) **i.i.d.** from the underlying joint distribution $p_{X,Y}$.

Read the symbols slowly: tilde / $\sim$ means “drawn from”; $p_{X,Y}$ is the joint of features and labels; **i.i.d.** is a new term whose definition is postponed to the **next lecture**. Densities remain ahead as well. Machine learning as a full formal definition is also not finished in this lecture.

If the dataset is treated as an arbitrary bag of files with no joint generative story, estimation has no target. Instead: $\{(x_i,y_i)\}$ as samples from $P_{X,Y}$.

We now have the standard supervised dataset notation. Still missing next time: precise i.i.d.; later: densities and algorithms that estimate the joint / conditionals.

### Analogy for this topic only

A folder of many patient charts:

- Each chart: one image plus one doctor’s mark  
- Modeling claim: charts behave like draws from one hospital process that produces image-and-mark pairs  

**What is not claimed yet?** The exact meaning of “i.i.d.” — that word is left for the next class.

Wrong move: mix charts from incompatible processes and still pretend one joint model. Right move: one joint story, many paired samples.

In lecture words: dataset $\{(x_i,y_i)\}\sim_{\mathrm{iid}} p_{X,Y}$; i.i.d. next.

### Local picture

```
  Feature space: x ∈ R^d
  Label space:   y ∈ R^k or discrete

  D = { (x1,y1), …, (xn,yn) }

  D  ~_iid  p_{X,Y}     (tilde = "sampled from")

  Next lecture: what iid means
  Later: densities; ML definition
```

**Notice:** the whole lecture prepared this one line of notation.

### Bridge

Unpack i.i.d.; continue toward densities and algorithms that estimate the distributions the mission requires.

---

## External references

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| [3Blue1Brown — But what is a neural network? (pixels as inputs)](https://www.youtube.com/watch?v=aircAruvnKk) | Topic 2 | Images as big input vectors (stacking intuition) |
| [StatQuest — The main ideas of fitting a line to data](https://www.youtube.com/watch?v=PaFPbb66DxQ) | Topic 5–7 | Data as samples; model fits a distributional/statistical story |
| [Seeing Theory — Random Variables](https://seeingtheory.brown.edu/probability-distributions/index.html) | Topics 1, 3–4 | RV vs realized numbers visually |
| [Lec 05 package (joints / labels prep)](../06-Lec05-Recap-Probability-Theory-Part2/NOTES.md) | Topics 5–6 | Joint of two RVs on one $\Omega$ |
| [StatQuest — Maximum Likelihood, clearly explained](https://www.youtube.com/watch?v=XepXtl9YKwc) | Topic 5 | “Likelihood of data under a model” language |
| [Google ML Crash Course — Training and Test Sets](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/video) | Topic 7 | Dataset of $(x,y)$ pairs in practice |

---

## Sources

- Video: [Lec 06 Understanding a Chest X-Ray as Sample from Distribution](https://www.youtube.com/watch?v=bdcvsSNAHIk)  
- Channel: NPTEL — Indian Institute of Science, Bengaluru  
- Skill: `youtube-lecture-tutor`  
- Captions cleaned via TRANSCRIPT.md / claim sheets  
