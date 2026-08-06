# Prerequisites — warm-up before Lec 06 (X-ray as sample from a distribution)

> **Do this first** if “data as random variables” still feels like jargon.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on [Lec 05](../06-Lec05-Recap-Probability-Theory-Part2/PREREQUISITES.md) (joints / conditionals / margins).  
> Still a **warm-up** — but detailed enough if you do not know the basics.  
> **Goal:** every map word in this video (range, stacking, data≠P, distribution specifies Ω, label RV, dataset ~ P_{X,Y}) feels necessary.

```
  After this warm-up you can say:

  "An image on disk is a grid of numbers I can stack into a vector in R^d."
  "That vector is a point in the range of a random variable X — not a probability."
  "Probability lives on events (via preimages), not on pixel intensity ink."
  "Knowing the distribution means knowing the experiment for practical purposes."
  "A label is another RV Y on the same Ω; together they have a joint."
  "A supervised dataset is n pairs (x_i,y_i) treated as samples from that joint."
```

**How to use:** top to bottom; every **Micro** on paper; read **Purpose for the video** in each section.

**Warm-up → lecture boxes**

```
  §1  (Ω,F,P) + RV + range     ──►  Topics 1–3
  §2  Grid → stack → R^d       ──►  Topic 2
  §3  Realizations             ──►  Topic 3
  §4  Data value ≠ probability ──►  Topic 4 (critical trap)
  §5  Preimages + “likelihood of image” ──► Topics 4–5
  §6  Distribution specifies Ω ──►  Topic 5
  §7  Two RVs: feature + label ──►  Topic 6
  §8  Dataset & joint sampling ──►  Topic 7
```

---

## 1. Probability triplet, RV, and range (reload)

<a id="p1-triplet-rv"></a>

### Purpose for the video

The lecture will say an X-ray on your computer is an **element of the range** of a random variable.  
If “range of a function” and “$P$ on events” are foggy, that sentence collapses.

### Core objects

| Object | Plain English |
|--------|----------------|
| Random experiment (RE) | Procedure that can produce different outcomes (image a patient) |
| Sample space $\Omega$ | Set of outcomes you admit for that RE |
| Events / $F$ | Allowed subsets of $\Omega$ you may score |
| Probability $P$ | Sizes events in $[0,1]$ |
| Random variable $X$ | **Deterministic map** $X:\Omega\to\mathbb{R}^{d}$ (same $\omega$ → same vector) |

**What is random?** Which $\omega$ occurs — not the rule $X$ flipping.

### Domain vs range (do not mix these)

| Word | For the map $X:\Omega\to\mathbb{R}^{d}$ |
|------|----------------------------------------|
| **Domain** | $\Omega$ — inputs (experiment outcomes) |
| **Codomain / range space** | $\mathbb{R}^{d}$ (or the set of vectors $X$ can hit) — outputs |
| **Value $X(\omega)$** | One specific output for one outcome |

Books say “observations from a random variable.” They mean: you got **outputs** $X(\omega)$, i.e. points in the range — **not** that the PNG file *is* the function $X$.

### Range and realizations

The **range** of $X$ is the set of vectors $X$ can output.  
A **realization / observation / data point** is one concrete vector you got when the experiment ran — a **point in the range**, not “the probability of that pixel.”

```
  domain Ω  --X-->  range R^d
                 data x = X(ω) ∈ R^d   ← what sits on the computer
```

### Analogy — purpose

A **sensor printout** after a clinic visit: the number “120” is a reading, not “the chance of a visit.”  
The process behind the visit ≈ $\Omega$; the sensor rule ≈ $X$; the printout ≈ range point.

**Why the video needs this:** Topics 1–2 redefine the X-ray file as exactly that printout story in $\mathbb{R}^{d}$.

---

## 2. Images as grids; stacking into vectors

<a id="p2-stack-image"></a>

### Purpose for the video

Computers store images as **matrices**. Probability in this course talks about **vectors in $\mathbb{R}^{d}$**.  
Stacking is the bridge — not a second physical experiment.

### Grid and pixels

An image of size $P\times Q$ is a $P\times Q$ table of numbers (pixels). Software (e.g. OpenCV) hands you that matrix.

### Stacking procedure

```
  Row1: a b c
  Row2: d e f
  Row3: g h i

  stack rows →  (a,b,c, d,e,f, g,h,i)  ∈ R^{9}
  here d = P·Q = 9
```

General: stack all rows → one long vector in $\mathbb{R}^{PQ}$. Set **$d = PQ$**.  
Any data point in this course is treated as a **$d$-dimensional vector**.

**Micro (work this):** $2\times 2$ image

```
  [ 10  20 ]
  [ 30  40 ]
  stack rows →  x = (10, 20, 30, 40) ∈ R^4
  d = 2·2 = 4
```

Those four numbers are still intensities (or whatever the sensor wrote). They did **not** become probabilities by being listed in a row.

### “Viewing,” not a second magical mapping

A student in class asks: “Are we *mapping* the rectangle to a vector?”  
Teacher: you are **viewing** the same $PQ$ numbers as a $PQ$-dimensional vector. Same pixels; different bookkeeping. (Later, models also treat text and other data as $d$-vectors the same way.)

### Analogy — purpose

A **spreadsheet** with $P$ rows and $Q$ columns.  
“Is it a table or a list of $PQ$ cells?” Both — concatenate rows and you have one list.  
The lecture does that to images so they sit in $\mathbb{R}^{d}$ like every other **feature vector**.

**Why the video needs this:** Topic 2’s take-home is “every image is a range point in $\mathbb{R}^{d}$.” Stacking is how the rectangle becomes that point.

**Trap:** thinking stacking “maps physics into probability.” Stacking is only a **view** of the same numbers.

---

## 3. Realizations and repeated experiments

<a id="p3-realizations"></a>

### Purpose for the video

If you photograph 100 students with a one-pixel camera, you get **100 realizations** — 100 points from the range of one RV.  
That is the statistical modeling move when physics is unknown.

### Definition

Each time the RE runs and you measure, you get one range point: a **realization**.

```
  RE run #1 → x^(1) ∈ R^d
  RE run #2 → x^(2) ∈ R^d
  …
  RE run #n → x^(n) ∈ R^d
```

### Why statistics here

When you **do not know the physics** of the full imaging process, you still can:

1. Repeat the experiment many times.  
2. Collect realizations.  
3. Model with distributions (later: estimate them).

### Analogy — purpose

**Weigh 100 apples** from a orchard with a scale.  
Each weight is a realization. You never “know the orchard’s soul,” but the histogram of weights tells you what the orchard tends to produce.

**Why the video needs this:** Topic 3’s single-pixel / 100-students story is exactly this apple-scale idea for pixels.

---

## 4. Critical trap: data values are not probabilities

<a id="p4-data-not-p"></a>

### Purpose for the video

This is the **most important beginner trap** in the lecture.  
Pixel intensities (0–255), or values normalized to $[0,1]$, are **still just numbers in the range**. They are **not** probabilities.

### Clear split

| Object | Is it a probability? |
|--------|----------------------|
| Pixel value 173 | **No** — a real number / coordinate |
| Vector $x\in\mathbb{R}^{d}$ | **No** — a range point |
| $P(\text{event})$ | **Yes** — sizes a set of outcomes |
| $P_X$ of a **region** of possible $x$ | **Yes** — pushforward measure of a set |

```
  WRONG:  "pixel=0.7 so probability is 70%"
  RIGHT:  "pixel=0.7 is the measured intensity;
           probability asks how often such intensities occur under the RE"
```

### Classroom trap (exactly what a student asked)

| Step people try | Why it fails |
|-----------------|--------------|
| Pixels run 0–255 (or 0–25 in a toy scale) | Still **coordinates**, not $P$ |
| Divide by 255 → values in $[0,1]$ | Still **coordinates** in $[0,1]$, not $P(\text{event})$ |
| “Now they look like probabilities” | Looking like $[0,1]$ does **not** make them event sizes |

A probability is a number assigned to an **event** (or to a region of possible images via the distribution). A bright pixel is just “how bright.”

### Analogy — purpose

A **thermometer** reading $37.0^\circ$C is not “37% chance of fever.”  
It is a number on the scale. Chance lives in how often readings fall in regions, under a model of who you measure.

**Why the video needs this:** Topic 4 exists almost entirely for this correction after a student asks about normalizing to $[0,1]$.

**Trap:** “If I normalize pixels to $[0,1]$, they become probabilities.” **False.**

---

## 5. Preimages: how probability attaches to images

<a id="p5-preimages"></a>

### Purpose for the video

If data values are not probabilities, **where** is probability?  
On **events** in $\Omega$, connected to number-regions by **inverse images**.

### Inverse image

$$
X^{-1}(S)=\{\omega:X(\omega)\in S\}
$$

$S$ = set of possible vectors (a region in $\mathbb{R}^{d}$).  
$X^{-1}(S)$ = outcomes that would have produced a measurement in $S$.  
$P(X^{-1}(S))$ = chance of landing in that region (distribution language).

For continuous models, talk about **subsets/regions** of the range, not “probability of one exact floating-point image” as a singleton casually.

### Analogy — purpose

A **postal code map**: regions of addresses (range regions) pull back to sets of streets (events).  
The post office sizes delivery zones; it does not treat “house number 12” as a probability.

**Why the video needs this:** Topics 4–5: “likelihood of obtaining this image” means sizing the event / region story via the distribution — not reading the pixel ink as $P$.

---

## 6. “Distribution completely specifies the sample space”

<a id="p6-dist-specifies"></a>

### Purpose for the video

The lecture’s **key slogan**: if you know the distribution of the RV, you know everything you need about the underlying experiment for this modeling game.  
That slogan becomes the **course mission**: estimate distributions from data.

### Why the slogan is taught

```
  RE + P on Ω
       --X-->  distribution P_X on range regions
```

$P$ answers: likelihood of events.  
$P_X$ answers: likelihood of measurement-regions / “seeing images like this.”  
If $P_X$ is known, pushforward questions about data are answered — that is “specifies the sample space” in the course’s operational sense.

### Course mission (preview)

Estimate distribution functions of various forms from observations of RVs.  
Engineer designs: pick RE packaging, collect data, fit problem into this narrative, then algorithm.

### Continuous care (one sentence)

For continuous images, “likelihood of this exact floating-point grid” is delicate; the clean objects are **regions / subsets** of possible images and their preimages. The lecture still says “likelihood of obtaining this image” as intuition — keep the subset picture underneath.

### Analogy — purpose

A **weather climate model**: if you truly know the distribution of tomorrow’s temperature readings, you know what the weather “experiment” produces for practical forecasting — even if you never list every microstate of the atmosphere.

**Why the video needs this:** Topic 5 installs the slogan and the course identity: ML math here is **distribution estimation**.

---

## 7. Two RVs: features $X$ and labels $Y$ on the same $\Omega$

<a id="p7-label-rv"></a>

### Purpose for the video

X-ray alone is not the full clinic story. Someone also assigns **disease / box / category**.  
That is a **second random variable** on the **same** experiment, not a second universe.

### Setup

```
  Same Ω (patient imaging + labeling story)
  X: Ω → R^d     image / features
  Y: Ω → R^k or discrete set    label / target
  Joint distribution of (X,Y)
```

“Label” is a **semantic name**. Math: just another RV on $\Omega$.

### Loose speech you will hear (and how to decode it)

| What people say | What they should mean |
|-----------------|------------------------|
| “The image is a random variable” | The image is a **range point** of a map $X:\Omega\to\mathbb{R}^{d}$ |
| “Observations from a random variable” | Sampled outputs $X(\omega)$ sitting in the range |
| “Label” | Another map $Y$ on the **same** $\Omega$ |

Saying “the PNG is the function $X$” is the **wrong sentence**. The PNG is one value $X$ produced.

### Range shapes for $Y$

| Task | Typical $Y$ |
|------|-------------|
| Binary disease | $k=1$, values in $\{0,1\}$ |
| 5 categories | discrete set $\{0,1,2,3,4\}$ (numbers are **tags**, not measurements) |
| Tumor box | e.g. center, height, width → often $k=3$ (here numbers *do* measure geometry) |

### Why this is still “distribution estimation”

Supervised learning with pairs $(x,y)$ is, in this course’s generalist view, about the **joint** of features and labels (or pieces of it: conditionals, margins — Lec 05 toolkit). The names “supervised / unsupervised” are packaging; the math target is still a distribution on the data the experiment produces. (He previews this; full argument later.)

### Analogy — purpose

One **shipping package**:

- Photo of the box ≈ $X$  
- “Fragile / not fragile” stamp ≈ $Y$  

Same package story ($\Omega$). Two readings. Joint asks about photo **and** stamp together.

**Why the video needs this:** Topic 6 is exactly this packaging for X-ray + label and tumor localization.

**Trap:** inventing a separate unrelated $\Omega$ for labels so “joint” has no shared story.

---

## 8. Dataset language and sampling from the joint

<a id="p8-dataset"></a>

### Purpose for the video

Practice hands you a **table of pairs** $(x_i,y_i)$.  
The lecture’s conventional story: those pairs are samples from the joint distribution $P_{X,Y}$.

### Names (feature / label / action)

| Name | Object |
|------|--------|
| **Feature space / data space** | where $x$ lives (for images: $\mathbb{R}^{d}$ after stacking) |
| **Label space** | where $y$ lives (tags, boxes, scores, …) |
| **Action space** | alternate name he uses for the $y$-side (returns later — think “what the system outputs / decides”) |
| **Dataset** | $\{(x_i,y_i)\}_{i=1}^{n}$ |

### Sampling slogan and the tilde

$$
(x_i,y_i)\;\sim_{\text{i.i.d.}}\; P_{X,Y}
\quad\text{or}\quad
(x_i,y_i)\;\tilde{\sim}\; P_{X,Y}
$$

| Symbol | Read as |
|--------|---------|
| $\sim$ or tilde | “is sampled from” / “is drawn from” |
| $P_{X,Y}$ or $p_{xy}$ | the **joint** distribution of features and labels |
| i.i.d. | independent and identically distributed — **defined next lecture** |

Read for now: each pair comes from the **same** joint story; pairs do not secretly use different generative rules. Precise i.i.d. axioms wait until next class. Densities also still ahead.

### End-to-end micro (one patient)

```
  1. RE: image patient + doctor assigns mark
  2. Image grid → stack → x ∈ R^d
  3. Label → y ∈ {0,1} or R^k
  4. One chart = one pair (x, y)   ← range points, not probabilities
  5. n charts = dataset ~ joint P_{X,Y}
  6. Course goal: estimate that joint (or pieces of it) from the charts
```

### Analogy — purpose

A **clinic archive of n folders**. Each folder has an image and a doctor’s mark.  
The modeling claim is: those folders behave like independent draws from one joint “how imaging+labeling works in this hospital world.”

**Why the video needs this:** Topic 7 writes the dataset notation and stops at the i.i.d. cliffhanger.

---

### Paper check (before NOTES)

1. Stack a $2\times 2$ image into a vector; what is $d$?  
2. Is a pixel value $0.8$ a probability? Why/why not? What if you only rescaled 0–255 into $[0,1]$?  
3. Where does probability live if not in the pixel ink?  
4. Decode: “the image is a random variable” → correct sentence in your own words.  
5. What does “distribution completely specifies the sample space” mean operationally?  
6. How is a disease label modeled relative to the image RV?  
7. Write a 3-line description of a dataset $\{(x_i,y_i)\}$ and what the tilde $\sim P_{X,Y}$ means.  
8. Tumor box: why might $k=3$? Why might class labels $0,1,2$ have no numeric “size”?

---

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).  
Prior: [Lec 05](../06-Lec05-Recap-Probability-Theory-Part2/NOTES.md).
