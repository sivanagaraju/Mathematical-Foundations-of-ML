# Lec 07 — IID Assumption

**Video:** [Lec 07 IID Assumption](https://www.youtube.com/watch?v=C83xmx80tMo) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 06 — X-ray as sample from a distribution](../07-Lec06-XRay-Sample-From-Distribution/NOTES.md)

---

## Table of Contents

1. [Topic 1 — Identically distributed; two views of N points](#topic-1-identically-distributed-two-views-of-n-points-0000–0200) (00:00–02:00)
2. [Topic 2 — Non-IID breaks; event independence](#topic-2-non-iid-breaks-event-independence-0200–0422) (02:00–04:22)
3. [Topic 3 — Independence across points, not dimensions](#topic-3-independence-across-points-not-dimensions-0422–0736) (04:22–07:36)
4. [Topic 4 — Two viewpoints, sampling, and when IID language applies](#topic-4-two-viewpoints-sampling-and-when-iid-language-applies-0736–1101) (07:36–11:01)
5. [Topic 5 — Design choice of dataset; test from the same P](#topic-5-design-choice-of-dataset-test-from-the-same-p-1101–1529) (11:01–15:29)
6. [Topic 6 — Labels vs one RV; supervised names; ML = estimate P](#topic-6-labels-vs-one-rv-supervised-names-ml--estimate-p-1529–2103) (15:29–21:03)
7. [Topic 7 — Labels synthetic; Q&A recap; formal ML problem](#topic-7-labels-synthetic-qa-recap-formal-ml-problem-2103–3041) (21:03–30:41)
8. [External references](#external-references)
9. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: unpack the **IID** line on a dataset so “$D\sim P$” is not a magic spell. Method: define **identical** (same distribution / stable experimental conditions) and **independent** (product of joints across **points**), then show two equivalent stories of $n$ samples. Fork: independence is **not** “pixels inside one image are independent.”

**Worldview arc:** from bare tilde notation (Lec 06) **to** precise IID + design of what counts as one experiment **to** the course slogan **given $D$, estimate $P$**.

### System context

```
  ╔══════════════════════════════════╗
  ║ Outside                          ║
  ║ · Lec 06: data as range points   ║
  ║ · densities / algorithms later   ║
  ╚══════════════╤═══════════════════╝
                 │ this lecture unpacks
                 ▼
        ┌────────────────────┐
        │ IID assumption     │
        │ + ML = estimate P  │
        └────────────────────┘
```

### Main blueprint

```
  Dataset of n points
       │
       ├─ identical: same P (conditions not shifting)
       └─ independent: across POINTS (not feature dims)
       │
       ▼
  Two views of the same D:
    A) n realizations of one RV
    B) one realization of n RVs (need full IID words)
       │
       ▼
  Sampling = multiple trials of RE;  ~  means sample from
       │
       ▼
  Design: what is one experiment? (faces vs buildings)
  Train/test assumed same P
       │
       ▼
  ML core: given D ~ unknown P, estimate P
  (joints / conditionals / margins / sample)
```

### Scenario walkthrough

1. Write $n$ X-rays as samples from one joint law.  
2. Identical: same hospital process story for all $n$.  
3. Independent: chart $i$ independent of chart $j$ (not pixel independence).  
4. Factor the joint of all $n$ as a product of individuals.  
5. Choose whether faces+buildings are one RE or two.  
6. Demand test data share the assumed $P$.  
7. State the job: given $D$, estimate $P$ (flavors: classifier, generative, …).

### Failure / contrast path

```
  Train lung X-ray, test brain MRI as if same P   ──X──► identical breaks
  Read IID as "pixels independent"                ──X──► wrong axis
  Test from a different experiment silently       ──X──► expected failure
  Treat supervised/unsupervised as deep ontology  ──X──► miss "estimate P"
```

### STOP / out of scope

Full density theory, full generalization/overfitting theory (hospital question deferred), rigorous algorithm list (next classes), philosophy of “good labels” as a solved math problem.

### Load-bearing claims (closed-book)

- Identical = same $P$ for all points; experimental conditions stable.  
- Event independence: $P(A\cap B)=P(A)P(B)$; RVs independent ⇒ joint factors.  
- IID independence = across **data points**, not across **dimensions/pixels**.  
- Two views: $n$ realizations of one RV vs one realization of $n$ RVs.  
- Tilde = sample from; sampling = multiple trials of the RE.  
- Train/test assumed same distribution by construction.  
- ML slogan: **given $D$, estimate $P$** (and optionally sample).

**Speaker / course:** NPTEL IISc · MFML · Lec 07.

---

## Topic 1: Identically distributed; two views of $N$ points (00:00–02:00)

### Where this sits on the master map

**IDENTICAL** box — first letter group of IID. Warm-up: [identical](./PREREQUISITES.md#p1-identical).

### Board / screenshot

![Identically distributed](./screenshots/composites/ch01-topic-01-identical-panel1of1.png)

**Figure — ~00:00–02:00:** same distribution for two RVs; $N$ points as $N$ RVs with same $P_{X,Y}$.

### What he is establishing

Welcome back: what is **IID**? Start with **identically distributed**.

Take two $d$-dimensional random variables on the **same** sample space. Each has a distribution. If those distributions are **exactly the same**, it is like observing the same random variable twice — or observing two different RVs that share one law.

For a dataset of $N$ points: **identically distributed** means you may view the $N$ points as **one realization of $N$ random variables** that all share the **same** underlying distribution $P_{X,Y}$. Equivalently: $N$ realizations of one random variable with that distribution.

Machine learning almost always **assumes** all $N$ data points come from the exact same distribution. In plain language: the **underlying experimental conditions are not changing** across the dataset.

If conditions drift (different machines, different protocols) but you still pretend one $P$, identical distribution is a lie. Instead: one shared law for all rows — or admit non-IID.

We now have “identical.” Still missing: independence, and what happens when identity fails across domains.

### Analogy for this topic only

A clinic uses the **same** scanner protocol every day this month.

- Monday’s patients and Friday’s patients are modeled with the **same** image law.  
- **What if Friday the scanner firmware changes?** Then Friday is not identically distributed with Monday.

In lecture words: same $P_{X,Y}$ for all $N$ points; experimental conditions stable.

### Local picture

```
  View A:  one RV  →  n realizations  x1,…,xn
  View B:  n RVs   →  one realization each, all with same P

  Identical ⇔ same underlying distribution P
  ML default: experimental conditions not changing
```

**Notice:** “identical” is about **laws**, not about the numbers looking similar by eye.

### Bridge

What if identity fails — lung X-rays vs brain MRI — and what does the **I** in IID (independence) mean formally?

---

## Topic 2: Non-IID breaks; event independence (02:00–04:22)

### Where this sits on the master map

**NON-IID + IND(EVENTS)**. Warm-up: [independence](./PREREQUISITES.md#p2-indep-events).

### Board / screenshot

![Non-IID and independence](./screenshots/composites/ch02-topic-02-noniid-events-panel1of1.png)

**Figure — ~02:00–04:20:** non-IID research; lung vs brain; $P(A\cap B)=P(A)P(B)$; joint = product of marginals.

### What he is establishing

**Non-IID data** is a known research pain. If the IID assumption breaks, methods fail across domains. Example: build an algorithm on **lung X-rays**, then apply it to **brain MRI** — those are **not** identically distributed; the random experiment is not the same.

Next: **independence**. He had not defined it carefully earlier. For **events**, statistical independence means

$$
P(A\cap B)=P(A)\,P(B)
$$

(product of measures — he asks the class sum **or** product; the answer is **product**, not sum). Same idea for distributions: if the **joint** is the **product of marginals**, the random variables are called statistically independent. Technical term — related to English “independence,” not identical to casual usage.

For the dataset: view $N$ points as one realization of $N$ RVs with the same distribution, and assume those RVs are **statistically independent**. Then the joint law of all $N$ data points factors as the **product of the individuals**:

$$
p(z_1,\ldots,z_N)=p(z_1)\cdots p(z_N)
$$

If you multiply when dependence is real, the joint model is wrong. Instead: product factorization is exactly the independence assumption.

We now have both letters of IID at event/RV level. Still missing: *which* variables are independent — points or pixels?

### Analogy for this topic only

Train a detector only on **chest** scans, then drop it into **brain** MRI as if nothing changed.

**Does “same algorithm” make the experiments the same?** No — identity already failed. Independence is a separate second ingredient for how multiple chest charts relate to each other.

In lecture words: non-IID is a real problem; independence = product rule.

### Local picture

```
  Events:  P(A∩B) = P(A) P(B)   ⇒ independent

  RVs:     joint = product of marginals  ⇒ independent

  Dataset of n points:
    joint of (z1,…,zn) = ∏ p(zi)   under independence
```

**Notice:** lung≠brain is an **identity** failure; independence is defined by the product rule.

### Bridge

Each point is a $d$-dimensional vector. Does independence mean the $d$ coordinates are independent?

---

## Topic 3: Independence across points, not dimensions (04:22–07:36)

### Where this sits on the master map

**IND AXES** — critical kill shot. Warm-up: [points vs dimensions](./PREREQUISITES.md#p3-points-vs-dims).

### Board / screenshot

![Independence axes](./screenshots/composites/ch03-topic-03-indep-axes-panel1of1.png)

**Figure — ~04:22–07:30:** independence not across dimensions; image1 ⊥ image2; pixels dependent.

### What he is establishing

Each data point $X_i$ is $d$-dimensional — an amalgamation of $d$ scalar RVs. You *could* talk about independence among those scalars. **Is that what IID says? No. Very important.**

Independence in the **IID assumption** does **not** mean the vector-valued RV is independent **across its dimensions**. It only means statistical independence **between the individual data points** you drew.

In images: **image one independent of image two**. **Not:** “first pixel independent of second pixel inside image one.”

Some algorithms *do* assume independence across dimensions — that is an extra modeling choice. **In general we do not.** Different pixels **almost always** have dependence.

Two axes of variability:

1. Inside one point: $d$ scalars (not the IID independence claim).  
2. Across points: $N$ observations (this *is* the IID independence claim).

Every data point is still a range element of a RV; $N$ of them share one distribution and are independent across the $N$. **That package is the IID assumption.**

If you force pixel independence, you may destroy the image’s structure. Instead: keep within-image dependence; assume across-patient independence (as a model).

We now have the critical axis correct. Still missing: how the two dataset viewpoints interact with “identical,” and what sampling means.

### Analogy for this topic only

Two patients in different beds.

- **Independent charts** (across patients) can still mean each patient’s **own** organs are highly dependent.  

**Do independent charts force independent organs inside one body?** No. Forcing every organ measurement inside one patient to be independent just because you heard “IID” confuses the axes. IID talks about the bed-to-bed story, not organ-to-organ inside one body.

In lecture words: independence across data points, not across data dimensions.

### Local picture

```
  One point:  (pixel1, pixel2, …, pixeld)  — usually DEPENDENT

  n points:   z1, z2, …, zn
              z1 ⊥ z2 ⊥ … under IID independence

  NOT claimed by IID:  pixel_a ⊥ pixel_b inside zi
```

**Notice:** $N\times d$ scalars exist if you flatten everything; independence is still among the $N$ vector blocks, not across the $d$ scalars.

### Bridge

How do the “$n$ rolls of one die” vs “$n$ dice” stories differ for the words independent and identical?

---

## Topic 4: Two viewpoints, sampling, and when IID language applies (07:36–11:01)

### Where this sits on the master map

**TWO VIEWS + SAMPLE**. Warm-up: [two views](./PREREQUISITES.md#p4-two-views), [sampling](./PREREQUISITES.md#p5-sampling).

### Board / screenshot

![Two viewpoints and sampling](./screenshots/composites/ch04-topic-04-two-views-sample-panel1of1.png)

**Figure — ~07:36–11:00:** independent + identical written; two views; tilde = sample from; sampling = multiple trials.

### What he is establishing

**Independent and identically distributed:** one observation of $N$ random variables, each with the exact same underlying distribution — **or** $N$ observations of one random variable.

Careful: if you take the viewpoint “$N$ observations of **one** random variable,” then “identically distributed” is **redundant** — there is only one RV, so you need not chant “identical.” Multi-RV **independence** language also sits more naturally in the other view (independence is a relation *between* RVs). Q&A line he corrects: under the single-RV / many-realizations packaging you need neither multi-RV independence talk nor identity talk the same way; under the multi-RV packaging you need full **IID**.

Textbooks write a dataset with a **tilde** meaning **sample from**. **Sampling** = conducting **multiple trials** of the underlying random experiment (toss the coin $N$ times). Those trials yield $N$ observations of a RV; the measure on $\Omega$ induces the distribution you later estimate.

From now on he will write: start with a dataset of $N$ tuples sampled from an underlying distribution — and you should hear either equivalent view. Independence is across different realizations / different RVs, **not** across data dimensions.

Q&A correction: if you only have multiple realizations of a **single** RV, you need not talk independence/identity as multi-RV properties. In the **multi-RV** interpretation you need full **IID**: identical + statistically independent, one realization each.

If you mix the two stories carelessly, “independence” becomes nonsense (one object cannot be independent of itself). Instead: pick a view; multi-RV view carries the full IID phrase.

We now can read $D\sim P$ carefully. Still missing: who decides what “same experiment” means for faces vs buildings, and test data.

### Analogy for this topic only

**One coin flipped 100 times** vs **100 fair coins flipped once**.

- Same spreadsheet of H/T.  
- The 100-coin story needs “all fair and independent.”  
- The one-coin story is just “100 trials of one experiment.”

In lecture words: two views; tilde = sample from; sampling = multiple trials.

### Local picture

```
  View A:  n trials of one RE/RV     → identity automatic
  View B:  n RVs, same P, independent → full "IID" language

  tilde ~  =  "sampled from"
  sampling = multiple trials of RE

  Independence across points, not dimensions
```

**Notice:** papers often write the multi-RV product form even when they think in trials.

### Bridge

Is “same hospital” required for identical? What about faces vs buildings — and test data?

---

## Topic 5: Design choice of dataset; test from the same $P$ (11:01–15:29)

### Where this sits on the master map

**DESIGN + TEST**. Warm-up: [train/test](./PREREQUISITES.md#p6-train-test).

### Board / screenshot

![Design choice and test distribution](./screenshots/composites/ch05-topic-05-design-test-panel1of1.png)

**Figure — ~11:01–15:20:** same machines interpretation; faces vs buildings; combine datasets; test from same distribution.

### What he is establishing

Student intuition: identical ≈ same hospital / same machines. He agrees it is **up for interpretation** — a **design choice**.

Example: dataset of **human faces** vs dataset of **buildings**. Same RE or not? You decide. Train on faces and test on buildings → **identical assumption breaks** by construction; expect failure. **Combine** both into one dataset and train jointly → you are packaging them as one experiment.

Joke/aside: if you include “all data in the world” (petabyte / internet scale), non-IID vanishes by packaging — like a huge lookup table (LLM-scale data as extreme design). Point: **what you call the dataset is your design choice**, but algorithms will **use** independence and identical properties you imposed.

**Test data:** new points you predict on must be ensured to come from the **same** distribution you assumed, because you will estimate that distribution; points from elsewhere make the algorithm **deemed to fail**. Use precise terms: “similarity” here means train/test same-distribution assumption by construction — not a fuzzy aesthetic.

If you silently change the experiment at test time, you are not testing the model — you are testing a broken assumption. Instead: match the assumed $P$, or admit domain shift / non-IID.

We now have design + test. Still missing: why split labels from data, and the one-line definition of ML.

### Analogy for this topic only

Study only **English essays**, then “test” on **Chinese essays** without saying so.

**Is the model stupid, or did you break same-$P$?** You broke same-$P$. Merging both languages into one training world is a different design.

In lecture words: packaging is a design choice; test must share the assumed distribution.

### Local picture

```
  Option 1: faces only  →  don't expect buildings
  Option 2: faces+buildings as one D  →  one broader RE

  Train ~ P_hat
  Test  should be ~ same P
  else assumption breaks → expected failure
```

**Notice:** “all data in the world” is a packaging joke, not a free lunch theorem.

### Bridge

Why call features and labels two RVs at all — and what *is* machine learning in one line?

---

## Topic 6: Labels vs one RV; supervised names; ML = estimate $P$ (15:29–21:03)

### Where this sits on the master map

**NOMENCLATURE + ML DEF**. Warm-up: [supervised names](./PREREQUISITES.md#p7-supervised-names), [given D estimate P](./PREREQUISITES.md#p8-given-d-estimate-p).

### Board / screenshot

![ML definition and nomenclature](./screenshots/composites/ch06-topic-06-ml-definition-panel1of1.png)

**Figure — ~15:29–21:00:** two RVs vs one; supervised/unsupervised/self-supervised; given D estimate P; classifier/generative flavors.

### What he is establishing

Q: why treat data and label as **two** RVs instead of one? **Mathematically you can** combine them. Historically people name problems: **data+labels = supervised**; **only data = unsupervised**. He finds unsupervised “only data” fuzzy for the same reason — it is still a RV and a distribution to estimate.

Fairer intuition: measurements are often **machine** readings; labels often inject **human judgment** → separate RV is natural. **Self-supervised learning:** take some dimensions of data as a fake label and estimate conditionals — pure interpretation. Example: **inpainting** — missing pixels are the “label”; train by masking and reconstructing. Dimensions of $x$ become $y$. That is why he did **not** open the course with a sacred supervised/unsupervised trichotomy.

Names he accepts as common speech:

- supervised — label exists  
- unsupervised — no label  
- self-supervised — some data dimensions used as label  

“Very pseudo distinction,” but people write that way. Label is still just another RV on the same $\Omega$ if you use that packaging.

**All machine learning (core slogan):** **given $D$, estimate $P$.** You sampled from an unknown underlying distribution; estimate it. Rest of the course is that.

He splits flavors of that job:

- **Discriminative** machine learning (as he uses the word here): estimate the distributional pieces you need (often conditionals), without necessarily learning to draw new full samples.  
- **Generative** machine learning: estimate the underlying distribution **and** learn to **sample** from it.

Concrete targets:

- $P(Y\mid X)$ — **classifier**; if $Y$ discrete → **classification**; if $Y\in\mathbb{R}^{k}$ → **regression** (**regressor**)  
- $P(X\mid Y)$ — conditional data estimation  
- $P_{X,Y}$ — generative joint  

Underneath: RE, $\Omega$, RV, distributions, samples — then “100 things” on those distributions.

If you memorize supervised vs unsupervised without “estimate $P$,” you miss the course spine. Instead: names are packaging; the math job is distribution estimation from $D$.

We now have the ML slogan. Still missing: how synthetic labels can be; formal write-up; close of probability arc.

### Analogy for this topic only

A photo with a sticky note.

- Photo = machine measurement  
- Sticky note = human tag  
- **Or** hide a corner of the photo and call the missing corner the “tag” (self-supervised)

**Is the sticky note a different universe?** No — same experiment packaging, optional second RV.

In lecture words: nomenclature is pseudo; core problem is given $D$, estimate $P$.

### Local picture

```
  Packaging:
    supervised:     (x,y) pairs
    unsupervised:   x only
    self-supervised: y carved from x (e.g. mask → inpaint)

  Core:
    D ~ unknown P
    job: estimate P  (joints / cond / marg)
    generative: also learn to sample

  P(Y|X)  → classifier / regressor
  P(X,Y)  → generative joint
```

**Notice:** folding $Y$ into $X$ is always legal; names change the conversation.

### Bridge

Are labels “real”? And can we write the formal problem on the board?

---

## Topic 7: Labels synthetic; Q&A recap; formal ML problem (21:03–30:41)

### Where this sits on the master map

**SEMANTICS + FORMAL + STOP**. Warm-up: [given D estimate P](./PREREQUISITES.md#p8-given-d-estimate-p).

### Board / screenshot

![Labels synthetic and formal ML problem](./screenshots/composites/ch07-topic-07-semantics-formal-panel1of2.png)

![Formal problem given D estimate P](./screenshots/composites/ch07-topic-07-semantics-formal-panel2of2.png)

**Figure — ~21:03–30:40:** synthetic labels; Q&A on axes; formal “given D estimate P”; probability arc closes.

### What he is establishing

“Good label space” is not a pure math object — too philosophical (how society divides people, spam definitions that change). Labels are often **extremely synthetic**. Contrast: flying an aircraft needs a handful of physical parameters; classifying man/woman or spam may need huge models because **semantics are pseudo** and shifting.

Assignment teaser: random labels on a dataset can still be fit by a network — shows fragility of human divisions. Engineers are not hired to settle philosophy: **given labels, build the algorithm asked for.**

Q&A recap: independence is across **data points**, not dimensions; if you flatten to $N\times d$ scalars (he says “$N$ cross $D$”), independence still groups by the $N$ **vector-valued** RVs — not across the $d$ scalar dimensions inside a point. Student concern that “test must match train distribution” might encourage overfitting to one hospital — **hold that question** for the rest of the course (good question; not answered here).

Notation hygiene (he is reminded of earlier board habits): **lowercase** for scalar coordinates of a vector RV ($x_1,\ldots,x_d$ along dimensions); **uppercase** for vector-valued RVs ($X$); course default is vector RVs.

**Formal problem (board):** all problems in ML — given dataset $D$ sampled from unknown underlying distribution $P_X$ (write $P_X$ not $P_{X,Y}$ because $Y$ can be folded into $X$), **estimate $P_X$**, and optionally **learn to sample** (generative). Estimate the distribution or conditionals/marginals on top of it.

This **concludes the basics of probability theory** arc. Next classes: rigorously define ML problems and walk algorithms.

If you leave with only “IID is good,” you missed the destination. Instead: IID is the sampling assumption; the job is estimate $P$ from $D$.

We now have the formal ML problem on the board and the probability primer closed. Still missing next: rigorous problem statements and the algorithms that estimate $P$.

### Analogy for this topic only

You are handed a box of tickets and told “estimate the mix, maybe draw more like it.”

- Ticket tags might be silly human stickers.  
- **Still:** estimate the unknown law of the box.  

**Must labels be philosophically pure before you estimate?** No — freezing for purity is not the engineering job. Solve the estimation problem you were given.

In lecture words: given $D\sim P$ unknown, estimate $P$ (and maybe sample); probability primer ends.

### Local picture

```
  Formal ML problem (this course spine):

    D = {z1,…,zn}  ~  unknown P
    job: estimate P
         (or P(y|x), P(x|y), margins, …)
         optional: learn to sample (generative)

  Labels: often synthetic; still engineer the ask
  IID: assumption on how D was drawn
  Next: rigorous ML problem statements + algorithms
```

**Notice:** writing $P_X$ with $Y$ folded in matches “one RV” packaging from Topic 6.

### Bridge

Probability tools are in place. Next lectures name ML problems rigorously and develop estimators/algorithms.

---

## External references

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| [StatQuest — The Fundamental Concepts of Machine Learning](https://www.youtube.com/watch?v=Gv9_4yMHFhI) | Topics 6–7 | High-level “learn patterns from data” before formal $P$ |
| [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | Topic 3 | Pixels as dependent coordinates in one vector |
| [Seeing Theory — Conditional Probability](https://seeingtheory.brown.edu/basic-probability/index.html#section2) | Topic 2 | Product intuition for independence |
| [Lec 06 package (dataset ~ P)](../07-Lec06-XRay-Sample-From-Distribution/NOTES.md) | Topics 1,4 | Tilde and range-point setup |
| [Google ML Crash Course — Generalization](https://developers.google.com/machine-learning/crash-course/generalization/video) | Topic 5 | Train/test same-distribution intuition |
| [StatQuest — Maximum Likelihood](https://www.youtube.com/watch?v=XepXtl9YKwc) | Topics 6–7 | “Fit a distribution to data” energy |

---

## Sources

- Video: [Lec 07 IID Assumption](https://www.youtube.com/watch?v=C83xmx80tMo)  
- Channel: NPTEL — Indian Institute of Science, Bengaluru  
- Skill: `youtube-lecture-tutor`  
- Captions cleaned via timed transcript / claim sheets  
