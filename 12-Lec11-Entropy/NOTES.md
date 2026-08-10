# Lec 11 — Entropy

**Video:** [Lec 11 Entropy](https://www.youtube.com/watch?v=P6wjLz4dRTs) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 10 — Challenge With ML](../11-Lec10-Challenges-of-ML/NOTES.md)

---

## Table of Contents

1. [Topic 1 — Recap: density/dist, recipe, W1/W2 fix](#topic-1-recap-densitydist-recipe-w1w2-fix-0003–0333) (00:03–03:33)
2. [Topic 2 — Need distributional divergences](#topic-2-need-distributional-divergences-0333–0604) (03:33–06:04)
3. [Topic 3 — Surprisal I(A)=−log P(A)](#topic-3-surprisal-ia−log-pa-0604–1032) (06:04–10:32)
4. [Topic 4 — Average surprisal for a discrete law](#topic-4-average-surprisal-for-a-discrete-law-1032–1552) (10:32–15:52)
5. [Topic 5 — Entropy named; thermo aside; next](#topic-5-entropy-named-thermo-aside-next-1552–1756) (15:52–17:56)
6. [External references](#external-references)
7. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: start filling recipe step 2 — we need a way to say how far two distributions are. Method: import information theory, define **surprisal** of an event as $-\log P(A)$, then **entropy** as the average surprisal of a discrete law. Fork: entropy is about *one* distribution; pairwise divergences (what ML optimizes) come next, built from these tools.

**Worldview arc:** from “need $d(P,P_\theta)$” **to** “$I(A)=-\log P(A)$ and $H(p)=\mathbb{E}[-\log p(X)]$.”

### System context

```
  ╔══════════════════════════════════╗
  ║ Outside                          ║
  ║ · Lec 10: recipe p_θ → d → train ║
  ║ · Later: divergences / f-div     ║
  ╚══════════════╤═══════════════════╝
                 │ this lecture
                 ▼
        ┌────────────────────┐
        │ surprisal → entropy│
        └────────────────────┘
```

### Main blueprint

```
  ML recipe step 2 empty
       │
       ▼
  quantify distance between two distributions
  (same sample space)
       │
       │ path: information theory
       ▼
  event A  ──►  I(A) = −log P(A)   (surprisal)
       │         rules:
       │         · likely ⇒ small I
       │         · I(Ω)=0, I(∅)=∞
       │         · indep ⇒ I adds
       ▼
  discrete X ~ p
       │
       ▼
  H(p) = E[−log p(X)] = −∑ p(x_i) log p(x_i)
       │
       │  name: ENTROPY
       ▼
  ┌ · · · next lectures · · · · · · ┐
  │ build pairwise divergences for d │
  └ · · · · · · · · · · · · · · · · ┘
```

### Scenario walkthrough

1. Recall: train by minimizing $d(\text{true},\text{model})$.  
2. Ask: what is $d$? Need distributional divergence.  
3. Define surprisal of rare vs common events.  
4. Average surprisal under a discrete PMF → entropy formula.  
5. Name it entropy; stop before full divergence construction.

### Failure / contrast path

```
  Confuse entropy H(p) with distance d(p,q)     ──X──► wrong object
  Treat density height as PMF probability         ──X──► discrete only here
  Forget minus on log                             ──X──► “info” goes negative
  Set both w1 and w2 in R^d for affine model      ──X──► dimension mismatch
```

### STOP / out of scope

Full $f$-divergence catalog; continuous differential entropy details; KL derivation; using $H$ inside training loss this lecture; ERM equivalence (other playlist items).

### Load-bearing claims (closed-book)

- Speech: distributions; equations often densities (≈ dual when both exist).  
- Recipe: parametric model → $d$ → $\mathrm{arg\,min}$.  
- $I(A)=-\log P(A)$ = surprisal.  
- Design rules: likely small; $I(\Omega)=0$; empty infinite; independent add.  
- Discrete entropy $H(p)=-\sum p_i\log p_i=\mathbb{E}[-\log p(X)]$ (use $0\log 0:=0$).  
- Entropy = average surprisal of **one** law (bits per draw under $p$); divergences compare **two**.

**Speaker / course:** NPTEL IISc · MFML · Lec 11.

---

## Topic 1: Recap — density/dist, recipe, $W_1$/$W_2$ fix (00:03–03:33)

### Where this sits on the master map

**RECAP** — reloads the course contract before new math. Warm-up: [recipe link](./PREREQUISITES.md#p7-recipe-link).

### Board / screenshot

![Recap recipe and dimension fix](./screenshots/composites/ch01-topic-01-recap-recipe-panel1of1.png)

**Figure — ~00:03–03:30:** density vs distribution disclaimer; parametric recipe; $w_1\in\mathbb{R}^d$, $w_2$ scalar.

### What he is establishing

**Disclaimer for the whole course:** when he *speaks*, he says **distributions**, because probability as a measure is defined at that level. When he *writes equations*, he often uses **densities**, assuming a largely **one-to-one** link when both exist. Every definition written with densities has an equivalent distribution version and vice versa. Objectively we want the distribution (the probability measure); algorithmically we often estimate densities.

**ML problem restated:** estimate the underlying distribution given samples from it.

**Three-step recipe (reload):**

1. Assume a **parametric** form $P_\theta$ / $p_\theta$ on the unknown law.  
2. Define/compute a **distance / divergence** between the true law and the assumed law.  
3. Choose $\theta^\star$ by **minimizing** that divergence (training).

**Nomenclature:** $p_\theta$ (or $P_\theta$) is the **model distribution** — also “model” or “assumed distribution.” The distance takes two distributions and maps to a positive real.

**Correction from last time:** for an affine form $w_1^\top x+w_2$ with $x\in\mathbb{R}^d$, if both $w_1$ and $w_2$ were in $\mathbb{R}^d$, dimensions fail ($w_1^\top x$ is scalar; cannot add a vector). Fix: **$w_1\in\mathbb{R}^d$**, **$w_2$ scalar**.

Rest of course: many models, many divergences, many optimizers — same recipe skeleton.

You can now restate the course problem and the three steps with correct $w_2$. Still missing: an actual definition of the distance between two laws.

### Analogy for this topic only

A map legend can use “elevation” in speech and “color scale” on paper — same terrain, two representations. Density and distribution are that pair when both exist.

**If $w_2$ were a vector, what breaks?** Adding a number to a vector — the model formula is ill-typed.

In lecture words: distributions in speech, densities in math; recipe; $w_2$ scalar.

### Local picture

```
  samples D  →  estimate true law P
       │
       ├─ (1) model P_θ / p_θ
       ├─ (2) d(P, P_θ) ≥ 0     ← still undefined today
       └─ (3) θ* = argmin d

  affine fix:  w1 ∈ R^d ,  w2 ∈ R  (scalar bias)
```

**Notice:** “model distribution” = the parametric stand-in for reality, not the truth.

### Bridge

What object does step (2) still lack — and what field historically built distances between distributions?

---

## Topic 2: Need distributional divergences (03:33–06:04)

### Where this sits on the master map

**NEED d** — why this entropy detour exists. Warm-up: [entropy vs divergence](./PREREQUISITES.md#p6-entropy-vs-div), [recipe](./PREREQUISITES.md#p7-recipe-link).

### Board / screenshot

![Distributional divergences objective](./screenshots/composites/ch02-topic-02-need-divergences-panel1of1.png)

**Figure — ~03:33–06:00:** need $d$ between distributions; $f$-divergences; same sample space.

### What he is establishing

The key unfinished piece of the recipe is a way to **quantify distance between distribution functions**. Parameters are chosen so that a **divergence metric** between true and model is minimized — so that metric must exist as a mathematical object.

**Distributional divergences** are a topic in themselves. One large family is **$f$-divergences**. He will study that family more in a later generative-models course. In *this* course he will pick **one** commonly used $f$-divergence example and use it as the working metric.

**Stated objective:** given two distributions defined on the **same sample space**, quantify how close or far they are.

**Path taken today:** historically, people defined distances between distributions using **information theory**. So we install some info-theory ideas first, then later turn them into pairwise distances.

If you try to “train” without a $d$, step (3) has nothing to minimize. Instead: build the information toolkit (surprisal, entropy), then divergences.

You can now say why the course opens information theory. Still missing: the actual surprisal definition.

### Analogy for this topic only

You cannot rank two weather forecasts without a scoring rule. Divergence is the scoring rule for whole probability laws.

**What must the two laws share for “same sample space” to make sense?** The same underlying set of outcomes they put mass on.

In lecture words: need $d$ on pairs of distributions; $f$-divergences later; start via information theory.

### Local picture

```
  true P     model P_θ
      \         /
       \       /
        v     v
      d(P, P_θ)  →  minimize over θ

  today: tools for that d
  later: one f-divergence example in this course
```

**Notice:** entropy (next topics) is not yet $d$ — it is a property of a *single* law.

### Bridge

How do we assign a number — information — to a single event $A$?

---

## Topic 3: Surprisal $I(A)=-\log P(A)$ (06:04–10:32)

### Where this sits on the master map

**SURPRISAL** — event-level information. Warm-up: [$-\log$](./PREREQUISITES.md#p1-neglog), [events](./PREREQUISITES.md#p2-events), [independence](./PREREQUISITES.md#p3-independence).

### Board / screenshot

![Information content of an event](./screenshots/composites/ch03-topic-03-surprisal-panel1of1.png)

**Figure — ~06:04–10:30:** $I(A)=-\log P(A)$; likely vs rare; $\Omega$ and null; independence.

### What he is establishing

For an event $A$ (in the event space / $\sigma$-algebra $\mathcal{F}$), the **information content** associated with $A$ is

$$
I(A)=-\log P(A)
$$

In information-theory language this is also called the **surprisal** of $A$ (sometimes “self-information”).

**Intuition:** probability measures how likely $A$ is. Information measures how much news $A$ carries when it happens.

Concrete contrast:

- “The sun rises in the east” — high probability, **little** information.  
- “You win a huge lottery tomorrow” — low probability, **lots** of information.

Numbers (base 2): if $P=1/2$, $I=1$ bit; if $P=1/1024$, $I=10$ bits — eight more “surprises” of coin-flip size stacked. So the metric must be designed so that **higher likelihood ⇒ smaller information number**.

**Three requirements** he lists (and why $-\log$ fits):

| # | Requirement | Check with $I=-\log P$ |
|---|-------------|-------------------------|
| 1 | $A=\Omega$ conveys **zero** info (“something happens”) | $P(\Omega)=1\Rightarrow I=0$ |
| 2 | Null event conveys **maximum** (infinite) info | $P(\emptyset)=0\Rightarrow I\to+\infty$ |
| 3 | Independent events: combined info = **sum** of individuals | $P(A\cap B)=P(A)P(B)\Rightarrow I$ adds |

The **minus** is required so that larger $P$ yields smaller $I$ (and so $I\ge 0$ for $P\in(0,1]$).

**Micro numbers (base 2):** $P=1/2\Rightarrow I=1$ bit; $P=1/4\Rightarrow I=2$ bits; $P=1\Rightarrow I=0$.  
Next topic will apply this to singletons $\{X=x_i\}$ of a discrete RV.

If you drop the minus, “information” goes negative for ordinary events — useless as a size of surprise. Instead: surprisal $=-\log P$.

You can now compute $I(A)$ and check the three rules. Still missing: lift from one event to a whole distribution.

### Analogy for this topic only

A fire alarm that rings every hour conveys almost no news. An alarm that almost never rings is very informative when it does.

**If two independent alarms each carry 1 bit, how many bits when both fire?** $2$ — additivity.

In lecture words: $I(A)=-\log P(A)$; surprisal; three requirements.

### Local picture

```
  P(A) high  ──►  I(A) low
  P(A) low   ──►  I(A) high
  P(Ω)=1     ──►  I(Ω)=0
  P(∅)=0     ──►  I → ∞
  A ⊥ B      ──►  I(A∩B)=I(A)+I(B)

  I(A) = −log P(A)   = surprisal
```

**Notice:** this is still about **one event**, not yet about comparing two distributions.

### Bridge

Given surprisal for each outcome of a discrete RV, how do we score the **entire** distribution?

---

## Topic 4: Average surprisal for a discrete law (10:32–15:52)

### Where this sits on the master map

**AVERAGE** — event → distribution. Warm-up: [PMF + expectation](./PREREQUISITES.md#p4-pmf-expectation), [entropy formula](./PREREQUISITES.md#p5-entropy-formula).

### Board / screenshot

![Average information of a distribution](./screenshots/composites/ch04-topic-04-average-entropy-formula-panel1of1.png)

**Figure — ~10:32–15:50:** communication motivation; discrete PMF; $H=-\sum p\log p$.

### What he is establishing

**Historical motive:** communication under limited bandwidth. Engineers asked: given a bit budget, what information should you send to minimize loss? That forced a definition of “how much information a whole distribution embeds.”

**For now: discrete random variables.** Continuous point events have probability zero, so naive $-\log P(X=x)$ blows up — discrete first is intentional. For a value $x_i$, the mass function $p(x_i)$ **is** a probability, so surprisal of the event $\{X=x_i\}$ is

$$
-\log p(x_i)
$$

**Whole distribution:** take the **average** of those surprisals. In probability, average means **expectation** under $p_X$:

$$
H(p)=\mathbb{E}_{X\sim p}\big[-\log p(X)\big]
$$

**Process reading:** sample $X\sim p$ repeatedly; each sample has a surprisal; $H$ is the long-run **average surprise per sample**.

For discrete $X$,

$$
H(p)=-\sum_i p(x_i)\log p(x_i)
$$

(sum over values the RV takes; if $p(x_i)=0$, take that term as $0$ — [convention](./PREREQUISITES.md#p5-entropy-formula)). He corrects the board: expectation of a function multiplies by $p(x_i)$ — definition $\mathbb{E}[f(X)]=\sum p(x)\,f(x)$ with $f=-\log p$.

**Micro — fair coin, $\log_2$:** $H=1$ bit.  
**Micro — deterministic:** $H=0$.  
**Micro — $p=(1/2,1/4,1/4)$, $\log_2$:** $H=1.5$ bits (full expand in warm-up).

This average information **is** what will be called entropy (next topic names it); the formula is already the definition.

If you average surprisals without weighting by $p(x_i)$, rare outcomes dominate wrongly. Instead: expectation weights by occurrence probability.

You can now write $H=-\sum p\log p$ and compute it for a small PMF. Still missing: the official name and the closing worldview.

### Analogy for this topic only

Each weather outcome has a shock level (surprisal). Climate’s “average shock” is the typical day-to-day uncertainty of that climate — one number for the whole law.

**Why weight by $p(x_i)$?** Because common outcomes dominate everyday experience; rare shocks are rare.

In lecture words: average surprisal under discrete $p$; $H=-\sum p_i\log p_i$.

### Local picture

```
  values x_i with p(x_i)
       │
       │  surprisal −log p(x_i)
       ▼
  weight by p(x_i) and sum
       │
       ▼
  H(p) = −∑_i p(x_i) log p(x_i)
       = E[−log p(X)]
```

**Notice:** continuous case needs more care (differential entropy); he stays discrete for clarity.

### Bridge

What is the standard name of this average surprisal — and why does he care beyond the formula?

---

## Topic 5: Entropy named; thermo aside; next (15:52–17:56)

### Where this sits on the master map

**ENTROPY + STOP**. Warm-up: [entropy formula](./PREREQUISITES.md#p5-entropy-formula), [entropy vs divergence](./PREREQUISITES.md#p6-entropy-vs-div).

### Board / screenshot

![Entropy of a distribution](./screenshots/composites/ch05-topic-05-entropy-name-aside-panel1of1.png)

**Figure — ~15:52–17:50:** entropy named; thermo / adapt aside; continue next lecture.

### What he is establishing

The average surprisal associated with a distribution is called the **entropy** of that distribution:

$$
H(p)=\mathbb{E}_{X\sim p}\big[-\log p(X)\big]
$$

**Thermodynamics (heuristic link, not a proof):** one law of thermo is that the entropy of the universe increases — he translates that loosely as “the universe never stops surprising us.”

**Worldview aside (AI / jobs):** technological breakthroughs rearrange the world order; because “entropy increases,” new problems keep appearing. Society still rewards value creation, but **what counts as value** changes over time. Survival lesson: **adapt** — he ties this to evolutionary “fitness,” not merely smartest or strongest.

He is **not** claiming this is a theorem of ML. It is a closing story connecting the word “entropy” to change and surprise.

**Stop:** “this is entropy of a particular distribution; we stop here and continue in the next lecture” — expect the toolkit to continue toward **divergences** for recipe step 2. A typical next brick (preview only): average $-\log q(X)$ under true $p$ (cross-entropy) compared with $H(p)$ builds a pairwise gap such as KL — [warm-up](./PREREQUISITES.md#p6-entropy-vs-div).

If you leave thinking entropy *is* the training loss between true and model, you jumped ahead. Instead: entropy = uncertainty of one law; pairwise $d$ is the next construction.

You can now name $H(p)$ and place it in the map relative to future divergences.

### Analogy for this topic only

Entropy is how mixed a deck’s “typical surprise” is. Divergence (later) is how different *two* decks’ laws are.

**Is a high-entropy world “far from” a model?** Not until you define a pairwise score — entropy alone is not $d(p,p_\theta)$.

In lecture words: entropy = average surprisal; continue next lecture.

### Local picture

```
  surprisal I(A)     →  one event
  entropy H(p)       →  one distribution (average I)
  divergence d(p,q)  →  two distributions  [next]

  this lecture stops at H(p)
```

**Notice:** thermo/AI remarks are worldview glue, not exam formulas.

### Bridge

Next: use information-theoretic measures to define distances between pairs of distributions for the ML recipe.

---

## External references

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| [3Blue1Brown — Essence of entropy / information](https://www.youtube.com/watch?v=ErfnhcEV1O8) | Topics 3–5 | Visual surprisal and average information |
| [StatQuest — Entropy](https://www.youtube.com/watch?v=YtebGVx-Fxw) | Topics 3–5 | Discrete entropy formula walkthrough |
| [Khan Academy — Information entropy](https://www.khanacademy.org/computing/computer-science/informationtheory/moderninfotheory/v/information-entropy) | Topics 3–4 | $-\mathrm{log}$ and average surprise |
| [Lec 10 package — ML recipe](../11-Lec10-Challenges-of-ML/NOTES.md) | Topics 1–2 | Why $d$ is needed at all |
| [Lec 09 package — Density](../10-Lec09-Density-Function/NOTES.md) | Topic 1 | Density vs distribution dual |
| [Seeing Theory — Discrete probability](https://seeingtheory.brown.edu/basic-probability/index.html) | Topic 4 | PMF intuition before $H$ |

---

## Sources

- Video: [Lec 11 Entropy](https://www.youtube.com/watch?v=P6wjLz4dRTs)  
- Channel: NPTEL — Indian Institute of Science, Bengaluru  
- Skill: `youtube-lecture-tutor`  
- Captions cleaned via timed transcript / claim sheets  
