# Prerequisites — warm-up before Lec 02 (probability foundations)

> **Do this first.** Then open [NOTES.md](./NOTES.md) at the **Executive Summary** map.  
> These are **basics** so the master map (random experiment → sample space → events → probability measure) does not freeze you.  
> Not a second lecture — short scenes only.

```
  After this warm-up you can say:

  "A set is a bag of allowed outcomes; a subset is some of them."
  "A measure is a fair way to size sets so we can compare them."
  "Uncertainty is about the situation; a decision is often still a single action."
  "An experiment can produce different outcomes; we list them before we 'score' them."
  "Last lecture's f was a mapping; this lecture sizes outcome-sets with a special measure P."
```

If you finished Lec 01, skim [function approximation warm-ups](../02-Lec01-Overview-Function-Approximation/PREREQUISITES.md) for domain/range/data pairs; the ideas below are the new probability vocabulary.

---

## 1. Sets: bags of things

<a id="p1-sets"></a>

A **set** is a collection of objects treated as a whole. Picture a bag with a label.

```
  Ω = { H, T }          coin faces
  D = {1,2,3,4,5,6}     die faces
```

You do not need fancy axioms yet. You need: “these are the allowed members.”

---

## 2. Subsets: some of the bag

<a id="p2-subsets"></a>

A **subset** is a smaller (or equal) bag made only from members of a larger set.

```
  Die Ω = {1,2,3,4,5,6}
  Even faces A = {2,4,6}   ⊂ Ω
  Empty set ∅ = { }        no members
```

In probability language later, many important subsets are called **events** (after we decide which subsets we are allowed to measure).

---

## 3. Functions: rules between sets

<a id="p3-functions"></a>

A **function** $f$ from set $X$ to set $Y$ assigns **exactly one** output in $Y$ to each input in $X$.

$$
f : X \to Y
$$

Last lecture: unknown $f$ mapped images to labels. This lecture will say a **random variable** is also a function (from outcomes to numbers) — deterministic as a map, even if the experiment feels “random.”

```
  outcome ω  ──g──►  number g(ω)
```

---

## 4. What “measure” means before probability

<a id="p4-measure"></a>

To compare sets you need a **sizing rule** — a **measure**.

On the real line, **length** is the familiar measure: the interval from 0 to 2 is “bigger” than from 0 to 1 because length 2 > length 1. In the plane, people use **area**; in space, **volume**.

```
  set A: [0,1]     length 1
  set B: [0,3]     length 3
  → B is larger under length measure
```

Probability will invent a special measure $P$ on outcome-sets, capped so the whole sample space gets size 1.

---

## 5. Uncertainty vs a final decision

<a id="p5-uncertainty"></a>

**Uncertainty:** you do not know which outcome will appear.  
**Decision:** you still often must pick **one** action (treat / do not treat; send to spam folder or not).

```
  uncertain model:  "maybe diseased"
  clinical decision: treat or not treat  ← one bit
```

The lecture’s joke: probability helps under uncertainty, but doctors and spam filters still output deterministic decisions.

---

## 6. Experiments and outcomes (pre–random experiment)

<a id="p6-outcomes"></a>

An **experiment** (everyday sense) is a procedure you run. An **outcome** is one possible result after it runs.

```
  Experiment: toss a coin once
  Outcomes:   H or T

  Experiment: open one email and read it
  Outcomes:   many possible email texts (abstract list)
```

Probability formalizes this as a **random experiment** whose full list of possible outcomes is the **sample space**.

---

## 7. Numbers between 0 and 1 as “sizes”

<a id="p7-unit-interval"></a>

Scores in $[0,1]$ are handy: 0 = none of the whole, 1 = the entire whole, 0.7 = seventy percent of a unit whole.

```
  0 --------0.7-------- 1
  empty               full sample space
```

When $P(A)=0.7$, people often *say* “70% chance,” but the math first only guarantees: $A$ got measure 0.7 under the rules of $P$.

---

## 8. From last lecture: why we need this tooling

<a id="p8-why-from-fa"></a>

**Function approximation (FA)** said: given pairs $D$, estimate unknown $f$. For planets and $F=ma$, physics sometimes writes $f$. For coin tosses and “diseased?” labels, physics is hard — so engineers use **many observations** (statistics).

```
  FA path:     D  →  estimate f
  stats path:  many outcomes  →  estimate patterns / measures
```

This lecture builds the probability objects under that statistical path. Distribution estimation as the twin of FA is the bridge the instructor wants you to feel.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
