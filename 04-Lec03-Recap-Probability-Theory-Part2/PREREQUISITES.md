# Prerequisites — warm-up before Lec 03 (random variables)

> **Do this first.** Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on [Lec 02 Part 1](../03-Lec02-Recap-Probability-Theory-Part1/PREREQUISITES.md) (RE, Ω, events, P).  
> This warm-up unlocks **why** we need a map from outcomes to numbers.

```
  After this warm-up you can say:

  "A function maps each input to exactly one output."
  "Ω lists abstract outcomes; numbers live in R or R^d."
  "A vector is an ordered list of numbers (an image can become one)."
  "A bridge object can turn abstract outcomes into numbers we can compute with."
  "Last lecture sized events with P; this lecture maps outcomes with X."
```

---

## 1. Function as a mapping

<a id="p1-functions"></a>

A **function** $f$ from set $A$ to set $B$ assigns **exactly one** element of $B$ to each element of $A$.

$$
f : A \to B
$$

```
  a in A  ──f──►  f(a) in B
```

This lecture’s **random variable** is “just” a carefully chosen function whose domain is the sample space.

---

## 2. Recap: sample space Ω (from Lec 02)

<a id="p2-omega"></a>

After a **random experiment**, collect every possible **outcome** in a set $\Omega$.

```
  Coin RE:  Ω = {H, T}
  Clinic RE: Ω = { abstract process outcomes … }
```

$\Omega$ can be **non-numeric**. That is intentional — and it creates the problem this lecture solves.

---

## 3. Recap: events and probability measure P

<a id="p3-events-p"></a>

**Events** are the subsets of $\Omega$ we size. **Probability measure** $P$ assigns each allowed event a number in $[0,1]$ with rules like $P(\Omega)=1$.

```
  (Ω, F, P)   ← the probability triplet (named in this lecture)
```

Having $P$ does **not** yet give you a pixel vector for an X-ray. Something else is missing.

---

## 4. Real numbers and vectors

<a id="p4-reals-vectors"></a>

**Real numbers** $\mathbb{R}$ are the usual continuous numbers you add and multiply.

A **vector** in $\mathbb{R}^{d}$ is an ordered list of $d$ reals:

```
  (2.0, -1.5, 0.0)   is a point in R^3
```

Engineers like vectors because computers and linear algebra work on them. An image, after stacking pixels, is one long vector in $\mathbb{R}^{d}$.

---

## 5. Abstract vs concrete (sensor intuition)

<a id="p5-abstract-concrete"></a>

**Abstract:** “someone has a lung condition and visits a clinic.”  
**Concrete:** a machine outputs a matrix of brightness numbers.

```
  abstract story  ──sensor──►  numbers on disk
```

You need both: the abstract story for probability, the numbers for code.

---

## 6. Domain and range of a function

<a id="p6-domain-range"></a>

For $f:A\to B$:

| Word | Meaning |
|------|---------|
| **Domain** | Input set $A$ |
| **Range** (here) | Output set $B$ (or the set of values $f$ actually hits) |

When $X:\Omega\to\mathbb{R}^{d}$, domain is sample space; range lives among real vectors. **Data points** will be described as elements of that range.

---

## 7. Deterministic map vs “random feeling”

<a id="p7-deterministic-map"></a>

A function is **deterministic**: same input → same output.  
An experiment can feel **uncertain**: which outcome occurs is not known in advance.

```
  Uncertain: which ω happens?
  Deterministic: once ω is fixed, X(ω) is fixed
```

That split is why “random variable” is a misleading name — the map $X$ itself is not random.

---

## 8. Why this lecture exists (bridge from Lec 02)

<a id="p8-why"></a>

Lec 02 stopped at “size events with $P$.”  
ML practice starts with **folders of vectors**.

```
  Lec 02:  RE → Ω → events → P
  Lec 03:           + X: Ω → R^d  → data vectors
```

Without $X$, the triplet and the dataset live in two different worlds.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz: [quiz.html](./quiz.html) Part A = this file.  
Prior package: [Lec 02 Part 1](../03-Lec02-Recap-Probability-Theory-Part1/NOTES.md).
