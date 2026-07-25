# Warm-up before Lec 02 (probability foundations)

> **Do this first** if sets, subsets, “size,” or functions feel shaky.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> This is a **slow basics** warm-up (you asked for stronger foundations) — still not a second full lecture.

```
  After this warm-up you should be able to say:

  "A set is a bag of distinct things; a subset is a smaller bag inside it.
   A function maps each allowed input to exactly one output.
   Function approximation means: estimate an unknown map f from data so we can
   predict on new inputs (this course’s ML goal before probability language).
   To compare bags we need a size (a measure), not just a list of members.
   Probability will be a special size on subsets of outcomes, with total size 1.
   Outcomes of a messy real process can live in an abstract bag — they need not
   start as numbers on a number line."
```

---

## Sets (bags of things)

<a id="p1-sets"></a>

A **set** is a collection of **distinct** objects. Order does not matter. Duplicates do not count twice.

**Membership.** We write $x \in S$ when $x$ is in the bag $S$, and $x \notin S$ when it is not.

**Empty set.** The empty bag $\emptyset$ contains nothing.

**Examples**

- Coin faces: $\Omega_{\text{coin}} = \{H, T\}$  
- Die faces: $\Omega_{\text{die}} = \{1,2,3,4,5,6\}$  
- Labels for disease: $\mathcal{Y} = \{0,1\}$

```
  bag S = { apple, banana, cherry }

  apple ∈ S     grape ∉ S     empty bag = ∅
```

In this lecture the big bag of **all possible outcomes** of an experiment is called the **sample space** and is written $\Omega$.

---

## Subsets and “events” (questions about outcomes)

<a id="p2-subsets"></a>

A **subset** $A \subseteq \Omega$ is a bag made only from members already in $\Omega$.

Everyday questions about an experiment become subsets:

| Everyday question | Subset of die $\Omega = \{1..6\}$ |
|-------------------|-----------------------------------|
| “Even?” | $A = \{2,4,6\}$ |
| “At least 5?” | $B = \{5,6\}$ |
| “Impossible?” | $\emptyset$ |
| “Anything possible?” | $\Omega$ |

**Union** $A \cup B$ = “in A or B (or both).”  
**Intersection** $A \cap B$ = “in both.”  
**Disjoint** means $A \cap B = \emptyset$ (no shared members).

```
  Ω = {1, 2, 3, 4, 5, 6}
       ┌─────────────┐
  A ── │ 2   4   6   │  even
       └─────────────┘
  B ── {5, 6}           ≥ 5
  A ∩ B = {6}
  A ∪ B = {2, 4, 5, 6}
```

The lecture will call (allowed) subsets of $\Omega$ **events**. You do not need the full formal theory of “which subsets are allowed” yet — just: **questions about outcomes = subsets**.

---

## Functions (reliable maps)

<a id="p3-functions"></a>

A **function** $f : \mathcal{X} \to \mathcal{Y}$ is a rule: each allowed input from $\mathcal{X}$ produces **exactly one** output in $\mathcal{Y}$.

```
  OK function                    NOT a function
  x=1 ──► 2                      x=2 ──► 5
  x=2 ──► 5                      x=2 ──► 9   ← two outputs for same x
  x=3 ──► 10
```

**Last lecture (recap):** machine learning as **function approximation** means estimate an unknown map $f$ from data pairs $(x,y)$.

**This lecture (preview):** a **random variable** will later be a function from outcomes $\omega \in \Omega$ to numbers or vectors — and the lecturer stresses it is a **deterministic** function of $\omega$, not a “random object” in casual English. You only need the *function* idea for now.

---

## Function approximation (the ML goal this lecture still serves)

<a id="p3b-fa"></a>

**Function approximation (FA)** is the short name for a simple job:

1. There is (or we hope there is) a map $f$ from inputs to outputs.  
2. We only see a finite list of pairs $(x_i, y_i)$.  
3. We **estimate** $f$ so that for a **new** $x$ we can still predict $y$.

```
  known pairs:  (x1 → y1), (x2 → y2), …
  new input:    x_new  →  ???   ← prediction needs an estimated f
```

**Tiny example.** Inputs = chest X-ray images; outputs = $\{0,1\}$ (no disease / disease). Function approximation means: estimate the image→label map so a **new** film still gets a predicted label.

This lecture does **not** replace that goal. It builds **probability tools** that will later describe the same project as “distribution estimation.” If you only remember one sentence from this warm-up: **FA = estimate a map for prediction.**

---

## Size / measure (why listing members is not enough)

<a id="p4-measure"></a>

Two subsets of the real line can have the same “count” story but different **size**.

Example: which is bigger, $[0,1]$ or $[0,10]$? Both are infinite sets of points. What we actually use is **length**:

$$
\text{length}([a,b]) = b - a
$$

So length($[0,1]$) $= 1$ and length($[0,10]$) $= 10$. **Length is a measure.**

```
  number line
  |====|                  length 1   ← set A
  |====================|  length 10  ← set B  (bigger under length)
```

Other settings use other sizes:

| World | Familiar measure |
|-------|------------------|
| subsets of $\mathbb{R}$ | length |
| subsets of $\mathbb{R}^2$ | area |
| subsets of $\mathbb{R}^3$ | volume |

**Key idea for the lecture:** once you have a set of outcomes $\Omega$, you still need a **measure** on its subsets before you can talk about “how large” an event is. Probability will be one special choice of measure.

---

## The special number range $[0,1]$ and “total size 1”

<a id="p5-unit"></a>

A length can be $100$ meters. A **probability measure** is built so every event gets a size between $0$ and $1$, and the **whole** sample space has size **exactly 1**:

$$
P(\Omega) = 1, \qquad 0 \le P(A) \le 1
$$

Think of the whole cake as size $1$; each slice is a fraction of the cake.

```
  whole cake Ω     size = 1
  piece A          size = P(A) ∈ [0, 1]
  empty piece ∅    size = 0
```

If two pieces **do not overlap** (disjoint events), their sizes **add**. That is the cake rule behind $P(A \cup B) = P(A)+P(B)$ when $A \cap B = \emptyset$.

You do **not** need a deep philosophy of “chance” to start. Treat $P$ first as a **size function** with these rules. Meaning (“70% likely”) can come later — that is exactly the lecturer’s order.

---

## Abstract outcomes (not everything starts as a number)

<a id="p6-abstract"></a>

Beginners often assume every outcome is already a number on a line. The lecture fights that habit.

- Coin outcomes $\{H,T\}$ are **symbols**, not numbers, until you *choose* to code them.  
- An X-ray “outcome” in the philosophical sample space can mean the whole messy story (person → hospital → image → label), not “a $512\times 512$ matrix” alone.  
- Numbers on disk (pixels, tokens) come later as **measurements** of those outcomes.

```
  abstract outcome ω  (story of the experiment)
           │
           │  later: random variable / measurement
           ▼
     numbers on disk  (image, email text, …)
```

If this feels abstract, that is intentional. Probability starts with **sets of outcomes**, then puts a **size** on questions (subsets). Numbers come when we map outcomes to measurements.

---

## Mini glossary (words the NOTES map will use)

| Word | Plain meaning (for now) |
|------|-------------------------|
| **Function approximation (FA)** | Estimate unknown map $f$ from data to **predict** on new inputs |
| **Random experiment** | Messy process we treat as the source of outcomes (not fully defined) |
| **Sample space $\Omega$** | Set of all possible outcomes of that experiment |
| **Event** | A subset of $\Omega$ — a yes/no question about the outcome |
| **Probability measure $P$** | A size on events with $P(\Omega)=1$ and $0\le P(A)\le 1$ |

You do **not** need to master formal measure theory before the NOTES. You need these words to read the master map without freezing on acronyms.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
