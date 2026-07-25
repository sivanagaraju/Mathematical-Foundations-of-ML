# Warm-up before Lec 01 (function approximation)

> **Do this first** if sets, functions, vectors, or “data as pairs” feel new.  
> Then open [NOTES.md](./NOTES.md) at the Executive Summary.  
> Basics only — not a second full lecture.

```
  After this warm-up you should be able to say:

  "A set is a bag of things. A function maps each allowed input to exactly one output.
   Data D is a finite list of (input, output) pairs we observed.
   Function approximation means: estimate the unknown map f so new inputs still get outputs.
   A vector is an ordered list of numbers; an image can be stacked into one long vector.
   When f is unknown we guess a class of shapes, then pick one member using data."
```

---

## Sets (bags of things)

<a id="p1-sets"></a>

A **set** is a collection of **distinct** objects. Order does not matter.

**Examples**

- Die faces: $\mathcal{X} = \{1,2,3,4,5,6\}$  
- Two labels: $\mathcal{Y} = \{0,1\}$ (e.g. no disease / disease)

**Membership:** $x \in \mathcal{X}$ means “$x$ is in the bag.”  
**Empty set:** $\emptyset$ has nothing in it.

```
  bag X (allowed inputs)     bag Y (allowed outputs)
     { times, images, … }       { positions, 0/1, … }
```

In the lecture, the **domain** is one set (inputs) and the **range** is another (outputs).  
(A **subset** $A \subseteq B$ is a smaller bag inside a bigger bag — useful later in probability; light for this video.)

---

## Functions (reliable maps)

<a id="p2-functions"></a>

A **function** $f : \mathcal{X} \to \mathcal{Y}$ is a rule: each allowed input from $\mathcal{X}$ produces **exactly one** output in $\mathcal{Y}$.

```
  OK function                    NOT a function
  x=1 ──► 2                      x=2 ──► 5
  x=2 ──► 5                      x=2 ──► 9   ← two outputs for same x
  x=3 ──► 10
```

Same $x$ always gives the same $y$. That reliability is why engineers like functions.

**Course examples**

- $f(t)$ = planet position at time $t$  
- $f(\text{X-ray}) \in \{0,1\}$ = disease / no disease  

---

## Ordered pairs and data $D$

<a id="p3-data"></a>

An **ordered pair** $(x,y)$ means: “when the input was $x$, we saw output $y$.”

**Data** is a finite list of such pairs:

$$
D = \{(x_1,y_1),\ (x_2,y_2),\ \ldots,\ (x_n,y_n)\}
$$

```
  SEE a tiny D (n = 3)

    time t=1 → position 2
    time t=2 → position 5
    time t=3 → position 10
    time t=4 → ???   ← why we need more than a table of old rows
```

We **do not** start by knowing the true map $f$. We only see evaluations of it (plus, later, noise).

**Goal (informal):** given $D$, find a useful estimate of $f$ so a **new** $x$ still gets a $y$.

---

## Vectors, stacking, and $\mathbb{R}^d$ (the board’s language)

<a id="p4-vectors"></a>

### Vector (Plain English)

A **vector** is a **long list of numbers** that stands for one thing (one image, one audio clip, …).

- 2 numbers → a point on a flat paper (2D)  
- 3 numbers → a point in ordinary space (3D)  
- 100 numbers → a point in a 100-dimensional space (you cannot draw it, but the math is the same idea: one list = one point)

### Euclidean space (Plain English)

**Euclidean space** means the normal “flat geometry” world where distance is the usual kind of distance (as on a graph paper), not a curved weird surface.

- $\mathbb{R}$ = all real numbers (a line)  
- $\mathbb{R}^2$ = plane (two real coordinates)  
- $\mathbb{R}^3$ = ordinary 3D  
- $\mathbb{R}^d$ = same idea with $d$ coordinates  

**How you specify it:** you only need the dimension $d$. Write $\mathbb{R}^d$. Every point is a list of $d$ real numbers.

### What $\mathbb{R}^{pq}$ means on the board

He often uses $p$ = number of **rows** (height) and $q$ = number of **columns** (width) of an image.

$$
d = p \times q
\qquad
x \in \mathbb{R}^{pq}
\quad\text{means:}\quad
\text{one list of } p\cdot q \text{ real numbers}
$$

Same as $\mathbb{R}^d$ with $d=pq$.

### Stacking (flattening) — the process he draws (~24:02)

Start with a **matrix** (table of pixels). Values might be brightness 0–255.

**Example: 2 rows × 3 columns**

```
  IMAGE MATRIX (what you see as a picture)

         col1   col2   col3
  row1    a      c      e
  row2    b      d      f

  [ a  c  e ]
  [ b  d  f ]
```

**Stack columns top-to-bottom, left-to-right** (his board process):

```
  col1 first     then col2     then col3
     a              c              e
     b              d              f
       \            |            /
        \           |           /
         \          |          /
          ▼         ▼         ▼

  ONE LONG COLUMN VECTOR  (length 6 = 2×3)

     ┌   ┐
     │ a │
     │ b │
     │ c │
     │ d │
     │ e │
     │ f │
     └   ┘

  This is one point in  ℝ^{6}  (also written ℝ^{2·3} or ℝ^{pq}).
```

**Tiny numbers version:**

```
  [ 10  30  50 ]
  [ 20  40  60 ]

  →  x = (10, 20, 30, 40, 50, 60)  ∈  ℝ^6
```

**Why do this?** Most ML math wants **one vector per example**. After stacking, the whole image is one input point; algorithms can measure “nearness,” average, fit functions $f:\mathbb{R}^{pq}\to\{0,1\}$, etc.

Real X-ray: same idea, $d$ is huge (thousands), still “one point in $\mathbb{R}^d$.” A **scalar** is only the special case $d=1$.

---

## Guess a class, then refine (models)

<a id="p5-models"></a>

When $f$ is unknown, people **guess a class of shapes** (all ellipses, all parabolas, later: all neural nets of a fixed size). That class is the menu of allowed candidates.

Each concrete choice inside the class is a **model** / estimate of $f$.

```
  CLASS = catalog                 DATA picks one member
  all ellipses / all lines        D favors one specific curve
        │
        ▼
  model f̂  →  use on new x
```

Field saying (quoted in class): *all models are wrong, but some are useful.*

---

## What “ill-posed / blind” means (preview)

<a id="p6-illposed"></a>

If you only have a few points and **no** assumption about the shape of $f$, many different curves can pass through the same points and **disagree** on a new $x$. That is the problem being **blind** or **ill-posed**.

```
  three dots → many curves fit them
  at a new x they disagree
```

Memorizing only the training pairs (table lookup) also fails on new $x$ — a preview of **overfitting**.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz: [quiz.html](./quiz.html) — Part A covers this file.
