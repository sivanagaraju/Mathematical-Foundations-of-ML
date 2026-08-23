# Tutorial 9 — Review of Basic Probability 3

**Video:** [Tutorial 9 : Review of Basic Probability 3](https://www.youtube.com/watch?v=eDSb3yObtB8) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Tutorial 8](../22-Tutorial08-Review-Basic-Probability-2/NOTES.md)  
**Course:** Mathematical Foundations of **Generative AI** (~73 min)  
**Speaker:** NPTEL IISc · Pairs, joints, marginals, conditionals, IID, Jacobian

---

## Table of Contents

1. [Topic 1 — Pair and joint CDF](#topic-1-pair-and-joint-cdf-0003–0553) (00:03–05:53)
2. [Topic 2 — Joint CDF properties](#topic-2-joint-cdf-properties-0553–0823) (05:53–08:23)
3. [Topic 3 — Joint PMF and two dice](#topic-3-joint-pmf-and-two-dice-0823–1348) (08:23–13:48)
4. [Topic 4 — Joint PDF on a triangle](#topic-4-joint-pdf-on-a-triangle-1348–1907) (13:48–19:07)
5. [Topic 5 — Marginals](#topic-5-marginals-1907–2634) (19:07–26:34)
6. [Topic 6 — Conditional discrete](#topic-6-conditional-discrete-2634–3519) (26:34–35:19)
7. [Topic 7 — Conditional continuous and Bayes](#topic-7-conditional-continuous-and-bayes-3519–4034) (35:19–40:34)
8. [Topic 8 — Mixed type, GMM, communication](#topic-8-mixed-type-gmm-communication-4034–5012) (40:34–50:12)
9. [Topic 9 — Independence and vector RVs](#topic-9-independence-and-vector-rvs-5012–6103) (50:12–61:03)
10. [Topic 10 — IID, Jacobian, recap](#topic-10-iid-jacobian-recap-6103–7317) (61:03–73:17)
11. [External references](#external-references)
12. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

This hour moves from one random variable to a **pair** (then a vector) that live on the **same** random experiment. A **joint** CDF / PMF / PDF answers questions about **both** at once; a rectangle of the plane is four corners of $F$; **marginals** peel one variable (unique from the joint; the converse is false). **Conditionals** slice the joint and re-normalize — a thin band if $Y$ is continuous, a mixture if the types are mixed. A **Gaussian mixture model (GMM)** is that mixture when each component is Normal; **independent and identically distributed (IID)** is the course’s default on a dataset; a monotone vector map moves the joint density by a **Jacobian**. Next hour is expectation of several variables, covariance, and the law of large numbers.

**Worldview arc:** from “one $X$ has $F$, $p$, $E$” **to** “a pair has a joint, unique marginals, conditionals, and (if IID) a product law.”

**Hour at a glance (whole video).** Tutorials 7–8 built one $X$: triplet, **cumulative distribution function (CDF)**, PMF or **probability density function (PDF)**, $E$, Var. Today $X$ and $Y$ are two maps on the **same** $(\Omega,\mathcal{F},P)$. If they sat on different experiments you would not have a pair. The **joint CDF** is $F(x,y)=P(X\le x,Y\le y)$ — “both stickers land south-west of $(x,y)$.” A **rectangle** (cylindrical set) uses four corners: $F_{22}-F_{21}-F_{12}+F_{11}$. $F$ is a valid probability object: zero if either argument is $-\infty$, one at $(+\infty,+\infty)$, nondecreasing and right-continuous in each slot. Two discrete maps store the joint as a **table of piles** (joint PMF): two dice with $X=\max$, $Y=\mathrm{sum}$ group 36 downstairs atoms into cells of $1/36$ or $2/36$. Two continuous maps store a **joint PDF** — height on a region, volume 1. Their running example is $p=2$ on the triangle $0<x<y<1$, and $P(Y>X+0.5)=1/4$. **Marginals** peel: sum a row or integrate the other variable. The joint gives unique margins; many joints share the same margins, so you cannot reconstruct the interior from the edges alone.

A **conditional** is a slice, re-normalized. Discrete: $p(x\mid y)=p(x,y)/p_Y(y)$ when $p_Y(y)>0$. Dice: if the sum is 3, $\max\le 4$ is certain; if there is one head in $n$ tosses, the first-head index is uniform on $\{1,\ldots,n\}$. Continuous $Y$ has $P(Y=y)=0$, so they take a thin band and let the width go to 0; on the triangle, $X\mid Y=y$ is Unif$(0,y)$. **Mixed type** (discrete $Y$, continuous $X$) has no single joint PDF; the marginal of $X$ is a **mixture** $\sum p(x\mid y)P(Y=y)$. If each component is Gaussian, that **is** a **Gaussian mixture model (GMM)**: roll a die for the component, then sample that Normal. A radio that hears 0 V or 5 V plus noise decodes bit 1 iff $x>2.5$ (equal priors, Gaussian noise). **Independence** is factorization for **all** windows, not one lucky cell. A **vector** $X:\Omega\to\mathbb{R}^n$ is just $n$ maps. **Independent and identically distributed (IID)** means independent **and** the same law — the dataset assumption. A map $Y=g(X)$ moves density by the absolute **Jacobian** $|J|$ of the inverse; the board’s sum-and-difference example has $|J|=1/2$. Next: $E$ of many, covariance, the law of large numbers.

### System context

```
  ╔══════════════════════════════════╗
  ║ Outside: Cov, LLN (next)         ║
  ║ Outside: full first-course proofs║
  ╚══════════════╤═══════════════════╝
                 │ this tutorial (~73 min)
                 ▼
        ┌────────────────────────────┐
        │ Pair / vector joint stack  │
        └────────────────────────────┘
```

### Main blueprint

```
  Same (Ω, F, P)
       │  X:Ω→ℝ , Y:Ω→ℝ
       ▼
  ┌──── Joint F(x,y)=P(X≤x, Y≤y) ────┐
  │ rectangle = F22−F21−F12+F11      │
  │ F(−∞,·)=0  F(∞,∞)=1  ↗ each arg  │
  └────────────┬─────────────────────┘
               │ discrete          continuous
               ▼                   ▼
         joint PMF table      joint PDF height
         dice max/sum         p=2 on 0<x<y<1
               │                   │
               └────────┬──────────┘
                        ▼
              Marginals (sum / ∫ other)
              unique ← joint;  joint ↛ unique from margins
                        │
                        ▼
              Conditional = joint / margin
              discrete: p(x|y)
              continuous: limit of a thin band
              mixed: GMM + 0 V / 5 V radio
                        │
                        ▼
              Independence: joint = product  (ALL windows)
              n-RVs / vector X:Ω→ℝⁿ
                        │
                        ▼
              IID  +  Y=g(X)  p_Y = p_X(h) |J|
```

### Scenario walkthrough

Walk this **one** story through the blueprint above. Each step answers “so what?” for the next box.

**Story:** you roll two dice on one table. $X$ is the **max** face, $Y$ is the **sum**. Same experiment, two stickers.

1. **Why a pair / joint CDF?** Both numbers come from one roll. The joint **cumulative distribution function (CDF)** $F(x,y)$ is “max $\le x$ **and** sum $\le y$.” A window on the plane uses four corners of that running total.

2. **Why a joint PMF table?** Discrete pair $\to$ piles on cells. 36 equally likely downstairs pairs group into upstairs cells of $1/36$ (doubles, $n=2m$) or $2/36$.

3. **Why a joint PDF at all (the triangle)?** When both are continuous, piles become height on a region. Same job: volume over a set is probability. $p=2$ on $0<x<y<1$ is the continuous cousin; $P(Y>X+0.5)=1/4$.

4. **Why marginals?** You may only care about the max, ignoring the sum. Sum the row (or integrate the other variable). You **cannot** go backwards: many interiors share the same margins.

5. **Why conditionals?** You learn the sum is 3. The world shrinks; $\max\le 4$ is now certain. Continuous cousin: a thin band, $X\mid Y=y\sim\mathrm{Unif}(0,y)$.

6. **Why mixed / GMM?** A **bit** (discrete) plus a **voltage** (continuous) is not a joint PDF. The voltage histogram is a mixture. If each piece is a bell, that is a **Gaussian mixture model (GMM)**. Decode 1 iff $x>2.5$.

7. **Why independence / IID?** Two maps are independent only if **every** window factors. **Independent and identically distributed (IID)** is stronger: independent **and** the same law — how the course treats a dataset.

8. **Why a Jacobian?** If you switch to $Y_1=X_1+X_2$, $Y_2=X_1-X_2$, areas stretch. Density is multiplied by $|J|=1/2$. Covariance and the law of large numbers are next hour.

```
  one table, two dice
         │  same Ω
         ▼
  X = max     Y = sum
         │  joint F / table of piles
         ▼
  rectangle = four corners of F
         │  peel
         ▼
  margins (unique ← joint; not the reverse)
         │  learn Y = 3
         ▼
  slice / re-normalize   =  conditional
         │  same idea, other types
         ▼
  mixed: GMM  (die of bells)     decode x > 2.5
  IID copies + Y = g(X) by |J|
```

Two coins are the same shape: one experiment, two stickers, a joint, then peel or slice.

### Failure / contrast path

```
  X and Y on different experiments          ──X──► not a pair
  “One factored window ⇒ independent”       ──X──► need ALL events
  “Two margins ⇒ unique joint”              ──X──► many joints share margins
  P(Y=y) for continuous Y without a limit   ──X──► divide by zero
  Mixed types called a joint PDF            ──X──► no single p
```

### STOP / out of scope

Expectation of many RVs, covariance, law of large numbers (next tutorial). Full Jacobian proof.

### Load-bearing claims (closed-book)

- A pair lives on the **same** $\Omega$; $F(x,y)=P(X\le x,Y\le y)$; a rectangle is $F_{22}-F_{21}-F_{12}+F_{11}$.
- A joint PMF / PDF is $\ge 0$ and totals $1$.
- The joint **uniquely** gives marginals; the converse is **false**.
- $p(x\mid y)=p(x,y)/p_Y(y)$ (mass or density).
- Independence is factorization **for all** windows.
- **IID** means independent **and** identically distributed.
- $p_Y(y)=p_X(h(y))\,|J|$.

**Speaker / course:** NPTEL IISc · Tutorial 9.

---

## Topic 1: Pair and joint CDF (00:03–05:53)

### Where this sits on the master map

**PAIR** — Two maps, one experiment. Warm-up: [same Ω](./PREREQUISITES.md#p1-same) · [window](./PREREQUISITES.md#p2-window).

### Board / screenshot

![Pair and joint CDF](./screenshots/composites/ch01-topic-01-pair-joint-cdf-panel1of1.png)

**Figure — ~03:48–05:34:** $F_{XY}:\mathbb{R}^2\to\mathbb{R}$, $F(x,y)=P(X\le x,Y\le y)$ as an intersection of events; rectangle $F_{22}-F_{21}-F_{12}+F_{11}$; cylindrical-set sketch.

### What he is establishing

This block is a **review**, not a first course. Viewers are assumed to have a rigorous probability class and a first ML class. Tutorials 7–8 already built the triplet, conditionals, Bayes, one RV (discrete and continuous), $E$, Var, LOTUS, and Markov/Chebyshev/Jensen. Today: a **pair**, then **several** random variables.

Let $X$ and $Y$ be random variables on the **same** $(\Omega,\mathcal{F},P)$. Each maps $\Omega\to\mathbb{R}$. The pair is a **vector-valued** map $\Omega\to\mathbb{R}^2$.

The **joint distribution function** (joint CDF) is $F_{XY}:\mathbb{R}^2\to\mathbb{R}$,

$$
F_{XY}(x,y) = P(X\le x,\; Y\le y) = P\bigl(\{X\le x\}\cap\{Y\le y\}\bigr)
$$

For $x_1<x_2$ and $y_1<y_2$ the probability of the **rectangle** (a **cylindrical set** in their language) is

$$
P(x_1<X\le x_2,\; y_1<Y\le y_2)
= F(x_2,y_2)-F(x_2,y_1)-F(x_1,y_2)+F(x_1,y_1)
$$

You can now write $F$ for a pair and open a window with four corners. Still missing: which axioms $F$ must obey.

A common trap is defining $X$ and $Y$ on two different experiments and still calling $F$ a joint.

### Analogy for this topic only

One photograph. $X$ is brightness, $Y$ is number of faces. $F(x,y)$ is “how often brightness is $\le x$ **and** faces $\le y$.” A window on the brightness–faces plane uses four corners of that running total, not one.

Question: **Why must $X$ and $Y$ share $\Omega$?**

In lecture words: this box is the pair and $F_{XY}$.

### Local picture

```
        y2 ┌────────┐
           │ window │     = F22 − F21 − F12 + F11
        y1 └────────┘
          x1        x2
```

**Notice:** the $+$ is the small corner you subtracted twice.

### Bridge

A 1-D CDF had $F(-\infty)=0$, $F(\infty)=1$, nondecreasing, right-continuous. What is the 2-D list?

---

## Topic 2: Joint CDF properties (05:53–08:23)

### Where this sits on the master map

**AXIOMS OF $F$** — Same jobs as 1-D, two arguments. Warm-up: [window](./PREREQUISITES.md#p2-window).

### Board / screenshot

![Joint CDF properties](./screenshots/composites/ch02-topic-02-joint-cdf-properties-panel1of1.png)

**Figure — ~05:53–08:23:** $F(-\infty,\cdot)=0$; $F(\infty,\infty)=1$; nondecreasing in each argument; right-continuous with left limits in each; rectangle $\ge 0$.

### What he is establishing

The joint CDF is a **valid probability measure** in the same sense as a 1-D CDF.

- If **either** argument is $-\infty$, $F=0$.  
- $F(+\infty,+\infty)=1$ (the whole plane).  
- $F$ is **nondecreasing in each argument**.  
- $F$ is **right-continuous** and has left limits **in each argument**.  
- Every rectangle probability is $\ge 0$.

Any function $\mathbb{R}^2\to\mathbb{R}$ with these properties **is** a joint CDF of some pair.

You can now reject a proposed $F$ that goes down in $x$ or forgets $F(\infty,\infty)=1$. Still missing: the discrete table (joint PMF).

A common trap is checking right-continuity in only one slot.

### Analogy for this topic only

The south-west pile of cloth never shrinks when you move the pin right or up. At the far north-east corner you have the whole tablecloth (mass 1). Off the left or bottom edge you have nothing.

Question: **What is $F(x,-\infty)$?**

In lecture words: this box is the joint-CDF axiom list.

### Local picture

```
  F(−∞, y)=0     F(x, −∞)=0     F(+∞,+∞)=1
  move x or y up-right  →  F cannot fall
  rectangle mass ≥ 0
```

**Notice:** two arguments ⇒ every 1-D axiom is stated twice.

### Bridge

How do two **discrete** RVs store the joint — a table of piles?

---

## Topic 3: Joint PMF and two dice (08:23–13:48)

### Where this sits on the master map

**TABLE** — Joint mass. Warm-up: [table](./PREREQUISITES.md#p3-table).

### Board / screenshot

![Joint PMF two dice](./screenshots/composites/ch03-topic-03-joint-pmf-two-dice-panel1of1.png)

**Figure — ~08:23–13:48:** $p(x_i,y_j)=P(X=x_i,Y=y_j)$; 0 off support; sums to 1; two dice $X=\max$, $Y=\mathrm{sum}$; masses $2/36$ or $1/36$.

### What he is establishing

Let $X$ take $x_1,\ldots,x_n$ and $Y$ take $y_1,\ldots,y_m$ ($n$ need not equal $m$). The **joint PMF** is

$$
p_{XY}(x_i,y_j) = P(X=x_i,\; Y=y_j)
$$

and $0$ at every other pair. Analogous to one PMF: every mass $\ge 0$, and the **double sum is 1**. The joint CDF is the sum of all masses with $x_i\le x$ and $y_j\le y$.

**Running example.** Roll two dice. $\Omega=\{(w_1,w_2): w_i\in\{1,\ldots,6\}\}$.  
$X=\max(w_1,w_2)\in\{1,\ldots,6\}$, $Y=w_1+w_2\in\{2,\ldots,12\}$.

The event $\{X=m,Y=n\}$ is the outcomes where the max is $m$ and the sum is $n$. The only candidates are $(m,n-m)$ and $(n-m,m)$. If $n=2m$ those coincide (one outcome). So

$$
p(m,n)=\frac{2}{36}\quad(n\neq 2m),\qquad p(m,n)=\frac{1}{36}\quad(n=2m)
$$

on the legal $(m,n)$, else $0$. Checking that all legal cells sum to 1 is left as an exercise. A quick sanity count: there are $36$ equally likely downstairs atoms, and every atom is assigned to exactly one $(m,n)$ cell, so the upstairs masses **must** add to $1$ if you grouped them correctly.

You can now fill a joint table and special-case the diagonal $n=2m$. Still missing: a **density** on a region of the plane.

A common trap is forcing $X$ and $Y$ to have the same number of values.

### Analogy for this topic only

A $6\times 11$ tray of cookie squares. Most squares empty. Two cookies in a square when both $(m,n-m)$ and $(n-m,m)$ exist; one cookie when they are the same face twice. All cookies together: 36 crumbs / 36 = 1.

Question: **When is the joint mass $1/36$ rather than $2/36$?**

In lecture words: this box is the joint PMF and the dice pair.

### Local picture

```
  (w1,w2)  →  X=max    Y=sum
  (2,5)    →  5        7     two ways for (5,7)
  (3,3)    →  3        6     one way  (n=2m)
```

**Notice:** 36 atoms downstairs; the table upstairs groups them.

### Bridge

What if $X$ and $Y$ are **continuous** — a height on a triangle instead of piles?

---

## Topic 4: Joint PDF on a triangle (13:48–19:07)

### Where this sits on the master map

**HEIGHT ON A REGION** — Joint density. Warm-up: [jam](./PREREQUISITES.md#p4-jam).

### Board / screenshot

![Joint PDF triangle](./screenshots/composites/ch04-topic-04-joint-pdf-triangle-panel1of1.png)

**Figure — ~13:48–19:07:** $p\ge 0$, $\iint p=1$; example $p=2$ on $0<x<y<1$; $P(Y>X+0.5)=0.25$.

### What he is establishing

$X,Y$ continuous type with joint CDF $F$ if there is a **joint PDF** $p_{XY}$ with

$$
F(x,y)=\int_{-\infty}^{y}\int_{-\infty}^{x} p(u,v)\,du\,dv
$$

Axioms: $p\ge 0$ everywhere, and $\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}p=1$. Any such $p$ is a joint density of some pair.

They **warn** (load-bearing): two continuous-type RVs do **not** automatically have a joint density. Continuous type for each margin is weaker than “the pair admits a joint $p(x,y)$.” The definition is the integral representation of $F$, not “both look continuous.”

**Example.** $p(x,y)=2$ on $0<x<y<1$, and $0$ elsewhere. Nonnegativity is obvious. The double integral reduces to the triangle:

$$
\int_{y=0}^{1}\int_{x=0}^{y} 2\,dx\,dy = 1
$$

The board sketch is the **support**, not a 3-D plot of $p$ — they say so.

$P((X,Y)\in A)=\iint_A p$. For $A=\{Y>X+0.5\}$ they integrate $y$ from $0.5$ to $1$ and $x$ from $0$ to $y-0.5$ and get **$0.25$**.

You can now certify a joint $p$ and integrate a subregion. Still missing: peeling one variable to get a **marginal**.

A common trap is treating the triangle drawing as a graph of height $p$.

### Analogy for this topic only

Frost a triangle of cake with 1 cup of frosting at height 2. Asking $P(Y>X+0.5)$ is “how much frosting sits in the upper sliver?” Answer: a quarter cup.

Question: **Why is height 2 legal?**

In lecture words: this box is the joint PDF.

### Local picture

```
  y
  1 |    /|
    |   / |   p=2 on 0<x<y<1
    |  /  |   area 1/2  →  volume 1
  0 +-----+ x
    0     1

  Y > X+0.5  →  smaller triangle, volume 0.25
```

**Notice:** endpoints of the triangle do not matter (zero area).

### Bridge

Given the joint, how do you recover the law of $X$ **alone**?

---

## Topic 5: Marginals (19:07–26:34)

### Where this sits on the master map

**PEEL** — Sum or integrate the other variable. Warm-up: [margin](./PREREQUISITES.md#p5-margin).

### Board / screenshot

![Marginals](./screenshots/composites/ch05-topic-05-marginals-panel1of1.png)

**Figure — ~19:07–26:34:** $F_X(x)=F(x,\infty)$; $p_X(x_i)=\sum_j p(x_i,y_j)$; $p_X(x)=\int p(x,y)\,dy$; “margin of the table”; joint $\Rightarrow$ unique margins, not conversely.

### What he is establishing

Set the unused argument of $F$ to $+\infty$:

$$
F_X(x)=F_{XY}(x,+\infty)=P(X\le x),\qquad F_Y(y)=F_{XY}(+\infty,y)
$$

**Discrete:** fix $x_i$, sum the row: $p_X(x_i)=\sum_j p(x_i,y_j)$.  
**Continuous:** $p_X(x)=\int p(x,y)\,dy$. Same for $Y$.

Name: old tables wrote row/column totals in the **margin**.

**Dice.** $p_X(m)$ = sum of $p(m,n)$ over legal $n=m+1,\ldots,2m$.  
**Triangle.** Fix $x\in(0,1)$; $y$ runs from $x$ to $1$; $p_X(x)=\int_x^1 2\,dy=2(1-x)$. Similarly $p_Y(y)=2y$ on $(0,1)$. Check $\int_0^1 2(1-x)\,dx=1$.

**Uniqueness.** A joint determines the marginals **uniquely**. The converse is **false**: many joints share the same pair of marginals. You cannot reconstruct $p(x,y)$ from $p_X$ and $p_Y$ alone (unless you add independence later).

You can now peel a joint and refuse “two margins ⇒ one joint.” Still missing: the **slice** $X\mid Y=y$.

A common trap is thinking two Gaussian margins force a unique bivariate Gaussian.

### Analogy for this topic only

A spreadsheet of cookie counts. The right-hand **margin** is cookies per row, ignoring columns. Many different interiors can share those same row and column totals.

Question: **Can two different joints have the same $p_X$ and $p_Y$?**

In lecture words: this box is marginals.

### Local picture

```
         y1  y2  y3   | p_X
    x1   ·   ·   ·    | row sum
    x2   ·   ·   ·    | row sum
                      +------
         col sums  =  p_Y

  triangle:  p_X(x)=2(1−x)   p_Y(y)=2y
```

**Notice:** joint → margins is easy; margins → joint is underdetermined.

### Bridge

If you **know** $Y=y$, how do you reweight $X$?

---

## Topic 6: Conditional discrete (26:34–35:19)

### Where this sits on the master map

**SLICE** — $X$ given $Y=y$. Warm-up: [slice](./PREREQUISITES.md#p6-slice).

### Board / screenshot

![Conditional discrete 1](./screenshots/composites/ch06-topic-06-conditional-discrete-panel1of2.png)

![Conditional discrete 2](./screenshots/composites/ch06-topic-06-conditional-discrete-panel2of2.png)

**Figure — ~26:34–35:19:** $F(x\mid y)=P(X\le x\mid Y=y)$ needs $p_Y(y)>0$; dice $P(X\le 4\mid Y=3)=1$, $P(X\le 4\mid Y=9)=0$; $p(x_i\mid y_j)=p(x_i,y_j)/p_Y(y_j)$; coin $P(Y=k\mid X=1)=1/n$; discrete Bayes.

### What he is establishing

$$
F_{X\mid Y}(x\mid y) = P(X\le x \mid Y=y)
$$

defined only when the conditioning object is legal — the same $P(B)>0$ idea. For two discrete RVs that means $p_Y(y)>0$. The map is $\mathbb{R}^2\to\mathbb{R}$; the bar is familiar abuse.

For **each fixed** $y$, $F(\cdot\mid y)$ is an ordinary CDF of $X$ (a new assignment).

**Dice.** $P(X\le 4\mid Y=3)=1$ (if the sum is 3 the max cannot exceed 2, hence $\le 4$ always). $P(X\le 4\mid Y=9)=0$ (a sum of 9 cannot have max $\le 4$).

**Conditional PMF:**

$$
p(x_i\mid y_j)=\frac{p(x_i,y_j)}{p_Y(y_j)},\qquad \sum_i p(x_i\mid y_j)=1 \text{ for every } j
$$

**Coin.** Toss $n$ times. $X=$ number of heads, $Y=$ toss index of the **first** head. Then

$$
P(Y=k\mid X=1)=\frac{1}{n},\qquad k=1,\ldots,n
$$

If there was only one head, it is equally likely to have been any of the $n$ tosses.

**Discrete Bayes.** Joint $=$ conditional $\times$ marginal, so you can flip $X\mid Y$ and $Y\mid X$.

You can now slice a table and run the $1/n$ coin. Still missing: continuous $Y$ where $P(Y=y)=0$.

A common trap is writing $p(x\mid y)$ when $p_Y(y)=0$.

### Analogy for this topic only

You learn the sum of two dice is 3. The world shrinks to $\{(1,2),(2,1)\}$. Asking “is the max $\le 4$?” is certain. Asking it when the sum is 9 is impossible. One head in $n$ tosses: the lonely head is equally likely on any toss.

Question: **What is $P(Y=k\mid X=1)$ for the $n$-coin experiment?**

In lecture words: this box is discrete $X\mid Y$.

### Local picture

```
  p(x|y) = p(x,y) / p_Y(y)     (row y, re-normalize)

  sum=3  →  max≤4  is certain
  one head in n tosses  →  first-head index ~ Unif{1..n}
```

**Notice:** each slice is itself a PMF.

### Bridge

If $Y$ is continuous, $\{Y=y\}$ has probability 0. How do they still define $X\mid Y=y$?

---

## Topic 7: Conditional continuous and Bayes (35:19–40:34)

### Where this sits on the master map

**THIN BAND** — Limit $\Delta\to 0$. Warm-up: [slice](./PREREQUISITES.md#p6-slice).

### Board / screenshot

![Continuous conditional](./screenshots/composites/ch07-topic-07-conditional-continuous-bayes-panel1of1.png)

**Figure — ~35:19–40:34:** limit definition; $p(x\mid y)=p(x,y)/p_Y(y)$; triangle $1/y$ and $1/(1-x)$; Bayes with an **integral** in the denominator.

### What he is establishing

$P(Y=y)=0$, so the old definition is undefined. They replace the point by a band $[y,y+\Delta]$ and let $\Delta\to 0$ (exists for $p_Y(y)>0$):

$$
F(x\mid y)=\lim_{\Delta\to 0}
\frac{P(X\le x,\; Y\in[y,y+\Delta])}{P(Y\in[y,y+\Delta])}
$$

The limit is the integral of the **conditional density**

$$
p(x\mid y)=\frac{p(x,y)}{p_Y(y)}
$$

Same construction for $Y\mid X$.

**Triangle again.** $p_Y(y)=2y$, $p(x,y)=2$ on $0<x<y$, so $p(x\mid y)=1/y$ on $(0,y)$: **$X\mid Y=y\sim\mathrm{Unif}(0,y)$**. Likewise $Y\mid X=x\sim\mathrm{Unif}(x,1)$.

**Continuous Bayes.** Same flip, but the “total law” in the denominator is an **integral**, not a sum.

Until now both coordinates were the same type. Mixed types are next.

You can now write $p(x\mid y)$ and the Unif slices. A common trap is dividing by $p_Y(y)=0$.

### Analogy for this topic only

You cannot condition on a knife-edge of cake (zero frosting). Take a **thin strip** of width $\Delta$ and shrink it. What remains is the height along that strip, re-normalized — uniform jam on $(0,y)$.

Question: **Given $Y=y$ in the triangle, what law is $X$?**

In lecture words: this box is continuous $X\mid Y$.

### Local picture

```
  band [y, y+Δ]  --Δ→0-->  p(x|y)=p(x,y)/p_Y(y)

  triangle:  X|Y=y  ~ Unif(0,y)
             Y|X=x  ~ Unif(x,1)
```

**Notice:** domain of $p(\cdot\mid y)$ is a function of $y$.

### Bridge

What if $Y$ is a **bit** and $X$ is a **voltage** — one discrete, one continuous?

---

## Topic 8: Mixed type, GMM, communication (40:34–50:12)

### Where this sits on the master map

**MIXED** — Discrete $Y$, continuous $X$. Warm-up: [slice](./PREREQUISITES.md#p6-slice).

### Board / screenshot

![Mixed GMM](./screenshots/composites/ch08-topic-08-mixed-gmm-comms-panel1of2.png)

![Communication](./screenshots/composites/ch08-topic-08-mixed-gmm-comms-panel2of2.png)

**Figure — ~40:34–50:12:** $p_X(x)=\sum p(x\mid y)P(Y=y)$ (GMM if Gaussians); 0 V / 5 V + noise; decide $1$ iff $x>2.5$.

### What he is establishing

So far both were discrete or both continuous. Now $X$ continuous, $Y$ discrete. Then $X\mid Y=y$ is a **density**, and the marginal of $X$ is the **mixture**

$$
p_X(x)=\sum_y p(x\mid y)\,P(Y=y)
$$

The board writes three components $Y\in\{1,2,3\}$ with weights $\lambda_i\ge 0$, $\sum\lambda_i=1$:

$$
p_X(x)=\lambda_1 p_1(x)+\lambda_2 p_2(x)+\lambda_3 p_3(x)
$$

If each $p_i$ is Gaussian, this **is** a Gaussian mixture model (GMM): roll a $3$-sided die for the component, then sample that Gaussian. Components need not all be Gaussian (one could be exponential) — but every $X\mid Y=y$ must stay **continuous**.

Mixed Bayes still holds:

$$
p(y\mid x)=\frac{p(x\mid y)\,P(Y=y)}{p_X(x)},\qquad \int p(y\mid x)\,p_X(x)\,dx=P(Y=y)
$$

(the integral is the mixed total-probability rule on the board).

**Communication.** Sender emits $0$ V (bit $Y=0$) or $5$ V (bit $Y=1$). Receiver measures $X=$ sent voltage $+$ channel noise. Want $P(Y=1\mid X=x)$.

In practice you only need the **ratio** $P(Y=1\mid x)/P(Y=0\mid x)$ (the $p_X$ cancels). If priors are equal and $X\mid Y$ is Gaussian centered at $0$ or $5$, the ratio exceeds $1$ iff

$$
x > 2.5
$$

Then $p_X$ itself is $\frac12\mathcal{N}(0,\sigma^2)+\frac12\mathcal{N}(5,\sigma^2)$.

You can now write a mixture and the $2.5$ V rule. Still missing: when the joint **factors**.

A common trap is calling a mixed pair a “joint PDF.”

### Analogy for this topic only

A radio hears a noisy voltage. If it sounds closer to 5 V than to 0 V (threshold halfway at 2.5), guess the sent bit was 1. The overall voltage histogram is two bells mixed fifty-fifty — a GMM with two components.

Question: **If priors are equal and noise is Gaussian, what threshold decides bit 1?**

In lecture words: this box is mixed type + GMM + decoder.

### Local picture

```
  Y = component (discrete)
  X | Y=k  ~ continuous density k
  p_X(x) = Σ π_k p_k(x)     (GMM if p_k Gaussian)

  send 0 or 5 V  →  hear x  →  say 1 iff x>2.5
```

**Notice:** you often need only a **ratio**, not $p_X$.

### Bridge

When does “knowing $Y$” tell you **nothing** new about $X$?

---

## Topic 9: Independence and vector RVs (50:12–61:03)

### Where this sits on the master map

**FACTOR** — Joint = product. Warm-up: [product](./PREREQUISITES.md#p7-prod).

### Board / screenshot

![Independence](./screenshots/composites/ch09-topic-09-independence-vector-panel1of2.png)

![Vector RVs](./screenshots/composites/ch09-topic-09-independence-vector-panel2of2.png)

**Figure — ~50:12–61:03:** $P(X\in B_1,Y\in B_2)=P P$ for **all** events; $F=F_X F_Y$; PMF/PDF products; $n$-RVs and $X:\Omega\to\mathbb{R}^n$; mixed types still have conditionals.

### What he is establishing

$X\perp Y$ when **every** pair of events factors:

$$
P(X\in B_1, Y\in B_2)=P(X\in B_1)\,P(Y\in B_2)\qquad\text{for all }B_1,B_2
$$

One lucky window is not enough. Then $F_{XY}=F_X F_Y$. Discrete: joint PMF $=$ product of marginal PMFs. Continuous: joint PDF $=$ product of marginal PDFs. Then $X\mid Y$ equals the unconditional law of $X$ (any type mix they have discussed).

**Three or $n$ variables.** Joint $F(x,y,z)=P(X\le x,Y\le y,Z\le z)$; joint PMF/PDF with the same $\ge 0$ and total-1 rules. Many marginals: even $F_{XY}(x,y)=F_{XYZ}(x,y,+\infty)$ is a **pair** marginal. Condition on one or several. Write a **vector** $X=(X_1,\ldots,X_n):\Omega\to\mathbb{R}^n$ when $n$ is large.

If some coordinates are discrete and some continuous there is **no** single joint PMF or PDF, but **conditional distributions** still exist and Bayes still extends.

Independence of $n$: events $B_1,\ldots,B_n$ all factor. Then — the one case — **marginals do determine the joint** (they multiply). A constant density on a region: choose $K$ so the integral is 1, then peel any marginals you want.

You can now test independence on **all** windows and write $X\in\mathbb{R}^n$. Still missing: IID and how a map $Y=g(X)$ moves $p$.

A common trap is claiming independence from one factored cell.

### Analogy for this topic only

Two switches are independent only if **every** pair of “on/off” combinations factors, not just “both on.” $n$ photocopies of the same experiment become one vector $X\in\mathbb{R}^n$. If the copies do not talk, the joint is the product of the margins — the rare time margins rebuild the joint.

Question: **Does one factored rectangle prove independence?**

In lecture words: this box is independence and vectors.

### Local picture

```
  X ⊥ Y  ⇔  F(x,y)=F_X(x)F_Y(y)   for all x,y
         ⇔  p(x,y)=p_X(x)p_Y(y)

  X = (X1,...,Xn) : Ω → ℝⁿ
  independence  ⇒  margins determine the joint
```

**Notice:** “for all events” is the load-bearing phrase.

### Bridge

The course’s default on a dataset is stronger than independence: **identical** copies. And $Y=g(X)$ needs a volume factor.

---

## Topic 10: IID, Jacobian, recap (61:03–73:17)

### Where this sits on the master map

**COPIES + STRETCH** — IID and $p_Y=p_X|J|$. Warm-up: [IID](./PREREQUISITES.md#p8-iid).

### Board / screenshot

![IID and Jacobian](./screenshots/composites/ch10-topic-10-iid-jacobian-recap-panel1of2.png)

![Transform example](./screenshots/composites/ch10-topic-10-iid-jacobian-recap-panel2of2.png)

**Figure — ~61:03–70:49:** Jacobian matrix of $h$; $Y_1=X_1+X_2$, $Y_2=X_1-X_2$; inverse halves; $J=\bigl(\begin{smallmatrix}0.5&0.5\\0.5&-0.5\end{smallmatrix}\bigr)$, $\det J=-0.5$.

### What he is establishing

$Z=g(X,Y)$ with $g:\mathbb{R}^2\to\mathbb{R}$ is an RV. Discrete: $p_Z(z)$ is the sum of $p(x_i,y_j)$ over all cells with $g(x_i,y_j)=z$.

**IID** = independent **and identically distributed**. Cornerstone assumption of the course. Two RVs: $X\perp Y$ and $p_X=p_Y$. $n$ RVs: mutually independent and every $p_{X_i}$ equals the same $p$.

If $X\perp Y$ then $g(X)\perp h(Y)$. More generally, a function of $(X_1,\ldots,X_m)$ is independent of a function of an independent block $(Y_1,\ldots,Y_n)$.

**Vector change of variables.** $Y=g(X)$ in $\mathbb{R}^n$, inverse $X=h(Y)$, Jacobian matrix $J=(\partial x_i/\partial y_j)$ **nonzero** on the range. Then (no proof today)

$$
p_Y(y) = p_X\bigl(h(y)\bigr)\,|J|
$$

**Example (board).** $Y_1=X_1+X_2$, $Y_2=X_1-X_2$. Inverse $h$:

$$
X_1=\frac{Y_1+Y_2}{2},\qquad X_2=\frac{Y_1-Y_2}{2}
$$

The Jacobian matrix of $h$ is

$$
J=\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & -0.5 \end{pmatrix},\qquad \det J=-0.5,\qquad |J|=\frac12
$$

so $p_Y(y)=\frac12\, p_X\bigl(\frac{y_1+y_2}{2},\frac{y_1-y_2}{2}\bigr)$.

**Recap.** Pairs; joints; marginals; conditionals; $n$-RVs; independence; IID; Jacobian. **Next tutorial:** expectation of several RVs, **covariance**, and the **law of large numbers** — then the probability recap ends.

You can now say IID in two words and stretch a joint by $|J|$. Leftover: $E(X,Y)$, $\mathrm{Cov}$, LLN.

A common trap is dropping $|J|$ or using a singular $g$.

### Analogy for this topic only

IID is $n$ independent photocopies of the same law — the dataset assumption. $Y_1=X_1+X_2$, $Y_2=X_1-X_2$ is a stretch-and-rotate of the plane; ink density is multiplied by $1/2$ because areas double ($|J|=1/2$ is the shrink of the inverse).

Question: **What two words does IID expand to?**

In lecture words: this box is IID + Jacobian + the close.

### Local picture

```
  IID:  X1,...,Xn independent and each ~ p

  Y = g(X),  X = h(Y),  det J ≠ 0
  p_Y(y) = p_X(h(y)) |J|

  Y1=X1+X2 , Y2=X1−X2    →   |J|=1/2

  Next: E of many · Cov · LLN
```

**Notice:** independence of blocks survives after you apply separate functions.

### Bridge

Averages of many coordinates — and how they **co-vary** — are the next tutorial, not this one.

---

## External references

Two layers, **both kept**.

1. **Start here** — the newer high-signal companions (famous teachers, mapped to this lecture’s hard boxes).
2. **Full topic map** — the previous per-topic list (2–3 companions each) **plus** any new entries already woven above. Use a group when one box still feels thin.

### Start here — high-signal companions

Only a few **widely used** companions — the ones people actually finish. Not a pile of random blogs. Use them after the matching topic, with this tutorial still closed.

This lecture is a **chalkboard recap**. It does **not** open a notebook.

**If a pair still feels like two separate experiments (Topics 1–3).** Brown’s [Seeing Theory — compound probability](https://seeing-theory.brown.edu/compound-probability/index.html) and [Khan Academy’s random-variables unit](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) keep $X$ and $Y$ on one $\Omega$. Taboga’s [Statlect — random vectors](https://www.statlect.com/fundamentals-of-probability/random-vectors) is the written joint / marginal page.

**If joint height vs table piles blur (Topics 3–5).** Khan plus Seeing Theory’s [distributions](https://seeing-theory.brown.edu/probability-distributions/index.html) for “cells vs area.” Statlect’s [factorization of joint PMFs](https://www.statlect.com/fundamentals-of-probability/factorization-of-joint-probability-mass-functions) and [factorization of joint PDFs](https://www.statlect.com/fundamentals-of-probability/factorization-of-joint-probability-density-functions) write the peel.

**If conditionals and Bayes still swap (Topics 6–7).** Grant Sanderson’s [3Blue1Brown — Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) is the famous ratio. Statlect’s [conditional probability distributions](https://www.statlect.com/fundamentals-of-probability/conditional-probability-distributions) covers $p(x\mid y)=p(x,y)/p_Y(y)$ for both types.

**If mixed type / GMM is just a name (Topic 8).** Josh Starmer’s [StatQuest — Gaussian Mixture Models](https://www.youtube.com/watch?v=qMTuMa86NzU) is the popular “die of bells.” Same Bayes video as above for the $2.5$ V decoder.

**If independence, IID, or $|J|$ slip (Topics 9–10).** Khan’s [independent events](https://www.khanacademy.org/math/statistics-probability/probability-library/multiplication-rule-independent/v/compound-sample-spaces) plus Seeing Theory’s [independence](https://seeing-theory.brown.edu/compound-probability/index.html#second). Statlect’s [independent random variables](https://www.statlect.com/fundamentals-of-probability/independent-random-variables) and [IID sequence](https://www.statlect.com/glossary/IID-sequence) expand the two words. 3Blue1Brown’s [change of variables](https://www.youtube.com/watch?v=okjYP_Uj-KM) is why $\lvert J\rvert$ multiplies density.

**How to use.** Joint fog → Seeing Theory or Khan *before* Topic 1. GMM name → StatQuest *after* Topic 8. One famous teacher per stuck idea. Do not open ten tabs.

---

### Full topic map — previous list plus new entries

**How to use:** finish NOTES first (video closed if you can). When one map box still feels thin, open **only that topic’s group** — **2–3 companions** (video + notes/blog). All links live **here**, not inside topic bodies.

This lecture is a **chalkboard recap**. It does **not** open a notebook.

### Topic 1 — Pair and joint CDF

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Seeing Theory — compound probability](https://seeingtheory.io/compound-probability/) | Interactive | Two events on one experiment |
| [Khan Academy — joint distributions intro](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) | Lesson | $P(X\le x,Y\le y)$ |
| [Stat 110 notes](https://stat110.hsites.harvard.edu/) | Notes | Pair on one $\Omega$ |

### Topic 2 — Joint CDF properties

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Stat 110 notes — CDF properties](https://stat110.hsites.harvard.edu/) | Notes | 1-D list lifted to two arguments |
| [Seeing Theory — distributions](https://seeingtheory.io/probability-distributions/) | Interactive | Running totals |
| [Khan Academy — CDF](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-discrete/v/discrete-probability-distribution) | Video | Nondecreasing / limits 0 and 1 |

### Topic 3 — Joint PMF / two dice

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Khan Academy — two-way tables](https://www.khanacademy.org/math/statistics-probability/probability-library) | Lesson | Cells, row/column totals |
| [Seeing Theory — compound](https://seeingtheory.io/compound-probability/) | Interactive | Two dice pictures |
| [Stat 110 Lec 7](https://www.youtube.com/watch?v=9PVn2auwXFw) | Video | Joint mass |

### Topic 4 — Joint PDF / triangle

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Stat 110 Lec 7 — joint PDF](https://www.youtube.com/watch?v=9PVn2auwXFw) | Video | Height vs volume |
| [Khan Academy — joint density](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) | Lesson | Integrate a region |
| [Seeing Theory — continuous](https://seeingtheory.io/probability-distributions/) | Interactive | Support vs plot of $p$ |

### Topic 5 — Marginals

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Seeing Theory — compound / margins](https://seeingtheory.io/compound-probability/) | Interactive | Sum a row |
| [Stat 110 notes](https://stat110.hsites.harvard.edu/) | Notes | Joint $\Rightarrow$ unique margins |
| [Khan Academy — marginal distributions](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) | Lesson | Why the converse fails |

### Topic 6 — Discrete conditional

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Khan Academy — conditional probability](https://www.khanacademy.org/math/statistics-probability/probability-library/conditional-probability-independence/v/calculating-conditional-probability) | Video | Slice and re-normalize |
| [Stat 110 Lec 7](https://www.youtube.com/watch?v=9PVn2auwXFw) | Video | $p(x\mid y)=p(x,y)/p_Y(y)$ |
| [Seeing Theory — conditional](https://seeingtheory.io/compound-probability/conditional-probability/) | Interactive | Given $Y=y$ |

### Topic 7 — Continuous conditional / Bayes

| Resource | Type | Why it helps |
|----------|------|--------------|
| [3Blue1Brown — Bayes](https://www.youtube.com/watch?v=HZGCoVF3YvM) | Video | Ratio, not a new axiom |
| [Stat 110 notes — continuous Bayes](https://stat110.hsites.harvard.edu/) | Notes | Integral in the denominator |
| [Khan Academy — conditional density](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) | Lesson | Thin-band intuition |

### Topic 8 — Mixed / GMM / comms

| Resource | Type | Why it helps |
|----------|------|--------------|
| [StatQuest — GMM](https://www.youtube.com/watch?v=qMTuMa86NzU) | Video | Die + Gaussians = $p_X$ |
| [3Blue1Brown — Bayes / tests](https://www.youtube.com/watch?v=lG4VkPoG3ko) | Video | Threshold from a likelihood ratio |
| [Stat 110 notes](https://stat110.hsites.harvard.edu/) | Notes | Mixed discrete/continuous |

### Topic 9 — Independence / vectors

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Stat 110 — independence](https://stat110.hsites.harvard.edu/) | Notes | Product for **all** windows |
| [Seeing Theory — independence](https://seeingtheory.io/compound-probability/independence/) | Interactive | Factorization picture |
| [Khan Academy — independent RVs](https://www.khanacademy.org/math/statistics-probability/probability-library/multiplication-rule-independent/v/compound-sample-spaces) | Video | When you may multiply |

### Topic 10 — IID / Jacobian

| Resource | Type | Why it helps |
|----------|------|--------------|
| [3Blue1Brown — change of variables](https://www.youtube.com/watch?v=okjYP_Uj-KM) | Video | Why $\lvert J\rvert$ |
| [Stat 110 notes — IID / transforms](https://stat110.hsites.harvard.edu/) | Notes | Identical copies; $Y=g(X)$ |
| [Khan Academy — transforming RVs](https://www.khanacademy.org/math/ap-statistics/random-variables-ap/transforming-random-variables/v/impact-of-transforming-random-variables) | Video | Linear maps first |

### Whole-map companions

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PREREQUISITES.md](./PREREQUISITES.md) | Warm-up | #p1–#p8 |
| [Tutorial 8 NOTES](../22-Tutorial08-Review-Basic-Probability-2/NOTES.md) | Prior unit | One CRV / $E$ / Var this hour assumes |
| [Stat 110 playlist](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) | Video course | Slower joints / IID |

---


## Sources

- Video: [Tutorial 9 : Review of Basic Probability 3](https://www.youtube.com/watch?v=eDSb3yObtB8)
- Channel: NPTEL — Indian Institute of Science, Bengaluru
- Duration: ~73 min (00:03–73:17)
- Skill: `youtube-lecture-tutor` · math_technical
- 10 topics · 45 claims · coverage receipt
- Previous: [Tutorial 8](../22-Tutorial08-Review-Basic-Probability-2/NOTES.md)
- Next (spoken): $E$ of many RVs, covariance, LLN
- Package: `23-Tutorial09-Review-Basic-Probability-3`
