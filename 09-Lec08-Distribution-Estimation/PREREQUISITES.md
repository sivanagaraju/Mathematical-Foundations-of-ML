# Prerequisites — warm-up before Lec 08 (distribution estimation)

> **Do this first** if “estimate a distribution,” “conditional,” or the symbols $D\sim P$ still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on [Lec 07](../08-Lec07-IID-Assumption/PREREQUISITES.md) (IID + given $D$ estimate $P$).  
> Still a **warm-up** — but every formula below is decoded in plain English.

```
  After this warm-up you can say:

  "D ~_iid P means the dataset is drawn independently from the same unknown law P."
  "Estimate P means recover that law (or a useful piece) from the finite sample D."
  "P(Y|X=x) = law of the label when the image is fixed at x."
  "P(Y) is prevalence; P(X|Y=y) is how data look when the label is fixed."
  "A moment (like E[X]) is a summary number computed from P, not the whole P."
  "In every conditional, the conditioner is fixed."
```

**Warm-up → lecture boxes**

```
  §1  Read D ~_iid P symbol-by-symbol   ──► Topics 1–2
  §2  Estimate P (core job formula)      ──► Topic 3
  §3  Joint / margin / conditional math  ──► Topics 4–5,7
  §4  Moments E[·] vs full P             ──► Topic 4
  §5  Which formula for which question   ──► Topic 5
  §6  Disc vs gen                        ──► Topic 3
  §7  Supervised packaging               ──► Topic 6
  §8  Conditioner fixed + P vs density   ──► Topic 7
```

---

## 1. Reading $D\sim_{\mathrm{iid}} P$ (symbol by symbol)

<a id="p1-dataset"></a>

### Purpose for the video

Every later formula assumes you can read this opening line without freezing.

### The setup formula

$$
D=\{(x_i,y_i)\}_{i=1}^{n}
\quad\sim_{\mathrm{iid}}\quad
P_{X,Y}
\quad\text{(unknown)}
$$

| Symbol | Read as | Plain English |
|--------|---------|----------------|
| $D$ | “dee” / dataset | The finite list of samples you hold |
| $\{(x_i,y_i)\}_{i=1}^{n}$ | pairs $i=1$ to $n$ | Point $i$ has feature vector $x_i$ and label $y_i$ |
| $x_i\in\mathbb{R}^{d}$ | “x-sub-i in R-dee” | $d$ real numbers (stacked image / features) |
| $y_i\in\mathbb{R}^{k}$ or discrete | “y-sub-i” | Continuous vector label **or** category tag $\{0,1,\ldots\}$ |
| $\sim$ | “is sampled from” | The tilde: drawn according to a law |
| $\mathrm{iid}$ | “eye-eye-dee” | **I**ndependent and **i**dentically **d**istributed (Lec 07) |
| $P_{X,Y}$ | “joint of X and Y” | The unknown probability law of the pair $(X,Y)$ |
| unknown | — | Nature does not hand you $P$; you only see $D$ |

**Labels optional.** If there is no $y$, write

$$
D=\{x_i\}_{i=1}^{n}\sim_{\mathrm{iid}} P_X
$$

### What “iid” contributes mathematically

If $Z_1,\ldots,Z_n$ are the data points (each $Z_i=(X_i,Y_i)$ or just $X_i$), **iid** means:

1. **Identical:** each $Z_i$ has the **same** distribution $P$.  
2. **Independent:** the joint of all $n$ points factors as a **product**:

$$
P(Z_1\in A_1,\ldots,Z_n\in A_n)
= P(Z_1\in A_1)\cdots P(Z_n\in A_n)
$$

In density-style writing (preview):

$$
p(z_1,\ldots,z_n)=p(z_1)\,p(z_2)\cdots p(z_n)
$$

**Product, not sum** (Lec 07 trap).

### Range-point reminder

Each $x_i$ is a **range point** of an RV $X:\Omega\to\mathbb{R}^{d}$ — a measurement, **not** a probability. Same for $y_i$.

### Analogy — purpose

A bag of tickets from a sealed drum.  
Bag = $D$; drum mix = $P$; iid ≈ each ticket drawn the same way without secret coupling.

---

## 2. The core job formula: estimate $P$

<a id="p2-estimate-p"></a>

### Purpose for the video

This is the one-line problem of the course.

### Formula

$$
\boxed{\;
\text{Given } D\sim_{\mathrm{iid}} P \text{ (unknown)},
\quad
\text{estimate } P
\;}
$$

| Piece | Meaning |
|-------|---------|
| Given $D$ | You start with samples only |
| $P$ unknown | The law is not observed directly |
| Estimate $P$ | Build $\hat{P}$ (or a useful piece of $P$) that could have produced $D$ |

**Second objective (generative):**

$$
\text{also learn to sample new } z\sim \hat{P}
$$

Estimating is **not** automatically sampling. Sampling needs a procedure that **draws** new points.

### What “estimate” does **not** mean

- Not “guess a single number and stop” (unless your target is a moment).  
- Not “memorize $D$ only” without a law for new points.  
- Not “receive $P$ from a file.”

### Analogy — purpose

Hear a song through a wall → rebuild the score (**estimate**).  
Build a piano that improvises new songs (**sample**).

---

## 3. Joint, margin, conditional — formulas decoded

<a id="p3-jcm"></a>

### Purpose for the video

Most of Lec 08 is “which of these objects am I estimating?”

### Joint

$$
P_{X,Y}(A,B)=P(X\in A,\ Y\in B)
$$

English: chance that $X$ lands in region $A$ **and** $Y$ lands in region $B$ (same experiment).

For discrete labels and a discrete toy feature, a **joint table** of cells $P(X=x,Y=y)$ is the joint.

### Marginals (remove one variable)

Discrete:

$$
P_X(x)=\sum_y P(X=x,Y=y),\qquad
P_Y(y)=\sum_x P(X=x,Y=y)
$$

Continuous (preview): replace $\sum$ by $\int$.

English: behavior of one variable after summing/integrating the other out (Lec 05 “table margins”).

### Conditional (fix the conditioner)

For events with $P(B)>0$:

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
$$

For RVs (discrete cells):

$$
P(Y=y\mid X=x)=\frac{P(X=x,Y=y)}{P(X=x)}
\quad\text{when }P(X=x)>0
$$

Density-style (preview, continuous):

$$
p(y\mid x)=\frac{p(x,y)}{p(x)}
$$

| Symbol | Meaning |
|--------|---------|
| $P(Y\mid X=x)$ | Law of $Y$ when $X$ is **fixed** at value $x$ |
| $P(X\mid Y=y)$ | Law of $X$ when $Y$ is **fixed** at value $y$ |

**Always:** the variable after the bar is held fixed.

### Micro numbers (joint table)

```
           Y=0    Y=1    row sum = P(X=·)
  X=a      0.1    0.2      0.3
  X=b      0.3    0.4      0.7
  col      0.4    0.6      1.0  = P(Y=·)
```

- Joint cell: $P(X=a,Y=1)=0.2$  
- Margin: $P(Y=1)=0.6$  
- Conditional: $P(Y=1\mid X=a)=0.2/0.3=2/3$  
- **Not** the same as $P(Y=1)=0.6$

### Analogy — purpose

Same hospital archive; different doors: both together / labels alone / labels given image / images given label.

---

## 4. Moments vs full distribution — formulas

<a id="p4-moments"></a>

### Purpose for the video

Sometimes the target is a **function of $P$**, not all of $P$.

### Expectation (first moment idea)

If $Z$ is a discrete RV with $P(Z=z)$:

$$
\mathbb{E}[Z]=\sum_z z\,P(Z=z)
$$

Continuous preview:

$$
\mathbb{E}[Z]=\int z\,p(z)\,dz
$$

English: probability-weighted average of the values $Z$ can take.

**Second moment** (idea): $\mathbb{E}[Z^2]$ — same weighted average using $z^2$.  
Variance-like summaries are built from moments. Moment-generating functions are another “function of $P$” family (name-drop only here).

| Job | Formula-level target |
|-----|----------------------|
| Full estimation | recover $P$ (or density $p$) |
| Moment estimation | recover $\mathbb{E}[g(Z)]$ for some $g$ (e.g. $g(z)=z$) |

### Analogy — purpose

Full height histogram of a city vs average height only.

---

## 5. Match each question to a formula

<a id="p5-match-question"></a>

### Purpose for the video

Casting real problems into math is the practitioner skill.

| Real question | Formula to estimate | English |
|---------------|---------------------|---------|
| Is **this** image diseased? | $P(Y\mid X=x)$ | label law given fixed image $x$ |
| Where is the tumor box? | $P(Y\mid X=x)$ with $Y=(c_x,c_y,h,w)$ or $(c,h,w)\in\mathbb{R}^{3}$ | continuous “label” geometry |
| How common is disease? | $P(Y=1)$ | label marginal; **no image** |
| How do pixels look (ignore labels)? | $P_X$ or $p(x)$ | data marginal |
| How do **diseased** images look? | $P(X\mid Y=1)$ | data law with disease fixed |
| Fill missing pixels | $P(Y\mid X=x)$ where $Y$ = missing coords of the same image | self-supervised packaging |

**Classification vs regression (when the target is $P(Y\mid X)$):**

| Name | $Y$ type |
|------|----------|
| Classification | $Y$ discrete (categories) |
| Regression | $Y\in\mathbb{R}^{k}$ continuous |

The **math job** is the same shape: estimate that conditional. Names track the type of $Y$.

### Analogy — purpose

Same archive, four questions, four formulas — do not use one door for all.

---

## 6. Discriminative vs generative (formulas of intent)

<a id="p6-disc-gen"></a>

### Purpose for the video

| Style | Intent formula |
|-------|----------------|
| **Discriminative** (this lecture’s use) | Produce $\hat{P}$ or $\widehat{P}(Y\mid X)$ etc. for decisions; **not** focused on drawing new $z$ |
| **Generative** | Also provide a way to draw $z_{\mathrm{new}}\sim \hat{P}$ |

Important: having $\hat{P}_{X}$ or $\hat{P}_{X,Y}$ as a mathematical object does **not** by itself mean you have a sampler. Sampling is a second capability.

This course ≈ estimation. Next generative course ≈ sampling.

---

## 7. Supervised / unsupervised (no new probability objects)

<a id="p7-sup-unsup"></a>

### Purpose for the video

| Name | Packaging |
|------|-----------|
| Supervised | $D$ includes labels $y_i$; algorithms use them |
| Unsupervised | $D=\{x_i\}$ only |

Math: still RVs on $\Omega$ and estimate a law. Inpainting shows $y$ can be **carved from $x$**. No new triple $(\Omega,F,P)$ appears just because of the name.

---

## 8. Conditioner fixed; $P$ vs density (notation)

<a id="p8-conditional-fixed"></a>

### Purpose for the video

### Conditional evaluation (how to read the board)

He writes objects like $P(X\mid Y)$ but means:

$$
\text{evaluate at } x \text{ with } Y \text{ fixed at } y
\quad=\quad
P(X\in\cdot \mid Y=y)
$$

| Symbol on the board | Role |
|---------------------|------|
| $x$ | **evaluation point** (dummy where you read the distribution) |
| $y$ | **fixed value** of the conditioner (not a free random input inside the same expression) |
| “$X$ given $Y$” | names the two RVs involved |

Change $y$ → you get a **different** conditional distribution.

### Distribution vs density (next lecture)

| Object | Role |
|--------|------|
| Distribution / measure $P$ or CDF-style $P_X$ | Sizes **sets** of outcomes / range regions |
| Density $p(x)$ | Continuous “height” function; probabilities come from **integrals** of $p$, not from $p(x)$ alone |

He uses script $\mathcal{P}$ for distribution functions; ML practice often works with densities next.  
**Trap:** $p(x)=0.7$ is not “probability 70%” of a point (Lec 04 / 06 trap).

### Paper check

1. Expand $D\sim_{\mathrm{iid}} P_{X,Y}$ into a full English sentence.  
2. Write the product formula for $n$ independent points.  
3. From the micro table, compute $P(Y=1\mid X=a)$.  
4. Write $\mathbb{E}[Z]$ for a discrete $Z$ and say what it means.  
5. Match: prevalence → $?$; diseased appearance → $?$.  
6. In $P(X\mid Y=y)$, what is fixed?  
7. Estimate $P$ vs sample from $P$ — one sentence each.

---

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).  
Prior: [Lec 07](../08-Lec07-IID-Assumption/NOTES.md).
