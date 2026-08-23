# Prerequisites — warm-up before Tutorial 9 (Review of Basic Probability 3)

> **Do this first** if “two numbers at once,” tables, or “product of laws” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues [Tutorial 7](../21-Tutorial07-Review-Basic-Probability-1/NOTES.md) and [Tutorial 8](../22-Tutorial08-Review-Basic-Probability-2/NOTES.md).  
> **Beginner deep warm-up:** definition · micro example · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "X and Y must live on the same experiment Ω."
  "A joint law is one number for a pair (x,y), not two separate stories."
  "A marginal is the joint with the other variable summed or integrated away."
  "A conditional is the joint sliced at a known y and re-normalized."
  "Independence means the joint factors for every window, not one lucky pair."
  "IID means independent and the same shape."
```

**Warm-up → tutorial boxes**

```
  §1  Same experiment, two labels        ──► Topics 1–2
  §2  A window on the plane              ──► Topics 1–4
  §3  Table of piles (joint PMF)         ──► Topic 3
  §4  Jam on a region (joint PDF)        ──► Topic 4
  §5  Summing a row or column            ──► Topic 5
  §6  Slice and re-normalize             ──► Topics 6–7
  §7  Product of two laws                ──► Topic 9
  §8  Copies + stretching a map          ──► Topic 10
```

---

## 1. Same experiment, two stickers

<a id="p1-same"></a>

### Purpose for the video

The first sentence of the definition is easy to skip: $X$ and $Y$ live on the **same** $(\Omega,\mathcal{F},P)$.

### Definitions

One roll of two dice is **one** $\omega=(w_1,w_2)$.  
$X(\omega)=\max(w_1,w_2)$ and $Y(\omega)=w_1+w_2$ are two functions of **that same** $\omega$.  
Together they are a map $\Omega\to\mathbb{R}^2$.

### Worked micro

$\omega=(2,5)$. Then $X=5$, $Y=7$. You do not roll once for $X$ and again for $Y$.

### Analogy — one photo, two measurements

One photograph (one $\omega$). Brightness is $X$. Number of faces is $Y$. Two numbers, **one** photo.

### Notice

- If $X$ came from Monday’s experiment and $Y$ from Tuesday’s, you have not defined a pair on one space.

### Mini-check

1. Can $X$ and $Y$ share $\omega=(3,3)$?  
2. Is “max” an outcome or a function of the outcome?

---

## 2. A window on the plane

<a id="p2-window"></a>

### Purpose for the video

The joint CDF is $P(\text{below }x\text{ and below }y)$. A **rectangle** of probability is four corners of $F$, not one.

### Picture

```
        y2  +--------+
            |        |
            |  want  |
        y1  +--------+
           x1        x2
```

$$
P(x_1<X\le x_2,\; y_1<Y\le y_2)
= F(x_2,y_2)-F(x_2,y_1)-F(x_1,y_2)+F(x_1,y_1)
$$

You add the big corner, subtract the two strips, add back the small corner you subtracted twice.

### Worked micro

If all mass sat in the unit square and $F$ were the area of $(-\infty,x]\times(-\infty,y]$, the formula is just the area of the window.

### Analogy — a tablecloth

$F(x,y)$ is “how much cloth is south-west of the pin $(x,y)$.” A window’s cloth is big rectangle minus two margins plus the overlap.

### Mini-check

1. Why add $F(x_1,y_1)$ back?  
2. What is $F(+\infty,+\infty)$?

---

## 3. A table of piles

<a id="p3-table"></a>

### Purpose for the video

Two discrete RVs have a **joint PMF**: a number in each cell $(i,j)$.

### Worked micro

Two fair dice, 36 equally likely pairs. Cell $(5,7)$ for $(\max,\text{sum})$ can happen as $(2,5)$ or $(5,2)$ — mass $2/36$. Cell $(3,6)$ is only $(3,3)$ — mass $1/36$.

### Analogy — a bakery tray

Each square of the tray holds some cookies (mass). Empty squares hold zero. All cookies together make 1.

### Notice

- Off the allowed $(m,n)$ pairs the mass is 0.  
- $n$ values of $X$ need not equal $m$ values of $Y$.

### Mini-check

1. Why is $n=2m$ a special case for $(\max,\text{sum})$?  
2. Must every cell be $\le 1$?

---

## 4. Jam on a region (not at points)

<a id="p4-jam"></a>

### Purpose for the video

A **joint PDF** $p(x,y)$ is a **height** over the plane. Probability is **volume** (area × height), never the height at one point.

### Worked micro

$p=2$ on the triangle $0<x<y<1$, else $0$. The triangle has area $1/2$, so volume $2\times\frac12=1$. Height 2 is legal.

$P(Y>X+0.5)$ is the volume over a **sub-triangle**, later $0.25$.

### Analogy — frosting

Spread 1 cup of frosting on a triangle of cake. Taller frosting on a smaller triangle. Asking $P(\text{region})$ is “how much frosting sits over that region?”

### Notice

- A picture of the **support** (the triangle) is not a 3-D plot of $p$. The lecture says so.

### Mini-check

1. Can $p(x,y)=2$ be legal?  
2. What must $\iint p\,dx\,dy$ equal?

---

## 5. The margin of the table

<a id="p5-margin"></a>

### Purpose for the video

A **marginal** is the joint with the other variable removed.

Discrete: $p_X(x_i)=\sum_j p(x_i,y_j)$ (sum the row).  
Continuous: $p_X(x)=\int p(x,y)\,dy$.

### Analogy — a spreadsheet

Each interior cell is a joint mass. The **right-hand margin** is the row sums — that is why they are called **marginals**.

### Notice

- From a joint you get **unique** marginals.  
- From two marginals you do **not** get a unique joint (many tables share the same row and column totals).

### Mini-check

1. How do you get $p_Y$ from a joint PMF?  
2. Do two Normal margins force a unique joint?

---

## 6. Slice and re-normalize

<a id="p6-slice"></a>

### Purpose for the video

$P(X\le x\mid Y=y)$ is the old “given $B$” idea with $B=\{Y=y\}$.

Need the slice to have **positive** mass (discrete) or a **positive density** (continuous). For a continuous $Y$, $P(Y=y)=0$, so they define the slice as a thin band $[y,y+\Delta]$ and let $\Delta\to 0$.

$$
p(x\mid y)=\frac{p(x,y)}{p_Y(y)}
$$

### Worked micro

Triangle $p=2$ on $0<x<y<1$. Given $Y=y$, $X$ is uniform on $(0,y)$: height $1/y$.

### Analogy — a loaf cut at $y$

You see only the slice $Y=y$. Re-share 100% of belief along that slice. Points off the slice are gone.

### Mini-check

1. Why divide by $p_Y(y)$?  
2. Why is $P(Y=y)=0$ a problem for continuous $Y$?

---

## 7. Product of two laws

<a id="p7-prod"></a>

### Purpose for the video

$X$ and $Y$ are **independent** when every pair of events factors:

$$
P(X\in B_1, Y\in B_2)=P(X\in B_1)P(Y\in B_2)
$$

**for all** $B_1,B_2$ — not one lucky pair. Then the joint CDF/PMF/PDF is the **product of the marginals**.

### Worked micro

Two separate coins: $P(\text{first H and second H})=\frac12\cdot\frac12$.  
Max and sum of the **same** two dice are **not** independent (the sum knows about the max).

### Analogy — two light switches

If the wires do not talk, “both on” is the product. If they share a hinge, you cannot multiply blindly.

### Notice

- Independence is the rare case where **marginals determine the joint**.

### Mini-check

1. Is one factored rectangle enough to claim independence?  
2. If $X\perp Y$, what is $p(x,y)$?

---

## 8. IID copies and a stretchy map

<a id="p8-iid"></a>

### Purpose for the video

**IID** = independent **and** identically distributed: same shape $p$, and they do not talk. The course’s default assumption on a dataset.

A map $Y=g(X)$ (vector to vector) moves the joint density by the **Jacobian** $|J|$ (how volumes stretch). Example: $Y_1=X_1+X_2$, $Y_2=X_1-X_2$ has $|J|=\frac12$.

### Analogy — photocopies vs a rubber sheet

IID: $n$ photocopies of the same law, stacked independently.  
Jacobian: draw a picture on a rubber sheet and stretch it; ink density changes by how much area grew.

### 2×2 Jacobian micro (Topic 10)

If $x_1=(y_1+y_2)/2$ and $x_2=(y_1-y_2)/2$, the four partials are $\pm 1/2$. The determinant is

$$
(0.5)(-0.5)-(0.5)(0.5)=-0.5
$$

The density multiplies by the **absolute** value $1/2$ (orientation can flip; mass cannot go negative).

### Mini-check

1. Can two independent Normals with **different** variances be IID?  
2. What does $|J|=\frac12$ do to density?  
3. Why take $\lvert\det J\rvert$ rather than $\det J$?

---

### Paper check

1. Must $X$ and $Y$ share $\Omega$?  
2. Formula for a rectangle of $F$?  
3. Joint $\to$ unique marginals? Reverse?  
4. $p(x\mid y)=$?  
5. Independence test in one line?  
6. IID expands to which two words?

**Peek:** (1) yes (2) $F_{22}-F_{21}-F_{12}+F_{11}$ (3) yes; no (4) $p(x,y)/p_Y(y)$ (5) joint $=$ product of marginals, all windows (6) independent and identically distributed

---

**Second teachers (names only here).** Khan Academy, Seeing Theory, StatQuest, 3Blue1Brown, Statlect. The actual pointers live at the end of [NOTES.md](./NOTES.md#external-references) — a few well-known items, not a link dump.

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html) Part A = this file.
