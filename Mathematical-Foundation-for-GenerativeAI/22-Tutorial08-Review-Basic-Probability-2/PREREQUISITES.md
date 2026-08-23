# Prerequisites — warm-up before Tutorial 8 (Review of Basic Probability 2)

> **Do this first** if density, integrals, expectation, or “height can be bigger than 1” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues [Tutorial 7](../21-Tutorial07-Review-Basic-Probability-1/NOTES.md) (discrete PMF/CDF).  
> **Beginner deep warm-up:** definition · micro example · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "A PDF is a height function whose area is probability."
  "A density can be taller than 1; a PMF cannot."
  "P(X equals one exact real) is 0 for a continuous-type RV."
  "Expectation is a weighted average (sum or integral)."
  "Variance is average squared distance from that average."
  "A histogram of many samples should look like the PDF."
```

**Warm-up → tutorial boxes**

```
  §1  Discrete leftover (PMF / CDF)     ──► Topic 1 recap
  §2  Height vs pile                    ──► Topics 1–2
  §3  Area under a curve                ──► Topics 2–3
  §4  Uncountable vs continuous-type    ──► Topic 1
  §5  Weighted average                  ──► Topic 5
  §6  Spread                            ──► Topic 6
  §7  Convex smile                      ──► Topic 7
  §8  Samples, seed, histogram          ──► Topics 8–10
```

---

## 1. Discrete leftover from Tutorial 7

<a id="p1-discrete"></a>

### Purpose for the video

The first two minutes only work if CDF and PMF still mean something.

### Definitions

| Object | Job |
|--------|-----|
| **PMF** $p(x)$ | Mass **at** a countable tick |
| **CDF** $F(x)=P(X\le x)$ | Running total from $-\infty$ to $x$ |
| Discrete $X$ | Completely specified by **either** $F$ or $p$ |

### Worked micro

Three fair coins, $X=$ heads. Masses $1/8,3/8,3/8,1/8$ at $0,1,2,3$. After $x=2$, $F(2)=7/8$.

### Analogy — buckets on a road

Discrete $X$ dumps whole spoonfuls at a few mile-markers. Continuous $X$ (this tutorial) **drips** along a stretch of road.

### Notice

- Same letter $p_X$ will mean **PDF** today. Read the type of $X$ first.

### Mini-check

1. Jump of $F$ at a tick equals what?  
2. Does a PMF have to sum to 1?

---

## 2. Height is not probability

<a id="p2-height"></a>

### Purpose for the video

The most common beginner crash: treating $p(x)$ like $P(X=x)$.

### Definitions

A **probability density function (PDF)** is a **height** function. Probability is **area**, not height.

A PMF value is already a probability (between 0 and 1). A PDF value can be **4**, or **100**, as long as the **area** is 1.

### Worked micro

Uniform on $[0,1/2]$: the interval is only half a unit long, so the height must be **2** to make area $2\times\frac12=1$. Height 2 is legal.

### Analogy — jam on toast

Spread 1 spoon of jam (total probability 1) on a **narrow** slice of toast. The jam gets **taller**. Tall jam is not “200% chance.”

### Notice

- $P(X=x)=0$ for a continuous-type RV, even when $p(x)$ is huge.  
- Lecture slogan: PDF is **not** a probability measure; CDF and PMF are.

### Mini-check

1. Can $p(0.1)=3$ for a legal PDF?  
2. What must $\int_{-\infty}^{\infty} p(x)\,dx$ equal?

---

## 3. Integral = area = probability of an interval

<a id="p3-area"></a>

### Purpose for the video

The CDF of a continuous-type RV is defined as an **integral**.

$$
F(x) = \int_{-\infty}^{x} p(t)\,dt
$$

$P(a < X < b)$ is the area of $p$ between $a$ and $b$.

### Worked micro

Unif$[0,1]$: $p=1$ on $[0,1]$. Area from $0$ to $0.3$ is a $0.3\times 1$ rectangle $=0.3$. So $F(0.3)=0.3$.

### Analogy — filling a measuring cup

Walk from $-\infty$ toward $x$ and pour every bit of area you pass into a cup. The water level is $F(x)$.

### Notice

- Where $p$ is continuous, $F'(x)=p(x)$ (fundamental theorem of calculus).  
- Because a single point has zero width, adding or dropping an endpoint does not change the area.

### Mini-check

1. For Unif$[0,1]$, what is $P(0.2 < X < 0.7)$?  
2. Why is $P(X=0.5)=0$ even though $p(0.5)=1$?

---

## 4. Uncountable range ≠ “continuous type”

<a id="p4-type"></a>

### Purpose for the video

The board warns: taking uncountably many values does **not** automatically make $X$ continuous-type.

### Definitions

| Phrase | Meaning here |
|--------|----------------|
| **Uncountable range** | $X$ can land on a continuum of numbers |
| **Continuous type** | There **exists** a PDF $p$ with $F=\int p$ |

You can invent mixed RVs that have both a jump and a density piece. Those are **not** continuous-type even if the range is uncountable.

### Analogy — a highway with a toll booth

A car can stop anywhere along a road (continuum) **and** pay a lump at one booth (an atom). Continuous-type means **no booths** — only road spray.

### Notice

- This lecture studies the clean case: a PDF exists, $F$ has no jumps, $P(X=x)=0$ everywhere.

### Mini-check

1. Does “uncountably many values” prove a PDF exists?  
2. If $F$ jumps, can $X$ be continuous-type?

---

## 5. Expectation as a weighted average

<a id="p5-mean"></a>

### Purpose for the video

$E[X]$ is one **number**: the balance point of the masses (discrete) or of the density (continuous).

### Definitions

Discrete: $E[X]=\sum x_i p(x_i)$.  
Continuous: $E[X]=\int x\, p(x)\,dx$.

Bernoulli with success $p$: $E[X]=0\cdot(1-p)+1\cdot p=p$.  
Indicator $1_B$: $E[1_B]=P(B)$.

### Worked micro

Fair die: $E[X]=(1+2+3+4+5+6)/6=3.5$. Not a face you can roll — still the average.

### Analogy — see-saw

Put mass $p(x)$ at position $x$. $E[X]$ is where you put the fulcrum so the plank does not tip.

### Notice

- $E[X]$ is not random. $X$ is the random object; $E[X]$ is a constant.  
- **LOTUS** (this lecture): to get $E[g(X)]$, do not find the law of $Y=g(X)$ first — plug $g(x)$ into the same weights.

### Mini-check

1. Why can $E[X]$ fail to be a possible value of $X$?  
2. $E[1_{\text{even}}]$ on a fair die?

---

## 6. Variance as spread

<a id="p6-var"></a>

### Purpose for the video

Variance asks: how far do values typically sit from $E[X]$?

$$
\mathrm{Var}(X)=E[(X-E[X])^2] = E[X^2]-(E[X])^2
$$

Always $\ge 0$. Adding a constant does not change spread. Multiplying by $c$ multiplies variance by $c^2$.

### Worked micro

Constant $X=5$: mean 5, every deviation 0, variance 0.  
Bernoulli $p$: $\mathrm{Var}=p(1-p)$ (largest at $p=1/2$).

### Analogy — classmates’ heights

Mean height is “typical.” Variance is “how spread out the class is.” Adding 10 cm of platform shoes to everyone (plus a constant) does not change spread.

### Notice

- Standard deviation is $\sigma=\sqrt{\mathrm{Var}}$ — same units as $X$.  
- Chebyshev later: no matter the shape, little mass lives many $\sigma$ away.

### Mini-check

1. $\mathrm{Var}(X+7)$ vs $\mathrm{Var}(X)$?  
2. $\mathrm{Var}(3X)$ vs $\mathrm{Var}(X)$?

---

## 7. Convex functions (for Jensen)

<a id="p7-convex"></a>

### Purpose for the video

Jensen’s inequality needs the word **convex**.

### Picture

A function is **convex** if its graph is a **smile** (or a straight V): the chord lies **above** the graph. $x^2$, $e^x$, $|x|$ are convex. $\log$ (on $(0,\infty)$) is concave (frown).

Jensen: for convex $g$,

$$
g(E[X]) \le E[g(X)]
$$

Average, then bend $\le$ bend, then average.

### Worked micro

$g(x)=x^2$ convex. Jensen says $(E[X])^2 \le E[X^2]$, which is $\mathrm{Var}(X)\ge 0$. Same fact, new name.

### Analogy — two highway exits

The elevation at the **average** mile-marker is below the **average of the elevations** if the road smiles.

### Mini-check

1. Is $x^2$ convex?  
2. What does Jensen say for $g(x)=x^2$?

---

## 8. Samples, seeds, and histograms

<a id="p8-sample"></a>

### Purpose for the video

The second half of the hour checks formulas with **numpy**.

### Definitions

| Idea | Meaning |
|------|---------|
| **Sample** | One simulated draw of $X$ |
| **Empirical mean** | Average of many draws $\approx E[X]$ |
| **Histogram** | Bar plot of how often draws land in bins $\approx$ PDF |
| **Seed** | Starting number so the “random” stream is repeatable (`42` in the notebook) |

### Worked micro

100,000 fair die rolls. About $1/6$ should be fours. Live board: $0.1669$ vs $0.1666\ldots$

### Analogy — poll a huge crowd

Theory says 50% even. Ask 100,000 rolls. You should sit near 50%, not exactly 50%. More asks → closer.

### Notice

- `rng.integers(1, 7)` in numpy is **1,2,3,4,5,6** (high end exclusive). That is why they wrote 7.  
- Small $n$ (10 tosses) will **not** match theory closely — they ask you to try it.

### Also unlock (same section — used in Topics 4 and 10)

- **Monotone inverse:** if $g$ only rises or only falls you can undo it. $x^2$ on all reals goes both ways — the board CoV formula stops.  
- **Product of means:** $E[XY]=E[X]E[Y]$ only when $X,Y$ are independent. $Y=2X+\text{noise}$ usually fails the product.  
- **Empirical CDF:** sort $n$ samples; jump $1/n$ at each. A Normal’s empirical $F$ looks like a **sigmoid**.

### Mini-check

1. Why fix a seed?  
2. Why is the second argument of `integers` equal to 7 for a six-face die?  
3. When may you write $E[XY]=E[X]E[Y]$?

---

### Paper check

1. PDF vs PMF: which can exceed 1?  
2. $F(x)$ for a CRV is which integral?  
3. $E[1_B]=$?  
4. $\mathrm{Var}(X+c)=$?  
5. Jensen needs $g$ to be what?  
6. When may you write $E[XY]=E[X]E[Y]$?  
7. What does an empirical CDF jump by at each sample?

**Peek:** (1) PDF (2) $\int_{-\infty}^x p$ (3) $P(B)$ (4) $\mathrm{Var}(X)$ (5) convex (6) independence (7) $1/n$

---

**Second teachers (names only here).** Khan Academy, Seeing Theory, StatQuest, 3Blue1Brown, Statlect. The actual pointers live at the end of [NOTES.md](./NOTES.md#external-references) — a few well-known items, not a link dump.

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html) Part A = this file.
