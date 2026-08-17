# Tutorial 8 — Review of Basic Probability 2

**Video:** [Tutorial 8 : Review of Basic Probability 2](https://www.youtube.com/watch?v=pQIbfyjSnFk) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Tutorial 7 — Probability 1](../21-Tutorial07-Review-Basic-Probability-1/NOTES.md)  
**Course:** Mathematical Foundations of **Generative AI** (~57 min)  
**Speaker:** NPTEL IISc · Continuous RVs, PDF, CoV, $E$, Var, inequalities, numpy

---

## Table of Contents

1. [Topic 1 — Continuous RV and PDF](#topic-1-continuous-rv-and-pdf-0003–0452) (00:03–04:52)
2. [Topic 2 — PDF properties versus PMF](#topic-2-pdf-properties-versus-pmf-0452–0843) (04:52–08:43)
3. [Topic 3 — Uniform, exponential, Gaussian](#topic-3-uniform-exponential-gaussian-0843–1357) (08:43–13:57)
4. [Topic 4 — Functions of RVs and change of variables](#topic-4-functions-of-rvs-and-change-of-variables-1357–2024) (13:57–20:24)
5. [Topic 5 — Expectation and LOTUS](#topic-5-expectation-and-lotus-2024–2610) (20:24–26:10)
6. [Topic 6 — Variance](#topic-6-variance-2610–2902) (26:10–29:02)
7. [Topic 7 — Markov, Chebyshev, Jensen](#topic-7-markov-chebyshev-jensen-2902–3317) (29:02–33:17)
8. [Topic 8 — Numpy dice and Bayes](#topic-8-numpy-dice-and-bayes-3317–4009) (33:17–40:09)
9. [Topic 9 — Discrete sampling families](#topic-9-discrete-sampling-families-4009–4603) (40:09–46:03)
10. [Topic 10 — Continuous simulation and recap](#topic-10-continuous-simulation-and-recap-4603–5713) (46:03–57:13)
11. [Apply it (scenarios)](#apply-it-scenarios)
12. [External references](#external-references)
13. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

**Job:** install continuous-type random variables after last hour’s discrete PMF/CDF.  
**Method:** define a PDF so $F$ is an integral, stock Unif/Exp/Normal, push $Y=g(X)$, then $E$, LOTUS, Var, and three inequalities.  
**Fork:** check the same formulas with numpy samples; pairs of RVs wait until next time.

**Worldview arc:** from “discrete $X$ specified by CDF or PMF” **to** “continuous $X$ specified by a PDF + $E$/Var checked in simulation.”

### System context

```
  ╔══════════════════════════════════╗
  ║ Outside: vector / pair RVs next  ║
  ║ Outside: inequality proofs       ║
  ╚══════════════╤═══════════════════╝
                 │ this tutorial (~57 min)
                 ▼
        ┌────────────────────────────┐
        │ CRV stack: p → E → Var     │
        │ + numpy histogram checks   │
        └────────────────────────────┘
```

### Main blueprint

```
  Discrete leftover (T7): F or p_mass specifies X
                 │
                 ▼
  ┌──────── Continuous type ─────────┐
  │ exists p:ℝ→ℝ                     │
  │ F(x)=∫_{-∞}^x p(t) dt            │
  │ F continuous; p=F′ where p cts   │
  │ P(X=x)=0; PDF can exceed 1       │
  └────────────┬─────────────────────┘
               │ named p
               ▼
        Unif / Exp / N(μ,σ²)
               │ Y=g(X)
               ▼
        CoV: p_Y(y)=p_X(g^{-1}(y)) | (g^{-1})′ |
               │
               ▼
        E[X]=∑xp  or  ∫ x p dx     (a number)
        LOTUS: E[g(X)] uses p_X
        linearity
               │
               ▼
        Var=E[(X-μ)²]=E[X²]-μ² ≥ 0
        Var(X+c)=Var(X)  Var(cX)=c²Var(X)
               │
               ▼
        Markov → Chebyshev → Jensen (convex)
               │
               ▼
        numpy: seed 42 · 1e5 rolls · hist ≈ p
```

### Scenario walkthrough

1. Recap discrete; define CRV via PDF.  
2. Contrast PDF with PMF; interval probabilities are areas.  
3. Write Unif, Exp, Gaussian formulas.  
4. Composite $Y=g(X)$; linear CDF; CoV theorem.  
5. $E$ as weighted average; LOTUS; linearity.  
6. Variance as mean squared deviation.  
7. Three inequalities, no proofs.  
8. Colab: die frequencies + Bayes $1\%/95\%/5\%$.  
9. Indicator, Bernoulli, choice, binomial samples.  
10. Unif/Normal/Exp hists; $Y=aX+b$; $E[XY]$; recap.

### Failure / contrast path

```
  Uncountable range ⇒ continuous type     ──X──► mixed RVs exist
  p(x)=3 declared illegal                 ──X──► height ≠ probability
  P(X=x) read as p(x)                     ──X──► point mass that is 0
  CoV when g′ changes sign                ──X──► theorem off
  E[XY]=E[X]E[Y] for Y=2X+noise           ──X──► dependence
```

### STOP / out of scope

Vector-valued RVs; pairs (next tutorial); proofs of Markov/Chebyshev/Jensen.

### Load-bearing claims (closed-book)

- Continuous type $\Leftrightarrow$ a PDF exists with $F=\int p$.  
- $P(X=x)=0$; PDF $\ge 0$, integrates to 1, **may exceed 1**.  
- Unif / Exp / $N(\mu,\sigma^2)$ formulas.  
- Monotone $g$: change-of-variable for $p_Y$.  
- $E$ + LOTUS + linearity; $\mathrm{Var}(X)=E[X^2]-\mu^2$.  
- Chebyshev is distribution-free.  
- Many i.i.d. samples make histograms sit on $p$.

**Speaker / course:** NPTEL IISc · Tutorial 8.

---

## Topic 1: Continuous RV and PDF (00:03–04:52)

### Where this sits on the master map

**NEW TYPE** — After discrete PMF/CDF, install a density. Warm-up: [discrete leftover](./PREREQUISITES.md#p1-discrete) · [type](./PREREQUISITES.md#p4-type).

### Board / screenshot

![Continuous RV definition](./screenshots/composites/ch01-topic-01-continuous-pdf-def-panel1of1.png)

**Figure — ~02:40–04:34:** $X$ continuous type if $\exists\, p_X:\mathbb{R}\to\mathbb{R}$ with $F_X(x)=\int_{-\infty}^x p_X(t)\,dt$; $F$ continuous; FTC $F'=p$ where $p$ is continuous; uncountable $\not\Rightarrow$ continuous type; $P(X=x)=F(x)-F(x^-)=0$.

### What he is establishing

Last tutorial built the probability space, conditionals, Bayes, independence, and **discrete** random variables: CDF, PMF, Bernoulli/binomial/geometric/Poisson. A discrete RV is completely specified by **either** its CDF or its PMF.

This hour continues with the other kind they will use most: **continuous-type** random variables, then functions of RVs, then expectation and variance. (Numpy sampling comes after the theory.)

$X$ is **continuous** (continuous type) if there exists a function $p_X:\mathbb{R}\to\mathbb{R}$ — the **probability density function (PDF)** — such that for every $x$,

$$
F_X(x) = \int_{-\infty}^{x} p_X(t)\,dt
$$

They reuse the same letters as the PMF. You must read the **type** of $X$ to know which object $p_X$ is.

By this definition $F$ is continuous at every $x$. Where $p_X$ itself is continuous, the fundamental theorem of calculus gives

$$
F_X'(x) = p_X(x)
$$

Continuous-type RVs take **uncountably many** values. The converse is false: an uncountable range does **not** force continuous type (mixed laws exist).

Because $F$ has the same left and right limit at every $x$,

$$
P(X=x) = F(x)-F(x^-) = 0 \qquad \text{for all } x
$$

You can now say what “continuous type” means and refuse the uncountable shortcut. Still missing: the two PDF axioms and how they differ from a PMF.

A common trap is calling every “real-valued” $X$ continuous-type.

### Analogy for this topic only

Discrete $X$ is a few water buckets on a road. Continuous-type $X$ is a **hose** spraying along the road. The water depth at a mile is the density; the water you collected up to that mile is $F$. A road with both a hose and a bucket is **not** continuous-type.

Question: **Does an uncountable range prove a PDF exists?**

In lecture words: this box is the PDF definition.

### Local picture

```
  F(x) = area of p to the left of x

  −∞ ========|======== x =======►
             [  area F(x)  ]

  P(X = one point) = width 0 = 0
```

**Notice:** $p$ is recovered by differentiating $F$ where $p$ is continuous.

### Bridge

We have a name for $p$. What **rules** must $p$ obey, and how is that different from a PMF?

---

## Topic 2: PDF properties versus PMF (04:52–08:43)

### Where this sits on the master map

**AXIOMS OF $p$** — Area 1; height free. Warm-up: [height](./PREREQUISITES.md#p2-height) · [area](./PREREQUISITES.md#p3-area).

### Board / screenshot

![PDF properties](./screenshots/composites/ch02-topic-02-pdf-properties-panel1of1.png)

**Figure — ~04:52–08:43:** $p\ge 0$, $\int p=1$; $p$ need not lie in $[0,1]$; PDF is not a probability measure; $F(-\infty)=0$, $F(+\infty)=1$, $F$ nondecreasing **and continuous**; $P(a<X\le b)=F(b)-F(a)=\int_a^b p$.

### What he is establishing

A PDF is an $\mathbb{R}\to\mathbb{R}$ function that must satisfy

$$
p(x) \ge 0 \quad \text{for all } x, \qquad \int_{-\infty}^{\infty} p(x)\,dx = 1
$$

Compare with a PMF: masses sit in $[0,1]$ and **sum** to 1. The integral replaces the sum. The **$[0,1]$ cap is gone**. $p(x)=3$ can be legal.

That is why they say the PDF is **not** a valid probability measure, while the CDF and the PMF are.

Any function meeting those two conditions is a PDF of some continuous-type RV. Then automatically $F(-\infty)=0$, $F(+\infty)=1$, $F$ is nondecreasing, and — unlike the general CDF, which is only right-continuous — $F$ is **continuous**.

For an interval,

$$
P(a < X \le b) = F(b)-F(a) = \int_a^b p(t)\,dt
$$

Because $P(X=a)=P(X=b)=0$, the four versions $<$ / $\le$ at either end are **equal**. They usually write $\le$ everywhere.

You can now check a proposed $p$ and compute interval probabilities as areas. Still missing: the three named densities they will sample later.

A common trap is rejecting $p>1$. Another is treating $p(x)$ as $P(X=x)$.

### Analogy for this topic only

1 kg of jam (total probability) on a 10 cm slice of toast must sit **10 times taller** than on a 100 cm slice. Height 10 is not “1000% chance.” Chance is how much jam sits **over an interval**.

Question: **Must every PDF value lie in $[0,1]$?**

In lecture words: this box is the PDF axioms.

### Local picture

```
  PMF:  piles at ticks, each pile in [0,1], piles sum to 1
  PDF:  height function ≥ 0, area = 1, height may be 2, 10, …

  P(a ≤ X ≤ b) = P(a < X < b) = area from a to b
```

**Notice:** endpoints are free because points have no area.

### Bridge

Theory is empty without examples. What do Unif, Exp, and Gaussian actually look like?

---

## Topic 3: Uniform, exponential, Gaussian (08:43–13:57)

### Where this sits on the master map

**NAMED DENSITIES** — Three workhorses. Warm-up: [area](./PREREQUISITES.md#p3-area).

### Board / screenshot

![Named continuous families](./screenshots/composites/ch03-topic-03-uniform-exp-gaussian-panel1of1.png)

**Figure — ~08:43–13:57:** Unif$[a,b]$ height $1/(b-a)$; Unif$[0,1]$ PDF $=1$, CDF $=x$ on $(0,1)$; Exp PDF/CDF; Normal $1/(\sigma\sqrt{2\pi})\exp(-(x-\mu)^2/(2\sigma^2))$; $X\sim N(\mu,\sigma^2)$; standard normal $\mu=0,\sigma^2=1$.

### What he is establishing

**Uniform** on $[a,b]$: $p(x)=1/(b-a)$ for $x\in[a,b]$, else $0$. Open or closed interval is the same (endpoints have probability 0). On $[0,1]$ the height is $1$; the CDF is $0$ for $x\le 0$, $x$ on $(0,1)$, and $1$ for $x\ge 1$.

**Exponential**$(\lambda)$ with $\lambda>0$: $p(x)=\lambda e^{-\lambda x}$ for $x\ge 0$, and $0$ for $x<0$. Split the integral at 0: the left half is 0, the right half is 1. The CDF is $0$ for $x<0$ and $1-e^{-\lambda x}$ for $x\ge 0$.

They flag that **numpy histograms** of samples from each family come in a few minutes.

**Gaussian / Normal** — “king of distributions” because the course uses it constantly:

$$
p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\Bigl(-\frac{(x-\mu)^2}{2\sigma^2}\Bigr), \qquad \sigma>0,\; \mu\in\mathbb{R}
$$

$\mu$ is the mean, $\sigma^2$ the variance, $\sigma$ the standard deviation. Write $X\sim N(\mu,\sigma^2)$. **Standard normal:** $\mu=0$, $\sigma^2=1$. Still **scalar** $X$; vector RVs later.

You can now write the three PDFs and the Unif/Exp CDFs. Still missing: what happens to $p$ after $Y=g(X)$.

A common trap is putting $\sigma$ (not $\sigma^2$) in the $N(\cdot,\cdot)$ slot.

### Analogy for this topic only

Uniform is **even jam** on a finite slice. Exponential is **jam piled at 0** that thins as you walk right. Gaussian is a **symmetric hill** centered at $\mu$ whose width is $\sigma$.

Question: **What two numbers specify a Normal family?**

In lecture words: this box is the named CRVs.

### Local picture

```
  Unif[0,1]     p:  ____1____          F:  ramp 0→1
  Exp(λ)        p:  drop from λ        F:  1-e^{-λx} (x>0)
  N(μ,σ²)       p:  bell at μ          F:  (no elementary closed form)
```

**Notice:** they will overlay histograms on these $p$ curves in Topic 10.

### Bridge

Often we do not observe $X$ but $Y=g(X)$ (a bill, a square, a scale). How does the law move?

---

## Topic 4: Functions of RVs and change of variables (13:57–20:24)

### Where this sits on the master map

**PUSH FORWARD** — $Y=g(X)$. Warm-up: Tutorial 7 [function](../21-Tutorial07-Review-Basic-Probability-1/PREREQUISITES.md#p2-fn).

### Board / screenshot

![Change of variables](./screenshots/composites/ch04-topic-04-functions-change-var-panel1of1.png)

**Figure — ~13:57–20:24:** $Y=g(X)$ as $\Omega\xrightarrow{X}\mathbb{R}\xrightarrow{g}\mathbb{R}$; $F_Y(y)=P(g(X)\le y)$; linear $a>0$; **change of variable theorem** with $g'>0$ or $g'<0$ everywhere.

### What he is establishing

$X:\Omega\to\mathbb{R}$. Take $g:\mathbb{R}\to\mathbb{R}$ and set $Y=g(X)$. Then $Y$ is also $\Omega\to\mathbb{R}$ — a composite.

Hotel picture (same world as Tutorial 7): you may order one dish. $X$ is the **bill**. Paying by UPI changes the account by a function of the bill. The account drop is $Y=g(X)$.

If $g$ is “nice,” $Y$ is a random variable. Its CDF is

$$
F_Y(y) = P(Y\le y) = P\bigl(g(X)\le y\bigr) = P\bigl(X\in \{z : g(z)\le y\}\bigr)
$$

Knowing the law of $X$ gives the law of $Y$ **in principle**.

Linear example: $Y=aX+b$ with $a>0$. Then $F_Y(y)=F_X((y-b)/a)$.

Work numbers: $a=2$, $b=1$, $X\sim\mathrm{Unif}[0,1]$. Then $Y$ lives on $[1,3]$ and $F_Y(2)=F_X((2-1)/2)=F_X(0.5)=0.5$. The later Colab uses $a=-4$ (decreasing) — the CDF formula above needs $a>0$; the **CoV** formula with the absolute value still handles $a<0$.

**Change of variable theorem** (used often later). Assume $g$ is differentiable and **either** $g'(x)>0$ for all $x$ **or** $g'(x)<0$ for all $x$ (one sign globally — not mixed). Let $X$ be continuous type and $Y=g(X)$. Then $Y$ is continuous type with

$$
p_Y(y) = p_X\bigl(g^{-1}(y)\bigr) \left|\frac{d}{dy} g^{-1}(y)\right|
$$

on the interval from $a=\min\{g(+\infty),g(-\infty)\}$ to $b=\max\{\ldots\}$. Remember the **conditions** before you use it.

You can now push a monotone $g$ and refuse CoV when $g$ folds. Still missing: a single number that summarizes $X$ — expectation.

A common trap is applying CoV to $Y=X^2$ on all of $\mathbb{R}$ ($g'$ changes sign at 0).

### Analogy for this topic only

$X$ is the hotel price tag. $g$ is “what the UPI app subtracts.” The map from menu $\to$ account is the **composite**. If $g$ is a steadily rising (or steadily falling) slider, you can read the new density off the old one. If $g$ goes up then down, two prices can share one output — the simple theorem stops.

Question: **What must be true of $g'$ for the board’s CoV theorem?**

In lecture words: this box is $Y=g(X)$ and CoV.

### Local picture

```
  Ω --X--> ℝ --g--> ℝ
            Y = g ∘ X

  a>0:   F_Y(y) = F_X((y-b)/a)

  CoV:   p_Y(y) = p_X(x(y)) · |dx/dy|
         need g′ always + or always −
```

**Notice:** the absolute value handles decreasing $g$.

### Bridge

We can move laws. The course also wants a **center**: $E[X]$.

---

## Topic 5: Expectation and LOTUS (20:24–26:10)

### Where this sits on the master map

**CENTER** — One number; LOTUS for $g(X)$. Warm-up: [mean](./PREREQUISITES.md#p5-mean).

### Board / screenshot

![Expectation and LOTUS](./screenshots/composites/ch05-topic-05-expectation-lotus-panel1of1.png)

**Figure — ~20:24–26:10:** $E[X]=\sum x_i p_i$ or $\int x p(x)\,dx$; Bernoulli $E=p$; $E[1_B]=P(B)$; LOTUS $\sum g(x_i)p_i$ / $\int g p$; positivity; $E[c]=c$; linearity.

### What he is establishing

**Discrete:** $E[X]=\sum_i x_i p(x_i)$ — a weighted average.  
**Continuous:** $E[X]=\int_{-\infty}^{\infty} x\, p(x)\,dx$.  
The same letter $p$ is PMF or PDF by type. $E[X]$ is a **scalar** (they still treat scalar $X$). They often write $EX$ without brackets.

Bernoulli: $E[X]=0\cdot(1-p)+1\cdot p=p$.  
Indicator: $E[1_B]=P(B)$.  
They list (without deriving here) means for the stock families. Standard values you should recognize when the list flashes by:

| Family | $E[X]$ | $\mathrm{Var}(X)$ |
|--------|--------|-------------------|
| Bernoulli $p$ | $p$ | $p(1-p)$ |
| Binomial$(n,p)$ | $np$ | $np(1-p)$ |
| Unif$[a,b]$ | $(a+b)/2$ | $(b-a)^2/12$ |
| Exp$(\lambda)$ | $1/\lambda$ | $1/\lambda^2$ |
| $N(\mu,\sigma^2)$ | $\mu$ | $\sigma^2$ |

(Poisson/geometric sit on the same list from Tutorial 7.)

**LOTUS** — law of the unconscious statistician. For $Y=g(X)$,

$$
E[Y] = \sum_i g(x_i)\,p_X(x_i) \quad\text{or}\quad \int g(x)\,p_X(x)\,dx
$$

You apply $g$ **unconsciously** to each $x$ and keep $X$’s weights. You do not first find $p_Y$. The proof is long (~3 pages); they ask you to look it up. “Unconscious” does not mean asleep.

Properties they want in muscle memory:

- $X\ge 0$ $\Rightarrow$ $E[X]\ge 0$  
- $E[c]=c$  
- $E[a\,g(X)]=a\,E[g(X)]$  
- **Linearity:** $E[a g_1(X)+b g_2(X)]=a E[g_1]+b E[g_2]$

You can now compute $E$ and $E[g(X)]$ without a new density. Still missing: a measure of **spread**.

A common trap is finding $p_Y$ first when LOTUS already gives $E[Y]$.

### Analogy for this topic only

Each possible bill $x$ has a chance $p(x)$. $E[X]$ is the long-run average bill. LOTUS: to get the average **UPI drop** $g(x)$, do not rebuild the drop’s histogram — just average $g(x)$ with the same bill-chances.

Question: **What is $E[1_B]$?**

In lecture words: this box is expectation + LOTUS.

### Local picture

```
  discrete:   E[X] = Σ x · (pile at x)
  continuous: E[X] = ∫ x · (height p) dx

  LOTUS:      E[g(X)] = Σ g(x) · pile(x)   (same piles)
```

**Notice:** $E[X]$ is not random.

### Bridge

Two RVs can share a mean and still feel different. How do we measure **spread**?

---

## Topic 6: Variance (26:10–29:02)

### Where this sits on the master map

**SPREAD** — Mean squared deviation. Warm-up: [variance](./PREREQUISITES.md#p6-var).

### Board / screenshot

![Variance](./screenshots/composites/ch06-topic-06-variance-panel1of1.png)

**Figure — ~26:10–29:02:** $\mathrm{Var}(X)=E[(X-EX)^2]=E[X^2]-(EX)^2$; $\mathrm{Var}\ge 0$; $\mathrm{Var}(X+c)=\mathrm{Var}(X)$; $\mathrm{Var}(cX)=c^2\mathrm{Var}(X)$.

### What he is establishing

$$
\mathrm{Var}(X) = E\bigl[(X-E[X])^2\bigr]
$$

$X-E[X]$ is the deviation from the mean; squaring, then taking $E$, is an **average squared deviation**.

Expand the square and use linearity ($E[X]$ is a constant):

$$
\mathrm{Var}(X) = E[X^2] - (E[X])^2
$$

$E[X^2]$ is $E[g(X)]$ for $g(x)=x^2$ — **LOTUS**.

$(X-EX)^2\ge 0$, so its expectation is $\ge 0$. Hence $\mathrm{Var}(X)\ge 0$ and $E[X^2]\ge (E[X])^2$.

Properties: $\mathrm{Var}(X+c)=\mathrm{Var}(X)$ (shift does not change spread). $\mathrm{Var}(cX)=c^2\mathrm{Var}(X)$. They ask you to substitute $X+c$ into the definition and watch $c$ cancel.

Micro check: Bernoulli $p=0.7$ has $\mathrm{Var}=0.7\cdot 0.3=0.21$. The notebook later estimates $\approx 0.209$ from 100,000 tosses. Fair die: $E[X]=3.5$, $E[X^2]=(1^2+\cdots+6^2)/6=91/6$, so $\mathrm{Var}=91/6-(3.5)^2=35/12\approx 2.92$.

You can now compute Var two ways and scale it. Still missing: inequalities that need only $E$ or Var, not the full law.

A common trap is writing $\mathrm{Var}(cX)=c\,\mathrm{Var}(X)$.

### Analogy for this topic only

Class mean height is $E[X]$. Variance is “how far typical students sit from that mean,” after you square so tall and short both count. Give everyone 5 cm platform shoes: mean moves, **spread does not**. Stretch every deviation by 3: squared spread becomes $9\times$.

Question: **Does adding 10 to $X$ change $\mathrm{Var}(X)$?**

In lecture words: this box is variance.

### Local picture

```
  deviation   X − μ
  square      (X − μ)²  ≥ 0
  average     Var(X)    ≥ 0

  Var(X+c)=Var(X)     Var(cX)=c² Var(X)
```

**Notice:** $E[X^2]$ is not $(E[X])^2$ unless Var is 0.

### Bridge

Sometimes you only know $E$ or Var and still want a **tail bound**.

---

## Topic 7: Markov, Chebyshev, Jensen (29:02–33:17)

### Where this sits on the master map

**BOUNDS** — No full law required. Warm-up: [convex](./PREREQUISITES.md#p7-convex).

### Board / screenshot

![Inequalities](./screenshots/composites/ch07-topic-07-inequalities-panel1of1.png)

**Figure — ~29:02–33:17:** Markov $P(|X|\ge c)\le E[|X|^k]/c^k$; Chebyshev $P(|X-\mu|\ge c)\le\mathrm{Var}/c^2$ and $P(|X-\mu|\ge k\sigma)\le 1/k^2$; Jensen $g(E[X])\le E[g(X)]$ for convex $g$.

### What he is establishing

They **state** three inequalities and skip proofs (easy to find; look them up).

**Markov.** For $c>0$ and suitable $k$,

$$
P(|X| \ge c) \le \frac{E[|X|^k]}{c^k}
$$

**Chebyshev** is Markov on $|X-\mu|$ with $k=2$:

$$
P(|X-\mu|\ge c) \le \frac{\mathrm{Var}(X)}{c^2}
$$

Set $c=k\sigma$ with $\sigma=\sqrt{\mathrm{Var}}$:

$$
P\bigl(|X-\mu|\ge k\sigma\bigr) \le \frac{1}{k^2}
$$

These hold for **all** random variables — no Gaussian assumption.

**Jensen.** If $g:\mathbb{R}\to\mathbb{R}$ is **convex**,

$$
g\bigl(E[X]\bigr) \le E\bigl[g(X)\bigr]
$$

They assume you know convex vs concave; if not, look it up. Jensen will be used “a lot” later.

Theory block ends. Next: **Colab / numpy** implementations.

You can now quote the three bounds and their hypotheses. Still missing: seeing the same $E$ and $p$ appear as sample averages.

A common trap is applying Chebyshev only to Gaussians. Another is Jensen with a **concave** $g$ and the same inequality direction.

### Analogy for this topic only

Markov/Chebyshev: “I only know the average (or the spread), and I still get a **speed limit** on how much mass can sit far away.” Jensen: on a smiling road, the elevation at the average mile is below the average of the elevations.

Question: **Does Chebyshev need $X$ to be Normal?**

In lecture words: this box is the three inequalities.

### Local picture

```
  Markov     tail of |X|     ≲  moment / c^k
  Chebyshev  tail of |X-μ|   ≲  σ² / c²     (any law)
  Jensen     g convex        g(EX) ≤ E[g(X)]
```

**Notice:** $g(x)=x^2$ in Jensen is $\mathrm{Var}\ge 0$ again.

### Bridge

Formulas need a sandbox. How do you **estimate** $P(4)$ on a die with 100,000 rolls?

---

## Topic 8: Numpy dice and Bayes (33:17–40:09)

### Where this sits on the master map

**CHECK ON A MACHINE** — Empirical $P$ vs theory. Warm-up: [samples](./PREREQUISITES.md#p8-sample).

### Board / screenshot

![Numpy dice and Bayes](./screenshots/composites/ch08-topic-08-numpy-dice-bayes-panel1of1.png)

**Figure — ~33:17–40:09:** `NPTEL_Gen_AI_Basics.ipynb`; numpy 2.0.2; `default_rng(42)`; `integers(1,7)` 100000 rolls; $P(4)\approx 0.1669$ vs $1/6$; Bayes $P(D)=0.01$, $P(T^+|D)=0.95$, $P(T^+|D^c)=0.05$.

### What he is establishing

```python
import numpy as np
import matplotlib.pyplot as plt
from math import comb, factorial, pi, sqrt, exp, log

rng = np.random.default_rng(42)          # reproducibility
np.set_printoptions(precision=4, suppress=True)
print("NumPy version:", np.__version__)  # 2.0.2 on the board
```

`comb` is for binomial PMFs; `pi`, `sqrt`, `exp` for Gaussian/Exp PDFs.

**Die experiment.** 100,000 rolls (underscores allowed: `100_000`). `rng.integers(1, 7)` is faces $1\ldots 6$ (high end exclusive — that is why they wrote 7).

```python
n_trials = 100_000
die_rolls = rng.integers(1, 7, size=n_trials)

p_four = np.mean(die_rolls == 4)
p_even = np.mean(die_rolls % 2 == 0)
p_greater_than_3 = np.mean(die_rolls > 3)
p_even_and_gt3 = np.mean((die_rolls % 2 == 0) & (die_rolls > 3))
```

Theory: $1/6$, $3/6$, $3/6$, $2/6$. Live: $P(4)\approx 0.1669$, $P(\text{even})\approx 0.4996$, $P(\text{even and }>3)\approx 0.332$. Pause and compute theory first.

**Bayes numeric.** $P(D)=0.01$, $P(T^+|D)=0.95$, $P(T^+|D^c)=0.05$. Then $P(T^+)=0.95\cdot 0.01+0.05\cdot 0.99$ and $P(D|T^+)=P(T^+|D)P(D)/P(T^+)$. Compute by hand, then compare.

```python
p_disease = 0.01
p_positive_given_disease = 0.95
p_positive_given_no_disease = 0.05
p_positive = (
    p_positive_given_disease * p_disease
    + p_positive_given_no_disease * (1 - p_disease)
)
p_disease_given_positive = (
    p_positive_given_disease * p_disease / p_positive
)
```

You can now estimate events with `np.mean(mask)` and flip Bayes in code. Still missing: named discrete families as `rng` samplers.

A common trap: `integers(1,6)` — that **drops face 6**.

### Analogy for this topic only

Theory is the recipe. 100,000 rolls are 100,000 taste tests. The kitchen should sit near $1/6$, not land on it exactly. Seed 42 is “start the same cookbook every time.”

Question: **Why is the high end of `integers` equal to 7?**

In lecture words: this box is empirical $P$ + numeric Bayes.

### Local picture

```
  mask == 4     →  mean = estimated P(4)
  1e5 rolls     →  0.1669  vs  0.1666…

  Bayes:  0.01, 0.95, 0.05  →  P(D|T+) via total law
```

**Notice:** `mean` of a Boolean array is a frequency.

### Bridge

The same `mean` trick should recover $E[1_A]=P(A)$ and Bernoulli $p$.

---

## Topic 9: Discrete sampling families (40:09–46:03)

### Where this sits on the master map

**DISCRETE RNG** — Indicator, Bernoulli, choice, binomial. Warm-up: [samples](./PREREQUISITES.md#p8-sample).

### Board / screenshot

![Discrete sampling](./screenshots/composites/ch09-topic-09-discrete-sampling-panel1of1.png)

**Figure — ~40:09–46:03:** 20 die outcomes + indicator of 6; $1_{\ge 5}$ mean $\approx P(\ge 5)$; `binomial(1,p)` as Bernoulli; `rng.choice` cat/dog/bird; Bin$(10,0.4)$.

### What he is establishing

A die is both an experiment and a RV ($1\mapsto 1,\ldots$). $Y$ can be even/odd. They print 20 outcomes and an indicator of “six” (and ask whether the die looks biased from 20 draws — ponder it).

**Indicator theorem in samples.** 100,000 rolls; $A=\{X\ge 5\}=\{5,6\}$. The mean of $1_A$ is the estimated $E[1_A]$; the frequency of $A$ is the estimated $P(A)$. They match, as $E[1_A]=P(A)$ demands.

```python
rolls = rng.integers(1, 7, size=100_000)
ind = (rolls >= 5).astype(float)     # 1_A
print(ind.mean(), np.mean(rolls >= 5))  # same number two ways ≈ 1/3
```

**Bernoulli via binomial.** `n=1` binomial **is** Bernoulli. 100,000 tosses with $p=0.7$: estimated mean $\approx 0.7006$ vs $0.7$; estimated variance $\approx 0.209$ vs $p(1-p)=0.21$. Histogram: about 70% ones. **Homework:** rerun with 10, 20, 30 trials and watch the match get worse.

**Categorical.** `rng.choice(["cat","dog","bird"], size=20000, p=[0.2,0.5,0.3])`. Frequencies $\approx 0.20, 0.50, 0.30$. Shrink `size` and watch the wobble.

**Binomial** $n=10$, $p=0.4$, 10,000 draws. Mean $\approx 3.9$ vs $np=4$; var $\approx 2.38$ vs $np(1-p)=2.4$. Poisson is in the notebook; they skip the walkthrough.

```python
# Bernoulli via binomial(n=1)
x = rng.binomial(1, 0.7, size=100_000)
# Categorical
labs = rng.choice(["cat", "dog", "bird"], size=20_000, p=[0.2, 0.5, 0.3])
# Binomial n=10, p=0.4
b = rng.binomial(10, 0.4, size=10_000)
print(b.mean(), b.var())  # ~4 and ~2.4
```

You can now sample the discrete catalog. Still missing: the same overlay for Unif/Normal/Exp and for $Y=g(X)$.

A common trap is expecting $n=10$ samples to match theory to two decimals.

### Analogy for this topic only

Bernoulli is one biased coin. Binomial is **ten** such coins counted together. `choice` is a three-sided spinner labeled cat/dog/bird. More spins, closer to the printed odds.

Question: **How do you get a Bernoulli out of `binomial`?**

In lecture words: this box is discrete sampling.

### Local picture

```
  1_A.mean()  ≈  P(A)           (same number two ways)
  Bern(0.7)   mean 0.70  var 0.21
  Bin(10,0.4) mean 4     var 2.4
```

**Notice:** small $n$ is a **feature** of the homework, not a bug.

### Bridge

Do histograms of **continuous** draws sit on the theoretical $p$?

---

## Topic 10: Continuous simulation and recap (46:03–57:13)

### Where this sits on the master map

**OVERLAY + CLOSE** — Hists vs PDF; $Y=g(X)$; $E[XY]$; next = pairs. Warm-up: [samples](./PREREQUISITES.md#p8-sample).

### Board / screenshot

![Continuous sim](./screenshots/composites/ch10-topic-10-continuous-sim-recap-panel1of2.png)

![Transforms and recap](./screenshots/composites/ch10-topic-10-continuous-sim-recap-panel2of2.png)

**Figure — ~46:03–57:13:** Unif$[-2,3]$ and $N(2,2.25)$ hist vs PDF; empirical CDF sigmoid; $Y=X^2$ and $Y=aX+b$ ($a=-4,b=5$, $EY=-7$); LOTUS; dependent vs independent $E[XY]$; $\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)$.

### What he is establishing

**Uniform** on $[-2,3]$, 100,000 samples: theoretical mean $(−2+3)/2=0.5$, theoretical $\mathrm{Var}=(5)^2/12\approx 2.08$. Histogram looks flat on that interval.

**Normal** — live cell matches the board:

```python
mu, sigma = 2.0, 1.5          # Var = sigma**2 = 2.25
samples = rng.normal(loc=mu, scale=sigma, size=100_000)
x_grid = np.linspace(mu - 5*sigma, mu + 5*sigma, 500)
pdf = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x_grid - mu) / sigma)**2)
print(samples.mean(), mu)
print(samples.var(), sigma**2)
```

Orange curve = that `pdf`; bars = histogram. **Exponential** is the same overlay.

**Empirical CDF** of standard normal (looks like a **sigmoid**):

```python
samples = rng.normal(0, 1, size=5000)
sorted_samples = np.sort(samples)
empirical_cdf = np.arange(1, len(samples) + 1) / len(samples)
plt.plot(sorted_samples, empirical_cdf)
```

**Sample-size ladder** on the board: `sample_sizes = [10, 100, 1_000, 100_000]` for $N(0,1)$. Live print: $n=10$ mean $\approx -0.29$, var $\approx 1.56$; $n=100$ already near $0$ and $1$; $n=10^5$ sits on $(0,1)$.

**Functions.** $Y=X^2$ (nonlinear) vs linear $Y=aX+b$ with **live values** $X\sim N(3,2^2)$, $a=-4$, $b=5$:

```python
x = rng.normal(loc=3, scale=2, size=1_000_000)
a, b = -4, 5
y = a * x + b
print(y.mean(), a * x.mean() + b)   # both near -7
print(y.var(), (a**2) * x.var())
```

$$
E[Y]=a E[X]+b = -4\cdot 3 + 5 = -7
$$

**LOTUS cell:** discrete support with masses — `exact_e_x2 = np.sum(x_values**2 * probabilities)` vs `np.mean(samples**2)` (board: exact $9.7$, sample $\approx 9.71$).

**Dependence.** $Y$ built from $X$: `e_xy = np.mean(X*Y)` vs `np.mean(X)*np.mean(Y)` — they **differ**. Independent pair: product of means matches. Live dependent print: $E[XY]\approx 26$ vs product $\approx 8$.

**Variance properties cell** uses a *second* pair $a=3.5$, $b=-10$: $\mathrm{Var}(Y)$ and $a^2\mathrm{Var}(X)$ both print $\approx 49.04$.

**Recap:** continuous RVs, functions, $E$, Var. **Next:** a **pair** of RVs + numpy. This hour is a **refresher**.

You can now overlay theory on samples and catch dependence in $E[XY]$. Leftover: **two** RVs at once (joints, later).

A common trap is claiming $E[XY]=E[X]E[Y]$ always.

### Analogy for this topic only

A huge crowd’s height histogram becomes the bell you drew on paper. Square everyone’s height ($Y=X^2$) and the picture changes shape. Stretch and shift ($aX+b$) and the center moves by the linearity you already trust. If $Y$ is built from $X$, their product average is **not** the product of averages.

Question: **If $a=-4$, $b=5$, $E[X]=3$, what is $E[aX+b]$?**

In lecture words: this box closes the CRV + simulation recap.

### Local picture

```
  more samples  →  hist sits on p  →  mean/var sit on E, Var

  E[aX+b] = a E[X]+b
  dependent Y=2X+noise :  E[XY] ≠ E[X]E[Y]
  independent          :  E[XY] = E[X]E[Y]

  Next: pairs of RVs
```

**Notice:** simulation **reinforces** formulas; it does not replace CoV or LOTUS.

### Bridge

One $X$ is in place. Joint laws of $(X,Y)$ are the next tutorial.

---

## Apply it (scenarios)

1. **PDF taller than 1.** Write a Unif$[0,0.2]$ PDF. Check area 1.  
2. **Endpoints.** For Exp$(1)$, compute $P(X\le 1)$ and $P(X<1)$.  
3. **CoV refuse.** Why the board theorem does not apply to $Y=X^2$ on $\mathbb{R}$.  
4. **Die high-end.** Change `integers(1,6)` and watch face 6 vanish.  
5. **Sample size.** Bernoulli $p=0.7$ with $n=10$ vs $n=10^5$.  
6. **Dependence.** Recreate $Y=2X+\varepsilon$ and compare $E[XY]$ to $E[X]E[Y]$.

---

## External references

**How to use:** finish the NOTES chain first (video closed if you can). When one map box still feels thin, open **only that topic’s group** — **2–3 companions each** (prefer **teaching video + notes/blog**). All links live **here**, not inside topic bodies.

### Topic 1 — Continuous type / PDF definition

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Khan Academy — PDF](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-continuous/v/probability-density-functions) | Video | $F$ as running area |
| [StatQuest — distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc) | Video | What a distribution is |
| [Seeing Theory — continuous](https://seeingtheory.io/probability-distributions/) | Interactive | Density vs histogram |

### Topic 2 — PDF vs PMF

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Khan Academy — PDF vs PMF](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-continuous/v/probability-density-functions) | Lesson | Height can exceed 1 |
| [Seeing Theory — discrete vs continuous](https://seeingtheory.io/probability-distributions/discrete-discrete/) | Interactive | Piles vs area |
| [3Blue1Brown — binomial setup](https://www.youtube.com/watch?v=8idr1WZ1A7Q) | Video | Mass language before density |

### Topic 3 — Unif / Exp / Gaussian

| Resource | Type | Why it helps |
|----------|------|--------------|
| [StatQuest — Normal](https://www.youtube.com/watch?v=rzFX5NWojp0) | Video | $\mu,\sigma$ as center and width |
| [Khan Academy — exponential](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/poisson-process/v/exponential-pdf) | Lesson | $1-e^{-\lambda x}$ CDF |
| [Seeing Theory — Normal](https://seeingtheory.io/probability-distributions/) | Interactive | Bell overlay |

### Topic 4 — $Y=g(X)$ / CoV

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Stat 110 notes](https://stat110.hsites.harvard.edu/) | Notes | Transformations of RVs |
| [Khan Academy — transforming RVs](https://www.khanacademy.org/math/ap-statistics/random-variables-ap/transforming-random-variables/v/impact-of-transforming-random-variables) | Video | Linear $aX+b$ first |
| [3Blue1Brown — change of variables (essence)](https://www.youtube.com/watch?v=okjYP_Uj-KM) | Video | Why $\lvert dx/dy\rvert$ appears |

### Topic 5 — Expectation / LOTUS

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Stat 110 Lec 9 — $E$ / indicators / LOTUS](https://www.youtube.com/watch?v=LX2q356N2rU) | Video | Same slogans as the board |
| [Blitzstein Stat 110 notes](https://stat110.hsites.harvard.edu/) | Notes | LOTUS proof pointer |
| [Khan Academy — expected value](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/expected-value-lib/v/expected-value) | Lesson | Weighted average |

### Topic 6 — Variance

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Khan Academy — variance](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/variance-standard-deviation-population/v/variance-of-a-population) | Video | Squared deviation |
| [StatQuest — standard deviation](https://www.youtube.com/watch?v=SzZ6GpcfoQY) | Video | $\sigma$ vs $\sigma^2$ |
| [Seeing Theory — variance](https://seeingtheory.io/basic-probability/variance/) | Interactive | Spread picture |

### Topic 7 — Inequalities

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Stat 110 — Markov / Chebyshev](https://projects.iq.harvard.edu/stat110/home) | Notes | Distribution-free tails |
| [Khan Academy — Jensen intuition](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/optimizing-multivariable-functions/v/convex-concave) | Video | Convex smile |
| [3Blue1Brown — Bayes / inequality mindset](https://www.youtube.com/watch?v=HZGCoVF3YvM) | Video | Bounds vs exact laws |

### Topic 8 — Numpy dice / Bayes

| Resource | Type | Why it helps |
|----------|------|--------------|
| [NumPy Generator docs](https://numpy.org/doc/stable/reference/random/generator.html) | Docs | `default_rng`, `integers` |
| [3Blue1Brown — Bayes](https://www.youtube.com/watch?v=HZGCoVF3YvM) | Video | Same $P(D\mid T^+)$ flip |
| [Tutorial 4 NumPy NOTES](../16-Tutorial02-Introduction-to-NumPy/NOTES.md) | Prior unit | Arrays / mean / masks |

### Topic 9 — Discrete sampling

| Resource | Type | Why it helps |
|----------|------|--------------|
| [StatQuest — Binomial](https://www.youtube.com/watch?v=J8jNoF-K8E8) | Video | $n$ independent yes/no |
| [NumPy `binomial` / `choice`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.binomial.html) | Docs | Exact APIs on the board |
| [Khan Academy — binomial](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/binomial-random-variables/v/binomial-distribution) | Lesson | $np$, $np(1-p)$ |

### Topic 10 — Continuous sim / recap

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Seeing Theory — distributions](https://seeingtheory.io/probability-distributions/) | Interactive | Hist vs PDF |
| [NumPy `normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html) | Docs | `loc`, `scale` |
| [Stat 110 Lec 9](https://www.youtube.com/watch?v=LX2q356N2rU) | Video | $E[XY]$ vs independence |

### Whole-map companions

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PREREQUISITES.md](./PREREQUISITES.md) | Warm-up | #p1–#p8 |
| [Tutorial 7 NOTES](../21-Tutorial07-Review-Basic-Probability-1/NOTES.md) | Prior unit | Discrete stack this hour assumes |
| [Stat 110 playlist](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) | Video course | Slower proofs of $E$/Var |

---

## Sources

- Video: [Tutorial 8 : Review of Basic Probability 2](https://www.youtube.com/watch?v=pQIbfyjSnFk)
- Channel: NPTEL — Indian Institute of Science, Bengaluru
- Duration: ~57 min (00:03–57:13)
- Skill: `youtube-lecture-tutor` · mixed (math + numpy)
- 10 topics · 46 claims · coverage receipt
- Previous: [Tutorial 7](../21-Tutorial07-Review-Basic-Probability-1/NOTES.md)
- Next (spoken): pairs of random variables
- Package: `22-Tutorial08-Review-Basic-Probability-2`
