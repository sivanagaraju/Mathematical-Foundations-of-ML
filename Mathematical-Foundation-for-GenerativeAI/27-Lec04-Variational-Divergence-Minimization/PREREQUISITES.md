# Prerequisites — warm-up before Lec 04 (Variational Divergence Minimization)

> **Do this first** if “expectation,” “conjugate,” “supremum,” or “saddle” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Complements [Lec 03](../25-Lec03-f-Divergence-Examples/NOTES.md) and [Tutorial 11](../26-Tutorial11-f-Divergence-Examples/NOTES.md).  
> **Beginner:** purpose · definition · micro numbers · analogy · ASCII · notice · mini-check.

This hour never gives you two closed-form densities. It only ever gives you **two piles of points** and asks you to **estimate a number** (an $f$-divergence) from those piles. Every idea below is a word you will meet on the master map. None of them is the lecture itself.

```
  After this warm-up you can say:

  "We have two piles of points, not two closed-form densities."
  "An integral of (function × density) is an expectation."
  "A sample average is an estimate of that expectation, not the same object."
  "LOTUS lets us average h on the original x’s, not on a new law of h(X)."
  "A convex cup never sits above its chords."
  "The conjugate of f peels the argument out of f."
  "A supremum is the least upper bound — it may not be attained."
  "A number t at every x is a function T(x)."
  "Min one set of weights and max another on the same score is a saddle."
```

```
  §1  Two sample clouds            ──► Topics 1, 8, 10
  §2  Expectation as ∫ h p         ──► Topics 2–3
  §3  LLN: sample average          ──► Topics 2–3
  §4  LOTUS                        ──► Topic 3
  §5  Convex cup                   ──► Topics 5–6
  §6  Fenchel conjugate            ──► Topics 5–6
  §7  Sup vs max; function T(x)    ──► Topics 7–8
  §8  Minmax / saddle              ──► Topic 10
```

**One scene through all eight.** Keep these two bags in your head. Every later idea is a job you can do *with the bags*, not with a factory recipe.

```
  BAG A  (album)     10 red marbles you inherited     = data D ~ p_x
  BAG B  (prints)    10 marbles a machine stamped
                     from noise z through a fixed die  = G_θ(Z) ~ p_θ

  §1  you have the bags, not the mixing recipe
  §2  a city-wide average is ∫ (measure × how-often)
  §3  polling n marbles estimates that average (LLN)
  §4  you score each marble where it sits (LOTUS) —
      you do not run a new experiment of scores
  §5  the scoring spring f is a cup
  §6  conjugate = a key that peels the locked ratio out of f
  §7  best score at each marble is a map T, not one number t
  §8  critic inflates one number J; forger deflates it (saddle)

  This hour has no Python. Next class writes the loop.
```

---

## 1. Two piles of points, not two formulas

<a id="p1-clouds"></a>

### Purpose

Last lectures wrote

$$
D_f(p_x\|p_\theta)=\int p_\theta(x)\,f\Bigl(\frac{p_x(x)}{p_\theta(x)}\Bigr)\,dx.
$$

You cannot type that integral into a computer if you do not have $p_x$ or $p_\theta$ as formulas. This lecture lives in the world of **files**, not formulas.

### Definitions

**Cloud of data.** $n$ **IID** (independent and identically distributed) draws $x_1,\ldots,x_n$ from the unknown law $p_x$. That *is* the dataset. “IID” means: same recipe for every draw, and one draw does not tell you the next.

**Cloud of fakes.** Draw $Z\sim\mathcal{N}(0,I)$ (a Gaussian vector you *do* know how to sample), push it through a **deterministic** net $G_\theta$, get $\hat x=G_\theta(Z)$. Those $\hat x$ are draws from $p_\theta$.

The net does not “roll dice.” All the chance lives in $Z$. Same $Z$ always gives the same $\hat x$.

### Micro numbers

Ten photos of cats on disk = cloud $p_x$. Ten `randn` vectors of length $2$ through a tiny map $G_\theta$ = cloud $p_\theta$.

```
  data cloud (n = 3 tiny “images” in R^2)
    x1 = (1.0, 0.2)
    x2 = (0.9, 0.3)
    x3 = (1.1, 0.1)

  fake cloud (same size, different pile)
    z1 = randn → G_θ → x̂1 = (0.4, 1.2)
    z2 = randn → G_θ → x̂2 = (0.5, 1.0)
    z3 = randn → G_θ → x̂3 = (0.3, 1.1)
```

Neither pile is a PDF you can plot in closed form. You still have **two files**.

The tablet draws $G_\theta$ as a **trapezoid**: typically $\dim(Z)<\dim(\text{data})$, so a short vertical edge on the noise side and a long edge on the $\hat x$ side. The net does not invent chance. Packages’ `randn` (or a **Box–Muller** map from Uniform) give you $Z$; same $Z$ through a frozen $G_\theta$ always prints the same fake.

### Analogy

Two bags of marbles. You never get the factory’s mixing recipe. You only get the bags. Divergence estimation this hour = compare the bags, not the recipes. If you wait for the recipe, the hour never starts.

### Local picture

```
  WANT:   number  D_f(p_x || p_θ)
  LACK:   formula for p_x, formula for p_θ

  HAVE:
    bag D        = {x1,...,xn}     samples of p_x
    bag G_θ(Z)   = {x̂1,...,x̂m}    samples of p_θ

  z ~ N(0,I)  --deterministic-->  G_θ  -->  x̂
                  (same z, same x̂)
```

### Notice

In **most** of this course the net **is** the sampler (its output *is* the fake). **Diffusion** is the named exception: that net estimates something else, not the sample itself.

### Mini-check

1. Where does randomness live — $G_\theta$ or $Z$?  
2. What are samples of $p_\theta$ on the board?  
3. If $G_\theta$ is frozen, how do you get a *new* fake?

---

## 2. An expectation is an integral of (function × density)

<a id="p2-expect"></a>

### Purpose

Every $f$-divergence is an integral against a density. The lecture’s move is: **name that integral an expectation**, then estimate the expectation from the bags.

### Definitions

$$
\mathbb{E}_{p}[h(X)]=\int h(x)\,p(x)\,dx.
$$

In words: walk every location $x$, measure $h(x)$, weight by how often the law $p$ visits $x$, add it up. If $h\equiv 1$, you get $1$. If $h$ is “how the spring $f$ reads,” the integral is a number you might try to drive down.

You must always name **which law** the average is under. $\mathbb{E}_{p_x}[T]$ and $\mathbb{E}_{p_\theta}[f^*(T)]$ are two different city-wide averages.

### Micro numbers

Fair six-sided die, $h(\text{face})=\text{face}$.

$$
\mathbb{E}[X]=\frac{1+2+3+4+5+6}{6}=3.5.
$$

Same idea as $\int$, but a sum. Height at each face is $1/6$; $h$ is the number on the face.

A second function on the same die: $h(x)=x^2$. Then $\mathbb{E}[h]=(1+4+9+16+25+36)/6=15.166\ldots$. The law did not change. The *thing you measure* did.

### Analogy

$p$ is how often you visit each street. $h$ is what you measure there (temperature, rent, a critic’s score). The expectation is the city-wide average. Two cities (data-town vs fake-town) give two averages even if you use the same thermometer $h$.

### Local picture

```
  streets x:     1     2     3     4     5     6
  visit-rate p:  1/6   1/6   1/6   1/6   1/6   1/6
  measure h:     1     2     3     4     5     6

  E_p[h] = 1*(1/6)+...+6*(1/6) = 3.5

  later in VDM there will be TWO cities:
    data-town   →  E_{p_x}[ T(x) ]
    fake-town   →  E_{p_θ}[ f*(T(x)) ]
```

### Notice

You still need to know **which law** the average is under. Average of $T(x)$ under **data** is not the average of $f^*(T(x))$ under **fakes**. Mixing the two piles is the wrong city.

Not every divergence is already an $\mathbb{E}$ you can poll. **Some** scores *are* exact expectations — invoke LLN directly. **Others cannot** — bound first, then poll the bound. That fork is the whole hour. $f$-divergence takes the bound path.

### Mini-check

1. Write $\mathbb{E}_{p_\theta}[f^*(T(X))]$ as an integral.  
2. Which cloud estimates that integral?  
3. If $h\equiv 1$, what must $\mathbb{E}_p[h]$ equal?

---

## 3. Law of large numbers: a sample average estimates an expectation

<a id="p3-lln"></a>

### Purpose

Once something is an expectation under a law you can sample, **IID sample averages** stand in for the integral. That is why “data is oil” in this lecture: large $n$ makes the stand-in honest.

### Definitions

If $x_1,\ldots,x_n$ are IID from $p$, then

$$
\frac1n\sum_{i=1}^n h(x_i)\;\longrightarrow\;\mathbb{E}_p[h(X)]
$$

as $n\to\infty$ (under standard conditions he does not spell out). The left side is a **sample average** — a *random* number that depends on which $x_i$ you drew. The right side is a **fixed** number. A statistician does **not** treat them as the same object.

The **law of large numbers (LLN)** is the statement that the left side approaches the right side when $n$ is huge *and* the $x_i$ came from the **same** law as the $\mathbb{E}$.

### Micro numbers

Die, $h=\text{face}$, true mean $3.5$. Ten rolls: $2,6,1,5,3,4,6,2,1,3$. Average $=3.3$. That is an **estimate** of $3.5$, not a proof that the mean is $3.3$. A thousand rolls will typically sit closer to $3.5$. A decade ago he would have called $10{,}000$ samples “large.” Models today train on **trillions**, so the asymptotic story is no longer a fairy tale.

### Analogy

Poll $n$ voters instead of counting a whole country. The poll is useful when $n$ is huge **and** you polled the country you claim to describe. Poll a hospital’s X-ray ward and report it as “all of medicine” and LLN’s ticket is void — that is Topic 4’s shift example, not a theorem today.

### Local picture

```
  TRUE (fixed):     E[h] = 3.5

  n=10 rolls:       average = 3.3     ← estimate, random
  n=1000 rolls:     average ≈ 3.49    ← still an estimate
  n→∞:              average → 3.5     ← LLN

  sample average  ≠  expectation
  (look the same on paper; behave differently)

  TICKET: draws must be IID from the SAME law as the E
```

### Notice

LLN needs **IID** draws from the **same** law the expectation uses. Shift the hospital’s scanner (non-IID) and the estimate can break. That is a Q&A in Topic 4, not a solved theorem today.

### Mini-check

1. Why is a sample average not “the same thing” as $\mathbb{E}[h]$?  
2. What assumption does LLN need that real hospitals often break?  
3. If you average $T$ on **fakes** and claim you estimated $\mathbb{E}_{p_x}[T]$, what did you break?

---

## 4. LOTUS: average $h$ on the original $x$’s

<a id="p4-lotus"></a>

### Purpose

Homework he assigns: why

$$
\mathbb{E}[h(X)]=\int h(x)\,p_x(x)\,dx
$$

and **not** $\int h\cdot p_{h(X)}$? If it were the second, you would need samples of $h(X)$, a new law. Because it is the first, **data $x_i$** are enough. That is the **law of the unconscious statistician (LOTUS)**.

### Definitions

**LOTUS:** you may average $h$ using the law of $X$, even though $h(X)$ has its own law. You do not have to build a new random experiment whose outcomes are the $h$-values.

Naive freeze: “$\mathbb{E}[X]=\int x\,p_x$, so $\mathbb{E}[h(X)]$ must use the density of $h(X)$.” That sounds careful and is the wrong extra work. The integral stays against $p_x$.

### Micro numbers

$X$ uniform on $\{1,2,3\}$, so $p_x(1)=p_x(2)=p_x(3)=1/3$. Let $h(x)=x^2$. Then $h(X)$ takes values $1,4,9$.

LOTUS way (walk the original $x$’s):

$$
\mathbb{E}[h(X)]=\frac{1^2+2^2+3^2}{3}=\frac{14}{3}.
$$

You never built the law of the squares first. If LOTUS failed, you would need a new sample of *squares*, not the original list $\{1,2,3\}$.

### Analogy

You have a list of cities (the $x$’s). $h$ is “temperature at that city.” You do not need a histogram of temperatures as a new random experiment — walk the city list and average the temperatures. The city list *is* the data file.

### Local picture

```
  x list:     1      2      3
  h(x)=x^2:   1      4      9

  LOTUS:  E[h] = (1+4+9)/3 = 14/3
          uses the x-list, not a new experiment of squares

  later:
    E_{p_x}[ T(X) ]        ← average T on DATA x_i
    E_{p_θ}[ f*(T(X)) ]    ← average f*(T) on FAKE x̂_j
```

### Notice

This is why the later bound can use **data $x$** for $\mathbb{E}_{p_x}[T(X)]$ and **fakes $G_\theta(Z)$** for $\mathbb{E}_{p_\theta}[f^*(T(X))]$. Two lists, two averages, no extra experiment.

### Mini-check

1. If LOTUS failed, which extra samples would you need?  
2. Which expectation in VDM uses the dataset, not the generator?  
3. For $X\in\{1,2,3\}$ uniform and $h(x)=2x$, what is $\mathbb{E}[h(X)]$ by LOTUS?

---

## 5. A convex cup

<a id="p5-convex"></a>

### Purpose

$f$ in $f$-divergence is **convex**. The conjugate in the next idea is defined using that cup. If $f$ were a hill (concave), the unzip would not be the same tool.

### Definitions

$f$ is convex if it never sits **above** a chord: for every $u,v$ and every $\alpha\in[0,1]$,

$$
f(\alpha u+(1-\alpha)v)\le \alpha f(u)+(1-\alpha)f(v).
$$

In words: the graph is a **cup**. Mix two inputs, evaluate $f$ — you sit **below** (or on) the straight line joining the two outputs.

A straight line is convex (the chord *is* the graph). A V-shape $|u-1|$ is convex but not strictly curved. $u\log u$ on $(0,\infty)$ is a smooth cup.

### Micro numbers

Take $f(u)=u^2$ on $\mathbb{R}$. Points $u=0$ and $u=2$. Midpoint $u=1$.

- $f(1)=1$.  
- Chord height at the midpoint: $\tfrac12 f(0)+\tfrac12 f(2)=0+2=2$.  
- $1\le 2$, so the cup sits **below** the chord. Legal.

A hill $f(u)=-u^2$ would reverse the inequality. That is **not** the $f$ of $f$-divergence.

### Analogy

A hanging chain (sags) vs a taut string (flat) vs a hill (bulges up). Cups sit below their chords. The lecture needs a cup so a **conjugate** exists and $(f^*)^*=f$ (homework).

### Local picture

```
        f(u)
          |           *  (u=2, f=4)
          |         /
          |       *      ← midpoint of the GRAPH is up here (height 2)
          |     /   *
          |   /       *  ← actual f(1)=1 sits BELOW the chord
          | /
    ------+------------→ u
          0     1     2

  cup never sits ABOVE its chords
```

### Notice

He will not reprove convexity. Last hour already required convex $f$ with $f(1)=0$. The recap tablet also says **left-semi-continuous** (values do not jump *down* as you approach a point). He needs the cup so the conjugate in §6 is legal; he does not spend the hour on that extra adjective.

### Mini-check

1. Is a straight line convex?  
2. Why does $f$-divergence need a convex $f$?  
3. At the midpoint of $0$ and $2$ for $f(u)=u^2$, which is larger — $f(1)$ or the chord?

---

## 6. Fenchel conjugate: peel $u$ out of $f(u)$

<a id="p6-conjugate"></a>

### Purpose

$D_f$ has $f(p_x/p_\theta)$ — the **ratio is stuck inside $f$**. You cannot sample a ratio of unknown densities. The **Fenchel conjugate** (also called the convex conjugate) **unzips** $u$ from $f$.

### Definitions

If $f$ is convex,

$$
f^*(t)=\sup_{u\in\mathrm{dom}(f)}\bigl(ut-f(u)\bigr).
$$

Board geometry: $ut-f(u)$ is a **family of lines**. **$t$ is the slope**; $u$ varies along the cup. The conjugate at a given $t$ is the **highest** of those lines (a supremum — the peak may not be attained).

**Homework properties** (he assigns; do not skip): $f^*$ is itself convex; **conjugate of the conjugate recovers $f$**:

$$
f(u)=\sup_{t}\bigl(tu-f^*(t)\bigr).
$$

That second line is the unzip: $f(u)$ becomes “best probe $tu$ minus a tax $f^*(t)$.” Then $u$ sits **outside** $f$, as a multiplier you can cancel against $p_\theta$.

### Micro numbers

A textbook friend (not required on his board): $f(u)=u^2/2$ has $f^*(t)=t^2/2$. Check the unzip at $u=3$:

$$
\sup_t\bigl(3t-t^2/2\bigr).
$$

Best $t$ is $t=3$; the value is $9/2=f(3)$. Same pattern at $u=1$: best $t=1$, value $1/2$. The lecture does **not** need this table; it needs the unzip: $u$ sits outside as a multiplier. Dummy $u$ is later $p_x/p_\theta$. Dummy $t$ is later the **output of a net** $T(x)$.

Treating $\mathbb{E}_{p_\theta}[f(\text{ratio})]$ as a finished rewrite is the wrong move: the ratio is still locked inside $f$, and you cannot sample a ratio of two unknown densities. The conjugate exists so $u$ can sit *outside* as a multiplier.

```
  u (locked ratio)    best t    unzipped value = f(u)
  1                   1         0.5
  3                   3         4.5
  0                   0         0
```

### Analogy

$f$ is a locked box with the ratio inside. You cannot sample the contents. The conjugate is a key that lets you write $f(u)$ as “best linear probe $tu$ minus a tax $f^*(t)$.” Then $u$ is a multiplier sitting *outside* the lock. Treating $\mathbb{E}_{p_\theta}[f(\text{ratio})]$ as a finished rewrite is putting the still-locked box on the table and calling it opened.

### Local picture

```
  locked:   f( p_x / p_θ )     ratio stuck inside f

  key:      f*(t) = sup_u ( u t − f(u) )
            t = slope,  u varies,  pick the HIGHEST line

  unzip:    f(u) = sup_t ( t u − f*(t) )
            now u is a multiplier, not an input hiding in f

  homework: (i) f* is convex
            (ii) (f*)* = f
```

### Notice

Dummy $u$ is the argument of $f$ (later $p_x/p_\theta$). Dummy $t$ is the argument of $f^*$ (later $T(x)$). He said **Fenchel**. He did not name extra hypotheses such as Fenchel–Moreau; biconjugate $=f$ is homework as assigned.

### Mini-check

1. In $ut-f(u)$, which symbol is the slope on his picture?  
2. What does $(f^*)^*=f$ let you substitute for $f(p_x/p_\theta)$?  
3. Why is $\mathbb{E}_{p_\theta}[f(p_x/p_\theta)]$ not yet a usable rewrite?

---

## 7. Supremum, and a number at every $x$ is a function

<a id="p7-sup-T"></a>

### Purpose

The algebra puts $\sup_t$ **inside** an integral over $x$. A single number $t$ cannot be the inner winner at **every** $x$. You need a **map** $T(x)$. If your bag of allowed maps is short, you only **lower-bound** $D_f$.

### Definitions

**Supremum** = least upper bound. It may **not be attained** (no maximizer sitting on the peak). That is why he says supremum, not “max,” even when he writes max on the board for intuition.

**Function $T$.** For each location $x$, pick a number in the **domain of $f^*$**. That list of picks *is* $T:\mathcal{X}\to\mathrm{dom}(f^*)$. Capital $T$ is the map; small $t$ was the dummy number.

If your bag of allowed $T$’s, written $\mathcal{T}$, **misses** the ideal $T^*$, the pulled-out expression is only a **lower bound** on $D_f$, not $D_f$ itself. A restricted supremum can only **undershoot**.

The last layer of the $T$-network must land in $\mathrm{dom}(f^*)$ — a legal output for $f^*$. That is not decoration.

### Micro numbers

At $x=0$ the best $t$ might be $2$; at $x=1$ the best $t$ might be $-1$. One global $t$ cannot do both.

```
  x          best inner t
  0          2
  1         -1
  2          0.5

  one number t=2  →  wins at x=0, loses at x=1
  a function T    →  T(0)=2, T(1)=-1, T(2)=0.5   (can win everywhere)
```

If your bag $\mathcal{T}$ only contains *constant* functions, you miss $T^*$ and you **lower-bound** the integral.

### Analogy

A restaurant wants the best wine **at each table**. One bottle for the whole dining room is a number $t$. A sommelier who walks table-to-table is $T(x)$. If your sommelier list is short, you only **lower-bound** “best possible dining.” You do not get to print “we served the ideal dinner.”

### Local picture

```
  inside the integral:   sup_t  { t · ratio(x) − f*(t) }
                         └── a NUMBER t, but the winner depends on x

  cannot pull ONE t out of ∫ dx

  T : X → dom(f*)     (a number at every x)

  bag 𝒯 may miss T*  →  LOWER bound, not equality

  last activation of the T-net must land in dom(f*)
```

### Notice

Direction of the inequality is a **lower** bound because a smaller bag can only make the inner sup *smaller*. An **upper** bound would have been the nicer object for a *minimization* of $D_f$. That trap is Topic 8.

### Mini-check

1. Why can’t you pull a scalar $\sup_t$ out of $\int dx$ unchanged?  
2. Why a **lower** bound, not an upper bound?  
3. What must the last activation of the $T$-net respect?

---

## 8. Min one player, max the other: a saddle

<a id="p8-saddle"></a>

### Purpose

After $T$ is itself a net $T_w$, the bound is a score $J$ that you **maximize in $w$** (tighter bound) and **minimize in $\theta$** (better sampler). Same formula, two directions. That shared critical point is a **saddle**. Ordinary training **avoids** saddles. This problem **seeks** them.

### Definitions

$$
\min_\theta\max_w\Bigl(\mathbb{E}_{p_x}[T_w(X)]-\mathbb{E}_{p_\theta}[f^*(T_w(X))]\Bigr).
$$

Call the inside $J(\theta,w)$. A **saddle** of a surface $J$ is a point where the two axes disagree: one direction wants to go up, the other down. He draws it as: move in the $w$ direction and $J$ **decreases** (you were maximizing over $w$); move in the $\theta$ direction and $J$ **increases** (you were minimizing over $\theta$).

$T_w$ is just an **approximator of $T$**. Board labels: **critic** / **discriminator**. Both mean that approximator. Not a Hollywood character.

### Micro numbers (shape, not a real train)

Imagine a toy surface $J(\theta,w)=w^2-\theta^2$ near $(0,0)$. Along $w$, it is a cup (minimum at $0$ if you *minimized* $w$ — but we *maximize* $w$, so we climb the flaps). Along $\theta$ it is a hill we want to sit at the bottom of. The interesting point is the middle of a horse saddle, not a bowl.

You do not need this formula in the lecture. You need the picture: **two knobs, opposite jobs, one number.**

### Analogy

A horse saddle: sit in the middle; left–right you go **up** the flaps; front–back you go **down** the horse. Two axes, opposite curvature. $T_w$ is a critic you train to **inflate** the score (expose fakes). $G_\theta$ is a forger you train to **deflate** the same score (fool the critic). They share one number. He does **not** want the “terrorist vs counter-terrorist” story — the score is the variational bound, not a morality play.

### Local picture

```
              max over w (critic T_w)
                    ▲
                    │  flaps go UP
                    │
   min over θ  ─────┼─────  horse goes DOWN
   (sampler G_θ)    │
                    │
                    ▼

  J(θ, w) = E_data[ T_w ] − E_fakes[ f*(T_w) ]

  typical opt AVOIDS saddles
  this problem SEEKS a saddle   →  hard to train

  STOP today: no training loop, no table of f*
```

### Notice

Implementation (how gradients flow, which $f^*$ to pick, freeze one net / train the other) is **next lecture**. Today ends at naming the saddle. Homework: brush **backpropagation** (backprop) — first-order gradient descent through a net.

### Mini-check

1. Which parameters does the inner $\max$ train?  
2. Why is “we seek saddles on purpose” a warning?  
3. What does “discriminator” mean on this board?

---

**Second teachers (names only here).** Original tablet: this lecture’s Drive notes. Original paper that *named* VDM: Nowozin, Cseke, Tomioka, $f$-GAN. Original variational characterization: Nguyen, Wainwright, Jordan 2010. University hours: Stanford CS236, Berkeley CS294-158 (2024), Cornell CS 6785. Conjugate picture: Boyd EE364A. LOTUS: Harvard Stat 110. Shortest path: a calculus-of-variations classroom video. Pointers (2–3 per topic) live at the end of [NOTES.md](./NOTES.md#external-references) — not under the warm-ups.

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
