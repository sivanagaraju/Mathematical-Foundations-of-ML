# Prerequisites — warm-up before W1_L4 (Variational divergence minimization)

> **Do this first** if “$p_x$,” “push-forward,” “convex $f$,” or “KL” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues [W1_L2 problem setting](../01-W1-L2-Introduction-Problem-Setting/NOTES.md).  
> **Beginner:** purpose · definition · micro numbers · analogy · ASCII · notice · mini-check.

The YouTube title says **variational divergence minimization (VDM)**. This sitting **names the yardstick** ($f$-divergence: KL, JS, TV). The actual variational **algorithm** (a critic, a conjugate, a minmax) is **next class**. Every idea below unlocks a map word. None of them is the lecture.

```
  After this warm-up you can say:

  "Data are samples from an unknown law p_x, not the law itself."
  "Generative modeling = estimate that law AND learn to draw new points."
  "p_θ is a catalog of candidate laws, usually a deep net (the model)."
  "A divergence scores how far two laws sit; 0 means they match."
  "Push z through G and you get a new random x̂ whose law depends on G."
  "Those outputs are SAMPLES from p_θ, not a printout of the density."
  "f-divergence plugs a convex spring f into an integral of the density ratio."
  "KL is one spring (u log u) and is NOT symmetric."
```

**Symbol card.** Keep this next to NOTES.

| Word | Picture | This sitting’s job |
|------|---------|-------------------|
| $p_x$ | unknown true pile | the law that made the training files |
| $D=\{x_1,\ldots,x_n\}$ | $n$ IID draws | all you see of $p_x$ |
| $p_\theta$ | candidate pile | a parametric family; usually a net |
| $G_\theta$ | the machine $z\mapsto\hat x$ | choosing $G$ **is** choosing $p_\theta$ |
| $D(\cdot\|\cdot)$ | a score of “how far” | he still says “metric”; it need not be a metric |
| $f$ | a convex spring | one $f$ $\Rightarrow$ one named $f$-divergence |
| $D_f$ | $\int p_\theta\,f(p_x/p_\theta)$ | the family this hour installs |
| KL / JS / TV | three named children | $u\log u$, a JS spring, $\tfrac12\|u-1\|$ |

```
  §1  Unknown law vs samples              ──► Topic 1
  §2  Two jobs: estimate AND sample       ──► Topics 1, 3
  §3  Model = parametric family p_θ       ──► Topic 1
  §4  Divergence as a score               ──► Topics 1–2, 5
  §5  Push-forward: z through a function  ──► Topic 2
  §6  Samples are not the density         ──► Topics 2–3
  §7  Convex f with f(1)=0                ──► Topics 4–5
  §8  KL is one spring; not symmetric     ──► Topics 6–8
```

---

## 1. Unknown law vs the pile of files

<a id="p1-px"></a>

### Purpose

The lecture starts from last time: you never hold $p_x$. You hold a **dataset**.

### Definitions

**Law / distribution $p_x$:** the hidden rule for which files show up.  
**Sample / draw:** one file produced by that rule.  
**IID:** each file is an independent draw from the **same** unknown $p_x$.

### Micro numbers

A bag of 60,000 MNIST digits. You do **not** have a formula for “probability of this 28×28 grid.” You have 60,000 grids. Those grids $\sim p_x$ (unknown).

Fair die: law is $P(k)=1/6$ for $k=1,\ldots,6$. Twenty rolls $\{2,5,5,1,\ldots\}$ are samples. The list is not the law. You cannot recover “$1/6$” from twenty rolls with certainty — you only get a histogram.

### Analogy

A bakery’s secret recipe is $p_x$. The display case is the dataset. Tasting every pastry still does not hand you the recipe on paper.

A second picture: a weather station prints one temperature each day. The printouts are samples. The **climate** (the law of temperatures) is $p_x$. Filing 365 printouts is not owning the climate formula.

### Local picture

```
  LAW (hidden)              SAMPLES (what you hold)
  p_x  =  rule              D = {x1, x2, …, xn}
  die: P(k)=1/6             twenty rolls 2,5,5,1,…
  MNIST: unknown density    60,000 grids

  unknown p_x  ──IID draws──►  x1, x2, …, xn
  you never plug p_x into a formula this hour
```

### Notice

He writes $D=\{x_1,\ldots,x_n\}\sim\mathrm{iid}\,p_x$ **(unknown)**. The word unknown is load-bearing: you cannot plug $p_x$ into a formula later.

### Mini-check

1. Is the dataset the same object as $p_x$?  
2. What does IID constrain — pixels inside one image, or images across the pile?  
3. Why does “unknown” make a divergence hard to compute?

---

## 2. Two jobs, not one: estimate and sample

<a id="p2-two-jobs"></a>

### Purpose

A classifier that labels $x$ does **not** emit a brand-new $x$. Generative modeling adds a second verb.

### Definitions

**Estimate $p_x$:** get close to the hidden law (maybe only *implicitly*).  
**Sample:** draw a **new** point that could have come from $p_x$ but was never in $D$.

### Micro numbers

You have 60,000 handwritten 7s. A new 7 that nobody wrote is a **sample**. A network that only says “this is a 7” has estimated a decision boundary, not learned to sample.

| Machine | Estimate? | Sample a new $x$? |
|---------|-----------|-------------------|
| Bayes classifier / “this is a 7” | a boundary | **no** |
| GMM (later weeks) | yes, a density | yes, but does not scale |
| $z\to G_\theta$ (this week) | **implicit** | **yes** — that is the point |

### Analogy

A weather archive vs a weather **generator**. Filing 10 years of temperatures is estimate-ish. Printing **tomorrow’s** plausible temperature that is not a copy-paste of last Tuesday is sampling.

A second picture: a photocopier vs a painter. The copier duplicates files already in $D$. The painter (the generator) must produce a **new** canvas that could have hung in the same gallery.

### Local picture

```
  job 1:  D  ──►  something close to p_x     (may be implicit)
  job 2:         draw x_new ~ that something   (never seen in D)

  classifier:  x ──► label          (no new x)
  generator:   z ──► x_new          (this course)
```

### Notice

He says explicit vs implicit estimate. This hour’s machine will be **implicit**: you get samples out of $G_\theta$, not a printout of $p_\theta(x)$.

### Mini-check

1. Can a Bayes classifier sample a new image?  
2. Which job needs $G_\theta$?  
3. If $p_\theta$ matches $p_x$, what do samples from $G_\theta$ look like?

---

## 3. A model is a parametric family

<a id="p3-model"></a>

### Purpose

Step (i) of his recipe: **assume** a family of candidate laws, indexed by knobs $\theta$. That family **is** the model. Today it is a deep net.

### Definitions

**Parametric family $\{p_\theta\}$:** a catalog of possible laws, one per $\theta$.  
**Model (his word):** $p_\theta$ represented by a neural net.

### Micro numbers

Gaussian family: $p_\theta=\mathcal{N}(\mu,\sigma^2)$, $\theta=(\mu,\sigma)$. Two knobs, infinitely many laws. Set $\mu=0,\sigma=1$: standard normal. Turn $\mu$ to $3$: the bell slides right. A net with a million weights is the same idea with a huge $\theta$.

### Analogy

A radio with a tuning dial. Each dial setting is one station-law. Training is turning the dial until the station sounds like the real broadcast. The radio **is** the model; the dial settings are $\theta$.

A second picture: a pasta catalog. “Spaghetti,” “penne,” “fusilli” are the **family**. Picking a shape is picking $\theta$. You cannot cook “the true Italian pasta” as a platonic object — you pick from the catalog and cook that.

### Local picture

```
  small catalog:   θ = (μ, σ)     p_θ = Normal(μ, σ²)
                   μ=0,σ=1        bell at 0
                   μ=3,σ=1        same bell, slid to 3

  this course:     θ = net weights (millions)
                   p_θ = law imposed by G_θ
```

### Notice

Choosing the **architecture** of $G_\theta$ **is** choosing which laws you can represent. That is one of the four leftover questions.

### Mini-check

1. Is “the model” a trained net or the whole family?  
2. If $\theta$ changes, does $p_\theta$ change?  
3. Why start with a family instead of “the true $p_x$”?

---

## 4. A divergence is a score of how far

<a id="p4-div"></a>

### Purpose

Steps (ii)–(iii): score $p_x$ against $p_\theta$, then **minimize** that score over $\theta$.

### Definitions

A **divergence** $D(p\|q)$ is a number that is supposed to be **large** when $p$ and $q$ disagree and **zero** when they match. He still says “distance metric.” A true **metric** also needs symmetry and the triangle inequality. $f$-divergence will **not** always be a metric (KL is not symmetric).

### Micro numbers

Wanted properties, which $f$-div will have: $D\ge 0$, and $D=0$ **iff** the two laws are equal. Then $\theta^*=\arg\min_\theta D(p_x\|p_\theta)$ is a sensible training problem: driving $D$ to 0 drives $p_\theta$ to $p_x$.

### Analogy

A bathroom scale that only reads **how different two piles of flour are**. Zero means same pile. It does not have to read the same if you swap the two bowls (KL will not). You still use it to bake: turn knobs until the scale hits zero.

### Local picture

```
  D(p_x || p_θ)  ≥  0
  D = 0   ⇔   p_x = p_θ     (the “iff match” property)

  train:  θ* = argmin_θ  D(p_x || p_θ)
```

### Notice

You still cannot **compute** $D$ if you do not know either density. That hole is why a **variational** method will be needed — **next** sitting, not this one.

### Mini-check

1. If $D$ can be negative, why is $\arg\min$ a bad idea?  
2. Does “metric” on the tablet mean triangle inequality?  
3. Why is $D=0$ iff match the property that makes sampling work after training?

---

## 5. Push-forward: a known $z$ through a function

<a id="p5-push"></a>

### Purpose

His running **example** of a generative model: start from a random variable you **can** sample, push it through a deterministic map.

### Definitions

**Push-forward:** if $Z$ is random and $g$ is a fixed function, $\hat X=g(Z)$ is a **new** random variable. Its law depends on $g$.

Typical $Z\sim\mathcal{N}(0,I)$ (easy to sample). $g=G_\theta$ a neural net. $G_\theta:Z\to\mathcal{X}$.

### Micro numbers

Fair die $Z\in\{1,2,3,4,5,6\}$. Map $g_{\text{bit}}(z)=1$ if $z$ odd, else $0$: then $P(\hat X=1)=3/6$. Map $g_{\text{hi}}(z)=1$ if $z\ge 5$: then $P(\hat X=1)=2/6$. Same easy $Z$, **different $g$**, different output law. You never rewrote a density on paper.

### Analogy

A pasta extruder. Flour (easy random $z$) goes in. The **die** $G$ shapes it. Change the die, change the pasta-law. You never wrote a formula for “probability of this noodle.” You designed the die.

A second picture: a rubber stamp. The ink pad is a simple random blot ($z$). The stamp’s **shape** is $G$. Press, get a letter. Change the stamp, change which letters appear.

### Local picture

```
  easy Z you CAN sample          known g              new RV
  die {1,2,3,4,5,6}     -->   odd? 1 else 0   -->   fair bit
  same die              -->   ≥5 ? 1 else 0   -->   P(1)=2/6

  this course:
  z ~ N(0, I)   -->   G_θ(z)   -->   x̂ ~ p_θ
                         └── the law of x̂ DEPENDS on G_θ
```

### Notice

Board: $\theta^*=\arg\min_\theta D(p_x\|p_\theta)$. After that, $p_{\theta^*}$ is close to $p_x$, so $z$ through $G_{\theta^*}$ **samples** from near $p_x$.

### Mini-check

1. If $G$ is the identity, what is the law of $\hat x$?  
2. Who chooses $p_\theta$ — the noise $z$, or the net?  
3. Why pick Gaussian $z$?

---

## 6. Samples out ≠ density on paper

<a id="p6-samples"></a>

### Purpose

He repeats: the net’s output is **samples from** $p_\theta$, **not** $p_\theta$ itself. Same wedge on the data side: $D$ is samples from $p_x$, not $p_x$.

### Definitions

**Implicit model:** you can **draw** $\hat x$, you cannot necessarily **evaluate** $p_\theta(\hat x)$ as a number.

### Micro numbers

Forward pass: draw 128 Gaussians, run $G_\theta$, get 128 fake images. You now have a **cloud**. You still do not have $p_\theta(x)$ at a pixel grid.

### Analogy

A camera vs a lighting equation. The camera **emits** photos (samples). It does not print the physics formula that made the light field (the density). MNIST is a camera dump of $p_x$. $G_\theta$ is a camera dump of $p_\theta$.

### Local picture

```
  have:   cloud from p_x     (training files)
          cloud from p_θ     (z through G)
  lack:   the two density FUNCTIONS
```

### Notice

This is why “just write $\int p_\theta f(p_x/p_\theta)\,dx$” is not yet an algorithm. Four questions on the tablet include **how to compute $D$ from clouds alone**.

### Mini-check

1. After a forward pass, do you hold $p_\theta$ or samples?  
2. Why is that a problem for the integral?  
3. Which of the four leftover questions is this?

---

## 7. A convex spring $f$ with $f(1)=0$

<a id="p7-f"></a>

### Purpose

$f$-divergence is one integral, many children. The child is chosen by a function $f$ on **positive reals**.

### Definitions

Board:

$$
D_f(p_x\|p_\theta)=\int_{\mathcal{X}} p_\theta(x)\,f\!\left(\frac{p_x(x)}{p_\theta(x)}\right)\,dx.
$$

$f:\mathbb{R}^+\to\mathbb{R}$, **convex**, **left semi-continuous**, $f(1)=0$.

**Why the ratio is a scalar:** even if $x\in\mathbb{R}^d$, a density at a point is one positive number. The ratio of two densities is one positive number. $f$ of that number is well-defined.

**Why $f(1)=0$:** when $p_x=p_\theta$, the ratio is $1$, and $f(1)=0$ makes the integrand vanish.

**Convex:** the shape that will later (not today) let a conjugate / variational bound exist. Today he only **lists** convex.

### Micro numbers

$u=p_x/p_\theta$. If the two laws match at $x$, $u=1$, $f(1)=0$, that location contributes $0$. If they mismatch, $u\neq 1$ and a convex $f$ with $f(1)=0$ sits **above** $0$, so the integral stays $\ge 0$.

Toy spring $f(u)=u\log u$ (natural log): $f(1)=0$. At $u=e\approx 2.7$, $f(e)=e$ (about $2.7$). TV is easier to picture: $f(u)=\tfrac12|u-1|$ is a V sitting at $u=1$, never negative. The **integral** $D_f$ is still $\ge 0$ for every legal $f$ (property he states, not proved today).

### Analogy

$f$ is a **spring** sitting at rest at $u=1$ (matched densities). Stretch $u$ away from $1$ and the spring stores energy. Different spring recipes (KL, JS, TV) store energy differently. The integral adds up the energy over the whole room $\mathcal{X}$.

A second picture: a speed-limit sign that reads **the ratio of two speedometers**, not the car’s GPS coordinates. $x$ may be a 784-vector; $f$ still eats **one** positive number.

### Local picture

```
  x may be a VECTOR in R^d
  p_x(x) and p_θ(x) are SCALARS (heights)
  u = height_real / height_fake     (one positive number)
  pay at x:   p_θ(x) · f(u)
  total:      ∫ pay   = D_f
  rest:       f(1)=0   (matched heights → pay 0)
```

### Notice

He assumes **continuous** RVs with **densities**. Discrete / no-density versions exist; he skips them because images are treated as continuous here.

### Mini-check

1. What is $u$ if the two densities match?  
2. Why is $f$ applied to a ratio, not to the vector $x$?  
3. Name the three conditions on $f$.

---

## 8. KL is one spring — and it is not symmetric

<a id="p8-kl"></a>

### Purpose

One choice $f(u)=u\log u$ **is** Kullback–Leibler (KL). Swapping the two arguments is a **different** number: **forward** vs **reverse** KL. That is why a **family** of $f$ is worth having.

### Definitions

**Forward KL:** $D_{\mathrm{KL}}(p_x\|p_\theta)=\int p_x\log(p_x/p_\theta)$.  
**Reverse KL:** $D_{\mathrm{KL}}(p_\theta\|p_x)$ — **not** equal.  
Board JS spring: $f(u)=\tfrac12\big(u\log u-(u+1)\log\frac{u+1}{2}\big)$.  
Board TV: $f(u)=\tfrac12|u-1|$.

### Micro numbers

Algebra he writes: plug $f(u)=u\log u$ into $D_f$:

$$
\int p_\theta\cdot\frac{p_x}{p_\theta}\log\frac{p_x}{p_\theta}\,dx=\int p_x\log\frac{p_x}{p_\theta}\,dx=D_{\mathrm{KL}}(p_x\|p_\theta).
$$

The $p_\theta$ cancels. That is the whole derivation.

**Two-bin toy (why swap matters).** Let $p=(0.9,0.1)$ and $q=(0.5,0.5)$. Forward $\sum p_i\log(p_i/q_i)$ is **not** reverse $\sum q_i\log(q_i/p_i)$. One direction yells when $p$ puts mass where $q$ is small; the other yells when $q$ puts mass where $p$ is small. He does **not** grind these numbers; he only writes the inequality of the two orientations.

### Analogy

Driving A→B vs B→A on a one-way street. The odometer (KL) reads different numbers. JS is a compromise road that looks at both directions. TV is a cruder “how much mass sits on mismatched addresses.”

A second picture: a one-way toll. Paying to enter the city is not paying to leave. If your training score secretly depends on which pile you call “true,” one KL is not enough — hence a **family** of $f$.

### Local picture

```
  f(u)=u log u     →  KL (forward when ordered p_x || p_θ)
  swap arguments   →  reverse KL   (NOT the same number)

  p=(0.9, 0.1)   q=(0.5, 0.5)
  D_KL(p||q)  ≠  D_KL(q||p)

  another f        →  JS   (board spring; GAN uses a VERSION later)
  another f        →  TV = ½|u−1|
```

### Notice

ASR said a garbled JS formula. **Trust the tablet:** $\tfrac12\big(u\log u-(u+1)\log\frac{u+1}{2}\big)$. He says a version of JS is what GANs use — **later**.

### Mini-check

1. Does $D_{\mathrm{KL}}(p\|q)=D_{\mathrm{KL}}(q\|p)$?  
2. Which $f$ recovers KL?  
3. Did this sitting train a GAN?

---

**Second teachers (names only here).** Nowozin et al. $f$-GAN / VDM (the paper that *does* the variational bound next sitting). Kullback–Leibler. Jensen–Shannon. Pointers (video + notes, 2–3 per topic) live at the end of [NOTES.md](./NOTES.md#external-references).

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
