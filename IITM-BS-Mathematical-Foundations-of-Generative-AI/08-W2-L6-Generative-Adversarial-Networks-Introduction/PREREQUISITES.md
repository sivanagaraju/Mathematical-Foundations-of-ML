# Prerequisites — warm-up before W2_L6 (GAN introduction)

> **Do this first** if “domain of f*,” “sigmoid,” “Jensen–Shannon,” or “composition” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues [W1_T3 two-net saddle](../05-W1-T3-PyTorch-Datasets-DataLoaders/NOTES.md).  
> **Beginner:** purpose · definition · micro numbers · analogy · ASCII · notice · mini-check.

YouTube says **GAN introduction**. This sitting is **chalk**: pick the usual GAN $f$, build the last activation so $T$ lands in $\mathrm{dom}(f^*)$, rearrange to the two-log loss with a **sigmoid** $D$. **No Python.** Every idea unlocks a map word.

```
  After this warm-up you can say:

  "T must spit numbers the conjugate is allowed to eat (dom f*)."
  "Split T into a shared net V (last layer linear) plus an f-specific activation σ_f."
  "GAN is one choice of f inside last hour’s min_θ max_w — not a new religion."
  "That f looks like Jensen–Shannon, with a constant’s difference."
  "f* lives on negative reals; σ_f is built to stay negative."
  "After algebra, the object in the loss is D = sigmoid(V) in (0,1)."
  "(0,1) is not R_−. D is not T. T is a deterministic function of D."
```

**Symbol card**

| Word | Picture | This sitting |
|------|---------|--------------|
| $T$ / $T_w$ | last hour’s judge | must land in $\mathrm{dom}(f^*)$ |
| $V_w$ | net ending in a linear scalar | common across all $f$ |
| $\sigma_f$ | last activation | chosen so range $=\mathrm{dom}(f^*)$ |
| $f$ | convex spring | GAN: $u\log u-(u+1)\log(u+1)$ |
| $f^*$ | conjugate | GAN: $-\log(1-e^t)$, domain $\mathbb{R}_-$ |
| $D_w$ | sigmoid of $V_w$ | output in $(0,1)$; **not** $T$ |
| $J_{\mathrm{GAN}}$ | two logs | $\mathbb{E}\log D+\mathbb{E}\log(1-D)$ |

```
  §1  Output range must match the conjugate     ──► Topic 1
  §2  Composition: backbone then activation     ──► Topics 2–3
  §3  Last layer linear; σ_f is f-specific      ──► Topic 2
  §4  GAN is one f inside VDM                   ──► Topic 4
  §5  Jensen–Shannon-like f                     ──► Topics 4–5
  §6  Negative reals; why −log(1+e^{−v})        ──► Topic 5
  §7  Sigmoid squeezes to (0,1)                 ──► Topics 6–8
  §8  One score, two verbs (saddle leftover)    ──► Topics 6, 9
```

**Running example (keep this).** A gallery of 60,000 handwritten 7s. A printer $G_\theta$ that eats Gaussian static. A judge who must write a number the **conjugate will accept**. Today that number is **negative**; after algebra the blog formula talks about a **percentage** $D\in(0,1)$ instead. Same game, two rulers.

```
  GALLERY (reals)     PRINTER (fakes)        JUDGE
  D ~ p_x             z → G_θ → x̂ ~ p_θ      first: T ∈ dom(f*) = R_−
                                             later: D = sigmoid(V) ∈ (0,1)

  WRONG:  mix the two rulers  (treat D as T)
  RIGHT:  T is a deterministic function of D; ranges differ
```

Someone asks: **is this a coding lab?** No. The pad stops at formulas and two triangles.

---

## 1. The judge may only write legal numbers

<a id="p1-range"></a>

### Purpose

Last hour’s $f^*$ is only defined on some set of numbers. $T$ must output **those** numbers.

### Definitions

**Domain of $f^*$:** the inputs $t$ for which $f^*(t)$ is allowed (finite).  
**Range of $T$:** the numbers $T(x)$ actually produces. They must sit inside $\mathrm{dom}(f^*)$.

### Micro numbers

If $\mathrm{dom}(f^*)=\{\,t<0\,\}$ and your net ends with a ReLU (outputs $\ge 0$), $f^*(T(x))$ is **undefined**. The bound formula breaks.

### Analogy

A vending machine that only takes **dimes**. Your judge writes a number on each photo. If the number is a quarter, the machine refuses. $\sigma_f$ is the adapter that turns the judge’s scribble into a dime.

Someone asks: **can I use the same last layer for every $f$?** Only if every $f^*$ shares a domain. They don’t.

### Local picture

```
  T(x)  ----must land in---->  dom(f*)
                 │
                 X  if last layer is “any real” and f* wants t<0
```

### Worked walk

$f^*(t)=-\log(1-e^{t})$ needs $t<0$.  
$T=+3$: $1-e^{3}<0$ → log of a negative → **garbage**.  
$T=-0.5$: $1-e^{-0.5}>0$ → $f^*$ finite → **legal**.

```
  WRONG:  any last layer (ReLU, sigmoid) for every f
  RIGHT:  last stamp chosen so range(T) ⊆ dom(f*)
```

### Notice

This is why the T-net is **tweaked** when $f$ changes — not the whole philosophy of VDM, just the last activation.

### Mini-check

1. If $f^*$ needs $t<0$, may $T$ output $+3$?  
2. Who chooses $\mathrm{dom}(f^*)$ — you, or the $f$ you picked?

---

## 2. Split the judge: backbone then stamp

<a id="p2-compose"></a>

### Purpose

Don’t rebuild a new net from scratch for every $f$. Share a backbone; swap the last stamp.

### Definitions

**Composition** $T=\sigma_f\circ V$: first $V(x)$ (a real), then $\sigma_f$ of that real.  
**$V_w$:** a net $X\to\mathbb{R}$.  
**$\sigma_f$:** a **scalar** function $\mathbb{R}\to\mathrm{dom}(f^*)$.

### Micro numbers

$V(x)=2.0$. If $\sigma_f(v)=-\log(1+e^{-v})$, then $\sigma_f(2)\approx -0.127<0$. Legal for $\mathrm{dom}=\mathbb{R}_-$.

### Analogy

A camera (shared $V$) plus a **filter** ($\sigma_f$) that depends on the contest. Same camera, different filter for black-and-white vs infrared.

Someone asks: **is $\sigma_f$ a whole second net?** No. It is a **fixed** formula on one number.

### Local picture

```
  x  →  [ V_w  last layer LINEAR ]  →  v ∈ R  →  [ σ_f ]  →  T(x) ∈ dom(f*)
           shared across f                 swapped when f changes
```

### Worked walk

Same photo $x$. Camera $V$ says $v=1.0$.  
Filter A (this $f$): $\sigma_f(1)=-\log(1+e^{-1})\approx -0.31$.  
Filter B (some other $f$) might map $1$ to $+2$. Same camera, different legal stamp.

```
  WRONG:  rebuild the whole critic for every f
  RIGHT:  share V (last linear); swap σ_f
```

### Notice

$T_w(x)=\sigma_f(V_w(x))$ is the whole critic in VDM letters. Later the GAN paper hides this split.

### Mini-check

1. What does $V_w$ output?  
2. Who depends on $f$ — $V$ or $\sigma_f$?

---

## 3. Last layer linear; activation is the $f$-specific part

<a id="p3-linear"></a>

### Purpose

He insists $V$’s **last layer is linear**. That is how you get “a real number,” not a probability yet.

### Definitions

**Linear last layer:** $v=w^\top h+b$ with no sigmoid/ReLU after it.  
**$f$-specific activation:** $\sigma_f$ **on top of** that scalar.

### Micro numbers

Hidden size 128 → one number $v=0.4$ (linear). Then $\sigma_f(0.4)$ or later $\mathrm{sigmoid}(0.4)\approx 0.60$. Two different last stamps, same $v$.

### Analogy

A thermometer that reports °C (linear scalar). You may then (a) clip to “below zero” for $f^*$, or (b) squash to a probability in $(0,1)$. The mercury column is $V$. The scale you glue on is $\sigma_f$ or sigmoid.

Someone asks: **why not sigmoid as the last layer of $V$?** Then $V$ would not be “$X\to\mathbb{R}$ common across $f$.” Sigmoid is a **later** rewrite for this one $f$.

### Local picture

```
  WRONG:  V ends with sigmoid for every f
  RIGHT:  V ends linear; σ_f (or later D) is extra
```

### Notice

“Common across all $f$-divergences” is load-bearing. Change $f$, keep $V$’s architecture, swap $\sigma_f$.

### Mini-check

1. Is the linear last layer part of $V$ or of $\sigma_f$?  
2. Does every $f$ share $V$’s last linear layer?

---

## 4. GAN is one $f$ inside last hour’s algorithm

<a id="p4-instance"></a>

### Purpose

Unlock the slogan: GAN is **not** a new training religion. It is **variational divergence minimization (VDM)** with one spring $f$.

### Definitions

**Instance / special case:** same two nets, same $\min_\theta\max_w$, one particular $f$.  
**Usual / vanilla GAN:** the original Goodfellow objective, which this sitting will recover.

### Micro numbers

Last hour: generic $J=\mathbb{E}T-\mathbb{E}f^*(T)$. This hour: pick one $f$, get $J_{\mathrm{GAN}}=\mathbb{E}\log D+\mathbb{E}\log(1-D)$. Same family, different $f$.

### Analogy

Last class: a restaurant with a **menu of sauces**. This class: “today’s special is sauce $f_{\mathrm{GAN}}$.” The kitchen (two nets, one scoreboard) did not change.

Someone asks: **did Goodfellow start from this $f$?** He says **no**. The 2014 paper wrote the two-log loss. We **arrive** there from VDM.

### Local picture

```
  VDM  =  min_θ max_w ( E T − E f*(T) )   for ANY convex f
                │
                │ pick one f
                ▼
  GAN  =  that algorithm with f(u)= u log u − (u+1) log(u+1)
```

### Notice

If you only memorize “generator vs discriminator,” you miss the course’s point: **one $f$, same bound**.

### Mini-check

1. What do you pick to get a GAN from VDM?  
2. Did the original paper motivate via $f$-div?

---

## 5. The GAN $f$ looks like Jensen–Shannon

<a id="p5-js"></a>

### Purpose

Name the spring and the hedge: **similar to JS, not exactly JS**.

### Definitions

**Jensen–Shannon divergence (JSD):** a symmetric mix of two KLs (last week’s $f$-div family).  
**GAN $f$ on the pad:** $f(u)=u\log u-(u+1)\log(u+1)$. He writes “(similar to JSD).”

### Micro numbers

At $u=1$: $f(1)=1\log 1-(2)\log 2= -2\log 2 <0$? Wait — standard $f$-div needs $f(1)=0$. Check: $1\log 1=0$, $(1+1)\log(1+1)=2\log 2$, so $f(1)=-2\log 2\neq 0$ as written...

Nowozin Table 1 GAN: $f(u)=u\log u-(u+1)\log(u+1)$ **plus** the understanding it is JS up to constants / affine. He says similar **with a constant**. Do **not** “fix” his formula. Teach: this is the $f$ he writes; it is JS-like.

### Analogy

Two recipes for chocolate cake. One uses 50 g cocoa, one 55 g. “Similar, not the same cake.” JS vs this $f$ is that kind of similar.

Someone asks: **is GAN exactly JS minimization?** He will not claim equality. He claims **this $f$** is what the usual GAN uses.

### Local picture

```
  JS  ≈  this f   (constant / slight difference)
  he writes:  f(u) = u log u − (u+1) log(u+1)
```

### Notice

The next two objects ($f^*$, $\sigma_f$) are **computed from this $f$**, not from a Wikipedia JS article.

### Mini-check

1. What $f$ does he write for GAN?  
2. Exact JS or similar?

---

## 6. $f^*$ lives on negative reals; $\sigma_f$ stays there

<a id="p6-neg"></a>

### Purpose

Why a funny last activation: $f^*$ **refuses** $t\ge 0$.

### Definitions

**$f^*(t)=-\log(1-e^{t})$** for this $f$. Needs $1-e^{t}>0$ $\Rightarrow$ $t<0$. Domain $\mathbb{R}_-=(-\infty,0)$.  
**$\sigma_f(v)=-\log(1+e^{-v})$** (speech: “$e$ power **minus** $v$”). Always negative: $1+e^{-v}>1$, log positive, overall minus.

### Micro numbers

$v=0$: $\sigma_f(0)=-\log(2)\approx -0.69<0$.  
$v=+10$: $\sigma_f\approx 0^-$ (tiny negative).  
$v=-10$: $\sigma_f\approx -10$. Never positive.

Tiny table:

| $v$ | $1+e^{-v}$ | $\sigma_f(v)$ |
|-----|------------|----------------|
| $0$ | $2$ | $-\log 2\approx -0.69$ |
| $+2$ | $\approx 1.14$ | $\approx -0.13$ |
| $-2$ | $\approx 8.4$ | $\approx -2.13$ |

Board at ~10:22 sometimes drops the minus on $v$. **Trust the speech and the sign argument.**

### Analogy

A basement that only stores boxes **below ground**. $\sigma_f$ is a chute that can only dump downward. $V$ can shout any real; the chute still delivers a negative.

Someone asks: **what if I skip $\sigma_f$ and feed $V=3$ to $f^*$?** $1-e^{3}<0$, log of a negative — garbage.

### Local picture

```
  v ∈ R  --σ_f-->  t < 0  --f*-->  −log(1−e^t)  OK
  v ∈ R  --skip--> t=3    --f*-->  undefined
```

### Notice

This activation is **not** the sigmoid yet. Sigmoid comes after algebra, as $D$.

### Mini-check

1. Why $\mathrm{dom}(f^*)=\mathbb{R}_-$?  
2. Why is $-\log(1+e^{-v})$ always negative?

---

## 7. Sigmoid squeezes a real to a probability-looking $(0,1)$

<a id="p7-sigmoid"></a>

### Purpose

After rearrangement the loss talks about $D_w=\sigma(V_w)\in(0,1)$, the **usual** last layer in GAN blogs.

### Definitions

**Sigmoid:** $D(v)=1/(1+e^{-v})\in(0,1)$.  
**$D_w(x)$:** sigmoid of $V_w(x)$. Letter $D$ matches the 2014 paper.

### Micro numbers

$V=0 \Rightarrow D=1/2$. $V=+5 \Rightarrow D\approx 0.993$. $V=-5 \Rightarrow D\approx 0.007$.

```
  V:   −∞ ........ 0 ........ +∞
  D:    0+ ...... 0.5 ...... 1−
```

### Analogy

A volume knob $V$ from $-\infty$ to $+\infty$. Sigmoid is the **percentage** “how sure this is a real photo,” from 0% to 100%, never quite 0 or 1.

Someone asks: **is that percentage $\mathrm{dom}(f^*)$?** No. $(0,1)$ vs $\mathbb{R}_-$. Different rulers. Topic 8’s warning.

### Local picture

```
  V = −2, 0, +2
  D = 0.12, 0.50, 0.88     all in (0,1)
```

### Notice

He introduces $D$ **so the formula looks like the GAN paper**. The VDM object was still $T=\sigma_f(V)$.

### Mini-check

1. Write $D$ in terms of $V$.  
2. Can $D$ equal $0$ or $1$ exactly?

---

## 8. Same scoreboard, opposite verbs (leftover saddle)

<a id="p8-saddle"></a>

### Purpose

$J_{\mathrm{GAN}}$ is still last hour’s bound: $\theta$ mins, $w$ maxes, **one** $J$.

### Definitions

**Saddle / adversarial:** $\min_\theta\max_w J$.  
**Two-log $J$:** $\mathbb{E}_{p_x}\log D_w(x)+\mathbb{E}_{p_\theta}\log(1-D_w(\hat x))$. First term likes $D\approx 1$ on reals; second likes $D\approx 0$ on fakes — the usual story, now as **this bound**.

### Micro numbers

Cartoon: $D(\mathrm{real})=0.9$, $D(\mathrm{fake})=0.2$.  
$\log 0.9+\log(1-0.2)\approx -0.105-0.223=-0.33$.  
If fakes fool $D$ to $0.5$: second log $\log 0.5\approx -0.69$ (worse for the judge’s max). Printer tries to move that.

### Analogy

Tug-of-war on **one rope** (last sitting). Today the rope is written with logs of $D$ and $1-D$. Same game, new letters.

Someone asks: **is this already a training loop?** No. Next video: implement in practice.

### Local picture

```
  reals:  want D close to 1     →  log D large
  fakes:  want D close to 0     →  log(1−D) large
  G tries the opposite on fakes
```

### Worked walk

$D(\mathrm{real})=0.9$, $D(\mathrm{fake})=0.1$:  
$\log 0.9+\log 0.9\approx -0.21$ (judge is happy).  
Printer fools $D(\mathrm{fake})\to 0.5$: second term $\log 0.5\approx -0.69$ (judge’s score drops). Printer **mins** $J$; judge **maxes** $J$. Same rope.

```
  WRONG:  two different losses glued later
  RIGHT:  one J, opposite verbs  (last hour’s saddle, new letters)
```

### Notice

**No Dataset, no `BCELoss` cell.** The pad stops at the formula and the two triangles.

### Mini-check

1. Who maximizes $J$ — $\theta$ or $w$?  
2. Is $J_{\mathrm{GAN}}$ a new loss or the old bound rewritten?

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
