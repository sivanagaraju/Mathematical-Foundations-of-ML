# Prerequisites — warm-up before W1_T3 (VDM in practice)

> **Do this first** if “IID,” “push-forward,” “lower bound,” “saddle,” or “adversary” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues [W1_L4 f-divergence](../03-W1-L4-Variational-Divergence-Minimization/NOTES.md) and the bound sitting (playlist T1).  
> **Beginner:** purpose · definition · micro numbers · analogy · ASCII · notice · mini-check.

The YouTube title says **datasets & dataloaders**. **This recording does not open PyTorch.** It is chalk: how to *use* last hour’s variational **lower bound** as two nets, a saddle, and the GAN names. Dataset / DataLoader live in [W1_T2](../04-W1-T2-Introduction-to-PyTorch-Tensors/NOTES.md). Every idea below unlocks a map word. None of them is the lecture.

```
  After this warm-up you can say:

  "Data are IID draws from an unknown law p_x — the pile is not the law."
  "A generator G_θ pushes easy z through a net; outputs have some law p_θ."
  "An f-divergence scores two laws; it is 0 iff they match."
  "You cannot type the integral. Averages of a function on samples approximate expectations (LLN)."
  "Last hour gave a LOWER BOUND on D_f, not D_f itself. Min of a floor is not min of the roof."
  "That bound is itself a MAX over a function T, then a MIN over generator weights θ."
  "A second net T_w stands in for T because we cannot search all functions by hand."
  "Min in θ and max in w of the SAME score is a saddle — two players pulling opposite ways."
```

**Symbol card.** Keep this next to NOTES.

| Word | Picture | This sitting’s job |
|------|---------|-------------------|
| $p_x$ | unknown true pile | law that made the training files |
| $D=\{x_1,\ldots,x_n\}$ | $n$ IID draws | all you see of $p_x$ |
| $z\sim\mathcal{N}(0,I)$ | easy noise | you *can* sample this |
| $G_\theta$ | the machine $z\mapsto\hat x$ | generator / sampler |
| $p_\theta$ | fake pile’s law | law of $\hat x=G_\theta(z)$ |
| $D_f$ | score of “how far” | $f$-divergence (one convex $f$) |
| $f^*$ | conjugate of $f$ | turns $f$ into a usable expectation |
| $T$ / $T_w$ | a scoring function / a net | inner max; later called critic |
| $J(\theta,w)$ | one number from two piles | $\mathbb{E}_{p_x}T_w-\mathbb{E}_{p_\theta}f^*(T_w)$ |
| saddle | min one way, max the other | what training actually seeks |

```
  §1  IID pile vs unknown law                 ──► Topic 1
  §2  Push-forward sampler G_θ                ──► Topics 1, 6
  §3  f-divergence as a yardstick             ──► Topics 1–2
  §4  Expectation ≈ sample average (LLN)      ──► Topic 2
  §5  Lower bound is not the quantity         ──► Topics 2–4
  §6  Nested min and max of one score         ──► Topics 4–6
  §7  A net can stand in for a function       ──► Topic 5
  §8  Saddle / two players                    ──► Topics 7–9
```

**Running example (keep this).** A folder of 60,000 handwritten 7s (the pile). A printer that eats Gaussian static and dumps a new 28×28 grid (the machine). A judge who writes one number per grid, real or dumped (the variational function). The lecture is: you cannot type the true mismatch of the two *laws*; you can average the judge on the two *folders*.

```
  FOLDER of real 7s     PRINTER of fake 7s      JUDGE writes a number
  D ~ unknown p_x       z → G_θ → x̂ ~ p_θ       T or T_w
         \                    /                      │
          └── two files ──────┘                      │
                    \                                /
                     └── one score J  =  avg_real T − avg_fake f*(T)
                          min_θ  and  max_w  of THAT same J
```

Someone asks: **is the folder the law?** No. **Does the printer print a density?** No. **Is lowering the judge’s number the same as matching the laws?** Not exactly — that is the red ≈.

---

## 1. The pile is not the law

<a id="p1-px"></a>

### Purpose

The first board line is always the same: you are given files, not a formula.

### Definitions

**Law / distribution $p_x$:** the hidden rule for which files show up.  
**Sample / draw:** one file produced by that rule.  
**IID (independent and identically distributed):** each file is a fresh draw from the **same** unknown $p_x$. Independence: one file does not leak the next. Identically distributed: they share one law.

### Micro numbers

A bag of 60,000 MNIST digits. You do **not** have $p_x(28\times 28\text{ grid})$. You have 60,000 grids. Those grids $\sim p_x$ (unknown).

Fair die: law is $P(k)=1/6$. Twenty rolls $\{2,5,5,1,\ldots\}$ are samples. The list is not $1/6$. Twenty rolls do not *become* the law.

### Analogy

A bakery’s secret recipe is $p_x$. The display case is $D$. Tasting every pastry still does not hand you the recipe on paper.

A second picture: a weather station prints one temperature each day. The printouts are samples. The **climate** is $p_x$. Filing 365 printouts is not owning the climate formula.

### Local picture

```
  LAW (hidden)                 SAMPLES (what you hold)
  p_x  =  rule                 D = {x1, x2, …, xn}
  die: P(k)=1/6                twenty rolls 2,5,5,1,…
  MNIST: unknown density       60,000 grids

  unknown p_x  ──IID draws──►  x1, x2, …, xn
```

### Worked walk (die, then digits)

Fair die, law $P(k)=1/6$. Four rolls: $2,5,5,1$. Histogram: two 5s, one 2, one 1. That histogram is **not** $1/6$. A fifth roll is not “copy one of the four”; it is a **new** draw from the hidden law.

Digits: 60,000 grids of 7. Pixel $(0,0)$ being ink in 12% of them is a **statistic of the pile**, still not $p_x(\text{this 784-vector})$. High $d$ ($784$ for MNIST) is why you will never type $\int p_x(x)\,dx$ as a training loss.

```
  WRONG:  D  =  p_x          (the pile is the law)
  RIGHT:  D  ~  p_x          (the pile is n draws)

  WRONG:  store D, emit a copy          (photocopier)
  RIGHT:  emit x_new never in D         (sampler)
```

### Notice

He writes $x_i\in\mathbb{R}^d$ sitting in a space $\mathcal{X}$. High $d$ (images) is why you will never type the integral later.

### Mini-check

1. Is $D$ the same object as $p_x$?  
2. What does IID constrain — pixels inside one image, or images across the pile?  
3. Why does “unknown” make $\int p_x(\cdot)\,dx$ unusable as a training loss?

---

## 2. Push noise through a net and you get a fake file

<a id="p2-push"></a>

### Purpose

The sampler is not a density printout. It is a **machine**: easy $z$ in, new $\hat x$ out.

### Definitions

**Known easy law:** something you *can* sample — here a **standard Gaussian** $\mathcal{N}(0,I)$ (“unit normal” on the board).  
**Deterministic $G_\theta$:** a neural net with weights $\theta$. Same $z$ always gives the same $\hat x$. Chance lives in $z$, not in $G$.  
**Push-forward:** the law of $\hat x=G_\theta(z)$ when $z\sim\mathcal{N}(0,I)$. That law is named $p_\theta$. Changing $\theta$ **is** changing $p_\theta$.

### Micro numbers

$z$ is 100 numbers $\sim\mathcal{N}(0,1)$. $G_\theta$ is a stack of matrix multiplies + nonlinearities that emit a $28\times 28$ grid $\hat x$. Repeat 64 times: a **mini-batch of fakes**. You still do not have a formula for $p_\theta(\hat x)$.

Tiny 1-D cartoon: $z\sim\mathcal{N}(0,1)$, $G_\theta(z)=\theta z+1$. Then $p_\theta$ is $\mathcal{N}(1,\theta^2)$. You see the pattern: the **function** $G$ **is** the model.

### Analogy

A pasta maker. Flour-water mix $z$ is random-ish dough you know how to scoop. The brass die $G_\theta$ is deterministic: that shape always makes fusilli. Changing the die (changing $\theta$) changes the **shape distribution** of the pasta on the plate ($p_\theta$). You never write a density of fusilli; you **make** fusilli.

A second picture: a photo printer with a noise knob. You twist the knob ($\theta$) until the prints look like the gallery ($p_x$). The printer never lists $P(\text{this pixel})$.

### Local picture

```
  easy, known                  learnable, deterministic         unknown law of output
  z ~ N(0,I)   ──────────►     G_θ(z)            ──────────►   x̂ ~ p_θ

  G is a triangle on the pad.  Chance is in z.  p_θ depends on θ.
```

### Worked walk (1-D, then 784-D)

$z\sim\mathcal{N}(0,1)$, $G_\theta(z)=\theta z+1$. Then $\hat x\sim\mathcal{N}(1,\theta^2)$. You **see** $p_\theta$ here because $G$ is affine. Twist $\theta$ from $1$ to $3$: the fake pile spreads. You still never “print $p_\theta(x)$” as a number at every $x$; you print **samples**.

MNIST: $z\in\mathbb{R}^{100}$, $G_\theta$ a stack of matrices, $\hat x\in\mathbb{R}^{784}$. Draw $z$ sixty-four times $\Rightarrow$ a batch of 64 fake 7s. Same $\theta$, new $z$ $\Rightarrow$ a **new** fake. Freeze $z$, change $\theta$ $\Rightarrow$ a different fake from the **same** noise.

```
  WRONG:  G_θ is a random net; chance lives in the weights
  RIGHT:  G_θ is deterministic; chance lives in z

  WRONG:  the net outputs p_θ(x) at every pixel
  RIGHT:  the net outputs a sample x̂ whose *law* is p_θ
```

Someone asks: **if I photocopy one training 7, is that $G_\theta$?** No. A photocopier does not eat Gaussian $z$.

### Notice

If $p_\theta=p_x$, then this machine **is** a sampler for $p_x$. That “if” is the whole course until GAN.

### Mini-check

1. Where does randomness live — $G$ or $z$?  
2. If you freeze $\theta$ and draw a new $z$, do you get a new $\hat x$?  
3. Why is $p_\theta$ not a printout you can plug into $D_f$?

---

## 3. A divergence is a score of “how far,” not a tape measure

<a id="p3-fdiv"></a>

### Purpose

Training needs a number that is small when $p_\theta$ matches $p_x$ and large otherwise.

### Definitions

**Divergence $D(p\|q)$:** a score of mismatch. He still says “metric.” It need **not** be a metric (KL is not symmetric).  
**$f$-divergence:** pick a convex function $f$ with $f(1)=0$. Then (one standard writing)

$$
D_f(p_x\|p_\theta)=\int p_\theta(x)\,f\!\left(\frac{p_x(x)}{p_\theta(x)}\right)\,dx.
$$

One $f$ $\Rightarrow$ one named child (KL, JS, TV, …).  
**Good yardstick:** $D_f\ge 0$, and $D_f=0$ **if and only if** $p_x=p_\theta$. Then minimizing $D_f$ really *is* matching the laws.

### Micro numbers

Two coins. $p=\mathrm{Bernoulli}(0.5)$, $q=\mathrm{Bernoulli}(0.9)$. Forward KL is a **positive** number. If $q=p$, every $f$-div with $f(1)=0$ returns **0**.

You do **not** need the KL formula this hour. You need: different $f$, same **family**.

### Analogy

A teacher’s red pen. Different rubrics ($f$) mark the same two essays differently, but every honest rubric gives **0** when the essays are copies of each other. $f$-divergence is a **family of rubrics**, not one mark.

A second picture: kitchen scales vs a “how spicy” scale. Both score difference. Neither has to obey the triangle inequality. He still says “metric”; you hear “score.”

### Local picture

```
  pick convex f, f(1)=0
           │
           ▼
  D_f(p_x || p_θ)  ≥  0
           │
           ▼
  = 0  iff  p_x = p_θ     ← then G_θ is a sampler for p_x
```

### Worked walk (two coins)

Let $p=\mathrm{Bernoulli}(1/2)$ (fair), $q=\mathrm{Bernoulli}(0.9)$. They are **different laws**, so any honest $f$-div is **strictly positive**. Set $q=p$: the ratio $p/q=1$ everywhere $f$ is evaluated, $f(1)=0$, so $D_f=0$.

That “zero iff equal” is why minimizing $D_f$ **is** matching the printer to the gallery. A made-up score that is zero for two different laws would let a bad printer look trained.

```
  WRONG:  “metric”  ⇒  D(p,q)=D(q,p) and triangle inequality
  RIGHT:  he says metric; you hear SCORE. KL is not symmetric.

  WRONG:  pick any loss (MSE on pixels) and call it a divergence
  RIGHT:  need ≥0 and =0 iff the two *laws* match
```

Someone asks: **can I compute that integral on 784-D images?** No — you do not have $p_x$ or $p_\theta$ as formulas. That is the next idea.

### Notice

The integral still contains **both densities**. That is why this sitting cannot just “compute $D_f$ and descend.”

### Mini-check

1. If $D_f=0$, what is true of $p_x$ and $p_\theta$?  
2. Does changing $f$ change the training problem?  
3. Why is “metric” a slightly sloppy word here?

---

## 4. Integrals of densities become averages on files (LLN)

<a id="p4-lln"></a>

### Purpose

The slogan last sitting, reused today: you cannot integrate a density you do not have, but you **can** average a function on samples you **do** have.

### Definitions

**Expectation:** $\mathbb{E}_{x\sim v}[h(x)]=\int h(x)\,v(x)\,dx$. In words: the average of $h$ if $x$ is drawn from $v$.  
**Law of large numbers (LLN):** if $x_1,\ldots,x_n$ are IID from $v$, then $\frac1n\sum_i h(x_i)$ approaches that expectation as $n$ grows.

So: **if** a training score is an expectation under $p_x$ (use real files) and/or under $p_\theta$ (use $G_\theta(z)$ fakes), you can **estimate** it without typing $p_x$ or $p_\theta$.

### Micro numbers

$v$ = fair die. $h(x)=1$ if $x$ is even, else $0$. True expectation $=1/2$. Twenty rolls: maybe $9/20=0.45$. Two thousand rolls: much closer to $0.5$. That *is* LLN. No density formula was needed — only rolls.

### Analogy

You want the average height of adults in a city. You do not have the city’s height-density formula. You measure 1,000 people. The sample mean **is** the integral, approximately.

A second picture: a poll. You cannot interview every voter ($p$). You interview a sample. The poll average stands in for $\mathbb{E}_p$.

### Local picture

```
  want:   ∫ h(x) v(x) dx     =  E_{x ~ v}[ h(x) ]

  hold:   x1, …, xn  ~ v     (IID)

  use:    (1/n) Σ h(xi)      ≈  the integral     (LLN)
```

### Worked walk (even faces of a die)

$h(x)=1$ if $x$ even, else $0$. True $\mathbb{E}=1/2$.

| rolls $n$ | example average of $h$ |
|-----------|------------------------|
| 4 | $1,2,5,6$ → two evens → $2/4=0.50$ (lucky) |
| 20 | maybe $9/20=0.45$ |
| 2000 | typically $0.49$–$0.51$ |

No density formula was used. Only rolls.

**Two piles, as the lecture will need:**

```
  E_{p_x}[ T(x) ]     ≈  (1/n) Σ T(x_i)           x_i in the REAL folder
  E_{p_θ}[ f*(T(x̂)) ] ≈  (1/m) Σ f*(T(G_θ(z_j)))  z_j ~ N(0,I), FAKE folder
```

That pair of averages **is** the computable object. It is not Python `Dataset` code. It is LLN applied twice.

```
  WRONG:  LLN hands you the density v(x) at every x
  RIGHT:  LLN hands you integrals of a function h you can evaluate

  WRONG:  average f(p_x/p_θ) on fakes   (you cannot evaluate the ratio)
  RIGHT:  average T on reals and f*(T) on fakes   (after conjugacy)
```

Someone asks: **I averaged 4 rolls and got 0.5 — is the law 1/2 for sure?** No. LLN is “in the long run,” not “four is enough.”

### Notice

This only helps **after** $D_f$ has been rewritten as expectations (or a bound made of expectations). A raw $\int p_\theta f(p_x/p_\theta)$ still has the **ratio of two unknown densities** inside $f$. That is why last hour needed a conjugate.

### Mini-check

1. What object does LLN replace — the density, or the integral of a function against the density?  
2. Which pile estimates $\mathbb{E}_{p_x}$? Which pile estimates $\mathbb{E}_{p_\theta}$?  
3. Why is “average $f(\text{ratio})$ on fakes” still blocked?

---

## 5. A lower bound is a floor, not the house

<a id="p5-bound"></a>

### Purpose

Last hour’s algebra did **not** give $D_f$ as two clean expectations. It gave a **lower bound**. Today they **minimize the floor** and say so out loud.

### Definitions

**Lower bound:** a number $B$ with $D_f\ge B$ always. Raising $B$ can make $B$ closer to $D_f$; it never overshoots.  
**Fenchel / convex conjugate $f^*$:** a second function built from convex $f$. Board name: conjugate of $f$. ASR says “fential.” You do **not** need the $\sup_u(ut-f(u))$ formula to follow today’s sitting — you need: $f^*$ is what appears in the second expectation.  
**Variational bound they reuse:**

$$
D_f(p_x\|p_\theta)\;\ge\;\sup_{T\in\mathcal{T}}\Bigl(\mathbb{E}_{p_x}[T(x)]-\mathbb{E}_{p_\theta}[f^*(T(x))]\Bigr).
$$

He will write $\max$ instead of $\sup$ **assuming the sup is attained**.

### Micro numbers

True height of a building $= 40$ m. You only know “at least $B$.” Today $B=12$. Minimizing $B$ (making the floor smaller) can send $B$ to $0$ while the building is still $40$. **Min of a floor $\neq$ min of the height.** That is his red $\approx$ on the pad.

A nicer case: if you could **maximize** $B$ over $T$ until $B$ **equals** $D_f$, then min over $\theta$ of that tight bound *would* match min $D_f$. The gap is: $\mathcal{T}$ may miss the best $T$, and even then they **min** the bound rather than the unknown $D_f$.

### Analogy

You want the cheapest flight (true $D_f$). The website only shows a **“from $B$”** teaser. Booking the smallest teaser is **not** booking the cheapest flight. It is the best the website lets you click.

A second picture: a high-jump bar you cannot see. You only see a **mat** under it. Lowering the mat does not lower the bar. Last hour built the mat. This hour trains on the mat.

### Local picture

```
  roof  =  D_f          (cannot touch: densities + high-d integral)

  floor =  max_T ( E_{real}[T] − E_{fake}[f*(T)] )

  D_f  ≥  floor  always

  they set:  θ*  ≈  argmin_θ  (that floor)
                    └── red ≈ on the board: NOT equivalent
```

### Worked walk (building vs mat)

True height $40$ m. Mat heights you can actually measure: $12$, then $8$, then $1$. You “minimized the mat.” The building is still $40$.

A nicer fantasy: if the mat could be **raised** until it **touches** the roof (inner **max** over judges $T$ finds a tight bound), *then* lowering that tight mat with $\theta$ would move the roof. Two gaps remain: (1) the judge class may miss the best judge, so the mat never touches; (2) they **minimize** the mat even when it is loose. Both sit in the red $\approx$.

```
  WRONG:  bound ≈ quantity, so argmin bound = argmin D_f
  RIGHT:  D_f ≥ bound always; min of a floor is a different problem

  WRONG:  “variational” means VAE
  RIGHT:  here variational = optimize over a *function* T, then a net T_w
```

Someone asks: **why not maximize the bound over $\theta$ too?** Because $\theta$ is supposed to **match** laws, which **shrinks** $D_f$. The inner player **builds** the floor (max); the outer player **lowers** it (min). Opposite verbs, one number.

### Notice

Every later GAN-looking formula in this video **is** that floor, not $D_f$. The conjugate $f^*$ is last hour’s unzip; today it is just “the function on the fake pile.”

### Mini-check

1. If the floor is $3$ and $D_f$ is $10$, and you drive the floor to $1$, did you minimize $D_f$?  
2. Why replace $\sup$ by $\max$?  
3. Which pile does $T$ get averaged on? Which pile does $f^*(T)$ get averaged on?

---

## 6. One score, two arrows: min $\theta$, max $T$

<a id="p6-nested"></a>

### Purpose

The floor is not a number you just read. It is itself an **optimization** over functions $T$. The generator problem sits **outside** that.

### Definitions

**Outer problem:** choose $\theta$ of $G_\theta$ to make the bound **small**.  
**Inner problem:** choose $T$ in a class $\mathcal{T}$ to make $\mathbb{E}_{p_x}T-\mathbb{E}_{p_\theta}f^*(T)$ **large** (that is how you *compute* a good floor).  
**Same objective:** both arrows act on **one** expression. Not two different losses glued later.

### Micro numbers

Pretend $J(\theta,T)=(\theta-2)^2-(T-3)^2$ (cartoon, not the lecture formula). Inner: for fixed $\theta$, pick $T$ to **max** $J$ (want $T=3$). Outer: pick $\theta$ to **min** that result (want $\theta=2$). Nested: $\min_\theta\max_T J$. You cannot drop the inner max and just “gradient $J$ wrt $\theta$” as if $T$ were a constant forever.

### Analogy

A landlord sets rent ($\theta$) to keep profit **low for the tenant**; the tenant’s lawyer ($T$) writes the **strongest** possible complaint so the “how overpriced” score is **high**. Same spreadsheet column. Opposite verbs.

A second picture: a debate. The critic tries to **inflate** the mismatch score; the generator tries to **deflate** it. One podium, two speakers.

### Local picture

```
  J(θ, T) =  E_{p_x}[T(x)]  −  E_{p_θ}[ f*(T(x)) ]

       inner:  max over T ∈ 𝒯     (build the floor)
       outer:  min over θ         (push G so the floor drops)

  θ* ≈ argmin_θ  [  max_T  J(θ, T)  ]
```

### Worked walk (cartoon $J$, not the lecture formula)

Let $J(\theta,T)=(\theta-2)^2-(T-3)^2$.

- Inner, $\theta$ frozen: max over $T$ wants $T=3$ (the $-(T-3)^2$ peak).  
- Outer: after that, $J=(\theta-2)^2$, min at $\theta=2$.

If you **drop** the inner max and pick a random $T=0$, you are optimizing a **different** number than the bound.

```
  spreadsheet column J
     lawyer T  →  make J LARGE   (inner)
     landlord θ →  make J SMALL  (outer)

  WRONG:  two different losses glued later
  RIGHT:  same J, opposite verbs

  WRONG:  freeze a random T forever, descend θ
  RIGHT:  T is part of the bound; dropping max_T collapses the floor
```

Someone asks: **who sees the real 7s?** The first expectation — $T$ on the real folder. Fakes go through $f^*(T)$.

### Notice

He marks the inner max as “w.r.t. a **class of functions**” and the outer min as “w.r.t. **parameters of $G_\theta$**.” Asymmetric on purpose: $\theta$ is already a vector of weights; $T$ is still a whole function.

### Mini-check

1. If you skip the inner max, what are you computing?  
2. Do $\theta$ and $T$ get different loss functions, or the same $J$?  
3. Which player sees real $x$’s in the first expectation?

---

## 7. A neural net can stand in for “all functions”

<a id="p7-approx"></a>

### Purpose

You cannot loop over every function $T:\mathcal{X}\to\mathbb{R}$. Practice replaces $\mathcal{T}$ by a **second net**.

### Definitions

**Function class $\mathcal{T}$:** the bag of allowed $T$’s in the inner max.  
**Parameterize:** pick a family $T_w$ indexed by weights $w$ (another neural net). Searching $w$ **stands in** for searching $T$.  
**Universal function approximator (spoken reason):** wide-enough nets can approximate many continuous functions on compact sets. This is a **license**, not a proof that your particular $T_w$ hit the ideal $T^*$.

### Micro numbers

$\mathcal{X}=\mathbb{R}$ for a cartoon. $\mathcal{T}=$ “all parabolas $T(x)=ax^2+bx+c$.” Then $w=(a,b,c)$ is already a parameterization. A net $T_w$ is the same idea with millions of $w$’s and a more flexible shape.

If the ideal $T^*$ is **not** in your net family, the inner max is **too small**, the floor is **looser**, and the red $\approx$ in Topic 4 gets worse.

### Analogy

You need the best lawyer in the country ($T^*$). You can only hire from **one firm** ($T_w$ with that firm’s staff $w$). A huge firm approximates “all lawyers.” A tiny firm does not. Universal approximation says “huge firm, in theory.” Your GPU is a finite firm.

A second picture: fitting a curve with a 3-layer MLP instead of writing $T(x)=\sin(3x)+x^2$ by hand. The MLP **is** the function, once $w$ is chosen.

### Local picture

```
  yesterday:   max over T in a huge bag 𝒯
  today:       max over weights w of a net T_w(x)

  T_w  takes x (real or fake)  →  a number T_w(x)

  now BOTH players are nets:
      G_θ   (sampler)     min_θ
      T_w   (variational) max_w
```

### Worked walk (parabolas, then a net)

Suppose $\mathcal{T}=$ all parabolas $T(x)=ax^2+bx+c$. Then $w=(a,b,c)$ **is** already a parameterization: max over $T$ became max over three numbers. A neural net is the same idea with millions of $w$’s and a more flexible shape.

Feed the same $T_w$ a real 7 and a fake 7. Two numbers come out. Average the real numbers; average $f^*$ of the fake numbers; subtract. That is $J$.

```
  WRONG:  T_w eats z  (noise)
  RIGHT:  T_w eats x  (data-space: real or fake)

  WRONG:  universal approximation ⇒ we hit T*
  RIGHT:  license to search a rich family; T* may still be outside

  WRONG:  T_w is “the discriminator from a blog”
  RIGHT:  T_w is last hour’s variational T, now with weights
```

Someone asks: **if my net is too small, is the inequality $D_f\ge J$ still true?** Yes (a smaller max is a looser floor). **Is it tight?** Usually no — the red $\approx$ gets worse.

### Notice

$T_w$ eats **$x$**, not $z$. Real files and fake files both go through the same $T_w$. $G_\theta$ eats **$z$**.

### Mini-check

1. Why not optimize $T$ “analytically”?  
2. What does the subscript $w$ record?  
3. If $T^*$ is outside the net family, is the bound still valid? Is it still tight?

---

## 8. A saddle is a min in one direction and a max in the other

<a id="p8-saddle"></a>

### Purpose

Ordinary training seeks a **bottom of a bowl**. This sitting seeks a **saddle**: sit so that walking in $\theta$ goes **up** and walking in $w$ goes **down**.

### Definitions

**Saddle point $(\theta^*,w^*)$ of $J$:** $J$ is minimized in $\theta$ and maximized in $w$ at that pair. He draws: move in the $\theta$ direction $\Rightarrow$ $J$ **increases**; move in the $w$ direction $\Rightarrow$ $J$ **decreases**. That matches $\min_\theta\max_w J$.  
**Saddle-point optimization:** the problem whose solutions **are** saddles.  
**Adversarial (preview of Topics 8–9):** two nets, **opposite verbs** on the **same** $J$. Each undoes the other. That opposition **is** the name “adversarial networks.”  
**Ordinary advice (contrast):** typical non-convex optimization **avoids** saddles (they are not local minima). Here they **seek** one **on purpose**.

### Micro numbers

$J(\theta,w)=\theta^2-w^2$. At $(0,0)$: along $\theta$, $J$ looks like a U (min); along $w$, $J$ looks like an upside-down U (max). Horse-saddle shape. Gradient zero, but **not** a bowl.

He warns: landscapes can have **many** saddles, and **finding** one is hard. That is why people say GAN training is unstable — already at the geometry, before any code.

### Analogy

A mountain pass. Walk north (the $\theta$-ridge): you climb. Walk east (the $w$-valley): you descend. The pass is the saddle. Hikers looking for a campsite (ordinary min) **avoid** the pass. This algorithm **wants** the pass.

A second picture: tug-of-war on one rope ($J$). Team $\theta$ pulls to **shorten** the score; team $w$ pulls to **lengthen** it. Equilibrium is not “everyone sits down.” It is a **stalemate**.

### Local picture

```
            w
            ^
            │     J decreases this way (max over w)
            │
            │        ╱  ╲
            │       ╱    ╲     ← horse saddle
            │      ╱  *   ╲      * = (θ*, w*)
            │         |
            +---------+------ θ →
                      J increases this way (min over θ)

  ordinary ML:  hide from saddles
  this sitting:  walk to a saddle on purpose
```

### Worked walk ($J=\theta^2-w^2$)

At $(0,0)$: along $\theta$, $J$ is a U (min). Along $w$, $J$ is an upside-down U (max). Gradient is zero, but it is **not** a bowl.

| step | $J$ |
|------|-----|
| stay at $(0,0)$ | $0$ |
| $\theta=\pm 1$, $w=0$ | $+1$ (went **up** — min in $\theta$) |
| $\theta=0$, $w=\pm 1$ | $-1$ (went **down** — max in $w$) |

Ordinary training: **flee** this point (not a local min). This sitting: **walk to it on purpose**. Many such passes on one range $\Rightarrow$ GAN training feels unstable **before any bug in code**.

```
  WRONG:  “loss went to 0”  ⇒  we won
  RIGHT:  a saddle need not look like a quiet bowl

  WRONG:  adversarial = a security attack / an angry net
  RIGHT:  opposite verbs on the SAME J

  WRONG:  critic and discriminator are already the same job today
  RIGHT:  critic = builds the bound (this hour)
          discriminator-as-classifier = one later f (not shown)
```

Someone asks: **why two names for $T_w$?** Critic matches **this** derivation (it *is* the bound). Discriminator matches **one** upcoming contest format where $T_w$ reads as a classifier. He stamps both so papers make sense. **No Python this hour.**

### Notice

He will later stamp $G_\theta$ **generator** and $T_w$ **discriminator / critic**. Those are **names** for the two players. The critic “constructs the bound.” The discriminator-as-classifier is **one** later $f$, not yet shown. **No Python this hour.**

### Mini-check

1. At a $\min_\theta\max_w$ saddle, does $J$ go up or down if you wiggle $\theta$?  
2. Why is this rarer than “run Adam until loss is small”?  
3. If two nets share one $J$ with opposite verbs, why does he say they are **adversaries**?

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
