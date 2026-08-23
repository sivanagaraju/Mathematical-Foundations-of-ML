# Prerequisites — warm-up before Lec 03 (f-Divergence and Examples)

> **Do this first** if “density,” “divergence,” or “generator” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues [Lec 02](../15-Lec02-Generative-Models-Problem-Formulation/NOTES.md) and the probability recaps.  
> **Beginner deep warm-up:** purpose · definition · micro numbers · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "A density height is not a probability."
  "We have a cloud of points, not the law itself."
  "A sample mean is a stand-in for an expectation."
  "A convex cup never sits above its chords."
  "Likelihood here means the density evaluated at a point."
  "A divergence can be one-way; a metric cannot."
  "A map G turns noise Z into a new random variable."
  "Missing a real mode and inventing junk are two different failures."
```

**Warm-up → lecture boxes**

```
  §1  Density vs probability vs samples   ──► Topics 1, 6, 8, 10
  §2  An IID cloud from unknown p         ──► Topics 1–2
  §3  Sample mean stands in for E         ──► Topic 4
  §4  A convex cup                        ──► Topics 8–9
  §5  Likelihood = a height               ──► Topic 10
  §6  Divergence is not a metric          ──► Topics 3, 8
  §7  Push a known Z through a map        ──► Topics 5, 7
  §8  Miss a mode vs invent junk          ──► Topic 10
```

---

## 1. Density, probability, and a pile of points

<a id="p1-density"></a>

### Purpose for the video

He writes small $p$ for a **density** and a script $P$ for a **distribution**, then evaluates $p(x)$ and calls that a **likelihood**. Those three words are not synonyms.

### Definitions

A **distribution** (CDF / law) answers: how much mass sits in this *region*? That number is in $[0,1]$.

A **density** $p$ is a *height function*. Area under $p$ over a region is the probability of that region. The height $p(x)$ at one point is a **non-negative real**, often **larger than 1**. It is **not** “the probability of $x$.”

A **sample** is one realized number (or image). A **dataset** is a pile of samples. The pile is not the density.

### Worked micro

Uniform on $[0,0.5]$ has height $p(x)=2$ on that interval. Height $2$ is legal. $P(X\in[0,0.25])=0.5$ is the probability.

### Analogy — jam on toast vs a bite

The density is how thick the jam is at a crumb. The probability is how much jam you get if you bite a *region*. A photo of ten crumbs is neither the recipe nor the jar.

### Notice

- Continuous points have probability **zero**; that is why we need densities.
- He assumes every law in this course has a density. Not every RV does.

### Mini-check

1. Can $p(x)=3$?  
2. Is “I have MNIST” the same as “I know $p_{\mathrm{MNIST}}$”?

---

## 2. An IID cloud from an unknown law

<a id="p2-iid"></a>

### Purpose for the video

The first board is $D=\{x_1,\ldots,x_n\}\sim_{\mathrm{iid}} p_x$ with $p_x$ **unknown**.

### Definitions

**Identical:** every $x_i$ comes from the *same* unknown $p_x$.  
**Independent:** seeing $x_1$ does not change the law of $x_2$.  
Together: **IID**.

The tilde $\sim$ means “sampled from.” He also uses **sample** as a *verb* (draw from $p$) and as a *noun* (a drawn point).

### Worked micro

Ten photos from one “MNIST factory.” You never get the factory’s density printout. You get ten tickets.

### Analogy — one espresso machine, many cups

Same grind all morning (identical). One cup does not change the next (independent). The logbook is not the machine.

### Notice

- Generative modeling adds a second job: **make more cups**, not only describe the machine.

### Mini-check

1. If image 2 is a crop of image 1, are they IID?  
2. Does “$n$ samples” tell you $p_x$?

---

## 3. A sample mean stands in for an expectation

<a id="p3-sample-mean"></a>

### Purpose for the video

Thirty–forty percent of the course is: rewrite a divergence so **samples** can estimate it. The baby case is already in your bones.

### Definitions

If $X$ has density $p$, $\mathbb{E}[X]=\int x\,p(x)\,dx$.  
You usually **cannot** do the integral (you do not have $p$).  
The **sample mean** $\bar x=(x_1+\cdots+x_n)/n$ is a **statistical estimator** of that expectation.

A **parametric family** is a menu of hills indexed by a knob vector $\theta$. An **explicit** family writes a formula (a two-Gaussian mixture is $\pi\,\mathcal{N}(\mu_1,\sigma_1^2)+(1-\pi)\mathcal{N}(\mu_2,\sigma_2^2)$). An **implicit** family only gives you a machine that *emits* points (a net $G_\theta(Z)$). Same word “model,” two kinds of access.

### Worked micro

True mean of a fair die is $3.5$. Five rolls $2,6,1,5,4$ give $\bar x=3.6$. Wrong a little, usable.

### Analogy — weighing a bag by grabbing handfuls

You never weigh every grain. A few handfuls estimate the typical weight. Later, $D(p_x\|p_\theta)$ will be rewritten as something that looks like an average over handfuls.

### Notice

- Estimator $\neq$ the thing estimated. $\bar x$ is a number; $\mathbb{E}[X]$ is a property of $p$.

### Mini-check

1. If you only have samples, can you still talk about $\mathbb{E}[\log p_\theta(X)]$?  
2. Why is this the escape hatch when $p_x$ is unknown?

---

## 4. A convex cup

<a id="p4-convex"></a>

### Purpose for the video

Every legal $f$ in **$f$-divergence** must be **convex** (and left semi-continuous) with $f(1)=0$.

### Definitions

$f$ is **convex** on an interval if the graph never sits *above* a chord:

$$
f(\alpha u_1+(1-\alpha)u_2)\le \alpha f(u_1)+(1-\alpha)f(u_2),\quad \alpha\in[0,1].
$$

(The board writes the same fact with two weights $\alpha_1+\alpha_2=1$.)

$f(1)=0$ is a **normalization**: when the two densities are equal, the ratio is $1$, and $f$ contributes nothing.

**Why the cup matters (preview of his homework).** Jensen says $f(\mathbb{E}[U])\le\mathbb{E}[f(U)]$. For $U=p_x/p_\theta$ under the $p_\theta$-weighted integral, $\mathbb{E}[U]=1$, so $\mathbb{E}[f(U)]\ge f(1)=0$. That is the shape of “$D_f\ge 0$.” You still have to write the proof; this is only why convexity is on the ticket.

### Worked micro

$f(u)=u\log u$ (for $u>0$) is a convex cup that hits $0$ at $u=1$. That single cup will become **KL**.

### Analogy — a hanging chain

A hanging chain sags. Any straight stick between two points of the chain stays *above* the chain. That sag is convexity. $f$-divergences need that sag so Jensen’s inequality can later prove $D_f\ge 0$ (homework).

### Notice

- Convex $\neq$ “positive.” $f$ may go negative; the *integral* $D_f$ will still be $\ge 0$.

### Mini-check

1. Is $f(u)=(u-1)^2$ convex with $f(1)=0$?  
2. Is $f(u)=\sin u$ a legal generator?

---

## 5. Likelihood here means a height

<a id="p5-likelihood"></a>

### Purpose for the video

He **redefines** the word for this course: “likelihood” = **$p$ evaluated at a point**. Not $P(\{x\})$.

### Definitions

If $p$ is a density, $p(\hat x)$ is how high the hill is at $\hat x$.  
He compares $p_x(\hat x)$ to $p_\theta(\hat x)$ to see whether the **model hill** matches the **data hill** at that crumb.

### Worked micro

True hill $p_x$ is tall at “the digit 3.” Model hill $p_\theta$ is almost flat there. He says: likelihood under the truth is high, under the model is low. KL will shout.

### Analogy — two topographic maps of the same town

At the church, map A shows a tall hill, map B shows a puddle. “Likelihood mismatch” means those two heights disagree — not that a church has probability $0.8$.

### Notice

- This is **not** the full statistical slogan “likelihood of the whole dataset.” It is a **pointwise height**.

### Mini-check

1. If $p_x(\hat x)=0.01$ and $p_\theta(\hat x)=4$, which hill is taller?  
2. Is $0.01$ a probability of that exact $x$?

---

## 6. A divergence is not a metric

<a id="p6-div-vs-metric"></a>

### Purpose for the video

He **fumbles on purpose** and then refuses the word **metric**.

### Definitions

A **distance / metric** $d$ obeys: $d\ge 0$; $d=0$ iff same point; **symmetry** $d(a,b)=d(b,a)$; **triangle** $d(a,c)\le d(a,b)+d(b,c)$.

A **divergence** $D(p\|q)$ usually keeps $\ge 0$ and $=0$ iff $p=q$, but **drops symmetry and/or the triangle**. $D(p\|q)$ and $D(q\|p)$ can be different jobs.

### Worked micro

KL$(p_x\|p_\theta)=\int p_x\log(p_x/p_\theta)$. Swap the two densities and you get a **different** number (reverse KL). One-way roads.

### Analogy — walking uphill vs downhill

Town A to town B uphill is not the same effort as B to A. Still useful. Do not call it a ruler.

### Notice

- He still says “how far.” The **feel** is distance; the **axioms** are not.

### Mini-check

1. If $D(p\|q)=0$, what must be true (once we prove the homework)?  
2. Why is “reverse KL” a slightly dishonest name?

---

## 7. Push a known $Z$ through a map

<a id="p7-push"></a>

### Purpose for the video

The primitive sampler is $Z\sim\mathcal{N}(0,I)$ then $\hat x=G_\theta(z)$.

### Definitions

A **deterministic map** $G$ applied to a random $Z$ produces a new random variable $G(Z)$. Its law is the **pushforward** of $Z$’s law. Change $G$ and you change the output law (Jacobians in Tutorial 9).

If $G_\theta$ is a **neural net**, $\theta$ is a huge knob board. **Universal approximation** is the slogan he leans on: a wide-enough net can mimic any continuous $G$ to arbitrary closeness *in theory*. That is why twisting $\theta$ might make $p_\theta$ equal $p_x$ — not because the net magically knows the density.

**Support** of $Z$ = where $Z$ is allowed to land. A continuous onto $G$ cannot create mass **outside** a bounded input support. That is why people start from a **Gaussian** (infinite support).

### Worked micro

$Z$ uniform on $[0,1]$, $G(z)=2z$. Output lives on $[0,2]$, still bounded. You never get $x=10$.

### Analogy — a pasta extruder

Noise $Z$ is the dough. $G_\theta$ is the die. Change the die, change the noodle shape. A die fed only from a *short* dough strip cannot extrude a noodle longer than the physics of that strip.

### Notice

- After training, **sampling** is cheap: draw $Z$, run $G_{\theta^*}$. You never “sample from $p_x$” directly.

### Mini-check

1. Why not start $Z$ as uniform on $[0,1]^k$ if images live on a huge box?  
2. Do outputs of $G_\theta$ *equal* the density $p_\theta$?

---

## 8. Miss a mode versus invent junk

<a id="p8-modes"></a>

### Purpose for the video

Forward KL and reverse KL punish **opposite** sins. That is why the $f$ you pick is not decoration.

### Definitions

**Mode-covering (typical of forward KL).** The model is punished if the **data hill is tall** and the **model hill is short**. It is pressured to put mass on *every* real cluster. It may smear extra junk in the gaps.

**Mode-seeking / no-junk (typical of reverse KL).** The model is punished if it puts mass where the **data hill is short**. It avoids junk, but it may **drop** a whole real cluster.

**Jensen–Shannon** is built to feel like an **average** of those two pressures.

### Worked micro

Data: two blobs, “3”s and “8”s.  
Forward-KL-ish model: covers both, plus some ugly 3–8 hybrids.  
Reverse-KL-ish model: perfect 3s, **zero** 8s.

### Analogy — a cover band

Missing a hit song the crowd wants (mode drop) vs playing a song nobody asked for (junk). A good setlist metric fines **both**.

### Notice

- These are **pointwise height** stories, not full proofs. The lecture uses them as interpretation.

### Mini-check

1. High $p_x$, low $p_\theta$: which failure?  
2. Why might someone still pick reverse KL on purpose?

---

**Second teachers (names only here).** Lilian Weng, StatQuest, Grosse, Hiroaki Hayashi, NannyML, Seeing Theory. The actual pointers live at the end of [NOTES.md](./NOTES.md#external-references) — a handful of well-known items, not a link dump.

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).
