# Prerequisites — warm-up before Lec 18 (Wasserstein GAN)

> **Do this first** if “manifold,” “Lipschitz,” “coupling,” or “earth mover” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues [Lec 05 GANs](../28-Lec05-Generative-Adversarial-Networks/NOTES.md).  
> **Beginner:** purpose · definition · micro numbers · analogy · ASCII · notice · mini-check.

Last hour’s GAN / VDM **saddle** can **saturate**: a perfect inspector leaves the forger with no slope. This hour changes the **yardstick** — Wasserstein / earth-mover — so the yardstick still sees *how far* two piles are even when they do not overlap. Every idea below unlocks a map word. None of them is the lecture.

```
  After this warm-up you can say:

  "A saddle still has one player maxing and one mining the same score."
  "Support = where the pile actually sits, not the whole ambient R^d."
  "A manifold is a low-dim sheet in a high-dim room (a thread in R^3)."
  "Lipschitz-1 means the slope never exceeds 1."
  "A joint with given row/column totals is a coupling / transport plan."
  "Work = mass moved times distance moved."
  "A min problem can have a dual that is a max over functions."
  "Weight-norm = shrink the critic’s weights so that slope stays legal."
```

**Symbol card (unlocks the map).** Keep this next to NOTES.

| Word / Symbol | Picture / Formal Concept | This hour’s job | Dedicated MathsTerm Guide |
| :--- | :--- | :--- | :--- |
| **$p_x$** | Real pile (training images) | The data distribution you wish to match | [Common Probability Distributions](../../../MathsTerms/Common_Probability_Distributions.md) |
| **$p_\theta$ / $p_{\hat x}$** | Fake pile from $G_\theta(z)$ | The generative push-forward law you can sample | [Autoregressive Models](../../../MathsTerms/Autoregressive_Models.md) |
| **$\text{Support}$** | Grass that actually has sand | The subset $\mathcal{X} \subset \mathbb{R}^d$ where density is non-zero | [Autoencoders & Latent Spaces](../../../MathsTerms/Autoencoders_and_Latent_Spaces.md) |
| **$\text{Manifold}$** | A thread / sheet in a 3D room | Low-dimensional geometry embedded in high-D ambient space | [Autoencoders & Latent Spaces](../../../MathsTerms/Autoencoders_and_Latent_Spaces.md) |
| **$D_f$** | $f$-divergence (KL, JSD, …) | Classical divergence yardstick; saturates to $\ln 2$ on disjoint supports | [f-Divergence](../../../MathsTerms/f_Divergence.md) |
| **$W_1(P, Q)$** | Wasserstein / Earth-Mover's Distance | Minimal shovel work $\inf_\gamma \mathbb{E}[\|x - y\|]$ between piles | [Wasserstein Distance & EMD](../../../MathsTerms/Wasserstein_Distance_and_EMD.md) |
| **$\pi / \gamma$** | Transport plan = joint coupling table | Conservation-of-mass joint distribution coupling $P$ and $Q$ | [Joint, Marginal & Conditional Dist](../../../MathsTerms/Joint_Marginal_Conditional_Dist.md) |
| **$T_w / D_w$** | 1-Lipschitz Critic Net | Scalar regression potential landscape; slope bounded by 1.0 | [Lipschitz Continuity](../../../MathsTerms/Lipschitz_Continuity.md) |
| **$\text{Saddle}$** | $\min_\theta \max_w J(\theta, w)$ | Minimax optimization seeking Nash equilibrium on zero-sum landscape | [Minimax Games & GANs](../../../MathsTerms/Minimax_Game_and_GANs.md) |

```
  §1  Saddle / minmax                 ──► Topics 1, 8–9
  §2  Support of a law                ──► Topics 3–4
  §3  Manifold as a thin sheet        ──► Topic 2
  §4  Perfect separator               ──► Topic 3
  §5  Two spikes that do not overlap  ──► Topic 4
  §6  Joint = transport table         ──► Topics 5–6
  §7  Work = mass × distance          ──► Topic 7
  §8  1-Lipschitz / bounded slope     ──► Topics 8–9
```

---

## 1. Two nets, one score: a saddle

<a id="p1-saddle"></a>

### Purpose

Wasserstein GAN is still **not** “train a classifier and stop.” Inner: **max** a score in the critic. Outer: **min** the **same** score in the sampler. That shared point is a **saddle**.

### Definitions

$$
\theta^*,w^*=\arg\min_\theta\max_w J(\theta,w).
$$

Ordinary training **avoids** saddles. This family **seeks** them. Last hour’s $J$ used $f^*$; this hour’s $J$ will be $\mathbb{E}T-\mathbb{E}T$ over **1-Lipschitz** $T$.

### Micro numbers

Toy $J=w^2-\theta^2$ near $(0,0)$: climb $w$, sit in $\theta$. Horse saddle, not a bowl.

**Why a huge net almost never sits in a bowl.** In 1-D, a max needs $f''<0$. In many dimensions you look at the **Hessian** (the matrix of second derivatives) and at the **signs of its eigenvalues**. A true minimum needs **every** eigenvalue positive (**positive definite**), not “mostly positive.” Pretend each sign is a coin with $P(+)=0.99$. For a billion-parameter net you need about a billion same-sign coins at once: $0.99^{10^9}$ is tiny. Ordinary training already cannot hunt a global min; a **saddle** is worse, because one player is climbing while the other is sitting. That is why people stop on a **surrogate** (sample quality), not on “did both losses go down.”

### Analogy

A horse saddle: flaps go **up** one way, horse goes **down** the other. One height, opposite jobs. **If you only watch “did both riders’ numbers go down?” you will wait forever** — one rider is supposed to climb. That is why GAN plots of “$D$ loss and $G$ loss both decreasing” are the wrong dashboard.

### Local picture

```
              max_w (critic T)
                    ▲
   min_θ  ──────────┼──────────
   (sampler G)      ▼
  J shared; ordinary opt AVOIDS saddles; here we SEEK one
```

### Notice

Instability of last hour is **because** it is a saddle in a trillion-parameter space — not because someone “tuned badly.”

### Mini-check

1. Which player maximizes $J$?  
2. Why is “seek a saddle” a warning?  
3. Does changing the yardstick from $f$-div to Wasserstein remove the saddle?

---

## 2. Support: where the pile actually sits

<a id="p2-support"></a>

### Purpose

A density written on all of $\mathbb{R}^d$ can still put **zero** mass on most of $\mathbb{R}^d$. **Support** = the set of points that actually get mass.

### Definitions

The **support** of $p$ is (informally) where $p>0$. Two laws are **perfectly aligned** (his phrase) if there is a bijection between their supports. If not, they sit on different sheets.

### Micro numbers

Fair die: support $\{1,2,3,4,5,6\}$, not all of $\mathbb{R}$. You can *write* a density on the whole line; the die still never lands on $7$ or $\pi$.

Two **Dirac** spikes: one at $0$, one at $3$ — supports $\{0\}$ and $\{3\}$, **not** aligned. A Dirac at $a$ is “all the sand in one grain at $a$.” If the grains sit on different spots, a threshold between them is a perfect classifier — no matter whether the gap is $3$ or $300$.

### Analogy

Two sand piles on a football field. The field is $\mathbb{R}^d$. The **support** is the two small patches of grass that actually have sand. Most of the field is empty.

### Local picture

```
  ambient R^d     ................................
  support of p_x     [###]
  support of p_θ                    [***]
  not aligned  →  a fence can sit between them
```

### Notice

$f$-divergence was defined as if both laws live on all of $\mathbb{R}^d$. Manifold hypothesis says real data **do not**.

### Mini-check

1. Is support the same as ambient dimension?  
2. Two spikes at $0$ and $3$: aligned or not?  
3. Why would a fence between them be a perfect classifier?

---

## 3. A manifold is a thin sheet in a big room

<a id="p3-manifold"></a>

### Purpose

**Manifold hypothesis:** practical data (MNIST digits, faces) sit on a **very thin** sheet inside a huge ambient cube. Not a theorem — a **conjecture without a proof** (his definition of hypothesis).

### Definitions

Rough picture: a **manifold** is a lower-dimensional sheet in a vector space. It is **not** a linear subspace (a thread is curved). A thread in the room $\approx$ 1-D manifold in $\mathbb{R}^3$. A paper of zero thickness $\approx$ 2-D manifold in $\mathbb{R}^3$.

### Micro numbers

MNIST: $28\times 28=784$ binary pixels. Coin-toss every pixel independently: $2^{784}$ possible grids — about $10^{236}$, far more than atoms in the observable universe. MNIST’s $60{,}000$ training digits occupy a **tiny** subset. Odds of a random grid being a handwritten $7$ are essentially zero. Same moral as infinite monkeys typing Shakespeare: probability **non-zero**, practically never.

### Analogy

Infinite monkeys at a typewriter: non-zero chance of Shakespeare, practically never. The **effective** space of English is a thin sheet in the space of all key-smash.

### Local picture

```
  ambient: all 28×28 binary grids   {0,1}^784   (huge)
  MNIST digits                      a thin curved sheet
  coin-toss filling pixels          almost never lands on the sheet
```

### Notice

He will **not** prove the hypothesis. He uses it to explain why a perfect discriminator exists.

### Mini-check

1. Is a manifold the same as a linear subspace?  
2. Why $28\times 28$ in the thought experiment?  
3. Hypothesis means proved, or not proved?

---

## 4. A perfect separator can kill the slope

<a id="p4-perfect-d"></a>

### Purpose

If two piles do not sit on the same sheet, you can often draw a fence that classifies **every** point right. Then the forger gets **no gradient**. That is **saturation**.

### Definitions

**Perfect discriminator:** a $T$ (or $D$) with **100%** accuracy on real vs fake. Then (homework this hour) the $f$-divergence **does not depend on** $\theta$. No signal for $G$.

Last hour’s **not $1{:}1$** (e.g. train $G$ five times, $D$ once) was a **hyperparameter** to avoid this, not a moral.

### Micro numbers

Two spikes on a line. A threshold between them is a perfect classifier **no matter how far** the spikes are, as long as they do not sit on the **same** point.

**$5{:}1$ is a knob.** Last hour: train $G$ five times, $D$ once, so $D$ does not become that fence. It is **not** a theorem. If you still hit $100\%$ $D$, $\theta$ has dropped out anyway.

### Analogy

Two flocks with a fence between them. The inspector is always 100% right. The forger can move the fake flock a mile and the inspector is **still** 100% right — no “warmer / colder.”

### Local picture

```
  p_x  X X X     |fence|     o o o  p_θ
                 100% D
  move o's farther  →  D still 100%  →  G has no slope
```

### Notice

Wasserstein will be built so the **distance of the fence** still counts.

### Mini-check

1. What happens to $G$ if $D$ is 100%?  
2. Why was GAN training not $1{:}1$?  
3. Is that ratio a theorem or a hyperparameter?

---

## 5. Two spikes: overlap vs how far

<a id="p5-dirac"></a>

### Purpose

$f$-div (KL, JSD, …) on two Diracs is **$0$ or “maxed out / $\pm\infty$”**, **independent of the gap $\theta$**. A better yardstick should read **how many metres of dirt** you must move.

### Definitions

A **Dirac** at $a$ is a spike: all mass at one point $a$. Two Diracs at $0$ and $\theta$: supports misaligned unless $\theta=0$.

### Micro numbers

Gap $\theta=1$ vs $\theta=100$. JSD (once they do not overlap) does **not** grow with $100$. Earth-mover work **does** — moving a tonne one metre is not moving it a hundred metres.

### Analogy

Two sandcastles on a beach. “Do they occupy the same footprint?” is a yes/no. “How far must I shovel the sand?” is a number. $f$-div asked the yes/no. Wasserstein asks the shovel.

### Local picture

```
  p_x:  spike at 0
  p_θ:  spike at θ

  f-div (KL/JSD):  maxed / independent of θ   (once θ ≠ 0)
  Wasserstein:     grows with |θ|             (homework this hour)
```

### Notice

Homework: **show** practical $f$-divs max out on two Diracs. He does not grind the algebra in class.

### Mini-check

1. If $\theta$ doubles, does JSD necessarily double?  
2. What question does earth-mover ask that JSD does not?  
3. Are two Diracs perfectly aligned?

---

## 6. A joint table is a transport plan

<a id="p6-coupling"></a>

### Purpose

There are **many** ways to shovel pile $A$ into pile $B$. Each way is a **table** of “how much mass left $x$ and arrived at $\hat x$.” That table **is** a **joint** whose **margins** are the two piles. Name: **coupling** / **transport plan**.

### Definitions

A **joint** $\pi(x,\hat x)$ with $\int \pi\,d\hat x=p_x$ and $\int \pi\,dx=p_{\hat x}$. **Marginalization** = sum a row or column of the table (literally the *margin*).

### Micro numbers

**One legal plan (only choice).** Two bins. $p=(1,0)$ all mass at bin 1. $q=(0,1)$ all mass at bin 2. The only plan: move **all** mass from 1 to 2. Table:

```
           dest 1   dest 2   | row = p
  src 1      0        1      | 1
  src 2      0        0      | 0
  col sums   0        1        = q
```

**Many legal plans.** $p=(0.5,0.5)$, $q=(0.5,0.5)$. You may leave everything in place (diagonal $0.5,0.5$), or swap everything, or send $0.3$ across and keep $0.2$ — infinitely many tables with the same row/column totals. Each table is one shovel scheme.

### Analogy

Shipping crates from warehouses to stores. Each spreadsheet of “warehouse $i$ sends this many tonnes to store $j$” is a plan. Row totals = warehouse stock. Column totals = store demand.

### Local picture

```
           dest x̂1   x̂2   | row = p_x
  src x1    π11     π12   | p_x(x1)
      x2    π21     π22   | p_x(x2)
      ---------------------
      col   q(x̂1)   q(x̂2)

  infinitely many π with those margins
  each π = one way to shovel
```

### Notice

Not yet “optimal.” First: **every** plan is a joint. Optimality is the **least-work** plan.

### Mini-check

1. What must row sums equal?  
2. Is there only one transport plan?  
3. Why the name “marginalization”?

---

## 7. Work = mass times distance

<a id="p7-work"></a>

### Purpose

High-school physics: work $\approx$ (how much you move) $\times$ (how far). Average work under a plan $\pi$ is $\mathbb{E}_\pi[\|x-\hat x\|]$. **Earth-mover** = you are moving dirt. **Wasserstein** = the **least** such average work.

### Definitions

$$
W(p,q)=\min_{\pi\in\Pi(p,q)}\mathbb{E}_\pi[\|x-\hat x\|].
$$

$\Pi(p,q)$ = joints with those margins. $p$-norm in the cost $\to$ $W_p$. He writes $W_2$ with $\|\cdot\|_2$.

### Micro numbers

Move $1$ tonne a distance $3$: work $3$. Move $0.5$ tonnes a distance $2$: work $1$. Average over the plan.

**Two Diracs, one legal plan.** Pile $A$: all mass $1$ at $0$. Pile $B$: all mass $1$ at $4$. The only table: send $1$ across distance $4$. Least work $=4$. Move $B$ to $8$: least work $=8$. The bill **tracks the gap**. JSD would have already maxed out at gap $4$ and stayed maxed at $8$.

### Analogy

You must flatten hill $A$ into hill $B$. Many truck routes. The earth-mover distance is the **cheapest** truck bill, not the average of all bills, and not a yes/no “are they the same hill.”

### Local picture

```
  one shipment:  mass π(x,x̂)  ×  distance ||x−x̂||  =  work of that cell
  average work of a plan = E_π[ ||x−x̂|| ]
  Wasserstein = MIN average work over legal plans
```

### Notice

Closer piles $\Rightarrow$ smaller **least** work. That is why $W$ is a meaningful **how far**.

### Mini-check

1. Why “earth mover”?  
2. Min over what?  
3. Does $W$ ignore distance once piles do not overlap?

---

## 8. Slope at most 1: Lipschitz-1

<a id="p8-lipschitz"></a>

### Purpose

The **dual** of Wasserstein is a **max** over functions $T$ whose **slope never exceeds 1**. Those are **1-Lipschitz** functions. WGAN’s critic must be one of those. Practice: **normalize weights** after each step so $\|w\|_2=1$ at each layer.

### Definitions

$T$ is **1-Lipschitz** if

$$
\frac{|T(x_1)-T(x_2)|}{\|x_1-x_2\|}\le 1
$$

for all distinct $x_1,x_2$ (derivatives bounded). Kantorovich–Rubinstein (named, **not proved** today):

$$
W(p_x,p_\theta)=\max_{\|T\|_{\mathrm{Lip}}\le 1}\bigl(\mathbb{E}_{p_x}[T]-\mathbb{E}_{p_\theta}[T]\bigr).
$$

No $f^*$ in this dual. Still a minmax on two nets.

### Micro numbers

$T(x)=0.5x$ on $\mathbb{R}$: slope $0.5\le 1$, legal. $T(x)=3x$: slope $3$, illegal for this dual.

**Why dual at all?** The primal $W=\min_\pi \mathbb{E}_\pi[\|x-\hat x\|]$ searches over **tables**. You do not know $p_x$ as a formula, so you cannot list tables in training. Duality (named **Kantorovich–Rubinstein**, **not proved** today) swaps that min over tables for a **max over functions** $T$ whose slope is at most $1$. Those two expectations of $T$ you *can* estimate from samples — same trick as last hour, **without** $f^*$.

### Analogy

A road whose grade is never steeper than 45° (slope 1). The critic is allowed to be any such road. If a layer’s weights blow up, the grade exceeds 1 — rescale.

**Primal vs dual in one picture.** Primal: search **spreadsheets** of dirt shipments (you cannot list them in $\mathbb{R}^{784}$). Dual: search **roads of slope $\le 1$** and read the height gap between the two piles. Same number $W$. Training uses the dual because two sample averages of $T$ are easy.

### Local picture

```
  W = max_{T 1-Lip} ( E_{p_x} T − E_{p_θ} T )

  practice: after every G/D step
            rescale critic weights so ||w||_2 = 1 at each layer

  conclusion he writes: WGAN more STABLE than naive GAN
```

### Notice

He does **not** prove Kantorovich–Rubinstein. Read the paper. Inversion and FID were **promised at minute 1** and **not taught** this hour.

### Mini-check

1. What extra constraint does WGAN’s $T$ have that vanilla GAN’s $D$ did not?  
2. How does he enforce it in a net?  
3. Did this sitting finish GAN inversion?

---

**Second teachers (names only here).** Original WGAN: Arjovsky, Chintala, Bottou 2017. OT textbook: Villani / Peyré–Cuturi. Kantorovich–Rubinstein duality. Original GAN + Nowozin $f$-GAN (he assigns both). Pointers (video + notes, 2–3 per topic) live at the end of [NOTES.md](./NOTES.md#external-references).

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
