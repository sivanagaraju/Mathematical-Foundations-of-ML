# Prerequisites — warm-up before W2_L7 (GAN formulation)

> **Do this first** if “ascent,” “freeze,” “$G(z)$,” or “sample average” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> IITM BS · Week 2 Lecture 7 · Prof. Prathosh.  
> **Beginner:** purpose · definition · micro numbers · analogy · ASCII · notice · mini-check.

He assumes you already know **backprop / gradient descent**. This sitting is **how to train two nets that fight**, not how a derivative works. YouTube title: GAN formulation. Tape: **implementation in practice** (~35 min of dense tablet; treat it as a full hour of procedure).

```
  After this warm-up you can say:

  "G turns noise z into a fake x-hat. The museum never sees z."
  "D scores a file in (0,1); sigmoid of a real-valued net V."
  "An expectation is a population mean; a batch average is what we type."
  "log D on reals, log(1-D) on fakes — that shared score is J_GAN."
  "Freeze means: do not update that net's weights this step (you may still run it)."
  "D climbs J (plus α); G walks downhill (minus α)."
  "We only have n IID files from unknown p_x. No labels."
  "One gradient step ≠ one epoch. Stop by looking at samples, not J=0."
```

```
  §1  Generator G(z)                 ──► Topics 1, 4, 6–7
  §2  Discriminator in (0,1)         ──► Topic 2
  §3  Expectation vs batch average   ──► Topics 3, 6
  §4  log D and log(1-D)             ──► Topics 2–4
  §5  Freeze one net                 ──► Topics 5–7
  §6  Ascent vs descent              ──► Topics 3, 5
  §7  IID data, unknown p_x          ──► Topic 1
  §8  Step vs epoch                  ──► Topics 8–10
```

**One scene through all eight.** A **forger** $G$ and a **detective** $D$. The museum has $n$ real paintings and no catalog of “true art.” The forger starts from random noise $z$ (scribbles). The detective scores “real?” as a number in $(0,1)$. They never train on the same day: freeze one, step the other.

```
  z ~ N(0,1)  →  G_θ  →  x̂          fake canvas   (forger)
  x  from D   →  D_w  →  score in (0,1)            (detective)
                 freeze one, step the other
```

---

## 1. Generator $G(z)$

<a id="p1-gen"></a>

**Purpose.** Fakes are not drawn from a formula $p_\theta(x)$. They are **made**.

**Definition.** A **generator** $G_\theta$ is a net that takes easy noise $z\sim\mathcal{N}(0,I)$ and outputs a file $\hat x=G_\theta(z)$. Those $\hat x$ are samples from $p_\theta$ (the **push-forward** of the noise through $G$). $\theta$ is every weight inside $G$.

**Micro.** Let $z$ be length 2 for a doodle: $z=(0.3,-1.1)$. $G$ is an MLP or a CNN — he does not care today. $\hat x$ has the **same shape** as a real $x$ (same pixels, same length). If real photos are $28\times 28$, so are fakes. The museum never sees $z$; it only sees $\hat x$.

**Analogy.** Three scribbles on the back of a napkin:

- scribble $z_1$ → canvas “almost a vase”
- scribble $z_2$ → canvas “blurry face”
- scribble $z_3$ → canvas “noise soup”

Someone asks: **can you paint a fourth canvas the museum would hang?** You do not have a density $p_\theta(x)$ to sample from. You run $G$ on a new $z_4$.

Wrong: wait for a closed-form $p_\theta$. Right: **make** the file with $G(z)$.

```
  z  (easy noise, e.g. Gaussian)
       │
       ▼
    ┌─────┐
    │ G_θ │   any net (MLP / CNN)
    └─────┘
       │
       ▼
  x̂ = G_θ(z)   same shape as a real x    ~ p_θ
```

**Notice.** When he writes $\hat x_j$, he means $G_\theta(z_j)$. That is how $\theta$ enters $J$. If you freeze $G$, you can still *draw* fakes; you just cannot *improve* them.

**Mini-check.** If you freeze $G$, can you still *draw* fakes? (Yes — you just cannot *improve* them.)

---

## 2. Discriminator in $(0,1)$

<a id="p2-disc"></a>

**Purpose.** Last lecture’s critic $T$ is rewritten as $D$.

**Definition.** $D_w(x)=\sigma(V_w(x))$ with **sigmoid** $\sigma(u)=1/(1+e^{-u})$, so $D\in(0,1)$. He calls this a **binary classifier / discriminator**. Classifier story is the **next** module; today $D$ is just a **score**. $w$ is every weight inside $V$ (and thus $D$).

**Micro.** $\sigma$ never leaves $(0,1)$:

| $V(x)$ | $D=\sigma(V)$ | Read as |
|--------|---------------|---------|
| $+3$ | $\approx 0.95$ | “pretty sure museum” |
| $0$ | $0.50$ | “coin flip” |
| $-3$ | $\approx 0.05$ | “pretty sure forger” |

$D$ cannot output $1.4$. Last layer of $V$ is linear; sigmoid sits on top. Architecture of $V$ can be MLP or CNN.

**Analogy.** A detective who only says a **probability**, not a paragraph.

- museum photo → $0.92$ (“I think this is from the wall”)
- last week’s forgery → $0.11$ (“forger”)
- tonight’s new canvas → $0.48$ (“not sure”)

Someone asks: **is $0.48$ a class label?** No. It is a score. The classifier *interpretation* of that score is next module.

```
  x  ──V_w──►  real number u  ──σ──►  D ∈ (0,1)
                 (any net)           never 1.4
```

**Notice.** $D(\text{real})\to 1$ is good for the detective. $D(\text{fake})\to 0$ is also good for the detective. The forger wants the opposite on fakes.

**Mini-check.** Can $D$ output $1.4$? (No — sigmoid.)

---

## 3. Expectation vs batch average

<a id="p3-batch"></a>

**Purpose.** $J_{\mathrm{GAN}}$ is written with $\mathbb{E}$. Code uses sums.

**Definition.** $\mathbb{E}_{p_x}[h(x)]$ is the **population** mean of $h$ if $x\sim p_x$. We **do not have** $p_x$. We have $n$ files. A **batch average** of size $B$ is $\frac1B\sum_{i=1}^B h(x_i)$ on a **random subset** — not necessarily consecutive files on disk.

**Micro.** Two plates, possibly different sizes.

- $B_1=2$ reals with $\log D = -0.1$ and $-0.3$ → first term $(-0.1-0.3)/2=-0.2$
- $B_2=3$ fakes with $\log(1-D)=-0.4,-0.2,-0.6$ → second term $-1.2/3=-0.4$

$n=1000$, $B_1=64$ → about 16 D-steps to walk the reals once. The last batch can be short. He assumes you know **batch gradient descent**: one batch → one gradient step, not a full epoch.

**Analogy.** You cannot poll every citizen (that would be $p_x$). You poll a handful of museum rooms and a handful of forgeries.

Someone asks: **must the two handfuls be the same size?** He writes $B_1$ and $B_2$ separately. They need not match.

```
  E_{p_x} log D(x)     ≈   (1/B1) Σ_{i=1}^{B1} log D(x_i)
  E_{p_θ} log(1-D(x̂))  ≈   (1/B2) Σ_{j=1}^{B2} log(1-D(x̂_j))

  x_i  = random subset of the n files   (not “rows 1..B1”)
  x̂_j = G(z_j),  z_j ~ N(0,1)
```

**Notice.** The real batch is a **subset** $B_1\subset D$. He says it out loud: “need not be continuous.”

**Mini-check.** Does the real batch have to be consecutive files? (He: no, a random subset of $D$.)

---

## 4. $\log D$ and $\log(1-D)$

<a id="p4-jgan"></a>

**Purpose.** The score both players share.

**Definition.**

$$
J_{\mathrm{GAN}}(\theta,w)=\mathbb{E}_{p_x}[\log D_w(x)]+\mathbb{E}_{p_\theta}[\log(1-D_w(\hat x))]
$$

First term wants $D$ **big** on reals. Second wants $D$ **small** on fakes. **Same** $J$ is **max** for $w$ and **min** for $\theta$.

**Micro.** Natural log. Bigger $D$ on a real $\Rightarrow$ $\log D$ closer to $0$ (less negative). Bigger $D$ on a fake $\Rightarrow$ $\log(1-D)$ a large negative (bad for D, which is what G wants).

| Input | $D$ | Term | Value (approx) | Who likes it? |
|-------|-----|------|----------------|---------------|
| real | $0.9$ | $\log D$ | $\log 0.9\approx -0.11$ | D |
| real | $0.1$ | $\log D$ | $\log 0.1\approx -2.3$ | G (D is failing) |
| fake | $0.1$ | $\log(1-D)=\log 0.9$ | $-0.11$ | D |
| fake | $0.9$ | $\log(1-D)=\log 0.1$ | $-2.3$ | G |

**Analogy.** Detective points for trusting museum pieces **and** for catching fakes. Forger only cares about the **fake** term — he cannot change how the detective scores the wall.

Someone asks: **which term uses $G(z)$?** Only the second. That is why G training drops the first.

```
  J =  (how sure D is on reals)  +  (how sure D is that fakes are fake)
         ↑ independent of θ              ↑ this is G(z) inside D
```

**Notice.** When training G, the first term does not depend on $\theta$ — **drop it**. Putting reals into the G loss is wasted motion.

**Mini-check.** Which term uses $G(z)$? (The second.)

---

## 5. Freeze one net

<a id="p5-freeze"></a>

**Purpose.** You cannot climb and descend the same hill in one simultaneous grab today. He **alternates**.

**Definition.** **Freeze** = do not **update** those weights this step. While D trains, $\theta$ is constant (you may still **run** $G$ to make fakes). While G trains, $w$ is constant (gradients may still **flow through** $D$). Frozen $\neq$ unused. Green on his board = the net being trained. Red = backprop.

**Micro.** Two different “offs”:

| Step | Run G? | Update $\theta$? | Run D? | Update $w$? | Grads through D? |
|------|--------|------------------|--------|-------------|------------------|
| train D | yes (make $\hat x$) | **no** | yes | **yes** (plus) | only inside D |
| train G | yes | **yes** (minus) | yes | **no** | **yes**, all the way into G |

In code this is `requires_grad=False` on the frozen net’s weights, **not** deleting the net.

**Analogy.** Forger paints while the detective’s **grade book is locked**. Next hour the detective studies while the forger’s **hand is locked**. Gradients through a frozen detective still tell the forger how the score would move — the grade book just does not change.

Someone asks: **when training G, do D’s weights change?** No. The canvas still walks through D.

```
  train D:  G frozen (run only)              D updates (green)
  train G:  D frozen (run + pass grads)      G updates (green)

  freeze  =  do not step those weights
  unused  ≠  freeze     (G is used every D step)
```

**Notice.** The trap is `loss.backward()` on a graph that contains both nets with both `requires_grad=True`. That unfreezes the one that should sit still.

**Mini-check.** When training G, do D’s weights change? (No.)

---

## 6. Ascent vs descent

<a id="p6-ascent"></a>

**Purpose.** Most of ML is **minus** $\alpha\nabla$. Discriminator is **plus**.

**Definition.** **Descent:** $w\leftarrow w-\alpha\nabla L$ **shrinks** $L$. **Ascent:** $w\leftarrow w+\alpha\nabla J$ **grows** $J$. D **maximizes** $J$ → **plus**. G **minimizes** $J$ → **minus**. He may say “gradient descent” while writing a plus — the **sign on the board** is the law.

**Micro.** $\alpha_1$ for D, $\alpha_2$ for G; same or different.

Example: scalar $w$, $\nabla_w J=+2$, $\alpha_1=0.1$.

- Ascent (correct for D): $w\leftarrow w+0.2$
- Descent (wrong for D): $w\leftarrow w-0.2$  ← stock `loss.backward()` if you treat $J$ as a loss to shrink

**Analogy.** Detective walks **uphill** on the shared scoreboard. Forger walks **downhill** on the **same** scoreboard. That opposite walk is “adversarial.” They do not walk at the same time (freeze).

Someone asks: **why is there a plus on the $w$ update?** Because D maximizes $J$. Minus would help the forger.

```
  D:  w  ←  w  + α1 ∇_w J     (ascent;  grow J)
  G:  θ  ←  θ  − α2 ∇_θ J     (descent; shrink J)

  board may say “grad descent” next to the plus. Follow the plus.
```

**Notice.** If you copy-paste a descent trainer for D without flipping the sign, D is helping the forger.

**Mini-check.** Why is there a plus on the $w$ update? (D maximizes $J$.)

---

## 7. IID data, unknown $p_x$

<a id="p7-iid"></a>

**Purpose.** The only input is a pile of files.

**Definition.** $D=\{x_1,\ldots,x_n\}$ drawn **IID** (independent and identically distributed) from unknown $p_x$. No labels $y$. No formula for $p_x$. Goal of the course recipe: make $G$ such that $G(z)$ looks like a **new** draw from $p_x$ — not a replay of a training file.

**Micro.** $n$ museum paintings. You never see the “true art law.” Architecture of $G$ and $D$ is **your** choice; this lecture does not pick CNN vs MLP (architecture-agnostic). Tutorials pick later.

**Analogy.** A closed museum: $n$ works on the wall. The forger must match that wall, not a textbook density.

- painting 1, painting 2, …, painting $n$ — unlabeled
- someone asks: **what is $p_x(x)$ at this new canvas?** You do not have it. You train $G$ so new $G(z)$ could have hung on the wall.

Wrong: need class labels. Right: unsupervised pile.

```
  Input:  D = {x1,...,xn} ~ iid p_x   (unknown; no y)
  Want:   G_θ(z)  ≈  a new museum piece   (not a copy of x_i)
  Nets:   G and D = any NN (MLP / feed-forward / CNN)
```

**Notice.** IID across files, not across pixels. One image is one $x$.

**Mini-check.** Do we need $y$-labels? (No.)

---

## 8. One gradient step vs an epoch

<a id="p8-epoch"></a>

**Purpose.** He writes **one step**, then says train many **epochs**.

**Definition.** **Gradient step:** one batch, one update. **Epoch:** one pass over the real data (many D-steps). G-steps do not even use reals, so “epoch” is a D-side word. Stopping: **not** a formula — look at generated quality / **metrics**. Step counts need not be 1:1 (five G and one D, or the reverse). That $k$:1 depends on the use case.

**Micro.** $n=1000$, $B_1=64$ → about $1000/64\approx 16$ D-steps per epoch if you cycle the reals once. You might do $k=5$ D-steps per G-step, or the reverse. There is **no** well-defined $J=0$ halt.

**Analogy.** One practice kick (step) vs walking the whole drill book (epoch). You may give the forger five kicks per detective quiz. You stop when the wall **looks** right, not when a number hits zero.

Someone asks: **is “one G step” the same as “one epoch”?** No.

```
  for epoch in range(E):          # walk the real pile once per epoch
      for _ in range(k_d):        # often k_d = 1; sometimes 5
          D-step                  # uses a real batch
      for _ in range(k_g):        # often k_g = 1; sometimes 5
          G-step                  # uses NO reals
  # STOP: look at x̂ / metrics. Next module: classifier view of “when.”
```

**Notice.** Next module: classifier interpretation, how to **stop**, improvisations, **inference / conditional**. This tape ends **training**, not sampling-with-a-prompt.

**Mini-check.** Is “one G step” the same as “one epoch”? (No.)

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
