# Lec 12 — Kullback–Leibler (KL) Divergence

**Video:** [Lec 12 Kullback-Leibler (KL) Divergence](https://www.youtube.com/watch?v=ihkGbIdbbxc) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 11 — Entropy](../12-Lec11-Entropy/NOTES.md)

---

## Table of Contents

1. [Topic 1 — Goal: pairwise distance; discrete PMFs](#topic-1-goal-pairwise-distance-discrete-pmfs-0003–0132) (00:03–01:32)
2. [Topic 2 — Cross-entropy H(p,q)](#topic-2-cross-entropy-hpq-0132–0457) (01:32–04:57)
3. [Topic 3 — KL = CE − H; formula](#topic-3-kl--ce--h-formula-0457–0858) (04:57–08:58)
4. [Topic 4 — Properties; not a metric](#topic-4-properties-not-a-metric-0858–1022) (08:58–10:22)
5. [Topic 5 — Continuous KL; board fix; stop](#topic-5-continuous-kl-board-fix-stop-1022–1649) (10:22–16:49)
6. [External references](#external-references)
7. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: turn “distance between two distributions” into a concrete formula for ML recipe step 2. Method: define **cross-entropy** $H(p,q)=\mathbb{E}_p[-\log q]$, then **KL** $D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p)=\mathbb{E}_p[\log(p/q)]$. Fork: KL is non-negative and zero iff $p=q$, but **asymmetric**, so it is a divergence, not a true metric; continuous data use the density twin.

**Worldview arc:** from entropy of **one** law **to** KL between a **pair** $(p,q)$ for training scores.

### System context

```
  ╔══════════════════════════════════╗
  ║ Outside                          ║
  ║ · Lec 11: entropy H(p)           ║
  ║ · Lec 10: recipe needs d         ║
  ╚══════════════╤═══════════════════╝
                 │ this lecture
                 ▼
        ┌────────────────────┐
        │ cross-entropy → KL │
        └────────────────────┘
```

### Main blueprint

```
  true p          model q
    │                │
    │   same sample space / alphabet
    └────────┬───────┘
             ▼
     H(p,q)=E_p[−log q]     cross-entropy
             │
             │  minus H(p)
             ▼
     D_KL(p‖q)=H(p,q)−H(p)
              =∑ p log(p/q)
             │
             ├─ ≥0 ; =0 iff p=q
             ├─ not symmetric → not metric
             └─ continuous: ∫ p log(p/q) dx
             │
             ▼
     recipe step 2: choose d = KL
```

### Scenario walkthrough

1. Need $d(p,q)$ for training.  
2. Samples from true $p$; score with $-\log q$ → cross-entropy.  
3. Subtract true entropy → KL extra surprise.  
4. Check: zero iff match; asymmetric; continuous formula same shape.  
5. Use as divergence in the ML recipe.

### Failure / contrast path

```
  Use H(p) alone as “distance to model”     ──X──► one-law score only
  Subtract H − CE (wrong order)             ──X──► can go negative
  Treat KL as symmetric metric              ──X──► triangle fails
  Plug density height as P(X=x)             ──X──► continuous misread
```

### STOP / out of scope

Full proof of non-negativity (homework); $f$-divergence catalog; optimizing KL in algorithms; differential entropy deep theory.

### Load-bearing claims (closed-book)

- $H(p,q)=\mathbb{E}_p[-\log q]$; $H(p,q)\ge H(p)$; not symmetric.  
- $D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p)=\sum p\log(p/q)$ (truth first).  
- $=0$ iff $p=q$; $\ge 0$ (homework).  
- Not a metric (asymmetry).  
- Continuous: $\int p\log(p/q)\,dx$.  
- Fills recipe step 2; $\arg\min$ KL $\equiv$ $\arg\min$ CE in $\theta$.

**Speaker / course:** NPTEL IISc · MFML · Lec 12.

---

## Topic 1: Goal — pairwise distance; discrete PMFs (00:03–01:32)

### Where this sits on the master map

**SETUP** — why entropy was not enough. Warm-up: [entropy](./PREREQUISITES.md#p1-entropy), [two PMFs](./PREREQUISITES.md#p2-two-pmfs).

### Board / screenshot

![Pair of distributions objective](./screenshots/composites/ch01-topic-01-setup-pair-pmf-panel1of1.png)

**Figure — ~00:03–01:30:** objective pair distance; $p,q$ as mass functions; same sample space.

### What he is establishing

Objective restated: quantify the **distance between a pair of distributions**. Entropy from last lecture gives average information **inside one** distribution. That does not yet compare two laws — so we need a new object.

**Notation discipline:** he previously used $P$ for a probability measure. Now $p$ and $q$ are **mass functions** for **discrete** random variables. Continuous RVs have **differential entropy** later — similar formulas, slightly different interpretation. For now: discrete.

Both laws are defined on the **same sample space** (same set of outcomes).

If you treat entropy as already “the distance,” you skip the pairwise goal. Instead: keep $H(p)$ as a one-law score; build a two-law score next.

You can now state the gap entropy left open. Still missing: the cross-entropy construction.

### Analogy for this topic only

Knowing how mixed a weather climate is (entropy) does not tell you how wrong a **forecast** climate is. You need a score that looks at both.

**What must $p$ and $q$ share?** The same menu of outcomes so $p(x)$ and $q(x)$ are comparable.

In lecture words: need pair distance; discrete mass functions $p,q$.

### Local picture

```
  H(p)  →  one law
  need d(p,q)  →  two laws, same space
  start: discrete PMFs p, q
```

**Notice:** capital $P$ measure vs small $p$ mass/density — he will restate at the end.

### Bridge

How do we score model $q$ when samples actually come from true $p$?

---

## Topic 2: Cross-entropy $H(p,q)$ (01:32–04:57)

### Where this sits on the master map

**CROSS** — middle brick. Warm-up: [cross-entropy](./PREREQUISITES.md#p3-cross-entropy).

### Board / screenshot

![Cross-entropy definition](./screenshots/composites/ch02-topic-02-cross-entropy-panel1of1.png)

**Figure — ~01:32–04:55:** $H(p,q)=\mathbb{E}_p[-\log q]$; wrong-mean micro; not symmetric.

### What he is establishing

Define the expectation of **surprisal under $q$**, averaged under **true** $p$:

$$
H(p,q)=\mathbb{E}_{X\sim p}\big[-\log q(X)\big]
$$

$-\log q(x)$ is the information/surprisal of outcome $x$ **as scored by $q$**. Taking $\mathbb{E}_p$ means: “my samples came from $p$, but I pretend the law is $q$ — what average surprisal do I get?”

**If $p=q$:** this collapses to ordinary entropy $H(p)$.

**If $p\neq q$:** it measures average information that $q$ “contains about” $p$ / the **mistake** of assuming reality is $q$ while it is $p$.

**Why expect CE $\ge$ entropy?** Scoring true data with a wrong codebook ($-\log q$) cannot beat the average surprise of the true codebook ($-\log p$). So $H(p,q)\ge H(p)$, equality only when $q=p$ (homework-level inequality).

**Micro:** true Gaussian mean $0$, model Gaussian mean $3$ — samples from mean $0$, surprisals computed under the mean-$3$ density, then averaged. That average is the cross-entropy story for continuous densities; discrete case is the same logic on a PMF.

**Name:** **cross-entropy** $H(p,q)$.  
**Not symmetric:** $H(p,q)\neq H(q,p)$ by definition (who is truth vs model flips).

If you average $-\log q$ under $q$ instead of under $p$, you get entropy of $q$, not cross-entropy. Instead: expectation always under the **true** law $p$.

You can now write $H(p,q)$ and explain the wrong-model story. Still missing: subtract entropy to get KL.

### Analogy for this topic only

You live in city $p$’s weather. You pack using guidebook $q$. Cross-entropy is how often you are surprised by the real weather when following $q$’s advice.

**If $q$ were perfect ($q=p$), what does that average become?** Ordinary entropy of the true climate.

In lecture words: $H(p,q)=\mathbb{E}_p[-\log q]$; not symmetric.

### Local picture

```
  truth p ──samples──► X
  model q ──score──► −log q(X)
         average under p
              │
              ▼
         H(p,q)  cross-entropy
         H(p,p)=H(p)
```

**Notice:** larger $H(p,q)$ ⇒ model $q$ is a worse “surprise machine” for data from $p$.

### Bridge

How do we turn cross-entropy into a non-negative gap that is zero only when $p=q$?

---

## Topic 3: KL $= \mathrm{CE}-H$; formula (04:57–08:58)

### Where this sits on the master map

**KL DEF** — the divergence itself. Warm-up: [KL](./PREREQUISITES.md#p4-kl), [micro](./PREREQUISITES.md#p8-micro).

### Board / screenshot

![KL divergence definition](./screenshots/composites/ch03-topic-03-kl-definition-panel1of1.png)

**Figure — ~04:57–08:55:** CE vs H; $p\log(p/q)$; KL named; homework properties.

### What he is establishing

Take entropy of $p$ and compare to cross-entropy between $p$ and $q$. The **Kullback–Leibler (KL) divergence** is

$$
D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p)
$$

(using the order that stays non-negative — **cross-entropy minus entropy**; the board’s subtraction order was corrected in class).

**Interpretation:** entropy = average information in $p$. Cross-entropy = average information scored by $q$ on data from $p$. Difference = **extra** bits (or nats) from wrongly using $q$ instead of $p$.

Using log algebra (see [warm-up bridge](./PREREQUISITES.md#p4-kl)):

$$
\begin{aligned}
H(p,q)-H(p)
&=\sum_i p_i\log p_i-\sum_i p_i\log q_i
=\sum_i p(x_i)\log\frac{p(x_i)}{q(x_i)}
=\mathbb{E}_{p}\Big[\log\frac{p(X)}{q(X)}\Big].
\end{aligned}
$$

**Limits:** if $p=q$, KL $=0$. If $p$ and $q$ are far, KL is large.

**Micro (binary, base 2):** true fair coin $p=(1/2,1/2)$, model $q=(3/4,1/4)$ yields $D_{\mathrm{KL}}(p\|q)\approx 0.207$ bits — not zero, because the model is wrong. The **swap** $D_{\mathrm{KL}}(q\|p)\approx 0.189$ bits is different — full expand in the [warm-up](./PREREQUISITES.md#p8-micro).

**Homework (must do with definitions):**  
1. $D_{\mathrm{KL}}(p\|q)\neq D_{\mathrm{KL}}(q\|p)$ in general (asymmetric).  
2. $D_{\mathrm{KL}}\ge 0$.  
3. $D_{\mathrm{KL}}(p\|q)=0$ **if and only if** $p=q$.

We now have the measure we wanted for “how far/close” two distributions are. For training, $\arg\min_\theta D_{\mathrm{KL}}(p\|p_\theta)=\arg\min_\theta H(p,p_\theta)$ because $H(p)$ does not depend on $\theta$.

If you reverse $H(p)-H(p,q)$ without care, the sign can go the wrong way. Instead: **CE minus H**, or write $\sum p\log(p/q)$ directly.

You can now state KL three ways (CE−H, sum, expectation). Still missing: metric status and continuous form.

### Analogy for this topic only

True map entropy is how complex the city is. Cross-entropy is how confusing a **wrong** map is on real streets. KL is the **extra** confusion beyond the city’s own complexity.

**When is extra confusion zero?** Only when the wrong map is actually the true map.

In lecture words: KL $=H(p,q)-H(p)=\sum p\log(p/q)$.

### Local picture

```
  H(p,q)   −   H(p)   =   D_KL(p‖q)
  (CE)        (entropy)   (extra surprise)

  ∑ p log(p/q)

  p=q  ⇒  0
  far  ⇒  large
```

**Notice:** first slot $p$ is the reference (expectation law).

### Bridge

Is this object a mathematical metric — and does the same idea work for continuous data?

---

## Topic 4: Properties; not a metric (08:58–10:22)

### Where this sits on the master map

**PROPS** — what KL can and cannot claim. Warm-up: [not a metric](./PREREQUISITES.md#p5-not-metric).

### Board / screenshot

![KL not a metric](./screenshots/composites/ch04-topic-04-properties-not-metric-panel1of1.png)

**Figure — ~08:58–10:20:** KL as closeness measure; asymmetry blocks metric axioms.

### What he is establishing

KL is **one** measure of how far or close a pair of distributions is — meeting the course objective for a working $d$.

It is **not a metric** in the strictest mathematical sense: it does **not** obey the triangle inequality. Why is that obvious? Because it is **not symmetric**. A true metric must be symmetric; without symmetry, triangle law already fails as a metric package.

Still, KL gives a useful sense of closeness — call it a **divergence**.

If someone insists “all distances are symmetric,” they are talking about metrics, not KL. Instead: use KL as directed “cost of using $q$ for $p$.”

You can now say “divergence not metric” with a reason. Still missing: continuous formula and final notation.

### Analogy for this topic only

One-way travel time city A→B need not equal B→A. Useful for planning; not a symmetric “distance” on a map legend.

**If $d(a,b)\neq d(b,a)$, can $d$ be a metric?** No — metrics require symmetry.

In lecture words: KL useful; not a metric because asymmetric.

### Local picture

```
  metric needs: ≥0, =0 iff equal, symmetric, triangle
  KL has:       ≥0, =0 iff equal, NOT symmetric → not metric
```

**Notice:** “quotes around metric” in Lec 10 now have a concrete example.

### Bridge

What is the continuous version used for densities, and how was the board subtraction fixed?

---

## Topic 5: Continuous KL; board fix; stop (10:22–16:49)

### Where this sits on the master map

**CONT+STOP** — density twin + cleanup. Warm-up: [continuous](./PREREQUISITES.md#p6-continuous), [recipe](./PREREQUISITES.md#p7-recipe).

### Board / screenshot

![Continuous KL and board correction](./screenshots/composites/ch05-topic-05-continuous-notation-stop-panel1of1.png)

**Figure — ~10:22–16:45:** differential CE/H; $\int p\log(p/q)$; CE−H order; capital vs small $p$.

### What he is establishing

For **continuous** RVs (most of our data), a similar definition uses **densities** and **differential** entropy / cross-entropy:

$$
h(p)=-\int p\log p,\qquad
H(p,q)=-\int p\log q,\qquad
D_{\mathrm{KL}}(p\|q)=\int p(x)\log\frac{p(x)}{q(x)}\,dx
$$

(again $H(p,q)-h(p)$ when defined). $p$ and $q$ are density heights, not probabilities of exact points — but the structure matches the discrete sum.

**Reconnect to his Gaussian micro:** samples from a mean-$0$ density, scored with a mean-$3$ density — that average $-\log q$ is continuous cross-entropy; subtract differential entropy of the true density to get continuous KL.

**Class correction (important):** subtraction order and $\log(p/q)$ vs $\log(q/p)$ were cleaned so that  
$D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p)$  
with the cross-entropy term producing $p\log(p/q)$. Cross-entropy is (when comparable) **at least** entropy — so you subtract the smaller from the larger to keep non-negativity. Both “differences” are differences; only **higher minus lower** stays non-negative by definition of the ordering.

**Notation recap:** capital for distribution / mass-function style writing; **small** $p$ for density functions.

**Stop:** continue next lecture. (Light aside: mistakes mark you as human, not AI.)

If continuous data make you abandon KL, you do not need to — replace sums by integrals over densities. Instead: same directed extra-surprise idea.

You can now write discrete and continuous KL and place KL in the ML recipe as step-2 $d$ with truth first, model second.

### Analogy for this topic only

Discrete KL counts extra surprise on a menu of dishes. Continuous KL does the same for a smooth “height map” of probability density over a continuum — like packing for climate $p$ with a weather app tuned to a shifted mean $q$.

**Why order CE − H?** Because CE is the higher (or equal) average surprise; difference stays non-negative.

In lecture words: continuous $\int p\log(p/q)$; CE−H; small $p$ densities; next lecture.

### Local picture

```
  discrete:  ∑ p log(p/q)
  continuous: ∫ p(x) log(p(x)/q(x)) dx

  H(p,q) ≥ H(p)  ⇒  KL = H(p,q)−H(p) ≥ 0

  notation:  P / mass   vs   small p density
  next lecture → continue
```

**Notice:** KL is the concrete $d$ the three-step recipe can minimize over $\theta$ when $q=p_\theta$.

### Bridge

Next lectures apply this $d$ inside training / models.

---

## External references

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| [StatQuest — Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qGfA) | Topics 2–3 | CE as average surprise under a model |
| [StatQuest — KL Divergence](https://www.youtube.com/watch?v=LCslWgIpz4E) | Topics 3–4 | KL as extra surprise / asymmetry |
| [Lec 11 package — Entropy](../12-Lec11-Entropy/NOTES.md) | Topics 1–3 | $H(p)$ and surprisal |
| [Lec 10 package — Recipe](../11-Lec10-Challenges-of-ML/NOTES.md) | Topics 1, 5 | Why $d$ exists in ML |
| [Christopher Olah — Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) | Topics 2–4 | Pictures of cross-entropy / KL ideas |
| [Lec 09 package — Density $p$ vs $P$](../10-Lec09-Density-Function/NOTES.md) | Topic 5 | Densities in continuous KL |

---

## Sources

- Video: [Lec 12 Kullback-Leibler (KL) Divergence](https://www.youtube.com/watch?v=ihkGbIdbbxc)  
- Channel: NPTEL — Indian Institute of Science, Bengaluru  
- Skill: `youtube-lecture-tutor`  
- Captions cleaned via timed transcript / claim sheets  
