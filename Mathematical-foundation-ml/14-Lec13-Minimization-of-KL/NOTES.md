# Lec 13 — Minimization of KL Divergence

**Video:** [Lec 13 Minimization of KL Divergence](https://www.youtube.com/watch?v=Ij4p5hLbfo4) · NPTEL / IISc · ~24:51  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 12 — KL Divergence](../13-Lec12-KL-Divergence/NOTES.md)

---

## Table of Contents

1. [Topic 1: Recipe return; IID data; general V](#topic-1-recipe-return-iid-data-general-v-0003-0305) (00:03–03:05)
2. [Topic 2: Model $p_\theta$; continuous KL; $\theta^\star$ argmin](#topic-2-model-p_theta-continuous-kl-theta-argmin-0305-0530) (03:05–05:30)
3. [Topic 3: Drop entropy of data](#topic-3-drop-entropy-of-data-0505-0640) (05:05–06:40)
4. [Topic 4: Unknown $p_V$; expectation form](#topic-4-unknown-p_v-expectation-form-0612-0750) (06:12–07:50)
5. [Topic 5: Law of large numbers; sample mean](#topic-5-law-of-large-numbers-sample-mean-0742-1210) (07:42–12:10)
6. [Topic 6: Min KL estimator; forward vs reverse; LOTUS](#topic-6-min-kl-estimator-forward-vs-reverse-lotus-1210-1700) (12:10–17:00)
7. [Topic 7: Likelihood; MLE equals min KL](#topic-7-likelihood-mle-equals-min-kl-1700-2451) (17:00–24:51)
8. [External references](#external-references)
9. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: turn KL from Lec 12 into a **trainable** objective for $\theta$.  
Method: minimize forward $D_{\mathrm{KL}}(p_V\|p_\theta)$, drop entropy of data, replace $\mathbb{E}_{p_V}[\log p_\theta(V)]$ by a sample average under **IID**, and read the result as **MLE**.  
Fork: the same $\theta^\star$ is both the minimum-KL estimator and MLE; reverse KL is not used because it leaves unknown $p_V$ uncomputable.

**Worldview arc:** from “KL is a distance we can write” **to** “training = minimize KL with data ≈ maximize log-likelihood.”

### System context

```
  ╔════════════════════════════════════╗
  ║ Outside                            ║
  ║ · Lec 12: D_KL(p‖q) defined        ║
  ║ · Recipe: need d between p and p_θ ║
  ╚══════════════╤═════════════════════╝
                 │ this lecture
                 ▼
        ┌────────────────────────┐
        │ min_θ KL → MLE algebra │
        └────────────────────────┘
```

### Main blueprint

```
  ┌─────────────────────────────────────┐
  │ D = {v1..vn}  ~iid  p_V             │
  │ V dummy: X, (X,Y), Y|X, …           │
  └─────────────────┬───────────────────┘
                    ▼
  ┌─────────────────────────────────────┐
  │ model family p_θ(v)  (evaluable)    │
  └─────────────────┬───────────────────┘
                    ▼
  θ* = argmin_θ  D_KL(p_V ‖ p_θ)
                    │ open
                    ▼
  ∫ p_V log p_V   −   ∫ p_V log p_θ
   [drop: no θ]         [keep]
                    │
                    ▼
  argmax_θ  E_{p_V}[ log p_θ(V) ]
                    │ weak LLN + IID
                    ▼
  argmax_θ  (1/n) Σ_i log p_θ(v_i)
           ┌────────┴────────┐
           ▼                 ▼
   min KL estimator    MLE (same math)
   L(D)=∏ p_θ(v_i) → log L = Σ log p_θ
```

### Scenario walkthrough

1. You have $n$ IID samples from an unknown law $p_V$ (images, pairs $(x,y)$, … — $V$ is a dummy).  
2. You pick a model family $p_\theta$ you can evaluate.  
3. You want $\theta$ that makes $p_\theta$ close to $p_V$ in **forward KL**.  
4. Expand KL: entropy of data does not depend on $\theta$ → drop it.  
5. You still cannot integrate against unknown $p_V$ → rewrite as expectation → **sample mean of $\log p_\theta(v_i)$**.  
6. Maximizing that average is MLE; the story is still minimum KL.

### Failure / contrast path

```
  Try reverse KL D(p_θ ‖ p_V)           ──X──► p_θ log p_V term unusable
  Plug unknown p_V into the integral    ──X──► begging the question
  Use non-IID / tiny n as if exact E    ──X──► LLN not helping
  Treat MLE as a different objective    ──X──► same math as min KL here
  Forget density-at-point = likelihood  ──X──► cannot read Topic 7
```

### STOP / out of scope

- How to design the architecture of $p_\theta$  
- Gradient algorithms / optimizers  
- Full proof of LOTUS (named; proof deferred)  
- When reverse KL is preferred in generative models (only contrast)

### Load-bearing claims

1. Goal: estimate $p_V$ given $D=\{v_i\}$ under IID.  
2. $\theta^\star=\arg\min_\theta D_{\mathrm{KL}}(p_V\|p_\theta)$.  
3. Drop $\int p_V\log p_V$ (entropy of data).  
4. Remaining piece is $\mathbb{E}_{p_V}[\log p_\theta(V)]$.  
5. Sample mean $\frac1n\sum\log p_\theta(v_i)$ under IID (weak LLN).  
6. Forward KL, not reverse, for computability.  
7. Likelihood = density at a point; $L(D)=\prod p_\theta(v_i)$.  
8. MLE $\equiv$ minimum KL estimator under this derivation.

**Speaker / course:** NPTEL — Indian Institute of Science, Bengaluru · Mathematical Foundations track (follows KL lecture).

---

## Topic 1: Recipe return; IID data; general V (00:03-03:05)

### Where this sits on the master map

This is the **SETUP** box. Lec 12 gave a distance between densities. Now we walk back into the machine-learning recipe and install the data object the rest of the lecture will optimize on. You need a clean picture of [IID samples](./PREREQUISITES.md#p6-lln) and of $V$ as a dummy variable.

### Board / screenshot

![Board: N IID samples, dataset D, goal estimate p_V](./screenshots/composites/ch01-topic-01-recipe-data-setup-panel1of1.png)
**Figure — ~01:50–02:59 (board-focused):** handwritten setup — “Suppose we have N samples drawn IID,”  
$D=\{v_1,\ldots,v_N\}\sim\mathrm{iid}\,p_V$, and **Goal: estimate $p_V$ given $D$**.

### What he is establishing

Lec 12 answered “how do we score two distributions?” With a usable score (KL), the course returns to the **machine learning recipe**: given data, estimate the unknown law that produced them.

There are two equivalent ways to say what “data” is. One: a single random variable and many draws from its distribution. Two: $n$ random variables, one realization each, that are **independent and identically distributed (IID)**. The lecture sticks to the second view for the whole derivation, because the IID assumption is used throughout machine learning practice.

That methodology does not care whether your problem is supervised or unsupervised — labels may exist or not. It also does not care whether you are estimating a **marginal**, a **joint**, or a **conditional**. The algebra will look the same.

So the board introduces a dummy letter $V$ and a dataset

$$
D=\{v_1,v_2,\ldots,v_n\},\qquad v_i\stackrel{\text{iid}}{\sim} p_V.
$$

Here $p_V$ can mean $p_X$, $p_{X,Y}$, $p_{X\mid Y}$, $p_{Y\mid X}$, or whatever the task needs. The goal of the recipe, stated once for all those cases, is:

> estimate $p_V$ given $D$.

Micro scene: three projects share one notebook line. Project A records only images $X$. Project B records pairs $(X,Y)$. Project C records labels given images. Wrong move: invent three unrelated training theories before the common core exists. Right move: keep one dummy $V$ and one goal until the estimator exists; only then specialize what $V$ stands for.

You can now write the data object and the goal in one line. Still missing: how the model and the KL score enter the recipe.

### Analogy for this topic only

Think of a lab notebook that always records $n$ independent trials of “the same kind of experiment,” but the experiment type changes by project:

- Project A: only images $X$  
- Project B: pairs $(X,Y)$  
- Project C: labels given images, $Y\mid X$

Someone asks: **what is the common mathematical goal?**

It is not “build three unrelated algorithms.” It is: from $n$ IID outcomes, estimate the law of the thing you recorded — whatever that thing is.

If you treat each project as a new universe with no shared structure, you will re-derive the same estimator three times. In lecture words: dummy $V$, dataset $D$, true density $p_V$.

### Local picture

```
  supervised? unsupervised?   marginal / joint / conditional?
              \                      /
               \                    /
                v                  v
              ┌──────────────────────┐
              │  D = {v1..vn} ~iid   │
              │       p_V            │
              │  goal: estimate p_V  │
              └──────────────────────┘
                         │
                         │  V is a dummy
                         ▼
              p_X | p_{X,Y} | p_{Y|X} | …
```

**Notice:** generality is installed *before* formulas so later KL algebra is not rewritten for every task.

### Bridge

You have $D$ and a goal “estimate $p_V$.” The recipe next needs a **parametric stand-in** $p_\theta$ and a score that says how far $p_\theta$ is from $p_V$ — that score is KL, and optimization over $\theta$ is the next box.

---

## Topic 2: Model $p_\theta$; continuous KL; $\theta^\star$ argmin (03:05-05:30)

### Where this sits on the master map

**OBJECTIVE** box. Setup is done; now install the [model family](./PREREQUISITES.md#p2-model) and the [KL](./PREREQUISITES.md#p3-kl) minimization problem that defines “best $\theta$.”

### Board / screenshot

![Board: model density, continuous KL, theta star argmin](./screenshots/composites/ch02-topic-02-kl-objective-panel1of1.png)
**Figure — ~03:16–05:18:** assume $p_\theta(v)$ as model density;  
$D_{\mathrm{KL}}(p_V\|p_\theta)=\int p_V(v)\log\frac{p_V(v)}{p_\theta(v)}\,dv$;  
$\theta^\star=\arg\min_\theta D_{\mathrm{KL}}$; opening into two integrals with the first term about to be struck.

### What he is establishing

Assume a **model density** $p_\theta$ that you can evaluate at any point $v$ (the dummy). The recipe’s next line is: form the Kullback–Leibler divergence between the true law and the model,

$$
D_{\mathrm{KL}}(p_V\|p_\theta).
$$

For continuous random variables with values in a $d$-dimensional real space (the “most general observation assumption” he uses here), that divergence is

$$
D_{\mathrm{KL}}(p_V\|p_\theta)
=\int p_V(v)\,\log\frac{p_V(v)}{p_\theta(v)}\,dv.
$$

(Auto-captions scramble the symbol order; the continuous form above matches the board and Lec 12.)

What we want is not the number KL for a fixed $\theta$, but the **best parameters**:

$$
\theta^\star
=\arg\min_\theta\,
D_{\mathrm{KL}}(p_V\|p_\theta).
$$

Open the logarithm:

$$
D_{\mathrm{KL}}(p_V\|p_\theta)
=\int p_V(v)\log p_V(v)\,dv
-\int p_V(v)\log p_\theta(v)\,dv.
$$

Worked micro of the *shape* (not a claim that these numbers appear on the board): if two candidate models give KL values $0.8$ and $0.3$ to the same truth, the recipe prefers the second — smaller forward KL. That ranking is exactly “argmin over $\theta$.”

Wrong move: treat $\theta^\star$ as “whatever optimizer Python returns” without writing the population objective first. Right move: write the KL objective in $\theta$, then ask which pieces can actually be computed from data.

You can now state the formal training goal as minimum forward KL. Still missing: why the first integral can be ignored, and how the second is estimated when $p_V$ is unknown.

### Analogy for this topic only

You design a thermostat family with a single knob $\theta$ (setpoint). Outdoor truth is a temperature distribution $p_V$ you do not control. Each knob setting defines a model “comfort law” $p_\theta$.

Question: **which knob makes the model law closest to outdoor truth under KL?**

That question *is* $\arg\min_\theta D_{\mathrm{KL}}(p_V\|p_\theta)$. Picking a knob because the housing looks nice is not solving the objective. In lecture words: model $p_\theta$, optimize $\theta$ by KL.

### Local picture

```
   p_V (unknown truth)     p_θ (model you control)
            \                   /
             \                 /
              v               v
         D_KL( p_V  ‖  p_θ )
                   │
                   │  minimize over θ
                   ▼
                 θ*
```

**Notice:** first slot of KL is always the data law; second is the model — that orientation becomes “forward KL” later.

### Bridge

The expanded KL has a term with only $p_V$ and a term with $p_\theta$. If the first does not depend on $\theta$, [argmin](./PREREQUISITES.md#p4-argmin) lets us drop it — that is the next move.

---

## Topic 3: Drop entropy of data (05:05-06:40)

### Where this sits on the master map

**SIMPLIFY** box. Same KL objective; algebra removes the part you cannot and need not control. Relies on [dropping constants in argmin](./PREREQUISITES.md#p4-argmin).

### Board / screenshot

![Board: first KL term struck; keep integral of p log p_theta](./screenshots/composites/ch03-topic-03-drop-entropy-panel1of1.png)
**Figure — ~05:35–06:34:** first integral marked independent of $\theta$ / entropy of data; remaining objective $-\int p_V\log p_\theta$.

### What he is establishing

Because integration is linear, the two-term expansion of KL is legitimate. Look at the first term $\int p_V\log p_V$. The optimization variable is $\theta$. That first term **does not mention $\theta$ at all**. For choosing $\theta^\star$, it can be ignored completely.

Interpretation: that term is (up to sign conventions) the **entropy of the data distribution**. The data are given. You cannot change how much entropy the true process has. If the job is to minimize divergence between $p_V$ and $p_\theta$, the only object you can twist is $p_\theta$.

Knock the entropy term off. The optimization that remains is

$$
\theta^\star
=\arg\min_\theta
\Bigl(-\int p_V(v)\log p_\theta(v)\,dv\Bigr)
=\arg\max_\theta
\int p_V(v)\log p_\theta(v)\,dv.
$$

A class question is already in the air: “Aren’t we begging the problem? We said we do not know $p_V$ — so how do we compute that integral?” Topic 4 answers with statistics. For now, own the logical move: **population KL minimization reduces to maximizing expected log model density under the true law.**

Wrong move: try to “learn the entropy of data” as if that were the training objective. Right move: accept entropy as fixed background and train only the model-dependent piece.

You can now reduce population KL training to maximizing expected log model density under the true law. Still missing: how to compute that expectation without knowing $p_V$.

### Analogy for this topic only

You grade students with a shared fifty-point bonus plus an exam. Ranking by total is the same as ranking by exam alone.

Question: **does changing the shared bonus change who ranks first?**

No. The bonus is background. Only the exam score ranks students. In lecture words: drop the entropy-of-data term; optimize expected log model density over theta.

### Local picture

```
  D_KL =  [ ∫ p_V log p_V ]  −  [ ∫ p_V log p_θ ]
              │                      │
              │ no θ                 │ has θ
              ▼                      ▼
           DROP                    KEEP
                                   │
                                   ▼
                    max_θ  ∫ p_V log p_θ  dv
```

**Notice:** dropping changes the objective’s *value*, not the *argmin* over $\theta$.

### Bridge

Even the remaining integral still multiplies by unknown $p_V$ in high dimension. How do you compute something that is both unknown and intractable? Rewrite as an [expectation](./PREREQUISITES.md#p5-expectation) and bring in sample averages.

---

## Topic 4: Unknown $p_V$; expectation form (06:12-07:50)

### Where this sits on the master map

**OBSTACLE → EXPECTATION** box. After dropping entropy, confront why the leftover integral is not plug-and-chug, then rewrite it as an [expectation](./PREREQUISITES.md#p5-expectation) that statistics can attack.

### Board / screenshot

![Board: integral as expectation; path to samples](./screenshots/composites/ch04-topic-04-expectation-obstacle-panel1of1.png)
**Figure — ~06:45–07:44:** remaining objective as $-\mathbb{E}_{p_V}[\log p_\theta(V)]$; pivot toward sample approximation / LLN.

### What he is establishing

Two blockers stop you from computing $\int p_V\log p_\theta\,dv$ directly.

1. **Intractability:** the integral lives in high-dimensional $\mathbb{R}^d$; numerical integration is hopeless for image-sized $d$.  
2. **Ignorance:** you do not know $p_V$ — estimating that law *is* the original problem. Plugging $p_V$ into the formula would beg the question.

Statistics offers a rewrite. By definition of expectation under a continuous law,

$$
\int p_V(v)\log p_\theta(v)\,dv
=\mathbb{E}_{p_V}\bigl[\log p_\theta(V)\bigr].
$$

So the optimization is

$$
\theta^\star=\arg\max_\theta\,\mathbb{E}_{p_V}\bigl[\log p_\theta(V)\bigr]
$$

(or $\arg\min$ of the negative). You still cannot evaluate that expectation by summing against $p_V$. The “biggest twist” is the next sentence of statistical practice: **expectations can be approximated from samples** drawn from the distribution that defines them. A theorem guarantees that with enough samples the approximation converges to the truth in probability — the **law of large numbers** (a student names it; the instructor accepts and invokes it).

Micro picture: average height in a city. You do not need the city’s full height-density formula. You measure fair samples and average. Same shape here with $f(v)=\log p_\theta(v)$.

Wrong move: invent a neural network “for the integral” while still needing $p_V$ as an input. Right move: stop needing $p_V$ as a function, and only need **samples** from $p_V$ (which you already have as $D$).

You can now say the population objective in expectation language and name the theorem that will replace $\mathbb{E}$ by an average. The explicit sample-mean formula and the big-data moral are Topic 5.

### Analogy for this topic only

You want the average height of everyone in a city (an expectation). You do not have the city’s full height census function $p_V$.

Question: **how do you estimate the average without the census formula?**

You measure heights of people who are fair samples from the city and average those measurements. You do not reconstruct the entire census first. In lecture words: $\mathbb{E}[\log p_\theta(V)]$ from samples, not from known $p_V$.

### Local picture

```
  Need:  ∫ p_V(v) log p_θ(v) dv
            │
            ├─ blocker 1: high-d integral
            └─ blocker 2: p_V unknown
            │
            ▼
  Rewrite:  E_{p_V}[ log p_θ(V) ]
            │
            ▼
  Still stuck until samples + LLN (Topic 5)
```

**Notice:** rewriting as $\mathbb{E}$ does not yet compute anything; it only changes the *shape* of the unknown into something LLN knows how to estimate.

### Bridge

Name the estimator: replace the expectation by a sample mean of $\log p_\theta(v_i)$ under IID draws — and face why machine learning hungers for enormous $n$.

---

## Topic 5: Law of large numbers; sample mean (07:42-12:10)

### Where this sits on the master map

**LLN ESTIMATOR** box. This is the practical heart: [weak LLN](./PREREQUISITES.md#p6-lln) turns the population objective into a sum you can code.

### Board / screenshot

![Board: LLN sample mean of log p_theta; theta hat](./screenshots/composites/ch05-topic-05-lln-sample-mean-panel1of1.png)
**Figure — ~08:10–11:49:**  
$\mathbb{E}_{p_V}[\log p_\theta(V)]\approx\frac1N\sum_i\log p_\theta(v_i)$ with $v_i\sim\mathrm{iid}\,p_V$;  
$\theta^\star=\arg\min_\theta\bigl(-\frac1N\sum_i\log p_\theta(v_i)\bigr)$; convergence as $N\to\infty$.

### What he is establishing

Invoke the **weak law of large numbers**. The expectation $\mathbb{E}_{p_V}[\log p_\theta(V)]$ is estimated by the **sample mean**

$$
\frac1n\sum_{i=1}^n \log p_\theta(v_i),
$$

where each $v_i$ is drawn IID from $p_V$. If the samples are not IID from that law, the theorem’s guarantee does not apply — so the lecture insists on the IID assumption.

Why is the sample mean computable when the integral was not? Because:

- $p_\theta(v_i)$ is known once you chose the model family and a candidate $\theta$;  
- the points $v_i$ are exactly the dataset.

So the optimization problem collapses to

$$
\hat\theta_n
=\arg\max_\theta
\frac1n\sum_{i=1}^n \log p_\theta(v_i)
\qquad (v_i\stackrel{\text{iid}}{\sim}p_V).
$$

That is the estimator. The cultural punchline: modern AI uses terabytes of data because sample averages approximate expectations reliably only when $n$ is huge. “What is infinity known to human beings? The entire internet.” The mathematics of LLN is centuries old; what is new is the ability to run the optimization arrow on that scale. Even large language models such as ChatGPT, he says, are trained using this same equation — what differs is the family $p_\theta$ and how the optimization is implemented, not the statistical skeleton.

Micro numbers: with $n=2$ and scores $\log p_\theta(v_1)=-1$, $\log p_\theta(v_2)=-3$, the sample average is $-2$. Change $\theta$ so both logs rise and the average rises — that is the training signal in one line.

Wrong move: treat “more data” as fashion. Right move: see large $n$ as buying a better Monte Carlo estimate of $\mathbb{E}[\log p_\theta]$.

You can now write the finite-sample training objective. Next: name it, defend the KL direction, and connect to likelihood language.

### Analogy for this topic only

You want the average temperature in a year. Measuring every infinitesimal moment is the “integral.” Recording noon temperature every day and averaging is the sample mean.

Question: **when does the daily average track the true mean well?**

When days are comparable draws from the same climate and you have many of them — not three days in a heat wave only. In lecture words: IID $v_i$, large $n$, $\frac1n\sum\log p_\theta(v_i)$.

### Local picture

```
  E[log p_θ(V)]
        │  weak LLN, v_i iid ~ p_V
        ▼
  (1/n) Σ_i log p_θ(v_i)
        │
        │  known: p_θ(·), data v_i
        ▼
  max over θ  →  θ̂_n
        │
        └─ same skeleton as ChatGPT training objective
```

**Notice:** computability came from *not* needing $p_V(v)$ as a formula — only samples.

### Bridge

Trace the derivation backward: this $\hat\theta_n$ is the parameter that (approximately) minimizes KL between truth and model. That naming, plus the student question “why not the other KL direction?”, is the next box.

---

## Topic 6: Min KL estimator; forward vs reverse; LOTUS (12:10-17:00)

### Where this sits on the master map

**NAMES + DIRECTION + LOTUS** box. Lock the estimator’s identity, answer forward vs reverse [KL](./PREREQUISITES.md#p3-kl), and justify why $\mathbb{E}[f(V)]$ with $f=\log p_\theta$ is legitimate ([LOTUS](./PREREQUISITES.md#p9-lotus)).

### Board / screenshot

![Board: minimum KL name; forward vs reverse discussion; LOTUS](./screenshots/composites/ch06-topic-06-min-kl-forward-lotus-panel1of1.png)
**Figure — ~12:33–16:36:** estimator named as minimal $D_{\mathrm{KL}}$; Kullback–Leibler names / $f$-divergence family; student Q on direction; LOTUS sketch with $f(v)=\log p_\theta(v)$.

### What he is establishing

Trace the chain: start from KL, drop entropy, replace expectation by sample mean. The parameter you obtain is the one that minimizes KL between the **true** distribution and the **model** distribution. Call it the **minimum KL divergence estimator** (minimal KL estimator).

$K$ and $L$ are people: **Kullback** and **Leibler**. More generally, KL sits inside a larger family called **$f$-divergences**; KL is one member. KL has known flaws and is “not the best” divergence in every theory sense — yet it is still widely used and often works empirically.

A student asks the sharp direction question: why $D_{\mathrm{KL}}(p_V\|p_\theta)$ and not $D_{\mathrm{KL}}(p_\theta\|p_V)$? In the literature the first is **forward KL**; the second is **reverse KL**. Reverse KL expands into terms that include pieces like $p_\theta\log p_V$ — and $p_V$ is still unknown, so you cannot compute what you need. Forward KL led to $\mathbb{E}_{p_V}[\log p_\theta]$, which samples can estimate. That is why this recipe uses forward KL.

Concrete direction check: true climate puts rainy days often; a model that almost never allows rain is punished hard by forward KL on real rainy samples (truth in the first slot). Reverse KL optimizes a different geometry and, in this data-driven recipe, is not computable the same way from $p_V$-samples alone.

To justify the expectation of a *function* of $V$, he recalls the **law of the unconscious statistician (LOTUS)**: if $X$ is a random variable and $f$ is a function, then $\mathbb{E}[f(X)]$ can be computed with respect to the law of $X$ without first deriving the full distribution of the new random variable $f(X)$. Here $f(v)=\log p_\theta(v)$. You estimate $\mathbb{E}[f(V)]$ by sample averages of $f(v_i)$; LLN sends those averages to the true expectation as $n\to\infty$. Full proof of LOTUS is left as prior probability homework / next-class fix; the lecture needs the statement and the use.

Wrong move: flip KL “because symmetry feels fair” and then discover an unusable objective. Right move: keep forward KL for computability in this data-driven setting.

You can now name the estimator (minimum KL), defend the forward direction, and state LOTUS for $f=\log p_\theta$. Still missing: the classical “likelihood product” name for the same maximizer.

### Analogy for this topic only

You score a student essay with a rubric that averages “how surprising the essay is under your model” using sentences drawn from the student’s true writing habits.

Question: **do you weight sentences by the student’s true habits, or by your model’s habits?**

Weighting by true habits (forward) uses samples of real writing. Weighting by the model’s habits (reverse) needs the true writing density inside the formula — which you do not have. In lecture words: forward $D(p_V\|p_\theta)$, not reverse.

### Local picture

```
  Forward KL:  D(p_V ‖ p_θ)  →  E_{p_V}[log p_θ]   ✓ samples
  Reverse KL:  D(p_θ ‖ p_V)  →  terms with log p_V  ✗ stuck

  LOTUS:  f(v)=log p_θ(v)
          E[f(V)] under law of V  (no need full law of f(V) first)
                │
                ▼
          sample avg of f(v_i) → E[f] as n→∞
```

**Notice:** “forward” here means truth in the first slot — the orientation that made the LLN step possible.

### Bridge

The same maximizer of $\sum\log p_\theta(v_i)$ has another famous name once you call $p_\theta(v_i)$ a **likelihood** and write a product over the dataset — Topic 7 closes the dual story.

---

## Topic 7: Likelihood; MLE equals min KL (17:00-24:51)

### Where this sits on the master map

**MLE ALIAS** box. Translate the min-KL estimator into maximum-likelihood language without changing the math. Uses [log monotonicity](./PREREQUISITES.md#p7-log) and [likelihood](./PREREQUISITES.md#p8-likelihood).

### Board / screenshot

![Board: likelihood definition; joint product; log-likelihood; MLE](./screenshots/composites/ch07-topic-07-mle-equals-min-kl-panel1of1.png)
**Figure — ~17:37–24:13:** $p_\theta(v_i)$ = likelihood of $v_i$; joint $L(D)=\prod_i p_\theta(v_i)$ under IID;  
$\ell(D)=\sum_i\log p_\theta(v_i)$; $\theta^\star=\arg\max_\theta\frac1N\sum\log p_\theta(v_i)$; named MLE and equated to minimal $D_{\mathrm{KL}}$ estimator.

### What he is establishing

Flip the sign if needed: maximizing $\frac1n\sum\log p_\theta(v_i)$ is the same decision as minimizing its negative. Now introduce the standard name: if $p_\theta$ is the model density, then **evaluating that density at a particular point** is called the **likelihood** of that point under the model.

So $p_\theta(v_i)$ is the likelihood of one data point. Finding $\theta$ that maximises the likelihood of **all** data points under the model is the **maximum likelihood estimator (MLE)**.

Derive it the classical way so the equivalence is obvious. For one point, use $p_\theta(v_i)$. For the whole dataset of $n$ IID random variables, the joint density factors as a **product**:

$$
L(D;\theta)=\prod_{i=1}^n p_\theta(v_i).
$$

Maximize $L$ over $\theta$. Products are unpleasant, so pass through a strictly increasing function: the logarithm. Maximizing $L$ is equivalent to maximizing $\log L$, and

$$
\log L(D;\theta)=\sum_{i=1}^n \log p_\theta(v_i).
$$

Seek

$$
\theta^\star=\arg\max_\theta\sum_{i=1}^n \log p_\theta(v_i)
$$

(or a scaled $1/n$ version — same argmax). People call this the MLE.

Worked micro: three scores $0.5,0.4,0.2$ give $L=0.04$ and $\log L=\log0.5+\log0.4+\log0.2$. Any $\theta$ that raises $L$ also raises $\log L$ — so you may optimize the sum of logs only.

The instructor is blunt: he is not a big fan of the slogan “maximize likelihood” as *motivation*, because it can sound ungrounded. He prefers: this is the parameter that **minimizes KL** between true and model distributions. Mathematically, under this derivation, **they are exactly the same thing**. He will still say “MLE” by convention, while remembering “minimum KL estimator.”

IID is always assumed; LLN says the sample objective tracks the population one for large $n$. The handwavy story people tell is: $p_\theta(v)$ is how likely $v$ is under the model, so pick $\theta$ that makes the whole dataset as likely as possible. That story is common; the KL story is the one he wants you to keep as the mathematical spine.

A student pushes on rare events and information: if low-probability events carry more information, does maximizing likelihood of a “weird” dataset force a bad model? The reply, carefully: look at *which* likelihood you maximize — the likelihood of the **observed data under the model** as a function of $\theta$. Confusions born from mixing “information of rare events” with MLE rhetoric are part of why he anchors on KL. Closing slogan of the lecture:

> Wherever you see maximum likelihood estimator, remember it is the minimum KL divergence estimator.

Stop; continue next lecture (architectures / further development of the recipe).

Wrong move: treat MLE as a mysterious second theory unrelated to KL. Right move: same $\theta^\star$, two speeches — prefer the KL grounding while using the MLE name in convention.

You can now write $L(D)=\prod p_\theta(v_i)$, pass to $\log L$, and equate MLE with the minimum-KL estimator. Still missing for later lectures: which family $p_\theta$ to pick and how to optimize it at scale.

### Analogy for this topic only

Two teachers grade the same homework stack.

- Teacher A: “Choose the grading curve that makes this stack of answers look as plausible as possible under the curve.”  
- Teacher B: “Choose the curve whose induced answer-distribution is KL-closest to the true student population.”

Question: **after the algebra of IID products and logs, do they pick different curves?**

No — same maximizer, two speeches. Prefer Teacher B’s grounding; still recognize Teacher A’s name on the syllabus. In lecture words: MLE $=$ minimum KL estimator.

### Local picture

```
  p_θ(v_i)  = likelihood of one point
       │
       │  IID joint
       ▼
  L(D) = Π_i p_θ(v_i)
       │  log (monotonic)
       ▼
  log L = Σ_i log p_θ(v_i)
       │  maximize over θ
       ▼
     θ_MLE
       │
       └════ identical ════┐
                           ▼
                    θ_min-KL
                    (from Topics 2–5)
```

**Notice:** product → log → sum is not a new objective; it is bookkeeping for the same average of $\log p_\theta$ that LLN already justified.

### Bridge

The recipe now has a named, computable estimator. What remains for later lectures is *which* family $p_\theta$ to use and *how* to optimize it at scale — not a second definition of training.

---

## External references

**How to use:** stay in NOTES for the chain first. When a box still feels thin, open **only that topic’s row group** below (video + blog/notes). Prior package [Lec 12 KL](../13-Lec12-KL-Divergence/NOTES.md) is the immediate prequel for the definition of $D_{\mathrm{KL}}$.

Links are **companions mapped to map boxes** — not a second lecture dump. Prefer one video *or* one short notes page per sticky point rather than watching everything.

### Topic 1 — IID data & general $V$ (setup)

| Resource | Type | Why it helps |
|----------|------|--------------|
| [IID: Independent and Identically Distributed (Data Talks)](https://www.youtube.com/watch?v=lhzndcgCXeo) | Video | Plain split of “independent” vs “identically distributed” with sample language |
| [Terms: IID (IntuitiveML)](https://www.youtube.com/watch?v=EGKbPww2_rc) | Video | One-minute definition hit for the two halves of IID |
| [Prior: Lec 07 IID Assumption package](../08-Lec07-IID-Assumption/NOTES.md) | Series notes | Same course thread on identical + independent across data points |

### Topic 2 — Model $p_\theta$ & continuous KL objective

| Resource | Type | Why it helps |
|----------|------|--------------|
| [StatQuest: KL Divergence](https://www.youtube.com/watch?v=q0AkK8aYbLY) | Video | Rebuilds $D(p\|q)$ before you minimize it |
| [StatQuest: Main Ideas behind Probability Distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc) | Video | Density/PMF picture for “model density you can evaluate” |
| [3Blue1Brown: Why “probability of 0” is not “impossible”](https://www.youtube.com/watch?v=ZA4JkHKZM50) | Video | Continuous density height vs probability of an exact point |

### Topic 3 — Drop entropy of data

| Resource | Type | Why it helps |
|----------|------|--------------|
| [3Blue1Brown: Entropy (info / stat mech feel)](https://www.youtube.com/watch?v=ErfnhcEV1O8) | Video | Why entropy is a property of the data law alone |
| [StatQuest: KL Divergence](https://www.youtube.com/watch?v=q0AkK8aYbLY) (CE − H view) | Video | Visual that $H(p)$ does not depend on the model $q$ |
| [Lec 11 Entropy package](../12-Lec11-Entropy/NOTES.md) | Series notes | Course definition of $H$ you are allowed to drop |

### Topic 4 — Unknown $p_V$; expectation form

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Monte Carlo integration notes (Louis Aslett)](https://www.louisaslett.com/Courses/DSSM/notes/monte-carlo-integration.html) | Notes | $\mathbb{E}[f]=\int p f$ estimated by sample averages — same rewrite as the board |
| [StatLect: Monte Carlo method](https://www.statlect.com/asymptotic-theory/Monte-Carlo-method) | Notes | Short formal statement that sample means approximate expectations |
| [Khan Academy: Probability density functions](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-continuous/v/probability-density-functions) | Video | Why continuous $p_V$ is a height, so “plug in $p_V$” is not a free gift |

### Topic 5 — Law of large numbers; sample mean

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Khan Academy: Law of large numbers](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/expected-value-lib/a/law-of-large-numbers) | Article | Beginner statement of why large $n$ stabilizes averages |
| [Monte Carlo integration notes (Louis Aslett)](https://www.louisaslett.com/Courses/DSSM/notes/monte-carlo-integration.html) | Notes | Explicit weak-LLN step from $\mathbb{E}[f]$ to $\frac1n\sum f(v_i)$ |
| [Seeing Theory — Law of Large Numbers](https://seeing-theory.brown.edu/probability-distributions/index.html) | Interactive | Visual demos of averages settling (use the LLN module on the site) |

### Topic 6 — Min KL name; forward vs reverse; LOTUS

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Reverse vs Forward KL (Tuan Anh Le notes)](https://www.tuananhle.co.uk/notes/reverse-forward-kl.html) | Blog/notes | Mode-covering vs mode-seeking geometry; matches “why not reverse” |
| [Forward vs Reverse KL interactive (Hiroaki H.)](https://hiroakih.me/kl-divergence.html) | Interactive | Drag distributions; see forward vs reverse behavior |
| [LOTUS — law of the unconscious statistician (bookdown)](https://bookdown.org/kevin_davisross/probsim-book/law-of-the-unconscious-statistician-lotus.html) | Notes | Clean statement: $\mathbb{E}[g(X)]$ without finding the law of $g(X)$ first |

### Topic 7 — Likelihood; MLE ≡ min KL

| Resource | Type | Why it helps |
|----------|------|--------------|
| [StatQuest: Maximum Likelihood, clearly explained](https://www.youtube.com/watch?v=XepXtl9YKwc) | Video | Product likelihood → log-likelihood with pictures |
| [Pieter Abbeel: Maximum Likelihood Examples](https://www.youtube.com/watch?v=BFHGIE-nwME) | Video | Short worked MLE examples (Berkeley CS188 style) |
| [MLE and loss functions (Rish blog)](https://rish-01.github.io/blog/posts/ml_estimation/) | Blog | Modern ML framing: MLE as the loss you already minimize |
| [Jordan notes ch.9 — MLE / KL link (Berkeley)](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter9.pdf) | Course notes | Graduate written confirmation that MLE ↔ min KL |

### Whole-map companion

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Lec 12 KL Divergence NOTES](../13-Lec12-KL-Divergence/NOTES.md) | Series notes | Definition and properties of $D_{\mathrm{KL}}$ this lecture minimizes |
| [Lec 10 Challenges of ML — recipe needs $d$](../11-Lec10-Challenges-of-ML/NOTES.md) | Series notes | Why the course was hunting a pairwise score in the first place |

---

## Sources

- Video: [Lec 13 Minimization of KL Divergence](https://www.youtube.com/watch?v=Ij4p5hLbfo4) (NPTEL IISc)  
- Captions: `raw/captions.en.vtt` / `raw/captions.en.timed.txt`  
- Claim sheets: `raw/claims/topic-0N.md`  
- Prior course package: [Lec 12 KL Divergence](../13-Lec12-KL-Divergence/NOTES.md)
