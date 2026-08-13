# Prerequisites — warm-up before Lec 13 (Minimization of KL)

> **Do this first** if “expectation,” “sample mean,” “likelihood,” or “arg min of KL” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Course: NPTEL / IISc · follows [Lec 12 KL Divergence](../13-Lec12-KL-Divergence/NOTES.md).  
> **Beginner deep warm-up:** each idea has definition · micro example · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "Data D = n points drawn IID from unknown true density p_V."
  "A model is a family p_θ that I can evaluate; θ are knobs I choose."
  "KL(p_V ‖ p_θ) scores how far the model is from truth."
  "When optimizing over θ, any term that does not depend on θ can be dropped."
  "∫ p log p_θ = E_p[log p_θ]; with data I replace E by a sample average (LLN)."
  "Likelihood = density evaluated at a data point; for IID data, joint likelihood is a product."
  "log turns products into sums; maximizing log-likelihood is the same as maximizing likelihood."
  "MLE and minimum-KL estimator are the same math under this recipe."
```

**Warm-up → lecture boxes**

```
  §1  Density vs probability              ──► Topics 2, 7
  §2  Model family p_θ                    ──► Topics 2–3
  §3  KL reload (why we minimize it)      ──► Topics 2–3, 6
  §4  Argmin / drop constants             ──► Topic 3
  §5  Expectation + LOTUS (light)         ──► Topics 4–6
  §6  IID + law of large numbers          ──► Topics 1, 5
  §7  Log is monotonic (products→sums)    ──► Topic 7
  §8  Likelihood at a point               ──► Topic 7
```

---

## 1. Density vs probability (continuous)

<a id="p1-density"></a>

### Purpose for the video

The lecture writes $p_V(v)$ and $p_\theta(v)$ and sometimes casually says “probability,” but for continuous data these are **densities**. Likelihood is “density at a point,” not “P(exactly this real number).”

### Definitions

| Idea | Meaning |
|------|---------|
| **Probability mass** (discrete) | $P(X=x)$ is a number in $[0,1]$; sums to 1 over all $x$ |
| **Density** $p(x)$ (continuous) | height of a curve; **area** under the curve over a set is probability |
| **Height is not probability** | $p(x)=2$ is allowed; $P(X=x)=0$ for continuous $X$ |

### Worked micro

Uniform on $[0,0.5]$ has density height $2$ on that interval:

$$
\int_0^{0.5} 2\,dx = 1
$$

So “evaluate density at $0.25$” returns $2$, not a probability of landing *exactly* on $0.25$.

### Analogy — heat map

A weather heat map colors every city block by temperature intensity. The color at one intersection is not “the chance it rains only on that pixel”; it is a **local intensity**. Densities work the same way for continuous outcomes.

### Notice

- Discrete: likelihood often means $P_\theta(X=x_i)$.  
- Continuous: likelihood means $p_\theta(x_i)$ (density height).  
- The lecture’s algebra is the same shape either way.

### Mini-check

1. Can a valid density exceed 1 at some points?  
2. Why is “probability of this exact photo” a slippery phrase for continuous pixel vectors?

---

## 2. Model family $p_\theta$

<a id="p2-model"></a>

### Purpose for the video

Everything after setup is “pick the best $\theta$.” You need a clear picture of what a **parametric model** is.

### Definitions

| Idea | Meaning |
|------|---------|
| **True law** $p_V$ | unknown process that generated data |
| **Model family** $\{p_\theta\}$ | candidate densities you can write down and evaluate |
| **Parameter** $\theta$ | knobs (means, variances, network weights, …) |
| **Evaluate** $p_\theta(v)$ | plug a concrete $v$ into the formula → a number |

### Worked micro

Suppose $V$ is a real number and you assume

$$
p_\theta(v)=\frac{1}{\sqrt{2\pi}\sigma}\exp\!\Big(-\frac{(v-\mu)^2}{2\sigma^2}\Big),\quad \theta=(\mu,\sigma)
$$

Then for data point $v_i=1.2$ and guess $(\mu,\sigma)=(0,1)$, you can compute $p_\theta(1.2)$ on a calculator. That number is ready for sums and logs later.

### Analogy — radio station presets

Truth is the real music on the airwaves (unknown full spectrum). Your model is a radio with knobs (station, bass, treble). Each knob setting $\theta$ defines a different “sound profile” $p_\theta$. Training = twist knobs until the profile best matches the music you recorded.

### Notice

- The model can be **wrong** (truth not in the family). KL still scores “best wrong model.”  
- Without ability to **evaluate** $p_\theta(v_i)$, the sample-mean trick cannot run.

### Mini-check

1. What must you be able to compute for every data point $v_i$?  
2. Is $\theta$ part of nature or part of your description?

---

## 3. KL reload — score between truth and model

<a id="p3-kl"></a>

### Purpose for the video

Lec 12 defined KL. Lec 13 **minimizes** it. You only need the working form.

### Definition (continuous, lecture shape)

$$
D_{\mathrm{KL}}(p_V\|p_\theta)
=\int p_V(v)\,\log\frac{p_V(v)}{p_\theta(v)}\,dv
=\underbrace{\int p_V\log p_V\,dv}_{\text{depends on data only}}
-\underbrace{\int p_V\log p_\theta\,dv}_{\text{depends on }\theta}
$$

Discrete twin:

$$
D_{\mathrm{KL}}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}=H(p,q)-H(p)
$$

### Properties you need

| Fact | Why it matters |
|------|----------------|
| $D_{\mathrm{KL}}\ge 0$ | zero means match (under mild conditions) |
| Not symmetric | $D(p\|q)\ne D(q\|p)$ in general → **forward vs reverse** in Topic 6 |
| First argument = truth | expectation is under $p_V$ (data law) |

### Worked micro (binary)

$p=(0.9,0.1)$, $q=(0.5,0.5)$ (use $\log_2$ for bits):

$$
D_{\mathrm{KL}}(p\|q)=0.9\log_2\frac{0.9}{0.5}+0.1\log_2\frac{0.1}{0.5}\approx 0.53\text{ bits}
$$

If $q=p$, KL is $0$.

### Analogy — two weather forecasts

True climate is $p$. Forecast app $A$ is $q$. KL asks: “If days really follow climate $p$, how much extra surprise do I suffer by believing app $A$?” It is **not** “how wrong would climate look if the app were true” (that swaps order).

### Notice

- Minimizing $D_{\mathrm{KL}}(p_V\|p_\theta)$ over $\theta$ is “make the model look good when samples come from truth.”  
- That is the **forward** KL used in this lecture.

### Mini-check

1. If $p_\theta=p_V$ for some $\theta$, what is KL?  
2. Why is “distance” a slightly dangerous English word for KL?

---

## 4. Argmin and dropping constants

<a id="p4-argmin"></a>

### Purpose for the video

The single algebraic move that removes entropy of data.

### Definitions

| Symbol | English |
|--------|---------|
| $\arg\min_\theta f(\theta)$ | the value of $\theta$ that makes $f$ smallest |
| $\arg\max_\theta f(\theta)$ | the value of $\theta$ that makes $f$ largest |
| Constant w.r.t. $\theta$ | term that does not change when you change $\theta$ |

### Rule (memorize)

If $c$ does not depend on $\theta$, then

$$
\arg\min_\theta\bigl(f(\theta)+c\bigr)=\arg\min_\theta f(\theta)
$$

Same for $\arg\max$. Signs flip when you maximize instead of minimize:

$$
\arg\min_\theta\bigl(-g(\theta)\bigr)=\arg\max_\theta g(\theta)
$$

### Worked micro

Cost $= 7 + (\theta-3)^2$. The $7$ never helps choose $\theta$; best $\theta$ is still $3$.

In KL:

$$
D_{\mathrm{KL}}(p_V\|p_\theta)=H_{\text{term}}(p_V)-\mathbb{E}_{p_V}[\log p_\theta(V)]
$$

$H_{\text{term}}$ is fixed by data → drop it when optimizing $\theta$.

### Analogy — race with a fixed head start

Every racer gets the same $+10$ seconds added to their time. Ranking by total time is the same as ranking by race time alone. Entropy of data is that shared $+10$.

### Notice

- Dropping a constant changes the **value** of the objective, not the **argmin**.  
- You still care about KL’s value when *comparing models* outside pure optimization — but for choosing $\theta^\star$ the constant is useless.

### Mini-check

1. $\arg\min_\theta(5-\log p_\theta)$ equals what in max form?  
2. Why can we not drop $\mathbb{E}[\log p_\theta]$?

---

## 5. Expectation as a weighted average

<a id="p5-expectation"></a>

### Purpose for the video

The remaining KL piece is rewritten as an expectation, then approximated by a sample mean.

### Definitions

**Discrete:**

$$
\mathbb{E}_{X\sim p}[f(X)]=\sum_x p(x)\,f(x)
$$

**Continuous:**

$$
\mathbb{E}_{X\sim p}[f(X)]=\int p(x)\,f(x)\,dx
$$

**Lecture case:** $f(v)=\log p_\theta(v)$, so

$$
\int p_V(v)\log p_\theta(v)\,dv=\mathbb{E}_{p_V}\bigl[\log p_\theta(V)\bigr]
$$

### Worked micro

Die fair, $f(x)=x$: $\mathbb{E}[X]=3.5$.  
If $f(x)=\log q(x)$ for some model $q$, you average the log-model-scores under the **true** faces you roll — not under the model.

### Analogy — class average of exam tricks

You do not know the full population of students, but you know each student and a score function $f$. The class average of $f$ estimates the population average of $f$ when the class is a fair sample.

### LOTUS (light) — expectation of a *function*

<a id="p9-lotus"></a>

Topic 6 names the **law of the unconscious statistician (LOTUS)**. Plain claim: for a nice function $f$,

$$
\mathbb{E}\bigl[f(V)\bigr]
$$

uses the law of $V$ and the rule $f$ — you need **not** first derive the full distribution of $W=f(V)$.

**Lecture use:** $f(v)=\log p_\theta(v)$. Sample averages of $f(v_i)$ estimate $\mathbb{E}_{p_V}[f(V)]$.

**Analogy:** average “after-tax tip” by applying tax to each bill, then averaging — no need for a new histogram of tips first.

### Notice

- Expectation uses weights from **$p_V$** (truth).  
- You rarely know $p_V$, so you cannot sum with those weights directly — that is why samples appear next.

### Mini-check

1. Rewrite $\int p_V(v)\log p_\theta(v)\,dv$ as an expectation.  
2. Whose law supplies the weights: truth or model?  
3. In the lecture, what is $f(v)$ for LOTUS?

---

## 6. IID samples and the law of large numbers

<a id="p6-lln"></a>

### Purpose for the video

This is why “more data” is not marketing fluff — it is the theorem that turns $\mathbb{E}$ into $\frac1n\sum$.

### Definitions

| Idea | Meaning |
|------|---------|
| **IID** | independent **and** identically distributed |
| **Sample mean** | $\bar f_n=\frac1n\sum_{i=1}^n f(v_i)$ |
| **Weak LLN (idea)** | if $v_i$ are IID from $p$ and $f$ is well-behaved, $\bar f_n$ approaches $\mathbb{E}_p[f(V)]$ as $n$ grows (in probability) |

### Worked micro

True mean of a coin’s numeric coding $H=1,T=0$ is $0.5$. After $n=1000$ fair flips, the fraction of heads is usually near $0.5$. After $n=4$, it can be $0$ or $1$ and look ridiculous.

Lecture application:

$$
\mathbb{E}_{p_V}[\log p_\theta(V)]
\;\approx\;
\frac1n\sum_{i=1}^n \log p_\theta(v_i)
\quad\text{when }v_i\stackrel{\text{iid}}{\sim}p_V
$$

### Analogy — tasting a soup pot

One spoonful is noisy. Many independent spoonfuls average toward the pot’s true saltiness. If you keep tasting the *same* spoonful (not independent), or stir only the top (not same distribution), the average lies.

### Second analogy — two coin jars

Jar A and jar B both claim “fair coins.” You flip coin #1 ten times, then reuse *that same sequence* ten more times and call it “twenty samples.” Those are **not** independent draws. IID would mean: flip a new coin from the same jar each time. LLN needs the second story.

### ASCII — what IID splits into

```
  IID = Independent  +  Identically distributed
           │                    │
           │                    └─ every v_i comes from the SAME p_V
           │
           └─ knowing v_1 does not change the law of v_2

  Break either  →  sample mean may not track E[f(V)]
```

### Notice

- **Identical:** same $p_V$ for every $i$.  
- **Independent:** knowing $v_1$ does not change the law of $v_2$.  
- Break either and LLN’s guarantee for this recipe is on thin ice.  
- Huge datasets (internet-scale) are the practical face of “$n\to\infty$.”

### Mini-check

1. Write the sample-mean estimator of $\mathbb{E}[\log p_\theta(V)]$.  
2. Why does the lecture insist samples are IID from $p_V$?

---

## 7. Log is monotonic — products become sums

<a id="p7-log"></a>

### Purpose for the video

IID joint likelihood is a **product**. Optimization hates products; log turns them into sums.

### Key facts

| Fact | Consequence |
|------|-------------|
| $\log$ is **strictly increasing** | $\arg\max_\theta L(\theta)=\arg\max_\theta \log L(\theta)$ for $L>0$ |
| $\log(ab)=\log a+\log b$ | product $\to$ sum |
| $\log(a^n)=n\log a$ | powers $\to$ multiples |

### Worked micro

$L(\theta)=0.2\times 0.4\times 0.1=0.008$.  
$\log L=\log 0.2+\log 0.4+\log 0.1$ (any fixed base).  
Maximizing $L$ or $\log L$ picks the same $\theta$; the log path is numerically stabler.

### Analogy — earthquake magnitudes

Multiplying many tiny probabilities is like multiplying many tiny numbers until the calculator underflows to zero. Logging is like reporting magnitudes: you add manageable pieces instead of multiplying dust.

### Notice

- Base of log only scales the objective; $\arg\max$ unchanged if base $>1$ fixed.  
- Never take $\log$ of a negative likelihood (densities should be positive where data live).

### Mini-check

1. Why is $\arg\max \prod_i p_\theta(v_i)$ the same as $\arg\max \sum_i \log p_\theta(v_i)$?  
2. What goes wrong if some $p_\theta(v_i)=0$?

---

## 8. Likelihood at a point

<a id="p8-likelihood"></a>

### Purpose for the video

The lecture renames $p_\theta(v_i)$ as **likelihood** and then maximises it — that is MLE language for the same min-KL estimator.

### Definitions

| Phrase | Meaning in this course |
|--------|-------------------------|
| **Likelihood of $v_i$ under $p_\theta$** | the number $p_\theta(v_i)$ (density or mass at that point) |
| **Joint likelihood of dataset $D$** (IID) | $L(D;\theta)=\prod_{i=1}^n p_\theta(v_i)$ |
| **Log-likelihood** | $\ell(\theta)=\sum_i\log p_\theta(v_i)$ (or $/n$ scaled) |
| **MLE** | $\theta_{\mathrm{MLE}}=\arg\max_\theta L(D;\theta)=\arg\max_\theta \ell(\theta)$ |

### Worked micro

Two points $v_1=0$, $v_2=1$. Model $p_\mu=\mathcal{N}(\mu,1)$.  
$L(\mu)=p_\mu(0)\,p_\mu(1)$.  
$\ell(\mu)=\log p_\mu(0)+\log p_\mu(1)$.  
Best $\mu$ maximises either; it also minimises empirical forward KL to the data’s empirical story (population version uses true $p_V$).

### Analogy — two explanations of the same exam score

“Minimum KL” says: make the model distribution as close as possible to the true one, scored by KL.  
“Maximum likelihood” says: make this dataset look as plausible as possible under the model.  
Same homework, two essay titles.

### Second analogy — restaurant menu vs kitchen inventory

Likelihood language: “How plausible is *this* order ticket under kitchen settings $\theta$?”  
KL language: “How far is the kitchen’s output distribution from the true customer demand?”  
For IID tickets, maximizing ticket-likelihood (product of per-item scores) is the same knob-turning as minimizing that KL after LLN.

### Worked product → log (numbers)

Suppose three points with model scores $p_\theta(v_i)\in\{0.5, 0.4, 0.2\}$.

$$
L=0.5\times 0.4\times 0.2=0.04,\qquad
\log L=\log 0.5+\log 0.4+\log 0.2
$$

Any other $\theta'$ that makes $L$ larger also makes $\log L$ larger. You never need to keep the tiny product if you optimize the sum of logs.

### Notice

- Likelihood is a function of **$\theta$** once data are fixed (data locked, knobs free).  
- Probability usually treats parameters fixed and data random; likelihood flips the emphasis.  
- Teacher’s slogan for Lec 13: wherever you see MLE, remember **minimum KL estimator**.

### Mini-check

1. Write $L(D;\theta)$ for IID data.  
2. In one sentence: why MLE and min-KL can name the same $\theta^\star$?

---

### Paper check (end-to-end)

Without notes, fill blanks:

1. Goal: estimate _____ given dataset $D=\{v_i\}$.  
2. $\theta^\star=\arg\min_\theta D_{\mathrm{KL}}(\,\_\_\_\,\|\,\_\_\_\,)$.  
3. Drop _____ because it does not depend on $\theta$.  
4. Remaining piece $\mathbb{E}[\log p_\theta(V)]\approx$ _____.  
5. $L(D)=\prod_i$ _____ ; MLE maximises _____ or its log.

**Answers (peek):** (1) $p_V$ (2) $p_V$, $p_\theta$ (3) entropy / $\int p_V\log p_V$ (4) $\frac1n\sum\log p_\theta(v_i)$ (5) $p_\theta(v_i)$; $L$ or $\sum\log p_\theta(v_i)$.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file · Part B = NOTES.
