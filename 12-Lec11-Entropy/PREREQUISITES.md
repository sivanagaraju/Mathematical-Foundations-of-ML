# Prerequisites — warm-up before Lec 11 (entropy)

> **Do this first** if “log of a probability,” “expectation,” or “surprisal” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on [Lec 10](../11-Lec10-Challenges-of-ML/PREREQUISITES.md) (recipe: model → distance → train).  
> Strong basics: every formula below is decoded with micro numbers.

```
  After this warm-up you can say:

  "Surprisal of an event A is I(A) = −log P(A): rare events surprise more."
  "I(Ω)=0 and I(empty) is infinite; independent events add surprisals."
  "Entropy of a discrete law is the average surprisal: H = −∑ p_i log p_i."
  "Use 0 log 0 := 0 when writing the sum (only sum over outcomes with p>0)."
  "Entropy is one law’s average surprise; a divergence scores two laws."
  "ML needs divergences for recipe step 2; entropy is the info-theory building block."
```

**Warm-up → lecture boxes**

```
  §1  Log of a probability (why −log)   ──► Topic 3
  §2  Events, Ω, empty set reload       ──► Topic 3
  §3  Independence → products of P      ──► Topic 3
  §4  Discrete PMF + expectation        ──► Topic 4
  §5  Surprisal → entropy (formula)     ──► Topics 3–5
  §6  Entropy vs divergence (not same)  ──► Topics 2, 5
  §7  Recipe step 2 needs d             ──► Topics 1–2
  §8  Log base (bits vs nats)           ──► Topics 3–4
```

---

## 1. Why $-\log$ of a probability?

<a id="p1-neglog"></a>

### Purpose for the video

The whole lecture hangs on one formula: information of event $A$ is $-\log P(A)$.

### What $\log$ does to products

$$
\log(ab)=\log a+\log b
$$

Independent events multiply probabilities; their informations should **add** — logs turn products into sums.

### Why the **minus** sign?

$P(A)\in(0,1]$ for ordinary events. $\log P(A)$ is then **negative** or zero (for $\log$ of numbers $\le 1$).  
Minus flips it so information is **non-negative**:

| $P(A)$ | $-\log P(A)$ (base 2, bits) |
|--------|------------------------------|
| $1$ | $0$ |
| $1/2$ | $1$ |
| $1/4$ | $2$ |
| $1/8$ | $3$ |
| $\to 0$ | $\to +\infty$ |

**Pattern:** more likely → smaller $-\log P$; almost impossible → huge number.

### Micro

Fair coin lands heads: $P=1/2$, surprisal $=1$ bit (if $\log_2$).  
Fair die shows $6$: $P=1/6$, surprisal $\approx 2.58$ bits — rarer, more surprising.

---

## 2. Events, $\Omega$, and empty set (reload)

<a id="p2-events"></a>

### Purpose for the video

He checks the formula on $\Omega$ and the null event. An **event** is a set of outcomes that we assign a probability (formally: a member of the event space / $\sigma$-algebra $\mathcal{F}$).

| Event | Probability | Surprisal should be… |
|-------|-------------|----------------------|
| Sample space $\Omega$ (“something happens”) | $P(\Omega)=1$ | **zero** information |
| Empty set $\emptyset$ (“impossible event”) | $P(\emptyset)=0$ | **infinite** information |
| Ordinary event $A$ | $0<P(A)<1$ | positive finite |

### Link to random variables (needed for entropy)

When $X$ is discrete and takes value $x_i$, the relevant event is the **singleton**

$$
A=\{X=x_i\}
$$

with $P(A)=p(x_i)$. So “surprisal of the outcome $x_i$” is just $I(\{X=x_i\})=-\log p(x_i)$.

### Analogy — purpose

Saying “tomorrow will be some day of the week” tells you nothing.  
Saying “a squared circle will appear” is an impossible claim — infinitely shocking if treated as certain.

---

## 3. Independence (for the additivity rule)

<a id="p3-independence"></a>

### Purpose for the video

Third design rule: independent events combine by **adding** informations.

If $A$ and $B$ are independent:

$$
P(A\cap B)=P(A)P(B)
$$

Then

$$
-\log P(A\cap B)=-\log P(A)-\log P(B)
$$

so

$$
I(A\cap B)=I(A)+I(B)
$$

### Micro

Two independent fair coins both heads:  
$P=1/4$, $I=2$ bits $=1+1$.

---

## 4. Discrete PMF and expectation

<a id="p4-pmf-expectation"></a>

### Purpose for the video

Entropy starts in the **discrete** case (he says it is easier).

### PMF

For discrete $X$ taking values $x_1,x_2,\ldots$:

$$
p(x_i)=P(X=x_i),\qquad \sum_i p(x_i)=1,\quad p(x_i)\ge 0
$$

Unlike a continuous density, **evaluating the PMF at a point is a probability**.

### Why discrete first? (continuous trap)

For a continuous RV, $P(X=x)=0$ for every exact $x$, so the naive point surprisal $-\log P(X=x)$ is **infinite** everywhere. That is why he starts with discrete mass functions, where $p(x_i)=P(X=x_i)$ is a real probability you can plug into $-\log$. Continuous “differential entropy” exists later but needs extra care — **out of scope today**.

### Expectation of a function

$$
\mathbb{E}[f(X)]=\sum_i p(x_i)\,f(x_i)
$$

English: for each possible value, take $f$ there, weight by how often it occurs, sum.

### Micro — expected number of spots (die)

$f(x)=x$, fair die: $\mathbb{E}[X]=(1+2+\cdots+6)/6=3.5$.

For entropy we will set $f(x)=-\log p(x)$ — same weighting idea, different $f$.

---

## 5. From surprisal of one event to entropy of a law

<a id="p5-entropy-formula"></a>

### Purpose for the video

This is the central definition of the lecture.

### One event

$$
I(A)=-\log P(A)
$$

### One outcome of a discrete RV

$$
I(X=x_i)=-\log p(x_i)
$$

### Whole distribution — average surprisal

$$
\boxed{H(p)=\mathbb{E}_{X\sim p}\big[-\log p(X)\big]=-\sum_i p(x_i)\log p(x_i)}
$$

| Piece | Meaning |
|-------|---------|
| $-\log p(x_i)$ | surprisal if $X=x_i$ |
| multiply by $p(x_i)$ | weight by how often that value occurs |
| sum | average information in the whole law |
| name | **entropy** of $p$ |

**Reading the formula as a process:** draw $X\sim p$ many times; each draw has surprisal $-\log p(X)$; the long-run **average surprise per draw** is $H(p)$. That is “how much uncertainty the law embeds,” not a distance to another law.

### Convention: $0\log 0$

If some outcome has $p=0$, the term $p\log p$ is taken as **$0$** (limit as $p\to 0^+$ of $p\log p$ is $0$). In practice: **only sum over outcomes with $p(x_i)>0$** (the support).

### Micro — fair coin

$p(\text{H})=p(\text{T})=1/2$, base 2:

$$
H=-\Big(\tfrac12\log_2\tfrac12+\tfrac12\log_2\tfrac12\Big)=1\text{ bit}
$$

### Micro — sure thing

$p(x_0)=1$ and $0$ elsewhere:

$$
H=-1\cdot\log 1=0
$$

No uncertainty → zero average surprisal.

### Micro — two-point skewed

$p=(0.9,0.1)$, base 2:

$$
H\approx -0.9\log_2 0.9-0.1\log_2 0.1\approx 0.47\text{ bits}
$$

Less mixed than fair coin → lower entropy than $1$.

### Micro — three outcomes (full arithmetic)

$p=(1/2,1/4,1/4)$, base 2:

$$
\begin{aligned}
H
&= -\tfrac12\log_2\tfrac12 -\tfrac14\log_2\tfrac14 -\tfrac14\log_2\tfrac14 \\
&= -\tfrac12(-1) -\tfrac14(-2) -\tfrac14(-2) \\
&= \tfrac12 + \tfrac12 + \tfrac12 = 1.5\text{ bits}
\end{aligned}
$$

More spread than a fair coin’s two sides in a non-uniform way — intermediate entropy. (Uniform on 3 symbols would be $\log_2 3\approx 1.58$ bits, the max for 3 labels.)

**Notice:** entropy is **one number attached to one distribution** (how spread / uncertain it is). Among laws on a fixed finite alphabet, more uniform ⇒ higher $H$; a spike ⇒ $H\to 0$.

---

## 6. Entropy vs divergence (do not mix)

<a id="p6-entropy-vs-div"></a>

### Purpose for the video

ML recipe needs a score between **two** distributions. Entropy is not that score yet.

| Object | Inputs | Meaning |
|--------|--------|---------|
| **Entropy** $H(p)$ | one distribution $p$ | average surprisal inside $p$ |
| **Divergence** $d(p,q)$ | two distributions | how far $p$ is from $q$ |

This lecture builds entropy. Later lectures use info-theory ideas to build divergences (e.g. related to **KL**).  
$f$-divergences = a whole family of such pairwise scores (he defers most of that family).

### How entropy becomes an ingredient of $d$ (preview only — not proved today)

A common next construction (cross-entropy / KL style):

- Score model $q$ by average surprisal of **true** samples under $q$: $\mathbb{E}_{X\sim p}[-\log q(X)]$ (cross-entropy).  
- Compare to entropy of truth $H(p)=\mathbb{E}_{p}[-\log p(X)]$.  
- Their difference is a non-negative gap that is $0$ only when $p=q$ (KL divergence, under mild conditions).

You do **not** need that formula today. You only need: **entropy alone is not recipe step 2**; step 2 needs a **pairwise** score. Entropy is the toolkit’s first brick.

---

## 7. Why this lecture exists in the ML recipe

<a id="p7-recipe-link"></a>

### Purpose for the video

Connect to Lec 10.

```
  (1) choose model p_θ
  (2) distance d(true, model)   ← NEED A DEFINITION
  (3) θ* = argmin d
```

Step (2) is empty without a way to score “how far is model from truth.”  
Historically people used **information theory** to invent such scores.  
Entropy is step one of that toolkit.

**Disclaimer he will stress:** speech often says “distribution”; math often writes **density** when a 1–1 link exists. Probability measure lives on the distribution side.

### Linear model dimension (correction from last class)

If $x\in\mathbb{R}^d$ and you write $w_1^\top x + w_2$:

| Symbol | Correct type |
|--------|----------------|
| $w_1$ | vector in $\mathbb{R}^d$ |
| $w_1^\top x$ | **scalar** |
| $w_2$ | **scalar** (bias), not a second $\mathbb{R}^d$ vector |

---

## 8. Log base: bits vs nats

<a id="p8-log-base"></a>

### Purpose for the video

He writes $\log$ without fixing the base. Units change; ideas do not.

| Base | Unit of entropy / surprisal |
|------|-----------------------------|
| $\log_2$ | **bits** (common in CS / coding) |
| $\ln=\log_e$ | **nats** (common in continuous math / ML papers) |
| $\log_{10}$ | rare for entropy |

Changing base multiplies all values by a constant:

$$
\log_b u=\frac{\ln u}{\ln b}
$$

So rankings and the qualitative story (rare = large surprisal) stay the same.

### Paper check

1. Compute $I(A)$ if $P(A)=1/8$ with $\log_2$.  
2. Why is $I(\Omega)=0$? Why is discrete first (not continuous points)?  
3. Independent fair coins both heads: check $I(A\cap B)=I(A)+I(B)$.  
4. Write $H$ for a fair coin, a sure outcome, and $p=(1/2,1/4,1/4)$.  
5. What do you do with a $p=0$ term in $-\sum p\log p$?  
6. Is entropy a distance between two distributions? What is a divergence?  
7. Which recipe step needs distributional divergences?  
8. Correct types: $w_1$ and $w_2$ in $w_1^\top x+w_2$.  
9. In one sentence: what does “average surprisal per draw” mean for $H(p)$?

---

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).  
Prior: [Lec 10 Challenges](../11-Lec10-Challenges-of-ML/NOTES.md).
