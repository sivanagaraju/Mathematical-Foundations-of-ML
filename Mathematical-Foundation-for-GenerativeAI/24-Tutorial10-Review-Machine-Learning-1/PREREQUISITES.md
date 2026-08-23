# Prerequisites — warm-up before Tutorial 10 (Review of Machine Learning 1)

> **Do this first** if “likelihood,” $\Phi$, “mixture,” or “EM” still feel like slogans.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues the first-course recap after [Tutorial 9](../23-Tutorial09-Review-Basic-Probability-3/NOTES.md).  
> **Beginner deep warm-up:** purpose · definition · micro numbers · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "A likelihood scores a parameter for data we already saw."
  "IID means the same machine, independent draws."
  "Φ(z) is the left-hand area under the standard Normal."
  "Log turns a product of tiny numbers into a sum we can differentiate."
  "A coin with m heads in n tosses has MLE p̂ = m/n."
  "A mixture first tosses a coin, then draws from that machine."
  "A latent Z is the missing label; complete data is (x, z)."
  "EM: guess soft labels, refit, repeat."
```

**Warm-up → tutorial boxes**

```
  §1  Likelihood scores a parameter     ──► Topics 2–5
  §2  IID copies of one machine         ──► Topics 2, 7
  §3  Standard Normal and Φ             ──► Topics 3, 5
  §4  Why we take the log               ──► Topics 4, 7, 9
  §5  Bernoulli / fraction MLE          ──► Topic 4
  §6  Mixture = coin then a machine     ──► Topic 6
  §7  Latent Z and complete data        ──► Topics 6–8
  §8  Bayes line + the EM loop          ──► Topics 8–10
```

---

## 1. Likelihood scores a parameter, given data

<a id="p1-likelihood"></a>

### Purpose for the video

The first numerical problem writes a **likelihood** $L(\mu)$ and then maximizes it. That is not “the probability of $\mu$.”

### Definitions

A **probability** $P(\text{data}\mid\text{parameter})$ looks forward: if I knew $\mu$, how often would I see this kind of data?

A **likelihood** $L(\text{parameter}\mid\text{data})$ looks backward: I already have the data; which $\mu$ makes that data least surprising?

Same formula, different job. Once the notebook is written, $\mu$ is the variable we slide, and the data are frozen.

### Worked micro

A biased coin has $P(\text{heads})=p$. You saw **H, H, T**.

$$
L(p)=p\cdot p\cdot(1-p)=p^2(1-p)
$$

Try $p=0.1$: $L=0.009$. Try $p=0.9$: $L=0.081$. Try $p=2/3$: $L\approx 0.148$. The last score is best among those three. The data did not change. The parameter did.

### Analogy — tasting three soup recipes after the pot is cooked

The pot is already on the table (the data). You cannot recook it. You can only ask: which recipe card (which $p$ or $\mu$) would most often have produced *this* pot?

Picking the recipe that best explains the pot is **maximum likelihood**. Guessing “the recipe *is* $p=0.7$ with probability 0.9” would be a different game.

That second game is **maximum a posteriori (MAP)**. He names MAP as homework and does **not** derive it. The one-line difference: MLE maximizes $L(\theta\mid\text{data})$; MAP multiplies that likelihood by a **prior** $P(\theta)$ (what you believed before the notebook) and then maximizes. If the prior is flat, MAP collapses to MLE. In this hour every derivative is MLE.

### Notice

- $L$ is a function of the **parameter**. The data sit still.
- You may only use what was **written down**. If the notebook recorded *signs* instead of numbers, $L$ is built from sign-probabilities, not from the missing $x_i$.
- A likelihood is **not** required to integrate to 1 over $\theta$. Do not treat $L(\mu)$ as a density of $\mu$.

### Mini-check

1. If $n$ changes but $\mu$ is fixed, is $L$ the same curve?  
2. Is $L(\mu)$ a density in $\mu$? (No — it need not integrate to 1 over $\mu$.)

---

## 2. IID: the same machine, independent draws

<a id="p2-iid"></a>

### Purpose for the video

Both problems say “$n$ IID realizations.” That one phrase lets us **multiply** $n$ one-point likelihoods.

### Definitions

**Identical:** every draw uses the *same* unknown law (same $\mu$, or same mixture $\theta$).

**Independent:** knowing $x_1$ does not change the law of $x_2$. Joint density $=$ product of one-point densities.

Together: **IID** = independent *and* identically distributed.

### Worked micro

Three tickets from one Normal copier $\mathcal{N}(\mu,1)$:

```
  copier θ = N(μ, 1)
     │
     ├── ticket 1  (does not change the copier)
     ├── ticket 2
     └── ticket 3
  joint = f(x1) · f(x2) · f(x3)
```

If ticket 2 came from a *different* mean, the product with one shared $\mu$ would be a lie.

### Analogy — one espresso machine, many cups

Same grind setting all morning (identical). One cup does not change the next cup (independent). The day’s logbook is $n$ cups from **one** setting, not $n$ different cafés.

### Notice

- Independence is *across observations*, not “$X$ independent of $\mu$.” $\mu$ is a parameter, not a second random variable in this lecture.
- When a hidden label $Z_i$ appears later, the pairs $(X_i,Z_i)$ are still IID.

### Mini-check

1. Same machine, but cup 2 is poured only if cup 1 overflowed — IID or not?  
2. Why does IID turn a joint into a product?

---

## 3. The standard Normal and the area $\Phi$

<a id="p3-phi"></a>

### Purpose for the video

The sign problem never needs the Normal *density* of the missing $x_i$. It needs the **area to the left** of a threshold.

### Definitions

$Z\sim\mathcal{N}(0,1)$ is the **standard Normal**: bell centered at 0, variance 1.

Its **CDF** is

$$
\Phi(z)=P(Z\le z)=\int_{-\infty}^{z}\frac{1}{\sqrt{2\pi}}e^{-t^2/2}\,dt
$$

In words: $\Phi(z)$ is the shaded area of the bell from $-\infty$ up to $z$.

If $X\sim\mathcal{N}(\mu,1)$, then $Z=(X-\mu)/1$ is standard. That one subtraction is **standardization** (a $z$-score with $\sigma=1$). Moving the mean slides the bell; the threshold $0$ stays put. The fence stays; the hill moves.

**Inverse** $\Phi^{-1}(q)$ is the number $z$ whose left-area is $q$. It is the $q$-quantile of $\mathcal{N}(0,1)$.

Two symmetries you will use with a pen:

$$
\Phi(-z)=1-\Phi(z),\qquad -\Phi^{-1}(q)=\Phi^{-1}(1-q)
$$

### Worked micro

Bell centered at $\mu=+1$. Threshold at $0$, which sits **left** of the peak.

```
         density of N(+1, 1)
  -----.------▲-----------
       0      μ=+1
       | left area |  right area
       |  Φ(-1)    |  Φ(+1)
       |  ≈0.16    |  ≈0.84
```

About 16% of draws are negative. If someone only tells you the sign, a pile of minuses is evidence that $\mu$ is itself negative.

**Inverse walk.** Suppose the left-area is $0.30$. Which $z$ has $\Phi(z)=0.30$? Tables (or `norm.ppf(0.3)`) give $z\approx -0.52$. That $z$ is $\Phi^{-1}(0.30)$. If $0.30=\Phi(-\hat\mu)$, then $-\hat\mu\approx -0.52$, so $\hat\mu\approx +0.52$. Area in, location out.

### Analogy — a hill and a fence

The hill is the Normal. The fence is $0$. $\Phi(-\mu)$ is “how much hill sits to the left of the fence.” Slide the hill (change $\mu$) and the left-of-fence area changes. You never climb the hill; you only count which side of the fence each sheep landed.

### Notice

- $\Phi$ is *not* the inverse function. $\Phi^{-1}$ is.
- $P(X>0)=1-\Phi(-\mu)=\Phi(\mu)$ uses symmetry, not a new table.

### Mini-check

1. If $\mu=0$, what is $P(X<0)$?  
2. If $\Phi(a)=0.9$, what is $\Phi(-a)$?

---

## 4. Why we take the log

<a id="p4-log"></a>

### Purpose for the video

Every maximization in this hour is done on a **log-likelihood**. The log is a calculator trick that does not change the winner.

### Definitions

$\log$ (any base $>1$) is **strictly increasing**, so

$$
\arg\max_\theta L(\theta)=\arg\max_\theta \log L(\theta)
$$

And $\log(ab)=\log a+\log b$, $\log(a^k)=k\log a$. A product of $n$ tiny densities becomes a **sum** of $n$ terms. Differentiating a sum is term-by-term.

### Worked micro

$L(p)=p^3(1-p)^2$. Ugly product rule.

$\ell(p)=3\log p+2\log(1-p)$. Derivative: $3/p-2/(1-p)$. Set to $0$: $p=3/5$. Same winner as maximizing $L$, less algebra.

### Analogy — adding decibels instead of multiplying faint whispers

Each observation is a faint “how much does this $\theta$ like me?” Multiplying 200 whispers underflows a calculator. Adding their logs is the same ranking, in a unit humans (and derivatives) can handle.

### Notice

- He writes capital $L$ for the product and script/lowercase $\ell$ (or still $L$) for the log. Track which one has the $\log$ already applied.
- Later, the **Q-function** is an *expected* complete-data log-likelihood — same “log so we can add” idea, plus an average over missing labels.

### Mini-check

1. Does $\log L$ have the same *value* as $L$ at the maximum? (No — same *location*.)  
2. Why is $\frac{d}{dp}\log p=1/p$ nicer than differentiating $p^m$?

---

## 5. A coin, $m$ successes, $\hat p=m/n$

<a id="p5-bernoulli"></a>

### Purpose for the video

After a substitution, the sign problem **is** a coin problem. You should already know its MLE in your sleep.

### Definitions

A **Bernoulli** trial is one flip: success with probability $p$, failure with $1-p$.

$n$ independent trials with $m$ successes have likelihood

$$
L(p)=p^m(1-p)^{n-m}
$$

(ignoring a binomial coefficient that does not depend on $p$).

Log, differentiate, set to zero:

$$
\frac{m}{p}-\frac{n-m}{1-p}=0\quad\Rightarrow\quad \hat p=\frac{m}{n}
$$

The MLE is just the **observed fraction**.

### Worked micro

$n=10$ flips, $m=3$ heads. $\hat p=0.3$. Not $0.5$. Not “maybe 0.4 because coins should be fair.” The data’s own fraction wins.

### Analogy — a jar of black and white stones

You drew 10 stones, 3 black. You are not allowed to look inside the jar. The single number that best explains *this* handful is $3/10$. That is MLE for a coin, dressed as a jar.

### Notice

- You may *rename* what “success” means. In the lecture, **negative sign = success**, so $m/n$ estimates $P(X<0)$, not $P(X>0)$.
- After you have $\hat p$, you still have to **invert** $\Phi$ to get back to $\hat\mu$. The coin is a waypoint, not the destination.

### Mini-check

1. If success is “negative,” and $m=7$, $n=10$, what is $\widehat{P}(X<0)$?  
2. Why does a constant $\binom{n}{m}$ not change the MLE?

---

## 6. A mixture is a coin, then a machine

<a id="p6-mixture"></a>

### Purpose for the video

Problem 2’s density is not “an exponential with a fancy $\beta$.” It is **two** exponentials glued by a coin.

### Definitions

A **two-component mixture**

$$
f(x)=\pi\,f_1(x)+(1-\pi)\,f_2(x)
$$

means: with probability $\pi$ draw from machine 1; otherwise from machine 2. The number $\pi$ is a **mixing weight**, not a mean.

Here each machine is **exponential** with **rate** $\beta>0$:

$$
f(x\mid\beta)=\beta e^{-\beta x},\qquad x>0
$$

Mean waiting time is $1/\beta$. A *larger* $\beta$ is a *faster* clock (shorter waits).

Constraints: $0\le\pi\le 1$, $\beta_1>0$, $\beta_2>0$.

### Worked micro

$\pi=0.3$, $\beta_1=2$, $\beta_2=0.5$. For a wait $x=1$:

```
  coin  P(machine 1)=0.3     P(machine 2)=0.7
           │                       │
        2 e^{-2}                 0.5 e^{-0.5}
        ≈0.271                   ≈0.303
  mixture height = 0.3·0.271 + 0.7·0.303 ≈ 0.293
```

You see $x=1$. You do **not** see which machine rang.

### Analogy — two kitchens, one dining room

A waiter flips a coin, then fetches a plate from kitchen 1 or kitchen 2. You only taste the plate. The menu of the dining room is a **blend**. Recovering both kitchens’ recipes plus the coin’s bias is the EM problem.

A second picture: two bus lines share one stop. Line A comes often (large $\beta$, short waits). Line B comes rarely (small $\beta$). You record the wait until *some* bus arrives, but you never wrote which line it was. The waiting-time histogram is a mixture. EM tries to un-blend the two clocks.

### Notice

- A mixture density is **one** number $f(x)$, already summed over the hidden kitchen.
- If you *knew* the kitchen, you would just do ordinary MLE on that kitchen’s plates. You do not know it.

### Mini-check

1. If $\pi=1$, what does the mixture collapse to?  
2. Is $\beta_1$ allowed to be negative? Why not?

---

## 7. Latent $Z$ and complete versus incomplete data

<a id="p7-latent"></a>

### Purpose for the video

EM invents a missing label so the math looks like ordinary complete-data MLE, then *averages* over the missingness.

### Definitions

A **latent** variable $Z$ is a quantity that *participated* in generating $X$ but was not written in the notebook.

Here $Z\in\{0,1\}$: $Z=1$ means “component 1 cooked this $x$.” $Z\sim\mathrm{Bernoulli}(\pi)$.

**Incomplete data:** the list $x_1,\ldots,x_n$ (what we have).

**Complete data:** the pairs $(x_1,z_1),\ldots,(x_n,z_n)$ (what would make MLE easy).

A useful algebra trick: if $z\in\{0,1\}$,

$$
\bigl[\pi\beta_1 e^{-\beta_1 x}\bigr]^{z}\bigl[(1-\pi)\beta_2 e^{-\beta_2 x}\bigr]^{1-z}
$$

keeps exactly one factor and kills the other (anything$^0=1$).

### Worked micro

$x=1.0$, $z=1$. Complete density uses only kitchen 1: $\pi\cdot 2\cdot e^{-2}$.  
Same $x$, $z=0$: only kitchen 2.  
Same $x$, $z$ unknown: you must use the mixture sum — or take an expectation over $z$.

### Analogy — graded homework with the name torn off

Each exam ($x$) was written by class A or class B ($z$). The pile on the desk has no names. **Complete** would be “exam + class tag.” **Incomplete** is the pile. EM pencil-guesses a *soft* tag (70% class A) instead of a hard stamp.

### Notice

- $Z$ is not “noise.” It is the missing switch.
- Expectation of a $0/1$ variable *is* a probability: $\mathbb{E}[Z_i\mid x_i]=P(Z_i=1\mid x_i)$.

### Mini-check

1. If $z=1$, what is $(1-z)$? What happens to the second factor?  
2. Why is $\mathbb{E}[Z\mid x]$ a number between 0 and 1?

---

## 8. The EM loop: soft labels, then refit

<a id="p8-em"></a>

### Purpose for the video

The second half of the hour is one EM cycle written in closed form, then “repeat.”

### Definitions

**Bayes, one line.** If a plate can come from kitchen 1 or kitchen 2,

$$
P(\text{kitchen 1}\mid\text{plate})=\frac{P(\text{plate}\mid\text{kitchen 1})\,P(\text{kitchen 1})}{P(\text{plate})}.
$$

Numerator = “how this kitchen cooks” times “how often we pick it.” Denominator = the mixture height — both kitchens added. That fraction *is* the E-step. Without Bayes, $\gamma_i$ is a slogan.

**E-step (expectation).** Freeze the current parameters $\theta^{\mathrm{old}}$. For each point, compute the posterior probability it came from component 1 (a **responsibility** $\gamma_i$). That is the Bayes line above.

**Q-function.** Replace every unknown $z_i$ in the complete log-likelihood by $\gamma_i$. Some books never say “$Q$”; they still do this replacement.

**M-step (maximization).** Treat the $\gamma_i$ as fixed weights. Maximize $Q$ in $\pi,\beta_1,\beta_2$. Here each update has a **closed form** (no gradient descent).

**Iterate.** The new $\theta$ becomes $\theta^{\mathrm{old}}$. Repeat until the numbers stop moving.

```
  guess θ_old
       │
       ▼
  E: γ_i = P(component 1 | x_i, θ_old)     ← Bayes
       │
       ▼
  Q: stick γ_i where z_i used to be
       │
       ▼
  M: π, β1, β2  ← closed-form weighted MLEs
       │
       └── loop ──►
```

### Worked micro

Two waits: $x=(1,4)$. Suppose $\gamma=(0.8,0.2)$ after an E-step.

- $\pi^{\mathrm{new}}=(0.8+0.2)/2=0.5$.
- $\beta_1^{\mathrm{new}}=(0.8+0.2)/(0.8\cdot 1+0.2\cdot 4)=1/1.6=0.625$.

Component 1 is being fit mostly to the short wait, with a little leak from the long wait. That *is* the M-step idea.

### Analogy — two clubs sharing a guest list with smudged ink

Each name is “probably Club A (80%) / Club B (20%).” You re-estimate each club’s typical arrival time using those soft memberships, then re-read the smudged ink with the new typical times. Nobody ever sees a hard stamp. The list still gets sharper.

**Hard question the ink cannot answer by memory:** after one rewrite of typical times, are the 80/20 notes still correct? No — the notes were computed from *last* week’s times. That is why you loop.

### Notice

- E uses **old** $\theta$ only. M uses **$\gamma$** only. Do not mix a half-updated $\beta$ into the same E-step.
- Closed form here is a gift of exponentials. A Gaussian mixture has a similar *shape* (weighted mean / variance) but different algebra.

### Mini-check

1. If every $\gamma_i$ is $0$ or $1$, what has EM collapsed to?  
2. Why must $\beta^{\mathrm{new}}$ have $x_i$ in the **denominator** for a *rate*?

---

**Second teachers (names only here).** StatQuest, Khan Academy, Seeing Theory, Do & Batzoglou. The actual pointers live at the end of [NOTES.md](./NOTES.md#external-references) — a handful of well-known items, not a link dump.

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).
