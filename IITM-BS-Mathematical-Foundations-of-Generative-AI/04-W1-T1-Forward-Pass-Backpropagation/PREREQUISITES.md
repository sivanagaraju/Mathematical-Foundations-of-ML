# PREREQUISITES — warm-up before the module

> **Do this first.** Then open [NOTES.md](./NOTES.md) at the Executive Summary map.
> One screen per idea. No prior probability or calculus assumed — every idea below
> is built from a small concrete scene.

```
  After this warm-up you can say in plain words:

  "A distribution is a law that generates samples; a density is its smoothed histogram."
  "An expectation is a weighted average, and a sample average converges to it (LLN)."
  "A convex function is bowl-shaped: the chord never dips below the graph."
  "sup means 'tightest lower bound of the answers', not always a maximum."
  "We can search over functions, not just numbers — a bag of functions."
```

---

## 1. Random variables, distributions, densities

<a id="p1-distributions-densities"></a>

A **random variable** is a thing whose value you do not know until nature "rolls" it:
the height of the next person who walks through the door, tomorrow's temperature,
the next image a camera takes. The **distribution** is the invisible law that decides
which values are common and which are rare.

A **density** is that law drawn as a curve: tall regions = likely values,
flat regions = rare values. Think of it as a smoothed histogram of *every possible
sample ever*. You never see the curve itself; you only see individual draws
(**samples**) that fall out of it.

```
        density curve (the LAW — invisible to us)
           ___---___
        --/         \--__                     
   ____-/                \------____
   height ↑  = region is likely
   samples:        •  •    •   •  •  (single draws we DO see)
```

- Micro example: human height. Law says "most people land near 170 cm".
  A sample is one actual person: 163.5 cm. The sample is real; the curve is invisible.
- In the lecture's language: the data's law is $p_x$ (unknown), the model's law is $p_\theta$.

---

## 2. iid samples — and making fresh samples on demand

<a id="p2-iid-sampling"></a>

**iid** = independent and identically distributed. It means every sample is drawn
from the *same* law and no draw influences another — like rolling the same fair die
again and again.

- **iid dataset:** 5 die rolls `3, 3, 6, 1, 3` — same die (identical), no memory (independent).
- **Not iid:** asking a crowd "how tall are you?" where friends answer after hearing
  each other (not independent), or mixing two different countries (not identical).

Why this matters for the lecture: a dataset $D = \{x_1, \ldots, x_n\}$ is *n iid
draws from the unknown law $p_x$*. And a neural network $G_\theta(z)$ is a machine
that can mint **fresh** samples of its own law $p_\theta$ whenever you want:
feed it easy random numbers $z$ (like die rolls) and it outputs new $\hat{x}$.

```
   easy z  (we sample it: 0.7, -1.2, 0.3, ...)
        │
        ▼
   ┌─────────┐
   │  G_θ    │   neural network with dials θ
   └────┬────┘
        ▼
   fresh x̂   samples of the model's law p_θ — on demand
```

Notice: we never *know* the formula of $p_\theta$; we can just **sample** it.

---

## 3. Expectation — the weighted average

<a id="p3-expectation"></a>

An **expectation** is a weighted average over *every possible value*, where each value
is weighted by how likely it is. It answers: "if I played this game a million times,
what would be the average payoff?"

Micro example — a game die that pays rupees equal to its face:

- Faces `1..6`, each with probability $1/6$.
- Average payoff:

$$
\mathbb{E}[X] = \tfrac{1}{6}\cdot 1 + \tfrac{1}{6}\cdot 2 + \cdots + \tfrac{1}{6}\cdot 6 = 3.5
$$

In words: add up each value times its probability. You never roll a "3.5" —
it is the *long-run average*, a property of the law, not of any single roll.

Now upgrade: instead of averaging the face itself, average some *function* of it.
If the payoff is "square of the face", the expectation of that function is

$$
\mathbb{E}_{p_x}[h(X)] = \sum_x h(x)\, p_x(x)
$$

In words: weight each transformed value $h(x)$ by its probability and add.
This "average of a function of the random thing" is exactly the object the
lecture keeps meeting.

---

## 4. Integrals — sums with infinitely small steps (and where they break)

<a id="p4-integrals"></a>

For a continuous law there are no single-value probabilities, only regions.
The sum from the previous idea becomes an **integral**: a limit of adding
weighted values with infinitely small steps.

$$
I = \int h(x)\, p_x(x)\, dx
$$

In words: over the whole line, add up $h(x)$ weighted by the local density $p_x(x)$.

```
   1-dimension:  area under a curve  → add thin strips  ▌▌▌▌▌▌
   2-dimensions: volume under a sheet → add thin tiles
   d-dimensions: integral over a d-box → must integrate EACH dimension
                 separately … the strip count explodes:
                 100 strips per axis → 100^d little boxes
                 d=100  →  100^100 boxes  (more than atoms in the universe)
```

That explosion is why "just compute the integral" fails for images and text,
where $d$ is in the thousands or millions. The lecture leans on this hard:
high-dimensional integrals involving densities are *practically impossible*,
so any algorithm that requires them is dead on arrival.

---

## 5. The law of large numbers — sample averages converge

<a id="p5-law-of-large-numbers"></a>

Here is the escape hatch. You want $\mathbb{E}_{p_x}[h(X)]$ but you do not know $p_x$.
The **law of large numbers (LLN)** says: draw iid samples from the very law you care
about, average $h$ on those samples, and the average approaches the true expectation
as the sample count grows.

$$
\frac{1}{n}\sum_{i=1}^{n} h(x_i) \;\approx\; \mathbb{E}_{p_x}[h(X)] \qquad (\text{equality as } n \to \infty)
$$

Micro example — estimate the average height of all adults in a city.
You cannot measure everyone (the true law is hidden). Measure 1000 random adults,
average: 168.9 cm. As the sample grows, the wiggle shrinks:

```
   n=10      average 167.2  (wobbly)
   n=100     average 168.7  (closer)
   n=10,000  average 169.0  ≈ the true expectation
```

The one condition the lecture repeats: the samples must come from **the same law**
you are averaging over. Averaging heights measured in country A tells you nothing
about country B's law.

Notice: the LLN converts "an integral I cannot compute" into "an average of numbers
I already hold". That conversion is the engine of the whole module.

---

## 6. Convex functions — bowls, chords, and combinations

<a id="p6-convexity"></a>

A **convex combination** of two points is any point strictly *between* them:
half of one plus half of the other, or 0.7 of one plus 0.3 of the other —
any split $\alpha_1, \alpha_2 \ge 0$ with $\alpha_1 + \alpha_2 = 1$.

A function is **convex** if, between any two of its points, the straight line
joining them (the chord) never dips below the curve. Bowls are convex; waves are not.

```
   CONVEX (bowl):                    NOT convex (wave):
   ●───────●   chord                  ●───────●
    \     /    stays above             \  ●  /   chord DIPS
     \   /      the curve               \___/    below the curve
      \_/                               
   x², |x|, e^x are convex.  sin(x) is not.
```

Formally, for any two inputs $u_1, u_2$ and convex-combination weights
$\alpha_1+\alpha_2=1$:

$$
\alpha_1 f(u_1) + \alpha_2 f(u_2) \;\ge\; f(\alpha_1 u_1 + \alpha_2 u_2)
$$

In words: "the average of the heights is at least the height of the average."
Convex functions are the friendly ones: no hidden dips, every local minimum is
the global one — which is why optimizers love them and why the lecture needs
this property before defining the conjugate.

---

## 7. sup vs max — the tightest answer, even if never touched

<a id="p7-supremum"></a>

**max** is the biggest value you actually attain. **sup** (supremum) is the smallest
ceiling that sits above all values — even if nothing ever touches that ceiling.

Micro example: the numbers $0.9, 0.99, 0.999, \ldots$ never reach $1$.
There is no max. But the sup is exactly $1$ — the tightest upper bound.

```
   values:  0.9   0.99   0.999   0.9999  ──→
   ceiling:  ═══════════════ 1.0 ═══  sup = 1 (never attained)
```

When the lecture writes $\sup_u \{ut - f(u)\}$, read it as: "scan every $u$,
record $ut - f(u)$, and keep the tightest answer" — call it the max when the
best value is actually attained (often the case here). The word "sup" just keeps
the statement honest for edge cases.

---

## 8. Functions as objects — a bag you can search

<a id="p8-functions-as-objects"></a>

You are used to optimizing **numbers**: find the best temperature, the best price.
The lecture will optimize over **functions**: whole input→output rules as single objects.

Picture a bag full of candidate rules for "how loud the ice-cream truck should be
given the temperature":

```
   the bag  𝒯 = { rule A, rule B, rule C, … }
     rule A: temp → 10 dB always        (flat rule)
     rule B: temp → 2 × temp            (ramps up)
     rule C: temp → 60 dB at 30°C only  (spiky rule)
   search = pick the ONE rule from the bag that does best EVERYWHERE
```

One member of the bag is written $T(x)$ — a rule that answers at every input $x$.
"Supremum over $\mathcal{T}$" means: try every rule in the bag, at every input,
and keep the best. The bag's contents matter enormously: if the perfect rule is not
inside the bag, the best answer you find is only a stand-in — that is why the
lecture's final equality becomes a bound.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).
Quiz later: [quiz.html](./quiz.html) — Part A covers this file.
