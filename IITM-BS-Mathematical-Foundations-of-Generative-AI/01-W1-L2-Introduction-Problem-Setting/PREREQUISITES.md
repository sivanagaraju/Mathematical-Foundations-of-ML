# Prerequisites — warm-up before W1_L2 (Introduction & problem setting)

> **Do this first** if “random variable,” “IID,” “density,” “parameter $\theta$,” or “sample from a model” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Course: IIT Madras BS **Mathematical Foundations of Generative AI** · after [W1_L1 course outline](https://www.youtube.com/watch?v=skWhn8W9P_Y).  
> **Beginner deep warm-up:** definition · worked micro · analogy · notice · mini-check for each idea.

```
  After this warm-up you can say:

  "A random variable is a rule that turns an unseen outcome into a number (or a list of numbers)."
  "A distribution is the rule for which values show up; a density is a height used to compute that rule."
  "IID means: each photo is an independent draw from the same unknown law — not that pixels inside one photo are independent."
  "An image is a long vector: stack every pixel-channel into R^D."
  "A parametric family p_θ is a catalog of candidate laws, indexed by knobs θ."
  "A divergence d(p,q) scores how far two laws sit; 0 usually means they match."
  "Pushing a known random z through a fixed function G produces a new random x whose law depends on G."
  "Training is argmin_θ of that score."
```

**Warm-up → lecture boxes**

```
  §1  Random variable vs a number          ──► Topics 4, 8
  §2  Distribution vs density              ──► Topics 2, 8–9
  §3  IID (across samples)                 ──► Topics 2, 4
  §4  Vectors / flattening an image        ──► Topic 3
  §5  Parametric family p_θ                ──► Topic 6
  §6  Divergence as “how far”              ──► Topics 7, 9
  §7  Function of a random variable        ──► Topics 8–9
  §8  Optimization / argmin                ──► Topics 7, 9–10
```

---

## 1. Random variable vs a number

<a id="p1-rv"></a>

### Purpose for the video

The lecture’s data points $x_i$ are **not** “just files.” They are **instantiations** of a random variable.

### Definitions

| Idea | Meaning |
|------|---------|
| **Outcome** | What actually happened once (this coin landed heads; this camera clicked now) |
| **Random variable (RV)** | A **function** that turns that unseen outcome into a number or a list of numbers |
| **Instantiation / sample** | One concrete value of that function after one run |

### Worked micro

Die roll: the hidden face is the outcome. $X=$ “the number showing” is the RV. Tonight you see $X=4$. The $4$ is an instantiation. The RV is the *rule*, not tonight’s $4$.

Photo: the hidden scene in front of the lens is the outcome. $X=$ “stack the pixels into a list.” One JPEG is one instantiation.

### Analogy — weather station

The atmosphere is messy and hidden. The station prints **one temperature**. That printout is a number. The **thermometer rule** (air → Celsius) is the random variable. You store printouts; you never store “the atmosphere.”

### Second analogy — lottery machine vs tonight’s ticket

The sealed drum of balls is the hidden experiment. The printed ticket “42” is **tonight’s draw** (instantiation). The rule “read the number on the ball” is the random variable. Training a generative model is studying thousands of tickets without opening the drum.

### Notice

- Randomness is *which* outcome occurred. The RV only **labels** it with numbers.  
- Later the lecture needs a **vector-valued** RV: one outcome → a $D$-list, not a single float.

### Mini-check

1. Is “$23^\circ$C on Tuesday” the RV, or an instantiation?  
2. What does the RV output for an image?

---

## 2. Distribution vs density

<a id="p2-density"></a>

### Purpose for the video

He writes a **script** $\mathbb{P}_x$ for a distribution and a **plain** $p_x$ for a density, then admits he will mix the words.

### Definitions

| Object | Plain meaning |
|--------|----------------|
| **Distribution** $\mathbb{P}_x$ | The full assignment of probability to regions of $\mathbb{R}^D$ (“how much mass in this blob of pixel-space”) |
| **Density** $p_x$ | A **height** function: probability of a small box ≈ height × volume. Height itself is **not** a probability |

### Worked micro

Uniform on $[0,0.5]$: the density height is $2$ (so area $=1$). Saying “probability $=2$” is wrong. The distribution says every sub-interval of length $\ell$ inside $[0,0.5]$ has probability $2\ell$.

### Analogy — jam on toast

The **amount of jam** on a region of toast is probability. How **thick** you spread it is density. A thick blob on a tiny crumb can still be a small amount of jam.

### Notice

- Continuous data (images as real vectors) almost always needs a **density** to write formulas.  
- The lecture will say “distribution” when it means “density” for ease. Keep the distinction in your pocket.

### Mini-check

1. Can a density be larger than $1$?  
2. What object is unknown at the start of generative modeling: the files, or $p_x$?

---

## 3. IID — independent and identically distributed

<a id="p3-iid"></a>

### Purpose for the video

The most common trap in this lecture: IID is **across photos**, not **across pixels**.

### Definitions

| Word | Meaning for a dataset $\{x_1,\ldots,x_n\}$ |
|------|---------------------------------------------|
| **Identically distributed** | Every $x_i$ is drawn from the **same** unknown law $p_x$ |
| **Independent** | Knowing $x_1$ does not change the law of $x_2$ |
| **IID** | Both at once |

### Worked micro

1000 vacation photos. Photo 1 and photo 100 are taken at different times by the same “world of photos” process: IID is reasonable. **Inside** photo 1, the sky pixel and its neighbour are **not** independent — they share a scene.

Split the two letters: **I** = photo 7 does not change the law of photo 8. **ID** = both photos came from the **same** unknown $p_x$, not from two different worlds. The lecture’s GPT remark is the same convenience at internet scale: one $p_x$ for “all documents,” because estimating many unknown laws at once is harder.

### Analogy — cookies from one recipe

Each cookie is pulled from the **same batter** (identical). Pulling cookie 7 does not tell you cookie 8’s chips (independent draws). Chocolate chips **inside** one cookie are clumped — that is *not* what IID forbids.

### ASCII

```
  OK to assume IID:     photo_1  ⟂  photo_2  ⟂  …  ⟂  photo_n
                        all ~ same p_x

  NOT claimed:          pixel_1  ⟂  pixel_1000   inside one photo
```

### Notice

- One shared $p_x$ is **mathematical ease**: estimating many unknown laws at once is harder.  
- GPT-scale: “one $p_x$ for the internet” is the same convenience, not a claim that the web is one tiny topic.

### Mini-check

1. Does IID say neighbouring pixels are independent?  
2. Why assume *one* $p_x$ rather than a different law per photo?

---

## 4. Vectors and flattening an image

<a id="p4-vectors"></a>

### Purpose for the video

A color photo is a **tensor**. The math of this course treats it as one point in $\mathbb{R}^D$.

### Definitions

| Object | Meaning |
|--------|---------|
| **Vector in $\mathbb{R}^D$** | An ordered list of $D$ real numbers |
| **Tensor (here)** | A 3-way array: rows $\times$ columns $\times$ 3 color channels |
| **Flatten / stack** | Read every entry into one long list; $D = R\cdot C\cdot 3$ |

### Worked micro

$400\times 400$ RGB image: $D = 400\times 400\times 3 = 480{,}000$. That **one** photo is one point in a 480{,}000-dimensional space. The lecture writes this number on the board.

A smaller check: MNIST is $28\times 28$ grayscale → $D=784$. Same idea, tiny $D$.

```python
# Stacking procedure (what the chalkboard cube becomes as a list).
# Not from this lecture's IDE — there isn't one — but this IS the map he draws.
# image.shape == (R, C, 3)   e.g. (400, 400, 3)
# x = image.reshape(-1)      # length D = R*C*3 = 480_000
```

### Analogy — packing a suitcase

A folded shirt has width, height, and layers (collar, pocket, lining). For the airline scale you still get **one weight**. Flattening is packing: geometry becomes one list so the same algorithms work for images, audio windows, or token embeddings.

### ASCII

```
  R rows
  ┌─────────────┐
  │  R × C × 3  │  ──stack──►  (x_1, x_2, …, x_{R·C·3})  ∈ R^D
  │   C cols    │
  └─────────────┘
     3 = RGB
```

### Notice

- Algorithms later **do not care** that the list “used to be” a rectangle.  
- High $D$ is the default (tens/hundreds of thousands), not a special case.

### Mini-check

1. Compute $D$ for a $28\times 28$ grayscale MNIST digit (one channel).  
2. Is $D$ the number of **images** or the length of **one** image-vector?

---

## 5. Parametric family $p_\theta$

<a id="p5-parametric"></a>

### Purpose for the video

You cannot search “all possible laws on $\mathbb{R}^{480000}$.” You pick a **family** with knobs $\theta$.

### Definitions

| Idea | Meaning |
|------|---------|
| **Parameter $\theta$** | A list of knobs (neural-net weights, or $\mu,\sigma$ for a Gaussian) |
| **Parametric family** $\{p_\theta\}$ | One candidate law for each knob setting |
| **Model** (this course’s slang) | The family $p_\theta$ you assumed for $p_x$ |

### Worked micro

1-D Gaussian family: $\theta=(\mu,\sigma)$. $p_\theta$ is a bell. Changing $\mu$ slides it; changing $\sigma$ stretches it. You still have to **pick** $\theta$ from data.

Deep net: $\theta$ is millions of weights. The net is just a very flexible $p_\theta$ (or a generator whose output **has** law $p_\theta$).

### Analogy — radio presets

The air has one true station ($p_x$). Your radio can only tune a **dial** $\theta$. Each dial setting is a candidate $p_\theta$. Training turns the dial until the candidate sounds like the true station.

### Notice

- Universal approximation (later in the lecture): a wide enough net can mimic almost any function, which is why modern $p_\theta$ is a **neural net**.  
- “Model” in this course **does not** mean “the whole trained product.” It means $p_\theta$.

### Mini-check

1. If $\theta$ changes, does $p_x$ (the unknown truth) change?  
2. What does the instructor mean by **model**?

---

## 6. Divergence as “how far two laws sit”

<a id="p6-divergence"></a>

### Purpose for the video

Training is: make $p_\theta$ close to $p_x$ by minimizing a score $d$.

### Definitions

| Idea | Meaning |
|------|---------|
| **Divergence** $d(p\|q)$ | A score of how unlike two probability laws are |
| **Typical properties** (this lecture) | $d\ge 0$, and $d=0$ **iff** the two laws match |
| **Not always a metric** | May fail symmetry or the triangle inequality (later weeks) |

### Worked micro

Two bells on the line. If they sit on top of each other, $d=0$. If one is far to the right, $d$ is large. Training slides the model bell toward the data bell.

### Analogy — two piles of sand

You want your sand pile $p_\theta$ to match the true pile $p_x$. Divergence is how many shovel-moves you still need. Zero shovels iff the piles coincide.

### Notice

- **Begging the question:** $d$ is written in terms of $p_x$, which we **do not know**. The rest of the course is “compute $d$ from **samples** only.”  
- Notation on the board: $d(p_x\|p_\theta)$ with two vertical bars.

### Mini-check

1. If $d(p_x\|p_\theta)=0$, what is true of the two laws (in this lecture’s setup)?  
2. Why is “just compute $d$ from the formula” blocked at the start?

---

## 7. A function of a random variable

<a id="p7-transform"></a>

### Purpose for the video

This is the **sampling engine**: $z$ known → $G_\theta(z)=\hat x$ whose law depends on $G_\theta$.

### Definitions

| Idea | Meaning |
|------|---------|
| **Deterministic function** $G$ | Same input always gives the same output (a neural net with frozen weights) |
| **Pushforward / transform** | If $Z$ is random and $G$ is fixed, then $G(Z)$ is **also** random, with a **new** law |

### Worked micro

$Z\sim \mathrm{Uniform}[0,1]$. Let $G(z)=z+2$. Then $G(Z)$ is uniform on $[2,3]$ — different law, still easy to sample: draw $z$, add $2$.

Standard trick: $Z\sim\mathcal N(0,I_k)$ in **$k$ dimensions** (you **know** how to draw it). $k$ is the noise size; the photo lives in $\mathbb{R}^D$. They need not match. A net $G_\theta:\mathbb{R}^k\to\mathbb{R}^D$ warps Gaussian blobs into image-shaped $\hat x$.

```python
# How "sample from a known Gaussian" looks as RNG (board: z ~ N(0, I)):
# k = 128          # noise dimension — a choice, not D
# z = numpy.random.randn(k)   # mean 0, variance 1, independent coordinates
# x_hat = G_theta(z)          # now length D, e.g. 480_000
```

### Analogy — pasta maker

You know how to dump flour ($z$ from a known bag). The machine $G$ (shape of the die) is deterministic. Changing the die changes the pasta shape (the law of $\hat x$). Training is choosing the die so the pasta looks like real photos.

### ASCII

```
  z  ~  known easy law (e.g. N(0,I))
   │
   │  G_θ  (deterministic net)
   ▼
  x̂ = G_θ(z)  ~  p_θ     (a different law)
```

### Notice

- You do **not** need a formula for $p_\theta(\hat x)$ to **draw** $\hat x$: run $G_\theta$ on a fresh $z$.  
- That is why implicit models can sample without writing the density.

### Mini-check

1. If $G$ is fixed and you draw a new $z$, do you get the same $\hat x$?  
2. Who determines the law of $\hat x$: $z$’s law, $G$, or both?

---

## 8. Optimization / $\arg\min$

<a id="p8-argmin"></a>

### Purpose for the video

The last recipe step: $\theta^\star=\arg\min_\theta d(p_x\|p_\theta)$.

### Definitions

| Idea | Meaning |
|------|---------|
| **Objective** | A number that depends on $\theta$ (here: the divergence) |
| **$\arg\min_\theta$** | The knob setting that makes that number as small as possible |
| **$\theta^\star$** | That winning setting; $G_{\theta^\star}$ is the **trained** net |

### Worked micro

Suppose $d(\theta)=(\theta-3)^2$. Then $\theta^\star=3$. In deep nets you cannot solve this on paper; you take gradient steps. The lecture does **not** yet pick SGD vs Adam — that is later “how to optimize.”

### Analogy — thermostat

$d$ is “how wrong the room feels.” $\theta$ is the thermostat. $\arg\min$ is the setting where the wrongness is smallest. After that, you **leave the thermostat there** and just live in the room (sample).

### Commented sketch of the whole recipe

The lecture is chalk, not an IDE. This is the **same algorithm** as the board, written so a beginner can see the moving parts:

```python
# Data: n vectors drawn IID from unknown p_x
# D = [x1, x2, ..., xn]    # we HAVE these
# p_x                      # we do NOT have this function

# Step 1 — model: a net G_theta that maps easy z -> x_hat
# Step 2 — score: d(p_x, p_theta)  (how we compute d is NEXT lectures)
# Step 3 — train:
#     theta_star = argmin_theta  d(p_x, p_theta)
#
# After training, SAMPLE (this we CAN do):
#     z = sample_standard_normal()   # known
#     x_new = G_theta_star(z)        # new point, not a copy of D
```

### Notice

- Success means $p_{\theta^\star}$ is **close** to $p_x$, so $x_{\mathrm{new}}$ behaves like a fresh draw from the true law — **not** a replay of the dataset.  
- Four leftover questions (last topic): compute $d$ without densities; choose $d$; choose $G_\theta$; how to optimize.

### Mini-check

1. After you have $\theta^\star$, how do you get a **new** image?  
2. Why is a new sample allowed to look unlike every training file?

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
