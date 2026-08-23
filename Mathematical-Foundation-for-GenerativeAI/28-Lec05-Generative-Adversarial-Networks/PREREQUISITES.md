# Prerequisites — warm-up before Lec 05 (Generative Adversarial Networks)

> **Do this first** if “saddle,” “sigmoid,” “freeze the critic,” or “conditional” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues [Lec 04 VDM](../27-Lec04-Variational-Divergence-Minimization/NOTES.md).  
> **Beginner:** purpose · definition · micro numbers · analogy · ASCII · notice · mini-check.

Last hour left a **score** $J(\theta,w)$ and a **saddle**: maximize in the critic’s weights $w$, minimize in the sampler’s weights $\theta$. This hour **implements** that saddle for **one** choice of $f$, and names the result a **GAN**. Every idea below is a word on the master map. None of them is the lecture itself.

```
  After this warm-up you can say:

  "Two nets share one number J — that is the VDM saddle."
  "The generator is a sampler: noise Z in, fake x̂ out."
  "The critic T builds a lower bound; for GAN’s f it becomes a D in (0,1)."
  "The last layer of T must land in the domain of f*."
  "A sigmoid squashes a real number into (0,1)."
  "A batch average stands in for an expectation."
  "Freeze one net while you step the other."
  "A prompt is a draw of the conditioner Y, not of Gaussian Z."
```

```
  §1  Two nets, one score (saddle)     ──► Topics 1, 5, 7
  §2  Generator = sampler from Z       ──► Topics 1, 4, 10
  §3  Critic T vs discriminator D      ──► Topics 1, 3, 6
  §4  Last activation / dom(f*)        ──► Topics 2–3
  §5  Sigmoid and log D                ──► Topics 3, 7
  §6  Batch average of an E            ──► Topics 4–5
  §7  Freeze one, step the other       ──► Topics 4–5
  §8  Condition on Y; discard D later  ──► Topics 9–10
```

---

## 1. Two nets share one score: a saddle

<a id="p1-saddle"></a>

### Purpose

Variational divergence minimization (VDM) from last hour is not “train a classifier.” It is: **max** a bound in one set of weights, **min** the same number in another. That shared critical point is a **saddle**.

### Definitions

$$
J(\theta,w)=\mathbb{E}_{p_x}[T_w(X)]-\mathbb{E}_{p_\theta}[f^*(T_w(X))].
$$

- $\theta$ = weights of the **sampler** $G_\theta$.  
- $w$ = weights of the **critic** $T_w$ (the $T$-approximator).  
- **Saddle-point / minmax:** $\theta^*,w^*=\arg\min_\theta\max_w J(\theta,w)$.

Ordinary training **avoids** saddles. This problem **seeks** one.

### Micro numbers (shape, not a real train)

Toy surface $J=w^2-\theta^2$ near $(0,0)$: along $w$ it is a cup you *climb* (max); along $\theta$ it is a hill you *sit in* (min). The interesting point is the middle of a horse saddle, not a bowl.

### Analogy

A horse saddle: sit in the middle; left–right the flaps go **up**; front–back the horse goes **down**. Two axes, opposite jobs, **one** height. The critic tries to **inflate** $J$; the sampler tries to **deflate** it.

### Local picture

```
              max over w (critic T_w)
                    ▲
                    │  flaps UP
   min over θ  ─────┼─────  horse DOWN
   (sampler G_θ)    │
                    ▼

  J(θ, w)  shared score
  typical opt AVOIDS saddles
  this problem SEEKS a saddle
```

### Notice

He will **implement** this saddle today. He will not prove it always converges. Oscillation is on the table.

### Mini-check

1. Which weights does the inner $\max$ train?  
2. Why is “seek a saddle on purpose” a warning?  
3. Is $J$ two different losses, or one number?

---

## 2. The generator is a sampler from noise

<a id="p2-generator"></a>

### Purpose

You never get a formula for $p_x$. You get a **file** of real $x$’s. Fakes are made by pushing **Gaussian noise** $Z\sim\mathcal{N}(0,I)$ through a deterministic net $G_\theta$. Those fakes *are* samples of $p_\theta$.

### Definitions

- **Generator / sampler network** $G_\theta$: $z\mapsto\hat x=G_\theta(z)$. Chance lives in $Z$. Same $z$, same $\hat x$.  
- **True cloud:** dataset $D=\{x_1,\ldots,x_n\}\sim_{\mathrm{iid}}p_x$.  
- **Fake cloud:** $\hat x_j=G_\theta(z_j)$.

### Micro numbers

Three noise vectors of length $2$, through a tiny map:

```
  z1 = (0.1, −0.4)  →  G_θ  →  x̂1 = (0.4, 1.2)
  z2 = (−1.2, 0.3)  →  G_θ  →  x̂2 = (0.5, 1.0)
  z3 = (0.0,  0.8)  →  G_θ  →  x̂3 = (0.3, 1.1)
```

Those three $\hat x$ are **not** a PDF. They are a **file**, just like MNIST is a file of real digits.

The tablet draws $G_\theta$ as a **trapezoid**: typically $\dim(Z)=k$ is **much smaller** than $\dim(x)=d$ ($k\ll d$). That is why a short noise edge becomes a long image edge. Why $k\ll d$ is the **manifold hypothesis** — named this hour, **derived next class**.

### Analogy

A print shop. The noise bag is blank paper with random speckle. The shop ($G_\theta$) is a **fixed recipe**. New speckle → new print. The shop does not roll dice; the speckle bag does.

### Local picture

```
  z ~ N(0,I)  --deterministic-->  G_θ  -->  x̂ ~ p_θ
                  (same z, same x̂)

  HAVE:  bag D          (real)
         bag G_θ(Z)     (fake)
  WANT:  p_θ close to p_x
```

### Notice

At **inference** you only need $G_\theta$. The critic is thrown away. Refreshing [thispersondoesnotexist](https://thispersondoesnotexist.com) is: draw $z$, run $G$.

### Mini-check

1. Where does randomness live — $G_\theta$ or $Z$?  
2. How do you get a *new* fake if $G_\theta$ is frozen?  
3. After training, which net do you keep?

---

## 3. Critic $T$ versus discriminator $D$

<a id="p3-critic"></a>

### Purpose

Last hour’s $T$ is **any** probe into $\mathrm{dom}(f^*)$. It is a **critic**: it scores the bound. For **one** $f$ (GAN’s), the last layer happens to land in $(0,1)$, so people also call it a **discriminator** — a binary classifier. That name is **not** general VDM.

### Definitions

- **Critic / $T$-approximator** $T_w$: the net that stands in for $T$. Always legal language.  
- **Discriminator** $D_w$: the same net **when** its output is in $(0,1)$ and you *interpret* it as $P(\text{real}\mid x)$. GAN’s $f$ licenses this. Change $f$ (e.g. least-squares GAN) and $T$ may be a **regressor**, not a classifier.

### Micro numbers

Two scores on the same photo:

```
  critic T(x)     = −2.1     (some number in dom(f*))
  discriminator D = 0.88     (after a sigmoid: “88% chance real”)
```

Same photo, two *readings*. GAN uses the second because its last activation *is* a sigmoid.

### Analogy

A food critic writes a score on a napkin (any real number in a legal range). A health inspector stamps **pass / fail** (a number in $0$–$1$). GAN’s $f$ turns the critic into an inspector. Least-squares GAN leaves you with the napkin score. Calling every $T$ a “discriminator” is the Hollywood shortcut he does not want as the *definition*.

### Local picture

```
  VDM (any f):     T_w : X → dom(f*)     critic
  GAN (this f):    D_w : X → (0,1)       critic + classifier reading

  change f  →  the classifier story can DIE
```

### Notice

He prefers the **VDM** story (lower bound, then min). The classifier story is a **special case** he will still derive, then distrust.

### Mini-check

1. Is “discriminator” legal for every $f$-GAN?  
2. What does $T_w$ approximate?  
3. Why does $D\in(0,1)$ invite a classifier reading?

---

## 4. Last activation must land in $\mathrm{dom}(f^*)$

<a id="p4-activation"></a>

### Purpose

Choose $f$ → $f^*$ is determined → **domain of $f^*$** is determined. The last layer of $T$ is a **Lego brick** that maps a free real $V(x)$ into that domain.

### Definitions

$$
T_w(x)=\sigma_f\bigl(V_w(x)\bigr),\qquad V_w:\mathcal{X}\to\mathbb{R},\quad \sigma_f:\mathbb{R}\to\mathrm{dom}(f^*).
$$

$V_w$ is a net with a **linear last layer** (output a real). $\sigma_f$ is the plug-in activation from the $f$-GAN table.

### Micro numbers

GAN’s $f^*$ lives on the **negative reals** $\mathbb{R}_-$. A legal last map is $\sigma_f(v)=-\log(1+e^{-v})$, which is always $\le 0$. A ReLU last layer ($\ge 0$) would be the **wrong hinge**.

### Analogy

A door and a hinge. $f$ picks the hinge shape. You may build any house ($V_w$), but the last door must fit **that** hinge. Swapping the Lego brick is how you change which $f$ you are using, without rebuilding the whole net.

### Local picture

```
  x  -->  V_w  -->  real v  -->  σ_f  -->  T(x) in dom(f*)
          (linear head)

  choose f  ⇒  f*  ⇒  dom(f*)  ⇒  which σ_f
```

### Notice

The original $f$-GAN paper lists $\sigma_f$ for each $f$. Today he only needs **GAN’s** brick.

### Mini-check

1. What does choosing $f$ fix besides $f$ itself?  
2. Why a linear head on $V_w$?  
3. What goes wrong if $T$ outputs outside $\mathrm{dom}(f^*)$?

---

## 5. Sigmoid: a real number crushed into $(0,1)$

<a id="p5-sigmoid"></a>

### Purpose

GAN’s algebra turns $T$ into

$$
D_w(x)=\frac{1}{1+e^{-V_w(x)}},
$$

the usual **sigmoid**. Output is always between $0$ and $1$. Then the score looks like **binary cross-entropy**: $\log D$ on reals, $\log(1-D)$ on fakes.

### Definitions

Sigmoid $\sigma(v)=1/(1+e^{-v})$: as $v\to+\infty$, $\sigma\to 1$; as $v\to-\infty$, $\sigma\to 0$; at $v=0$, $\sigma=1/2$.

**Log $D$:** if $D=0.9$, $\log 0.9\approx -0.105$ (small penalty). If $D=0.1$ on a *real* photo, $\log 0.1\approx -2.3$ (large penalty). The critic is rewarded for $D\approx 1$ on reals and $D\approx 0$ on fakes.

### Micro numbers

```
  V =  2.0   →  D ≈ 0.88
  V =  0.0   →  D = 0.50
  V = −2.0   →  D ≈ 0.12
```

### Analogy

A volume knob from $-\infty$ to $+\infty$, then a squash into a **probability-like** $0$–$1$ reading. “$88\%$ real” is that reading — not a court verdict. The generator’s job, in this $f$, is to push the reading toward $1/2$ (confused inspector).

### Local picture

```
  real x   --D-->  want D ≈ 1   -->  log D   close to 0
  fake x̂  --D-->  want D ≈ 0   -->  log(1−D) close to 0

  J_GAN = E_real[log D] + E_fake[log(1−D)]
          critic MAXes this; generator MINs it
```

### Notice

People say “GAN minimizes JSD.” He will say: the $f$ is **similar** to Jensen–Shannon, **missing a constant**. Not exactly JSD.

After the algebra, GAN’s score is a **plus** of two logs, not last hour’s **minus** of $T$ and $f^*(T)$. Same saddle, different writing of $J$.

### Mini-check

1. What is $D$ if $V=0$?  
2. Why does $D\in(0,1)$ license a classifier story?  
3. Is GAN *exactly* JSD?

---

## 6. A batch average stands in for an expectation

<a id="p6-batch"></a>

### Purpose

$J$ is two expectations. You cannot walk the whole city. You poll a **batch**.

### Definitions

- **Batch** $B_1$ from the dataset: $x_1,\ldots,x_{B_1}$ (with replacement / shuffle — “sample from the $n$ points”).  
- **Batch** $B_2$ of fakes: draw $z_j\sim\mathcal{N}(0,I)$, $\hat x_j=G_\theta(z_j)$.  

$$
\mathbb{E}_{p_x}[\log D(X)]\;\approx\;\frac1{B_1}\sum_{i=1}^{B_1}\log D(x_i).
$$

The left side is a **fixed** number (for frozen nets). The right side is a **random** estimate.

### Micro numbers

$B_1=2$ reals with $D=0.8$ and $0.6$: average $\log D \approx (\log 0.8+\log 0.6)/2$. $B_2=2$ fakes with $D=0.3$ and $0.2$: average $\log(1-D)$. Add. That **sum** is one step of the inner $\max_w$.

### Analogy

Poll $32$ addresses instead of the whole city. The poll is useful when the addresses came from the **same city** the $\mathbb{E}$ used. Polling fakes to estimate the **real** $\mathbb{E}$ is the wrong city.

### Local picture

```
  E_{p_x}[log D]     ≈  (1/B1) Σ log D(x_i)      x_i from MNIST file
  E_{p_θ}[log(1−D)]  ≈  (1/B2) Σ log(1−D(x̂_j))  x̂_j = G_θ(z_j)
```

### Notice

Same $D$ net, two inputs: real $x$ and fake $\hat x$. That is why a $D$-step needs **two** forward passes through $D$.

### Mini-check

1. How do you get $B_2$ fakes?  
2. Is a batch average the same object as $\mathbb{E}$?  
3. Which cloud estimates $\mathbb{E}_{p_x}[\log D]$?

---

## 7. Freeze one net, step the other

<a id="p7-freeze"></a>

### Purpose

A saddle is not “add both losses and step everything.” Inner: **max** $w$ with $\theta$ frozen. Outer: **min** $\theta$ with $w$ frozen. In code: turn **off** gradients on the frozen net.

### Definitions

- **$D$-step (inner max):** $\theta$ constant. Forward $G$ once (make fakes). Forward $D$ twice (reals + fakes). Backward **only** through $D$.  
- **$G$-step (outer min):** $w$ constant. **No** real batch needed — the $\mathbb{E}_{p_x}[\log D]$ term does not depend on $\theta$. Forward $G$, forward $D$, backward from $D$’s output **through** $G$. $D$’s weights do not move.

Naive practice: one $D$-step, one $G$-step, alternate. Real practice is often **not** $1{:}1$ (he flags this; tutorials show it).

### Micro numbers (pass count)

```
  D-step:  1 fwd G  +  2 fwd D  +  1 bwd D
  G-step:  1 fwd G  +  1 fwd D  +  1 bwd G (via D)
```

### Analogy

Two rooms, one shared scoreboard. Inspector trains with the forger **frozen**. Then the forger trains with the inspector **frozen** (you still *walk through* the inspector’s room to read the score — you just do not move the inspector’s furniture).

### Local picture

```
  D-step:  freeze G_θ
           real x  ──► D
           z → G → x̂ ──► D
           add logs;  ∇_w only

  G-step:  freeze D_w
           z → G → x̂ ──► D  (D is a path, not a student)
           ∇_θ through G; first term dropped
```

### Notice

You still **evaluate** $D(G_\theta(z))$ on a $G$-step. Frozen ≠ deleted.

### Mini-check

1. Why drop the real-data term when training $G$?  
2. How many $D$ forwards in a $D$-step?  
3. Frozen $D$ still sits in the $G$ backward path — true or false?

---

## 8. Condition on $Y$; throw $D$ away at inference

<a id="p8-condition"></a>

### Purpose

Everything so far samples a **marginal** $p_x$ (any MNIST digit). A **conditional** sampler draws $x\mid y$ (the digit **3**, or “two people crossing a road”). The prompt of a chatbot is a draw of that $Y$, not of Gaussian $Z$.

### Definitions

- **Conditioner** $Y$: class label, text, any semantic you have as **pairs** $(x_i,y_i)$.  
- **One-hot:** class $3$ of $10$ → vector $(0,0,0,1,0,\ldots)$.  
- **Embedding:** a sentence mapped to a vector of reals (because nets eat numbers).  
- **Concat trick:** feed $(z,y)$ to $G$ and $(x,y)$ to $D$. $D$ then scores **co-occurrence**.  
- **Inference:** discard $D$. Sample $z$ (and $y$ if you want a class) through trained $G$ only.

### Micro numbers

MNIST, ten classes, one-hot length $10$:

```
  class   one-hot y
  0       (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  3       (0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
  9       (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
```

Concatenate class $3$ to $z\in\mathbb{R}^{100}$ → $G$ input length $110$. At test time you pick the slot you want, draw a **new** $z$, print a three. A caption like “two people crossing the road” is the same idea after the sentence has been turned into a vector of reals (an **embedding**).

### Analogy

A teacher (the discriminator) walks with the student (the generator) during training, then **leaves**. A good teacher becomes redundant. At showtime you only bring the student — plus, if you asked for “draw a 3,” you also bring that request $Y$.

### Local picture

```
  TRAIN (conditional)
    z  ⊕  y  -->  G_θ  -->  x̂ | y
    x  ⊕  y  -->  D_w  -->  score of the pair

  INFERENCE
    z_test  ⊕  y_wanted  -->  G_θ*  -->  new x
    D is gone
```

### Notice

You cannot condition on text you never paired with images. **COCO** is the named dataset of image–caption pairs. ChatGPT is **not** this GAN recipe — it is autoregressive, later in the course.

### Mini-check

1. What extra data does a conditional GAN need?  
2. Who is concatenated into $G$ and into $D$?  
3. After training, do you still run $D$?

---

**Second teachers (names only here).** Original GAN paper: Goodfellow et al. 2014. VDM / $f$-GAN: Nowozin, Cseke, Tomioka 2016. DCGAN: Radford et al. Conditional GAN: Mirza & Osindero. StyleGAN: Karras / NVIDIA. University hours: Stanford CS236, Berkeley CS294-158, MIT 6.S191. Pointers (video + notes) live at the end of [NOTES.md](./NOTES.md#external-references).

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
