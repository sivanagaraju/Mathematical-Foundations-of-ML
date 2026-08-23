# Lec 20 — Latent Variable Models and Introduction to VAE

> **Video:** [Lec 20 Latent Variable Models and Introduction to Variational Autoencoder (VAE)](https://www.youtube.com/watch?v=4djE9goJtKs) · **~55 min**  
> **Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md) · **Quiz:** [quiz.html](./quiz.html)

**Do PREREQUISITES before this article.**  
**Previous:** [Lec 19 inversion / FID](../31-Lec19-Inversion-GANs-FID/NOTES.md)  
**Course:** Mathematical Foundations of **Generative AI** · Prof. Prathosh A. P. · NPTEL IISc  
**Boards:** captions only (video 403) — ASCII reconstructions.

This is a **chalk hour**. He **does not** implement a VAE. Reparameterization, backprop, and “one instantiation on a computer” are **next class**. Commented blocks below are **his algebra / algorithms**, not invented PyTorch.

| When the lecture hits… | Warm-up |
|------------------------|---------|
| Observed vs hidden | [p1-observed-latent](./PREREQUISITES.md#p1-observed-latent) |
| $p(x)$ as a marginal | [p2-joint-marginal](./PREREQUISITES.md#p2-joint-marginal) |
| Cluster vs knobs | [p3-discrete-continuous](./PREREQUISITES.md#p3-discrete-continuous) |
| KL = MLE | [p4-kl-mle](./PREREQUISITES.md#p4-kl-mle) |
| $q(z\mid x)$ | [p5-posterior](./PREREQUISITES.md#p5-posterior) |
| Jensen / ELBO | [p6-jensen](./PREREQUISITES.md#p6-jensen) |
| GMM die | [p7-gmm-die](./PREREQUISITES.md#p7-gmm-die) |
| Net = samples or parameters | [p8-nn-two-ways](./PREREQUISITES.md#p8-nn-two-ways) |

---

## Table of Contents

1. [Topic 1 — LVM family and definition](#topic-1-lvm-family-and-definition-0003–0255) (00:03–02:55)
2. [Topic 2 — Discrete $z$ clusters; continuous $z$ features](#topic-2-discrete-z-clusters-continuous-z-features-0255–0812) (02:55–08:12)
3. [Topic 3 — Embedding; both arrows](#topic-3-embedding-both-arrows-0812–1009) (08:12–10:09)
4. [Topic 4 — Min KL is max MLE](#topic-4-min-kl-is-max-mle-1009–1329) (10:09–13:29)
5. [Topic 5 — Incomplete likelihood; $q(z\mid x)$](#topic-5-incomplete-likelihood-qzmid-x-1329–1646) (13:29–16:46)
6. [Topic 6 — Log of an expectation; Jensen](#topic-6-log-of-an-expectation-jensen-1646–2011) (16:46–20:11)
7. [Topic 7 — ELBO and the word variational](#topic-7-elbo-and-the-word-variational-2011–2650) (20:11–26:50)
8. [Topic 8 — Boxed ELBO; GMM](#topic-8-boxed-elbo-gmm-2650–3410) (26:50–34:10)
9. [Topic 9 — EM; when VAE starts](#topic-9-em-when-vae-starts-3410–4353) (34:10–43:53)
10. [Topic 10 — Three jobs; two NN styles; stop](#topic-10-three-jobs-two-nn-styles-stop-4353–5524) (43:53–55:24)
11. [External references](#external-references)
12. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

This hour defines a latent variable model: $p_\theta(x)$ is the marginal of a joint with hidden $z$. Jensen on $q(z\mid x)$ yields an ELBO over $\theta$ and $q$. EM if the posterior is a formula; VAE if not. Implementation is next class.

**Worldview arc:** from decoder-only adversarial sampling **to** a probabilistic LVM (encoder+decoder, ELBO). Deterministic one-$z$ stamp (k-means) vs full $q(z\mid x)$.

### The approach

```
  HAVE     n IID x's from unknown p
  WANT     estimate p and SAMPLE new x
           AND infer a hidden z for each x

  DEFINE   p_θ(x) = ∫ p_θ(x,z) dz     (sum if z discrete)
           z jointly learned with θ

  LEARN    min KL(p_x || p_θ)  ≡  max E[log p_θ(x)]
           log ∫ p(x,z) is incomplete (no z samples)

  BOUND    introduce q(z|x)
           log E_q[p(x,z)/q]  ≥  E_q[log p(x,z)/q]   = ELBO
           (θ*, q*) = argmax ELBO     ← boxed problem
                     for GMM, VAE, DDPM, flows, …

  FORK     if p_θ(z|x) tractable  →  EM  (q* = p_θ(z|x) by Bayes)
           if not                 →  VAE / Auto-Encoding Variational Bayes

  USE      sample:  z ~ p(z), then x ~ p(x|z)
           infer:   q(z|x)  = clustering (discrete) or embedding (continuous)

  STOP     reparameterization + backprop implementation  NEXT lecture
```

### System context

```
  ╔══════════════════════════════════════════╗
  ║ Lec 04–19: f-div, VDM, GAN, WGAN, invert ║
  ║ Next: reparam trick + VAE on a computer  ║
  ║ Later: DDPM, score, flows (still LVMs)   ║
  ╚════════════════════╤═════════════════════╝
                       │ this lecture (~55 min)
                       ▼
        ┌──────────────────────────────────┐
        │ LVM definition + ELBO + EM/VAE   │
        └──────────────────────────────────┘
```

### Main blueprint

```
  ╔══ JOB ══╗
  ║ estimate║
  ║ p and   ║
  ║ sample  ║
  ╚═══╤═════╝
      │ n IID x
      ▼
  ┌─ LVM ─────────────────────────────┐
  │ p_θ(x) = marginal of p_θ(x,z)     │
  │ z hidden; jointly learned with θ  │
  │ discrete z → clustering           │
  │ continuous z → embedding          │
  │ encoder AND decoder by construction│
  └──────────────┬────────────────────┘
                 │ min KL ≡ max MLE
                 │ log ∫ p(x,z)  incomplete
                 ▼
  introduce q(z|x)  (latent / variational posterior)
                 │ Jensen (log concave)
                 ▼
  ┌─ ELBO J_θ(q) ─────────────────────┐
  │ E_q log p(x|z) − KL(q(z|x) || p(z))│
  │ maximize over θ AND q             │
  └──────────────┬────────────────────┘
         ┌───────┴────────┐
         ▼                ▼
   p_θ(z|x) known     p_θ(z|x) unknown
   EM (GMM, PGMs)     VAE (AEVB paper)
   q* = model posterior
         │
  ┌ · · ·┴ · · · · · · · · · · ┐
  │ STOP: reparam + code next  │
  └ · · · · · · · · · · · · · ┘
```

### Scenario walkthrough

**Story:** 128 unlabeled MNIST digits. You want new digits **and** a code for each existing digit.

1. Write $p_\theta(x)=\int p(x,z)\,dz$. $z$ might be a cluster id or thickness knobs.
2. MLE wants $\log\int p(x,z)$. You have no $z$ samples — incomplete.
3. Invent $q(z\mid x)$, apply Jensen, maximize ELBO in $\theta$ and $q$.
4. If $z$ is a die + Gaussian (GMM), EM: Bayes for $q$, closed form for $\mu,\alpha,\Sigma$.
5. If $q$ has no formula, you are in the VAE paper. Nets will output **parameters** of $q$ and $p(x\mid z)$. Sampling trick is **next** hour.

### Failure / contrast

```
  WRONG  treat last class's G as already an encoder
  WRONG  sample-average a log-of-expectation
  WRONG  “EM always increases likelihood”   (never decreases; may saturate)
  WRONG  implement reparameterization today  (he deferred it)
```

### STOP / out of scope

- Jensen proof and two EM proofs = homework / previous course.
- DDPM, score, flows named as later LVMs, not derived.
- Reparameterization + computer instantiation = **next lecture**.

### Load-bearing claims (closed-book)

- $p_\theta(x)=\int p_\theta(x,z)\,dz$ (sum if discrete); $z$ jointly learned with $\theta$.
- Discrete $z$ ⇒ clustering; continuous $z\mid x$ ⇒ embedding.
- LVM has **encoder and decoder by construction**.
- $\min\mathrm{KL}\equiv\max$ MLE; drop $H(p_x)$.
- $q(z\mid x)$ is the latent (variational) posterior; GMM soft vs k-means Dirac.
- ELBO $= \mathbb{E}_q[\log p(x,z)/q]$; also $\mathbb{E}_q\log p(x\mid z)-\mathrm{KL}(q\|p(z))$.
- EM: $q^\star=p_\theta(z\mid x)$ when that posterior is tractable.
- VAE paper: **Auto-Encoding Variational Bayes** — intractable posterior.
- Two NN styles: samples vs parameters.

**Speaker / course:** Prof. Prathosh · NPTEL IISc · Lec 20.

---

## Topic 1: LVM family and definition (00:03–02:55)

### Where this sits on the master map

This is the **FAMILY + DEFINITION** box. Adversarial / VDM is done. The new object is a model whose $p_\theta(x)$ is a [marginal of a joint with hidden $z$](./PREREQUISITES.md#p2-joint-marginal).

### Board / screenshot

No content frame (video 403). Reconstruction ~00:03–02:55.

```
  last block:  VDM / GAN / WGAN / invert
  this block:  LATENT VARIABLE MODELS

  family:  autoencoders, VAEs,
           mixture models / mixture densities,
           later: DDPM, score, flows

  DEFINITION
  p_θ(x)  =  marginal of joint p_θ(x, z)
  z hidden / unobserved / latent
  discrete z → SUM     continuous z → INTEGRAL
  z jointly learned with θ
```

Caption: $p_\theta$ is how you *look at* the parametric model — as a marginal over a hidden $z$.

### What he is establishing

Welcome, then a clean cut: last class finished **variational divergence minimization / adversarial learning**. From now, a **broad paradigm**: **latent variable models**. Autoencoders, **variational autoencoders** and variants, even classical **mixture models / mixture densities**. Immediately in this stretch of the course: VAEs. Then **DDPMs** (denoising diffusion probabilistic models), **score-based** models, **flow-based** models — all LVMs.

The GenAI job does not change: $n$ IID draws from unknown $p$; **estimate** and **sample**. $p_\theta$ is still a parametric model.

What changes is the **mathematical definition of $p_\theta$**. An LVM says: introduce an auxiliary **hidden / latent / unobserved** $z$, write a joint of observed $x$ and $z$, and take **$p_\theta(x)$ to be the marginal of that joint**. If $z$ is discrete the marginal is a **sum**; if continuous, an **integral**. You are **not given** $z$. In most cases $z$ is **jointly learned with $\theta$**.

The trap is hearing “latent” and thinking last class’s GAN noise you *sample first*. Here $z$ is a hidden cause you did **not** write in the dataset.

You can now write $p_\theta(x)=\int p_\theta(x,z)\,dz$ and name the family. What is still open: what $z$ *means* for a digit.

### Analogy for this topic only

128 digit photos on the table. You invent a hidden recipe card $z$ for each photo, then define the chance of a photo by adding up (or integrating) all recipe cards.

Someone asks: **did the JPEG file contain $z$?** No. That is the whole point of “latent.”

The wrong move is to wait for a $z$ column in the CSV.

In lecture words: recipe card = latent $z$; photo = $x$; $p_\theta(x)$ = marginal of the joint.

### Local picture

```
  x observed          z hidden
    \                /
     \   joint p(x,z)
      v
   erase z (sum or ∫)
      v
   p_θ(x)   ← this is the LVM

  Notice: z is learned with θ, not downloaded.
```

### Bridge

If $z$ is a stamp with $M$ faces, grouping falls out. If $z$ is a slider, you get knobs. Those two readings are the next box.

---

## Topic 2: Discrete $z$ clusters; continuous $z$ features (02:55–08:12)

### Where this sits on the master map

This is **WHAT $Z$ MEANS**. One [hidden $z_i$ per $x_i$](./PREREQUISITES.md#p1-observed-latent). Discrete ⇒ [clustering](./PREREQUISITES.md#p3-discrete-continuous); continuous ⇒ features, maybe uninterpretable.

### Board / screenshot

No content frame. Reconstruction ~02:55–08:12.

```
  each x_i  has  a z_i

  z discrete, 1..M
     after inference: x's sit in M piles
     = CLUSTERING
     GMM, k-means, hierarchical, …

  z continuous, z in R^K
     z_i = K-vector for x_i
     MNIST example: thickness, orientation, …
     interpretable dims = OPEN PROBLEM
```

Caption: $x$ may still be $\mathbb{R}^D$ (an image). Only the type of $z$ changed.

### What he is establishing

Conceptually: for each realization $x_i$ there is a hidden realization $z_i$.

**Discrete** $z$: the variable takes **one of $M$ values**. $x$ can still be $\mathbb{R}^D$ — an image. After you attach a $z_i$ to every $x_i$, every point sits in one of $M$ groups. That exercise **is clustering**. All clustering algorithms he names — **Gaussian mixture models, k-means, hierarchical clustering** — are LVMs with discrete $z$.

**Continuous** $z$: $z_i$ can be a **$K$-dimensional** vector. It **may** correspond to features; it **may have no physical meaning**. MNIST: four numbers might be thickness, orientation, and two unnamed factors. **Coming up with interpretable latent dimensions is an open problem.** Discrete $z$ is always readable as clustering; continuous $z$ is not always readable as named sliders.

The trap is expecting every trained $z$-dim to be a English noun. He said it might not.

You can now split discrete vs continuous. What is still open: the word **embedding**, and why this family already has both arrows.

### Analogy for this topic only

Stamp each digit red/blue/green ($M=3$). You clustered.

Or give each digit four sliders (fat/thin, tilt, …). You embedded.

Someone asks: **must slider 2 mean “tilt”?** No. Interpretable dims are extra work.

In lecture words: stamp = discrete $z$ = clustering; sliders = continuous $z$ = possible features.

### Local picture

```
  128 digits
     discrete z →  M buckets
     continuous z → 128 arrows in R^K

  Notice: k-means is an LVM with a spike posterior (next topics).
```

### Bridge

He will call $z\mid x$ an **embedding**, and contrast with last class’s decoder-only GAN.

---

## Topic 3: Embedding; both arrows (08:12–10:09)

### Where this sits on the master map

This is **BOTH WAYS**. Continuous $z\mid x$ = [embedding](./PREREQUISITES.md#p3-discrete-continuous). Unlike last class, encoder **and** decoder are in the definition.

### Board / screenshot

No content frame. Reconstruction ~08:12–10:09.

```
  z | x   =  embedding / feature
  extracting embeddings = LVM + estimate/sample p(z|x)

  last class FID: Inception vectors = embeddings
                 (that net as an LVM)

  LVM by construction:
     encoder  x --> z     embedding
     decoder  z --> x     sampling
```

Caption: both purposes — invert and sample — are native, not bolted on.

### What he is establishing

If $z$ is continuous, $z_i$ given $x_i$ is a **feature vector**. That is **embedding**. Whenever you extract embeddings you are (in this language) building an LVM and estimating $z\mid x$ or sampling $p(z\mid x)$. Last class’s **FID** vectors from a pretrained net can be seen as embeddings; that net as an LVM.

Then the contrast with inversion week: LVMs **give you both ways**. Encoder: $x$ onto $z$ (embedding). Decoder: $z$ onto data (sampling). Both happen **by construction**.

The trap is thinking you still need BiGAN’s third net to ingest $x$. That was the GAN gap. Here the encoder is part of the object.

You can now say “LVM = encoder + decoder.” What is still open: the **learning** objective (still KL / MLE).

### Analogy for this topic only

Last week the press could print but not read a customer JPEG. This week the shop is born with a locksmith **and** a press.

Someone asks: **do I still hire BiGAN’s extra encoder?** Not as a patch — the LVM already owns $q(z\mid x)$.

In lecture words: $z\mid x$ = embedding; both arrows native.

### Local picture

```
  GAN (last class):     z --> G --> x̂     missing x --> z
  LVM (this class):     x --> encoder --> z
                        z --> decoder --> x̂

  Notice: FID's Inception tap is an embedding, hence an LVM reading.
```

### Bridge

How do you *fit* $\theta$? Typical answer: KL, which is MLE after you drop entropy.

---

## Topic 4: Min KL is max MLE (10:09–13:29)

### Where this sits on the master map

This is **OBJECTIVE**. Typical LVM divergence is [KL](./PREREQUISITES.md#p4-kl-mle). Entropy of data is not a knob.

### Board / screenshot

No content frame. Reconstruction ~10:09–13:29.

```
  learn LVMs by MLE
  typical d = KL(p_x || p_θ)

  KL = ∫ p_x log p_x  −  ∫ p_x log p_θ
         H(data)            − E[log p_θ]
         independent of θ

  drop H  →  maximize E_{p_x}[log p_θ(x)]
  min KL  ≡  max likelihood
```

Caption: “you cannot change the universe”; $\theta$ is self-tweaking.

### What he is establishing

General principle: **maximum likelihood**. Typical metric: **KL** (other divergences exist). Expand KL into entropy of $p_x$ minus expected log $p_\theta$. The entropy term **does not depend on $\theta$**. Data is given; you cannot change its entropy — “you cannot change the universe… self-tweaking is $\theta$.” Drop $H$. With the minus sign, **minimizing KL is maximizing** $\mathbb{E}_{p_x}[\log p_\theta(x)]$. **MLE is min-KL.** He will not keep repeating that.

Plug the LVM in and the thing you wanted to max is $\log\int p_\theta(x,z)\,dz$. That integral is the next obstacle.

Work 128 MNIST files. Their entropy is a property of the pile. Switching $p_\theta$ from one Gaussian to a mixture does **not** tidy the album; it only changes how well the model’s log-prob scores those files.

The trap is optimizing $H(p_x)$ or treating KL and MLE as different jobs today.

You can now write the MLE line. What is still open: the integral over unseen $z$.

### Analogy for this topic only

The album’s messiness is weather. Your model is an umbrella. Moving the umbrella does not edit the weather.

Someone asks: **should the loss include entropy of MNIST?** No. It cancels.

In lecture words: $H(p_x)$ dropped; leftover = expected log-likelihood.

### Local picture

```
  KL(p_x || p_θ) = H(p_x) − E[log p_θ]
                      ignore     maximize

  Notice: same slogan as earlier MFML weeks; now p_θ has a hidden z inside.
```

### Bridge

$L_\theta=\log\int p(x,z)\,dz$ has no $z$ samples. That is incompleteness, and it forces $q(z\mid x)$.

---

## Topic 5: Incomplete likelihood; $q(z\mid x)$ (13:29–16:46)

### Where this sits on the master map

This is **OBSTACLE**. The integral is not a usable number. The missing object is the [latent posterior](./PREREQUISITES.md#p5-posterior).

### Board / screenshot

No content frame. Reconstruction ~13:29–16:46.

```
  L_θ = log p_θ(x) = log ∫ p_θ(x,z) dz

  we do NOT have z samples
  we do NOT know the joint as a table

  need:  a law on z GIVEN x
         q(z|x)  =  LATENT POSTERIOR

  GMM: q is a distribution over M clusters  (soft)
  k-means: q is a Dirac spike               (one z per x)
```

Caption: jointly estimating $z$ **means** estimating $q(z\mid x)$, not one spreadsheet cell.

### What he is establishing

Write $\mathcal{L}_\theta=\log p_\theta(x)=\log\int p_\theta(x,z)\,dz$. This marginalization is **not well-defined in practice**: you do not know the joint, and you have **no realizations of $z$** — only samples from $p_x$. Yet the LVM story was: jointly estimate latents with the model.

Translate “find $z$ for this $x$” into probability: a **distribution on $z$ conditioned on $x$**. That is the **latent posterior**. There is **not** one single $z$ per $x$ in general. **GMM** is **soft clustering**: given $x$, a whole distribution on the discrete labels. If $q(z\mid x)$ is **degenerate / Dirac**, one $z$ per $x$ — **k-means**.

When he says we estimate the latent **jointly** with $\theta$, he means we estimate **$q(z\mid x)$**. A primary LVM goal is that posterior. It has to enter the optimization.

A digit that looks halfway 3 and 8 is the picture: GMM returns two weights (0.6 / 0.4). K-means slaps a single stamp. Same $x$, two different $q$s.

The trap is solving $\arg\max_z p(x\mid z)$ and calling that GMM.

You can now name $q(z\mid x)$ and the soft/hard fork. What is still open: algebra that puts $q$ inside $\log\int$.

### Analogy for this topic only

A digit looks sixty percent like a three and forty percent like an eight. Soft $q$ keeps both numbers. A spike $q$ stamps “3” and lies.

Someone asks: **does “estimate $z$” return one integer?** Only if $q$ collapsed.

In lecture words: $q(z\mid x)$ = latent posterior; Dirac = k-means; soft = GMM.

### Local picture

```
  have:   x_1..x_n
  want:   q(z|x) for each x
          not a single z* unless q is a spike

  Notice: this q is still arbitrary at the next step — then we optimize it.
```

### Bridge

Multiply and divide the integral by $q$, and you will meet a **log of an expectation**.

---

## Topic 6: Log of an expectation; Jensen (16:46–20:11)

### Where this sits on the master map

This is **ALGEBRA START**. Rewrite the likelihood using [arbitrary $q$](./PREREQUISITES.md#p5-posterior); the course’s sample-average weapon fails on $\log\mathbb{E}$; [Jensen](./PREREQUISITES.md#p6-jensen) saves it.

### Board / screenshot

No content frame. Reconstruction ~16:46–20:11.

```
  start with arbitrary density q(z|x) ≥ 0
  multiply and divide the integral by q

  L = log E_{q(z|x)} [ p_θ(x,z) / q(z|x) ]

  problem: LOG of an expectation
           (sample average wants E, not log E)

  log is CONCAVE  →  Jensen
  log E[Y]  ≥  E[log Y]     (lower bound)
  homework: prove Jensen
```

Caption: multiplying by $q/q$ does not change the integral; it changes what you can bound.

### What he is establishing

Pick an arbitrary density $q(z\mid x)$ (he first says “random,” then takes it back — randomness has a meaning). Multiply and divide the integral by $q$. Group terms: the inner integral is an **expectation under $q$**. The log-likelihood becomes

$$
\log\mathbb{E}_{q(z\mid x)}\Bigl[\frac{p_\theta(x,z)}{q(z\mid x)}\Bigr].
$$

The course’s go-to weapon is: see an expectation, replace it by a sample average. Here you see a **log of an expectation**. That is hard. Push $\mathbb{E}$ outside $\log$.

**Jensen:** for a **concave** $f$, $f(\mathbb{E} Y)\ge \mathbb{E} f(Y)$. Log is concave, so $\log\mathbb{E}\ge\mathbb{E}\log$ — a **lower bound**. Convex $f$ flips it. **Homework: prove Jensen.**

In the digit workshop, two equally likely hidden stories score 2 and 8 on a toy $Y$. Log of the average is log five; average of the logs is log four. The first number is the likelihood rewrite; the second is what you can hope to sample-average.

The trap is taking many draws of $Y$, averaging, then logging, and claiming that estimates $\mathbb{E}\log Y$.

You can now write $\log\mathbb{E}$ vs $\mathbb{E}\log$. What is still open: naming the bound **ELBO** and why $q$ is “variational.”

### Analogy for this topic only

Balances two and eight. Log of the average (log five) is bigger than average of the logs (log four).

Someone asks: **can I swap those two?** Not if you want a valid lower bound.

In lecture words: $\log\mathbb{E}[p/q]$ is the likelihood rewrite; $\mathbb{E}\log(p/q)$ is the bound we will keep.

### Local picture

```
  ∫ p(x,z) dz
    = ∫ q(z|x) · [p(x,z)/q(z|x)] dz
    = E_q[ p/q ]

  log E_q[p/q]   ≥   E_q[ log(p/q) ]
       ↑ likelihood      ↑ next: ELBO

  Notice: q was arbitrary; we will optimize it.
```

### Bridge

The lower bound on **evidence** gets a name, a pun, and a calculus.

---

## Topic 7: ELBO and the word variational (20:11–26:50)

### Where this sits on the master map

This is **ELBO**. The bound $J_\theta(q)$ is the new objective. “Variational” means [optimize a function inside an integral](./PREREQUISITES.md#p6-jensen), not “VAE layer names.”

### Board / screenshot

No content frame. Reconstruction ~20:11–26:50.

```
  extra motive: exponential family → log cancels exp
  GMM teachers: "log of sums vs sum of logs"
  he wants the ABSTRACT Jensen story first

  J_θ(q) = lower bound on log-likelihood
  log-likelihood  =  EVIDENCE
  bound           =  Evidence Lower BOund  =  ELBO
  paper pun: "ELBO surgery"

  q(z|x) = variational latent posterior
  why variational: q sits inside an ∫
                   calculus of variations / Euler–Lagrange
  engineers: replace q by a net, optimize WEIGHTS (variables)
             that is NOT functional optimization
```

Caption: VAE is named because we estimate a **function** $q$; then we cheat with neural nets.

### What he is establishing

Another reason to push log inside: many model densities are **exponential family**; log cancels the exp; algebra is nicer (GMM). Mixture-EM lectures often start from “we have log of sums, we need sum of logs.” He wants the **abstract** Jensen move first, special cases later — “infinite special cases; find the integral idea.”

Name the bound $J_\theta(q)$: function of $\theta$ **and** of unknown $q$. It lower-bounds log-likelihood. Literature calls log-likelihood **evidence**. So this is an **evidence lower bound**, abbreviated **ELBO** (ASR “elbow”). Whole algorithms are “ELBO optimization.” There is a paper literally titled **ELBO surgery** — another way to carve the bound.

$q(z\mid x)$ is the latent posterior, also **variational** latent posterior: last class he mentioned **variational calculus** — optimize a **function** sitting inside an integral (Euler–Lagrange), not a finite list of numbers. Variational machine learning. **Variational autoencoder** is named from estimating that **function**. Engineers then approximate $q$ by a neural net and descend on **parameters**. That is still optimization over variables, **not** true functional optimization. “That’s what engineers do.”

The trap is thinking “variational” means “has an encoder layer.” It means **$q$ is a function you optimize**.

You can now say ELBO and why $q$ is variational. What is still open: the boxed $\arg\max$ and the GMM worked case.

### Analogy for this topic only

Evidence is the height of the mountain. ELBO is a platform you build underneath. Raise the platform (in $\theta$ and in $q$) and you never overshoot the peak.

Someone asks: **is the platform equal to the peak?** Only for special $q$ (next EM tightness).

In lecture words: $J_\theta(q)$ = ELBO; $q$ = variational posterior.

### Local picture

```
  log p_θ(x)   =  evidence
       ≥
  J_θ(q)       =  ELBO

  maximize J over θ  AND  over the function q

  Notice: NN weights later fake the “over functions” part.
```

### Bridge

Box the problem. Then see it on a Gaussian mixture so you feel why neural models had to happen.

---

## Topic 8: Boxed ELBO; GMM (26:50–34:10)

### Where this sits on the master map

This is **PROBLEM + CLASSICAL CASE**. Same boxed ELBO for GMM, VAE, DDPM, flows. GMM is the [die + Gaussians](./PREREQUISITES.md#p7-gmm-die) LVM.

### Board / screenshot

No content frame. Reconstruction ~26:50–34:10.

```
  BOXED  (θ*, q*) = arg max  E_q [ log p_θ(x,z) / q(z|x) ]
         same box for GMM, VAE, DDPM, flows

  GMM: z discrete 1..M
       p(z=j) = α_j              (mass; die)
       p(x|z=j) = N(μ_j, Σ_j)
       roll die, then that Gaussian
       p_θ(x) = Σ_j α_j N(x; μ_j, Σ_j)
       convex combination BECAUSE z is discrete
```

Caption: continuous $z$ is an integral, not a finite mix.

### What he is establishing

Original MLE: max log-likelihood in $\theta$. After the bound: find $\theta$ **and** $q$ maximizing ELBO. **Boxed** and reused: GMM, any mixture, VAE, DDPM, flow-based.

Classical example, assuming you know it, to see **why people moved to neural models**. Mixture densities are LVMs with discrete $z\in\{1,\ldots,M\}$. Factor $p(x,z)=p(z)p(x\mid z)$. Discrete $\Rightarrow$ $p(z)$ is a **mass** $\alpha_j$. In GMM, $p(x\mid z{=}j)$ is Gaussian $(\mu_j,\Sigma_j)$. Picture: **$M$-face die**; roll it; pick that Gaussian. $M$ mean vectors in $\mathbb{R}^D$, $M$ covariance $D\times D$ matrices, $\alpha\in[0,1]$ summing to 1.

Why “mixture”: the **marginal** is a **linear / convex combination** of the conditionals — **only because $z$ is discrete**. Continuous $z$ cannot be seen that way.

Optimize ELBO by iterating: freeze $\theta$, step $q$; freeze $q$, step $\theta$. “How do we optimize over functions $q$?” Hold that. For mixtures it becomes **tractable**. If you **cannot** find optimal $q$ analytically, that gap **is exactly why VAEs exist**.

The trap is calling a VAE “a GMM with extra layers” without the tractability fork.

You can now sample a GMM and point at the boxed $\arg\max$. What is still open: the EM proofs and the intractable-posterior paper.

### Analogy for this topic only

Three paint bowls. Die lands on 2; dip bowl 2. The overall cloud is 30% bowl 1, 50% bowl 2, 20% bowl 3.

Someone asks: **if $z$ is a slider, how many bowls?** Infinitely many — you left the mixture picture.

In lecture words: die = $p(z)$; bowl = $p(x\mid z)$; mix = discrete LVM.

### Local picture

```
  z ~ Categorical(α)
  x | z=j ~ N(μ_j, Σ_j)

  p(x) = Σ_j α_j N(x; μ_j, Σ_j)

  Notice: finite mix ⇔ discrete z.
          VAE lives on the other side of that ⇔.
```

### Bridge

Write the iteration, two homeworks, then the sentence the VAE paper starts with.

---

## Topic 9: EM; when VAE starts (34:10–43:53)

### Where this sits on the master map

This is **EM vs GAP**. [Iterate $q$ and $\theta$](./PREREQUISITES.md#p7-gmm-die) while $p_\theta(z\mid x)$ is a formula. When it is not, read **Auto-Encoding Variational Bayes**.

### Board / screenshot

No content frame. Reconstruction ~34:10–43:53.

```
  EM:  for t = 1..T
         q_{t+1}  ←  best q given θ_t
         θ_{t+1}  ←  best θ given q_{t+1}

  homework 1: likelihood NEVER DECREASES
              (equality allowed → can SATURATE)
  homework 2: q*(·|x) = p_θ(z|x)   (model posterior)
              tightness: ELBO = likelihood at that q

  Bayes: p(z|x) ∝ p(x|z) p(z)
  GMM: both known → closed μ, α, Σ  (or gradients)

  EM needs: tractable p_θ(z|x)  AND  tractable θ-step
  PGMs: designed so posterior is tractable

  IF p_θ(z|x) unknown  →  cannot EM
  VAE paper title: Auto-Encoding Variational Bayes
                   (not “variational autoencoders”)
```

Caption: two skipped proofs = previous course / homework; VAE intro is the intractable-posterior sentence.

### What he is establishing

Call the iteration **expectation–maximization (EM)**. $\theta_t,q_t$ at round $t$; freeze $\theta$, update $q$; freeze $q$, update $\theta$.

Two skipped proofs (previous course; homework if you did not take it):

1. This iteration makes the likelihood **never decrease**. He did **not** say it always **increases**. Equality is allowed. EM **can saturate** — known for mixtures in high dimension.
2. The $q$ that maximizes ELBO for **fixed** $\theta$ is **analytical**: $q^\star(z\mid x)=p_\theta(z\mid x)$, the **model posterior**. How to see it: ELBO’s maximum possible value is the likelihood (tight bound). Ask which $q$ makes the bound **equal** the likelihood; that $q$ is $p_\theta(z\mid x)$.

Algorithmically: start with some $\theta$; set $q_{t+1}=p_{\theta_t}(z\mid x)$ by **Bayes** $p(x\mid z)p(z)/\mathrm{norm}$. For GMM both pieces are known (Gaussian and $\alpha$). Plug $q$ into ELBO; **differentiate** in $\mu,\alpha,\Sigma$ (or use gradients if the mix is not closed form). Complete GMM-EM as assumed prerequisite.

EM is not GMM-only. Constraints: (1) you can **compute** $p_\theta(z\mid x)$ for a given $\theta$; (2) after plugging $q^\star$, you can **optimize** in $\theta$. **Probabilistic graphical models (PGMs)** are *constructed* so the posterior is tractable, then EM.

**If $p_\theta(z\mid x)$ is not tractable, EM dies.** Read the **introduction of the VAE paper**: they want LVMs whose **variational posterior is intractable**. The title is **not** “variational autoencoders.” It is **Auto-Encoding Variational Bayes**. For now, only the intro.

The trap is running “EM” with a guessed neural $q$ and calling it GMM, or skipping the never-decrease vs always-increase distinction.

```python
# EM — as he describes it. Not a VAE train loop.
# Constraint: p_theta(z|x) MUST be computable.

for t in range(1, T+1):
    # E-step (homework: this q maximizes ELBO at frozen theta)
    q = posterior_z_given_x(theta)   # Bayes: p(x|z) p(z) / Z
    # M-step
    theta = argmax_theta ELBO(theta, q)   # GMM: d/dμ, d/dα, d/dΣ
# homework: likelihood(theta) never decreases (may plateau)
```

You can now state EM’s two guarantees and the VAE gap. What is still open: what a **neural** LVM must *do*, and how nets represent $q$ and $p(x\mid z)$.

### Analogy for this topic only

If you can name the doctor’s exact posterior over diseases given symptoms, you run the two-desk iteration (diagnosis desk, then parameter desk).

Someone asks: **what if the posterior has no formula?** You cannot sit at the diagnosis desk. That sentence **is** the VAE paper’s intro.

In lecture words: $q^\star=p_\theta(z\mid x)$; intractable $\Rightarrow$ Auto-Encoding Variational Bayes.

### Local picture

```
  tractable p_θ(z|x)          intractable
         │                         │
         ▼                         ▼
        EM                      VAE / AEVB
   q* by Bayes                 next: nets + reparam
   θ by dELBO/dθ               (implementation NEXT class)

  Notice: PGM = architecture chosen so the left column works.
```

### Bridge

List three jobs every neural LVM must finish, rewrite ELBO as recon minus KL, then stop before the sampling trick.

---

## Topic 10: Three jobs; two NN styles; stop (43:53–55:24)

### Where this sits on the master map

This is **SPECS + STOP**. Neural LVM: learn without explicit $p_\theta(z\mid x)$, [sample](./PREREQUISITES.md#p7-gmm-die), [infer $q$](./PREREQUISITES.md#p5-posterior). ELBO’s two-term form is the VAE loss. Nets represent laws as [samples or parameters](./PREREQUISITES.md#p8-nn-two-ways). Reparameterization is **next**.

### Board / screenshot

No content frame. Reconstruction ~43:53–55:24.

```
  neural LVM must:
    1. learn without explicit p_θ(z|x)
    2. SAMPLE: z ~ p(z), then x ~ p(x|z)     (GMM: die then Gaussian)
    3. posterior inference q(z|x)
         discrete → clustering
         continuous → embedding

  ELBO =  E_q[log p_θ(x|z)]  −  KL( q(z|x) || p(z) )
          ↑ recon-like            ↑ keep q near prior
          = VAE / DDPM loss form

  NN represents a distribution TWO ways
    A. output SAMPLES     (classifier; GAN G)
    B. output PARAMETERS  (μ, σ of a Gaussian)
  VAE: probabilistic nets, style B, assumed parametric family

  NEXT: E_q with q being optimized → reparameterization
        ("differentiate through sampling")
        then one instantiation, backprop, implement
```

Caption: good stopping point; no computer this hour.

### What he is establishing

Same data: $n$ IID $x\in\mathbb{R}^D$. Same ELBO $J_\theta(q)$. A **neural** LVM must:

1. Learn the LVM **without** knowing $p_\theta(z\mid x)$ explicitly (the EM-breaking case).
2. **Enable sampling** from $p_\theta(x\mid z)$. If it cannot, it is not a generative model. Mixtures and PGMs already do: sample $z\sim p_\theta(z)$ (roll die), then $x\sim p_{\theta^\star}(x\mid z)$. **GMMs are samplers / generative models.**
3. **Posterior inference** = embedding extraction = produce $q(z\mid x)$. GMM: for each $x$, a distribution over $1..M$ — **clustering** (likelihood of each group). Continuous $z$: **embedding**.

All three.

Break ELBO further: $p(x,z)=p(x\mid z)p(z)$. Log product minus log $q$ becomes

$$
J=\mathbb{E}_{q(z\mid x)}[\log p_\theta(x\mid z)]-\mathrm{KL}\bigl(q(z\mid x)\,\|\,p(z)\bigr).
$$

**This is the so-called loss of a VAE and of DDPM.** In English: expected log **conditional data likelihood** under $q$, minus KL between variational posterior and the **marginal / prior** on $z$.

To use nets: represent $p_\theta(x\mid z)$ and $q(z\mid x)$ with neural nets. **What does that mean?** Two ways:

- **Deterministic / sample output:** the net **emits draws**. A classifier represents $p(y\mid x)$ by outputting a label (a sample). A **GAN generator** outputs $\hat x$, a sample of $p_\theta$.
- **Parameter output:** the net emits **parameters of a named family**, not a draw. Naive VAE: $q(z\mid x)$ is Gaussian; the net maps $x$ to **mean and variance**.

This course will use **both**. In a VAE he calls them **probabilistic neural networks** in the sense that they output **parameters of assumed parametric forms** for $q(z\mid x)$ and $p(x\mid z)$.

**Stop.** Next: the ELBO has an expectation **under a $q$ that is itself being optimized**. Turning that into a sample average is **non-trivial**. The trick is **reparameterization** — people say “differentiate through the sampling step.” Then one instantiation, backprop, implement on a computer — **as he did for GAN**. Thank you.

The trap is coding `z = mu + sigma * eps` today and claiming it was this lecture. He named the trick and deferred it.

```python
# VAE-shaped ELBO (math he wrote). Not a training loop.
# J = E_{z ~ q(z|x)}[ log p_theta(x|z) ]  -  KL( q(z|x) || p(z) )

# Represent q(z|x) and p(x|z) by nets that output PARAMETERS
# (e.g. Gaussian mean/var) — not GAN-style samples.
# How to backprop through z ~ q(...)  =  NEXT lecture (reparam).
```

You can now list the three jobs, the two-term ELBO, and the two NN styles. What is still missing is the reparameterization algebra and a coded VAE — that is the leftover on purpose.

### Analogy for this topic only

The shop must (1) learn recipes without a closed-form diagnosis table, (2) bake a new cake from a die roll, (3) still tell you *which bowl* an old cake probably came from.

Someone asks: **does the net hand me a cake or a recipe card?** VAE: recipe card $(\mu,\sigma)$. GAN: cake.

The wrong move is backpropping a VAE tonight.

In lecture words: three jobs; ELBO = recon term minus KL; parameter-output nets; reparam next.

### Local picture

```
  J =  E_q log p(x|z)   −   KL(q(z|x) || p(z))

  sample path:   p(z) --> z --> p(x|z) --> new x
  infer path:    x --> q(z|x) --> embedding / cluster weights

  NN fork:  samples  |  parameters
            GAN G      VAE q-net, decoder family

  Notice: expectation under a moving q  →  reparam next hour.
```

### Bridge

The leftover problem is **differentiating through a draw from $q_\phi$**. Next lecture: reparameterization, then an instantiation you can implement — not a sixth EM proof.

---

## External references

How to use: after Topics 7–9 read Kingma & Welling’s **intro** (as he said). After Topic 10, CS236 / CS231n for the same ELBO board. Implementation videos wait until the next NPTEL hour.

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| Instructor notes — [Drive PDF](https://drive.google.com/file/d/1Kebb00VehPBMyleuw2ubp2qbvrBqyvZZ/view) | Whole hour | Board companion linked in the YouTube description (LVM, GMM, VAE). |
| Kingma & Welling — *Auto-Encoding Variational Bayes* — [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) | Topics 9–10 | The paper he names; title is AEVB, not “VAE”; intro = intractable posterior. |
| Stanford CS236 — VAE course notes — [notes/vae](https://deepgenerativemodels.github.io/notes/vae/) | Topics 5–8, 10 | ELBO, $q(z\mid x)$, why the bound exists — same map boxes. |
| Stanford CS236 2023 Lecture 6 — VAEs — [video](https://www.youtube.com/watch?v=8cO61e_8oPY) | Topics 7–10 | University hour on ELBO and why it is called a variational autoencoder. |
| Stanford CS231n (2026) Lecture 13 slides — VAE/ELBO — [lecture_13.pdf](https://cs231n.stanford.edu/slides/2026/lecture_13.pdf) | Topics 7, 10 | Latest CS231n ELBO derivation: $\mathbb{E}\log p(x\mid z)-\mathrm{KL}(q\|p(z))$. |
| MIT 6.S191 Lecture 4 — Deep generative modeling — [video](https://www.youtube.com/watch?v=R8V8CbuxryI) | Topics 1, 3, 10 | Places VAE next to GAN/diffusion — the family he opened with. |
| Doersch — *Tutorial on VAEs* — [arXiv:1606.05908](https://arxiv.org/abs/1606.05908) | Topics 6–8, 10 | Written ELBO + the two-term loss he reaches at 48:35. |
| CS231n Spring 2025 Lecture 13 — Generative models I — [notes](https://raimbekovm.github.io/cs231n-2025-notes/lectures/13-generative-models.html) | Topics 1, 8 | VAE/ELBO in the 2025 vision course, before the diffusion lecture. |

---

## Sources

- Video: [Lec 20 LVM and VAE intro](https://www.youtube.com/watch?v=4djE9goJtKs) · playlist [MFGAI](https://www.youtube.com/playlist?list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK) (YouTube `&index=1`; learning order 21)
- Captions: `raw/captions.en.vtt` / `raw/captions.en.timed.txt` (ASR: ELBO, KL, Gaussian, Jensen, Auto-Encoding Variational Bayes)
- Claim sheets: `raw/claims/topic-01.md` … `topic-10.md`
- Course notes: [Drive](https://drive.google.com/file/d/1Kebb00VehPBMyleuw2ubp2qbvrBqyvZZ/view)
- Ingest: captions yes · video no (HTTP 403) · frames no · `ingest_evidence: E2`
