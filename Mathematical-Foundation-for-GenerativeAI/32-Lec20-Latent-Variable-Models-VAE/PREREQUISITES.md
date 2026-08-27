# Prerequisites — warm-up before Lec 20 (LVM and VAE intro)

> **Do this first** if “marginal,” “posterior,” “Jensen,” or “ELBO” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues [Lec 19 inversion / FID](../31-Lec19-Inversion-GANs-FID/NOTES.md). Adversarial / VDM is closed.  
> **Beginner:** purpose · definition · micro · analogy · ASCII · notice · mini-check.

This hour is **chalk**, not a Colab. He derives why a hidden $z$ forces a **lower bound** (ELBO) and why **EM** dies when the posterior is intractable. **Implementation and the reparameterization trick are next class.** Do not expect a training loop today.

```
  After this warm-up you can say:

  "Observed x is on the table; latent z is a hidden story for each x."
  "p(x) is the marginal of the joint p(x,z) — sum or integral over z."
  "Discrete z groups data (clustering); continuous z is a feature vector."
  "Min KL on p_x vs p_θ is the same as max expected log p_θ (MLE)."
  "q(z|x) is a whole distribution of hidden stories given this x."
  "log of an average is not the average of logs; Jensen gives a lower bound."
  "A GMM rolls an M-face die, then samples that Gaussian."
  "A net can emit samples of a law, or emit the parameters of a law."
```

```
  §1  Observed vs latent                 ──► Topics 1–2
  §2  Joint vs marginal (sum / integral) ──► Topics 1, 8
  §3  Discrete z vs continuous z         ──► Topics 2–3
  §4  KL drop entropy = MLE              ──► Topic 4
  §5  Posterior q(z|x)                   ──► Topics 5, 9
  §6  Jensen: log E vs E log             ──► Topics 6–7
  §7  Mixture / die + Gaussians          ──► Topics 8–9
  §8  NN: samples vs parameters          ──► Topic 10
```

**One scene through all eight.** A **digit workshop**. You have 128 unlabeled photos of handwritten 3s and 8s ($x$). You never wrote down *why* a 3 looks fat or tilted. That missing why is $z$.

```
  ALBUM     128 digits x in R^D          observed
  STORY     z  (cluster id OR knobs)     hidden / latent
  TODAY     write p(x) using z, without ever seeing z
```

---

## 🪨 Latent Variable Models & VAE Rosetta Stone

| Symbol / Term | Theoretical Concept | Plain-English Software Meaning | Dedicated MathsTerm Guide |
| :--- | :--- | :--- | :--- |
| **$x \in \mathcal{X}$** | Observed Data Point | Visible pixel tensor in dataset | [Tensors & Shapes](../../MathsTerms/Tensors_and_Shapes.md) |
| **$z \in \mathcal{Z}$** | Latent Random Variable | Unobserved hidden recipe / code vector | [Autoencoders & Latent Spaces](../../MathsTerms/Autoencoders_and_Latent_Spaces.md) |
| **$p_\theta(x, z)$** | Generative Joint Distribution | Model probability of pair $(x, z) = p_\theta(x \mid z) p(z)$ | [Latent Variable Models](../../MathsTerms/Latent_Variable_Models.md) |
| **$p_\theta(x) = \int p_\theta(x, z) dz$** | Marginal Likelihood (Evidence) | Total observable evidence across all possible hidden causes | [Likelihood & Log-Likelihood](../../MathsTerms/Likelihood_and_Log_Likelihood.md) |
| **$p_\theta(z \mid x)$** | True Intractable Posterior | Exact posterior probability of code $z$ given image $x$ | [Joint, Marginal & Conditional Dist](../../MathsTerms/Joint_Marginal_Conditional_Dist.md) |
| **$q_\phi(z \mid x)$** | Variational Posterior (Encoder) | Neural approximation parameterized by $\mu_\phi(x), \sigma_\phi^2(x)$ | [ELBO & Variational Inference](../../MathsTerms/ELBO_and_Variational_Inference.md) |
| **$z = \mu + \sigma \odot \epsilon$** | Reparameterization Trick | Differentiable sampling enabling backpropagation through encoder | [Reparameterization Trick](../../MathsTerms/Reparameterization_Trick.md) |
| **$\mathcal{L}_{\text{ELBO}}(\theta, \phi; x)$** | Evidence Lower Bound | Tractable lower bound $\le \ln p_\theta(x)$ optimized by gradient ascent | [ELBO & Variational Inference](../../MathsTerms/ELBO_and_Variational_Inference.md) |
| **$D_{\text{KL}}(q_\phi \parallel p)$** | Kullback-Leibler Divergence | Information penalty regularizing latent space towards prior | [KL Divergence](../../MathsTerms/KL_Divergence.md) |

---

## 1. Observed vs latent

<a id="p1-observed-latent"></a>

**Purpose.** Every later formula has two rooms: the photo, and a hidden story about the photo.

**Definition.** **Observed** $x$ is what the dataset actually stores (pixels). **Latent / hidden / unobserved** $z$ is a variable the *model* invents. You never get a file of $z$ values. You still treat each $x_i$ as having some $z_i$.

**Micro.** $x$ = $28\times 28$ digit. $z$ might be “this is cluster 2” or a 4-vector (thickness, tilt, …). The JPEG does not contain those numbers.

**Analogy.** You see 128 finished cakes. You do not see the recipe card. The recipe card is latent. The cake is observed.

```
  seen:   x_1, x_2, …, x_n     (album)
  unseen: z_1, z_2, …, z_n     (stories)
  model still talks about both
```

**Notice.** Last class a GAN had $z$ as **input noise** to a decoder. Here $z$ is a **hidden cause** you must *infer* from $x$.

**Mini-check.** If someone hands you only JPEGs, do you have samples of $z$?

---

## 2. Joint vs marginal (the integral)

<a id="p2-joint-marginal"></a>

**Purpose.** The LVM slogan is one line: the model of $x$ is obtained by **erasing** $z$.

**Definition.** A **joint** $p(x,z)$ is a law on the pair. The **marginal** of $x$ is that joint after you sum or integrate $z$ away:

$$
p_\theta(x)=\sum_z p_\theta(x,z)\quad\text{(discrete $z$)},\qquad
p_\theta(x)=\int p_\theta(x,z)\,dz\quad\text{(continuous $z$)}.
$$

**Micro.** Two clusters. Joint says “this 3 and cluster 1 together.” Marginal says “this 3,” after adding both cluster stories.

**Analogy.** A list of (student, club) pairs. The club column is $z$. If you hide the club column, you still have a list of students — that hidden-column list is the marginal.

```
  joint p(x,z)  --erase z-->  marginal p(x)
  that marginal IS the LVM's p_θ(x)
```

**Notice.** You almost never write $p(x,z)$ on a whiteboard as a table. You factor $p(z)\,p(x\mid z)$ and then mix.

**Mini-check.** If $z$ has 3 values, is $p(x)$ a sum of 3 terms or an integral?

---

## 3. Discrete $z$ vs continuous $z$

<a id="p3-discrete-continuous"></a>

**Purpose.** Same definition, two readings.

**Definition.** **Discrete** $z\in\{1,\ldots,M\}$: after you assign each $x_i$ a $z_i$, you have **M piles**. That *is* clustering (k-means, GMM, hierarchical). **Continuous** $z\in\mathbb{R}^K$: each $x_i$ gets a $K$-vector — a **feature / embedding**. Dims *might* mean thickness and tilt on MNIST; they might mean nothing readable. Interpretable dims are an **open problem**.

**Micro.** $M=2$: pile of 3s vs pile of 8s. $K=4$: four knobs per digit.

**Analogy.** Discrete $z$ is a **stamp** (red / blue / green). Continuous $z$ is a **mixing board** with sliders. Stamps always group. Sliders may or may not be named.

```
  discrete z = 1..M     →  M groups     (clustering)
  continuous z in R^K   →  K knobs      (embedding)
```

**Notice.** $x$ can stay in $\mathbb{R}^D$ (an image) in both cases. Only $z$’s type changes.

**Mini-check.** Is k-means an LVM? What type of $z$?

---

## 4. Min KL is max likelihood

<a id="p4-kl-mle"></a>

**Purpose.** Why the LVM objective is “maximize expected log $p_\theta$.”

**Definition.** $\mathrm{KL}(p_x\|p_\theta)=\mathbb{E}_{p_x}[\log p_x]-\mathbb{E}_{p_x}[\log p_\theta]$. The first term is **entropy of the data**. It does not contain $\theta$. Drop it. Minimizing KL is exactly **maximizing** $\mathbb{E}_{p_x}[\log p_\theta(x)]$ — **maximum likelihood**.

**Micro.** You cannot change how messy the album is. You can only change the model.

**Analogy.** You cannot edit the weather (entropy of the universe). You can only carry an umbrella ($\theta$). Optimization only sees the umbrella.

```
  KL =  H(data)  −  E[log p_θ]
           ↑ ignore               ↑ maximize this
```

**Notice.** He says LVMs *typically* use KL; other divergences exist.

**Mini-check.** If you change $\theta$, does $H(p_x)$ move?

---

## 5. Posterior $q(z\mid x)$

<a id="p5-posterior"></a>

**Purpose.** “Estimate $z$” in probability language is not a single number.

**Definition.** The **latent posterior** is a density $q(z\mid x)$: given *this* photo, a **distribution** over hidden stories. **Soft** (GMM): $x$ has probabilities over all $M$ clusters. **Hard / Dirac** (k-means): $q$ is a spike — one $z$ only.

**Micro.** Digit looks halfway 3/8. Soft: $q(z{=}3)=0.6$, $q(z{=}8)=0.4$. Hard: stamp “3” and stop.

**Analogy.** A doctor’s differential diagnosis (distribution over diseases) vs a single stamp in the chart.

```
  k-means:   q = spike     one z per x
  GMM:       q = weights   M numbers that sum to 1
```

**Notice.** Jointly learning $z$ with $\theta$ **means** learning $q(z\mid x)$, not a spreadsheet of $z_i$.

**Mini-check.** Can two different $z$ values both be plausible for one $x$?

---

## 6. Jensen: $\log\mathbb{E}$ vs $\mathbb{E}\log$

<a id="p6-jensen"></a>

**Purpose.** Why we get a **bound** instead of the true log-likelihood.

**Definition.** **Jensen:** if $f$ is **concave** (log is), $f(\mathbb{E}[Y])\ge \mathbb{E}[f(Y)]$. So

$$
\log\mathbb{E}_q[Y] \;\ge\; \mathbb{E}_q[\log Y].
$$

Left: true (intractable) log-likelihood after rewriting. Right: something you can hope to sample-average. The right-hand side is the start of **ELBO**.

**Micro.** $Y$ takes 2 and 8, equal chance. $\log\mathbb{E}[Y]=\log 5$. $\mathbb{E}[\log Y]=(\log 2+\log 8)/2=\log 4$. $\log 5>\log 4$. Bound is **lower**.

**Analogy.** Average the bank balances, then take log (true). Take log of each balance, then average (easier, smaller). You are not allowed to pretend they are equal.

```
  log( average )   ≥   average( log )
  true evidence        ELBO (lower bound)
```

**Notice.** He assigns **prove Jensen** as homework. Convex $f$ flips the inequality.

**Mini-check.** If you replace log by $y^2$ (convex), does the inequality flip?

---

## 7. Mixture: die then Gaussian

<a id="p7-gmm-die"></a>

**Purpose.** The classical LVM he actually writes.

**Definition.** **Gaussian mixture model (GMM):** $z$ is discrete $1..M$. $p(z{=}j)=\alpha_j$ (die faces). $p(x\mid z{=}j)=\mathcal{N}(\mu_j,\Sigma_j)$. Sample: **roll the die**, then draw from that Gaussian. The marginal $p(x)$ is a **convex combination** of $M$ Gaussians.

**Micro.** $M=2$, $D=2$. Two blobs on the plane. $\alpha=(0.7,0.3)$: 70% of points from blob 1.

**Analogy.** $M$ bowls of paint. Roll a die, dip from that bowl. The overall color cloud is a mix. If $z$ were a *slider* (continuous), you would not have a finite list of bowls.

```
  roll z ~ (α_1,…,α_M)
  then  x ~ N(μ_z, Σ_z)
  p(x) = Σ_j α_j N(x; μ_j, Σ_j)
```

**Notice.** Continuous $z$ cannot be written as a finite mix. That is why VAE is not “GMM with a neural net” in a trivial sense.

**Mini-check.** How many mean vectors does a GMM in $\mathbb{R}^D$ with $M$ components have?

---

## 8. Two ways a net represents a distribution

<a id="p8-nn-two-ways"></a>

**Purpose.** He stops the hour on this fork. VAE uses the second way.

**Definition.** A network can “be” $p$ in two styles:

| Style | Net outputs | Examples he names |
|-------|-------------|-------------------|
| **Samples** | a draw from the law | classifier’s label; **GAN generator** prints $\hat x$ |
| **Parameters** | numbers that *index* a named family | mean and variance of a Gaussian $q(z\mid x)$ |

**Micro.** Input $x$. Style A: emit a $z$ vector. Style B: emit $\mu(x)\in\mathbb{R}^K$ and $\sigma(x)\in\mathbb{R}^K$, and you still have to sample $z\sim\mathcal{N}(\mu,\mathrm{diag}\,\sigma^2)$ yourself.

**Analogy.** A chef who **hands you a plate** (sample) vs a chef who **hands you the recipe card** (parameters). VAE’s probabilistic nets, in his naming, hand you recipe cards for assumed families.

```
  GAN G:     z_noise --> G --> x̂          (sample)
  VAE q-net: x --> net --> (μ, σ)         (parameters)
                         then z ~ N(μ,σ)
```

**Notice.** Next class: sampling from a $q$ that is **itself being trained** needs the **reparameterization trick**. Not today.

**Mini-check.** If the net outputs $(\mu,\sigma)$ of a Gaussian, did it output a sample of $z$ yet?

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
