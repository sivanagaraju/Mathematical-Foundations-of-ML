# Prerequisites — warm-up before Lec 19 (Inversion with GANs and FID)

> **Do this first** if “manifold,” “joint,” “decoder-only,” or “Fréchet” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Continues [Lec 05 GANs](../28-Lec05-Generative-Adversarial-Networks/NOTES.md) and [Tutorial 12](../29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/NOTES.md). Wasserstein geometry is [Lec 18](../30-Lec18-Wasserstein-GAN/) (package in progress).  
> **Beginner:** purpose · definition · micro · analogy · ASCII · notice · mini-check.

This hour is **theory**, not a training loop. Last lectures built a **sampler**: noise in, fake image out. Today we need the **other direction** — given a real photo, find the noise that would have printed it — and a **score** for “how real do my prints look?” with no likelihood.

```
  After this warm-up you can say:

  "Pixels live in a huge grid R^D; the useful codes live in a small R^K."
  "The manifold hypothesis: faces do not fill every pixel combination."
  "Sampler = K→D; encoder = D→K."
  "A joint is a law on pairs; a marginal is one side of that pair."
  "PCA is encoder-only: squash, no printer."
  "A vanilla GAN is decoder-only: printer, no ingest."
  "Wasserstein-2 of two Gaussians is a closed formula of means and covariances."
  "FID runs that formula on Inception embeddings, not on raw pixels."
```

```
  §1  Ambient R^D vs latent R^K         ──► Topics 1–2
  §2  Manifold hypothesis               ──► Topic 1
  §3  Sampler vs encoder                ──► Topics 2, 5
  §4  Joint vs marginal                 ──► Topics 7–8
  §5  PCA = encoder-only                ──► Topics 2, 5
  §6  Decoder-only vs both directions   ──► Topics 5–6
  §7  W2 of two Gaussians               ──► Topic 9
  §8  Embed then measure (FID idea)     ──► Topics 9–10
```

**One scene through all eight.** A **passport-photo studio**. A customer walks in with a face. The printer can invent new faces from 10 knobs. Today’s job is: recover those knobs for *this* face, walk them a little, and score whether the prints look like the album.

```
  ALBUM     128 real passport photos          x in R^D
  KNOBS     10 Gaussian numbers               z in R^K
  PRESS     G turns knobs into a print        x̂ = G(z)
  CLERK     D used to judge single photos     (last lectures)
  TODAY     we also need a code for THIS photo
            and a museum score for the whole pile of prints
```

---

## 🪨 Mathematical & GAN Inversion Rosetta Stone

| Symbol / Term | Theoretical Concept | Plain-English Software Metaphor | Dedicated MathsTerm Guide |
| :--- | :--- | :--- | :--- |
| **$x \in \mathbb{R}^D$** | Ambient Image Vector | High-dimensional pixel array ($D=784$ or $12288$) | [Tensors & Shapes](../../../MathsTerms/Tensors_and_Shapes.md) |
| **$z \in \mathbb{R}^K$** | Latent Representation Vector | Low-dimensional continuous knobs ($K \ll D$) | [Latent Variable Models](../../../MathsTerms/Latent_Variable_Models.md) |
| **$G(z)$** | Push-Forward Generator (Decoder) | Printing press generating photo $\hat{x}$ from code $z$ | [Autoregressive Models](../../../MathsTerms/Autoregressive_Models.md) |
| **$E(x)$** | Inversion Encoder Network | Ingest camera mapping real photo $x$ to latent code $z$ | [Autoencoders & Latent Spaces](../../../MathsTerms/Autoencoders_and_Latent_Spaces.md) |
| **$z^* = \arg\min_z \|x - G(z)\|^2$** | Direct Optimization Inversion | Gradient descent on latent code to reconstruct target photo | [Gradient Descent](../../../MathsTerms/Gradient_Descent.md) |
| **$W_2^2(P_r, P_g)$** | 2-Wasserstein Distance | Quadratic optimal transport cost between feature Gaussians | [Wasserstein Distance & EMD](../../../MathsTerms/Wasserstein_Distance_and_EMD.md) |
| **$\text{FID}(r, g)$** | Fréchet Inception Distance | $\|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$ | [Fréchet Inception Distance](../../../MathsTerms/Frechet_Inception_Distance.md) |
| **$\text{BiGAN} / \text{ALI}$** | Joint Adversarial Matching | Discriminator matching joint pairs $(x, E(x))$ vs $(G(z), z)$ | [Joint, Marginal & Conditional Dist](../../../MathsTerms/Joint_Marginal_Conditional_Dist.md) |

---

## 1. Ambient space vs latent space

<a id="p1-ambient-latent"></a>

**Purpose.** Every picture in this hour is two numbers: how big the **photo** is, and how big the **code** is.

**Definition.** **Ambient space** is where the data actually live — pixels. Write it $\mathbb{R}^D$. For a flattened MNIST digit, $D=28\times 28=784$. For a color $64\times 64$ face, $D=64\times 64\times 3=12288$. **Latent space** (also called the **code**, **embedding**, or **representation**) is a much smaller list of numbers $\mathbb{R}^K$ with $K\ll D$. Typical GAN noise is $K=100$ or even $K=10$.

**Micro.** One passport photo: $D=12288$ numbers between black and white. The studio’s knobs: $K=10$. Ten knobs cannot spell every possible 12288-pixel grid. They can still spell a lot of *faces*, if faces do not use all 12288 degrees of freedom. A flattened MNIST 3 is $D=784$; a typical GAN noise is $K=100$. $100\ll 784$.

**Analogy.** A sheet of graph paper with 12,288 squares is the ambient grid. A combination lock with 10 dials is the latent code. The press reads the lock and paints the sheet. Almost every random painting of the sheet is TV static, not a face. A ZIP file is the same idea: a small list of numbers that *stands for* a huge photo, not a second copy of every pixel.

```
  latent knobs          ambient photo
  z in R^K              x in R^D
  K = 10                D = 12288
  "what kind of face"   "every pixel"
```

**Notice.** $K\ll D$ is a *design choice* justified by the next idea. It is not a law of nature.

**Mini-check.** If you flatten a $28\times 28$ digit, what is $D$? If $K=100$, is $K$ smaller than $D$?

---

## 2. Manifold hypothesis

<a id="p2-manifold"></a>

**Purpose.** Why bother with a small $K$ at all?

**Definition.** The **manifold hypothesis** says: the data you care about (faces, digits, speech) are **well supported on far fewer dimensions** than the ambient grid. Almost every random point in $\mathbb{R}^D$ is garbage. The interesting points sit on a thin “sheet” (a **manifold**) of dimension near $K$.

**Micro.** There are $256^{784}$ possible 8-bit MNIST grids. Almost none of them look like a handwritten 3. The 3s form a tiny subset. A generator that starts in $\mathbb{R}^{100}$ is betting that 100 numbers are enough to crawl around that subset.

**Analogy.** Every GPS coordinate on Earth is three numbers in a huge box, but humans live on a **thin crust** (the surface). You do not need a 3-D warehouse of coordinates to describe “places a person can stand.” Two numbers (lat, lon) plus a tiny height already cover the useful set.

A second picture: a bowl of spaghetti in a kitchen. The pasta traces a 1-D curve that *bends through* 3-D space. Digit photos are like that pasta inside the 784-D cube. $K$ is “how many numbers to crawl along the pasta,” not “how many pixels the camera has.” Too small a $K$ (one dial) cannot tell a 3 from an 8. Too large a $K$ (784 dials) wastes dimensions on empty air. He says an **optimal $K$** can exist; this lecture does not compute it.

```
  ambient cube R^D          (almost empty of faces)

        .  .  .  noise  .  .  .
           .  [ thin face sheet ]  .
        .  .  .  noise  .  .  .

  start in R^K, then G paints onto that sheet
```

**Notice.** He also says there can exist an **optimal** $K$ (too small: blurry / mode-drop; too large: wasted dimensions). This lecture does not compute that $K$.

**Mini-check.** If the manifold hypothesis were false, would starting from small $K$ still be a good idea?

---

## 3. Sampler vs encoder: two arrows

<a id="p3-sampler-encoder"></a>

**Purpose.** The whole hour is about *which way* the map points.

**Definition.** A **sampler** (generator, **decoder**) maps **latent → data**:

$$
G:\mathbb{R}^K\to\mathbb{R}^D,\qquad \hat x = G(z),\quad z\sim\mathcal{N}(0,I).
$$

An **encoder** (inverter, embedding map) maps **data → latent**:

$$
E:\mathbb{R}^D\to\mathbb{R}^K,\qquad \hat z = E(x),\quad x\sim p_x.
$$

**Micro.** Customer hands you photo $x$. Sampler alone cannot eat $x$. Encoder returns a 10-number code $\hat z$. Push $\hat z$ through $G$ and you hope to recover the same photo. That round-trip $G(E(x))\approx x$ is the inversion test. Sampling a *new* face is $G(z_{\mathrm{fresh}})$ — different job, no $x$ in the input.

**Analogy.** The press turns lock settings into a print. The locksmith looks at a print and guesses the lock settings. Until today you only owned a press. Asking the press “reprint *this* JPEG” is like stuffing a photograph into a combination lock. Wrong slot.

```
  SAMPLER / decoder          ENCODER / inverter
  z  -->  G  -->  x̂          x  -->  E  -->  ẑ
  knobs to photo             photo to knobs
```

**Notice.** Later he will use **embedding**, **representation**, **feature**, **latent**, and **code** as names for the same $K$-vector. They are interchangeable in this lecture.

**Mini-check.** Which arrow does a vanilla GAN give you for free?

---

## 4. Joint vs marginal

<a id="p4-joint-marginal"></a>

**Purpose.** BiGAN does not match photos alone. It matches **pairs**.

**Definition.** A **joint** is a law on a **tuple** $(a,b)$ together. A **marginal** is the law of one coordinate after you ignore the other. If two joints are equal, **every** marginal of those joints is equal (the photo-side, the code-side, everything).

**Micro.** Pair A: `(print G(z), the knobs z that made it)`. Pair B: `(real photo x, the code E(x) the locksmith guessed)`. Matching the two *clouds of pairs* is stronger than matching prints to photos.

**Analogy.** Three students:

```
  name   height   weight
  Ann      180      60     tall-light
  Bob      160      90     short-heavy
  Cat      170      70     average
```

You can match the **height list** $\{180,160,170\}$ and the **weight list** $\{60,90,70\}$ and still staple Ann to 90 kg. That coupling is the **joint**. Matching joints forces every side to match; matching sides does **not** force the joints.

Same studio: matching “prints look like photos” is only the image-marginal. Matching “this print came with *these* knobs” is the pair-joint BiGAN actually trains.

```
  JOINT of pairs                 MARGINALS (sides)

  (x̂, z)  =  (G(z), z)           x̂-cloud   and   z-cloud
  (x, ẑ)  =  (x, E(x))           x-cloud   and   ẑ-cloud

  if joints match  =>  all sides match
```

**Notice.** Treating $z$ as “extra dimensions of the data you generate” is his slogan for this trick. The critic now sees a longer vector, not a lone image.

**Mini-check.** Can two joints differ even when both photo-marginals match?

---

## 5. PCA is encoder-only

<a id="p5-pca"></a>

**Purpose.** Classical ML already knew **data → small space**. It did not automatically give a sampler.

**Definition.** **Principal component analysis (PCA)** is a **linear** dimensionality reduction: it finds directions of largest variance and **projects** $x\in\mathbb{R}^D$ onto $\mathbb{R}^K$. That is **encoding**. A reconstruction $\tilde x = \text{basis}\cdot\text{codes}$ is a linear undo, not a trained nonlinear generator of *new* faces from Gaussian noise.

**Micro.** Stack 128 passport photos as rows. PCA might keep the first 10 principal axes. Each photo becomes 10 scores. You can *approximate* a photo you already have. You do not get a trained $G$ that turns fresh $\mathcal{N}(0,I)$ noise into a never-seen face.

**Analogy.** PCA is a **document shredder that keeps the 10 fattest strips**. You can tape those strips back into a blurry original. You cannot hand the shredder 10 random numbers and expect a new customer’s face.

```
  PCA:     x in R^D  --linear project-->  codes in R^K
           (encoder-only; no Gaussian sampler)

  GAN:     z in R^K  --nonlinear G-->     x̂ in R^D
           (decoder-only until today)
```

**Notice.** He flags **variational autoencoders (VAEs)** as later models that bake **both** arrows in from the start. That is next class, not this one.

**Mini-check.** If someone says “PCA generates images,” what arrow are they mixing up?

---

## 6. Decoder-only, encoder-only, encoder–decoder

<a id="p6-decoder-only"></a>

**Purpose.** He borrows language from language models so the GAN gap has a name.

**Definition.**

| Name | Arrow | Example in this hour |
|------|--------|----------------------|
| **Decoder-only** | latent → data (samples; no embedding of a given $x$) | vanilla GAN; he also names transformer LMs as decoder-only samplers |
| **Encoder-only** | data → latent (no sampler back) | PCA |
| **Encoder–decoder** | both directions | BiGAN / ALI today; VAE later |

**Micro.** You cannot paste a customer’s JPEG into $G$. $G$ only accepts $z\sim\mathcal{N}(0,I)$. So “put spectacles on *this* photo” is illegal until you have an inverter. “Sample a new face” is **not** inversion — inversion’s extra input is a **fixed** $x$.

**Analogy.** A radio that only **transmits** (decoder-only) cannot tell you the station a song came from. A radio that only **receives** (encoder-only) cannot broadcast a new song. You want both. ChatGPT-style transformer LMs, in his naming, are also decoder-only: they emit tokens; they do not hand you an embedding of *this* paragraph the way an encoder would.

```
  decoder-only   z --> G --> x̂          cannot ingest x
  encoder-only   x --> E --> ẑ          cannot sample new x
  both           z --> G --> x̂
                 x --> E --> ẑ          round-trip G(E(x)) ≈ x
```

**Notice.** **Inversion** is the named job: given $x$ from $p_x$, find $z$ such that $G(z)\approx x$. “What point in the normal would have generated this under my GAN?”

**Mini-check.** Is “sample a new face” inversion? What extra input does inversion need?

---

## 7. Wasserstein-2 between two Gaussians

<a id="p7-w2-gaussians"></a>

**Purpose.** FID is not a mysterious brand name. It is a **closed-form** distance after a modeling choice.

**Definition.** The **Wasserstein-2** distance $W_2$ (also called Fréchet distance when the laws are Gaussian) measures how much work it takes to move one cloud of mass onto another, with cost = squared Euclidean distance. For **two Gaussians** $\mathcal{N}(\mu_r,\Sigma_r)$ and $\mathcal{N}(\mu_g,\Sigma_g)$ there is an **algebraic formula** — means, traces of covariances, and a matrix square-root term. No inner min-max to run at test time.

**Micro.** Cloud R: 128 real-photo embeddings, mean $\mu_r$, covariance $\Sigma_r$. Cloud G: 128 fake-photo embeddings, mean $\mu_g$, covariance $\Sigma_g$. Plug into the formula; get one number.

Tiny numbers. Suppose both clouds are 1-D Gaussians with variance 1. If $\mu_r=\mu_g=0$, the closed form is $0$. If $\mu_r=0$ and $\mu_g=3$, the mean term is $3^2=9$ and FID is at least 9. **Lower is better.** That is why “FID 21 vs FID 90” is a ranking, not a probability.

**Analogy.** Two oval ink-blots on the desk. How far are the **centers**, and how different are the **stretches**? If both blots are perfect ovals (Gaussians), that question has a calculator answer. If the blots are weirdly shaped, you do **not** get that free formula — which is a later caveat.

```
  two clouds of vectors
       μ_r, Σ_r     μ_g, Σ_g
            \         /
             W2 closed form
             (means + traces − sqrt term)
```

**Notice.** Last lectures used $W$ as a **training** discrepancy (WGAN). Today $W_2$ is reused as an **evaluation** number on **already trained** samples. Different job, same geometry family.

**Mini-check.** Why is “assume Gaussian” doing so much work here?

---

## 8. Measure in an embedding space, not in pixels

<a id="p8-inception"></a>

**Purpose.** Pixel $W_2$ is the wrong museum. FID changes the museum, then applies §7.

**Definition.** **Inception** is a large convolutional net trained on **ImageNet** (a big labeled-photo dataset). You freeze it. You push every real image and every generated image through it and **tap a vector from some internal layer**. Those vectors are the new “locations.” Fit two Gaussians. The $W_2$ of those Gaussians is the **Fréchet Inception Distance (FID)**. **Lower is better.**

**Micro.** Two photos of the same person, one shifted by two pixels: pixel $W_2$ can look large; Inception embeddings often stay close (same face). Two photos that are pixel-close static vs a face: pixel $W_2$ can look small; embeddings separate them.

Worked contrast. Photo A = a 3. Photo B = the same 3 slid one pixel. Photo C = white noise with the same average brightness as A. Pixel distance can rank B far and C close. A museum guide who has seen ImageNet digits will usually put A near B and C far away. FID uses that guide, then the oval-blot formula.

**Analogy.** Do not judge two novels by counting matching characters on the page. Feed both through a **critic who already read a million books** (Inception), then compare the critic’s notes (embeddings) with the oval-blot formula. A two-letter typo is a huge pixel (character) distance and a tiny meaning distance. FID wants the meaning distance.

```
  pixels x, x̂
       |  frozen Inception
       v
  vectors  φ(x), φ(x̂)
       |  fit two Gaussians
       v
  W2 = FID     (lower ⇒ closer clouds ⇒ better prints)
```

**Notice.** He **does not remember** which Inception layer to tap (“68th or something”). Do not invent a layer index from memory. The idea is: **some** perceptual layer, then Gaussian $W_2$.

**Mini-check.** If FID is huge, are you looking at pixels or at embeddings? Which direction of FID is “better”?

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
