# Lec 19 — Inversion with GANs and FID

> **Video:** [Lec 19 Inversion with GANs and FID](https://www.youtube.com/watch?v=zw2DUzD0TLE) · **~28 min**  
> **Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md) · **Quiz:** [quiz.html](./quiz.html)

**Do PREREQUISITES before this article.**  
**Previous:** [Lec 05 GANs](../28-Lec05-Generative-Adversarial-Networks/NOTES.md) · [Tutorial 12 implementations](../29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/NOTES.md) · [Lec 18 WGAN](../30-Lec18-Wasserstein-GAN/) (ingest in progress)  
**Course:** Mathematical Foundations of **Generative AI**  
**Speaker:** NPTEL IISc · Prof. Prathosh A. P. · inversion, BiGAN/ALI, FID  
**Boards:** captions only (video 403) — ASCII reconstructions, no content frames.

This is a **chalk hour**. There is **no training loop** on screen. The FID recipe below is his **evaluation procedure** with comments, not invented PyTorch.

| When the lecture hits… | Warm-up |
|------------------------|---------|
| $K\ll D$, ambient vs code | [p1-ambient-latent](./PREREQUISITES.md#p1-ambient-latent) |
| Why small $K$ works | [p2-manifold](./PREREQUISITES.md#p2-manifold) |
| Sampler vs encoder | [p3-sampler-encoder](./PREREQUISITES.md#p3-sampler-encoder) |
| Pairs and joints | [p4-joint-marginal](./PREREQUISITES.md#p4-joint-marginal) |
| PCA as one-way squash | [p5-pca](./PREREQUISITES.md#p5-pca) |
| Decoder-only GAN | [p6-decoder-only](./PREREQUISITES.md#p6-decoder-only) |
| Closed-form $W_2$ | [p7-w2-gaussians](./PREREQUISITES.md#p7-w2-gaussians) |
| Inception then FID | [p8-inception](./PREREQUISITES.md#p8-inception) |

---

## Table of Contents

1. [Topic 1 — Sampler, manifold, $K\ll D$](#topic-1-sampler-manifold-kll-d-0001–0155) (00:01–01:55)
2. [Topic 2 — Inverse as representation](#topic-2-inverse-as-representation-0155–0446) (01:55–04:46)
3. [Topic 3 — Why embeddings: cluster, retrieve, edit](#topic-3-why-embeddings-cluster-retrieve-edit-0446–0707) (04:46–07:07)
4. [Topic 4 — Latent traversal](#topic-4-latent-traversal-0707–0929) (07:07–09:29)
5. [Topic 5 — Inversion; GAN is decoder-only](#topic-5-inversion-gan-is-decoder-only-0929–1117) (09:29–11:17)
6. [Topic 6 — BiGAN and ALI](#topic-6-bigan-and-ali-1117–1343) (11:17–13:43)
7. [Topic 7 — Tuple discriminator and joints](#topic-7-tuple-discriminator-and-joints-1343–1709) (13:43–17:09)
8. [Topic 8 — Encode then sample; algebra homework](#topic-8-encode-then-sample-algebra-homework-1709–1832) (17:09–18:32)
9. [Topic 9 — FID = $W_2$ of Inception Gaussians](#topic-9-fid--w_2-of-inception-gaussians-1832–2446) (18:32–24:46)
10. [Topic 10 — FID caveats, agents, next LVMs](#topic-10-fid-caveats-agents-next-lvms-2446–2812) (24:46–28:12)
11. [External references](#external-references)
12. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Given a photo $x$, recover the knobs $z$ that would reprint it, then score a pile of prints with no likelihood. Vanilla $G$ only samples, so BiGAN/ALI add an encoder and a critic on **pairs**. Matching those joints inverts $G$. FID is Wasserstein-2 of two Gaussians on frozen Inception embeddings; lower is better.

**Worldview arc:** from GAN-only sampling to a three-net tuple inverter, then FID.

### The approach (read this first)

```
  HAVE     G: z --> x̂     decoder-only sampler   (K ≪ D, manifold)
  BLOCK    x  -X->  z     G will not ingest a JPEG
  WANT     inversion      find z with G(z) ≈ x
           + a score      how close is the print pile to the album?

  METHOD   hire a locksmith E, change the clerk D
           D now stamps PACKETS, not lone photos
             fake packet:  (G(z), z)
             real packet:  (x, E(x))
           same saddle as GAN: min over (G AND E), max over D
           joints match  =>  all marginals match

  USE      sample:   z ~ N --> G --> new x̂
           invert:   ẑ = E(x);  x_rec = G(ẑ) ≈ x
           edit:     lerp z1 to z2 (spectacles), then G

  SCORE    n real, n fake  (post-training)
           φ = frozen ImageNet Inception (layer unnamed here)
           fit two Gaussians on φ-vectors
           closed-form W2 = FID     lower ⇒ better

  STOP     joint=opt algebra is homework
           no training loop on screen
           VAE / diffusion / AR next
```

### System context

```
  ╔══════════════════════════════════════════╗
  ║ Lec 05 / Tut 12: G samples; D on images  ║
  ║ Lec 18: Wasserstein as a training d      ║
  ║ Lec 20: VAE encoder+decoder baked in     ║
  ╚════════════════════╤═════════════════════╝
                       │ this lecture (~28 min)
                       ▼
        ┌──────────────────────────────────┐
        │ INVERT G  and  SCORE samples     │
        │ BiGAN/ALI tuples + FID           │
        └──────────────────────────────────┘
```

### Main blueprint

```
  ╔════ JOB ════╗
  ║ given photo ║
  ║ x, recover  ║
  ║ knobs z,    ║
  ║ then score  ║
  ║ print piles ║
  ╚════╤════════╝
       │ vanilla GAN only has
       ▼
  z ~ N(0,I) ══► G_θ ══► x̂     SAMPLER (decoder-only)
       │
       │ missing arrow
    ──X──►  x  -?->  z          cannot ingest a photo
       │
       ▼
  ┌─ THREE NETS (BiGAN / ALI) ─────────┐
  │  G: z → x̂     decoder / sampler    │
  │  E: x → ẑ     encoder / inverter   │
  │  D: pair → score                   │
  │     (x, E(x))  vs  (G(z), z)       │
  └──────────────┬─────────────────────┘
                 │ match joints
                 ▼
  p(x̂, z)  =  p(x, ẑ)   ⇒  all marginals match
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
  USE E then G           SCORE piles
  G(E(x)) ≈ x            FID path:
  walk z1 → z2           pixels ─► Inception
  (spectacles)           ─► two Gaussians ─► W2
                         lower is better
       │
  ┌ · · · ┴ · · · · · · · · · · · ┐
  │ STOP: joint algebra homework  │
  │       VAE / diffusion / AR    │
  └ · · · · · · · · · · · · · · · ┘
```

### Scenario walkthrough

**Story:** a customer uploads a passport photo and wants **spectacles** added. You also need a number that says whether your press’s prints look like the album.

1. **Sampler?** Noise through $G$ prints new faces. That is the box you already own.
2. **Need knobs of *this* photo?** Vanilla $G$ will not take a JPEG. That is the blocked arrow.
3. **Invert?** Train BiGAN/ALI so $D$ cannot tell `(real photo, E(photo))` from `(print, knobs)`. Joints match.
4. **Edit?** $z_1=E(\text{no glasses})$, $z_2=E(\text{glasses})$. Walk $z_1\to z_2$, decode with $G$.
5. **Score?** Push album and prints through frozen Inception, fit two Gaussians, report $W_2$. That number is FID. Lower wins.

### Failure / contrast

```
  WRONG  treat G as an encoder          (it only prints)
  WRONG  match image clouds only        (need pair joints)
  WRONG  pixel W2 as FID                (must embed first)
  WRONG  invent "layer 2048 / 68"       (he said he forgets)
```

Matching $p_{\hat x}$ to $p_x$ does **not** by itself give you $E$. Pixel $W_2$ is **not** FID.

### STOP / out of scope

- No PyTorch this hour (Tutorial 12 already coded vanilla/DC/cGAN).
- The algebra “optimum ⇔ joints match with the right marginals” is **homework / papers / exams**.
- Exact Inception tap-layer is unnamed.
- VAEs, diffusion, autoregressive encoder–decoder families start **next class**.
- Image generation SOTA today is **diffusion**; minmax is not retired as an *idea*.

### Load-bearing claims (closed-book)

- $K\ll D$ because of the **manifold hypothesis**; an optimal $K$ can exist.
- **Sampler** = latent → data; **embedding / representation** = data → latent.
- **Inversion:** find $z$ with $G(z)\approx x$. A vanilla GAN is **decoder-only**; PCA is **encoder-only**.
- **BiGAN** and **ALI** are the **same idea**, two concurrent papers.
- Critic on tuples $(x,E(x))$ vs $(G(z),z)$; matching **joints** matches **marginals**.
- After training: $G$ samples, $E$ inverts; $G(E(x))$ should recover $x$.
- **FID** = closed-form $W_2$ of two Gaussians on **Inception** embeddings; **lower is better**.
- Caveats: why Gaussian, why Inception, which layer; Inception Score drops the Gaussian.

**Speaker / course:** Prof. Prathosh · NPTEL IISc · Mathematical Foundations of Generative AI · Lec 19.

---

## 📐 Chalkboard & PyTorch Rosetta Stone

| Symbol / Term | Theoretical Meaning | PyTorch / Software Implementation | Role in GAN Inversion & FID | Dedicated MathsTerm Guide |
| :--- | :--- | :--- | :--- | :--- |
| **$x \in \mathbb{R}^D$** | Ambient Data Vector | `x = batch_images.view(B, -1)` | High-dimensional observable pixel vector ($D=784, 12288$) | [Tensors & Shapes](../../../MathsTerms/Tensors_and_Shapes.md) |
| **$z \in \mathbb{R}^K$** | Low-Dimensional Latent Code | `z = torch.randn(B, K)` | True underlying degrees of freedom on data manifold ($K \ll D$) | [Autoencoders & Latent Spaces](../../../MathsTerms/Autoencoders_and_Latent_Spaces.md) |
| **$G(z)$** | Push-Forward Generator (Decoder) | `x_fake = generator(z)` | Maps low-D latent Gaussian code to high-D synthetic pixels | [Autoregressive Models](../../../MathsTerms/Autoregressive_Models.md) |
| **$E(x)$** | Inversion Encoder Network | `z_hat = encoder(x)` | Maps high-D real images to latent feature representations | [Autoencoders & Latent Spaces](../../../MathsTerms/Autoencoders_and_Latent_Spaces.md) |
| **$q(x, z)$ vs $p(x, z)$** | Joint Empirical vs Model Measures | `(x, E(x))` vs `(G(z), z)` | BiGAN / ALI joint distribution matching pairs | [Joint, Marginal & Conditional Dist](../../../MathsTerms/Joint_Marginal_Conditional_Dist.md) |
| **$D(x, z)$** | Joint Discriminator Network | `score = discriminator(x, z)` | Classifies joint image-latent pairs $(x, z)$ as real or fake | [Jensen-Shannon Divergence](../../../MathsTerms/Jensen_Shannon_Divergence.md) |
| **$\phi(x) \in \mathbb{R}^{2048}$** | Inception-v3 pool3 Embedding | `feats = inception_v3(x)` | Perceptual feature representation capturing semantic realism | [Convolution & Pooling](../../../MathsTerms/Convolution_and_Pooling.md) |
| **$\text{FID}$** | Fréchet Inception Distance | `FrechetInceptionDistance()` | Evaluates generator sample quality & diversity via 2-Wasserstein | [Fréchet Inception Distance](../../../MathsTerms/Frechet_Inception_Distance.md) |

---

## Topic 1: Sampler, manifold, $K\ll D$ (00:01–01:55)

### Where this sits on the master map

This is the **SAMPLER** box. Last lectures already train $G_\theta:z\mapsto\hat x$. He pauses to name **why** the input is tiny compared with pixels, using the [manifold hypothesis](./PREREQUISITES.md#p2-manifold) and the two rooms [ambient vs latent](./PREREQUISITES.md#p1-ambient-latent). Without that, “GAN inversion” has nothing to invert *into*.

### Board / screenshot

No content frame (video HTTP 403; retry still forbidden). Chalk reconstruction from timed captions ~00:01–01:55.

```
  ┌──────────────┐         G          ┌──────────────┐
  │  z  in R^K   │ ───────────────►   │  x̂ in R^D    │
  │  latent RV   │    SAMPLER         │  ambient     │
  └──────────────┘   "projects an     └──────────────┘
                      arbitrary RV
                      onto ambient"
  K much less than D     why?  MANIFOLD HYPOTHESIS
  (student Q, class 1)         data supported in
                               far fewer dims than
                               the pixel grid
  prior work: an OPTIMAL K can exist  (not computed today)
```

Caption: $G$ is a **sampler** that **projects** an arbitrary random variable onto ambient space; $K\ll D$ is the manifold bet, not a UI default.

### What he is establishing

He opens with **GAN inversion**, then admits he should have introduced **latent-variable models** first. The rough intro is enough: a GAN takes an arbitrary random variable and **projects it onto the ambient space**.

Typical picture: input lives in $\mathbb{R}^K$, output in $\mathbb{R}^D$, and **$K$ is much less than $D$**. Someone asked in the first class why $K<D$. The answer is the **manifold hypothesis**: we believe the data can be **supported well in far fewer dimensions** than the ambient grid. That is why he has been drawing generators whose input width is tiny compared with pixels.

He mentions prior work of his group: there can exist an **optimal $K$** for performance — not “as small as possible,” not “as large as the pixels.” He does not compute that $K$ today.

The slogan for this box: **typical samplers start in a lower-dimensional space and project onto ambient**, because the manifold hypothesis is believed true. **That is what a GAN sampler does.**

If you treat $K\approx D$ as the default, you missed the geometric bet the whole family is making. If you treat $K\ll D$ as a coding trick with no geometry, you will not see why an inverse into $\mathbb{R}^K$ is even a sensible compression.

You can now draw $z\in\mathbb{R}^K\to G\to x\in\mathbb{R}^D$ and say *why* $K$ is small. What is still open: many ML jobs want the **other arrow**.

### Analogy for this topic only

A twenty-eight by twenty-eight digit is 784 pixels. Almost every random 784-vector is static, not a 3.

The studio’s lock has $K=10$ dials. The press paints a sheet from those dials.

Someone asks: **why not 784 dials?** Because the useful 3s do not fill the cube. Ten dials are a bet that the 3-sheet is thin.

If you insist on 784 dials “because there are 784 pixels,” you paid for empty space. If you use 1 dial, you cannot tell a 3 from an 8.

In lecture words: lock = $z\in\mathbb{R}^K$, sheet = ambient $\mathbb{R}^D$, press = sampler $G$, thin 3-sheet = manifold hypothesis.

### Local picture

```
  cube of all 784-pixel grids
  .... noise .... noise ....
       [  thin 3-sheet  ]
  .... noise .... noise ....

  10 knobs --G--> a point ON the sheet
  (not a random point in the cube)

  Notice: K≪D is a geometric bet, not a UI default.
          He also says an optimal K can exist.
```

### Bridge

A sampler only answers “give me a new photo.” Retrieval, clustering, and “put spectacles on *this* JPEG” need a vector **for a photo you already hold**. The missing arrow is the next box.

---

## Topic 2: Inverse as representation (01:55–04:46)

### Where this sits on the master map

This is the **INVERSE** box: ambient $\mathbb{R}^D\to$ lower $\mathbb{R}^K$. It names that arrow [embedding / encoder](./PREREQUISITES.md#p3-sampler-encoder) and parks [PCA as the classical encoder-only move](./PREREQUISITES.md#p5-pca). Compression-as-intelligence is the philosophy, not a theorem.

### Board / screenshot

No content frame. Reconstructed ~01:55–04:46.

```
  last box:     z in R^K  --sampler-->  x in R^D

  this box:     x in R^D  --project-->  z in R^K
                PCA = linear way
                name: embedding / representation
                      / feature extraction
```

Caption: two arrows on the board; lower→data labeled **samplers**; data→lower labeled **embedding**.

### What he is establishing

Some jobs demand the **other way around**: take data in ambient dimension and project onto a lower-dimensional space. Classical machine learning already knows a linear tool: **principal component analysis (PCA)** — **dimensionality reduction** from $\mathbb{R}^D$ to $\mathbb{R}^K$.

Later in the course, **variational autoencoders** will do **both**: data down to a subspace **and** back up to ambient. He splits the names now so they do not blur:

- lower-dimensional space → data space = **samplers**
- data space → lower space = **embedding** / **feature extraction**

The offline name of this course is **representation learning**. The **projection of data onto a lower-dimensional space** *is* the **representation** or **embedding**. He will use **embedding space**, **latent space**, and **representation space** interchangeably.

A concrete modern fact: text models are not fed raw ASCII characters. They are fed an **embedding** — start from raw data, project onto a lower-dimensional representation, *then* run the net.

Then a short philosophy: one theory of **intelligence** is **compression**. Humans have produced a humongous pile of data; models compress it into on the order of a **trillion** parameters (he also says a few billion). Trillion is large as a number and **small compared with the data pile**. Compression of information is **one way to look at knowledge**. That is why his lab is named representation learning. It is a worldview, not a proof that $K=10$ is optimal.

The trap is calling the GAN’s $z$ “just noise” and the PCA codes “features” as if they were unrelated. They are the same geometry: lower-dimensional coordinates for high-dimensional files. The wrong move is treating PCA as a sampler of new faces.

You can now name the reverse arrow and the course’s offline title. What is still open: *why* a practitioner would pay for that reverse arrow on images.

### Analogy for this topic only

You have 128 passport photos, each 12,288 pixels.

PCA keeps the 10 fattest directions and writes **10 scores per photo**. A language model does the same idea to words: not the letters, the scores.

Someone asks: **what do we feed the next net — the JPEG bytes or the 10 scores?**

The right move is the scores (the embedding). The wrong move is dumping raw ASCII / raw pixels and calling that “representation learning.”

In lecture words: 10 scores = representation / embedding; JPEG grid = ambient; PCA = linear encoder-only version of this arrow.

### Local picture

```
  ASCII characters / pixels     D huge
           |
           |  project  (PCA today; nets later)
           v
  embedding vector              K small
           |
           |  this is what nets eat
           v
  downstream model

  Notice: sampler is the opposite arrow.
          VAE later owns both; not this hour.
```

### Bridge

We have a name for $x\mapsto z$. We do not yet have a list of jobs that *break* if we only own $z\mapsto x$. Clustering, search, and editing are that list.

---

## Topic 3: Why embeddings: cluster, retrieve, edit (04:46–07:07)

### Where this sits on the master map

This is the **WHY** box. The sampler already handles latent→data. He asks the room why we also need a vector for every image in a space with a [good distance](./PREREQUISITES.md#p1-ambient-latent). Three answers: cluster, retrieve, edit.

### Board / screenshot

No content frame. Reconstructed ~04:46–07:07.

```
  embedding / latent / representation
  (words he will mix)

  why a vector per image?
    1. semantic clustering  (unsupervised categories)
    2. image retrieval      (e-commerce, vector DB)
    3. image editing        (style, spectacles)
                            need embedding of THIS photo
```

Caption: embedding = each data point as a vector in a space **endowed with a good distance**.

### What he is establishing

He restates the interchangeable names: **embedding space**, **latent space**, **representation space**. The sampler we have can project **from latent to data**. What if we need the other way **also**?

He asks for applications. First student-facing answer: **semantic clustering**. You have a large pile of images and want **categories in an unsupervised way**. If each image is a vector with a decent distance, nearby vectors are nearby meanings, and clusters become groups.

Second application: **image-based retrieval / search**. You run an e-commerce site. Someone **uploads a photo**. You want a similar product in the catalog. Convert catalog images **and** the query into vectors, then do a **vector-DB search**. Retrieval is nearest-neighbor in the embedding, not pixel-by-pixel template matching.

Third — the one that will drive the next two topics — **image editing**. Upload your picture; change style; **put spectacles on**. To do that you must first, **given that particular image, find the corresponding embedding**. Without $E(x)$ you cannot start a walk in $z$-space that is *about this person*.

A trap: treating embeddings as “a layer in the GAN” rather than **a coordinate for this file**. Clustering and search never touch $G$ at inference; they only need the vectors. Editing needs both the vectors **and** a decoder.

You can now list three jobs that die if you only own a sampler. What is still open: how walking in $z$ changes the photo.

### Analogy for this topic only

Three studio jobs, same 10-number codes:

- **Pile of 10,000 unlabeled faces** → group them (clustering).
- **Customer uploads a shoe photo** → find the nearest catalog shoe (vector search).
- **This passport photo, add glasses** → you must first *own the codes of this photo*.

Someone asks: **can the press alone do job 3?** No. The press only accepts knobs, not a JPEG.

In lecture words: vector with a good distance = embedding; job 3 is the teaser for traversal + inversion.

### Local picture

```
  catalog photos --> E --> vectors in R^K
  query photo    --> E --> one vector
                         |
                         v
                    nearest neighbor
                    (vector DB)

  editing still needs: E(this photo), then walk, then G

  Notice: clustering/search use distance in z.
          editing also needs the decoder G.
```

### Bridge

Editing is not “find $z$ and stop.” People found that **directions** in $z$ mean something (glasses, pose). That walk is the next box — still assuming we *had* $z_1$ and $z_2$.

---

## Topic 4: Latent traversal (07:07–09:29)

### Where this sits on the master map

This is the **TRAVERSAL** box. Given two embeddings, a **linear mix of codes** is supposed to be a **semantic mix of images**. It sits after WHY (edit) and before INVERSION, because the walk is useless until we can [ingest a photo](./PREREQUISITES.md#p6-decoder-only).

### Board / screenshot

No content frame. Reconstructed ~07:07–09:29.

```
  image X1  <-->  embedding Z1
  image X2  <-->  embedding Z2

  mix:  z(t) = (1-t) Z1 + t Z2     (linear / convex)

  direction Z1 --> Z2
    tracks semantic change X1 --> X2
    e.g. face     -->  face + spectacles

  name: latent space traversal
```

Caption: after GAN training, **some directions in latent space correspond to semantic variations**.

### What he is establishing

To edit, find the embedding of the photo and **traverse** that space **while generating**. Empirically, once a GAN is trained, **particular directions in the latent space correspond to semantic variations**.

Write two pairs: image $X_1$ has embedding $Z_1$; image $X_2$ has embedding $Z_2$. He has **not** yet told you how to get those $Z$s from a GAN. Assume for a moment that you can.

Take a **linear combination** — he says **linear / convex combination** — of the two embeddings. What people have shown: the **direction from $Z_1$ to $Z_2$ tracks the semantic variation** between $X_1$ and $X_2$. Example: a person, and the same person **with spectacles**. Walking $Z_1\to Z_2$ converts the first person into the second. That walk is **latent space traversal**.

So: there exist **semantically meaningful directions** in $z$. Traversing them imposes a style or a semantic change on the decoded image.

For every application in this family you need to know **which vector corresponded to this image**. The GAN constructed so far **does not have that capability**. It **cannot ingest an image as input**. It only takes a vector from the normal (or whatever latent) and maps it to data. It cannot do the other way around.

If you mix **pixels** of two faces and expect glasses to appear smoothly, that is the wrong space. If you think $G$ already accepts JPEGs because you saw pretty samples, you confused sampling with inversion.

The walk itself, once you *have* the two codes, is a **convex / linear combination** in $z$, then decode. This lecture has no training loop; the procedure is:

```python
# Latent traversal (after you already own z1, z2).
# Mix CODES, then paint. Do not mix PIXELS.
# t = 0 → first face; t = 1 → second face (e.g. spectacles).
for t in (0.0, 0.25, 0.5, 0.75, 1.0):
    z = (1 - t) * z1 + t * z2   # convex combo in R^K
    x_t = G(z)                  # semantic walk, not a pixel smear
```

You can now describe traversal with two named pairs $(X_1,Z_1)$, $(X_2,Z_2)$. What is still missing is a definition of **finding $Z$ given $X$**.

### Analogy for this topic only

Two prints on the counter: you without glasses, and you with glasses. Suppose a locksmith already wrote ten knobs for each print.

Walk the knobs in a straight line: $z(t)=(1-t)Z_1+t Z_2$, reprint at each $t$.

Someone asks: **do the reprints grow glasses, or do they smear pixels?**

The claim is they grow glasses (semantic walk). Smearing pixels would be mixing $X_1$ and $X_2$ in ambient space — the wrong corridor.

In lecture words: $Z_1\to Z_2$ = latent traversal; glasses = semantic direction; $G$ still cannot eat $X_1$ by itself.

### Local picture

```
  z-space (K=2 sketched)

  Z1 *---------->* Z2
     you           you + glasses

  G(Z1) = face
  G(Z2) = face with spectacles
  G(0.5 Z1 + 0.5 Z2) ≈ halfway glasses

  Notice: the line is in z, then G paints.
          The GAN so far cannot produce Z1 from the JPEG.
```

### Bridge

We keep saying “given $X_1$, its $Z_1$.” The GAN cannot compute that. The named problem is **inversion**.

---

## Topic 5: Inversion; GAN is decoder-only (09:29–11:17)

### Where this sits on the master map

This is the **INVERSION** box: given $x\sim p_x$, find $z$ such that $G(z)\approx x$. It also hangs the nameplates [decoder-only / encoder-only / encoder–decoder](./PREREQUISITES.md#p6-decoder-only) so PCA and language models sit on the same rack as GANs.

### Board / screenshot

No content frame. Reconstructed ~09:29–11:17.

```
  INVERSION
  given x from p_x
  find z  s.t.  G(z) ≈ x
  "what point in the normal generated this?"

  data --> latent   = encoding
  latent --> data   = decoding

  GAN              = decoder-only
  transformer LMs  = decoder-only (sample, no embeddings)
  PCA              = encoder-only
  encoder-decoder  = both arrows
```

Caption: a sampler should also have an **inverter**; that is when the story is complete.

### What he is establishing

**Inversion** of the sampler: given an image from $p_x$, find the corresponding $z$ such that passing that $z$ through the sampler recovers the $x$ you started from. You have a **fixed** image. You want: **what point in my normal would have generated this image under my GAN?** Those techniques are **inversion techniques**. If you have a sampler, you should also have an **inverter**.

He then maps the arrows onto a vocabulary students have heard from language models:

- data → latent = **encoding**
- latent → data = **decoding**

A GAN is a **decoder-only** model. The transformer-based **autoregressive language models** he names are also **decoder-only**: they **sample**; they **cannot give you embeddings / representations**. An **encoder–decoder** model handles **both** directions. An **encoder-only** model: **PCA** is the example — it projects data to latent and **not** the other way around (no Gaussian sampler of new points as a trained $G$).

The trap is calling ChatGPT-style models “encoders” because they internally have vectors, or calling a GAN an autoencoder because $G$ looks like a decoder in a diagram. Ownership here is the **available arrow at inference**: can I ingest *this* $x$? Vanilla $G$ cannot.

The job as a comment, not a Colab (none on screen):

```python
# INVERSION — given a fixed image x from p_x, find knobs z
# such that G(z) reprints x.  "Which point in the normal
# would have generated this under my GAN?"
#
# Vanilla G cannot do this in one forward pass:
#   G(x) is a type error  (x is R^D, G wants R^K)
#
# Two later routes (only the second is this lecture):
#   (A) optimize z to min ||G(z) - x||     # not trained here
#   (B) learn E so  z_hat = E(x)           # BiGAN / ALI next
# Round-trip test of (B): G(E(x)) should recover x.
```

You can now state inversion in one sentence and place GAN / PCA / LM on the encoder–decoder rack. What is still open: how to **modify the adversarial game** so an inverter exists.

### Analogy for this topic only

The press prints a face from 10 Gaussian knobs. A customer drops a JPEG on the counter: “reprint *this*, then add glasses.”

Someone asks: **which knobs would have printed this JPEG?**

The right move is to search (or learn a map) for those knobs — inversion. The wrong move is to shove the JPEG into $G$’s noise slot (wrong size, wrong meaning) or to run PCA and pretend you sampled a new face.

In lecture words: that search = inversion; $G$ = decoder-only; PCA = encoder-only; both arrows = encoder–decoder.

### Local picture

```
  decoder-only (GAN, AR LM)
     z --> G --> x̂          sample yes, ingest no

  encoder-only (PCA)
     x --> E --> ẑ          ingest yes, sample no

  inversion job
     given x*,  find z* with G(z*) ≈ x*

  Notice: "inverter" is the missing half of the sampler story.
```

### Bridge

We know the job (find $z$ for this $x$). Vanilla two-net adversarial learning never trained an $E$. How do you change the game so inversion is possible? Two papers answered at once.

---

## Topic 6: BiGAN and ALI (11:17–13:43)

### Where this sits on the master map

This is the **PAPERS** box. Feature extraction, manipulation, and editing already motivated inversion. Now the architecture grows from two nets to **three**. Names: **bidirectional GAN (BiGAN)** and **adversarially learned inference (ALI)**.

### Board / screenshot

No content frame. Reconstructed ~11:17–13:43.

```
  vanilla adversarial
    G:  z --> x
    D:  classifier on x   (real vs fake images)

  BiGAN / ALI   (same idea, two papers, same venue, same year)
    G:  z --> x     decoder / sampler
    E:  x --> z     encoder / inverter   (NEW, a neural net)
    D:  critic that builds the bound
    three neural networks
```

Caption: concurrent submissions; research overlap is normal.

### What he is establishing

Question: **how do you modify the adversarial framework so that inversion is possible?** It can be done with a **bidirectional GAN**, also called **BiGAN**. The **exact same approach** appeared in **two papers**, **same conference**, **same year** — **concurrent submissions**. One is **adversarially learned inference (ALI)**; the other is **BiGAN**. Same idea. He tells a researcher meme: you are rarely the only person working on a problem, and sometimes others solve it better. That is the process.

Vanilla adversarial learning: a **sampler** that takes $z$ and projects onto $x$, and a **critic** that is a classifier on the space where $x$ lives.

In BiGAN, **in addition** to sampler and critic, there is another function: the **encoder** or **inverter**, approximated by a **neural network**. **Three** nets:

1. Sampler / **decoder**: projector $z\to x$
2. **Encoder**: takes $x$, emits a latent sample
3. **Critic**: constructs the bound (as in VDM / GAN)

The trap is keeping only two nets and hoping $G$ is invertible — that does not modify the framework. Training $E$ with a pixel reconstruction loss and calling it BiGAN is the wrong move: you skipped the adversarial change (tuples) he is about to write. Names matter: BiGAN and ALI are **not** two different theories in this lecture; they are **one idea with two titles**.

You can now list the three nets and the concurrent-paper anecdote. What is still open: **what $D$ classifies** once $E$ exists.

### Analogy for this topic only

Vanilla studio: a press and a clerk who looks at photos.

Someone asks: **who reads a customer JPEG and writes knobs?** Hire a locksmith ($E$). Now three employees.

Two journals print the same hire-a-locksmith design in the same year. That is BiGAN and ALI.

The wrong move is renaming the clerk “encoder” and claiming you inverted $G$.

In lecture words: locksmith = encoder/inverter; BiGAN = ALI = three nets.

### Local picture

```
  vanilla:   z --> G --> x̂
             x or x̂ --> D --> score

  BiGAN:     z --> G --> x̂
             x --> E --> ẑ
             pairs --> D --> score   (details next)

  Notice: third net is a function X→Z, trained, not a
          numerical solver run after G is frozen.
```

### Bridge

Three nets are not yet an objective. The critic must see **pairs**, because we need $z$ and $x$ to be consistent **together**. That is the tuple game.

---

## Topic 7: Tuple discriminator and joints (13:43–17:09)

### Where this sits on the master map

This is the **JOINTS** box — the load-bearing mechanism of BiGAN/ALI. $D$ classifies [tuples, not images](./PREREQUISITES.md#p4-joint-marginal). Matching joints is stronger than matching photo-marginals, and it is what makes $E$ an inverter rather than a random encoder.

### Board / screenshot

No content frame. Reconstructed ~13:43–17:09.

```
  G:   z ~ Normal  -->  x̂ ~ p_θ
  E:   x ~ p_x     -->  ẑ ~ p_{ẑ}     (dim(ẑ)=dim(z))

  D classifies TUPLES
     fake pair:  ( G_θ(z) , z )
     real pair:  ( x      , E(x) )

  goal:  joint of (x̂, z)  =  joint of (x, ẑ)
         then ALL marginals match

  treat z as extra dimensions of the data you generate

  saddle: min over (θ AND encoder f),  max over D
```

Caption: previously $D$ saw $x$ vs $\hat x$; now it sees $(x,\hat z)$ vs $(\hat x,z)$.

### What he is establishing

He first writes the two pushforwards. Input to the generator is (say) **normal**; output samples $p_\theta$. Input to the encoder is $p_x$; output is some $\hat z$ with law $p_{\hat z}$. **Dimensionality of $\hat z$ must match $z$**. That goes without saying, and it is the first bug if you code this later.

The discriminator is designed to classify **data tuples** of the form $(x, E(x))$ versus $(G_\theta(z), z)$. (Captions once say “$G_\theta$ of $x$ comma $z$”; the pair he means is **generated image and the $z$ that produced it**.)

**Goal:** the **joint** of $(\hat x, z)$ should match the **joint** of $(x,\hat z)$. That is when **both sides** of the story are done. If those joints match, **all the marginals match** — the image law, the latent law, everything.

Mechanism slogan: **treat $z$ as some additional dimensions of the data you are trying to generate.** $D$ no longer classifies samples of $p_x$ vs $p_\theta$. It classifies tuples of the two **pair** distributions.

The **optimization problem is the same shape** (the VDM/GAN saddle), but **minimization is over both $\theta$ and $f$**, where $f$ is the encoder $x\mapsto\hat z$. Classification is between the tuples: $D$ takes $(x,E)$ and $(G_\theta(z),z)$. Previously: $x$ vs $\hat x$. Now: $(x,\hat z)$ vs $(\hat x,z)$.

The trap is matching only $p_{\hat x}$ to $p_x$. Then $E$ can ignore $x$ and emit unrelated noise; the photo-marginal still looks fine while pairs are garbage. That is the wrong move. If you match joints, a fake pair must look like “a real photo **with its own code**.”

Same saddle shape as the vanilla GAN, with two changes he names: **$D$ sees tuples**, and **the min is over both $\theta$ and the encoder $f$**. No live training loop; the packets as comments:

```python
# BiGAN / ALI  — not a new loss family; same saddle, longer vectors.
# G input:  z ~ Normal            ->  x_hat ~ p_theta
# E input:  x ~ p_x               ->  z_hat ~ p_zhat   (dim MUST match z)
#
# D classifies PACKETS, not photos:
fake_pair = (G(z), z)       # print stapled to the knobs that made it
real_pair = (x, E(x))       # album photo stapled to guessed knobs
#
# Goal: joint p(x_hat, z)  matches  joint p(x, z_hat)
# then ALL marginals match (image side AND code side).
# Slogan: treat z as extra dimensions of the data you generate.
#
# min over (theta AND encoder f),  max over D
# previously D saw x vs x_hat; now (x, z_hat) vs (x_hat, z)
```

You can now write the two pair types and the joint-matching goal. What is still open: how you *use* $E$ and $G$ after this saddle is trained, and the algebra he will not do on the board.

### Analogy for this topic only

Clerk no longer stamps “real photo / fake photo.”

Stamp A: a **print** stapled to the **knobs that made it**.  
Stamp B: a **real album photo** stapled to the **knobs the locksmith guessed**.

Someone asks: **when is the studio consistent?** When the clerk cannot tell those *stapled packets* apart — not when the photo piles merely look similar.

Matching only the photo piles is the wrong move: the locksmith can guess random knobs and the photo-marginal still matches.

In lecture words: stapled packet = tuple; matching packets = matching joints; sides of the packet = marginals.

### Local picture

```
  pair cloud FAKE                 pair cloud REAL
  ( G(z),  z )                    ( x,  E(x) )
     x̂-axis, z-axis                 x-axis, ẑ-axis

  D: "which cloud did this pair come from?"

  if clouds coincide:
     p(x̂) = p(x)     image marginals
     p(z) = p(ẑ)     latent marginals
     and the COUPLING matches too

  Notice: z is extra coordinates of "the data".
          min is over sampler AND encoder.
```

### Bridge

Suppose the joints really match. Then $G$ should still sample, and $E$ should invert. He wants a round-trip you can say in one line — and an algebra homework he will put on exams.

---

## Topic 8: Encode then sample; algebra homework (17:09–18:32)

### Where this sits on the master map

This is the **USE** box. After BiGAN training, split the three nets by job: $G$ samples, $E$ inverts. The [round-trip](./PREREQUISITES.md#p3-sampler-encoder) $G(E(x))\approx x$ is the inversion test. The saddle’s **optimum characterization** is left as algebra.

### Board / screenshot

No content frame. Reconstructed ~17:09–18:32.

```
  after training
    G  = sampler
    E  = inversion / embedding extractor

  given image x
    ẑ = E(x)
    x_rec = G(ẑ)     should be the image you started with

  homework / exams / papers
    optimum  <=>  joints match
    those joints have the right marginals
```

Caption: “do the algebra… most of your exams will depend on the ones left as algebra.”

### What he is establishing

Once the bidirectional GAN is trained, the objective was to have **both** the sampler **and** the embedding extractor. Usage is then easy: **treat the generator as sampling** and **the encoder as inversion**. Given an image, pass it through the encoder to get the embedding. Take **that same embedding** and put it through the sampler: it **should give you the image you started with**.

That recovery is a **result you can show**. He will not write the proof. **Do the algebra.** What can be shown: the **optima** of this optimization problem are achieved **if the joint distributions match**, and those joints are such that the **marginals correspond to the individual distributions**. Please do it. **Read the papers.** He assumes you do the left-as-algebra items when he writes question papers.

The trap is skipping the homework and only memorizing “$E$ inverts.” Using $E$ at inference without the joint-training story is the wrong move: you are guessing a code, not running BiGAN.

Inference after the three nets are trained (still not a training loop):

```python
# USE the split:  G samples,  E inverts.
z_hat = E(x)          # embedding of THIS image
x_rec = G(z_hat)      # should be the image you started with

# Sampling is unchanged:
x_new = G(z_fresh)    # z_fresh ~ Normal, no x in the input

# Homework / exams / papers (he skips writing it):
# saddle optimum  <=>  the two joints match
#                   and those joints have the right marginals
```

You can now run the two-line inference recipe and you know the exam threat. What is still open: how to **score** a pile of samples when the model has no likelihood.

### Analogy for this topic only

Customer JPEG in. Locksmith writes 10 knobs. Press reprints from those knobs.

Someone asks: **is it the same face?** At a good BiGAN, yes — that is the round-trip test.

The wrong move is to declare victory because $G$ still prints *some* face from *some* noise, unrelated to this JPEG.

In lecture words: $E$ then $G$ = inversion use; joint match at opt = the algebra he will grade.

### Local picture

```
  x  --> E --> ẑ --> G --> x_rec ≈ x

  sampling (unchanged):  z ~ N --> G --> new x̂

  algebra left on purpose
    saddle opt  =>  p(x̂,z) = p(x,ẑ)
                 =>  correct marginals

  Notice: he will skip writing the proof on the board.
          Papers + exams own that step.
```

### Bridge

You built a generative model. **How good are the generated samples?** No $p_\theta(x)$ to evaluate. That is the FID box — the longest stretch of the hour.

---

## Topic 9: FID = $W_2$ of Inception Gaussians (18:32–24:46)

### Where this sits on the master map

This is the **FID** box: a **sample estimate** of how close the generated cloud is to the real cloud when you have no closed-form laws. It reuses [Wasserstein-2 of Gaussians](./PREREQUISITES.md#p7-w2-gaussians) after a [frozen Inception embedding](./PREREQUISITES.md#p8-inception). Lower is better. This is **evaluation**, not the WGAN training objective.

### Board / screenshot

No content frame. Reconstructed ~18:32–24:46.

```
  n real images     n generated images     (post-training)

  NOT W2 on pixels
  pass both through pretrained Inception (ImageNet)
  tap embeddings at some layer   (he does not remember which;
                                  "68th or something")
  assume those vectors are Gaussian
  closed-form W2 of the two Gaussians  =  FID
  lower FID  =>  better perceived quality
```

Caption: FID **is** Wasserstein-2 between true and generated **with a twist** — the twist is the Inception space plus the Gaussian closed form.

### What he is establishing

After you have constructed a generative model, how do you know **how good the generated samples are**? Multiple metrics exist. A widely used one: **FID**, the **Fréchet Inception Distance**. (ASR says “fret / fresh / fresher”; the name is **Fréchet**.)

Setup: you are given **$n$ samples from the true data** and **$n$ samples from the generated distribution**, **post-training**. You need a **sample** estimate of closeness. You do **not** have distributional-level closed forms of $p_x$ and $p_\theta$.

High-level idea: **reuse the same family of metrics you used for optimization**, now as a **score on samples**. In particular you can compute the **Wasserstein-2** metric between the post-training generated samples and the true samples. (ASR “versus stains / science” is **Wasserstein**.) **That is what FID is doing** — $W_2$ between actual and generated — **but with a twist**.

The twist: it is **not** $W_2$ on the **images** themselves. First take true and generated images and **extract embeddings on a pretrained neural net** trained for some other task. Here: the **Inception** network, a large architecture trained on **ImageNet**. (Captions say “unsupervised way on the ImageNet task”; ImageNet Inception is the usual pretrained classifier — take the **ImageNet + freeze** idea, not a new unsupervised theorem.) Pass **both** clouds through that frozen net and **extract embeddings from a particular layer**. People have found that **perceptually** some layer corresponds to human vision. He **does not remember the exact index** — “some 68th layer or something.” **Do not invent a layer number** and attribute it to this board.

Now every image is a vector whose dimension equals that layer’s width. They **assume** the distribution in that Inception latent space is **Gaussian**. Is that fair? If the net used **batch normalization** and similar, and if you have **enough samples**, you can wave at the **central limit theorem** and call it fairly Gaussian. **Procedure-wise, this is what is done** even if the assumption is crude.

You now have samples from **two high-dimensional Gaussians**. $W_2$ between two Gaussians has a **known algebraic / deterministic form**: means of both, **trace** involving **covariances** of true and generated, minus a **matrix square-root** term. That number **is FID**. You have everything you need because you have the vectors: empirical mean and covariance of each cloud.

Recipe he restates at the end of the stretch:

1. Pretrained Inception on ImageNet (frozen).
2. Pass true and generated images through it.
3. Tap embeddings at some hidden layer.
4. Assume both clouds Gaussian; compute mean and covariance of each.
5. Plug into the closed-form $W_2$. That is FID.
6. It is called **Inception** distance because you tap Inception.
7. **The lower the FID, the better** the perceived quality of generated data, because it tracks $W_2$.

The closed form he points at as “given by this” is the standard $W_2$ between Gaussians (Fréchet distance of Gaussians):

$$
\mathrm{FID}=\|\mu_r-\mu_g\|_2^2+\mathrm{Tr}\bigl(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\bigr).
$$

In words: squared distance of the two means, plus a trace that compares the two covariances via a matrix square root.

If you compute $W_2$ on pixels and call it FID, you dropped the twist. If you think FID is a likelihood, you dropped the whole point of a **sample** score. If you train Inception on your GAN samples, you cheated the “pretrained / other task” freeze.

This lecture has **no training code**. The procedure as a commented checklist (not a library you must run) is:

```python
# FID = evaluation AFTER training. Not a training loss today.
# He: n true images and n generated images (equal n), post-training.
# Reuse W2 (the family used for WGAN training) as a SAMPLE score.

# 1. Frozen Inception, pretrained on ImageNet (other task).
#    He does NOT name the tap layer ("68th or something" = he forgets).
#    Do not invent 2048 / pool3 and pin it on this board.
phi = inception.tap_some_layer

# 2. Embed BOTH clouds. Twist: NOT Wasserstein-2 on pixels.
real_vec = [phi(x)     for x in real_images]   # album
fake_vec = [phi(G(z))  for z in noises]        # prints

# 3. Assume Gaussian (BN + "enough samples" / CLT story).
#    Crude, but procedure-wise this is what is done.
mu_r, Sigma_r = mean_and_cov(real_vec)
mu_g, Sigma_g = mean_and_cov(fake_vec)

# 4. Closed-form W2 of two high-dim Gaussians = FID.
#    ||mu_r - mu_g||^2  +  Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^{1/2})
#    Lower FID => better perceived quality (it tracks W2).
fid = w2_gaussians(mu_r, Sigma_r, mu_g, Sigma_g)
```

You can now run the six-step recipe and say “lower is better.” What is still open: why this recipe is **flawed**, and where the adversarial *idea* goes after GANs stop being image SOTA.

### Analogy for this topic only

Two piles on the counter: 128 real passport photos, 128 prints.

Do **not** compare them by counting matching pixels.

Send every sheet through a **museum guide who already walked ImageNet** (Inception). The guide writes a note-vector per sheet. Pretend the real notes form an oval blot and the print notes form another oval blot. The **calculator distance between those two ovals** is FID.

Someone asks: **is a smaller number better?** Yes. A huge number means the print blot is far from the album blot.

The wrong move is measuring blots on the raw pixel grid, or claiming you know he said “layer 68” as a fact (he said he **forgets**).

In lecture words: museum guide = Inception; oval blot = Gaussian on embeddings; calculator distance = closed-form $W_2$ = FID.

### Local picture

```
  real x_1..x_n          fake G(z_1)..G(z_n)
         \                    /
          \   frozen Inception
           v                  v
        φ(x) cloud         φ(G(z)) cloud
           \                  /
            assume Gaussian
            μ_r,Σ_r     μ_g,Σ_g
                  \
                   W2 closed form
                   = FID   (lower better)

  Notice: evaluation, not a training loss this hour.
          Layer index is unknown on this board.
```

### Bridge

The recipe has three loaded assumptions (Gaussian, Inception, which layer). He will name those flaws, then close the whole **$f$-divergence / adversarial** block and point at **VAE / diffusion**.

---

## Topic 10: FID caveats, agents, next LVMs (24:46–28:12)

### Where this sits on the master map

This is the **STOP / CARRY** box. FID is a handy score, not a sacred one. The **minmax idea** outlives image-GAN SOTA (agents, PEFT, even op-amp feedback). The next family is [latent-variable models with **both** arrows baked in](./PREREQUISITES.md#p6-decoder-only): VAE, diffusion, autoregressive.

### Board / screenshot

No content frame. Reconstructed ~24:46–28:12.

```
  FID flaws
    why Gaussians?
    why Inception?
    which layer?

  other scores (e.g. Inception Score)
    drop the Gaussian assumption
  broad recipe remains:
    embed  ->  assume a law  ->  compute a distance

  end of f-divergence min / adversarial learning
  image SOTA now: diffusion  (minmax not SOTA there)
  idea still used: adversarial agents, PEFT, ...
  next class: LVMs with encoder AND decoder
              VAE, diffusion, autoregressive, ...
```

Caption: critique + sampler is a recurring idea; electrical-engineering bias (negative feedback, Kalman / state-space later).

### What he is establishing

Lower FID tracks better perceived quality **because it corresponds to $W_2$**. Of course there are **flaws**: **why should they be Gaussians**, **why Inception**, **what layer to tap**. That is why people invented **other metrics**, e.g. **Inception Score**, where they **give up the Gaussianity assumption**. The **broad idea** remains: take some embedding, assume a distribution, compute a distance.

That brings him to the **end of $f$-divergence minimization / adversarial learning**. Adversarial **minmax is not the state of the art today for image generation** — **diffusion models** are. The **adversarial idea** is still used in many places.

Example he wants you to hear even if it is ahead of the course: **autonomous agents** with multiple LLMs. Typical pattern: **adversarial / self-correcting agents** — one LLM **proposes a hypothesis**, another **negates / critiques**. You solve an optimization on **prompt space**, or you do **parameter-efficient fine-tuning (PEFT)** — train a **small set of parameters** on those agents so performance improves. He will return to PEFT later.

The whole idea — a **critique** network and a **sampler**, with an adversarial optimization between them — **carries to multiple SOTA models**. Life slogan: to generate good data you should always have a good critique. Control systems: a system stabilizes with **negative feedback**. Electrical engineers know **operational amplifiers** with negative feedback; computer-science undergrads may not. Later the course will touch **state-space models**, with roots in **linear dynamical systems** and **Kalman filters** from control. He admits a slight EE bias because those ideas cross-pollinate. Point: a **critique network is a handy idea** with many applications.

**That’s it for adversarial learning.** Next: another class of models, **latent-variable models**, which **by construction** have **both encoders and decoders baked in**. Multiple models in that family: **variational autoencoder**, **diffusion**, some **autoregressive** models, and so on. End of class.

The trap is walking out thinking “GANs are dead, delete the saddle.” Minmax is not image SOTA; the critique idea still carries. Treating FID as the last word is the wrong move — he already listed the flaws. You can now close the adversarial block and name the next family. What is still missing is a model that is *born* with both arrows: that is the VAE hour, not a sixth net on this GAN.

### Analogy for this topic only

The museum score (FID) used an oval-blot assumption and one particular guide (Inception). Other scores fire the oval assumption.

Image printing’s current champion is **diffusion**, not the clerk-vs-press fight. The *fight* still shows up: one intern proposes a plan, another intern tears it apart (adversarial agents); you only tweak a few knobs (PEFT).

Someone asks: **what is the next machine in this course?** A machine that is **born with both a locksmith and a press** (VAE), then diffusion, then autoregressive samplers.

The wrong move is to treat this hour as the end of generative modeling.

In lecture words: flaws of FID = Gaussian / Inception / layer; next family = LVMs with both arrows.

### Local picture

```
  adversarial block CLOSES
    f-div / VDM / GAN / WGAN / BiGAN / FID

  idea CARRIES
    proposer  <-->  critic
    agents, PEFT, negative feedback

  image-gen SOTA
    diffusion   (not minmax)

  NEXT family: latent variable models
    encoder AND decoder baked in
      VAE  |  diffusion  |  autoregressive ...

  Notice: inversion was the missing arrow of GANs.
          Next models refuse to omit that arrow.
```

### Bridge

The leftover problem is **likelihood with a hidden $z$**: $p(x)=\int p(x\mid z)p(z)\,dz$ is incomplete if $z$ is unknown. That is the VAE hour (Lec 20), not a sixth net on this GAN.

---

## External references

All companions live **here**, not under the topics. Mix of **video**, **blog/notes**, and **original papers**. No Wikipedia.

**Start here (if you only open three).** Instructor Drive PDF → BiGAN + ALI papers → Heusel FID.

This video is **~28 min of chalk**, not a 60-minute coding lab. The extra links are second passes of the **same ten boxes**.

### Per-topic companions (2–3 each)

Use **after** the matching topic. Do not open thirty tabs.

| Topic / map box | Type | Resource | Why it helps |
|-----------------|------|----------|--------------|
| **1 · sampler / manifold** | blog | [Olah — Visualizing MNIST](https://colah.github.io/posts/2014-10-Visualizing-MNIST/) | $D=784$ cube; MNIST as a thin manifold — the $K\ll D$ picture. |
| **1 · sampler / manifold** | blog | [Olah — NNs, manifolds, topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) | Why a net can crawl a sheet inside a huge ambient space. |
| **1 · sampler / manifold** | notes | [Stanford CS236 GAN notes](https://deepgenerativemodels.github.io/notes/gan/) | Implicit sampler $z\mapsto x$; decoder-only GAN. |
| **2 · representation / PCA** | video | [StatQuest — PCA step-by-step](https://www.youtube.com/watch?v=FgakZw6K1QQ) | Linear encoder-only squash he names as the classical reverse arrow. |
| **2 · representation / PCA** | blog | [Olah — Visualizing representations](https://colah.github.io/posts/2015-01-Visualizing-Representations/) | Embeddings as the thing nets eat, not raw pixels/ASCII. |
| **3 · cluster / retrieve / edit** | video | [Google MLCC — embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) | Vector with a good distance; nearest-neighbor retrieval. |
| **3 · cluster / retrieve / edit** | code | [FAISS (Facebook)](https://github.com/facebookresearch/faiss) | The vector-DB search he names for e-commerce photo query. |
| **4 · latent traversal** | video | [InterFaceGAN demo](https://www.youtube.com/watch?v=uoftpl3Bj6w) | Walking a face-GAN latent direction (glasses / pose) — his spectacles example at scale. |
| **4 · latent traversal** | notes | [InterFaceGAN project](https://genforce.github.io/interfacegan/) | Linear directions in $z$ ↔ semantic edits; needs an inverted code first. |
| **4 · latent traversal** | paper | [Radford et al. DCGAN (arXiv:1511.06434)](https://arxiv.org/abs/1511.06434) | Early latent interpolations: mix $z$, then decode — not mix pixels. |
| **5 · inversion / decoder-only** | paper | [Xia et al. — GAN inversion survey (arXiv:2101.05278)](https://arxiv.org/abs/2101.05278) | Named job: find $z$ with $G(z)\approx x$; later encoder vs optimization routes. |
| **5 · inversion / decoder-only** | video | [Zhou — inverting GANs for real-image editing (MIT)](https://www.youtube.com/watch?v=zyBQ9obuqfQ) | University seminar: invert a real photo, then edit. |
| **5 · inversion / decoder-only** | notes | [Instructor Drive PDF](https://drive.google.com/file/d/1Kebb00VehPBMyleuw2ubp2qbvrBqyvZZ/view) | Board companion for this NPTEL hour. |
| **6 · BiGAN / ALI** | paper | [Donahue et al. BiGAN (arXiv:1605.09782)](https://arxiv.org/abs/1605.09782) | Original three-net inverse mapping; joints of pairs. |
| **6 · BiGAN / ALI** | paper | [Dumoulin et al. ALI (arXiv:1606.00704)](https://arxiv.org/abs/1606.00704) | Concurrent “same idea, same year, same venue.” |
| **6 · BiGAN / ALI** | notes | [ALI project page](https://ishmaelbelghazi.github.io/ALI/) | Paper + code pointer for adversarially learned inference. |
| **7 · tuple $D$ / joints** | paper | BiGAN paper (Topic 6) | $D$ on $(x,E(x))$ vs $(G(z),z)$; joint match ⇒ marginals. |
| **7 · tuple $D$ / joints** | demo | [Seeing Theory — compound / pairs](https://seeing-theory.brown.edu/compound-probability/index.html) | Joint vs a side (marginal) with a picture, not a slogan. |
| **7 · tuple $D$ / joints** | notes | CS236 GAN notes (Topic 1) | Adversarial saddle shape before the tuple upgrade. |
| **8 · $E$ then $G$** | code | [jeffdonahue/bigan](https://github.com/jeffdonahue/bigan) | Official BiGAN training/eval code — round-trip lives here, not in this chalk hour. |
| **8 · $E$ then $G$** | code | [IshmaelBelghazi/ALI](https://github.com/IshmaelBelghazi/ALI) | Official ALI repo; encoder as inference net. |
| **8 · $E$ then $G$** | paper | ALI paper (Topic 6) | Algebra he leaves as homework is in the papers. |
| **9 · FID recipe** | paper | [Heusel et al. FID (arXiv:1706.08500)](https://arxiv.org/abs/1706.08500) | Fréchet Inception Distance; $W_2$ of Gaussians on Inception features; lower better. |
| **9 · FID recipe** | slides | [Stanford CS236 — eval / FID](https://deepgenerativemodels.github.io/assets/slides/lecture15.pdf) | Same claim in a Stanford lecture: Fréchet on pretrained-classifier features. |
| **9 · FID recipe** | code | [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid) | Runnable FID; default tap is a later convention — he did **not** name the layer. |
| **10 · caveats / next** | paper | [Salimans et al. (arXiv:1606.03498)](https://arxiv.org/abs/1606.03498) | Inception Score — the alternate metric he says drops the Gaussian assumption. |
| **10 · caveats / next** | video | [MIT 6.S191 2026 Lec 4 — deep generative models](https://www.youtube.com/watch?v=R8V8CbuxryI) | GAN sampler next to VAE/diffusion — the fork at 27:34. |
| **10 · caveats / next** | video | [Stanford CS231n 2025 Lec 14](https://www.youtube.com/watch?v=Edr4uZFh4EE) | Latest large-course GAN hour, then why the course moves to diffusion. |

**How to use.** After Topic 1, Olah MNIST. After Topic 4, InterFaceGAN demo (glasses walk). After Topics 6–8, BiGAN **and** ALI (he insists they are concurrent). After Topic 9, Heusel + CS236 slides; pytorch-fid only if you want to *run* a score — this lecture has no Colab. After Topic 10, MIT 6.S191 or CS231n 2025 for the VAE/diffusion handoff.

---

## Sources

- Video: [Lec 19 Inversion with GANs and FID](https://www.youtube.com/watch?v=zw2DUzD0TLE) · playlist [MFGAI](https://www.youtube.com/playlist?list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK) (YouTube slot `&index=2`; learning order 20)
- Captions: `raw/captions.en.vtt` / `raw/captions.en.timed.txt` (ASR cleaned: Wasserstein, Fréchet)
- Claim sheets: `raw/claims/topic-01.md` … `topic-10.md`
- Course notes linked in the description: [Drive](https://drive.google.com/file/d/1Kebb00VehPBMyleuw2ubp2qbvrBqyvZZ/view)
- Ingest: captions yes · video file no (HTTP 403) · frames no · `ingest_evidence: E2`
