# Prerequisites — warm-up before Tutorial 12 (Vanilla GAN, DCGAN, cGAN)

> **Do this first** if “logits,” “detach,” or “FID” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Complements [Lec 04 VDM](../27-Lec04-Variational-Divergence-Minimization/NOTES.md) and [Lec 05 GANs](../28-Lec05-Generative-Adversarial-Networks/) (theory).  
> **Beginner:** purpose · definition · micro · analogy · ASCII · notice · mini-check.

This hour is **code**, not a new theorem. Two nets share one score. You will write PyTorch that **maximizes** one block of weights and **minimizes** the other — using an optimizer that only knows how to **descend**.

```
  After this warm-up you can say:

  "G turns noise into an image; D is a real-vs-generated classifier."
  "At a good GAN, D’s probability sits near 1/2 on both piles."
  "Logits are pre-sigmoid scores; BCEWithLogits folds σ in."
  "detach() cuts the tape so D-step cannot train G."
  "tanh images live in [−1,1]; MNIST tensors start in [0,1]."
  "A label embedding is a learned row of numbers, not the digit itself."
  "Transpose conv grows the grid; conv shrinks it."
  "FID is lower-is-better on Inception embeddings."
```

```
  §1  Two nets G and D              ──► Topics 1, 6
  §2  Saddle / two steps per batch  ──► Topics 1, 4–5
  §3  Logits and BCEWithLogits      ──► Topics 3–5
  §4  Computational graph / detach  ──► Topic 4
  §5  tanh range vs [0,1] pixels    ──► Topics 2, 6
  §6  Label embeddings              ──► Topics 7, 9
  §7  Conv vs transpose conv        ──► Topics 8–9
  §8  FID (lower better)            ──► Topics 2, 10
```

**One scene through all eight.** Keep the print shop. Every idea is a job you can do *with two piles of 28×28 digits*, not with a PDF of $p_x$.

```
  BAG A  album     128 MNIST digits, mapped to [−1,1]     = real x
  BAG B  prints    G(z), z is 100 Gaussian numbers        = x̂

  §1  G is the press; D is the clerk (0.5 = shrug)
  §2  each batch: teach clerk, then teach press (saddle)
  §3  clerk turns in a raw score (logit); loss folds σ in
  §4  photocopy prints before the clerk’s lesson (detach)
  §5  tanh press speaks [−1,1]; album started [0,1]
  §6  “print a 7” needs a learned name-tag, not a lonely integer
  §7  conv shreds the photo; transpose-conv blows crumbs up
  §8  FID is a third museum; lower score wins

  This hour is not live 25-epoch training. The Colab already ran.
```

---

## 1. Two nets: sampler and clerk

<a id="p1-two-nets"></a>

**Purpose.** The lecture never trains “a GAN.” It trains **G_θ** and **D_w**.

**Definitions.** **Generator** $G_\theta$: $z\sim\mathcal{N}(0,I)$ in, image $\hat x$ out. **Discriminator** $D_w$: image in, **one number** out — “how real.” After sigmoid that number is in $(0,1)$.

**Micro.** $z$ is length 100. MNIST is $28\times 28=784$. G must expand 100 numbers into 784.

**Analogy.** Noise bag → printing press (G). Clerk (D) sees a photo from the album **or** a print. Success is when the clerk shrugs: **0.5**.

```
  z ~ N(0,I) --> G_θ --> x̂
  album x  or  x̂  --> D_w --> logit --> σ --> P(real)
```

**Notice.** Math does not care if G is an MLP or a conv net. That is today’s fork.

**Mini-check.** Where does chance live — G or $z$? What does 0.5 mean?

---

## 2. A saddle: two updates, opposite jobs

<a id="p2-saddle"></a>

**Purpose.** One batch: **raise D’s score on reals / lower it on prints**, then **fool D**.

**Definitions.** $\max_w$ of a two-term score, then $\min_\theta$ of (part of) the same score. That is last lecture’s **saddle**. PyTorch `optimizer.step()` only **descends**, so the D-loss is written so **descending it** is the same as **ascending** the theory score.

**Micro.** Each batch: one D step, one G step (he also assigns: try $k$ G steps per D).

**Analogy.** Clerk studies fakes, then forger studies the clerk. They share one grade book $J$.

```
  batch:  zero_grad D → loss_D.backward → step D
          zero_grad G → loss_G.backward → step G
```

**Notice.** The first term of the GAN score **does not depend on θ**. G only sees generated images.

**Mini-check.** Which optimizer updates on the D step — G, D, or both?

---

## 3. Logits and BCE with logits

<a id="p3-logits"></a>

**Purpose.** D ends in **one raw number** (a **logit** $A$). Do **not** put `Sigmoid` on D if the loss is `BCEWithLogitsLoss`.

**Definitions.** $\sigma(A)=1/(1+e^{-A})$. Binary cross-entropy with logits:

$$
y\log\sigma(A)+(1-y)\log(1-\sigma(A)).
$$

$y=1$ (real) keeps the first term; $y=0$ (generated) keeps the second. PyTorch **negates** this into a **minimization**.

**Micro.** $A=2$, $y=1$: $\sigma(2)\approx 0.88$ (clerk says “probably real”). $A=-2$, $y=0$: $\sigma(-2)\approx 0.12$ (clerk says “probably generated”). Both are *good* clerk answers. $A=-2$, $y=1$ is a *bad* G-step: the press wanted the clerk to say real and failed.

**Analogy.** Logit = raw exam score. Sigmoid = percentage. The loss already converts score→percentage. Grading twice (sigmoid then BCE) double-counts.

```
  D  -->  A (logit)  -->  BCEWithLogits  -->  number to descend
                 (σ lives INSIDE the loss)
```

**Notice.** G-step will **lie about y** (set ones on prints) on purpose. That is Topic 5.

**Mini-check.** Why no `nn.Sigmoid` on D?

---

## 4. The tape and detach

<a id="p4-detach"></a>

**Purpose.** Fake images $G(z)$ still remember G’s weights. On the **D step** you must **cut** that memory.

**Definitions.** Autograd builds a **tape**. `.backward()` walks the tape. **`.detach()`** returns a tensor that looks the same but has **no tape** into G.

**Micro.** `fake = G(z); D(fake)` would train G if you backward D-loss. `D(fake.detach())` trains **only D**.

**Analogy.** The print still has the press’s serial number. For the clerk’s lesson you photocopy the print (detach) so the press is not fined.

```
  z --> G --> x̂ --detach--> D --> D-loss   (only w)
  z --> G --> x̂ ----------> D --> G-loss   (θ; w frozen by not stepping D)
```

**Notice.** G-step does **not** detach — G needs the tape through D (with D’s weights **not** stepped).

**Mini-check.** On which step do you detach?

---

## 5. tanh range vs [0,1] pixels

<a id="p5-tanh"></a>

**Purpose.** G’s last layer is **tanh** ∈ [−1,1]. MNIST after `ToTensor` is [0,1].

**Definitions.** Map reals with `Normalize((0.5,),(0.5,))`: $(x-0.5)/0.5$. At plot time undo: $(x+1)/2$.

**Micro.** Pixel 0 → −1; pixel 1 → +1. Pixel 0.5 stays 0.

**Analogy.** Two thermometers: album in Celsius [0,1], press in Fahrenheit-like [−1,1]. Convert before you compare.

```
  ToTensor:     0..255  →  0..1
  Normalize:    0..1    →  −1..1     (match tanh G)
  show:         −1..1   →  0..1      ((x+1)/2)
```

**Notice.** He misspeaks “std 0.1” once; the **arithmetic he walks** is 0.5 / 0.5. Three-channel nets: ImageNet stats.

**Mini-check.** What does pixel 0 become after Normalize(0.5, 0.5)?

---

## 6. A label is a learned row, not a digit

<a id="p6-embed"></a>

**Purpose.** Conditional GAN feeds **Y** (digit 0–9). A single integer next to 100 noise numbers is too weak.

**Definitions.** `nn.Embedding(10, 10)` is a **10×10 matrix**. Row $k$ is the vector for digit $k$. **Learned**. Concatenate with $z$ (G) or with flat $x$ (D). G and D may have **different** matrices.

**Micro.** $z$ is 100, embed is 10 → G input **110**.

**Analogy.** “7” is a name tag. The net learns a 10-number **accent** for each tag, then tapes it to the noise.

```
  y=2  -->  Embedding  -->  row 2 (10 numbers)
  z (100) ++ row  -->  110  -->  G
```

**Notice.** On DCGAN’s **D**, he later makes the embed a **28×28 map** and stacks it as a **second channel**, not a 10-vector.

**Mini-check.** Who updates G’s embedding — the D-step or the G-step?

---

## 7. Conv shrinks; transpose conv grows

<a id="p7-convt"></a>

**Purpose.** DCGAN: D is a CNN (images → small maps). G must go **the other way** (tiny map → 28×28): **ConvTranspose2d**.

**Definitions.** `Conv2d` with stride 2: 28→14→7. `ConvTranspose2d` with stride 2: 7→14→28. BatchNorm between.

**Micro.** His G: Linear → $128\times 7\times 7$ cube, then two transpose blocks to $1\times 28\times 28$.

**Analogy.** D is a paper shredder (big photo → crumbs). G is a **blow-up copier** (crumb grid → photo).

```
  G:  100  →  128×7×7  →  64×14×14  →  1×28×28
  D:  1×28×28  →  64×14×14  →  128×7×7  →  logit
```

**Notice.** Same **loss/train loop** as vanilla. Only the **body** of G and D changes.

**Mini-check.** Which layer grows spatial size?

---

## 8. FID: inspector who never uses D

<a id="p8-fid"></a>

**Purpose.** Pretty grids lie. **FID** (Fréchet Inception Distance) compares **Inception embeddings** of real vs generated. **Lower is better.**

**Definitions.** Pass images through Inception, take a mid-layer (he uses **2048** features), fit two Gaussians, measure a distance. Needs RGB: **repeat** MNIST’s 1 channel to 3. Clamp to [0,1].

**Micro.** He reports vanilla **92.93**, cGAN-MLP **104** (looks nicer, **worse FID**), a DCGAN run **21.5**.

**Analogy.** The clerk D can be fooled. FID is a **third museum** that scores oil-paint statistics, not the clerk’s stamp.

```
  images → clamp 0..1 → repeat 3 ch → Inception 2048 → FID
  lower FID  =  closer embedding clouds
```

**Notice.** ~5000 generated images. Notebook in the video description.

**Mini-check.** If samples look sharper but FID rose, who wins?

---

Ready → [NOTES.md](./NOTES.md). Quiz: [quiz.html](./quiz.html) Part A.
