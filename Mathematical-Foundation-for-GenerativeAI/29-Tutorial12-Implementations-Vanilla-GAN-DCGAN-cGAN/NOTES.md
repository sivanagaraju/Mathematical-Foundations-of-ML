# Tutorial 12 — Implementations of Vanilla GAN, DCGAN and Conditional GAN

> **Video:** [Tutorial 12 : Implementations of Vanilla GAN, DCGAN and Conditional GAN](https://www.youtube.com/watch?v=dBcURX7GrwE) · **~78 min**  
> **Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md) · **Quiz:** [quiz.html](./quiz.html)

**Do PREREQUISITES before this article.**  
**Previous math:** [Lec 04 VDM](../27-Lec04-Variational-Divergence-Minimization/NOTES.md)  
**Course:** Mathematical Foundations of **Generative AI** · NPTEL / IISc  
**Colab (description):** [notebook](https://colab.research.google.com/drive/15nVkKu1mySDHzEj4NaqDjDAUUvZsTxzR?usp=sharing)

| When the lecture hits… | Warm-up |
|------------------------|---------|
| G vs D | [p1-two-nets](./PREREQUISITES.md#p1-two-nets) |
| Two steps / saddle | [p2-saddle](./PREREQUISITES.md#p2-saddle) |
| BCEWithLogits | [p3-logits](./PREREQUISITES.md#p3-logits) |
| `.detach()` | [p4-detach](./PREREQUISITES.md#p4-detach) |
| tanh vs [0,1] | [p5-tanh](./PREREQUISITES.md#p5-tanh) |
| Embeddings | [p6-embed](./PREREQUISITES.md#p6-embed) |
| ConvTranspose | [p7-convt](./PREREQUISITES.md#p7-convt) |
| FID | [p8-fid](./PREREQUISITES.md#p8-fid) |

**Boards:** video file 403 from YouTube this session — **no screenshot files**. ASCII reconstructs the Colab he walked. `ingest_evidence: E2`.

---

## Table of Contents

1. [Topic 1 — VDM saddle and today’s three nets](#topic-1-vdm-saddle-and-todays-three-nets-0003–0600) (00:03–06:00)
2. [Topic 2 — FID, Colab, MNIST into tanh range](#topic-2-fid-colab-mnist-into-tanh-range-0600–1230) (06:00–12:30)
3. [Topic 3 — MLP generator and discriminator](#topic-3-mlp-generator-and-discriminator-1230–1945) (12:30–19:45)
4. [Topic 4 — Discriminator step, detach, sign flip](#topic-4-discriminator-step-detach-sign-flip-1945–3135) (19:45–31:35)
5. [Topic 5 — Non-saturating generator loss](#topic-5-non-saturating-generator-loss-3135–4627) (31:35–46:27)
6. [Topic 6 — Sampling vanilla; starting cGAN](#topic-6-sampling-vanilla-starting-cgan-4627–5109) (46:27–51:09)
7. [Topic 7 — Conditional MLP: embeddings](#topic-7-conditional-mlp-embeddings-5109–6241) (51:09–62:41)
8. [Topic 8 — DCGAN conv and transpose](#topic-8-dcgan-conv-and-transpose-6241–6724) (62:41–67:24)
9. [Topic 9 — Conditional DCGAN](#topic-9-conditional-dcgan-6724–7233) (67:24–72:33)
10. [Topic 10 — FID numbers, noise probe, notebook](#topic-10-fid-numbers-noise-probe-notebook-7233–7833) (72:33–78:33)
11. [External references](#external-references)
12. [Apply it (scenarios)](#apply-it-scenarios)
13. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Make $G_\theta$ draw MNIST-like digits without a density formula. Implement last lecture’s **saddle** in PyTorch: one D-step, one G-step, each batch. Do it four times — MLP, MLP+$Y$, conv, conv+$Y$ — **25 epochs**, notebook already run. Score with **FID** (lower better); do not ship on a pretty 32-image grid.

**Worldview arc:** from “GAN is VDM on two nets” **to** “four Colab variants + FID + a same-noise / different-label probe.”

### Method card

```
  1. HOLD two clouds     D ~ p_x (MNIST, mapped to [−1,1])
                         x̂ = G(z), z ~ N(0,I) in R^100
  2. D = binary clerk    logit only; NO sigmoid on the module
  3. D-STEP              BCE(D(real), 1) + BCE(D(G(z).detach()), 0)
                         minus inside BCE turns max_w into descent
  4. G-STEP              do NOT min log(1−D(G)); min −log D(G)
                         code trick: BCE(D(G(z)), ones)
  5. SAMPLE              throw D away; (x̂+1)/2; make_grid
  6. ADD Y               nn.Embedding; concat (MLP) or extra channel (DC)
  7. SWAP BODY           Linear stack → ConvTranspose G / Conv D
  8. SCORE               FID on Inception-2048; RGB-repeat MNIST
  STOP  3-channel data = homework; notebook is in the description
```

### System context

```
  ╔════════════════════════════════════╗
  ║ Lec 04/05: VDM saddle, GAN theory  ║
  ║ Next: your 3-channel experiments   ║
  ╚════════════════╤═══════════════════╝
                   │ this tutorial (~78 min)
                   ▼
        ┌──────────────────────────┐
        │ Colab: 4 GAN variants    │
        │ + FID + label probe      │
        └──────────────────────────┘
```

### Main blueprint

```
  ╔══ JOB ══╗
  ║ sample  ║
  ║ MNIST   ║
  ╚═══╤═════╝
      │ z ~ N(0,I)
      ▼
  ┌─ G_θ ─────────────┐     ┌─ D_w ──────────────┐
  │ MLP or convT      │     │ MLP or conv        │
  │ tanh out [−1,1]   │     │ 1 logit, no σ      │
  └─────────┬─────────┘     └─────────┬──────────┘
            │ x̂                        │
            └──────────►  BCEWithLogits ◄── real x
                         detach on D-step
                         fake-ones on G-step (non-sat)
            │
            ├─ + Y embed  →  cGAN (MLP or DC)
            └─ FID (Inception 2048)  lower better

  four Colab bodies (same losses):
    vanilla MLP     no Y, Linear stacks     FID 92.93
    cGAN MLP        Embedding concat        FID 104   (looks nicer, worse number)
    DCGAN           convT G / conv D        (one run 21.5)
    cDCGAN          110-d G; D is 2-ch 28×28
```

### Scenario walkthrough

Train a forger of handwritten **3**s.

1. Map MNIST to [−1,1] so tanh G matches.  
2. MLP G: 100→784. D: 784→1 logit.  
3. Each batch: D-step with detach; G-step with ones.  
4. Samples look weak (vanilla FID **92.93**).  
5. Concat a learned 10-d **Y**. Now you can *request* a 3. Looks nicer; FID **104** (worse).  
6. Swap convT/conv. cDCGAN looks best; one run **21.5**.  
7. Freeze one $z$, change labels 0–9: the digit follows **Y**.

### Failure / contrast

```
  sigmoid on D + BCEWithLogits     ──X──► double σ
  skip detach on D-step            ──X──► G updated while training D
  G-step targets = zeros           ──X──► saturating VDM G-loss
  trust a pretty grid over FID     ──X──► cGAN-MLP 104 > vanilla 92
  feed 1-ch MNIST to Inception     ──X──► must repeat to 3 ch
  live-code 25 epochs on CPU       ──X──► he already ran Colab
```

### STOP

- Full Colab source is behind a Google login; layers below are **as spoken**.  
- Three-channel training is **homework**.  
- No screenshot tiles this package (video 403).

### Load-bearing claims

- Equilibrium: $D\approx 0.5$ on both clouds.  
- D-step: ones/zeros + **detach**. BCE minus ⇒ descent implements ascent.  
- G-step: **non-saturating** $-log D(G)$; targets **ones**. Same direction as VDM G-loss, stronger when $P$ is small.  
- cGAN: `nn.Embedding`; G/D embeddings **need not match**.  
- DCGAN: transpose G, conv D; cDCGAN D is **two-channel** 28×28.  
- FID **lower better**; 92.93 / 104 / 21.5.

**Speaker:** NPTEL IISc tutorial (not live coding).

---

## Topic 1: VDM saddle and today’s three nets (00:03–06:00)

### Where this sits on the master map

**JOB / ROADMAP.** Theory already made GAN a **VDM** saddle. This hour writes it. Warm-up: [two nets](./PREREQUISITES.md#p1-two-nets), [saddle](./PREREQUISITES.md#p2-saddle).

### Board / screenshot

No content frame (video 403). Reconstruct:

```
  z ~ N(0,I) --> G_θ --> x̂ ~ p_θ
  x or x̂     --> D_w --> P(real | image)
  trained well:  D(real) ≈ D(x̂) ≈ 0.5

  max_w  E_real[log D] + E_z[log(1−D(G(z)))]
  min_θ  (second term only)
```

**Figure — spoken ~01:04–05:49:** generator from Gaussian, D as binary clerk, 0.5 test, two-term loss, three architectures.

### What he is establishing

The tutorial’s job is **implementation**, not a new divergence. GAN is last lecture’s **variational divergence minimization** with a chosen $f$, written as a **saddle** in $\theta$ and $w$.

$G_\theta$ samples $z\sim\mathcal{N}(0,I)$ and emits $\hat x\sim p_\theta$. $D_w$ is a **binary classifier**: $x$ or $\hat x$ in, probability “this came from $p_x$” out. After training you **want D to fail**: both clouds should score about **0.5**.

The board loss has **two expectations**: real data and generated. $\hat x=G(z)$. Both terms depend on $w$ → **maximize $w$**. Replace Es by **batch averages**. The real term does **not** depend on $\theta$, so G **minimizes only the second term**.

No architecture is forced by the math. Today: (1) **vanilla GAN** with **MLP** G and D; (2) **conditional GAN** — same MLPs plus label $Y$; (3) **DCGAN** — **transpose conv** G, **plain conv** D.

You can now name the two nets and the three bodies. You cannot yet match tanh to MNIST.

### Analogy for this topic only

Three prints on the table: a real `3`, a generated `3`, a generated smudge. The clerk outputs 0.99, 0.02, 0.40.

**Is 0.99 on the album a finished GAN?** No. You want the clerk to **shrug on both piles** (~0.5). A sure clerk means the press has not caught up.

In lecture words: forger $=G_\theta$, clerk $=D_w$, shrug $=0.5$.

### Local picture

```
  TODAY
    1. MLP vanilla     (no Y)
    2. MLP + Y         (conditional)
    3. conv vanilla / conv + Y   (DCGAN)

  MATH (unchanged):  max_w  then  min_θ  of the two-term score
```

Notice: “generated” is the word he prefers; “fake” still appears in code names.

### Bridge

A 0.5 clerk is a **feeling**. He wants a **number**: FID, and a Colab that already ran.

---

## Topic 2: FID, Colab, MNIST into tanh range (06:00–12:30)

### Where this sits on the master map

**SETUP.** Metric, device, pixel range. Warm-ups: [FID](./PREREQUISITES.md#p8-fid), [tanh range](./PREREQUISITES.md#p5-tanh).

### Board / screenshot

No content frame. Spoken Colab: `pip` torchmetrics, `device`, MNIST `Normalize`.

### What he is establishing

**FID** (Fréchet Inception Distance): push images through **InceptionNet**, take a **middle embedding**, compare real vs generated clouds. Package: **torchmetrics**. Training is slow; he **already ran** the notebook. **Not live coding.** Link in the YouTube description.

```python
# packages he names (not a full requirements.txt)
# torch, torch.optim, DataLoader, datasets, transforms
# torchvision.utils.make_grid, matplotlib, numpy, torchmetrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CPU works if you shallow the nets; it will take too long
```

**What this does:** pick GPU like every earlier PyTorch tutorial.

MNIST after `ToTensor` is [0,1]. G ends in **tanh** ∈ [−1,1]. Map reals by **subtract 0.5, divide 0.5**. ASR once says “std 0.1”; the **walked arithmetic** is 0.5. One-channel: one mean/std. Three-channel: **ImageNet stats**.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # 0→−1, 1→+1; match tanh
])
loader = DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transform),
    batch_size=128, shuffle=True,
)
# plot first 32 of a batch as the real reference grid
```

**What this does:** every real pixel uses the same numeric range as G’s output.

Hypers for vanilla: $z\in\mathbb{R}^{100}$, $28\times 28$, lr **0.0002** for **both** nets (two lrs / Adam vs SGD are legal), **25 epochs**. Skipping Normalize so tanh G faces [0,1] album pixels is the wrong comparison. You can now match ranges and name FID. You cannot yet write G and D.

### Analogy for this topic only

Album pixel 0 (black) vs tanh press −1 (also “black”). **If you skip Normalize, is the press “wrong” or is the comparison?** The comparison. Convert first: 0→−1, 1→+1. FID is a later museum that never asks the clerk.

In lecture words: Normalize $(0.5,)/(0.5,)$; FID = Inception embeddings.

### Local picture

```
  0..255  --ToTensor-->  0..1  --(x-0.5)/0.5-->  −1..1
  G tanh already in −1..1
  batch 128; show 32 reals
```

Notice: he will not sit through 25 epochs on camera.

### Bridge

Range matches. Next: the two MLP bodies, and why D has **no** sigmoid.

---

## Topic 3: MLP generator and discriminator (12:30–19:45)

### Where this sits on the master map

**VANILLA NETS.** Setup gave files and a tanh range. This box is the two MLP bodies that will play the saddle. Warm-up: [logits](./PREREQUISITES.md#p3-logits).

### Board / screenshot

No content frame. Spoken Sequential widths.

### What he is establishing

```python
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),   nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),  nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),  # last: [−1, 1]
        )
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256),     nn.LeakyReLU(0.2),
            nn.Linear(256, 1),       # logit; NO sigmoid
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
```

**What this does:** G inflates noise to a tanh image. D is a **binary clerk** ending in one raw score.

**Do not** put sigmoid on D. Loss is **`BCEWithLogitsLoss`**, which already applies $\sigma$. Real **1**, generated **0** (he prefers the word generated). Two **Adam** opts at 0.0002; mixing Adam/SGD is allowed.

$$
\ell(A,y)= y\log\sigma(A)+(1-y)\log(1-\sigma(A)).
$$

$y=1$ keeps the first term; $y=0$ the second. Putting sigmoid on D **and** BCEWithLogits is the wrong clerk (σ twice). You can now write both Sequential bodies. What is still missing is the **train step** that flips max into min.

### Analogy for this topic only

G stretches 100 noise numbers into 784 pixels. D turns in a **raw exam score**, not a percentage.

**Should D end with Sigmoid if the loss is BCEWithLogits?** No. That grades twice. The loss already applies $\sigma$.

In lecture words: Sequential widths as printed; no sigmoid on D.

### Local picture

```
  G:  100 → 256 → 512 → 1024 → 784 → tanh → 28×28
  D:  784 → 512 → 256 → 1 logit
```

Notice: leak 0.2 is explicit on D; G also uses LeakyReLU.

### Bridge

Nets exist. The D-step must **cut the tape** into G and **flip max into min**.

---

## Topic 4: Discriminator step, detach, sign flip (19:45–31:35)

### Where this sits on the master map

**D STEP.** Warm-ups: [saddle](./PREREQUISITES.md#p2-saddle), [detach](./PREREQUISITES.md#p4-detach).

### Board / screenshot

No content frame. Spoken loop: flatten, ones, randn, detach, zeros, backward, `optimizer.d.step`.

### What he is establishing

`G.train(); D.train()`. 25 epochs. Batch on CPU → `.to(device)`. MLP wants a **vector**, so flatten 28×28. The $y$ in BCE is **not** the digit — only real vs generated.

```python
opt_d.zero_grad()
real_logits = D(real)                         # D_w(x_i)
real_loss = criterion(real_logits, torch.ones(B, 1, device=device))
z = torch.randn(B, 100, device=device)
fake = G(z).detach()                          # cut tape into G
fake_logits = D(fake)
fake_loss = criterion(fake_logits, torch.zeros(B, 1, device=device))
(d_loss := real_loss + fake_loss).backward()
opt_d.step()                                  # only w
```

**What this does:** first sum = reals ($p_x$); second = generated ($p_\theta$).

`optimizer.step` **only descends**. Theory **maximizes** $w$. BCE’s **minus** turns that max into a min so descent is the right direction. Sign mismatch trains a **different** objective.

`.detach()`: fake images still remember G. D-step must **not** update $\theta$. Skipping detach is the wrong photocopy: the press gets fined on the clerk’s lesson. You can now write the D-step. $\theta$ has not moved.

### Analogy for this topic only

The print still has the press’s serial number. **If the clerk’s lesson backprops into G, who paid?** The forger, on the clerk’s clock. Photocopy (`.detach()`) first. On the **G-step** you must **not** photocopy — the press needs the tape through the clerk (without stepping the clerk).

In lecture words: detach prevents gradients flowing into G.

### Local picture

```
  real:  D(x) vs ones     →  −log σ(D(x))
  fake:  G(z).detach() vs zeros  →  −log(1−σ(D(x̂)))
  sum.backward(); step D        →  descent ≡ theory ascent
```

Notice: one D update per batch here.

### Bridge

$w$ moved. $\theta$ still needs a step — and he **refuses** the VDM G-loss as written.

---

## Topic 5: Non-saturating generator loss (31:35–46:27)

### Where this sits on the master map

**G STEP.** D just moved $w$. The leftover problem is $\theta$: theory wrote $\min \log(1-D(G))$, and he will **refuse** that line in code. Warm-up: [logits](./PREREQUISITES.md#p3-logits).

### Board / screenshot

No content frame. Spoken $P$ table and `torch.ones` on fakes.

### What he is establishing

Theory G-loss: $\frac1B\sum\log(1-D_w(G_\theta(z_j)))$. He **does not implement it**. **Non-saturating** objective (engineering, **not** a new VDM theorem):

$$
\theta^\star=\arg\min_\theta\;-\frac1B\sum_j \log D_w(G_\theta(z_j)).
$$

Let $P=\sigma(D(G(z)))$. Both losses want **larger $P$**. Gradients wrt the activation: saturating $\partial L_1 \propto -P$, non-sat $\propto -(1-P)$.

| $P$ | sat $\sim -P$ | non-sat $\sim -(1-P)$ |
|-----|----------------|------------------------|
| 0.01 | −0.01 | −0.99 |
| 0.10 | −0.10 | −0.90 |
| 0.50 | −0.50 | −0.50 |
| 0.90 | −0.90 | −0.10 |

**Same direction.** When D is winning ($P$ small), non-sat is **stronger**. Code trick: even on generated images set **$y=1$** so BCEWithLogits becomes $-\log\sigma(A)$.

```python
opt_g.zero_grad()
fake = G(z)                                   # NO detach — G needs the tape
g_loss = criterion(D(fake), torch.ones(B, 1, device=device))
g_loss.backward()
opt_g.step()                                  # only θ; do not step D
```

**What this does:** G tries to make D say “real” on prints.

One batch, both steps together (the loop he walks; two Adam opts):

```python
# per batch of real images (already on device, flattened if MLP)
opt_d.zero_grad()
real_loss = criterion(D(real), torch.ones(B, 1, device=device))
z = torch.randn(B, 100, device=device)
fake = G(z)
fake_loss = criterion(D(fake.detach()), torch.zeros(B, 1, device=device))
(real_loss + fake_loss).backward()
opt_d.step()                       # w only

opt_g.zero_grad()
g_loss = criterion(D(fake), torch.ones(B, 1, device=device))  # non-sat
g_loss.backward()
opt_g.step()                       # θ only; D not stepped
# homework: wrap G or D in an inner for-loop k times; watch FID
```

**What this does:** one clerk lesson, one press lesson. Homework: try $k$ G updates per D (or reverse).

Homework: inner-loop $k$ G steps per D (or reverse); watch FID. Implementing $\log(1-D(G))$ as written is the weak shove when $P\approx 0$. You can now write both steps of the batch. You cannot yet **draw** without D.

### Analogy for this topic only

The clerk is already sure the print is fake ($P=0.01$).

**Which shove moves the press faster — the VDM line $\log(1-P)$ or $-\log P$?** The table: saturating nudge $\approx 0.01$, non-sat $\approx 0.99$, **same direction**. He keeps `arg min` but swaps the formula. Code: pretend generated $y=1$.

In lecture words: non-saturating; fake targets are ones.

### Local picture

```
  VDM G:     min  log(1−P)     weak when P≈0
  non-sat:   min −log P        strong when P≈0
  both want P ↑
```

Notice: two different functions, same desired end.

### Bridge

Both nets stepped. How do you **draw** without D?

---

## Topic 6: Sampling vanilla; starting cGAN (46:27–51:09)

### Where this sits on the master map

**SAMPLE**, then leftover $Y$. Warm-up: [two nets](./PREREQUISITES.md#p1-two-nets).

### Board / screenshot

No content frame. Spoken: 32 noises, denorm, “quite bad” digits.

### What he is establishing

**Discard D.** At inference the clerk is gone. Sample 32 noises of length 100 on the device, pass G, reshape to $32\times 1\times 28\times 28$. Values still sit in $[-1,1]$; **add 1 and divide by 2** to plot in $[0,1]$. `.cpu()` then `make_grid`. He agrees the vanilla grid is **quite bad**; a few 0s, 1s, 9s are only “to a certain extent” OK. That is what 25-epoch MLP looks like, not a bug in your copy-paste.

Conditional: the **game** stays. The only change is that every time you push a real through D or noise through G, you also pass **label $Y$**. A single integer taped to a 100-d $z$ (or a 784-d $x$) is too weak — one number against a hundred. Options: **one-hot** the ten digits, or **learn a row-vector** per digit. He will take embeddings.

You can now sample without D. You cannot yet make the press obey “print a 7.”

### Analogy for this topic only

Throw the clerk out. Sample 32 noises, print, undo tanh, make a grid. **Can you ask for a “7”?** Not on vanilla — the press has no name-tag slot. A lonely integer next to 100 noise numbers will not steer it. That leftover is cGAN.

In lecture words: inference $=z\to G$; $Y$ needs a vector.

### Local picture

```
  infer:  randn(32,100) → G → view 28×28 → (x+1)/2 → grid
  next:   how to feed Y without it being one lonely number
```

Notice: D is training-only.

### Bridge

A lonely integer next to 100 noise numbers will not steer the press. The leftover problem is: **what vector should stand in for $Y$?**

---

## Topic 7: Conditional MLP: embeddings (51:09–62:41)

### Where this sits on the master map

**CGAN-MLP.** Vanilla can sample but cannot **request a digit**. This box tapes a learned name-tag onto $z$ and onto flat $x$. Warm-up: [embeddings](./PREREQUISITES.md#p6-embed).

### Board / screenshot

No content frame. Spoken `nn.Embedding(10,10)`, concat dim 1, `randint`.

### What he is establishing

```python
class CondGenerator(nn.Module):
    def __init__(self, z_dim=100, n_classes=10, emb=10):
        super().__init__()
        self.lab = nn.Embedding(n_classes, emb)     # 10×10 rows
        self.net = nn.Sequential(                   # input 110
            nn.Linear(z_dim + emb, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh(),
        )
    def forward(self, z, y):
        yv = self.lab(y)                            # (B, 10)
        return self.net(torch.cat([z, yv], 1)).view(-1, 1, 28, 28)
```

**What this does:** digit $k$ looks up **row $k$**. Rows are **parameters**. On the G-step, grads must flow **through G into the embedding** (D not stepped).

D has a **separate** `nn.Embedding`; they need not match. D input **794**. Same BCE. Batch now includes **digit labels**. Fakes: `y_fake = torch.randint(0, 10, (B,))`. Still **detach** on D-step, **ones** on G-step.

Inference: `requested = list(range(10))*3` — 0–9 three times. He says this is **far better** than unconditional MLP. Taping a raw integer onto $z$ is the weak name-tag; the 10-row matrix is the right one. You can now request a digit. The press still has no **spatial** inductive bias.

### Analogy for this topic only

“7” is a name tag. The press learns a 10-number **accent** (row of a 10×10 matrix). The clerk learns a **different** accent.

**Must those two matrices agree?** No. Taping a raw integer `7` onto $z$ is the weak tag. `nn.Embedding` is the right one. Fake labels at train time are `randint(0,9)`, not the real digit.

In lecture words: 10×10 matrix; concat to 110 / 794.

### Local picture

```
  y=2 → row 2 (10) ++ z(100) → 110 → G
  fake y ~ Uniform{0..9}
  request 0,1,…,9 at sample time
```

Notice: loss **formula** unchanged; only **inputs** grew.

### Bridge

The MLP press can now be asked for a 7, but it still treats the image as a **784-bag of pixels**. The leftover problem is spatial structure: **conv and transpose-conv**.

---

## Topic 8: DCGAN conv and transpose (62:41–67:24)

### Where this sits on the master map

**DCGAN.** Warm-up: [conv vs transpose](./PREREQUISITES.md#p7-convt).

### Board / screenshot

No content frame. Spoken k=4, s=2, p=1; 128×7×7.

### What he is establishing

Label machinery **unchanged**. Only G/D **bodies**.

```python
class DCGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128 * 7 * 7)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),
        )
    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.conv(x)                       # → 1×28×28

class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)      # line ~72 in his notebook
        return self.fc(h)
```

**What this does:** G **grows** 7→14→28; D **shrinks** 28→14→7 then one logit. Train loop **identical**. He shows DC samples. Flattening 28×28 into an MLP is the wrong geometry for a digit’s strokes; conv keeps neighborhood. You can now swap the body. Labels on this body are still open.

### Analogy for this topic only

D shreds 28×28 → 14×14 → 7×7 crumbs. G is a **blow-up copier** 7→14→28.

**Could you keep the MLP 784-bag and just train longer?** You can, but strokes have neighbors. Conv keeps them. Same courtroom (BCE, detach, ones), new buildings (k=4, s=2, p=1).

In lecture words: transpose G, conv D.

### Local picture

```
  G: 100 → 128×7×7 → 64×14×14 → 1×28×28 tanh
  D: 1×28×28 → 64×14×14 → 128×7×7 → flatten → 1
```

Notice: ReLU on G (Leaky OK); Leaky 0.2 on D.

### Bridge

Put $Y$ on this conv body — D’s concat will **not** be a 10-vector.

---

## Topic 9: Conditional DCGAN (67:24–72:33)

### Where this sits on the master map

**CGAN-DC.** Conv bodies exist; they still cannot **request a digit** until $Y$ is glued on. MLP glued a 10-vector. Conv D wants a **spatial** sheet. Warm-up: [embeddings](./PREREQUISITES.md#p6-embed).

### Board / screenshot

No content frame. Spoken 110-d G; D as **two-channel** 28×28.

### What he is establishing

G: Linear input **110** (100+10), same convT stack, tanh. Labels → embed → cat with $z$ → 128×7×7 → image.

D: he first says “add,” then **corrects to concatenate**. Embed $Y$ as a **28×28 map**, stack as channel 2.

```python
# D input: real/fake image (B,1,28,28) and label-map (B,1,28,28)
x_in = torch.cat([x, y_map], dim=1)   # (B, 2, 28, 28)
# then the same conv stack, first Conv2d in_channels=2
```

**What this does:** each pixel sees the image **and** a spatial label sheet.

Four systems: vanilla MLP, cGAN MLP, DCGAN, cDCGAN. **cDCGAN far superior** (not perfect; 9s messy). Only **25 epochs**.

### Analogy for this topic only

Tape a **stencil** the size of the photo onto the photo (second channel), not a 10-number sticker on the noise. **Can the clerk use a tiny sticker on a 28×28 grid?** He chose a full-size sheet instead.

In lecture words: two-channel image into conv.

### Local picture

```
  G: z(100)++emb(10) → FC → 128×7×7 → convT → 28×28
  D: [image | label-map] 2×28×28 → conv → 128×7×7 → 1
```

Notice: train loop = cGAN MLP.

### Bridge

The cDCGAN grid looks best to the eye. The leftover problem is whether a **third inspector** (FID) agrees — and it will not always.

---

## Topic 10: FID numbers, noise probe, notebook (72:33–78:33)

### Where this sits on the master map

**SCORE / HOMEWORK.** Four variants exist. This box refuses to ship on a 32-image grid: **FID**, a same-noise probe, and the Colab link. Warm-up: [FID](./PREREQUISITES.md#p8-fid).

### Board / screenshot

No content frame. Spoken 21.5, 92.93, 104; same-$z$ strip.

### What he is establishing

```python
from torchmetrics.image.fid import FrechetInceptionDistance
# generated in [−1,1] → clamp [0,1]; repeat 1ch → 3ch (Inception is RGB)
# fid = FrechetInceptionDistance(feature=2048)
# ~5000 (he got 5120) generated images; lower is better
```

**What this does:** numeric quality without D.

He reports **21.5** on one DCGAN-scale run. Vanilla **92.93**, conditional MLP **104** — looks nicer, **worse FID**. DCGAN reverses that. **Lower is better.**

Probe: **one** 100-d $z$, labels 0–9 → different digits (cDCGAN). Then **same label, many $z$**. Output depends on **both**, more on **labels**.

Close: math↔code (BCE, non-sat). MNIST is 1-ch. Homework: **3-channel** if you have compute. **Notebook in the description** — don’t recode; enhance; comment samples.

### Analogy for this topic only

Same clay ($z$), ten cookie-cutters (labels 0–9) → ten different digits. Same cutter, many clays → variants of **one** digit.

**Which knob mostly picks the class?** The label. Eyes said cGAN-MLP looked nicer; FID said **104 vs 92.93** — the museum disagreed. Lower FID wins.

In lecture words: same noise, different labels; lower FID better.

### Local picture

```
  FID:  vanilla MLP 92.93 | cGAN MLP 104 | DC-scale 21.5
  lower better; pretty ≠ FID

  same z, y=0..9  →  different digits
  same y, many z  →  style of one digit
```

Notice: 25 epochs is small; he does not claim SOTA.

### Bridge

Four Colab variants + FID + a label probe. Next sitting is whatever the course names; your homework is **3-channel** on that notebook.

---

## External references

All companions live **here**, not under the topics. Mix of **video**, **blog/docs**, and **original papers**. No Wikipedia.

**Start here (if you only open three).** Course Colab → CS236 notes (non-sat G) → PyTorch DCGAN tutorial.

### Per-topic companions (2–3 each)

Use **after** the matching topic. Do not open thirty tabs.

| Topic / map box | Type | Resource | Why it helps |
|-----------------|------|----------|--------------|
| **1 · VDM saddle** | paper | [Goodfellow et al. GAN (arXiv:1406.2661)](https://arxiv.org/abs/1406.2661) | Original two-term minmax; $D\to\tfrac12$ at equilibrium. |
| **1 · VDM saddle** | notes | [Stanford CS236 GAN notes](https://deepgenerativemodels.github.io/notes/gan/) | Written minimax + why saturating G-loss is weak. |
| **1 · VDM saddle** | video | [Stanford CS231n 2025 Lec 14](https://www.youtube.com/watch?v=Edr4uZFh4EE) | Latest large-course GAN + DCGAN hour. |
| **2 · FID / tanh range** | paper | [Heusel et al. FID (arXiv:1706.08500)](https://arxiv.org/abs/1706.08500) | Original Fréchet Inception Distance; lower better. |
| **2 · FID / tanh range** | docs | [TorchMetrics FID](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html) | The `FrechetInceptionDistance` API he imports. |
| **2 · FID / tanh range** | notes | Course Colab (video description) | Pre-run MNIST pipeline; not live 25 epochs. |
| **3 · MLP G/D** | docs | [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) | σ lives **inside** the loss; no module sigmoid. |
| **3 · MLP G/D** | video | TensorFlow DCGAN tutorial (MNIST loop) — see Topic 8 row if you want Keras side-by-side | Same two-player loop, different body. |
| **4 · detach / D-step** | docs | [Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html) | Cuts the tape so D-step cannot train G. |
| **4 · detach / D-step** | tutorial | [PyTorch autograd basics](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | `detach` vs `no_grad`; why the photocopy exists. |
| **5 · non-sat G** | notes | CS236 GAN notes (same as Topic 1) | Saturating $\log(1-D(G))$ vs $-\log D(G)$. |
| **5 · non-sat G** | paper | [Goodfellow NIPS 2016 GAN tutorial (arXiv:1701.00160)](https://arxiv.org/abs/1701.00160) | Names the non-saturating heuristic he codes with **ones**. |
| **6 · sample / start cGAN** | docs | [make_grid](https://pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html) | The 32-image reference grid. |
| **6 · sample / start cGAN** | paper | [Mirza & Osindero cGAN (arXiv:1411.1784)](https://arxiv.org/abs/1411.1784) | Original “feed $Y$ to G and D.” |
| **7 · embeddings** | docs | [nn.Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) | 10×10 learned rows; concat to 110 / 794. |
| **7 · embeddings** | paper | Mirza cGAN (same as Topic 6) | Why a raw integer next to $z$ is too weak. |
| **8 · DCGAN** | paper | [Radford et al. DCGAN (arXiv:1511.06434)](https://arxiv.org/abs/1511.06434) | BN, no pooling, tanh G, conv D. |
| **8 · DCGAN** | tutorial | [PyTorch DCGAN (faces)](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) | Same k=4, s=2, p=1 ConvTranspose pattern. |
| **8 · DCGAN** | tutorial | [TensorFlow DCGAN (MNIST)](https://www.tensorflow.org/tutorials/generative/dcgan) | MNIST conv GAN with an explicit train loop. |
| **9 · cDCGAN** | paper | Radford DCGAN + Mirza cGAN (above) | Conv body + label channel. |
| **9 · cDCGAN** | video | CS231n 2025 Lec 14 (same as Topic 1) | DC-GAN architecture board in a 2025 course. |
| **10 · FID numbers** | paper | Heusel FID (same as Topic 2) | Why ~5k images and Inception-2048. |
| **10 · FID numbers** | docs | TorchMetrics FID (same as Topic 2) | Clamp [0,1], repeat 1ch→3ch. |
| **10 · FID numbers** | code | [Course Colab](https://colab.research.google.com/drive/15nVkKu1mySDHzEj4NaqDjDAUUvZsTxzR?usp=sharing) | **Original notebook** — enhance; don’t recode from this article. |

**How to use.** After Topic 4, `detach` docs. After Topic 5, Goodfellow 2016 tutorial or CS236. After Topic 8, Radford + PyTorch DCGAN. After Topic 10, open **their Colab**. Do not invent a 3-channel loop today — that is homework.

---

## Apply it (scenarios)

*Workplace-style situations that use ideas from this video only.*

### Scenario 1: Sign mismatch on D
**Context:** intern’s D loss goes down, samples get worse.  
**Challenge:** they used `−BCE` “because theory maximizes.”  
**Questions:** 1. What does `optimizer.step` do? 2. What is BCE’s built-in minus for?

<details><summary>Show solution sketch</summary>

- Topic 4: step only **descends**. BCE already flips max→min. Extra minus trains the wrong $w$.

</details>

### Scenario 2: G updates on the clerk’s lesson
**Context:** both nets’ weights move during `d_loss.backward()`.  
**Challenge:** forgot `.detach()` on fakes.  
**Questions:** 1. Who is on the tape? 2. When must you *not* detach?

<details><summary>Show solution sketch</summary>

- Topic 4: detach on **D-step**. Topic 5: **keep** the tape on **G-step**.

</details>

### Scenario 3: Pretty cGAN, worse FID
**Context:** product wants “sharper 7s.” cGAN-MLP grid looks better; FID 104 vs vanilla 92.93.  
**Challenge:** ship on looks?  
**Questions:** 1. What does he say lower FID means? 2. Why can eyes and FID disagree at 25 epochs?

<details><summary>Show solution sketch</summary>

- Topic 10: lower better; he flags this reversal. Don’t ship on a 32-image grid.

</details>

### Scenario 4: Inception rejects MNIST
**Context:** FID code crashes or is nonsense on 1×28×28.  
**Challenge:** Inception wants 3-channel [0,1].  
**Questions:** 1. Clamp? 2. Repeat?

<details><summary>Show solution sketch</summary>

- Topic 10: clamp [0,1], `repeat` to 3 channels, feature 2048, ~5k images.

</details>

---

## Sources

- Video: [Tutorial 12 Implementations](https://www.youtube.com/watch?v=dBcURX7GrwE) · NPTEL IISc  
- Colab: [drive/15nVkKu1mySDHzEj4NaqDjDAUUvZsTxzR](https://colab.research.google.com/drive/15nVkKu1mySDHzEj4NaqDjDAUUvZsTxzR?usp=sharing) (description)  
- Captions: `raw/captions.en.timed.txt` (cleaned: VDM, tanh, detach, BCEWithLogits, FID, ConvTranspose)  
- **Ingest:** captions yes; video **403 Forbidden** → E2, ASCII boards  
- **Code audit:** reconstructed from spoken widths/losses; not a dump of the private Colab. Do not treat snippets as the only runnable file — use the notebook.
