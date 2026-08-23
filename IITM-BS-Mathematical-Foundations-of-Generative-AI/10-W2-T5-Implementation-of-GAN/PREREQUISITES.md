# Prerequisites — warm-up before W2_T5 (vanilla GAN in PyTorch)

> **Do this first** if “$z\sim\mathcal{N}(0,I)$,” “sigmoid as $P(\text{real})$,” “BCE,” “ascent vs descent,” “`.detach()`,” or “Tanh $\leftrightarrow[-1,1]$” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> IITM BS · Week 2 Tutorial 5 · TA live-codes (Chandan). Theory already wrote $J(\theta,w)$. This hour **runs it**.  
> **Beginner:** purpose · definition · micro with numbers · analogy · ASCII · notice · mini-check.

Previous sheets left a **minmax** on a generator $G_\theta$ and a discriminator $D_w$. This hour **implements the vanilla flavor** on **MNIST** ($28\times 28$ digits) with two MLPs, two Adams, and BCE.

```
  After this warm-up you can say:

  "D is a finite IID sample from unknown p_x; MNIST train is ~60,000 such 28×28 grids."
  "Flatten stacks a 28×28 photo into 784 numbers; a batch of 128 stays 128 rows."
  "z is a 100-number recipe card from N(0,I); G_θ turns the card into a fake photo."
  "D_w is a detective: sigmoid out is P(this photo is real), not which digit 0–9."
  "BCE is how shocked the detective is when you tag the photo 1 (real) or 0 (fake)."
  "D climbs J (ascent, plus the gradient). G descends only the fake term (minus)."
  "Freeze the other player: two optimizers; .detach() cuts G off D's backward."
  "Normalize((0.5,), (0.5,)) and Tanh agree on pixels in [−1, 1]."
```

```
  §1  IID sample from p_x + flatten     ──► Topics 1, 3, 6, 9
  §2  Noise z ~ N(0,I)                  ──► Topics 2, 5, 8, 9
  §3  Forger G vs detective D (MLP)     ──► Topics 2, 7
  §4  Sigmoid = P(real)                 ──► Topics 2, 7
  §5  BCE as log D and log(1−D)         ──► Topics 3–5, 8–9
  §6  Ascent vs descent                 ──► Topics 4–5
  §7  Freeze, two opts, detach          ──► Topics 4–5, 8–9
  §8  Tanh matches Normalize [−1,1]     ──► Topics 6–8
```

**One scene (reuse all hour).** A **bank** with a vault of real handwritten **digit photos** (MNIST). A **forger** ($G$) is handed a sealed envelope of 100 random numbers ($z$) and must draw a photo. A **teller** ($D$) looks at a photo — real or forged — and whispers a number between 0 and 1: “how sure am I this came from a real customer?” They take **turns**. The teller wants to be right. The forger wants the teller to say “real” on fakes. Digit stickers 0–9 on the slips are **ignored**.

```
  REAL pile     x ~ p_x          MNIST 28×28, flattened to 784
  ENVELOPE      z ~ N(0, I)      100 numbers
  FORGER        xhat = G_θ(z)    784 numbers → a 28×28 picture
  TELLER        D_w(photo) → p   one number in (0,1) after sigmoid
  TAGS          1 = “this is real”,  0 = “this is fake”
                (NOT the digit 0–9)
  CART          128 slips at a time (batch)
```

---

## 1. Data is an IID sample from $p_x$; a photo is 784 numbers

<a id="p1-iid-px"></a>

**Purpose.** The tutorial’s first object is the dataset $D$, not the nets. You also need **flatten** because an MLP cannot eat a $28\times 28$ grid.

**Definition.** Unknown true law $p_x$ over photos. The file you download is a **finite list** $D=\{x_1,\ldots,x_n\}$ drawn **independently and identically (IID)** from that law. “Independent” = knowing photo 7 does not tell you photo 8. “Identically” = every photo is a draw from the **same** $p_x$, not a mix of two vaults. MNIST **training** is about **60,000** $28\times 28$ grayscale digits. ASR “28 + 28” is **$28\times 28$**. **Flatten** lays the 28 rows end to end: $28\times 28=784$. A **batch** of 128 is 128 such vectors stacked, shape `(128, 784)` after `view`. The GAN does **not** use the digit class (0–9). Those labels exist on disk; vanilla GAN unpacks `real, _` and throws `_` away.

**Micro.** One “7”:

```
  28 rows × 28 cols of brightness
  row 1: pixels 1–28
  row 2: pixels 29–56
  …
  row 28: pixels 757–784
  → one column of 784 numbers
```

A cart of 128 such columns is one **batch** $B_1$ from $p_x$. He uses **shuffle=True** so the cart is not the same 128 digits every epoch. **No test set**: they are generating, not scoring accuracy.

**Analogy.** The bank vault holds 60,000 real deposit slips. You never see the infinite law of “all possible handwriting.” You only see this vault. Training means: learn to print **new** slips that could have come from the same vault — not memorize slip #42. Flatten is copying each crossword page into **one spreadsheet column**. Sixty-four (or 128) pages on a cart stay 128 columns; flatten does not smash the cart into one photo.

Someone asks: **if you only recite slip #42, can you print a new one?** Memory of $D$ cannot. That is why $G$ must exist.

```
  unknown p_x  ──IID draws──►  D = {x1, x2, …, xn}   ~60,000 train
                                    │
                                    ▼
                              one xi:  28 × 28 grid
                                    │  flatten
                                    ▼
                                 784 numbers
                                    │  DataLoader batch 128, shuffle
                                    ▼
                              B1 = cart of 128 reals
```

**Notice.** He picks MNIST because **GPU is limited**. Miniature on purpose so you can **run it and see** digits appear. The interesting law is still $p_x$; MNIST is a $p_x$ that fits Colab.

**Mini-check.** After flatten, is the batch still 128 photos, or one giant photo of size $128\times 784$?

---

## 2. Noise $z\sim\mathcal{N}(0,I)$ is a recipe card

<a id="p2-noise-z"></a>

**Purpose.** You cannot sample $p_x$ directly. You sample something **easy**, then push it through $G$.

**Definition.** $z$ is a vector of length **100** (he says “a random thing”; you may change it). $\mathcal{N}(0,I)$ means: each coordinate is an independent **standard normal** — mean 0, variance 1, no correlation between coordinates. $I$ is the $100\times 100$ identity covariance. In code: `torch.randn(batch_size, 100)` — that is **standard normal**, not uniform $[0,1]$.

**Micro.** One $z$ might start `(0.21, −1.43, 0.80, …)` — 100 numbers. $G_\theta$ is a trained function that turns **that** 100-list into a 784-list that looks like a digit. Different $z$ → different fake photo. **Same** $z$ later in training → you can watch one photo improve. That taped envelope is **fixed noise**: the notebook draws `torch.randn(64, 100)` **once**, then plots an $8\times 8$ grid (`nrow=8`) every 10 epochs.

**Analogy.** The forger never copies a vault slip. The manager hands a **sealed envelope** of 100 random numbers. The forger must invent a slip from the envelope alone. If you keep one envelope in a drawer and ask for a drawing every 10 days, you can compare “the same idea” as the forger gets better. If you handed a **fresh** envelope every time, “better looking” might just mean “luckier numbers.”

Someone asks: **did the photo get better because $G$ improved, or because $z$ got easier?** Fixed noise answers that.

```
  z ~ N(0, I)     100 numbers     torch.randn(B, 100)
        │
        ▼
     G_θ(z)
        │
        ▼
     xhat          784 numbers  →  reshape 28×28

  fixed_noise: 64 envelopes taped to the wall
               same 64, every 10 epochs, G has changed
```

**Notice.** $z$ is **not** a compressed MNIST photo (that would be an autoencoder). You never encode a real $x$ to get $z$ in this vanilla loop. You only **sample** $z$ and push forward. He corrects himself later: images are **generated**, not reconstructed.

**Mini-check.** If you pass the **same** $z$ through $G$ twice with frozen weights, do you get the same $\hat{x}$?

---

## 3. Generator forges; discriminator judges (two MLPs)

<a id="p3-g-d"></a>

**Purpose.** Two networks, two jobs, two parameter piles. An **MLP** (multi-layer perceptron) is stacked linear maps with a bend between them.

**Definition.** **Generator** $G_\theta$: input $z$, output $\hat{x}$ (a fake image). Parameters $\theta$ (all the $W,b$ in $G$). **Discriminator** $D_w$: input a photo (real $x$ **or** fake $\hat{x}$), output a score that becomes $P(y=1\mid\text{photo})$. Parameters $w$. Both are **MLPs** / fully connected stacks in this tutorial. **CNN** is homework, not the coded net. A **Linear** layer maps a vector $u$ to $W^\top u + b$. Stack several Linears with **ReLU** (or **LeakyReLU**) between them so the whole map is not just one big Linear.

**Micro.** Tonight’s sizes (Colab):

| Net | In | Hidden | Out | Last activation |
|-----|----|--------|-----|-----------------|
| $G$ | 100 | 256, 512, 1024 | 784 | **Tanh** |
| $D$ | 784 | 512, 256 | **1** | **Sigmoid** |

After $G$, reshape 784 back to $28\times 28$ for a picture. $D$ never needs the grid — it eats the flat 784. $D$ does **not** take $z$ as input.

**Analogy.** Forger and teller. The forger only sees envelopes. The teller sees photos and must not be told in advance which pile they came from. Four desks for the forger (100→256→512→1024→784). Three desks for the teller (784→512→256→1). If you let the forger rewrite the teller’s rulebook while the teller is studying a fake, the game is rigged — that is why we **freeze** one player (warm-up 7).

Someone asks: **does the teller ever open the envelope?** No. Only the slip.

```
  envelope z          real x from vault
       │                    │
       ▼                    │
    G_θ (forger, MLP)       │
       │                    │
       ▼                    ▼
     xhat  ─────────►  D_w (teller, MLP)  ──►  p in (0,1)
                       ▲
                       └── also eats real x

  G never sees real x in the vanilla loop.
  D never sees z.
```

**Notice.** Vanilla = **basic flavor**. Later tutorials swap the loss, the architecture, or the data. Same two-player picture. Last $D$ layer is **1**, not 10 — this is not last week’s classifier.

**Mini-check.** Does $D$ ever take $z$ as input in this vanilla GAN?

---

## 4. Sigmoid is $P(\text{this is real})$

<a id="p4-sigmoid-real"></a>

**Purpose.** $D$’s last number must be a probability, not a digit class.

**Definition.** **Sigmoid** $\sigma(t)=\frac{1}{1+e^{-t}}$ maps any real $t$ into $(0,1)$. He puts it at the **end of $D$**. The output is $P(y=1\mid\text{input})$ — “how much probability it belongs to the **real** data.” $y=1$ means **real**, $y=0$ means **fake**. This is **binary**, not 10-way MNIST classification.

**Micro.** Work three teller whispers:

| Raw $t$ (last Linear) | $\sigma(t)$ | Meaning |
|-----------------------|-------------|---------|
| $+3$ | $\approx 0.95$ | almost sure real |
| $0$ | $0.50$ | coin-flip confused |
| $-3$ | $\approx 0.05$ | almost sure fake |

A batch of 128 photos → 128 probabilities, shape `(128, 1)`. **Not** `(128, 10)`.

**Analogy.** The teller does not shout “this is a 7.” The teller whispers “72% chance this slip came from the vault.” A 7 that looks traced still might score **low** (fake handwriting). A blob that looks like ink from a real customer might score **high**. Digit identity is the wrong question.

Someone asks: **if the photo is a perfect “3” that the forger drew, should $D$ output 3 or 0.9?** The second. Real-vs-fake, not class.

```
  D's last Linear → 1 raw number t  (any real)
                      │  Sigmoid
                      ▼
                   p = σ(t) ∈ (0,1)
                   “P(y=1 | photo)”

  10-way softmax would answer “which digit?”  ← wrong head
```

**Notice.** Do **not** put a 10-way softmax on $D$. The 0–9 labels on the DataLoader are discarded (`real, _`).

**Mini-check.** If $D(x)=0.9$ for a real $x$, is the teller doing well or poorly on that example?

---

## 5. Binary cross-entropy (BCE) is the log terms

<a id="p5-bce"></a>

**Purpose.** The chalkboard writes $\log D$ and $\log(1-D)$. PyTorch writes `nn.BCELoss`.

**Definition.** For a predicted probability $p\in(0,1)$ and a target $y\in\{0,1\}$,

$$\mathrm{BCE}(p,y)=-\bigl(y\log p+(1-y)\log(1-p)\bigr).$$

If $y=1$ (tell the net “this is real”), BCE $=-\log p=-\log D_w(x)$. If $y=0$ (“this is fake”), BCE $=-\log(1-p)=-\log(1-D_w(\hat{x}))$. That is why he says the two $J$ terms are a **straightforward BCE**.

**Micro.** Tiny numbers (natural log):

| $p$ | tag $y$ | BCE $=-(y\log p+(1-y)\log(1-p))$ | Story |
|-----|---------|----------------------------------|--------|
| $0.9$ | $1$ real | $-\log 0.9\approx 0.11$ | teller right, small shock |
| $0.1$ | $1$ real | $-\log 0.1\approx 2.30$ | tagged real, whispered fake — huge |
| $0.1$ | $0$ fake | $-\log 0.9\approx 0.11$ | tagged fake, whispered fake — good |
| $0.9$ | $0$ fake | $-\log 0.1\approx 2.30$ | tagged fake, whispered real — fooled |

Real batch: `criterion(D(real), ones)` ≈ first term of $J$ (PyTorch already has the minus). Fake batch for **$D$**: `criterion(D(fake), zeros)` ≈ second term. Add them. For the **generator trick**, call BCE on **fakes** but with **ones** — you *claim* they are real.

**Analogy.** After each slip, you tell the teller the official tag. If you tagged it **real** and the teller said $p=0.01$, BCE is huge (shocked). If you tagged it **fake** and the teller said $p=0.01$, BCE is tiny (correct suspicion). The forger’s trick is: show a fake, but **tell the scoring rule the tag is real**, so the only way to lower BCE is to make $D(\hat{x})$ climb toward 1.

Someone asks: **the drawing is fake — shouldn’t the tag be 0?** For the **teller’s** study hall, yes. For the **forger’s** practice, no: the tag is 1 on purpose.

```
  y=1 (real tag):   BCE = −log p        = −log D(x)
  y=0 (fake tag):   BCE = −log(1−p)     = −log(1−D(xhat))

  D's total:   BCE(real, ones) + BCE(fake, zeros)
  G's trick:   BCE(D(G(z)), ones)     ← tag fakes as real
```

**Notice.** PyTorch `BCELoss` expects **probabilities** in $(0,1)$ because $D$ already has Sigmoid. (`BCEWithLogitsLoss` would want raw $t$; he does **not** use that.)

**Mini-check.** For the generator trick, is the BCE target `ones` or `zeros`?

---

## 6. Discriminator climbs; generator descends

<a id="p6-ascent-descent"></a>

**Purpose.** Same $J(\theta,w)$, two opposite moves. The **plus vs minus** on the learning-rate line is the whole game.

**Definition.** $J$ is large when $D$ is right: $\log D(\text{real})$ big (real scores near 1) and $\log(1-D(\text{fake}))$ big (fake scores near 0). **Discriminator:** $w^\star=\arg\max_w J$ → **gradient ascent** $w\leftarrow w+\eta_D\nabla_w J$. **Generator:** $\theta^\star=\arg\min_\theta J$ → **gradient descent** $\theta\leftarrow\theta-\eta_G\nabla_\theta J$. He allows **two different learning rates** (the Colab happens to set both to $2\times 10^{-4}$).

**Micro.** First term of $J$ is $\mathbb{E}_{x\sim p_x}[\log D_w(x)]$. There is **no $G$** in it, so **no $\theta$**. $G$ therefore **drops term 1** and only fights $\log(1-D(G(z)))$. Descent on that term, or equivalently the **trick** of BCE vs ones, pushes $D(G(z))$ up.

If both players took a **minus** step on $J$, the teller would get **worse on purpose**. That is the classic mix-up.

**Analogy.** Teller wants a **high score** for “I can tell vault from forgery.” Forger wants that score **low**. Same exam, opposite wishes. Climbing a hill vs walking down it — same hill, opposite feet.

Someone asks: **why doesn’t PyTorch write `w += lr * grad`?** Frameworks **minimize** a loss. BCE already carries the minus-log, so **minimizing** `BCE(real,ones)+BCE(fake,zeros)` **is** ascent on $J$.

```
  J(θ, w) = E_real[ log D(x) ] + E_fake[ log(1 − D(xhat)) ]
             term 1: only w          term 2: θ and w

  D:  w  ←  w  +  η_D  ∇_w J     (ASCENT, plus)
  G:  θ  ←  θ  −  η_G  ∇_θ J     (DESCENT, minus)
                  └── only term 2 depends on θ
```

**Notice.** You will not see a literal plus in Colab. You will see `d_loss.backward(); d_optimizer.step()` on a BCE that already flipped the sign.

**Mini-check.** If you used `w ← w − η ∇J` for the discriminator, would $D$ get better or worse at the game?

---

## 7. Freeze the other player: two optimizers and `.detach()`

<a id="p7-detach-freeze"></a>

**Purpose.** One backward must not rewrite **both** rulebooks.

**Definition.** While $D$ trains, **keep $\theta$ constant**. While $G$ trains, **keep $w$ constant**. **Two Adam optimizers**: `g_optimizer` only owns `generator.parameters()`; `d_optimizer` only owns `discriminator.parameters()`. Calling `d_optimizer.step()` cannot change $\theta$. **`.detach()`** on `fake` during the $D$ step **cuts the tape** from $\hat{x}$ back into $G$, so even `d_loss.backward()` does not **compute** $\nabla_\theta$. On the $G$ step you **do not** detach: the gradient must flow **through $D$ into $G$**, but you still **do not step** $w$.

**Micro.** Two different locks:

| Lock | What it does | When |
|------|----------------|------|
| Two Adams + only one `.step()` | **Parameters** of the other net cannot change | both steps |
| `fake.detach()` | Autograd **does not even compute** $\nabla_\theta$ | $D$ fake path only |
| Live tape `D(G(z))` | Grad flows through $D$ into $G$ | $G$ step only |

Without detach, graph is `z → G → xhat → D → loss` — backward would touch $\theta$ **and** $w$. With detach, `xhat` is a **constant picture**. `zero_grad()` clears old grads so they do not accumulate from the previous batch.

**Analogy.** During the teller’s study hall, the forger’s pencils are **locked**. The teller may look at a forged slip, but must not sneak into the forger’s studio and “help” by moving the forger’s stamps. `.detach()` is locking the studio door. During the forger’s practice, the teller stands still (frozen $w$) so the forger can feel **which way to push the drawing** to fool this particular teller.

Someone asks: **on the teller’s turn, is the studio door locked?** If no, $G$ is being updated on $D$’s objective — the opposite of freeze-$\theta$. Losses still print. That is the #1 silent bug.

```
  D step:
    fake = G(z)
    D(fake.detach())     ← tape cut; θ not in graph
    d_optimizer.step()   ← only w moves

  G step:
    D(G(z))              ← tape through D into G
    g_optimizer.step()   ← only θ moves; w not stepped
```

**Notice.** Forgetting `.detach()` on extra $D$ steps in `five_disc_one_gen` is the same bug, four more times.

**Mini-check.** On the generator step, should `fake` be detached?

---

## 8. Tanh and `Normalize((0.5,), (0.5,))` share $[−1,1]$

<a id="p8-tanh-norm"></a>

**Purpose.** $G$’s last layer and the dataset transform must speak the same pixel language.

**Definition.** MNIST files are bytes 0–255. `ToTensor()` maps them to $[0,1]$. `Normalize((0.5,), (0.5,))` does $(x-0.5)/0.5$, so pixels land in **$[−1,1]$**. One value in the tuple because MNIST is **one channel**; RGB would need three means and three stds. **Tanh** on $G$’s last layer also lands in **$(-1,1)$**. Match. To **plot**, he **renormalizes**: `fake * 0.5 + 0.5` back toward $[0,1]$ for `imshow`. He says he will not teach plotting in depth.

**Micro.** Work one pixel:

| File byte | After ToTensor | After Normalize | Tanh end of $G$ |
|-----------|----------------|-----------------|-----------------|
| $0$ black | $0.0$ | $-1.0$ | near $-1$ |
| $128$ gray | $\approx 0.5$ | $\approx 0.0$ | near $0$ |
| $255$ white | $1.0$ | $+1.0$ | near $+1$ |

He says “mean 0.5 and **variance** 0.5.” In PyTorch the second argument is **std**, not variance. $(x-0.5)/0.5$ is what runs.

**Analogy.** The vault stores slips in **pencil gray 0 to 1**. The forger’s last stamp is a **signed ink** that only writes $-1$ to $+1$. If you forget Normalize, the teller sees vault slips in $[0,1]$ and forgeries in $[-1,1]$ — two different paper stocks — and “real vs fake” becomes “which scale was used,” not “does it look like a digit.” $D$ can get “high accuracy” from epoch 1 and the pictures still look like noise.

Someone asks: **must the press and the vault use the same gray scale?** Yes. That is this whole warm-up.

```
  file 0–255  →  ToTensor [0,1]  →  Normalize  [−1, 1]
                                              ▲
  z → … → Linear 784 → Tanh ──────────────────┘  same range

  plot:  (Tanh out) * 0.5 + 0.5  →  [0,1] for the picture
```

**Notice.** If $G$ ended in **Sigmoid** $[0,1]$ instead, you would **drop** Normalize (or both ends stay $[0,1]$). Mixing Tanh with un-normalized $[0,1]$ reals is the fail.

**Mini-check.** If $G$ ended in Sigmoid $[0,1]$ instead of Tanh, would `Normalize((0.5,), (0.5,))` still be the matching transform?

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
