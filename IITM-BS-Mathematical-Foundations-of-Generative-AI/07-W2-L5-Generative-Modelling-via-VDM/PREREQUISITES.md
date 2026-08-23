# Prerequisites — warm-up before this recording (MLP forward / backprop)

> **Do this first** if “chain rule,” “activation,” “gradient,” or “epoch” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> IITM BS · playlist **W2_L5** · TA live-board (Chandan).  
> **Beginner:** purpose · definition · micro · analogy · ASCII · notice · mini-check.

YouTube **title** says variational divergence minimization. This **recording** is **~55 minutes** (study it as a one-hour lab). The tablet is the **first DGM tutorial**: a tiny fully-connected net, a **forward pass**, **MSE**, **backprop** by the chain rule, then a **Python-looking training loop**. Warm-up matches **the board**, not the title.

```
  After this warm-up you can say:

  "Supervised data is pairs (x, y); we learn a map f."
  "A net is nested weighted sums, then a bend (activation)."
  "z is the mix; a = g(z) is the activation. Input nodes just hold x."
  "Sigmoid squashes to (0,1); ReLU is a hinge at 0."
  "Loss is a number that says how wrong ŷ is. Error need not hit zero."
  "A derivative asks: if I nudge this weight, how does loss move?"
  "Chain rule: multiply the slopes along the unique path (off-path nodes stay out)."
  "Gradient descent: step opposite the derivative, size α > 0."
  "A batch is a plate of examples; an epoch is one full pass; an iteration is one update."
```

```
  §1  Supervised pairs (x, y)     ──► Topic 1
  §2  Weighted sum + bend         ──► Topics 3–5
  §3  Sigmoid / ReLU              ──► Topic 5
  §4  Loss / MSE                  ──► Topic 6
  §5  Derivative as a nudge       ──► Topic 7
  §6  Chain rule                  ──► Topics 8–9
  §7  Gradient descent / α        ──► Topic 7
  §8  Batch vs epoch              ──► Topic 10
```

**One scene through all eight.** A **penalty kick**. You know where the ball *should* go (the label). You kick (forward pass). You miss (loss). You adjust plant-foot and hip (weights) a little **opposite** the miss (gradient descent), using the chain of joints from hip to ball (chain rule). You take shots in **packs** (batches) and walk the whole drill book **once** (an epoch).

```
  know the answer  ──►  kick  ──►  miss  ──►  adjust joints  ──►  kick again
  (y)                (forward)   (loss)     (backprop + GD)     (next batch)
```

**Walk one kick from whistle to notebook** (this is the whole lecture in one shot):

```
  coach notebook:  x = [wind, distance] = [1, 0]
                   y = where it SHOULD land          (known answer)
                         │
                         │  FORWARD  (Topics 3–5)
                         ▼
  mix z = (weights · x) + bias
  bend a = g(z)     g = sigmoid or ReLU
  ŷ = last a        ≈ 0.88 if σ(2) on his numbers
                         │
                         │  LOSS  (Topic 6)
                         ▼
  L = ½ (y − ŷ)²          miss, squared
                         │
                         │  BACK  (Topics 7–9)
                         ▼
  highlight the ONE hip-joint on the path
  multiply five local slopes  →  ∂L/∂w
  w ← w − α ∂L/∂w             α > 0, minus = downhill
                         │
                         │  LOOP  (Topic 10)
                         ▼
  next plate of kicks (batch)
  after every plate in the book: one epoch
  STOP: procedure ready. Next sitting = PyTorch.
```

If any box is foggy, the matching § below is the unpack.

---

## 1. Supervised pairs

<a id="p1-pairs"></a>

**Purpose.** He starts: you already know the right answer.

**Definition.** **Supervised learning:** each example is a pair $(x,y)$. $x$ is the **feature** (what you see). $y$ is the **target label** (the number you should have said). A finite list of $m$ such pairs is the training set $D$. **Regression** here: $y$ is a real number, not a class name. He assumes you know this before the MLP.

He also says data has **two parts**. ASR repeats “training”; at the **end** of the hour he grades a **test** split with a metric of the problem. Same idea as later Dataset lectures: practice pile vs sealed final.

**Micro.** Three kicks you already filmed:

```
  x = (wind, distance)     y = where the ball should land
  pair 1:  (2.0, 1.1)
  pair 2:  (0.5, 0.4)
  pair 3:  (3.1, 2.0)

  m = 3 examples in D
```

On his later numeric: $x$ is a **column** $\begin{bmatrix}1\\0\end{bmatrix}$ — two features, one example. The net never sees the English word “should.” It sees numbers in a real space.

**Analogy.** A coach with a notebook of past penalties. Each line: conditions → where it *did* belong. You are not inventing the sport; you are copying the notebook’s map.

Wrong move: hiding $y$ and still calling it supervised. That is a different course chapter.

```
  D = { (x1,y1), (x2,y2), ..., (xm,ym) }
           │  learn on TRAIN
           ▼
        ŷ = f(x)    so   f(x) ≈ y  on new x
           │
           ▼
        later: TEST pile, grade once  (end of this tape)
```

```
  human: "that's where it should have gone"
  net:   y = 1.1     (a number, not a sentence)
```

**Notice.** Generative AI will later drop $y$ and model $x$ itself. This tape is the **supervised warmup** the later nets sit on.

**Mini-check.** If someone hides $y$, is the task still supervised? What is $m$? Where does test data show up on this tape — the first minute or the last board?

---

## 2. A net is nested weighted sums, then a bend

<a id="p2-net"></a>

**Purpose.** The “function $f$” is an MLP: layers of nodes.

**Definition.** Each **node** computes a **weighted sum** of the previous layer, adds a **bias**, then applies an **activation** $g$. **Fully connected / multilayer perceptron:** every node in layer $\ell$ talks to every node in layer $\ell+1$. **Input nodes** just *hold* $x$ — no weights on the sockets themselves.

He draws **two features**, **one hidden layer with two nodes**, **one output**. You *may* stack more hidden layers; today one is enough.

**Micro.** Count knobs before biases: $2\times 2=4$ edges into hidden, $2$ edges into $\hat y$ → **six weights**. Plus three biases (two hidden + one output) if every non-input node has one.

```
  x = [ x¹ ]          orange sockets   (input nodes)
      [ x² ]

  x¹ ──w──► h1 ──w──► ŷ     blue = hidden, red = output
  x² ──w──► h2 ──w──┘
```

Hidden 1 is not “the answer.” It is an intermediate mix.

**Analogy.** A kitchen line: prep station (hidden) then plating (output). Each cook uses *every* ingredient from the previous counter (fully connected), with their own spoon sizes (weights) and a pinch of salt (bias). The pantry shelves (input nodes) do not have spoons.

```
  layer ℓ outputs  ──weighted mix──►  z   ──g──►  layer ℓ+1 outputs
                                          └── activation (the bend)
```

```
  WRONG:  treat input circles as weights
  RIGHT:  weights live on EDGES; inputs are sockets
```

**Notice.** Letter $a$ on the board = **activation**. Layer number $i$, node number $j$. Edges wait for Topic 4’s $w_{i,j}^{k}$.

**Mini-check.** If the hidden layer has two nodes and the input has two features, how many edges sit between those two layers? (Count: $2\times 2=4$.) Do input nodes have their own $w$?

---

## 3. Sigmoid and ReLU

<a id="p3-act"></a>

**Purpose.** He writes $g$ as sigmoid or ReLU. The mix before $g$ is **$z$** (same indices as $a$).

**Definition.** **Activation** $g$: a fixed bend after the sum $z$. **Sigmoid:** $\sigma(z)=1/(1+e^{-z})$, always in $(0,1)$. **ReLU:** $0$ if $z\le 0$, else $z$ (a hinge). **Tanh** exists; he does not dwell. Why they exist: **smoothness**, or to model **more complex** input–output maps.

**Micro.** $z=2$: $\sigma(2)\approx 0.88$ (the number he computes on the board). $z=-3$: $\sigma$ near $0$, ReLU is $0$. $z=4$: ReLU is $4$, sigmoid is still $<1$.

Tiny table:

```
   z      sigmoid σ(z)     ReLU
  -3      ≈ 0.05            0
   0       0.50             0
   2      ≈ 0.88            2
   4      ≈ 0.98            4
```

**Analogy.** Sigmoid is a squash bottle: huge $z$ still pours at most 1. ReLU is a one-way door: negative $z$ is blocked at 0, positive $z$ walks through unchanged.

```
  sigmoid:   0 ──────── ● ──────── 1     (never leaves (0,1))
  ReLU:      ____●/                   0 until z=0, then a 45° ramp
```

```
  z  (the mix:  w·incoming + b)
  a = g(z)     (the bend)
  ŷ = last a
```

**Notice.** He prefers the **same** $g$ on every node. Mix-and-match is allowed in theory; not today’s drill. Convention on the board: **weight first**, then the incoming activation.

**Mini-check.** What is ReLU$(-2)$? What interval does sigmoid live in? Is $z$ the prediction?

---

## 4. Loss / MSE

<a id="p4-loss"></a>

**Purpose.** “Minimize some error” needs a formula. He says the error **will reduce**; it **will not become zero**.

**Definition.** A **loss** $L(y,\hat y)$ is a number that is small when $\hat y$ matches $y$. **Mean squared error (MSE)** for one point, as he writes:

$$
L=\frac12(y_i-\hat y_i)^2
$$

Many points: sum the $L$’s and divide by how many (an average). **Cross-entropy** is the usual classification loss; he names it, then stays on MSE. Generative work will later use **reconstruction** and **KL**; not this hour’s formula.

**Micro.** $y=1$, $\hat y=0.88$: $L=\frac12(0.12)^2=0.0072$. $y=1$, $\hat y=0.1$: $L=\frac12(0.9)^2=0.405$. Farther $\hat y$ → bigger $L$. Two points with those losses: average $=(0.0072+0.405)/2=0.2061$.

**Analogy.** The miss distance to the goal, squared so overshoot and undershoot both count, times $1/2$ so the derivative is tidy (the $2$ from $x^2$ cancels).

Wrong: calling a perfect $L=0$ the goal of this tape. He says it will **not** become zero.

```
  y = 1.0     ŷ = 0.88     miss = 0.12     L = 0.0072
  y = 1.0     ŷ = 0.10     miss = 0.90     L = 0.405

  one point  →  that L
  m points   →  (L1 + … + Lm) / m
```

**Notice.** He assumes **one** training point for the derivative, then says: if you have many, average. $f$ could be linear, polynomial, or a net; today it is the MLP.

**Mini-check.** If $\hat y=y$, what is MSE? Why the $1/2$? Did he promise $L=0$?

---

## 5. A derivative is a nudge question

<a id="p5-deriv"></a>

**Purpose.** He will not write Lagrange multipliers. He will write $\partial L/\partial w$.

**Definition.** $\partial L/\partial w$ asks: **if I increase this one weight a tiny bit, holding the others fixed, which way does the loss move, and how steeply?** Positive slope: raising $w$ *increases* $L$ (bad). Negative: raising $w$ *decreases* $L$.

**Micro.** Suppose $\partial L/\partial w=+0.4$. Nudge $w$ up by $0.1$ → loss goes up about $0.04$. You wanted loss *down*, so you should move $w$ the other way.

**Analogy.** A volume knob and a noise meter. Turn the knob a hair; watch the meter. That ratio is the derivative.

Second picture: you cannot nudge **$x$**. $x$ is the kick already taken. Only $\theta$ (weights and biases) is a knob.

```
  w  ----nudge +ε----►  L changes by ≈ (∂L/∂w)·ε

  WRONG:  edit the dataset x to shrink L
  RIGHT:  only θ moves
```

**Notice.** “How much does a change in this weight affect the loss?” is his spoken intuition. The five-factor product later *computes* that number.

**Mini-check.** If $\partial L/\partial w$ is negative, do you increase or decrease $w$ to reduce $L$? Can you gradient-descend on $x$?

---

## 6. Chain rule

<a id="p6-chain"></a>

**Purpose.** One weight sits several hops from the loss. He traces a **path** first, then multiplies.

**Definition.** If $L$ depends on $\hat y$, which depends on $z$, which depends on $a$, which depends on $w$, then

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial\hat y}\cdot
\frac{\partial\hat y}{\partial z}\cdot
\frac{\partial z}{\partial a}\cdot
\frac{\partial a}{\partial w}
$$

Multiply the slopes **along the unique path** from that weight to the loss. Links *off* the path (the other hidden node) do not enter **this** product. He assumes you know this before he highlights $w_{1,1}^1$.

**Micro.** His actual five hops for $w_{1,1}^1$:

```
  w¹₁₁ → z²₁ → a²₁ → z³₁ → ŷ → L
         (x¹)   (σ)   (w²₁₁) (σ)
```

Hidden-2 is a spare water bottle: on the bike, **not** on this chain.

**Analogy.** A bicycle: pedal → chain → wheel → ground. The miss at the goal depends on the pedal through **that** chain.

Wrong: multiplying by the off-path hidden node “because it is in the net.”

```
  ON path:   w¹₁₁  →  a²₁  →  ŷ  →  L
  OFF path:  a²₂                    (different weight’s story)
```

**Notice.** He does this by **hand** for **one** weight. Autograd will do every weight later. This tape is the picture autograd implements.

**Mini-check.** If a hidden node is not on the highlighted path, does its activation appear in this weight’s product? How many factors does he write for $w_{1,1}^1$?

---

## 7. Gradient descent and learning rate

<a id="p7-gd"></a>

**Purpose.** Once you have $\partial L/\partial w$, you step.

**Definition.** **Gradient descent (GD):**

$$
w \leftarrow w - \alpha\,\frac{\partial L}{\partial w}
$$

Spoken: **new weight = previous weight minus learning rate times the derivative.** **Learning rate** $\alpha$ is a **positive** real (step size). Too big: you jump over the valley. Too small: you crawl. He says choosing $\alpha$ takes **knack**; later tutorials.

**Micro.** $w=1$, $\partial L/\partial w=+0.5$, $\alpha=0.1$ → new $w=1-0.05=0.95$. You moved *downhill*. $\alpha=50$ with the same slope → new $w=1-25=-24$ (a leap, not a step).

**Analogy.** Walk opposite the slope on a hill, with stride $\alpha$. The **minus** sign is “opposite.”

```
  wrong:  w ← w + α ∂L/∂w     (climb the loss)
  right:  w ← w − α ∂L/∂w     (descend)

  α ∈ R⁺     “how big a step”
```

**Notice.** The same $\alpha$ is used for every weight in his story. Fancy per-weight schemes (Adam) are later. He will **not** write a general optimization problem — just this update.

**Mini-check.** Why must $\alpha$ be positive? What happens if you drop the minus? Is $\alpha$ a weight?

---

## 8. Batch vs epoch

<a id="p8-epoch"></a>

**Purpose.** The last board is a Python loop. That *is* the coding of this tape (no Colab).

**Definition.** **Batch** of size $b$: a plate of $b$ examples processed together before one update. If you have $m$ examples, you get about $k=m/b$ plates $d_1,\ldots,d_k$ (last plate may be short if $m$ is not divisible by $b$ — he says fine). **Epoch:** one full walk through every plate. **Iteration:** one update (one plate). `max_epoch` is how many times you reread the whole book.

Then: **test** data, predict, measure with a **metric of the problem**. Mix-and-match later; overall structure stays.

**Micro.** $m=128$, $b=32$ → $k=4$ batches per epoch. 10 epochs → 40 updates. $m=100$, $b=64$ → one full plate of 64 plus a short plate of 36.

**Analogy.** Flashcards in packs of 32. Going through every pack once is an epoch. Flipping one pack is an iteration. Then a sealed final (test).

```
  for e in range(max_epoch):      # epochs = full passes
      for di in D:                # batches = plates
          X, y = di
          ŷ = f_θ(X)              # forward
          L(y, ŷ)                 # loss
          obtain gradients        # five-factor, every w
          update weights          # minus α
```

```
  m=128, b=32, max_epoch=10

     epoch 1:  d1 d2 d3 d4     ← 4 iterations
     ...
     epoch 10: d1 d2 d3 d4     ← 40 iterations total
```

**Notice.** He writes `for e in range(max_epoch)` assuming you know Python. Calling that header an “iteration” is the trap.

**Mini-check.** If $m=100$, $b=64$, how many batches (allow a short last one)? Is “iteration” the same word as “epoch” on his recap? What happens after training on his last board?

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
