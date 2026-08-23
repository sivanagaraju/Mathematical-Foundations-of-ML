# W2_L5 — Generative modelling via variational divergence minimization  
*(this recording: MLP forward pass and backprop)*

> **Video:** [W2_L5: Generative modelling via variational divergence minimization](https://www.youtube.com/watch?v=stZC0Zk5KYo) · **~55 min**  
> **Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md) · **Quiz:** [quiz.html](./quiz.html)

**Do PREREQUISITES before this article.**  
**Course:** IIT Madras B.S. · **BSDA5002** · Prof. Prathosh A. P. · tutorial live-board (Chandan)  
**Previous in this playlist:** [W1_L4 $f$-divergence](../03-W1-L4-Variational-Divergence-Minimization/NOTES.md) (the *title’s* math) · [W1_T2 Dataset/DataLoader](../04-W1-T2-Introduction-to-PyTorch-Tensors/NOTES.md)

**Honest title note:** YouTube says **variational divergence minimization**. The 55-minute tablet he actually opens is **Tutorial 1**: a 2–2–1 MLP, **forward pass**, **MSE**, **chain-rule backprop** for one weight, then a **Python-shaped** batch/epoch loop. Next sitting, he says, is **PyTorch**. These notes follow the **speech and the board**. The actual VDM algorithm (conjugate, critic, saddle) is [W1_L4](../03-W1-L4-Variational-Divergence-Minimization/NOTES.md) plus NPTEL Lec 04 — see Sources.

| When he hits… | Warm-up |
|---------------|---------|
| Pairs $(x,y)$ | [p1-pairs](./PREREQUISITES.md#p1-pairs) |
| Hidden nodes | [p2-net](./PREREQUISITES.md#p2-net) |
| Sigmoid / ReLU | [p3-act](./PREREQUISITES.md#p3-act) |
| MSE | [p4-loss](./PREREQUISITES.md#p4-loss) |
| $\partial L/\partial w$ | [p5-deriv](./PREREQUISITES.md#p5-deriv) |
| Path / chain | [p6-chain](./PREREQUISITES.md#p6-chain) |
| $w\leftarrow w-\alpha\partial L/\partial w$ | [p7-gd](./PREREQUISITES.md#p7-gd) |
| Epoch loop | [p8-epoch](./PREREQUISITES.md#p8-epoch) |

---

## Table of Contents

1. [Topic 1 — Supervised setup: $D$, $f$, loss](#topic-1-supervised-setup-d-f-loss-0011–0541) (00:11–05:41)
2. [Topic 2 — Football; forward vs backprop](#topic-2-football-forward-vs-backprop-0541–0854) (05:41–08:54)
3. [Topic 3 — MLP diagram; names $a$](#topic-3-mlp-diagram-names-a-0854–1306) (08:54–13:06)
4. [Topic 4 — Weight indices](#topic-4-weight-indices-1306–1606) (13:06–16:06)
5. [Topic 5 — Forward pass: $z$, sigmoid, ReLU](#topic-5-forward-pass-z-sigmoid-relu-1606–2228) (16:06–22:28)
6. [Topic 6 — $\theta$ and MSE](#topic-6-θ-and-mse-2228–2743) (22:28–27:43)
7. [Topic 7 — Gradient descent](#topic-7-gradient-descent-2743–3320) (27:43–33:20)
8. [Topic 8 — Path of $w_{1,1}^1$; $\sigma(2)\approx 0.88$](#topic-8-path-of-w111-σ2--088-3320–3858) (33:20–38:58)
9. [Topic 9 — Five-factor product](#topic-9-five-factor-product-3858–4818) (38:58–48:18)
10. [Topic 10 — Batch, epoch, Python loop](#topic-10-batch-epoch-python-loop-4818–5453) (48:18–54:53)
11. [External references](#external-references)
12. [Apply it (scenarios)](#apply-it-scenarios)
13. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

You hold labeled pairs and want a map $f$ whose $\hat y$ sits close to $y$. This recording (title: VDM) installs a **2–2–1 MLP**: **forward**, **MSE**, **chain-rule** on one weight, **gradient descent**. Nested **epoch / batch** loops finish the procedure; PyTorch is next.

**Worldview arc:** from “minimize some error on $D$” **to** “multiply five partials for one $w$, then loop plates of data.”

### The approach

```
  1. SETUP     D = {(x_i, y_i)};  ŷ = f(x);  minimize L(y, ŷ)
  2. NET       2 inputs → 2 hidden → 1 output   (fully connected)
               names: a (activation), w_{i,j}^{k} (dest, src, prev layer)
  3. FORWARD   z = weighted sum + bias
               a = g(z)     g = sigmoid or ReLU
               ŷ = last a
  4. LOSS      L = (1/2)(y − ŷ)^2     (one point; average if many)
  5. BACK      highlight path of one w
               ∂L/∂w = product of local slopes on that path
               w ← w − α ∂L/∂w
  6. LOOP      for epoch:
                 for batch in D:
                   forward; loss; grads; update
  STOP         procedure ready; next = PyTorch
```

### Whole-sheet recipe (commented 2–2–1)

```python
# Transcribed from the tablet. He did not run Colab; this is the hour in code.
import math

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))   # Topic 5: σ(z) = 1/(1+e^{-z})

# --- Topic 5 FORWARD: x has two features; g = sigmoid throughout ---
def forward(x1, x2, W, b):
    # Hidden node 1  (a²₁)
    z21 = W["w111"] * x1 + W["w121"] * x2 + b["b11"]
    a21 = sigmoid(z21)
    # Hidden node 2  (a²₂)
    z22 = W["w211"] * x1 + W["w221"] * x2 + b["b12"]
    a22 = sigmoid(z22)
    # Output ŷ = a³₁
    z31 = W["w211_out"] * a21 + W["w221_out"] * a22 + b["b21"]
    yhat = sigmoid(z31)
    return yhat, a21, a22, z21, z22, z31

def mse(y, yhat):
    return 0.5 * (y - yhat) ** 2        # Topic 6: one point; average if many

# --- Topic 9 BACK: five factors for ONE weight w¹₁₁ (purple path) ---
def dL_dw111(y, yhat, a21, W, x1):
    f1 = (y - yhat)                     # as he writes ∂L/∂ŷ
    f2 = yhat * (1 - yhat)              # sigmoid' at output
    f3 = W["w211_out"]                  # ∂z³₁/∂a²₁
    f4 = a21 * (1 - a21)                # sigmoid' at hidden-1
    f5 = x1                             # ∂z²₁/∂w¹₁₁  (this w multiplies x¹)
    return f1 * f2 * f3 * f4 * f5

# Topic 8 numbers: x=[1,0], all w,b = 1  →  a21 = σ(2) ≈ 0.88

# --- Topic 10 LOOP ---
# D = {d1, ..., dk}, k = m/b  (last plate may be short)
for e in range(max_epoch):              # epoch = one full pass
    for di in D:                        # batch = one plate
        X, y = di
        yhat, a21, a22, *_ = forward(X[0], X[1], W, b)
        loss = mse(y, yhat)
        W["w111"] = W["w111"] - alpha * dL_dw111(y, yhat, a21, W, X[0])
        # similarly every other knob in θ  (Topic 9: "repeat for all weights")
# Then test split + a metric of the problem. Next sitting: PyTorch.
```

### System context

```
  ╔════════════════════════════════════╗
  ║ W1_L2/L4: estimate p_x, f-div      ║  (title of THIS video)
  ║ W1_T2/T4: Dataset, nn.Module       ║  (next tutorials)
  ╚════════════════╤═══════════════════╝
                   │ this 55 min (playlist index 9)
                   ▼
        ┌──────────────────────────┐
        │ MLP by hand              │
        │ forward + backprop + GD  │
        └──────────────────────────┘
```

### Main blueprint

```
  D = {(x_i, y_i)}                 known answers (supervised)
       │
       ▼
  ┌─ FORWARD ─────────────────────┐
  │  x → z → g(z)=a → … → ŷ      │
  │  ŷ = f_θ(x)                   │
  └──────────┬────────────────────┘
             │ compare
             ▼
  L = ½ (y − ŷ)²
             │
             ▼
  ┌─ BACK ────────────────────────┐
  │  highlight path of one w      │
  │  ∂L/∂w = 5 local slopes       │
  │  w ← w − α ∂L/∂w              │
  └──────────┬────────────────────┘
             │ repeat on batches
             ▼
  for e in epochs:
      for di in D:  forward; L; grads; update
             │
  ┌ · · · · ┴ · · · · · · ┐
  │ STOP: procedure ready │
  │ next: PyTorch         │
  └ · · · · · · · · · · · ┘
```

### Scenario walkthrough

$x_i=[1,0]^\top$, every weight and bias equal to $1$. Hidden $a^2_1=\sigma(2)\approx 0.88$. Highlight $w_{1,1}^1$: the path is that edge → hidden-1 → output → loss (hidden-2 is off-path). Multiply five partials, step $w$ by $-\alpha$ times the product, then do the same for every other knob in $\theta$.

### Failure / contrast

```
  WRONG  expect f-GAN / conjugate / critic in this 55 min   (title trap)
  WRONG  try to change x to reduce L                         (x is data)
  WRONG  skip highlighting the path before chaining
  WRONG  call one batch update an "epoch"
  WRONG  drop the minus in w ← w − α ∂L/∂w
```

### STOP / out of scope

- PyTorch `nn.Module`, autograd, `backward()` (next tutorials).
- Conjugate $f^*$, critic $T$, minmax VDM / GAN (the **title**; see W1_L4 and NPTEL Lec 04).
- Adam, momentum, learning-rate schedules (he flags $\alpha$ as later knack).

### Load-bearing claims (closed-book)

- Supervised: $D$ of pairs; $\hat y=f(x)$; minimize $L$.
- Forward = input→output; backprop = error→weights.
- 2–2–1 MLP; $a$ names activations; $w_{i,j}^{k}$ = dest, source, previous layer.
- $z$ then $g$; sigmoid $1/(1+e^{-z})$; ReLU hinge.
- MSE $\frac12(y-\hat y)^2$; $\theta$ is all knobs; $x$ is not a knob.
- GD: $w\leftarrow w-\alpha\partial L/\partial w$, $\alpha>0$.
- Five-factor product for $w_{1,1}^1$; $\sigma(2)\approx 0.88$ on his numbers.
- Nested loops: epoch, then batch; iteration ≠ epoch.

**Speaker:** IITM BS tutorial (Chandan) · course Prof. Prathosh.

---

## Topic 1: Supervised setup: $D$, $f$, loss (00:11–05:41)

### Where this sits on the master map

This is the **JOB** box. [Labeled pairs](./PREREQUISITES.md#p1-pairs) and a map $f$ whose miss we will later differentiate. He calls it the first tutorial of the DGM course — MLP, forward, backprop — and assumes you already know a fully-connected net and **regression** (predict a **number**).

### Board / screenshot

![Topic 1 — MLP/FC, regression, (x_i, y_i), ŷ = f(x), loss L](./screenshots/composites/ch01-topic-01-supervised-setup-d-f-loss-panel1of1.png)

Caption: top-right, assumed background: MLP / fully connected, regression. Training data as two blobs (input vs ?). Bottom: $x_i$ is the input feature, $y_i$ the target label, goal “minimize some error,” then $\hat y_i=f(x_i)$ and $L(y_i,\hat y_i)$.

### What he is establishing

He welcomes you to the **first tutorial** in the deep generative models course. The hour is a **simple MLP**, how the **forward pass** happens, and how **backprop using the error** happens. Two assumptions: you know the basic structure of an MLP (fully connected network), and you know **regression** — supervised learning where the thing you predict is a **number**. Data, he says, has **two parts**; ASR repeats “training,” and the **last** board grades **test**.

In supervised learning you **know the right answer**. Data splits into training (and later test). Training is how you understand the mapping from input to output. Call that mapping **function $f$**. Represent training data as $D$: $m$ pairs $(x_i,y_i)$, $x_i$ a feature, $y_i$ a target label. For regression both live in a real space.

The **goal** of the whole supervised game: **minimize some error** using the training data. Picture: input $x_i$ into a neural net (preferably an MLP) and a prediction $\hat y_i$. You can write $\hat y_i=f(x_i)$. The **loss** $L$ sits between the actual target $y_i$ and that prediction. That is the standard setting of any supervised task.

The trap is waiting 55 minutes for $f$-divergence. You will not see it. This is the **supervised engine** later generators still use.

You can now write $D$, $f$, $\hat y$, $L$. What is still open: a picture of “error” that is not an equation.

### Analogy for this topic only

A notebook of past penalties: wind and distance ($x$) paired with where the ball **should** have gone ($y$). You are asked to copy that map, not to invent football.

Someone asks: **do I need the answer key?** Yes — that is what “supervised” means on this tape.

In lecture words: $D$ of pairs, learn $f$, minimize $L(y,\hat y)$.

### Local picture

```
  D = {(x1,y1), ..., (xm,ym)}
           │  learn
           ▼
        ŷ = f(x)
           │
           ▼
        L(y, ŷ)   ← make this small

  Notice: y is known. That is the whole point of “supervised.”
```

### Bridge

A formula for $L$ will come. First he wants you to *feel* a miss — a football that does not hit the post.

---

## Topic 2: Football; forward vs backprop (05:41–08:54)

### Where this sits on the master map

This is **TWO PASSES**. [Nudge intuition](./PREREQUISITES.md#p5-deriv) starts here as a picture, before derivatives.

### Board / screenshot

![Topic 2 — miss arc; forward i/p→o/p; backprop error→weights](./screenshots/composites/ch02-topic-02-football-forward-vs-backprop-panel1of1.png)

Caption: left, a kick that should go to the post and a miss; the gap is error. Right: $x_i\to\mathrm{NN}\to\hat y_i$; forward pass is input to output; backward pass / backprop is from the error to the weights.

### What he is establishing

A football field, a penalty (or hockey — he is not precious). The **actual trajectory** should hit the goalpost. You go off a bit and hit somewhere else. That **deviation is the error**. The aim is to reduce it by practicing on the training data. The error **will reduce**. It **will not become zero**.

Given the input you predict — that process is the **forward pass**. From the outcome you calculate the error and try to update things — **backward pass / backprop**. Short slogans on the board: forward is **from input to output**; backprop is **from error to weights**. Each model has weights; you update them.

The trap is treating “backprop” as a PyTorch button. Today it is a **direction**: error flows back into knobs.

You can now name the two passes. What is still open: a net small enough to name every knob.

### Analogy for this topic only

Intended path to the post versus the actual miss. Practice shrinks the miss; perfect zero is not promised.

Someone asks: **is the miss the same as the weight?** No. The miss is $L$. The weights are how you kick next time.

In lecture words: forward input→output; backprop error→weights.

### Local picture

```
  intended  ●━━━━━━━━━━●  post
  actual    ●━━━━●  miss = error

  x ──FORWARD──► ŷ ──L──► error ──BACK──► weights

  Notice: error need not hit zero. He says so out loud.
```

### Bridge

Slogans are not a computation. He draws a 2–2–1 fully connected net so every edge can get a name.

---

## Topic 3: MLP diagram; names $a$ (08:54–13:06)

### Where this sits on the master map

This is **NET**. [Weighted sum + bend](./PREREQUISITES.md#p2-net). Why now: you cannot differentiate a cloud; you need nodes.

### Board / screenshot

![Topic 3 — two inputs, two hidden, one output; a = activation](./screenshots/composites/ch03-topic-03-mlp-diagram-activation-names-panel1of1.png)

Caption: orange $x_i=\begin{bmatrix}x_i^1\\x_i^2\end{bmatrix}$ into **input nodes**; blue **hidden**; red **output** $\hat y_i$. Fully connected (crossed purple, then green). Bottom-right: $a$ with $i$ = **layer number**, $j$ = **node number**. Top-left pane is still empty — read the other three.

### What he is establishing

Take $x_i$ with **two features**, $x_i^1$ and $x_i^2$. For example, two numbers you can hold. **Input nodes** just hold those values — no weights on the input itself. Then **hidden nodes**. He takes **one hidden layer with two nodes** (you may stack more layers; today one is enough). Then a simple **output** $\hat y_i$.

**Fully connected / multilayer perceptron:** everything from the previous layer connects to everything in the next. Two inputs × two hidden nodes is **four** edges into the hidden layer, then **two** edges into $\hat y$ — six knobs before you even count biases.

Names: each node is an **activation** $a$, with a **layer number** and a **node number**. First layer first node, first layer second node, second layer first and second, third layer first (the output). He uses $a$ *because it is an activation*.

The trap is calling the input circles “weights.” They are sockets. Weights live on the **edges**.

You can now draw 2–2–1 and label $a$. What is still open: how to index an edge.

### Analogy for this topic only

Two ingredients, two prep cooks, one plate. Every prep cook tastes both ingredients. The plate tastes both prep cooks.

Someone asks: **can I add a second hidden layer?** Yes — he said so — not today.

In lecture words: input nodes, two hidden, one output; $a$ with layer and node.

### Local picture

```
  x¹  →  ○ ──┐
             ├──►  ○ (hidden) ──┐
  x²  →  ○ ──┤                  ├──►  ○  →  ŷ
             └──►  ○ (hidden) ──┘

  Notice: four edges in, two edges out. Fully connected.
```

### Bridge

Those colored edges need a systematic name or the chain rule will drown in “this one” and “that one.”

---

## Topic 4: Weight indices (13:06–16:06)

### Where this sits on the master map

This is **INDEXING**. Same net, now every edge has $w_{i,j}^{k}$.

### Board / screenshot

![Topic 4 — w_{i,j}^{k}: dest, source, prev layer](./screenshots/composites/ch04-topic-04-weight-indices-wij-k-panel1of1.png)

Caption: $k$ = previous layer, $i$ = destination node, $j$ = source node. First expansion of $a^2_1$ as a weighted mix of $x_i^1,x_i^2$ plus bias (activation $g$ still to come).

### What he is establishing

Weights are $w$. Consider a cable from input 1 into hidden 1. His notation $w_{i,j}^{k}$: **$k$** is the **previous layer**; **$i$** is the **destination** node; **$j$** is the **source** node. Concrete: dest 1, source 1, previous layer 1 is the edge from the first input into the first hidden node — $w_{1,1}^{1}$, the purple cable he will differentiate. The other three hidden-layer edges get the remaining $(i,j)$ pairs; the output edges live at layer $k=2$.

**Biases** are there too; he names them when he writes the activation formula. On an **input** node, the activation *is* the feature: $a^1_1=x_i^1$.

The trap is swapping dest and source. Swap them and the chain in Topic 9 points at the wrong $x$.

You can now read $w_{1,1}^{1}$ out loud: “into hidden-1, from input-1, previous layer 1.” What is still open: the formula that uses those names.

### Analogy for this topic only

A labeled cable: “from this wall socket into that lamp, on this floor.” Unplug the label and you cannot trace the miss.

Someone asks: **is the floor number the lamp’s floor?** He says previous layer — the socket you came from.

In lecture words: dest $i$, source $j$, previous layer $k$.

### Local picture

```
  w_{i, j}^{k}
     │  │  └── previous layer
     │  └───── source node
     └──────── destination node

  Notice: input activations equal the features. No extra w on the raw x sockets.
```

### Bridge

A name is not a computation. Next: $z$, then $g$, then $\hat y$.

---

## Topic 5: Forward pass: $z$, sigmoid, ReLU (16:06–22:28)

### Where this sits on the master map

This is **FORWARD**. [Activation](./PREREQUISITES.md#p3-act). This *is* the forward pass he promised.

### Board / screenshot

![Topic 5 — a = g(w x + b); sigmoid; ReLU; ŷ = a³₁](./screenshots/composites/ch05-topic-05-forward-pass-sigmoid-relu-panel1of1.png)

Caption: $a^2_1=g(w_{1,1}^1 x_i^1+w_{1,2}^1 x_i^2+b_1^1)$. Sigmoid $\sigma(z)=1/(1+e^{-z})$. ReLU $0$ if $z\le 0$ else $z$. Same pattern for $a^2_2$ and output $a^3_1=\hat y_i$. $\theta$ = all parameters.

### What he is establishing

How is $a^2_1$ computed? Suppose two incoming numbers. Mix the two inputs with their weights, **plus a bias** $b$, then pass through activation **$g$**:

$$
a^2_1 = g\big(w_{1,1}^1 x_i^1 + w_{1,2}^1 x_i^2 + b_1^1\big)
$$

The inner sum is **$z$** (same indices as $a$). **Sigmoid:** if the input to $g$ is $z$, then $1/(1+e^{-z})$. **ReLU:** $0$ when $z\le 0$, else $z$. Why activations: **smoothness**, or to model **more complex** input–output maps. **Tanh** and others exist; he will not go through every one.

People prefer the **same** activation throughout. $a^2_2$ is the same shape with the other dest index. Preview of Topic 8’s numbers: if $x=(1,0)$ and every weight/bias is $1$, the first hidden $z$ is $1+0+1=2$, and sigmoid of that is about $0.88$. The output

$$
a^3_1 = g\big(w_{1,1}^2 a^2_1 + w_{1,2}^2 a^2_2 + b_1^2\big)
$$

**is** $\hat y_i$. That whole walk, if you have more layers, is **forward propagation**. Inputs are **fixed** (they are data). If you want to reduce error you modulate **weights**.

```python
# Topic 5 — one example, sigmoid throughout (weight first, then incoming)
import math
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

# a²₁ = g(w¹₁₁ x¹ + w¹₁₂ x² + b¹₁)     board ~16:06
z21 = w111 * x1 + w121 * x2 + b11
a21 = sigmoid(z21)
# a²₂ analogously
z22 = w211 * x1 + w221 * x2 + b12
a22 = sigmoid(z22)
# ŷ = a³₁ = g(w²₁₁ a²₁ + w²₁₂ a²₂ + b²₁)
z31 = w2_11 * a21 + w2_12 * a22 + b21
yhat = sigmoid(z31)
```

**What this does:** one forward pass. ReLU instead of sigmoid is `max(0, z)`. He prefers the same $g$ on every node.

The trap is stopping at $z$ and calling it the prediction. The prediction is $g(z)$ at the **last** node.

You can now write the three $a$ formulas. What is still open: packing knobs as $\theta$ and scoring $\hat y$ with MSE.

### Analogy for this topic only

Each cook: weigh both ingredients, pinch of salt, then a fixed “style” (squash or hinge). The plate is the last cook’s dish, not the prep bowl.

Someone asks: **must hidden and output use the same $g$?** Preferred, not a law on this tape.

In lecture words: $z$ then $g$; sigmoid and ReLU; $\hat y$ is the last $a$.

### Local picture

```
  z²₁ = w¹¹₁ x¹ + w¹₁₂ x² + b¹₁
  a²₁ = g(z²₁)

  z³₁ = w²₁₁ a²₁ + w²₁₂ a²₂ + b²₁
  ŷ   = a³₁ = g(z³₁)

  Notice: convention “weight first, then the incoming activation.”
```

### Bridge

You cannot change $x$. All the knobs together get one name, and the miss gets a formula.

---

## Topic 6: $\theta$ and MSE (22:28–27:43)

### Where this sits on the master map

This is **LOSS**. [MSE](./PREREQUISITES.md#p4-loss). Forward is done; now score it.

### Board / screenshot

![Topic 6 — MSE ½(y−ŷ)²; one training point; minimize by a derivative](./screenshots/composites/ch06-topic-06-theta-mse-one-point-panel1of1.png)

Caption: $L=\frac12(y_i-\hat y_i)^2=\frac12(y_i-f_\theta(x_i))^2$. Assuming only one training point. “This is what we want to minimize.” Later: we minimize a function by a derivative wrt what we can change.

### What he is establishing

Inputs are fixed. Say the two features are already on the page. Every weight (and bias) that *can* change is listed as **$\theta$**. Then $\hat y_i=f_\theta(x_i)$. That $f$ could be linear regression, polynomial, or a neural net — here it is the MLP.

**Loss depends on the task.** Classification: often **cross-entropy**. Reconstruction (most generative cases he flags): **MSE** / reconstruction losses; also **KL**. Today: a **simple** loss, **mean squared error**:

$$
L=\frac12\big(y_i-\hat y_i\big)^2
$$

**One data point.** Micro: $y=1$, $\hat y=0.88$ → $L=\frac12(0.12)^2=0.0072$. If you have many, **accumulate, sum, divide by the number of points** — the average. That average is what you minimize.

The trap is changing $x$ to shrink $L$. You cannot. Only $\theta$.

You can now write MSE for one row. What is still open: *how* a derivative becomes a step.

### Analogy for this topic only

Miss distance to the post, squared so both sides of the post count, times $1/2$ so the later derivative is clean. One kick today; many kicks → average the misses.

Someone asks: **is CE illegal?** No — wrong task today.

In lecture words: $\theta$ is all parameters; MSE on one point; average if many.

### Local picture

```
  θ = { every w and b }
  ŷ = f_θ(x)          x frozen
  L = ½ (y − ŷ)²      one point
  many points → mean of those L's

  Notice: he names CE and KL as other menus, then stays on MSE.
```

### Bridge

Minimizing a function, in this course’s first tutorial, means a derivative with respect to a weight, then a step.

---

## Topic 7: Gradient descent (27:43–33:20)

### Where this sits on the master map

This is **UPDATE**. [GD and $\alpha$](./PREREQUISITES.md#p7-gd).

### Board / screenshot

![Topic 7 — backprop with GD; learning rate in R⁺; pick w¹₁₁](./screenshots/composites/ch07-topic-07-gradient-descent-learning-rate-panel1of1.png)

Caption: back propagation with GD. Learning rate $\alpha\in\mathbb{R}^+$. Then: consider $w_{1,1}^1$ to understand how loss changes.

### What he is establishing

You minimize a function by computing its **derivative with respect to what you can optimize**. Imagine a volume knob and a noise meter. You cannot modify the input. So: derivatives wrt **parameters**. For a specific weight $w_{i,j}^{k}$ (from node $j$ of layer $k$ into node $i$ of layer $k+1$), find $\partial L/\partial w$. Then **gradient descent**:

$$
w \leftarrow w - \alpha\,\frac{\partial L}{\partial w}
$$

```python
# Topic 7 — spoken: previous weight minus learning rate times the derivative
# alpha is a positive real; choosing it takes knack (later tutorials)
w = w - alpha * dL_dw
# WRONG: w = w + alpha * dL_dw   # climbs the miss
```

**$\alpha$ is the learning rate**, a **positive real**. Choosing it takes **knack**; coming tutorials. Intuition of the derivative: if you **change this parameter**, **how does it affect the loss** — rate of change of loss with respect to the weight.

He will not write a general optimization problem. He will **pick one weight**, $w_{1,1}^1$, and grind the derivative by hand. Micro of the step: if that weight is $1$, the derivative is $0.5$, and $\alpha=0.1$, the new weight is $1-0.05=0.95$.

The trap is $w\leftarrow w+\alpha\partial L/\partial w$ (climbing the miss).

You can now write the update. What is still open: the **path** from that one edge to $L$.

### Analogy for this topic only

Stride a positive step **opposite** the slope. If turning the knob up makes the miss worse, turn it down.

Someone asks: **can the step size be negative?** He puts the learning rate in the positive reals.

In lecture words: GD with learning rate; derivative = how loss changes when you change $w$.

### Local picture

```
  w_new = w_old − α · (∂L/∂w)

  α > 0     “how big a step”
  minus     “downhill”

  Notice: same idea for every weight; he computes only one fully.
```

### Bridge

Before multiplying slopes you must know **which gears** that weight actually turns.

---

## Topic 8: Path of $w_{1,1}^1$; $\sigma(2)\approx 0.88$ (33:20–38:58)

### Where this sits on the master map

This is **PATH**. [Chain rule](./PREREQUISITES.md#p6-chain) as a highlighted wiring, then numbers.

### Board / screenshot

![Topic 8 — highlighted path of w¹₁₁; x=[1,0]; σ(2)≈0.88](./screenshots/composites/ch08-topic-08-chain-path-w111-numeric-panel1of1.png)

Caption: purple edge $w_{1,1}^1$ from $x^1$ into $a^2_1$, then $w_{1,1}^2$ into $a^3_1=\hat y$. Numeric: $x_i=[1,0]^\top$, all $w,b=1$, $a^2_1=\sigma(2)\approx 0.88$. Chain on the board: $\partial L/\partial w_{1,1}^1$ as a product along $\hat y,z^3_1,a^2_1,z^2_1,w$.

### What he is establishing

Consider **$w_{1,1}^1$**. First: **what path is this weight involved in?** Highlight it: that edge into hidden-1, then hidden-1 into the output. **$a^2_2$ is not on this path.**

He assumes **chain rule**. Loss depends directly on the output $a^3_1=\hat y$. That $a^3_1$ depends on $z^3_1$. That $z^3_1$ depends on $a^2_1$ (not $a^2_2$). $a^2_1$ depends on $z^2_1$. $z^2_1$ depends on $w_{1,1}^1$. Trace it **back**. That is backprop using the chain rule.

Worked numbers he precomputes: features **$1$ and $0$**; **all weights and biases $=1$**. Then $a^2_1=\sigma(1\cdot 1+1\cdot 0+1)=\sigma(2)\approx 0.88$. Same style for the output and the loss.

The trap is including $a^2_2$ in this weight’s product because “it is in the net.” Off-path nodes do not appear.

You can now point at the purple path and compute $\sigma(2)$. What is still open: each local $\partial$ as a formula, then the product.

### Analogy for this topic only

Only the gears on the chain from this pedal to the wheel. The other pedal is a different story.

Someone asks: **why $x=[1,0]$?** So one incoming feature is live ($1$) and the other is off ($0$) — $\partial z/\partial w$ will be $x^1=1$, easy.

In lecture words: highlight the path; $a^2_2$ stays off; $\sigma(2)\approx 0.88$.

### Local picture

```
  x¹=1  --w¹₁₁-->  a²₁=σ(2)≈0.88  --w²₁₁-->  ŷ
  x²=0  --         a²₂  (off this path)

  Notice: all w and b are 1. Only this path is differentiated for this w.
```

### Bridge

The board now writes five fractions. Multiply them and you have $\partial L/\partial w_{1,1}^1$.

---

## Topic 9: Five-factor product (38:58–48:18)

### Where this sits on the master map

This is **PRODUCT**. The whole backprop computation for **one** weight.

### Board / screenshot

![Topic 9 — five partials; product (y−ŷ) ŷ(1−ŷ) w²₁₁ a(1−a) x](./screenshots/composites/ch09-topic-09-five-factors-one-weight-panel1of2.png)

![Topic 9 continued — sigmoid derivative; ∂z/∂a = w; ∂z/∂w = x](./screenshots/composites/ch09-topic-09-five-factors-one-weight-panel2of2.png)

Caption: $\partial L/\partial w_{1,1}^1$ as five factors. He writes $\partial L/\partial\hat y=(y_i-\hat y_i)$ from $L=\frac12(y-\hat y)^2$. $\partial\hat y/\partial z^3_1=\hat y(1-\hat y)$. $\partial z^3_1/\partial a^2_1=w_{1,1}^2$. $\partial a^2_1/\partial z^2_1=a^2_1(1-a^2_1)$. $\partial z^2_1/\partial w_{1,1}^1=x_i^1$. Green product on the last pane.

### What he is establishing

Copy the chain and replace $a^3_1$ by $\hat y_i$ for practical purposes. Then grind, left to right.

**1.** $L=\frac12(y_i-\hat y_i)^2$ (he notices a missing square, then puts it). He writes

$$
\frac{\partial L}{\partial\hat y_i}=(y_i-\hat y_i)
$$

(the $2$ from $x^2$ cancels the $1/2$). Textbooks that differentiate **wrt $\hat y$** often keep an extra minus; **the product he multiplies uses $(y-\hat y)$ as factor 1.** Follow the board for this quiz.

**2.** $\hat y=\sigma(z^3_1)$, so

$$
\frac{\partial\hat y}{\partial z^3_1}=\sigma(z^3_1)\big(1-\sigma(z^3_1)\big)=\hat y_i(1-\hat y_i).
$$

**3.** $z^3_1=w_{1,1}^2 a^2_1+w_{1,2}^2 a^2_2+b$, so $\partial z^3_1/\partial a^2_1=w_{1,1}^2$.

**4.** $a^2_1=\sigma(z^2_1)$, so $\partial a^2_1/\partial z^2_1=a^2_1(1-a^2_1)$.

**5.** $z^2_1=w_{1,1}^1 x_i^1+\cdots$, so $\partial z^2_1/\partial w_{1,1}^1=x_i^1$.

Label them 1–5 and **multiply**. That is $\partial L/\partial w_{1,1}^1$. **Similarly** for every other weight in $\theta$, then apply the GD update. That is the overall training idea for the weights.

```python
# One weight, five factors — matches the green board (Topic 9).
# yhat = sigmoid(z3); a21 = sigmoid(z2)

dL_dw111 = (
    (y - yhat)                # 1  as he writes ∂L/∂ŷ
    * (yhat * (1 - yhat))     # 2  sigmoid' at output
    * w2_11                   # 3  ∂z³₁/∂a²₁
    * (a21 * (1 - a21))       # 4  sigmoid' at hidden-1
    * x1                      # 5  ∂z²₁/∂w¹₁₁
)
w111 = w111 - alpha * dL_dw111
# Repeat analogously for every other entry of theta.
```

The trap is stopping after one weight and calling the net trained.

You can now multiply the five. What is still open: doing it on **plates** of data, many times.

### Analogy for this topic only

Five gear ratios on one chain, times each other, is how hard the miss pulls on this one bolt. Every other bolt gets its own chain.

Someone asks: **where did hidden-2 go?** Off-path. Its $w$ gets a *different* five-factor story.

In lecture words: five terms; multiply; update all of $\theta$.

### Local picture

```
  ∂L/∂w¹₁₁ = (y−ŷ) · ŷ(1−ŷ) · w²₁₁ · a²₁(1−a²₁) · x¹

  1          2           3         4              5

  Notice: factor 5 is x¹, not x² — this weight multiplies x¹.
```

### Bridge

One example at a time is slow. He batches $D$ and writes the loop in Python.

---

## Topic 10: Batch, epoch, Python loop (48:18–54:53)

### Where this sits on the master map

This is **LOOP**. [Batch vs epoch](./PREREQUISITES.md#p8-epoch). After this, **the procedure is the one they will keep**. Next tutorials: PyTorch.

### Board / screenshot

![Topic 10 — for e in range(max_epoch): for di in D: forward, loss, grads, update](./screenshots/composites/ch10-topic-10-batch-epoch-python-loop-panel1of1.png)

Caption: $m/b=k$, $D=\{d_1,\ldots,d_k\}$, `epochs: max_epoch`. Nested Python: `for e in range(max_epoch):` / `for di in D:` then $\hat y=f_\theta(X)$ # Forward Pass, $L(y,\hat y)$ # Loss, obtain gradients, update the weights.

### What he is establishing

Rather than one example at a time, take a **batch**. $m$ examples, batch size $b$, about $k=m/b$ batches (assume $m$ divisible by $b$; else a **short last batch** — fine). $D=\{d_1,\ldots,d_k\}$, each $d_i$ holds $b$ examples.

Decide how many times to iterate over the **whole** data: **`max_epoch`**. He writes it **in a Python way**, assuming you know Python:

```python
# Overall training procedure — transcribed from the tablet (~50:16–52:14)
# m examples, batch size b, k = m/b batches (last may be short)

# D = {d1, ..., dk}
for e in range(max_epoch):       # how many full passes
    for di in D:                 # each plate of b examples
        X, y = di                # inputs and targets in this batch
        yhat = f_theta(X)        # FORWARD PASS
        loss = L(y, yhat)        # LOSS COMPUTATION
        grads = obtain_gradients(loss, theta)  # chain rule, every w
        theta = update_weights(theta, grads)   # GD: minus alpha
```

**Iteration** = how many times weights are updated (one batch). **Epoch** = how many times you iterate over the **whole** data. Unless specified, this is the **standard training procedure**. Then use **test** data to predict and measure with a **metric of the problem**. Mix-and-match later; **overall structure remains**.

**Recap he lists:** structure of the simple MLP; mapping; error; names; forward pass; how you compute the error; GD; chain rule for gradients; a simple numeric forward; gradients for one weight, same for all weights, update once; overall supervised training procedure.

**Thank you.** Next session / next tutorials: **PyTorch**.

The trap is calling `for e in range(max_epoch)` an iteration. That header is epochs; the inner loop’s updates are iterations.

You can now write the nested loops. What is still missing is a library that computes the five factors for you — that is the next sitting, not more tablet algebra.

### Analogy for this topic only

Flashcard packs of $b$. Walking every pack once is an epoch. Flipping one pack is an iteration. Then a sealed final (test) with a score of your choice.

Someone asks: **100 images, batch 64?** Last pack has 36. He shrugged: fine.

In lecture words: batches $d_1..d_k$; `max_epoch`; forward, loss, grads, update; next is PyTorch.

### Local picture

```
  m examples, batch b, k = m/b

  epoch 1:  d1 → d2 → ... → dk     (k updates)
  epoch 2:  d1 → d2 → ... → dk
  ...
  max_epoch times

  Notice: iteration = one update. Epoch = one full pass.
```

### Bridge

The leftover problem is **not** more chain rule. It is typing this loop in **PyTorch** so autograd owns the five factors. The leftover problem of the **YouTube title** — variational $D_f$, critic, saddle — is a **different lecture** ([W1_L4](../03-W1-L4-Variational-Divergence-Minimization/NOTES.md)).

---

## External references

All companions live **here**, not under the topics. Mix of **video**, **blog/notes**, and **university lectures**. No Wikipedia.

**Start here (if you only open three).** 3Blue1Brown ch. 1 (what a net is) → CS231n backprop notes → Karpathy micrograd (the Python loop).

This recording is **~55 min**. Extra links are second passes of the **same ten boxes**. The YouTube title’s VDM math is in **Sources**, not here.

### Per-topic companions (2–3 each)

| Topic / map box | Type | Resource | Why it helps |
|-----------------|------|----------|--------------|
| **1 · supervised $D$, $f$, $L$** | video | [StatQuest — Essential ideas of NNs](https://www.youtube.com/watch?v=CqOfi41LfDw) | Pairs, a map, a miss — his JOB box without the title trap. |
| **1 · supervised $D$, $f$, $L$** | notes | [Stanford CS229 — Supervised learning notes](https://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf) | University write-up of $(x,y)$, $f$, regression. |
| **1 · supervised $D$, $f$, $L$** | notes | [Nielsen ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html) | Original free chapter: labeled digits as the same job. |
| **2 · football; two passes** | video | [3Blue1Brown — Gradient descent, how nets learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Cost as a miss; practice shrinks it; need not hit zero. |
| **2 · football; two passes** | video | [3Blue1Brown — Backprop, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U) | Forward vs “how would this example like to nudge the weights.” |
| **2 · football; two passes** | blog | [3Blue1Brown lesson text — GD](https://www.3blue1brown.com/lessons/gradient-descent) | Same hour in prose if you want pause-able diagrams. |
| **3 · 2–2–1 MLP; names $a$** | video | [3Blue1Brown — What is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | Layers, why hidden nodes exist, ReLU vs sigmoid at the end. |
| **3 · 2–2–1 MLP; names $a$** | notes | [CS231n — Neural nets part 1](https://cs231n.github.io/neural-networks-1/) | Fully connected layers, activations, counting parameters. |
| **3 · 2–2–1 MLP; names $a$** | video | [Stanford CS231n 2025 L4](https://www.youtube.com/watch?v=25zD5qJHYsk) | Latest university lecture: hidden layers + graphs. |
| **4 · weight indices** | notes | [CS231n — Neural nets 1, “named entity” weights](https://cs231n.github.io/neural-networks-1/#layers) | $W$ as dest × source; same dest/source idea as $w_{i,j}^{k}$. |
| **4 · weight indices** | notes | [Nielsen ch. 1 — notation](http://neuralnetworksanddeeplearning.com/chap1.html) | $w_{jk}^{l}$ cousin of his dest/source/layer. |
| **4 · weight indices** | video | [StatQuest — NNs, multiple inputs](https://www.youtube.com/watch?v=83LYR-1IcjA) | Two features in; named edges. |
| **5 · forward; sigmoid; ReLU** | notes | [Nielsen ch. 1 — sigmoid / $\sigma$](http://neuralnetworksanddeeplearning.com/chap1.html) | $1/(1+e^{-z})$ and why a bend. |
| **5 · forward; sigmoid; ReLU** | notes | [CS231n — activation functions](https://cs231n.github.io/neural-networks-1/#actfun) | Sigmoid, ReLU, tanh — the three he names. |
| **5 · forward; sigmoid; ReLU** | video | [MIT 6.S191 2026 L1](https://www.youtube.com/watch?v=II4giR4vOOo) | University lab: $z$ then $g$, a first forward. |
| **6 · $\theta$ and MSE** | video | [3Blue1Brown ch. 2 (cost)](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Squared miss; average over examples. |
| **6 · $\theta$ and MSE** | notes | [CS231n — Loss functions](https://cs231n.github.io/neural-networks-2/#losses) | MSE vs CE — he names both, stays on MSE. |
| **6 · $\theta$ and MSE** | notes | [Nielsen ch. 1 — quadratic cost](http://neuralnetworksanddeeplearning.com/chap1.html) | $\frac12\|y-a\|^2$; $1/2$ for a tidy derivative. |
| **7 · GD and $\alpha$** | video | [StatQuest — Gradient descent](https://www.youtube.com/watch?v=sDv4f4s2SB8) | Minus the slope; step size as a knob. |
| **7 · GD and $\alpha$** | notes | [CS231n — Gradient descent](https://cs231n.github.io/optimization-1/) | $w\leftarrow w-\alpha\nabla L$; $\alpha\in\mathbb{R}^+$. |
| **7 · GD and $\alpha$** | blog | [3Blue1Brown — GD lesson](https://www.3blue1brown.com/lessons/gradient-descent) | Downhill picture matching his update. |
| **8 · path; $\sigma(2)\approx 0.88$** | video | [StatQuest — Chain rule](https://www.youtube.com/watch?v=wl1myxrtQHQ) | Path first, then multiply — his Topic 8 order. |
| **8 · path; $\sigma(2)\approx 0.88$** | notes | [CS231n — Backprop / circuits](https://cs231n.github.io/optimization-2/) | Highlight gates on the path; off-path stays out. |
| **8 · path; $\sigma(2)\approx 0.88$** | blog | [Olah — Calculus on computational graphs](https://colah.github.io/posts/2015-08-Backprop/) | Same purple-path idea as a graph. |
| **9 · five-factor product** | notes | [Nielsen ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html) | Original four-step backprop; $\sigma'=\sigma(1-\sigma)$. |
| **9 · five-factor product** | video | [3Blue1Brown — Backprop calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) | The five local slopes in animation. |
| **9 · five-factor product** | video | [Karpathy — micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | Builds the product in Python, then steps $w$. |
| **10 · batch / epoch loop** | notes | [CS231n — Training / SGD](https://cs231n.github.io/neural-networks-3/#sgd) | Minibatch vs full pass; iteration vs epoch. |
| **10 · batch / epoch loop** | video | [Karpathy — micrograd training loop](https://www.youtube.com/watch?v=VMj-3S1tku0) | Nested epoch/batch; then autograd (his “next PyTorch”). |
| **10 · batch / epoch loop** | notes | [CS231n — Gradient check](https://cs231n.github.io/neural-networks-3/#gradcheck) | Why a short last batch still yields a plate. |

**How to use.** After Topic 3, 3Blue1Brown ch. 1. After Topic 5, Nielsen sigmoid/ReLU. After Topic 7, StatQuest GD. After Topic 9, Nielsen ch. 2 *and* CS231n optimization-2. After Topic 10, micrograd’s loop. Open CS231n 2025 L4 if you want the same objects in a 2025 university room.

### If you opened this video for the **YouTube title** (VDM)

This tape does not teach variational divergence minimization. Use **Sources** below for W1_L4, NPTEL Lec 04, the Nowozin f-GAN paper, and the MSR talk.

---

## Apply it (scenarios)

*Workplace-style situations that use ideas from this video only.*

### Scenario 1: Intern updates $x$ instead of $\theta$

**Context:** MSE will not go down in a training plot.  
**Challenge:** the intern treats the features as knobs.  
**Questions:**  
1. What is frozen?  
2. What lives in $\theta$?

<details><summary>Show solution sketch</summary>

- Topic 6: inputs are fixed; only weights/biases in $\theta$ change. GD cannot edit the dataset.

</details>

### Scenario 2: Gradient includes the off-path hidden node

**Context:** they backprop $w_{1,1}^1$ but multiply by $a^2_2$.  
**Challenge:** the numeric gradient disagrees with a tiny finite difference.  
**Questions:**  
1. Which hidden node is on the path of $w_{1,1}^1$?  
2. What is factor 5?

<details><summary>Show solution sketch</summary>

- Topics 8–9: path is $w_{1,1}^1\to a^2_1\to\hat y$. Factor 5 is $x_i^1$, not $a^2_2$.

</details>

### Scenario 3: They report “10 iterations” but ran `range(10)` on epochs

**Context:** batch size 64, $m=640$, `max_epoch=10`.  
**Challenge:** the skip-connection paper wanted 10 *updates*.  
**Questions:**  
1. How many updates is one epoch?  
2. How many updates is 10 epochs?

<details><summary>Show solution sketch</summary>

- Topic 10: $k=m/b=10$ batches per epoch = 10 iterations per epoch. 10 epochs = 100 updates. `for e in range(10)` is epochs, not iterations.

</details>

### Scenario 4: Loss explodes after one step

**Context:** $\alpha=50$, $\partial L/\partial w$ about $0.4$.  
**Challenge:** $w$ jumps wildly.  
**Questions:**  
1. Which sign should the update use?  
2. What did he say about choosing $\alpha$?

<details><summary>Show solution sketch</summary>

- Topic 7: $w\leftarrow w-\alpha\partial L/\partial w$, $\alpha\in\mathbb{R}^+$, choosing it takes knack (later tutorials). 50 is not illegal on the tape, but it is a huge stride.

</details>

---

## Sources

- Video: [W2_L5 VDM title / recording = MLP tutorial](https://www.youtube.com/watch?v=stZC0Zk5KYo) · [IITM playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu) index 9
- Captions: `raw/captions.en.vtt` / `raw/captions.en.timed.txt`
- Claim sheets: `raw/claims/topic-01.md` … `topic-10.md`
- Course page: https://study.iitm.ac.in/ds/course_pages/BSDA5002.html
- Actual VDM (title, not this tape): [W1_L4](../03-W1-L4-Variational-Divergence-Minimization/NOTES.md) · [NPTEL Lec 04](../../Mathematical-Foundation-for-GenerativeAI/27-Lec04-Variational-Divergence-Minimization/NOTES.md) · Nowozin et al., [*f-GAN*](https://arxiv.org/abs/1606.00709) (NeurIPS 2016) · [MSR page](https://www.microsoft.com/en-us/research/publication/f-gan-training-generative-neural-samplers-using-variational-divergence-minimization/) · [Nowozin talk](https://www.youtube.com/watch?v=y7pUN2t5LrA)
- Ingest: captions yes · video yes · 10 unique topic composites (Topic 9 has two panels) · `ingest_evidence: E3`
