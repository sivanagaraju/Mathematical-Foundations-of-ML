# Prerequisites — warm-up before W1_T4 (PyTorch model building)

> **Do this first** if “logit,” “ReLU,” “`nn.Module`,” “bias,” or “zero_grad” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> IITM BS · Week 1 Tutorial 4 · TA live-codes (Chandan).  
> **Beginner:** purpose · definition · micro · analogy · ASCII · notice · mini-check.

Previous sheet left **data ready** (Fashion-MNIST tensors in batches of 64). This hour **builds an MLP**, **trains** it, **tests** it, and **saves** weights. CNN is a preview, not the coded model.

```
  After this warm-up you can say:

  "An MLP is stacked W^T x + b maps with a ReLU bend between them."
  "Flatten stacks the 28×28 photo into 784 numbers; batch 64 stays 64."
  "Logits are raw scores; softmax turns them into 10 probabilities that sum to 1; argmax is the index."
  "nn.Module is the parent; you write __init__ and forward; you call model(x), not model.forward(x)."
  "Gradient descent: W ← W − α ∇L. Cross-entropy is the 'how wrong' for 10 classes."
  "One epoch = every training batch once. α = 1e-3. Batch = 64."
  "Train computes grads; eval + no_grad does not. loss.item() is the number, not the graph."
  "Save the weights (.pth); do not retrain from scratch every morning."
```

```
  §1  MLP = stacked linear + ReLU     ──► Topics 1–5
  §2  Flatten 28×28 → 784             ──► Topic 3
  §3  Logits, softmax, argmax         ──► Topics 3, 5, 6, 9
  §4  nn.Module / super / forward     ──► Topics 3, 6
  §5  Gradient descent + CE loss      ──► Topic 7
  §6  Epoch, batch, learning rate     ──► Topics 7–8
  §7  Train vs eval / no_grad / item  ──► Topics 8–9
  §8  Save weights (.pth)             ──► Topic 10
```

**One scene.** A **clothing warehouse** with 10 bins (T-shirt … ankle boot). Each photo is 28×28 pixels. You must **score** which bin a photo belongs to, then **nudge the stamps** using 64-photo carts until the guesses get better.

```
  PHOTO     28×28 brightness grid
  VECTOR    784 numbers after flattening
  SCORES    10 logits (one shout per bin)
  LABEL     integer 0–9 (true bin)
  BATCH     64 photos on one cart
```

---

## 1. An MLP is stacked linear maps with a bend

<a id="p1-mlp"></a>

**Purpose.** He builds a **multi-layer perceptron (MLP)** before mentioning CNNs.

**Definition.** A **linear layer** maps a vector $x$ to $W^\top x + b$. $W$ is a **weight matrix** (many numbers you will train). $b$ is a **bias** — one offset per output slot, so the map can shift even if $x$ is all zeros. An **MLP** stacks several of those maps and puts a **nonlinearity** (here **ReLU**) between them. Without the bend, three Linears collapse into **one** linear map: $W_3^\top W_2^\top W_1^\top x + \text{combined bias}$.

**Micro.** Tonight’s net: 784 numbers in → 512 → 512 → 10 scores.

| Stage | In | Out | Has ReLU after? |
|-------|----|-----|-----------------|
| Linear 1 | 784 | 512 | yes |
| Linear 2 | 512 | 512 | yes |
| Linear 3 | 512 | 10  | **no** (logits) |

Work a 2-in, 3-out Linear by hand: $x = (1, 0)^\top$. If $W^\top$ is $3\times 2$ and $b = (0.1, 0, -0.2)^\top$, the three outputs are three different mixes of the two inputs, then shifted by $b$. ReLU would then zero any negative mix.

**Analogy.** Three warehouse desks in a row.

- Desk 1 reads the 784-column and writes a 512-number summary (plus 512 offsets).
- A bouncer (ReLU) paints every negative number **zero**.
- Desk 2 restamps 512 → 512. Another bouncer.
- Desk 3 writes **10 shouts**. **No bouncer** after desk 3 — a negative shout is still information.

Skip the bouncers and the three desks might as well be one desk.

```
  x (784)
    --Linear W1,b1-->  512  --ReLU-->
    --Linear W2,b2-->  512  --ReLU-->
    --Linear W3,b3-->  10 logits     (no ReLU)

  If every ReLU is deleted:  x  |-->  one fat Linear
```

**Notice.** A **CNN** is a later sibling: sliding stamps on the 28×28 grid, then a small MLP. Same train loop. **ViT** he parks as “future.”

**Mini-check.** If you remove every ReLU, what kind of map is the whole net?

---

## 2. Flatten: 28×28 becomes 784

<a id="p2-flatten"></a>

**Purpose.** Fashion-MNIST photos are grids; an MLP wants a list.

**Definition.** **Flatten** lays the rows of a matrix end to end. $28\times 28 = 784$. The **batch** axis is kept: 64 photos of shape `(64, 28, 28)` become `(64, 784)`. A single photo as a column is $784\times 1$ (ASR “784 plus 1”).

**Micro.** Row 1 (pixels 1–28), then row 2 (29–56), …, row 28 (757–784). He draws a square, an arrow “flatten,” then a tall column. Neighbors in the grid are **not** neighbors in the vector except along a row — that is why CNNs exist later.

**Analogy.** A crossword page vs the same 784 letters copied into **one spreadsheet column**. Same cells; the MLP only reads the column. Sixty-four crossword pages on a cart stay sixty-four columns side by side — flatten does not smash the cart into one photo.

```
  one photo:     28 × 28 grid
                     │  nn.Flatten  (row 1, row 2, …, row 28)
                     ▼
                 784 × 1 column

  a cart:   (64, 28, 28)  -->  (64, 784)
            batch kept          64 rows of 784
```

**Notice.** ASR “28 plus 28” is **times**. Flatten is not a trained layer (no $W$); it only reshapes.

**Mini-check.** After flatten, is the batch size (64 photos) still 64?

---

## 3. Logits, softmax, argmax

<a id="p3-logits"></a>

**Purpose.** The net does not spit the word “coat.” It spits 10 numbers.

**Definition.** **Logits** = raw unbounded scores (can be negative). **Softmax** (along `dim=1`, the class axis) turns a 10-vector into 10 **probabilities** that are ≥ 0 and **sum to 1**. He writes $P(Y\mid X)$ — “for this photo, how sure is each bin?” **Argmax** = the **index** of the largest entry (0–9), not the probability value itself.

**Micro.** Three-class toy (he uses this on the tablet in the test loop): $p = (p_1, p_2, p_3)$. If $p_2$ is largest, argmax is **index 1** (zero-based). Compare that index to the true label $y$. Fashion-MNIST has **ten** bins, same idea.

Work numbers: logits `[0.0, 2.0, −1.0]`. Softmax puts almost all mass on index 1. Argmax returns `1`. You need `1`, not `0.78`.

**Analogy.** Ten bins get a shout (logit). Softmax turns shouts into “how sure” fractions that add to 100%. Argmax is “**which bin** shouted loudest?” If you report the shout volume instead of the bin number, the accuracy tally breaks.

```
  logits (10 numbers, any real)
       │  softmax(dim=1)
       ▼
  P(Y|X)  (10 numbers ≥ 0, sum = 1)
       │  argmax  →  index 0..9
       ▼
  predicted class   compared to   true y
```

**Notice.** **Do not ReLU the last layer.** ReLU would clip negative logits to 0 and wreck the scores. PyTorch **cross-entropy** eats **logits**, not probabilities — so `forward` **returns logits**. Softmax is for humans / argmax, or inside CE.

**Mini-check.** If softmax outputs `[0.1, 0.2, 0.7]`, what does argmax return?

---

## 4. `nn.Module`, `super`, and `forward`

<a id="p4-module"></a>

**Purpose.** Every PyTorch net is a child of `nn.Module`.

**Definition.** **`nn.Module`** is the parent that **tracks parameters** (every `W` and `b` you assign to `self.`). You subclass it. **`super().__init__()`** runs the parent constructor — skip it and tracking breaks; `model.parameters()` will look empty. **`forward(self, x)`** is the math on one input. You **call `model(x)`**, which runs extra bookkeeping; **do not call `model.forward(x)`** as the public API.

**Micro.**

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()          # parent: start tracking
        self.flatten = nn.Flatten() # registered
        # self.linear_relu_stack = nn.Sequential(...)
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

logits = model(x)   # correct
# logits = model.forward(x)   # works but skips Module hooks; don't
```

`model.parameters()` later feeds **SGD**. That list exists only because layers live on `self`.

**Analogy.** The warehouse union contract is `nn.Module`. Your building is an employee who signed it. `forward` is “what you do when a photo arrives.” The radio still calls the employee **by name** (`model(x)`), not by shouting the method title. `super().__init__()` is signing the contract on day one. Forget the signature and payroll (`parameters()`) cannot find your stamps.

```
  nn.Module          (contract: track W, b)
      ▲
  NeuralNetwork
      __init__   layers on self.   +  super().__init__()
      forward    x → logits
  usage:  logits = model(x)
          SGD(model.parameters(), lr=1e-3)
```

**Notice.** `nn.Sequential` is itself an `nn.Module` — a taped-together pipeline you store as one `self.` field.

**Mini-check.** Who calls `forward` — you, or `model(x)`?

---

## 5. Gradient descent and cross-entropy

<a id="p5-gd"></a>

**Purpose.** Training is this update, over and over.

**Definition.** **Loss** $L$ is “how wrong.” For 10-class clothes he uses **cross-entropy (CE)**: it punishes the net when the true bin’s logit is not the loudest. **Gradient** $\nabla_W L$ is the slope of $L$ vs each weight. **Gradient descent:** $W \leftarrow W - \alpha \nabla_W L$. **SGD** does this on a **batch** of 64, not the whole 60,000. **Adam** / **RMSprop** are fancier solvers; he **names** them and does not derive them.

**Micro.** $\alpha = 10^{-3} = 0.001$. He writes $W = W - \alpha \partial L / \partial W$ and labels α **learning rate**. `loss.backward()` = compute every slope. `optimizer.step()` = take the step on **all** `model.parameters()` (or a subset later).

Tiny picture: if increasing $W_{17}$ makes CE worse, the slope is positive, so you **subtract** and $W_{17}$ gets smaller.

**Analogy.** Foggy hill (the loss). You feel downhill (gradient) and take a small step (α). A cart of 64 photos is 64 local weather reports averaged into one step. Cross-entropy is the hill’s height when the true bin is “coat” but the net shouted “sandal.” Adam is a fancier forklift with momentum; he points at it on the shelf and does not unpack it.

```
  CE(logits, y)  =  how wrong the 10 shouts are vs true bin y

  W  ←  W  −  α  *  (slope of CE vs W)     α = 1e-3
           ▲
           SGD on a batch of 64
```

**Notice.** He skips autograd internals (`grad_fn`). You will still type `backward` / `step`. CE wants **logits**, not `softmax(logits)`.

**Mini-check.** If you set α = 0, do the weights move?

---

## 6. Epoch, batch, learning rate, enumerate

<a id="p6-epoch"></a>

**Purpose.** The outer clocks of training.

**Definition.** **Batch** = 64 examples (DataLoader). **Epoch** = one full pass over the **training** set. **Learning rate** α = step size. He first says 5 epochs, then the cell runs **10**. **`enumerate(dataloader)`** yields `(batch_index, (X, y))` so you know you are on cart 0, 100, 200… for printing.

**Micro.** 60,000 train images / 64 ≈ 938 batches per epoch. Ten epochs ≈ 9,380 updates. Print uses `batch * batch_size + len(X)` as “how many examples seen this epoch.” Test is **one** sitting, not 10 test-epochs.

**Analogy.** One **night of practice** (epoch) is many **carts of 64** (batches). Learning rate is how far you move the stamps after each cart. Too big: you overshoot the valley. Too small: you barely move. `enumerate` is numbering the carts so you can shout the loss every hundredth cart.

```
  epoch   =  all training batches once     (he runs 10)
  batch   =  64 pairs (X, y)               D1, D2, …, Dk
  α       =  1e-3
  enumerate →  (0, D1), (1, D2), …         index + cart
```

**Notice.** Shuffle (from T2) changes **order** of carts each night. It does not mix train into test.

**Mini-check.** After 2 epochs, has every training image been seen twice (ignoring shuffle leftovers)?

---

## 7. Train vs eval, `no_grad`, and `loss.item`

<a id="p7-eval"></a>

**Purpose.** Two modes, two jobs — plus “number vs graph.”

**Definition.** **`model.train()`**: compute **gradients**; dropout/batch-norm (if any) behave as training. **`model.eval()`**: evaluation behaviour. **`torch.no_grad()`**: do not build a **computational graph** — saves memory; no grads. He says eval may already skip grads; `no_grad` is **best practice**.

**`loss.item()`**: CE is not a bare Python float. It carries a **graph** used by `backward`. `.item()` peels off the **number** for printing. You do not need the graph in a log line.

**`zero_grad`:** slopes **add** across batches unless you refresh them. He calls this a **programming** step, not extra math.

**Micro.** Train loop: `model.train()` then four steps then `zero_grad`. Test loop: `model.eval()` then `with torch.no_grad():` then argmax vs $y$. Never `backward` inside the test block.

**Analogy.** Practice night: keep a notebook of “which way the stamps should move” (the graph / grads). Every cart, **erase the notebook** (`zero_grad`) so tomorrow’s arrows are not added to today’s. Exam night: lock the notebook (`eval` + `no_grad`) and only mark the paper. `loss.item()` is copying the score onto the whiteboard **without** photocopying the whole notebook.

```
  train:  model.train()
          pred → CE → backward → step → zero_grad
          print(loss.item())     # number only

  test:   model.eval()
          with torch.no_grad():
              pred;  argmax vs y;  accumulate .item()
          no step, no zero_grad
```

**Notice.** Forgetting `zero_grad` on train is a different bug from forgetting `no_grad` on test.

**Mini-check.** Should `loss.backward()` run inside the test `no_grad` block?

---

## 8. Saving weights (`.pth`)

<a id="p8-save"></a>

**Purpose.** You do not want to retrain every morning.

**Definition.** **`state_dict`** = a dictionary of all parameter tensors (`linear_relu_stack.0.weight`, `.bias`, …). **`torch.save(..., "model.pth")`** writes it. **Load:** build a **fresh** `NeuralNetwork()` (same architecture), then **`load_state_dict(torch.load("model.pth"))`**. Extension **`.pth`**. This does **not** save Fashion-MNIST images.

**Micro.** Two statements to save; two to load (instance + copy weights). Then `model.eval()` before predicting.

**Analogy.** After a long night, photocopy the **stamp settings** into a folder named `model.pth`. Tomorrow, hire a **new** employee (new class instance) and give them yesterday’s photocopies. You do not rebuild the warehouse, and you do not photocopy the garment photos. If tomorrow’s employee has different desk sizes (CNN vs MLP), the photocopies will not fit.

```
  tonight:  train  -->  model.state_dict()  -->  model.pth

  tomorrow:
      m = NeuralNetwork().to(device)     # empty desks, same layout
      m.load_state_dict(torch.load("model.pth"))
      m.eval()
```

**Notice.** Architecture source (the class) + `state_dict` file must match. He will later say a CNN uses the **same** train loop but a **different** class — that CNN would get its own `.pth`.

**Mini-check.** Does `torch.save` of `state_dict` store the Fashion-MNIST images?

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
