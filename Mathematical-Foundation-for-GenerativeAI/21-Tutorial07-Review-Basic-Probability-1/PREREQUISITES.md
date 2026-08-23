# Prerequisites — warm-up before Tutorial 7 (Review of Basic Probability 1)

> **Do this first** if sets, mappings, “given that,” or “countable” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Course: NPTEL Mathematical Foundations of Generative AI · Tutorial 7.  
> Builds on the professor’s earlier probability lectures; this tutorial is a **recap + discrete families**.  
> **Beginner deep warm-up:** definition · micro example · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "Ω is the bag of every possible outcome of one experiment."
  "An event is a subset of that bag; P assigns it a number in [0,1]."
  "P(A|B) shrinks the world to B and reweighs what is left."
  "Independent means P(both) = P(A)×P(B), not 'they feel unrelated.'"
  "A random variable is a function from outcomes to real numbers."
  "A CDF is a running total of probability from −∞ to x."
  "A PMF is the pile of mass sitting on one number (discrete only)."
```

**Warm-up → tutorial boxes**

```
  §1  Sets, subsets, union, intersection     ──► Topics 1–4
  §2  Functions / mappings                   ──► Topics 7–8
  §3  One experiment, many outcomes          ──► Topic 1
  §4  Numbers in [0,1] as mass               ──► Topics 1, 9
  §5  Countable vs continuum                 ──► Topic 9
  §6  “Given that” as shrinking the world    ──► Topics 2–4
  §7  Product vs sum                         ──► Topics 3, 5–6
  §8  Running totals on a number line        ──► Topics 8–10
```

---

## 1. Sets: bags, pieces, glue, overlap

<a id="p1-sets"></a>

### Purpose for the video

Almost every formula is a sentence about **sets**. If union and intersection feel foggy, Bayes will look like alphabet soup.

### Definitions

| Word | Picture |
|------|---------|
| **Set** | A bag of things with no repeats |
| **Subset** | A smaller bag made only from things already in the big bag |
| **Union** $A \cup B$ | Everything in $A$ or $B$ or both (glue) |
| **Intersection** $A \cap B$ | Things in **both** (overlap) |
| **Complement** $A^c$ | Everything in the big bag that is **not** in $A$ |
| **Empty set** $\emptyset$ | A bag with nothing |
| **Disjoint / mutually exclusive** | Overlap is empty: $A \cap B = \emptyset$ |
| **Power set** | The set of *all* subsets of a set |

### Worked micro

Universe = a die $\Omega=\{1,2,3,4,5,6\}$.  
Even $A=\{2,4,6\}$. High $B=\{4,5,6\}$.

- $A \cup B = \{2,4,5,6\}$  
- $A \cap B = \{4,6\}$  
- $A^c = \{1,3,5\}$  
- $A$ and $\{1,3,5\}$ are **disjoint**.

### Analogy — fruit bowl

The whole bowl is $\Omega$. “Citrus” is a subset. Union is dumping two labeled piles into one. Intersection is the fruit that got **both** stickers. Complement is “everything that is not citrus.”

### Notice

- $\Omega$ itself is a subset of $\Omega$. The empty set is also a subset.  
- “Mutually exclusive” is about **overlap**, not about probability yet.  
- A **partition** of $\Omega$ is a list of nonempty disjoint subsets whose union is all of $\Omega$ — like cutting a pie so every crumb sits in exactly one slice. Topic 3 uses this constantly.

```
  Ω pie:   [ B1 | B2 | B3 | B4 ]
  every outcome in exactly one slice
```

### Mini-check

1. If $A \cap B = \emptyset$, what is in both $A$ and $B$?  
2. Name two subsets of $\{H,T\}$.  
3. Is $\{2,4\}$ a subset of $\{1,2,3,4,5,6\}$?  
4. Do $\{1,2,3\}$ and $\{3,4,5,6\}$ partition a die? Why not?

---

## 2. Functions: stickers from one world to another

<a id="p2-fn"></a>

### Purpose for the video

A **random variable** is just a function. If “function” still means “$y=mx+b$ only,” Topic 7 will feel mystical.

### Definitions

A **function** $f : S \to T$ is a rule that gives **exactly one** output in $T$ to each input in $S$.

- $S$ = domain (where you start)  
- $T$ = codomain (where labels live)

### Worked micro

Outcomes $S=\{H,T\}$. Rule $X$: $X(H)=1$, $X(T)=0$.  
That is a function $S \to \mathbb{R}$. Nothing “random” lives inside the rule — the **experiment** is random; the sticker is fixed.

Hotel menu $S=\{\text{dosa}, \text{idli}, \text{tea}\}$.  
Price $X$ maps each dish to rupees. Calories $Y$ maps each dish to kcal. **Two functions, same domain.**

### Analogy — coat-check tickets

Each coat (outcome) gets **one** ticket number (real). Two counters can issue two different ticket systems on the same coat rack (price vs calories).

### Notice

- One input → one output. Two inputs may share an output (two dishes can cost ₹120).  
- The lecture slogan: a random variable is **neither random nor a variable**.

### Mini-check

1. Can one dish have two prices under one function $X$?  
2. Can two dishes share one price?  
3. Domain of the coin map $H\mapsto 1$, $T\mapsto 0$?

---

## 3. One random experiment, many outcomes

<a id="p3-re"></a>

### Purpose for the video

The first object on the board is a **random experiment (RE)** and its **sample space** $\Omega$.

### Definitions

| Term | Meaning |
|------|---------|
| **Random experiment** | A procedure you can repeat whose result is not known in advance |
| **Outcome** $\omega$ | One possible complete result |
| **Sample space** $\Omega$ | The set of **all** outcomes |

### Worked micro

- Coin once: $\Omega=\{H,T\}$.  
- Coin three times: $\Omega=\{HHH,HHT,HTH,THH,HTT,THT,TTH,TTT\}$ — **eight** strings, not the number 3.  
- “Number of heads” is **not** $\Omega$; it is a **function of** $\Omega$ (Topics 7 and 9).  

Count the eight strings: $2\times 2\times 2=8$. Exactly one of them has zero heads (`TTT`); three have one head (`HTT, THT, TTH`). Those counts become the $1/8$ and $3/8$ jumps on the lecture staircase.

### Analogy — every legal chess finish

$\Omega$ is the list of every game that could happen. One game you actually play is one $\omega$. “White won” is a **bundle** of many finishes — an **event**, not a single outcome.

### Notice

- Outcomes can be words, sequences, dishes — they need not be numbers.  
- That is why we later invent random variables: to land in $\mathbb{R}$.

### Mini-check

1. For a fair die, how many outcomes are in $\Omega$?  
2. Is “even” an outcome or an event?  
3. Why isn’t $\{0,1,2,3\}$ the sample space of “toss three coins”?

---

## 4. Probability as mass in $[0,1]$

<a id="p4-mass"></a>

### Purpose for the video

$P$ is a **function on events** that hands each event a number. The axioms are the rules that mass must obey.

### Definitions

Think of $P(A)$ as **how much clay** sits on the subset $A$.

- No pile is negative.  
- The whole $\Omega$ pile is exactly $1$ (100% of the clay).  
- If two piles do not overlap, their glued pile is just the **sum**.

### Worked micro

Fair die. $P(\{4\})=1/6$. $P(\{2,4,6\})=1/2$.  
$P(\Omega)=1$. You cannot have $P(\text{even})=-0.3$.

### Analogy — 1 kg of dough

You may cut the dough into slices. Slice weights $\ge 0$. All slices together weigh 1 kg. Non-overlapping slices add.

### Notice

- $P$ eats **sets**, not raw numbers. $P(3)$ is meaningless unless you mean $P(\{3\})$.  
- Later, a **PMF** puts mass on **points** of $\mathbb{R}$; $P$ itself still lives on events.

### Mini-check

1. Why is $P(\Omega)=1$ called normalization?  
2. If $A$ and $B$ overlap, is $P(A\cup B)$ equal to $P(A)+P(B)$?  
3. Can an event have probability $1.2$?

---

## 5. Countable vs a continuum

<a id="p5-count"></a>

### Purpose for the video

**Discrete** random variables take **countably many** values. The lecture assumes you know finite vs countably infinite vs “the whole line.”

### Definitions

| Kind | Example | Can you list them? |
|------|---------|--------------------|
| **Finite** | $\{0,1,2,3\}$ | Yes, it ends |
| **Countably infinite** | $\{0,1,2,3,\ldots\}$ or $\{1,2,3,\ldots\}$ | Yes, like a never-ending roll call |
| **Uncountable / continuum** | all real $x$ in $[0,1]$ | No complete list |

A set is **countable** if it is finite **or** countably infinite.

### Worked micro

Number of heads in 3 tosses: $\{0,1,2,3\}$ — finite.  
Number of coin tosses until first head: $\{1,2,3,\ldots\}$ — countably infinite (geometric).  
A random height in centimeters as a real: continuum (continuous RV; **next** tutorial).

### Analogy — hotel rooms vs a ruler

You can number every hotel room $1,2,3,\ldots$ (countable). You cannot give every point on a 1-meter stick a unique roll-call number without leaving gaps (continuum).

### Notice

- “Infinite” is not one thing. Countably infinite is still discrete-friendly.  
- PMF language is for countable support. Densities wait until Tutorial 8.

### Mini-check

1. Is $\{0,1,2,\ldots\}$ countable?  
2. Is every real in $[0,1]$ countable as a set?  
3. Why can a Poisson RV still be called discrete?

---

## 6. “Given that” shrinks the world

<a id="p6-given"></a>

### Purpose for the video

**Conditional probability** is not a new kind of magic number. It is **the same mass idea after you throw away everything outside $B$**.

### Pattern

You learn $B$ happened. The new universe is $B$.  
The piece of $A$ that still matters is $A \cap B$.  
Re-normalize by the leftover total $P(B)$ (must be $>0$).

$$
P(A\mid B) = \frac{P(A \cap B)}{P(B)}
$$

### Worked micro

Die. $B=\{\text{even}\}=\{2,4,6\}$, $P(B)=1/2$.  
$A=\{4,5,6\}$. $A\cap B=\{4,6\}$, $P(A\cap B)=1/3$.  
Then $P(A\mid B)=(1/3)/(1/2)=2/3$.  
Among evens, two of the three faces are in $A$.

### Analogy — locked rooms

A hotel has 6 rooms. You learn the guest is in an **even-numbered** room. You no longer care about rooms 1,3,5. You re-share 100% of your belief across 2,4,6 only.

### Notice

- Need $P(B)>0$. Conditioning on a zero-mass event is undefined here.  
- $P(\cdot\mid B)$ is a **new** probability assignment to *every* event.

Work the same die with numbers: $P(A\cap B)=2/6=1/3$, $P(B)=3/6=1/2$, so $P(A\mid B)=(1/3)/(1/2)=2/3$. You can do this on paper — this video never opens a notebook.

### Mini-check

1. Why divide by $P(B)$?  
2. What is $P(B\mid B)$?  
3. If $A \subset B$, what is $P(A\mid B)$ in words?  
4. Why is $P(A\mid B)$ illegal if $P(B)=0$?

---

## 7. Sum for glue, product for independence

<a id="p7-arith"></a>

### Purpose for the video

The instructor’s slogan: **union ~ plus**, **intersection ~ times** (when events do not overlap, or when they are independent).

### Two different multiplications

| Situation | Formula |
|-----------|---------|
| **Disjoint union** | $P(A \cup B)=P(A)+P(B)$ |
| **Independence** | $P(A \cap B)=P(A)\,P(B)$ |
| **Always (definition of conditional)** | $P(A \cap B)=P(A\mid B)\,P(B)$ |

Independence is a **special case** where $P(A\mid B)=P(A)$ (when $P(B)>0$).

**Indicator preview (Topic 10):** pick an event $B$ and define a function that is $1$ if the outcome landed in $B$ and $0$ otherwise. That function is an **indicator random variable**. Its “success” chance is just $P(B)$ — a Bernoulli with $p=P(B)$.

### Worked micro

Two fair coins, assumed independent.  
$P(\text{first H and second H})=\frac12\cdot\frac12=\frac14$.  
If they shared a sticky mechanism, the product would be **wrong**.

### Analogy — two light switches

If the switches do not talk to each other, “both on” is the product of the two “on” rates. If one switch is wired to the other, you **cannot** multiply blindly.

### Notice

- Pairwise independence (every couple multiplies) is weaker than **total** independence (every subset multiplies).  
- “They feel unrelated” is not a proof.

### Mini-check

1. When may you add $P(A)+P(B)$ for a union?  
2. Write the independence test for two events.  
3. If $P(A\mid B)=P(A)$, what does that say (when $P(B)>0$)?

---

## 8. Running totals on the number line

<a id="p8-cdf"></a>

### Purpose for the video

The **cumulative distribution function (CDF)** is a **running total** of probability as you walk from $-\infty$ toward $+\infty$.

### Picture

```
  −∞ -------- x -------- +∞
  [ all mass of values ≤ x ]  =  F(x)
```

$F$ starts near $0$ on the far left and ends near $1$ on the far right. It never goes down (**nondecreasing**). It may stay **flat** between jumps.

### Worked micro

$X=$ number of heads in one fair coin: $0$ or $1$, each $1/2$.

- $F(-3)=0$ (nothing yet)  
- $F(0.5)=P(X\le 0.5)=P(X=0)=1/2$  
- $F(2)=1$ (everything is already behind you)

For discrete $X$, $F$ looks like a **staircase**. The **height of a jump** at $x$ is $P(X=x)$ — that pile is the **probability mass function (PMF)** at $x$.

### Analogy — filling a measuring cup

Walk along the number line and pour every probability you pass into a cup. The water level at position $x$ is $F(x)$. A discrete RV dumps whole spoonfuls at a few ticks (jumps). A continuous RV (next tutorial) drips smoothly.

### Notice

- Nondecreasing $\neq$ always increasing. Flat stretches are legal.  
- CDF exists for **every** RV. PMF exists only for **discrete** ones.

### Mini-check

1. Why is $F(-\infty)$ thought of as $0$?  
2. If $F$ is flat from $0.1$ to $0.9$, what mass sits inside $(0.1,0.9)$?  
3. Jump of $F$ at $2$ equals what probability?

---

### Paper check

1. Sample space of one die?  
2. Formula for $P(A\mid B)$ and the $P(B)>0$ warning?  
3. Independence of $A,B$ in one equation?  
4. Is a random variable a set or a function?  
5. CDF vs PMF in one line each?  
6. Name one finite discrete RV and one countably infinite one from later in the lecture.

**Peek:** (1) $\{1,\ldots,6\}$ (2) $P(A\cap B)/P(B)$ (3) $P(A\cap B)=P(A)P(B)$ (4) function $\Omega\to\mathbb{R}$ (5) running total vs mass at a point (6) 3-coin heads; geometric wait-for-head

---

**Second teachers (names only here).** Khan Academy, Seeing Theory, StatQuest, 3Blue1Brown, Statlect. The actual pointers live at the end of [NOTES.md](./NOTES.md#external-references) — a few well-known items, not a link dump.

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html) Part A = this file.
