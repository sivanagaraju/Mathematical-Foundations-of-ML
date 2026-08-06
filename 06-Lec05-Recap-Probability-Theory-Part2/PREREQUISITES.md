# Prerequisites — warm-up before Lec 05 (joints, conditionals, margins)

> **Do this first** if multi-RV language feels new, or if “joint / conditional / marginal” is still a fog of words.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on: [Lec 02](../03-Lec02-Recap-Probability-Theory-Part1/PREREQUISITES.md) · [Lec 03](../04-Lec03-Recap-Probability-Theory-Part2/PREREQUISITES.md) · [Lec 04](../05-Lec04-Recap-Probability-Theory-Part3/PREREQUISITES.md).  
> Still a **warm-up**, not a full probability course — but detailed enough if you do not know the basics.  
> **Goal of this file:** make every box on the lecture map feel *necessary*, not decorative.

```
  After this warm-up you can say:

  "Probability lives on (Ω, F, P); data are range points of maps X."
  "One experiment story (one Ω) can host many readings (many RVs)."
  "A joint scores both RVs via intersection of preimage events — not default multiply."
  "A vector RV is the same idea as d scalar RVs on the same Ω."
  "Conditioning shrinks the world: P(A|B)=P(A∩B)/P(B) when P(B)>0."
  "A marginal is the joint with the other variable summed/integrated out."
  "Capital P sizes sets; lowercase p is density-style notation (densities next)."
  "X-ray + disease need this toolkit: joint, conditional, marginal on shared structure."
```

**How to use**

1. Read top to bottom (each idea unlocks the next).  
2. Do every **Micro** on paper.  
3. Read the **Purpose for the video** block — that is *why* the idea exists.  
4. Paper check at the end of idea 8 before NOTES.

**Map of this warm-up → lecture boxes**

```
  §1  (Ω,F,P) + “and”     ──►  joints need ∩ of events
  §2  RV + CDF pushforward ──►  recap Topic 1
  §3  many maps, one Ω     ──►  multi-RV Topics 2–3
  §4  joint = both at once ──►  Topic 4
  §5  vector ≡ d scalars   ──►  Topics 2 + 5
  §6  shrink-the-world     ──►  Topic 6 events
  §7  cond + marg + P vs p ──►  Topics 6–7
  §8  X-ray purpose        ──►  Topic 7 bridge
```

---

## 1. Sets, experiments, and the triplet $(\Omega,F,P)$

<a id="p1-sets-and"></a>

### Purpose for the video

The lecture will say: joint probability is $P$ of an **intersection of events** on one sample space.  
If “event,” “intersection,” and “$P$ sizes a set” are foggy, every later formula is just symbols.  
This section builds the **floor** those formulas stand on.

### Sets and subsets

A **set** is a bag of objects. Members are **in**; everything else is **out**.

```
  Die faces:     Ω = {1, 2, 3, 4, 5, 6}
  Even faces:    A = {2, 4, 6}     ← a subset of Ω
  Empty bag:     ∅ = { }          ← no members
```

**Subset:** every member of the smaller bag is already in the bigger bag ($A \subset \Omega$).

**Micro:** Is $\{1,7\}$ a subset of the die $\Omega$? **No** — $7$ is not a face.

### Random experiment and sample space

A **random experiment (RE)** is a procedure you treat as able to produce different outcomes (toss a die, image a patient, open an email).

The **sample space** $\Omega$ is the set of **all outcomes you admit** for that experiment.

```
  One die toss:        Ω = {1,2,3,4,5,6}
  Coin once:           Ω = {H, T}
  Clinic process:      Ω = { process stories … }   (need not be numbers!)
```

You **choose** the experiment: “one toss” and “five tosses” are different REs with different $\Omega$s.

### Events and the menu $F$

An **event** is a subset of $\Omega$ we are allowed to score with probability.

The **event space** $F$ is the collection of allowed events — think of it as the **menu** of questions you are allowed to ask (“even?”, “six?”, “≤ 3?”).  
For this course you only need: inverse images of “reasonable” number-regions should land in $F$ so $P$ can score them. (You do not need full σ-algebra theory here.)

### Intersection and union (“and” / “or”)

| Symbol | English | Die micro |
|--------|---------|-----------|
| $A \cap B$ | both | even **and** in $\{1,2,3\}$ → $\{2\}$ |
| $A \cup B$ | either (or both) | even **or** in $\{1,2,3\}$ → $\{1,2,3,4,6\}$ |

```
  Die Ω = {1,2,3,4,5,6}
  A = even = {2,4,6}
  B = {1,2,3}
  A ∩ B = {2}
```

**Micro:** Even **and** greater than 4 → $A\cap B=\{6\}$. One outcome, later one score with $P$.

### Probability measure $P$

$P$ assigns each allowed event a number in $[0,1]$ with the usual rules ($P(\Omega)=1$; disjoint pieces add).  
$P$ sizes **sets of outcomes**, not free-floating adjectives (“lucky,” “typical”).

**Triplet to memorize:**

```
  (Ω, F, P)  =  outcomes · allowed events · how we size them
```

### Analogy — purpose of this section

A **library catalog** of every book that could be checked out today:

- $\Omega$ = the full catalog of possible checkouts.  
- An event = a shelf query (“all mystery novels,” “books under 200 pages”).  
- $P$ = how large that query is under today’s borrowing traffic.  
- $\cap$ = “mystery **and** under 200 pages” — only books that match **both** filters.

**Why does the video need this?** Because a joint will be: “reading 1 in region A **and** reading 2 in region B” — two filters, one catalog, one size.

If you invent two unrelated catalogs with no shared library, “both” has no honest home.

---

## 2. Functions, inverse images, RVs, and pushforward CDF

<a id="p2-rv-cdf"></a>
<a id="p2-functions"></a>

### Purpose for the video

Data vectors on the screen are not free-floating. They are **outputs** of a map from an underlying experiment.  
The lecture’s recap is: realizations live in the **range** of $X$; the CDF is $P$ **pushed forward** through $X$.  
Joints later are the same idea with **two** maps and an **intersection**.

### Function = one input → one output

A **function** $f:A\to B$ assigns each input in $A$ **exactly one** output in $B$.

```
  Same day  →  temperature number
  Same ω    →  X(ω)   (never two different X-values for one ω)
```

### Inverse image (preimage) — the video’s main tool

Given a set $S$ of **outputs**, pull back to inputs:

$$
X^{-1}(S)=\{\omega\in\Omega:X(\omega)\in S\}
$$

English: “which outcomes would have produced a reading inside $S$?”

```
  Coin: X(H)=1, X(T)=0
  S = (-∞, 0.5] = numbers ≤ 0.5
  X^{-1}(S) = {T}     because only T maps to 0
```

**Micro:** If $X(H)=1$, $X(T)=0$, what is $X^{-1}((-\infty,0.5])$? **$\{T\}$**.

### Half-lines (why “cumulative”)

$(-\infty, x]$ means **all reals ≤ $x$** (closed at $x$).  
A **CDF** is cumulative because it scores the whole pile “everything up through here,” not only the single point $x$.

```
  number line:  ··· ═══════]x
                 (-∞, x]
```

### Random variable and realizations

A **random variable** $X:\Omega\to\mathbb{R}$ (or $\mathbb{R}^{d}$) is a **deterministic function**.  
What is “random”? Which $\omega$ occurs — not the rule $X$ flipping on the same $\omega$.

**Realizations** = measured values = points in the **range** of $X$ (what you actually write down).

### CDF / distribution (scalar pushforward)

$$
P_X(x)=P\big(X^{-1}((-\infty,x])\big)=P(\{\omega:X(\omega)\le x\})
$$

In English: pull the half-line back to $\Omega$, then size that event with $P$.  
That is the **pushforward** of $P$ through $X$ — also called the distribution / CDF of $X$.

```
  (Ω, F, P) --X--> (numbers, Borel-ish regions, P_X)
  data = points in range(X)
```

**Micro (fair coin):** $X(H)=1$, $X(T)=0$.  
$P_X(0.5)=P(X\le 0.5)=P(\{T\})=1/2$.  
$P_X(1)=P(X\le 1)=P(\{H,T\})=1$.

**Trap:** the printed number is a realization; probability still lives on **sets** of outcomes (or on range regions via $P_X$), not on the ink of a single coordinate alone.

### Analogy — purpose of this section

A **weather station** prints a temperature:

- The atmosphere process ≈ $\Omega$ (what actually happened).  
- The sensor rule ≈ $X$ (always turns that process into a number the same way).  
- The printed “22°C” ≈ realization.  
- “Chance the printout is ≤ 20°C” ≈ $P_X(20)$ — pull “≤ 20” back to which weather stories would have produced that, then size them.

**Why does the video need this?** Every joint/conditional later still sits on this spine: maps + preimages + $P$. Multi-RV tools do not replace it; they **stack** on it.

---

## 3. Multiple readings from one experiment (many maps, one $\Omega$)

<a id="p3-multi-maps"></a>

### Purpose for the video

Machine learning almost never sees one lonely number. It sees **bundles**: pixels + label, features + target, temperature + humidity.  
The lecture’s move: put those readings as **several functions on the same sample space**, so “both at once” is legal.

### Many functions, same domain

Between two sets you can define **many** functions, not just one.

```
  Same Ω = {days in a year}
  X1(day) = temperature
  X2(day) = humidity
```

Both $X_1$ and $X_2$ are legal maps $\Omega\to\mathbb{R}$.  
In probability language: **multiple random variables on the same sample space**.

Each gets its own distribution $P_{X_1}$, $P_{X_2}$ from the **same** underlying $P$ (each pushforward uses its own map).

### Why share one $\Omega$?

Because “temperature and humidity **today**” is one experiment story.  
Inventing a brand-new unrelated $\Omega$ for every number makes “both at once” harder to define cleanly — there is no shared day that both readings come from.

**Micro:** One school day. Map 1 = temperature. Map 2 = humidity. Same day, two readings — two RVs, one $\Omega$.

### Notation preview from the lecture

He sometimes writes scalar RVs in **lowercase** ($x_1,x_2$) when stressing coordinates/scalars, and **capital** $X$ when stressing a vector map. Same idea; different emphasis.

### Analogy — purpose of this section

A **patient visit** is one story ($\Omega$).

- Reading 1: blood pressure  
- Reading 2: heart rate  
- Reading 3: temperature  

**Can you ask “high BP *and* high temperature on that visit”?** Only if both readings share the **same visit** as domain.  
If BP is from Monday’s clinic and temperature is from a random day in another city with no link, “and” is storytelling, not a joint event.

**Why does the video need this?** Topics 3–4 build joints from multiple maps on one $\Omega$. Without shared domain, the intersection construction collapses.

**Trap:** inventing a second unrelated $\Omega$ every time you need a second number — that move fails when you ask about both numbers at once.

---

## 4. Inverse images, product regions, and joints (“both at once”)

<a id="p4-joint-preimage"></a>

### Purpose for the video

This is the **core new definition** of the lecture: what “joint distribution” *is*.  
Not a software table first — a **probability of one event** built from two preimages and $\cap$.

### One RV (review)

$X^{-1}(S)=\{\omega:X(\omega)\in S\}$.

### Two RVs — the joint skeleton

“$X_1$ lands in $S_1$ **and** $X_2$ lands in $S_2$” is the intersection:

$$
X_1^{-1}(S_1)\;\cap\;X_2^{-1}(S_2)
$$

For the **joint CDF** at $(a,b)$, the regions are cumulative half-lines:

$$
P_{X_1,X_2}(a,b)
= P\big(X_1^{-1}((-\infty,a])\;\cap\;X_2^{-1}((-\infty,b])\big)
= P(X_1\le a,\ X_2\le b)
$$

English pipeline:

1. Ask both cumulative questions.  
2. Pull each question back to $\Omega$ (two events).  
3. Intersect → one event $E$.  
4. Score $P(E)$.

### Product region in the plane (see it)

$(-\infty,a]\times(-\infty,b]$ is the infinite southwest rectangle: all points with first coordinate ≤ $a$ **and** second ≤ $b$.

```
        b |
          |████████
          |████████  ← region (X1≤a and X2≤b)
          |████████
        --+-------- a -->
```

On $\Omega$, that rectangle becomes the **intersection** of the two preimages. The joint scores **one** subset of $\Omega$.

### Worked micro (tiny discrete joint)

Two binary readings from the **same** toss story:

```
              Y=0    Y=1   | row sum = P(X=·)
  X=0         0.1    0.2   |  0.3
  X=1         0.3    0.4   |  0.7
  col sum     0.4    0.6   |  1.0
```

- Joint cell $P(X=0,Y=1)=0.2$ — both at once.  
- Product of marginals $P(X=0)P(Y=1)=0.3\times 0.6=0.18$ — **different**.

So the joint is **not** “always multiply the separate one-variable chances.”

### Independence (only a special case)

If (and only if) the variables are **independent**, the joint factors as a product of marginals.  
Independence is a **property** you may assume or test later — **not** the definition of joint.

### Analogy — purpose of this section

Two exam scores for the **same** student on the **same** exam day:

- Threshold 1: score1 ≤ 70  
- Threshold 2: score2 ≤ 80  

**Which students count for “both thresholds”?** Only students who clear **both** cuts on that one day.

Wrong story: take everyone who ever scored ≤70 in *any* class, multiply by everyone who ever scored ≤80 in *another* class, and call that a joint.  
Right story: one student-day, intersect the two conditions, then size that set.

**Why does the video need this?** Topic 4 *is* this construction. Everything later (conditionals, margins) assumes you can form that joint event.

**Trap:** defining joint as “two separate $P$s multiplied” without checking dependence.

---

## 5. Vectors $\mathbb{R}^{d}$ vs $d$ numbers (two views, one object)

<a id="p5-vector-vs-scalars"></a>

### Purpose for the video

Images are points in $\mathbb{R}^{d}$. The lecture wants you to hear **two equivalent stories**:

1. One map $X:\Omega\to\mathbb{R}^{d}$ (vector-valued RV).  
2. $d$ scalar maps $X_1,\ldots,X_d$ on the same $\Omega$ with a joint.

That equivalence is the **license** for the rest of the course (“WLOG work with a $d$-dim vector RV”).

### One point, two languages

A point $(x_1,\ldots,x_d)\in\mathbb{R}^{d}$ is **one vector** or **$d$ coordinates**. Same object.

Same for RVs:

```
  View A:  one map  X: Ω → R^d
  View B:  d maps   X1,...,Xd : Ω → R  (same Ω)
```

These views are **equivalent** when the scalars live on the same sample space.

- **Cartesian product** on the range: $(-\infty,x_1]\times\cdots\times(-\infty,x_d]$  
  = “all coordinates ≤ their thresholds at once.”  
- On $\Omega$: that product becomes the **intersection** of the individual preimage events.

```
  Range:   product of half-lines
  Ω side:  ∩ of scalar preimages
  Same cumulative question, two geometries
```

### Micros

**Clinic form:** $d$ fields after one visit (age, BP, labs…). One form **or** $d$ answers — both correct. You do not need $d$ hospitals.

**Image:** stack pixels into one long vector in $\mathbb{R}^{d}$. That is one vector RV **or** $d$ coordinate RVs sharing the imaging experiment.

### Different dimensions still allowed

Two different RVs may have **different** range dimensions (image in $\mathbb{R}^{d}$, label in $\mathbb{R}^{k}$ or $\{0,1\}$). They still need a **shared** $\Omega$ to interact.

### Course license (preview)

Without loss of generality, the lecture treats a $d$-dimensional vector-valued RV as the standard object. When he says that, hear either View A or View B — not a third competing theory.

### Analogy — purpose of this section

A **shipping form** with $d$ fields after one package:

- weight  
- length  
- width  
- …  

**Is that one shipment record or $d$ numbers?** Both.  
Treating “one form” and “$d$ answers” as enemies fails; they describe the same shipment.

**Why does the video need this?** Topics 2 and 5 exist so that multi-pixel / multi-feature data are not a new species of math — they are the same multi-RV joint idea in vector packaging.

**Trap:** treating the vector view and the $d$-scalar view as incompatible formalisms that cannot talk to each other.

---

## 6. Conditional probability — shrink the world (events first)

<a id="p6-conditional-events"></a>

### Purpose for the video

ML constantly asks: “given the label / given the image side-info, what about the other variable?”  
That is **conditioning**. The lecture starts with events, then lifts the same idea to RVs.

### Definition (events)

If $A,B$ are events and **$P(B)>0$** (cannot condition on an impossible event):

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
$$

**English:** given that $B$ has **already** occurred, how large is $A$ **inside** the shrunken world $B$?

```
  Full space Ω
     └── shrink to B          (evidence arrived)
           └── how much of that is also A?
```

### Why divide by $P(B)$?

After evidence $B$, the total probability mass of the remaining world is $P(B)$, not $1$.  
You renormalize so probabilities in the smaller world still add up properly.  
If $P(B)=0$, there is **no** positive-probability world to shrink into — division by zero is undefined.

### Worked micro (fair die)

$B=\{\text{even}\}=\{2,4,6\}$, $A=\{6\}$.

| Quantity | Value |
|----------|-------|
| $P(A\cap B)=P(\{6\})$ | $1/6$ |
| $P(B)$ | $1/2$ |
| $P(A\mid B)$ | $(1/6)/(1/2)=1/3$ |

Not $1/6$ anymore — the world shrank to three faces, and only one is a six.

### Analogy — purpose of this section

You walk into a room of people who **all already passed a filter** (event $B$ = “even die,” or “patients with symptom S”).

**Question:** among *those* people, how many also have property $A$?  
Not: among everyone in the city, how many have $A$.

Wrong move: keep using the city-wide rate after the filter already applied.  
Right move: throw away everyone outside $B$, then recompute the rate of $A$ inside the room.

**Why does the video need this?** Topic 6 is this idea for RVs: fix $Y=y$ (the filter), then describe $X$. Without event conditioning, “given disease” is empty English.

**Trap:** conditioning does **not** mean “multiply by a density height.” Start from events: intersect, then renormalize by $P(B)$.

---

## 7. Conditional distributions, marginals, and $P$ vs $p$

<a id="p7-conditional-marginal"></a>

### Purpose for the video

The lecture’s continuous write-up is:

$$
p(x\mid y)=\frac{p(x,y)}{p(y)}
$$

with $y$ **fixed**, and $p(y)$ a **marginal**.  
This section makes those three words (conditional, joint, marginal) a **toolkit**, not three disconnected slogans — and warns about $P$ vs $p$ before densities arrive.

### Conditional distribution of RVs

For RVs $X,Y$ on the same $\Omega$, **$P(X\mid Y=y)$** means: **fix** $Y$ at value $y$, then describe the distribution of $X$ in that restricted world.

**Discrete (exact cell formula):**

$$
P(X=x\mid Y=y)=\frac{P(X=x,\ Y=y)}{P(Y=y)}\qquad (P(Y=y)>0)
$$

**Continuous notation (as written in class; densities next lecture):**

$$
p(x\mid y)=\frac{p(x,y)}{p(y)}
$$

Always: the conditioner is **fixed**. The lecture writes $p(x\mid y)$ without spelling “$y=y$,” but $y$ is still held fixed — the effective sample space shrinks to “$Y=y$.”

**Micro scene:** Image features $X$ and disease label $Y$.  
$p(x\mid y=1)$ = “what do images look like **given** disease is present?” — label fixed, describe $X$.  
$p(y\mid x)$ would fix the image side and ask about the label — same toolkit, swapped roles.

### Marginal — remove the other variable

**Marginal of $X$:** behavior of $X$ alone after “removing” $Y$.

- Discrete: **sum** the joint table over $y$  
- Continuous: **integrate** the joint over $y$

$$
p(x)=\sum_y p(x,y)\quad\text{or}\quad p(x)=\int p(x,y)\,dy
$$

(same idea for marginal of $Y$ by summing/integrating over $x$).

**Why the word “marginal”?** For a discrete joint **table**, summing along the **margins** (edges) of the table gives the single-variable distributions. Literally the margin of the table.

```
         Y=0   Y=1   | row sum → marginal of X
  X=a    0.1   0.2   | 0.3
  X=b    0.3   0.4   | 0.7
  -------+-----+-----+
  col →  0.4   0.6     ← margins for Y
```

**Paper micro:** $P(X=a)=0.3$, $P(Y=1)=0.6$, $P(X=a\mid Y=1)=0.2/0.6=1/3$.

Plain idea: nullify the other RV by adding/integrating over all values it can take; what remains is one RV alone. Conditionals **need** this object in the denominator.

### Notation care: capital $P$ vs lowercase $p$

| Symbol | Meaning in this course arc |
|--------|----------------------------|
| Capital $P$, $P_X$, $P(A\mid B)$ | Sizes **events / sets** (or CDF-style evaluations of those sets) |
| Lowercase $p(x)$, $p(x,y)$ | **Density-style** heights (continuous “how thick is the cloud here”) — full story next lecture |

The lecture already writes $p(x\mid y)=\text{joint}/\text{marginal}$ while postponing density theory. Treat that line as **structural preview**: same shrink-and-renormalize idea as $P(A\mid B)$, written in density notation.  
Do **not** confuse density height with an event probability by itself (Lec 04 trap).

### Notation care (integrals)

Capitals $X,Y$ are **functions**. Integrals are over their **ranges** (sets of numbers), not “over the function object.” Writing $\int dx$ means integrate over the range of $X$. If the range is $\mathbb{R}^{3}$, that is a triple integral — dimension follows the range.

### How the three pieces fit (one picture)

```
           joint p(x,y)
          /            \
   fix y, divide      sum/∫ out y
   by p(y)               \
      ↓                   ↓
  conditional         marginal p(x)
   p(x|y)
```

You cannot honestly compute the continuous conditional without a joint and a marginal. That is why the lecture introduces all three as one family.

### Analogy — purpose of this section (conditional)

A **filtered patient ward**: only disease-present patients ($Y=1$) are in the room.  
You walk the room and study image patterns $X$.  
That walk is $p(x\mid y=1)$ — not “images of all humans on Earth.”

### Analogy — purpose of this section (marginal)

A joint table of **weather × ice-cream sales**.

- rainy + low / rainy + high / sunny + low / sunny + high  

**What is “sales alone” if you no longer care about weather?** Sum along the weather direction — the edge of the table.  
Weather did not leave the universe; you **averaged it out of the table**.

**Why does the video need this?** Topic 6 introduces conditionals that **literally divide by a marginal**; Topic 7 defines that denominator and names it. Without both, the continuous formula is half-empty.

**Trap (conditional):** leave $y$ random inside $p(x\mid y)$ instead of fixing it.  
**Trap (marginal):** treat the marginal as a second unrelated experiment instead of “joint with a coordinate removed.”

---

## 8. Why this lecture exists (ML X-ray thread + how the map fits)

<a id="p8-why"></a>

### Purpose for the video

Probability so far (Lec 02–04) gave you **one** RV and its CDF.  
This lecture equips **multi-coordinate / multi-variable** data with:

- joints (“both / all at once”)  
- conditionals (“given this, what about that?”)  
- margins (“this alone, after averaging the other out”)

Then it points back at the course’s running **X-ray → disease** problem: next class will ground that story in this language and introduce **densities**.

### What changed

| Earlier (Lec 02–04) | Now (this lecture) |
|---------------------|--------------------|
| One RV + CDF | Several RVs / vector RV |
| Single pushforward | Joints, conditionals, margins |
| Abstract tools | Ready to ground **X-ray → disease** |

```
  Image features X ∈ R^d     (vector RV or d pixel coordinates)
  Label Y ∈ {0,1}            (disease / non-disease)
  Joint p(x,y)
  Conditional p(y|x) or p(x|y)
  Marginals p(x), p(y)
```

### What questions this toolkit answers (purpose)

| Question in ML words | Probability name |
|----------------------|------------------|
| How do image and label co-occur? | Joint of $(X,Y)$ |
| Given disease, what do images look like? | $p(x\mid y=1)$ |
| Given this image, how likely is disease? | $p(y\mid x)$ (same toolkit, swapped) |
| Ignore labels — how do images look alone? | Marginal $p(x)$ |
| Ignore images — how common is disease? | Marginal $p(y)$ |

### Analogy — purpose of the whole lecture

A **clinic archive**:

- Every row is one patient visit (shared experiment story).  
- Columns: image-like measurements + disease bit.  

Without joints, you only have separate column histograms that do not know they came from the same visit.  
Without conditionals, you cannot say “given disease…” honestly.  
Without margins, you cannot drop a column and still have a valid single-column story from the same table.

**That three-tool kit is the purpose of Lec 05.** Densities (how continuous “height” objects work) wait until next time.

### What is *not* this lecture (so you do not feel lost)

- Full density theory / change of variables  
- Full independence theory as a chapter  
- Finished formal X-ray model write-up (started only; grounded next)  
- A substitute for a whole probability course (primer / refresher)

### How the eight warm-ups unlock the video map

```
  §1 floor: sets, ∩, (Ω,F,P)
       ↓
  §2 data = range of X; CDF = pushforward
       ↓
  §3 many maps on one Ω
       ↓
  §4 joint = P(pre1 ∩ pre2)
       ↓
  §5 vector packaging ≡ d scalars
       ↓
  §6–7 condition + margin toolkit (+ P vs p)
       ↓
  §8 why: X-ray/disease needs all of the above
```

### Paper check (do before NOTES)

1. Write $A\cap B$ for two die events and compute $P(A\cap B)$ if fair.  
2. Why can one $\Omega$ host two RVs? One sentence + example.  
3. State $P(A\mid B)$ including $P(B)>0$; interpret “shrink then score.”  
4. From the table in §4, compute $P(X=1\mid Y=0)$.  
5. What does a table **margin** mean for a discrete joint?  
6. In one line: vector RV $X:\Omega\to\mathbb{R}^{d}$ ≡ $d$ scalars on the same $\Omega$.  
7. Why is $P(X=0)P(Y=1)$ not automatically the joint cell?  
8. In one sentence: purpose of joints + conditionals + margins for X-ray/disease.  
9. Capital $P$ vs lowercase $p$ — one careful sentence.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz: [quiz.html](./quiz.html) (Part A = this file).  
Prior: [Lec 04 Part 3](../05-Lec04-Recap-Probability-Theory-Part3/NOTES.md).
