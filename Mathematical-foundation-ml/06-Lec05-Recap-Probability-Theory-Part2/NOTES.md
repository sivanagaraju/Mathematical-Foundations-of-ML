# Lec 05 — Recap of Probability Theory Part 2

**Video:** [Lec 05 Recap of Probability Theory Part 2](https://www.youtube.com/watch?v=R69wew8RrPo) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Lec 04 — CDF / pushforward](../05-Lec04-Recap-Probability-Theory-Part3/NOTES.md)

---

## Table of Contents

1. [Topic 1 — Recap: realizations and pushforward CDF](#topic-1-recap-realizations-and-pushforward-cdf-0000–0204) (00:00–02:04)
2. [Topic 2 — Vector-valued random variables](#topic-2-vector-valued-random-variables-0204–0353) (02:04–03:53)
3. [Topic 3 — Multiple RVs on one sample space](#topic-3-multiple-rvs-on-one-sample-space-0353–0724) (03:53–07:24)
4. [Topic 4 — Joint distribution via inverse-image intersection](#topic-4-joint-distribution-via-inverse-image-intersection-0724–1117) (07:24–11:17)
5. [Topic 5 — Vector RV equals d scalars on the same Ω](#topic-5-vector-rv-equals-d-scalars-on-the-same-ω-1117–1311) (11:17–13:11)
6. [Topic 6 — Conditional probability and distributions](#topic-6-conditional-probability-and-distributions-1311–1721) (13:11–17:21)
7. [Topic 7 — Marginals and the X-ray bridge](#topic-7-marginals-and-the-x-ray-bridge-1721–2142) (17:21–21:42)
8. [External references](#external-references)
9. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Job: equip multi-coordinate data (images, labels) with joint, conditional, and marginal language. Method: put several RVs on the **same** $\Omega$, form joints by **intersecting preimage events**, then condition and marginalize. Fork: a vector RV $X:\Omega\to\mathbb{R}^{d}$ is the **same object** as $d$ scalar RVs — two views, one structure.

**Worldview arc:** from single-RV pushforward CDF **to** joints / vector RVs / conditionals / margins (X-ray ML grounding + densities next).

### System context

```
  ╔══════════════════════════════════╗
  ║ Outside                          ║
  ║ · Lec 02–04: Ω,F,P,X,P_X         ║
  ║ · densities (next class)         ║
  ║ · ML as distribution learning    ║
  ╚══════════════╤═══════════════════╝
                 │ this lecture adds
                 ▼
        ┌────────────────────┐
        │ joints · cond. ·   │
        │ margins on RVs     │
        └────────────────────┘
```

### Main blueprint

```
  Recap:  (Ω,F,P) --X--> P_X (CDF)
                 │
                 ▼
         X : Ω → R^d
                 │
        ┌────────┴────────┐
        ▼                 ▼
   one vector RV     d scalar RVs
   Ω → R^d           X1..Xd on same Ω
        └────────┬────────┘
                 ▼
   Joint: P(X1≤a, X2≤b)
        = P( X1^{-1}((-∞,a]) ∩ X2^{-1}((-∞,b]) )
                 │
        ┌────────┼────────┐
        ▼                 ▼
   P(X|Y=y)           margins
   = joint/marginal   sum/integrate other out
                 │
                 ▼
   STOP: ground X-ray disease problem; densities next
```

### Scenario walkthrough

1. Data vectors are realizations of RVs with underlying $\Omega$ and $P_X$.  
2. An image in $\mathbb{R}^{d}$ is one vector RV **or** $d$ coordinate RVs.  
3. Joint scores “all coordinates / both variables” via $\cap$ of preimages.  
4. Condition: fix label $Y=y$, describe image $X$.  
5. Marginal: ignore one variable by summing/integrating it out.  
6. Next: write the X-ray disease task in this language; then densities.

### Failure / contrast path

```
  Treat coordinates as unrelated Ωs always  ──X──► breaks joint story
  Confuse density height with P(A|B)        ──X──► wrong object
  Forget conditioner is fixed at a value    ──X──► fuzzy conditional
```

### STOP / out of scope

Density definitions, full multi-RV calculus course, finished ML problem statement (started only). Primer, not a substitute for a rigorous probability class.

### Load-bearing claims (closed-book)

- Realizations = range points of RV; CDF is pushforward of $P$.  
- Vector RV: $X:\Omega\to\mathbb{R}^{d}$.  
- Many RVs can share one $\Omega$.  
- Joint = $P$ of intersection of preimage events (not default product).  
- Vector RV ≡ $d$ scalars on same $\Omega$ (Cartesian product ↔ $\cap$).  
- $P(A\mid B)=P(A\cap B)/P(B)$ ($P(B)>0$); RVs: fix $Y=y$; ranges may differ.  
- Marginal = joint with other variable summed/integrated out (table edges).  
- $P$ sizes sets; $p$ is density-style (densities next).  
- Next: X-ray grounding + densities.

**Speaker / course:** NPTEL IISc · MFML · Lec 05.

---

## Topic 1: Recap — realizations and pushforward CDF (00:00–02:04)

### Where this sits on the master map

**RECAP** box — reload the stack before multi-RV tools. Warm-up: [RV + CDF](./PREREQUISITES.md#p2-rv-cdf).

### Board / screenshot

![Recap of RV and CDF](./screenshots/composites/ch01-topic-01-recap-panel1of1.png)

**Figure — ~00:00–02:00:** sample space, measures, RV realizations as range vectors; pushforward CDF reminder.

### What he is establishing

This class continues probability fundamentals and begins relating **vector-valued random variables** and **joint distributions** to machine learning examples.

Quick stack from earlier: sample space, event spaces, probability measures; random variables exist because practice works with **realizations** — mathematically, **vectors from the range** of the function called a random variable. Whenever you measure from a probabilistic standpoint, remember there is an **underlying sample space** that gave rise to the RV, and a probability measure **pushed forward** into a **probability distribution function / CDF**.

The trap is treating a data vector as free-floating with no $\Omega$ behind it — that story fails. Instead keep the full picture: the vector is a range point of $X$, and $P_X$ is the pushforward of $P$.

We now have the single-RV stack restated. Still missing: multi-coordinate / multi-RV structure.

### Analogy for this topic only

A clinic prints one blood-pressure number after a visit.

- Visit A → 120  
- Visit B → 135  
- Visit C → 128  

Someone asks: **what process produced those three readings?** Memory of the numbers alone does not answer “why these values and not others.”

Right move: treat each printout as a reading from an underlying process, not as “the process itself.” Wrong move: pretend the printed number floated into existence with no experiment behind it.

In lecture words: process ≈ $\Omega$; reading map ≈ $X$; printed number ≈ realization; CDF ≈ pushforward of $P$.

### Local picture

```
  Ω --X--> R^d
  P  ──pushforward──►  P_X (CDF / distribution)
  data = points in range(X)
```

**Notice:** every measured vector quietly carries $\Omega$ and $P$ with it.

### Bridge

A single scalar range cannot describe an image stack of $d$ coordinates. The leftover problem is how to let $X$ land in $\mathbb{R}^{d}$.

---

## Topic 2: Vector-valued random variables (02:04–03:53)

### Where this sits on the master map

**VECTOR RV** box. Warm-up: [vectors vs scalars](./PREREQUISITES.md#p5-vector-vs-scalars).

### Board / screenshot

![Vector-valued RV definition](./screenshots/composites/ch02-topic-02-vector-rv-panel1of1.png)

**Figure — ~02:04–03:50:** $X:\Omega\to\mathbb{R}^{d}$; RV as function; vector-valued language.

### What he is establishing

Earlier, $X$ was often scalar-valued: $\Omega\to\mathbb{R}$. Generalize: the range/codomain can be **$\mathbb{R}^{d}$**, a $d$-dimensional real space. He keeps calling the object a **function** — that is what a random variable is, despite the misnomer.

So $X$ maps sample-space elements to points in $\mathbb{R}^{d}$. Such objects are **vector-valued random variables**.

The trap is treating “vector-valued” as something mystical beyond “outputs are vectors.” Instead: the range is $\mathbb{R}^{d}$, and the map is still deterministic given $\omega$. Writing $X:\Omega\to\mathbb{R}^{d}$ is the whole definition so far.

We now have vector RVs. Still missing: how this relates to several scalar RVs and joints.

### Analogy for this topic only

One clinic visit fills $d$ boxes on one form:

- age  
- height  
- blood pressure  
- …  

**Is that one form or $d$ separate numbers?** Memory of “$d$ loose numbers with no visit” fails — you lose that they came from the **same** visit.

Right move: one form is one vector (or $d$ coordinates of one visit). Wrong move: invent $d$ unrelated stories when one visit produced all fields.

In lecture words: one map $\Omega\to\mathbb{R}^{d}$.

### Local picture

```
  Ω  ──X──►  R^d
  ω  ──X──►  (x1,...,xd)
```

**Notice:** $d$ is the dimension of the **range**, not “how random” the experiment is.

### Bridge

One vector can also be read as $d$ scalar maps — but then “both coordinates at once” needs joint language.

---

## Topic 3: Multiple RVs on one sample space (03:53–07:24)

### Where this sits on the master map

**MULTI-RV** box — after the vector definition, open the second view ($d$ scalars) and the set tools that make joints possible. Warm-up: [multiple maps](./PREREQUISITES.md#p3-multi-maps), [sets and ∩](./PREREQUISITES.md#p1-sets-and).

### Board / screenshot

![Multiple RVs on same Ω](./screenshots/composites/ch03-topic-03-multi-rv-panel1of1.png)

**Figure — ~03:53–07:20:** one vector map vs $d$ scalars; $x_1,x_2$ on same $\Omega$; set operations enable combining RVs.

### What he is establishing

A vector RV can be seen two ways: (1) **one** function $\Omega\to\mathbb{R}^{d}$; (2) a **collection of $d$ scalar-valued** RVs. To connect those views we need **multiple random variables** and **joint distributions**.

Fix one sample space for a random experiment. Define two functions $x_1,x_2:\Omega\to\mathbb{R}$ (he deliberately uses lowercase for these scalars). You can always define many functions between the same pair of sets — so many RVs can share $\Omega$.

Each has its own distribution function, because each induces a pushforward of $P$.

From set theory: if events $A,B$ have measures, $P$ also handles **unions and intersections**. Inverse images of RVs land in the event space $F$. Therefore there should be a way to **combine** two RVs using those set operations — that combination is the idea of **joint distributions**.

The trap is inventing a second unrelated $\Omega$ every time you need a second number — that move fails when you ask about both numbers at once. Instead prefer multiple maps on one $\Omega$ when they share one experiment — that is where joints live.

We now can place several RVs on one $\Omega$ and expect joints. Still missing: the explicit joint formula $P_{X_1,X_2}(a,b)$.

### Analogy for this topic only

One school day is the experiment.

- Reading 1: temperature 28°C  
- Reading 2: humidity 70%  

**Can you ask “was it hot *and* humid that day” from two unrelated calendars?** No — without a shared day, “and” has no home.

Right move: same day $\Omega$, two maps. Wrong move: two separate $\Omega$s with no link, then pretend “both” is well-defined.

In lecture words: multiple functions on one sample space.

### Local picture

```
  Ω ──x1──► R     with P_{x1}
  Ω ──x2──► R     with P_{x2}
  Events from preimages live in F
  ∩ and ∪ on F ⇒ room for joints
```

**Notice:** “joint” starts as set intersection of preimage events, not as a software table first.

### Bridge

We can place two RVs on one $\Omega$, but we still lack a clean formula for “both land in their cumulative regions.” That formula is the joint $P_{X_1,X_2}(a,b)$.

---

## Topic 4: Joint distribution via inverse-image intersection (07:24–11:17)

### Where this sits on the master map

**JOINT** box — core definition. Warm-up: [joint preimage](./PREREQUISITES.md#p4-joint-preimage).

### Board / screenshot

![Joint distribution definition](./screenshots/composites/ch04-topic-04-joint-panel1of1.png)

**Figure — ~07:24–11:10:** joint $P_{X_1,X_2}$ at two points; inverse images; intersection event; extend to $d$ RVs.

### What he is establishing

The **joint distribution** of $X_1$ and $X_2$, evaluated at two points (he writes $a,b$ for notation), is the probability measure of an event $E$ built as the **intersection of inverse images** under $X_1$ and $X_2$ of the corresponding regions.

Which regions? The same cumulative half-lines used for a scalar CDF: $(-\infty,a]$ for $X_1$ and $(-\infty,b]$ for $X_2$. On the plane that is the southwest product rectangle $(-\infty,a]\times(-\infty,b]$. On $\Omega$ it is the intersection of two preimage events.

In plain form:

$$
P_{X_1,X_2}(a,b)
= P\big(X_1^{-1}((-\infty,a])\;\cap\;X_2^{-1}((-\infty,b])\big)
= P(X_1\le a,\ X_2\le b)
$$

In words: both RVs land in their cumulative regions; pull each region back to $\Omega$; intersect; apply $P$. One joint score, one subset of $\Omega$.

Extend from two to **$n$ or $d$** scalar RVs on the same sample space — a joint among all of them uses the intersection of **all** their preimages (or equivalently the product region in $\mathbb{R}^{d}$).

The trap is defining a joint as “two separate $P$s multiplied” without checking dependence — that is a special case (independence), not the definition. Instead always go through a **single event** on the shared $\Omega$ (intersection of preimages).

We now have the joint via $\cap$ of preimages. Still missing: explicit identity with vector RVs.

### Analogy for this topic only

Two exam scores on the **same** student day:

- Score 1 threshold: ≤ 70  
- Score 2 threshold: ≤ 80  

**Which students count for “both thresholds”?** Only students who clear **both** cuts on that one day — not “anyone who ever scored ≤70 in some other class unioned with someone else who scored ≤80.”

Right move: one student-day story, intersect the two conditions. Wrong move: multiply two separate class-wide rates and call it a joint by default.

In lecture words: joint = $P$ of the intersection event.

### Local picture

```
  Range view (plane):
        b |████
          |████  (-∞,a] × (-∞,b]
        --+---- a -->

  Sample-space view:
  E = X1^{-1}((-∞,a]) ∩ X2^{-1}((-∞,b])
  P_{X1,X2}(a,b) = P(E)

  Extend: d scalars → ∩ of d preimages (one joint)
```

**Notice:** the joint is still a measure of **one** subset of $\Omega$ — product geometry on the range, intersection geometry on $\Omega$.

### Bridge

Show that “vector RV” and “$d$ scalar RVs with a joint” are the same object.

---

## Topic 5: Vector RV equals $d$ scalars on the same $\Omega$ (11:17–13:11)

### Where this sits on the master map

**EQUIVALENCE** box — license for the rest of the course: vector view and $d$-scalar view are the same structure. Warm-up: [vector vs scalars](./PREREQUISITES.md#p5-vector-vs-scalars).

### Board / screenshot

![Equivalence of vector and scalar RVs](./screenshots/composites/ch05-topic-05-equivalence-panel1of1.png)

**Figure — ~11:17–13:10:** vector RV as $d$ scalars; Cartesian products ↔ intersection of events; course assumes $d$-dim vector RV.

### What he is establishing

A **vector-valued random variable** can be seen as a **collection of $d$ scalar-valued random variables** defined on the **same** sample space — they are **exactly the same** idea.

For the rest of the course, without loss of generality, work with a $d$-dimensional vector-valued RV. When he says that, you should hear either: one map $\Omega\to\mathbb{R}^{d}$ with its distribution, **or** $d$ coordinate maps with a joint.

The cumulative region in $\mathbb{R}^{d}$ is a **Cartesian product** of half-lines (from last time). Going back to $\Omega$, that product corresponds to the **intersection** of the individual scalar preimage events.

The trap is treating the two views as competing theories that cannot both be true. Instead switch freely when convenient: a product region on the range is an intersection of scalar preimage events on $\Omega$.

We now can translate vector CDF regions into intersections of scalar events. Still missing: conditioning.

### Analogy for this topic only

A patient form has $d$ fields after one visit:

- age  
- BP  
- lab marker  
- …  

**Do you need $d$ hospitals to have $d$ answers?** No — one visit, $d$ fields. Treating “one form” and “$d$ answers” as enemies fails; they describe the same visit.

Right move: one $\Omega$, $d$ coordinates (or one vector). Wrong move: two incompatible formalisms that cannot talk to each other.

In lecture words: vector RV ≡ $d$ scalars on the same sample space.

### Local picture

```
  X: Ω → R^d
    ≡
  X1,...,Xd : Ω → R   (same Ω)
    with
  joint of (X1,...,Xd)

  Region ∏(-∞,xi]  ↔  ∩_i Xi^{-1}((-∞,xi])
```

**Notice:** Cartesian product on the range ↔ intersection on the sample space.

### Bridge

Conditionals: shrink the world given one event/RV value.

---

## Topic 6: Conditional probability and distributions (13:11–17:21)

### Where this sits on the master map

**CONDITIONAL** box. Warm-up: [event conditionals](./PREREQUISITES.md#p6-conditional-events), [RV conditionals](./PREREQUISITES.md#p7-conditional-marginal).

### Board / screenshot

![Conditional probability and distributions](./screenshots/composites/ch06-topic-06-conditional-panel1of1.png)

**Figure — ~13:11–17:20:** $P(A|B)$; conditional distributions $P(X|Y)$; conditioner fixed; joint/marginal form.

### What he is establishing

With multiple RVs on the same $\Omega$, you can define **conditional distributions** because **conditional probabilities** exist for events. Set-theoretic probability tools transfer into the pushforward / distribution world (same idea as last classes: what works for events works after pushforward).

**Event definition:** for events $A,B$ with **$P(B)>0$** (cannot condition on an impossible event),

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
$$

Interpret: given $B$ has already occurred (evidenced), what is the probability $A$ occurs?

**For RVs:** if $X$ and $Y$ are RVs on the same $\Omega$, define conditional distribution $P(X\mid Y)$. Both may be **vector-valued**, and their ranges need **not** match — one can live in $\mathbb{R}^{d}$ and the other in $\mathbb{R}^{k}$ (image vs label is the ML picture). Evaluate at some $x$ **given $Y$ fixed to a value $y$**. The conditioner is always fixed; that shrinks the effective sample space.

He writes $p(x\mid y)$ without spelling “$Y=y$,” but **always** read $y$ as held fixed. Defined (continuous case, as written in class) using the **joint** divided by the **marginal** of the conditioner:

$$
p(x\mid y)=\frac{p(x,y)}{p(y)}
$$

**Notation preview:** capital $P$ sizes events / CDFs; lowercase $p$ is density-style writing. Densities are the next lecture — treat $p(x\mid y)$ here as the **same shrink-and-renormalize idea** as $P(A\mid B)$, not as “height equals probability of a point.” Discrete analog for later tables: $P(X=x\mid Y=y)=P(X=x,Y=y)/P(Y=y)$.

From now on, “a random variable” in this course usually means a **$d$-dimensional vector-valued** RV $\Omega\to\mathbb{R}^{d}$ unless said otherwise.

The trap is treating $p(x\mid y)$ as if $y$ were still random inside the same expression — that fails. Instead fix $y$, then describe $x$.

We now have conditionals for events and for RVs. Still missing: what “marginal” is and why the name (needed in the denominator).

### Analogy for this topic only

Given the die landed even, what is the chance it is a six?

- Full world: faces $\{1,2,3,4,5,6\}$, six has chance $1/6$.  
- Given even: world shrinks to $\{2,4,6\}$.  

**Is six still $1/6$ inside the shrunken world?** No — only one of three even faces is a six → $1/3$.

Right move: intersect with “even,” then renormalize by $P(\text{even})$. Wrong move: keep using $1/6$ after the evidence arrived, or treat “given even” as a density height with no set.

In lecture words: condition = intersect then renormalize by $P(B)$; for RVs, fix $y$ the same way.

### Local picture

```
  Events:  P(A|B)=P(A∩B)/P(B)   (need P(B)>0)

  RVs:     p(x|y) = p(x,y) / p(y)
           y is fixed  (even if we only write "p(x|y)")
           world shrinks to {Y=y}

  Ranges:  X in R^d , Y in R^k allowed
```

**Notice:** conditional is still a probability/distribution in the smaller world.

### Bridge

Define marginals carefully; then return to X-ray ML.

---

## Topic 7: Marginals and the X-ray bridge (17:21–21:42)

### Where this sits on the master map

**MARGINAL + STOP**. Warm-up: [marginal name](./PREREQUISITES.md#p7-conditional-marginal), [why / X-ray](./PREREQUISITES.md#p8-why).

### Board / screenshot

![Marginals and course bridge](./screenshots/composites/ch07-topic-07-marginal-xray-panel1of1.png)

**Figure — ~17:21–21:40:** marginal via integral; discrete table margins; notation abuse; X-ray disease problem teaser; densities next.

### What he is establishing

**Marginal distribution** of one RV: from the joint, remove the other by **integrating** (continuous case, integrals assumed well-defined) or **summing** (discrete).

Continuous write-up (as on the board):

$$
p(x)=\int p(x,y)\,dy,\qquad p(y)=\int p(x,y)\,dx
$$

(integrals over the **range** of the variable being removed). Discrete twin: replace $\int$ by $\sum$ over the finite/countable values of the other variable.

**Why the word “marginal”?** For discrete RVs the joint is a **table**. Evaluating along the **margins** (edges) of the table yields the single-variable distributions. Literally the margin of the table — that etymology is why the continuous object keeps the same name.

Plain idea: if two RVs share $\Omega$, nullify one by adding/integrating over all values it can take; what remains is the behavior of the other alone. Conditionals need this object in the denominator ($p(x\mid y)=p(x,y)/p(y)$).

You can also form the marginal of $Y$ by integrating out $X$. Dimension: if the range is $\mathbb{R}^{3}$, you face triple integrals, etc. Writing $\int dx$ is notation for integrating over the **range** of $X$, not “integrating a function object.” Capitals $X,Y$ are functions; integration is over ranges — he calls the sloppy $\int dy$ an **abuse of notation** and corrects it in class.

**Primer disclaimer:** this is a quick refresher, not a full probability course; he assumes a prior rigorous class.

**Bridge to ML:** remember the opening problem — X-ray and disease / non-disease. Next: ground that example in joint/conditional language, then go to **density functions**.

The trap is treating a marginal as a second unrelated experiment. Instead: the marginal is the joint with a coordinate integrated or summed out. That definition, together with joints and conditionals, is exactly the language the X-ray disease task needs.

We now have margins. Still missing (next lectures): full grounding of the disease/X-ray problem in this notation, and density functions.

### Analogy for this topic only

A joint table of weather × ice-cream sales:

- rainy + low sales  
- rainy + high sales  
- sunny + low sales  
- sunny + high sales  

**What is “sales alone” after you no longer care about weather?** Sum along the weather direction — the edge of the table.

Right move: nullify weather by summing/integrating it out. Wrong move: invent a brand-new sales experiment with no joint history.

In lecture words: margin = edge sum / integral of the joint.

### Local picture

```
  Continuous:  p(x) = ∫ p(x,y) dy   (over range of Y)
               p(y) = ∫ p(x,y) dx   (over range of X)
  Discrete:    sum the table over the other variable
               table edges = "margins"

  Abuse: "∫ dY" means integrate over range(Y), not over the function Y

  Next: X-ray + disease in this language; then densities
```

**Notice:** conditional needs marginals in the denominator; joints, conditionals, and margins are one toolkit.

### Bridge

Ground the disease/X-ray problem in joints and conditionals; introduce densities next.

---

## External references

| Resource | Matches lecture… | Why it helps |
|----------|------------------|--------------|
| [StatQuest — Conditional Probability, Clearly Explained](https://www.youtube.com/watch?v=H02B3aMNKzE) | Topic 6 | Event-first $P(A\mid B)$ intuition |
| [StatQuest — Probability vs Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4) | Topic 6–7 | Keeps “fixed conditioner / scored data” clean |
| [3Blue1Brown — Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) | Topic 6 conditionals | Visual conditioning |
| [MIT 6.041SC — Probabilistic Systems](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/) | Topics 4–7 | Joints / conditionals / margins depth |
| [Lec 04 package (this series)](../05-Lec04-Recap-Probability-Theory-Part3/NOTES.md) | Topic 1 recap | Pushforward CDF stack |
| [Seeing Theory — Conditional Probability](https://seeingtheory.brown.edu/basic-probability/index.html#section2) | Topics 6–7 | Interactive conditioning |

---

## Sources

- Video: [Lec 05 Recap of Probability Theory Part 2](https://www.youtube.com/watch?v=R69wew8RrPo)  
- Channel: NPTEL — Indian Institute of Science, Bengaluru  
- Skill: `youtube-lecture-tutor`  
- Captions cleaned in TRANSCRIPT.md / claim sheets  
