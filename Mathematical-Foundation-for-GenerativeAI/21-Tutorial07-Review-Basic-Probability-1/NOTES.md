# Tutorial 7 — Review of Basic Probability 1

**Video:** [Tutorial 7 : Review of Basic Probability 1](https://www.youtube.com/watch?v=owlWCCgYx50) · NPTEL / IISc
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)
**Previous:** [Tutorial 6 — Transfer Learning](../20-Tutorial06-Transfer-Learning-PyTorch/NOTES.md)
**Course:** Mathematical Foundations of **Generative AI** (~50 min)
**Speaker:** NPTEL IISc · Triplet, conditionals, Bayes, independence flavors, RVs, discrete CDF/PMF

---

## Table of Contents

1. [Topic 1 — Probability triplet and three axioms](#topic-1-probability-triplet-and-three-axioms-0003–0450) (00:03–04:50)
2. [Topic 2 — Conditional probability](#topic-2-conditional-probability-0450–0747) (04:50–07:47)
3. [Topic 3 — Law of total probability](#topic-3-law-of-total-probability-0747–1039) (07:47–10:39)
4. [Topic 4 — Bayes rule](#topic-4-bayes-rule-1039–1206) (10:39–12:06)
5. [Topic 5 — Independence of two events](#topic-5-independence-of-two-events-1206–1431) (12:06–14:31)
6. [Topic 6 — Total, pairwise, and conditional independence](#topic-6-total-pairwise-and-conditional-independence-1431–1900) (14:31–19:00)
7. [Topic 7 — Random variables and pushforward](#topic-7-random-variables-and-pushforward-1900–2512) (19:00–25:12)
8. [Topic 8 — CDF definition and properties](#topic-8-cdf-definition-and-properties-2512–3235) (25:12–32:35)
9. [Topic 9 — Discrete RV, staircase CDF, and PMF](#topic-9-discrete-rv-staircase-cdf-and-pmf-3235–4247) (32:35–42:47)
10. [Topic 10 — Named discrete families and recap](#topic-10-named-discrete-families-and-recap-4247–5006) (42:47–50:06)
11. [External references](#external-references)
12. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

This hour recaps the probability objects the rest of the course will keep using. Start from a **random experiment (RE)** — a procedure you can run again without knowing the result — whose outcomes live in a sample space $\Omega$, whose events $P$ scores, and whose three axioms make $(\Omega,\mathcal{F},P)$ a **probability space**. After you learn that $B$ already happened you reweigh every event; a partition rebuilds $P(A)$; Bayes flips the easy lab number into the number you actually want. Independence is a product test (with total, pairwise, and conditional cousins); a random variable is a function $X:\Omega\to\mathbb{R}$, not a floating number. A **cumulative distribution function (CDF)** $F(x)=P(X\le x)$ specifies the law; a discrete $X$ is equally specified by a **probability mass function (PMF)**. You leave with named discrete families (Bernoulli, indicator, binomial, Poisson, geometric); continuous $X$, expectation, and variance wait until next time.

**Worldview arc:** from “named objects $\Omega$, $\mathcal{F}$, $P$” **to** “discrete $X$ completely specified by a staircase CDF or a mass function.”

**Hour at a glance (whole video).** The first half is the *stage*. A **random experiment (RE)** is a repeatable procedure whose result is not known in advance. Every complete result is an outcome; the bag of all outcomes is the **sample space** $\Omega$; an **event** is a subset; the **event space** $\mathcal{F}$ is the menu of subsets you may score (here they take every subset). **Probability** $P$ assigns each event a number and must stay nonnegative, give the whole of $\Omega$ the score 1, and add on events that cannot happen together. Those three objects are the **probability triplet**. The triplet alone does not update. If you learn that $B$ already happened, **conditional probability** throws away $\Omega\setminus B$ and stretches the leftover so $B$ now has mass 1 — but only if $P(B)>0$. If you instead cut $\Omega$ into disjoint rooms (a **partition**), the **law of total probability** rebuilds $P(A)$ as a weighted mix of the room-wise conditionals. **Bayes** is that mix used backwards: the lab published “positive given disease”; you want “disease given positive.” Independence is not “they feel unrelated”: two events are independent when $P(A\cap B)=P(A)P(B)$. For a list you must say *which* test — **total** (every subcollection factors), **pairwise** (only pairs), or **conditional independence** given $C$ (tests can be independent given the disease and still dependent if you do not know the disease).

The second half leaves the abstract bag. Computation wants **real numbers**, so a **random variable** is a deterministic function $X:\Omega\to\mathbb{R}$ (the slogan: neither random nor a variable). It **pushes** $P$ onto the line as a new measure $P_X$. One working function packages that law: the **cumulative distribution function (CDF)** $F(x)=P(X\le x)$, the running total of mass from $-\infty$ up to $x$. The CDF specifies the law of $X$. About 80% of later course time is continuous and 20% discrete; today they finish the discrete side. A discrete $X$ takes countably many values; its CDF is a **staircase**; each jump is the mass at that point — the **probability mass function (PMF)**. Either the CDF or the PMF completely specifies a discrete law. Named piles close the hour: **Bernoulli**, **indicator** $1_B$, **binomial**, **Poisson**, **geometric**. Next tutorial replaces piles by densities and adds expectation.

### System context

```
  ╔══════════════════════════════════════╗
  ║ Outside: GenAI sampling / densities  ║
  ║ Outside: full first-course proofs    ║
  ╚══════════════╤═══════════════════════╝
                 │ this tutorial (~50 min)
                 ▼
        ┌────────────────────────────┐
        │ Recap probability stack    │
        │ triplet → condition → RV   │
        └────────────────────────────┘
                 │
                 ▼
        next: continuous RVs, E, Var
```

### Main blueprint

```
  Random experiment
          │ outcomes
          ▼
  ┌──────────── Ω ────────────┐
  │ sample space of ω         │
  └────────────┬──────────────┘
               │ subsets
               ▼
  ┌──────────── F ────────────┐
  │ events (practice: 2^Ω)    │
  └────────────┬──────────────┘
               │ P : F → ℝ
               ▼
  ┌──────────── P ────────────┐
  │ ≥0 · P(Ω)=1 · σ-additivity│
  │ (Ω, F, P) = space/triplet │
  └──────┬──────────┬─────────┘
         │          │
         │ given B  │ partition {B_i}
         ▼          ▼
   P(·|B) new    total law
   assignment    P(A)=∑ P(A|B_i)P(B_i)
         │          │
         └──── Bayes ┘
               │
               ▼
        independence tests
        (2 / total / pair / |C)
               │
               ▼
        X : Ω → ℝ   (deterministic)
               │ pushforward
               ▼
        (ℝ, B, P_X)
               │
               ▼
        CDF F  (all RVs) · PMF p (discrete)
               │
               ▼
        Bernoulli / 1_B / Binomial / Poisson / Geometric
```

### Scenario walkthrough

Walk this **one** story through the blueprint above. Each step answers “so what?” for the next box.

**Story:** a clinic wants to know whether a rare disease is present after a lab test comes back positive. (A die or a coin is the same shape of objects; they use those for the arithmetic.)

1. **Why a triplet?** “A person walks in” is a **random experiment (RE)**. $\Omega$ is every possible person-state. “Has the disease” and “test is positive” are **events**. $P$ scores them. That is the SETUP box.

2. **Why conditionals?** You hear the test is positive. That is event $B$. You throw away everyone who tested negative and reweigh. That is $P(\text{disease}\mid\text{positive})$.

3. **Why a partition?** You do not know $P(\text{positive})$ directly. Cut the world into disease / no-disease. Mix the two rooms. That is the total law.

4. **Why Bayes?** The lab published the easy number $P(\text{positive}\mid\text{disease})$. You want the flipped number. Cartoon numbers on the board: prior $1\%$, true-positive $99\%$, false-positive $1\%$ $\to$ posterior $1/2$, **not** $99\%$.

5. **Why independence flavors?** Two tests can be **conditionally independent given the disease** and still dependent if you do not know the disease (they share a hidden cause). Pairwise independence of three events is weaker than total independence.

6. **Why a random variable?** “Has disease” as a yes/no sticker is $X:\Omega\to\{0,1\}$. “Test positive” is another sticker $Y$ on the **same** $\Omega$.

7. **Why a CDF / PMF?** Walk the line: $F(x)=P(X\le x)$ is the **cumulative distribution function (CDF)**. For this Bernoulli, $F$ is a two-step staircase and the **probability mass function (PMF)** is the two piles $p$ and $1-p$.

8. **Why named families?** The same piles get names: Bernoulli / indicator for one test, binomial for $n$ independent tests, geometric for “wait until first positive,” Poisson for an uncapped count. That is the catalog. Continuous $X$ is next hour.

```
  person walks in  =  random experiment
         │
         ▼
  Ω = possible person-states     events: disease, test+
         │  learn test+
         ▼
  reweigh (conditional)  →  mix rooms (total law)
         │  flip the lab number
         ▼
  Bayes: P(disease | test+)   (1% / 99% / 1%  →  1/2)
         │  stickers on the same Ω
         ▼
  X = 1{disease}   Y = 1{test+}
         │  CDF = running total; PMF = piles
         ▼
  Bernoulli / binomial / geometric / Poisson
```

Same chain for a die or three coins: experiment $\to$ $\Omega$ $\to$ update $\to$ function $X$ $\to$ CDF / PMF $\to$ named piles.

### Failure / contrast path

```
  “I’ll write P(A|B) even if P(B)=0”              ──X──► undefined here
  “Just add P(A)+P(B)” when A and B overlap       ──X──► you double-count
  “Pairwise independent ⇒ totally independent”    ──X──► false
  “X is just a random number”                     ──X──► you hid the map Ω → ℝ
  Mixing script P (the measure) with P_X (the CDF)──X──► two different objects
  “A continuous X has a PMF”                      ──X──► wrong object
```

### STOP / out of scope

Continuous random variables and densities; expectation and variance; a full measure-theoretic $\sigma$-algebra course; live coding (mentioned as a plan, not run in this video).

### Load-bearing claims (closed-book)

- $(\Omega,\mathcal{F},P)$ is the **probability space**; $P$ obeys **three axioms** (nonnegative, whole space $=1$, add on disjoint events).
- $P(A\mid B)=P(A\cap B)/P(B)$ only if $B$ is a legal event with $P(B)>0$; it is a **new assignment**, not decoration on $P(A)$.
- The **total law** rebuilds $P(A)$ from a partition; **Bayes** rewrites the other conditional.
- Independence is the **product test** $P(A\cap B)=P(A)P(B)$; total, pairwise, and conditional cousins are not the same word.
- A **random variable** is a function $X:\Omega\to\mathbb{R}$ and induces a pushforward $P_X$.
- The **cumulative distribution function (CDF)** accumulates $P(X\le x)$; it **specifies** the law of $X$.
- A **discrete** $X$ has a **probability mass function (PMF)**; the CDF is a **staircase** whose jumps are the masses.

**Speaker / course:** NPTEL IISc · Tutorial 7.

---

## Topic 1: Probability triplet and three axioms (00:03–04:50)

### Where this sits on the master map

**SETUP** — Recap the three objects every later formula sits on. Warm-up: [sets](./PREREQUISITES.md#p1-sets) · [experiment](./PREREQUISITES.md#p3-re) · [mass](./PREREQUISITES.md#p4-mass).

### Board / screenshot

![Triplet and axioms](./screenshots/composites/ch01-topic-01-triplet-axioms-panel1of1.png)

**Figure — ~01:40–04:50:** Sample space $\Omega=\{\omega_1,\omega_2,\ldots\}$; event space $\mathcal{F}\subseteq 2^\Omega$ (they take $\mathcal{F}=2^\Omega$); $P:\mathcal{F}\to\mathbb{R}$; axioms non-negativity, normalization $P(\Omega)=1$, $\sigma$-additivity for mutually exclusive $A_i$; red label $(\Omega,\mathcal{F},P)$ **probability space**. Early spoken minutes are face-cam; the board tiles carry the definitions.

### What he is establishing

This block is a **recap**. The professor already introduced sample space, event space, and probability measure. The tutorial restates them so the rest of the course can move fast, and it will stress **when a formula is legal** more than it will invent new theory.

A **random experiment** is a repeatable procedure whose result is not known in advance. Every complete result is an **outcome**. The set of all outcomes is the **sample space** $\Omega$. Outcomes need not be numbers — heads/tails, menu dishes, bit strings are fine.

Work one die: $\Omega=\{1,2,3,4,5,6\}$. “Even” is not an outcome; it is the subset $\{2,4,6\}$. “Roll a 4” is the singleton $\{4\}$. Both will later get masses $1/2$ and $1/6$.

An **event** is a subset of $\Omega$. The **event space** $\mathcal{F}$ (script $F$) is the collection of events you are allowed to talk about. Formally $\mathcal{F}$ is a subset of the **power set** $2^\Omega$. For this recap they take the practical default

$$
\mathcal{F} = 2^\Omega

$$

so **every** subset of $\Omega$ is an event. That is enough for discrete examples in this hour.

**Probability** is a function that assigns each event a number in $[0,1]$ and must obey three **axioms**:

$$
P : \mathcal{F} \to \mathbb{R}

$$

1. **Non-negativity.** $P(A)\ge 0$ for every event $A$. Negative probability is not allowed.
2. **Normalization.** $P(\Omega)=1$. The whole sample space is the certain event.
3. **$\sigma$-additivity.** If $A_1,A_2,\ldots\in\mathcal{F}$ are **mutually exclusive** ($A_i\cap A_j=\emptyset$ for $i\neq j$), then

$$
P\Bigl(\bigcup_i A_i\Bigr) = \sum_i P(A_i)

$$

Mutual exclusion is the green light for turning union into a sum. If the sets overlap you **cannot** add blindly.

The three objects together $(\Omega,\mathcal{F},P)$ are the **probability triplet**, also called the **probability space**. Every later definition — conditional, independence, random variable — lives on this triple.

You can now name the three objects and refuse a negative $P$. What is still open: how to **update** $P$ after you learn that some event $B$ already happened.

A common trap is treating $\Omega$ as “the numbers we care about.” $\Omega$ is outcomes; numbers arrive later via a map $X$. Another trap is adding probabilities of overlapping events and calling it $\sigma$-additivity.

### Analogy for this topic only

A bakery has a list of every pastry that can come out of the oven ($\Omega$). “Chocolate” is a subset (an event). You have 1 kg of dough to spread over the list ($P$). You never assign a negative weight. The whole list gets the full kilogram. If two labels never share a pastry, their weights add.

Question: **What three rules must that 1 kg assignment obey?**

In lecture words: this box is $(\Omega,\mathcal{F},P)$ plus the three axioms.

### Local picture

```
  RE ──outcomes──► Ω = {ω}
                    │ subsets
                    ▼
                   F  (practice: every subset)
                    │ P
                    ▼
              numbers in [0,1]
              P(Ω)=1 · P≥0 · disjoint ⇒ add
```

**Notice:** the triplet is the *stage*. Conditionals and RVs are actors on that stage.

### Bridge

The axioms tell you how $P$ behaves on a **fixed** space. They do not yet tell you how to change $P$ after you **observe** an event $B$.

---

## Topic 2: Conditional probability (04:50–07:47)

### Where this sits on the master map

**UPDATE** — New assignment given $B$. Warm-up: [given that](./PREREQUISITES.md#p6-given).

### Board / screenshot

![Conditional probability](./screenshots/composites/ch02-topic-02-conditional-panel1of1.png)

**Figure — ~04:50–07:47:** $B\in\mathcal{F}$ and $P(B)>0$; $P(A\mid B)=P(AB)/P(B)$; intersection written $AB$; $P(B\mid B)=1$; “new probability assignment.”

### What he is establishing

Let $B$ be an event. Two gates must both open before you write $P(A\mid B)$:

- $B\in\mathcal{F}$ (it is a legal event), and
- $P(B)>0$ (it has positive mass).

Then, for any event $A$,

$$
P(A\mid B) = \frac{P(A\cap B)}{P(B)}

$$

The board often writes $AB$ for $A\cap B$. That is **notational abuse**; they use it on purpose and want you to notice.

Read the formula as a **new probability assignment**. After you know $B$ occurred, every event gets a new number. The new mass of $A$ is only the part of $A$ that sits inside $B$, stretched so that $B$ itself now has mass $1$:

$$
P(B\mid B) = 1

$$

In words: the new probability of each event is determined by **what is common with $B$**. You have already learned $B$; you throw away $\Omega\setminus B$ and re-normalize.

The same idea extends to more than two events. The last board line writes a three-letter version

$$
P(A\mid BC) = \frac{P(ABC)}{P(BC)}

$$

which is the same definition after you treat “$B$ and $C$ both happened” as the new world (still need $P(BC)>0$).

Work a die so the formula is not abstract. $B=\{\text{even}\}=\{2,4,6\}$, $A=\{4,5,6\}$. Then $A\cap B=\{4,6\}$, $P(A\cap B)=1/3$, $P(B)=1/2$, so $P(A\mid B)=2/3$. Among the three remaining faces, two sit in $A$.

You can now refuse $P(A\mid B)$ when $P(B)=0$, and you can reweigh a die after “even” is announced. Still missing: how to recover $P(A)$ when you only know conditionals on pieces of a **partition**.

A common trap is treating $P(A\mid B)$ as “$P(A)$ with extra decoration.” It is a **different function** of $A$. Another trap: writing $P(A)/P(B)$ and dropping the intersection.

### Analogy for this topic only

Six hotel rooms. You learn the guest is in an **even** room. Rooms 1, 3, 5 drop out. Belief that was spread over six doors is now spread over three. The chance they are in $\{4,5,6\}$ becomes two-out-of-three, not the old $1/2$.

Question: **What two conditions must $B$ satisfy before $P(A\mid B)$ is defined here?**

In lecture words: this box is a new $P$ given $B$.

### Local picture

```
  Ω  [======== B ========][ leftover ]
         A∩B sits here

  new mass of A  =  (old mass of A∩B) / (old mass of B)
```

**Notice:** divide by $P(B)$ so the new total on $B$ is $1$.

### Bridge

One $B$ updates one assignment. Often you do not have a single $B$ — you have a **cover of $\Omega$ by disjoint slices**. How do those slices rebuild $P(A)$?

---

## Topic 3: Law of total probability (07:47–10:39)

### Where this sits on the master map

**MIX** — Rebuild $P(A)$ from a partition. Warm-up: [sum vs product](./PREREQUISITES.md#p7-arith).

### Board / screenshot

![Total probability](./screenshots/composites/ch03-topic-03-total-probability-panel1of1.png)

**Figure — ~07:47–10:39:** Partition $B_1,\ldots,B_m$; $A$ as a blue region cut into pie slices $AB_i$; $\sigma$-additivity then $P(A)=\sum_i P(A\mid B_i)P(B_i)$. Board title: **Total probability rule**.

### What he is establishing

A **partition** of $\Omega$ is a list $B_1,\ldots,B_m$ of events that

- do not overlap: $B_i\cap B_j=\emptyset$ for $i\neq j$, and
- cover the whole space (their union is $\Omega$).

Draw four rooms $B_1,\ldots,B_4$ (think: four weather types, or four lab machines). Paint an event $A$ in blue that crosses several rooms — “the test is positive,” or “the die shows at least 5.” Then $A$ is exactly the union of the overlaps

$$
A = \bigcup_{i=1}^{m} (A \cap B_i)

$$

Each slice $A\cap B_i$ lives in a different room, so the slices are **disjoint**. $\sigma$-additivity therefore turns the union into a sum:

$$
P(A) = \sum_i P(A \cap B_i)

$$

Feed in the definition of conditional probability, $P(A\cap B_i)=P(A\mid B_i)P(B_i)$, and you get the **law of total probability**

$$
P(A) = \sum_i P(A\mid B_i)\, P(B_i)

$$

In words: pick a room at random with chance $P(B_i)$, then look at the chance of $A$ **inside that room**, and mix.

They again treat union and “and” notation loosely; a first course should already be comfortable with that abuse.

You can now cut $P(A)$ along a partition. Still missing: the two-way swap that turns $P(A\mid B)$ into $P(B\mid A)$.

A common trap is using a list of $B_i$ that **leak** (overlap) or **miss** part of $\Omega$. Then the pie identity fails.

### Analogy for this topic only

A circular pizza is the whole sample space. Four cuts make four slices. Event $A$ is “has olives.” Olives sit on several slices. Total olive-area is olive-area on slice 1 plus slice 2 plus … — **because slices do not overlap**.

Question: **Why may you replace the probability of the glued olive pieces by a sum?**

In lecture words: this box is the total probability rule.

### Local picture

```
   Ω = B1 | B2 | B3 | B4     (disjoint, cover)
   A  = a1 | a2 | a3 | a4     a_i = A ∩ B_i

   P(A) = Σ P(A | B_i) P(B_i)
```

**Notice:** the weights $P(B_i)$ themselves add to $1$.

### Bridge

Total law expands a **denominator**. Bayes uses that expansion to **flip** a hard conditional into an easier one.

---

## Topic 4: Bayes rule (10:39–12:06)

### Where this sits on the master map

**FLIP** — Interchange the easy and hard conditional. Warm-up: [given that](./PREREQUISITES.md#p6-given).

### Board / screenshot

![Bayes rule](./screenshots/composites/ch04-topic-04-bayes-panel1of1.png)

**Figure — ~10:39–12:06:** $P(A\mid B)=P(AB)/P(B)=P(B\mid A)P(A)/P(B)$; denominator expanded with partition $\{A,A^c\}$; independence heading starts at the bottom.

### What he is establishing

Start from the definition and the multiplication rule:

$$
P(A\mid B) = \frac{P(A\cap B)}{P(B)} = \frac{P(B\mid A)\,P(A)}{P(B)}

$$

If $P(B)$ is ugly, expand it with the **two-set partition** $\{A, A^c\}$. Those two events are mutually exclusive and cover $\Omega$, so the total law applies:

$$
P(B) = P(B\mid A)\,P(A) + P(B\mid A^c)\,P(A^c)

$$

That is **Bayes’ rule** in the form they write on the board.

The operational reason to care: **one conditional is often easier than the other**. You want “disease given a positive test.” The lab published “positive test given disease.” Bayes swaps them after you mix in false alarms on the no-disease side.

Cartoon numbers (not from a notebook — from the same story they tell): suppose $P(A)=0.01$ (rare disease), $P(B\mid A)=0.99$ (true positive), $P(B\mid A^c)=0.01$ (false positive). Then

$$
P(A\mid B) = \frac{0.99\cdot 0.01}{0.99\cdot 0.01 + 0.01\cdot 0.99} = \frac12

$$

The flipped number is **not** $99\%$. That is why the denominator’s $A^c$ term exists.

You can now flip a conditional and expand the denominator on $\{A,A^c\}$. Still missing: when the flip is unnecessary because $A$ and $B$ **do not inform** each other.

A common trap is forgetting $A^c$ in the denominator and writing only $P(B\mid A)P(A)$. Another trap: applying Bayes when $P(B)=0$.

### Analogy for this topic only

You hear a fire alarm. You want the chance there is a fire given the alarm. The factory published the chance of an alarm given a fire, which is easy. Bayes turns the factory number into the number you actually need, after you account for false alarms when there is no fire.

Question: **Which two events form the partition used in the board’s denominator?**

In lecture words: this box is Bayes as “use the easy conditional.”

### Local picture

```
  want P(A|B)
       │
       ▼
  P(B|A) P(A)  /  [ P(B|A)P(A) + P(B|A^c)P(A^c) ]
                     └── total law on {A, A^c} ──┘
```

**Notice:** $\{A,A^c\}$ is the smallest useful partition.

### Bridge

Sometimes $P(A\mid B)$ equals $P(A)$ already. That special case has a name, and a product test that does not mention conditionals at all.

---

## Topic 5: Independence of two events (12:06–14:31)

### Where this sits on the master map

**NO INFORMATION** — Product test for two events. Warm-up: [product](./PREREQUISITES.md#p7-arith).

### Board / screenshot

![Two-event independence](./screenshots/composites/ch05-topic-05-two-event-independence-panel1of1.png)

**Figure — ~12:06–14:31:** $A,B$ independent iff $P(AB)=P(A)P(B)$; then $P(A\mid B)=P(A)$ as a **consequence**; complement exercise.

### What he is establishing

Events $A,B\in\mathcal{F}$ are **independent** when

$$
P(A\cap B) = P(A)\,P(B)

$$

That is the **definition**. Intersection on the set side becomes **multiplication** on the probability side. The instructor’s side remark: union/intersection in set space correspond, loosely, to plus/times in probability space.

If $P(A)>0$ and $P(B)>0$, plug the definition into the conditional formula:

$$
P(A\mid B) = \frac{P(A)P(B)}{P(B)} = P(A)

$$

and likewise $P(B\mid A)=P(B)$. So **conditional equals unconditional**. That is a **consequence** of the product definition, not a replacement for it.

Exercise they leave you: if $A$ and $B$ are independent, so are $(A,B^c)$, $(A^c,B)$, and $(A^c,B^c)$. Sketch for $(A,B^c)$: $P(A\cap B^c)=P(A)-P(A\cap B)=P(A)-P(A)P(B)=P(A)(1-P(B))=P(A)P(B^c)$. The other two pairs are the same trick.

You can now test two events with a product and explain why $P(A\mid B)=P(A)$ follows. Still missing: **more than two** events, where “every pair multiplies” is not the same as “every subset multiplies.”

A common trap is starting from “they feel unrelated” instead of the product. Another trap: treating $P(A\mid B)=P(A)$ as the definition when $P(B)=0$ (the product definition still makes sense).

### Analogy for this topic only

Two coins that do not share a hinge. Chance both land heads is $\frac12\times\frac12$. If they were welded, “both heads” would not factor. Learning the first is heads would **not** change the second only when the product test holds.

Question: **Is “$P(A\mid B)=P(A)$” the definition or a corollary?**

In lecture words: this box is two-event independence.

### Local picture

```
  definition:   P(A ∩ B) = P(A) P(B)
  if P(B)>0:    P(A|B) = P(A)     (corollary)

  also independent: (A,B^c), (A^c,B), (A^c,B^c)
```

**Notice:** plus is for disjoint unions; times is for independence.

### Bridge

Two events have one product test. A list $A_1,\ldots,A_M$ has **several** tests that students keep collapsing.

---

## Topic 6: Total, pairwise, and conditional independence (14:31–19:00)

### Where this sits on the master map

**FLAVORS** — Do not collapse three different independence notions. Warm-up: [product](./PREREQUISITES.md#p7-arith).

### Board / screenshot

![Flavors of independence](./screenshots/composites/ch06-topic-06-flavors-independence-panel1of1.png)

**Figure — ~14:31–19:00:** Total independence (every subset’s intersection factors); pairwise (every pair); conditional independence given $C$; $P(A\mid B,C)=P(A\mid C)$.

### What he is establishing

**Total (mutual) independence** of a list of events (say three exam-day students, or $M$ coin tosses): take any $k$ between $1$ and $M$, and any $k$ of the events. Their “all happen” probability must equal the product of the $k$ individual probabilities. You cannot pick $k=M+2$ — there are only $M$ events. Intuition: **every subcollection** factors.

**Pairwise independence** is weaker. You only require the product test for **pairs** $i\neq j$:

$$
P(A_i \cap A_j) = P(A_i)\,P(A_j)

$$

Events may be pairwise independent and **fail** total independence. Exercise: if they are totally independent, are they pairwise independent? (Yes — pairs are allowed subcollections.)

**Conditional independence** of $A$ and $B$ **given** $C$:

$$
P(A\cap B\mid C) = P(A\mid C)\,P(B\mid C)

$$

A short algebra chase they sketch: $P(A\mid B,C)=P(A\mid C)$. Given $C$, learning $B$ does not change $A$.

Two corner cases to keep on the desk:

- $A,B$ may be **conditionally independent given $C$** but **not** independent. Classic story: several **independent medical tests** for a disease. Given the disease status, the tests do not talk; unconditionally they are dependent because they share the hidden disease.
- $A,B$ may be **independent** but **not** conditionally independent given some other $C$.

You can now name three tests and refuse to treat them as one word. Still missing: why we leave the abstract $\Omega$ and map into **real numbers**.

A common trap is “pairwise ⇒ total.” Another is “conditionally independent ⇒ independent.”

### Analogy for this topic only

Three classmates. Pairwise: every couple’s joint attendance factors. Total: the whole trio and every pair and every singleton factor. Conditional given “exam day”: given that fact, two students’ attendance no longer informs each other — even if, without that fact, they tend to skip together.

Question: **Can pairwise independence hold when total independence fails?**

In lecture words: this box is the independence menu.

### Local picture

```
  total     every subset   ∩  →  product
  pairwise  every pair     ∩  →  product     (weaker)
  cond. |C  inside C, A ⟂ B

  tests | disease   :  cond. indep., not ordinary indep.
```

**Notice:** always say *which* independence you mean.

### Bridge

Events live in $\Omega$, which may be dishes or bit-strings. Computation wants **real numbers**. That is the job of a random variable.

---

## Topic 7: Random variables and pushforward (19:00–25:12)

### Where this sits on the master map

**LAND IN ℝ** — Deterministic map, new measure $P_X$. Warm-up: [functions](./PREREQUISITES.md#p2-fn).

### Board / screenshot

![Random variable and pushforward](./screenshots/composites/ch07-topic-07-random-variable-panel1of1.png)

**Figure — ~19:00–25:12:** $X:\Omega\to\mathbb{R}$ on $(\Omega,\mathcal{F},P)$; coin $X(H)=1$, $X(T)=0$; new space $(\mathbb{R},\mathcal{B},P_X)$; $P_X(B)=P(X\in B)=P(\{\omega:X(\omega)\in B\})$.

### What he is establishing

We want to compute in **real numbers**. A **random variable** on $(\Omega,\mathcal{F},P)$ is a **real-valued function**

$$
X : \Omega \to \mathbb{R}

$$

The famous slogan (already in the lectures): a random variable is **neither random nor a variable**. It is a **deterministic function**. Randomness lives in which $\omega$ the experiment produces; once $\omega$ is in hand, $X(\omega)$ is fixed.

Coin: $\Omega=\{H,T\}$, $X(H)=1$, $X(T)=0$.

Hotel: $\Omega$ is the menu. One random variable maps each dish to its **price**. Another maps each dish to **calories**. Multiple random variables may live on the **same** probability space. Billing and calories are two different stickers on one menu.

Any $X$ **induces a new probability space** $(\mathbb{R},\mathcal{B},P_X)$. Here $\mathbb{R}$ is the new sample space, $\mathcal{B}$ a new event collection on the line, and $P_X$ a new measure. For $B\subset\mathbb{R}$ with $B\in\mathcal{B}$,

$$
P_X(B) = P\bigl(\{\omega\in\Omega : X(\omega)\in B\}\bigr) = P(X\in B)

$$

Go back to $\Omega$, collect every outcome whose sticker landed in $B$, and reuse the **old** $P$.

Coin check: $P_X(\{1\})=P(H)$.
Hotel check (one dish only): $P(\text{bill}=120)=P(\text{dishes priced }120)$.

They write $\{X\in B\}$ for the pre-image as notational convenience. $P_X$ satisfies the three axioms — it is a genuine probability measure.

You can now say $X$ is a function and compute $P_X$ by pulling $B$ back to $\Omega$. Still missing: a single function of $x\in\mathbb{R}$ that **packages** the whole law — the CDF.

A common trap is calling $X$ “the random number 0 or 1” and forgetting the pre-image. Another is thinking one $\Omega$ can carry only one $X$.

### Analogy for this topic only

Coat check. Each coat gets a ticket number. Asking “probability the ticket is 120” means: go back to the rack and add the probabilities of **every coat that received 120**. A second counter can issue calorie tickets on the same coats.

Question: **Where is the randomness — in the ticket rule, or in which coat appeared?**

In lecture words: this box is $X:\Omega\to\mathbb{R}$ and $P_X$.

### Local picture

```
  Ω --X--> ℝ
  {ω : X(ω)∈B}  maps to  B ⊂ ℝ

  P_X(B) = P( those ω )
  coin: P_X({1}) = P(H)
```

**Notice:** $P_X$ is **pushed** from $P$ along $X$.

### Bridge

$P_X$ still eats **sets** on the line. The working function in the rest of the hour eats a **point** $x$ and returns a running total.

---

## Topic 8: CDF definition and properties (25:12–32:35)

### Where this sits on the master map

**RUNNING TOTAL** — One function that specifies the law. Warm-up: [running totals](./PREREQUISITES.md#p8-cdf).

### Board / screenshot

![CDF properties](./screenshots/composites/ch08-topic-08-cdf-properties-panel1of1.png)

**Figure — ~25:12–32:35:** Script $P$ = measure; non-script $P_X$ = CDF; $P_X(x)=P(X\le x)$; values in $[0,1]$; limits $0$ and $1$; **nondecreasing**; $P(a<X\le b)=F(b)-F(a)$.

### What he is establishing

**Notation alert (load-bearing).** Script $\mathbb{P}$ (or script $P$) means a **probability measure**. The same letters **without** the script, written $P_X$, mean the **cumulative distribution function (CDF)** of $X$ in this tutorial. Do not mix them. The pushforward measure from Topic 7 is the scripted object; the CDF is an ordinary function $\mathbb{R}\to\mathbb{R}$.

They also say “distribution function” and “CDF” interchangeably.

$$
P_X(x) = P\bigl(\{\omega : X(\omega)\le x\}\bigr) = P(X\le x)

$$

Walk the real line from $-\infty$ up to $x$ and **accumulate** every bit of mass whose value is $\le x$. That is why it is called cumulative.

Coin $X\in\{0,1\}$: $P_X(-3)=0$ (nothing yet). At $0.5$ you have passed $0$ but not $1$, so $P_X(0.5)=P(X=0)$. After $1$ the CDF is $1$.

The CDF **completely specifies** the probability assignment of $X$: if you know $F(x)$ for every $x$, you can recover the law.

Properties they insist on:

- $0\le F(x)\le 1$ for all $x$ (it is a probability).
- $F(-\infty)=0$, $F(+\infty)=1$.
- $F$ is **nondecreasing**, not necessarily **increasing**. It may stay flat. On the coin, $F(0.5)=F(0.6)$.
- If $x_1\le x_2$ then $\{X\le x_1\}\subset\{X\le x_2\}$, so $F(x_1)\le F(x_2)$.
- $F$ is **right-continuous** and has left limits — left as an exercise (why needed; what fails without it).
- $P(a<X\le b)=F(b)-F(a)$. If you include the point $a$, add $P(X=a)$.

You can now evaluate a coin CDF and refuse the word “increasing” when the graph is flat. Still missing: the discrete case where $F$ is a **staircase** and the jump heights get their own name (PMF).

A common trap is reading $P_X$ as the pushforward measure. Another is claiming $F$ must strictly rise.

### Analogy for this topic only

Walk east along a road and pour every probability you pass into a measuring cup. The water level at mile $x$ is $F(x)$. Between two houses with no one living in between, the water level **does not change**. At a house, it may jump.

Question: **Why can $F(0.5)$ equal $F(0.6)$ on a coin?**

In lecture words: this box is the CDF and its properties.

### Local picture

```
  −∞ ----0---- 0.5 ----1---- +∞
  F:  0    ½     ½     1        (fair 0/1 coin)

  P(a < X ≤ b) = F(b) − F(a)
```

**Notice:** nondecreasing allows flats; increasing would forbid them.

### Bridge

The course will live ~80% in **continuous** RVs later, ~20% in **discrete**. Discrete $X$ makes the flats and jumps visible — and introduces the PMF.

---

## Topic 9: Discrete RV, staircase CDF, and PMF (32:35–42:47)

### Where this sits on the master map

**ATOMS** — Countable values; jump = mass. Warm-up: [countable](./PREREQUISITES.md#p5-count) · [running totals](./PREREQUISITES.md#p8-cdf).

### Board / screenshot

![Discrete RV and staircase](./screenshots/composites/ch09-topic-09-discrete-pmf-panel1of2.png)

![PMF vs CDF](./screenshots/composites/ch09-topic-09-discrete-pmf-panel2of2.png)

**Figure — ~32:35–42:47:** Discrete vs continuous (CDF for all); $X$ discrete iff countably many values; 3-coin $X\in\{0,1,2,3\}$; $q_i=P(X=x_i)$, $q_i\ge 0$, $\sum q_i=1$; jump $F(x)-F(x^-)=P(X=x)$; staircase sketch; PMF $p_X$ = mass at a point; $F(x)=\sum_{x_i\le x}p(x_i)$.

### What he is establishing

They will study **two kinds** of random variable (not claiming these are the only kinds): **discrete** and **continuous**. About 80% of later course time is continuous, 20% discrete. The **CDF is defined for every kind**.

$X$ is **discrete** if it takes only **countably many** distinct values — finite or countably infinite. Any RV defined on a **countable** $\Omega$ is discrete.

Example: toss a fair coin **three** times. $\Omega=\{H,T\}^3$ has eight strings. Let $X$ be the **number of heads**. Then $X\in\{0,1,2,3\}$ only. Another RV $Y=$ number of tails is a different map on the same $\Omega$ (exercise: list $Y$’s values).

Order the support $x_1<x_2<\cdots$ (rename if needed). Put mass $q_i=P(X=x_i)$ with $q_i>0$ and $\sum q_i=1$. The CDF jumps at each $x_i$ by exactly that mass:

$$
P_X(x) - P_X(x^-) = P(X=x)

$$

So the graph is a **staircase / step function**. For three fair coins the jumps are


| $x$ | jump$P(X=x)$          | $F(x)$ after the jump |
| ----- | ----------------------- | ----------------------- |
| $0$ | $1/8$ (TTT)           | $1/8$                 |
| $1$ | $3/8$ (HTT, THT, TTH) | $4/8$                 |
| $2$ | $3/8$                 | $7/8$                 |
| $3$ | $1/8$ (HHH)           | $1$                   |

The **probability mass function (PMF)** $p_X$ (small $p$) is the mass **at** a point: $p_X(x_i)=P(X=x_i)$, and $p_X=0$ off the support. CDF = probability **until** $x$; PMF = probability **at** $x$.

Relationship:

$$
F(x) = \sum_{x_i \le x} p_X(x_i)

$$

Example: $F(2.5)=p(0)+p(1)+p(2)$.

Any function $p\ge 0$ that sums to $1$ over a countable set **is** a PMF of some discrete RV. A discrete RV is **completely specified** by giving either its CDF or its PMF — you can reverse-engineer the atoms either way.

**PMF is only for discrete $X$.** Continuous RVs do not get a PMF (next tutorial).

You can now draw the three-coin staircase and move between $F$ and $p$. Still missing: the **named** PMFs the course will keep using.

A common trap is thinking CDF is only for continuous RVs. Another is putting a PMF on a continuous $X$.

### Analogy for this topic only

Four buckets sit at miles 0, 1, 2, 3. You dump one-eighth, three-eighths, three-eighths, one-eighth of a liter. The PMF is “how much in this bucket.” The CDF is “how much I have dumped **so far** as I walk east.” Between buckets the CDF is flat. At a bucket it jumps.

Question: **What is the running total after mile 2 for the three-fair-coin heads count?**

In lecture words: this box is discrete $X$, staircase, PMF.

### Local picture

```
  p:   1/8   3/8   3/8   1/8
  x:    0     1     2     3
  F:   1/8   4/8   7/8    1     (after each jump)

  F(x) = sum of p at all ticks ≤ x
```

**Notice:** understand a discrete RV = know every atom’s mass.

### Bridge

Some PMFs appear so often they get names and two-letter parameters. The last block is that catalog, then a recap.

---

## Topic 10: Named discrete families and recap (42:47–50:06)

### Where this sits on the master map

**CATALOG + CLOSE** — Bernoulli, indicator, binomial, Poisson, geometric; next = continuous. Warm-up: [mass](./PREREQUISITES.md#p4-mass).

### Board / screenshot

![Named families and recap](./screenshots/composites/ch10-topic-10-named-families-recap-panel1of1.png)

**Figure — ~42:47–50:06:** Bernoulli $p$ and $1-p$; indicator $1_B$; binomial $n,p$ with $C(n,k)p^k(1-p)^{n-k}$; Poisson PMF; geometric wait-for-head $(1-p)^{k-1}p$; spoken recap list.

### What he is establishing

**Bernoulli.** Two values, success/failure. PMF puts $p$ on success and $1-p$ on failure, with $p\in[0,1]$. Coin (tail $0$, head $1$), pass/fail, a COVID test — all Bernoulli stories.

**Indicator** $1_B$. On $(\Omega,\mathcal{F},P)$ pick $B\in\mathcal{F}$. Set $1_B(\omega)=1$ if $\omega\in B$ and $0$ otherwise. Then

$$
P(1_B = 1) = P(B)

$$

They will use indicators constantly later.

**Binomial$(n,p)$.** $n$ independent Bernoulli trials. $X=$ number of successes, $X\in\{0,1,\ldots,n\}$.

$$
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad k=0,\ldots,n

$$

Parameters: $n$ a positive integer, $p\in[0,1]$. Reading: $k$ successes each costing $p$, $n-k$ failures each costing $1-p$, and $\binom{n}{k}$ ways to arrange them. Number of heads in $n$ independent tosses with $P(H)=p$ is binomial. It **is** $n$ independent Bernoulli trials.

**Poisson.** Support $0,1,2,\ldots$ (countably infinite). The board writes one parameter $\lambda>0$ and

$$
p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k=0,1,2,\ldots

$$

Whenever you propose a PMF, **check the axioms**: each mass in $[0,1]$, and the sum over the support is $1$. They check Poisson with the series $\sum_{k=0}^{\infty} \lambda^k/k! = e^{\lambda}$, so the $e^{-\lambda}$ in front makes the masses add to $1$. Bernoulli/indicator is the easy check $p+(1-p)=1$.

**Geometric.** Support $1,2,3,\ldots$. Toss until the first head.

$$
P(X=k) = (1-p)^{k-1} p

$$

The first $k-1$ tosses are tails, then a head. That is “probability the first head occurs at the $k$th toss.”

**Recap of this tutorial:** definition of probability, conditional probability, Bayes, independence and its variants, random variables, and — specifically — discrete random variables. **Next tutorial:** continuous random variables, then expectation and variance.

You can now recognize the five named PMFs and the homework of checking $\sum p=1$. The leftover problem is the **continuum**: no PMF, a density instead, and averages $E[X]$.

A common trap is mixing geometric “trial of first success” (starts at 1) with a “number of failures” version that starts at 0. Use the lecture’s $k=1,2,\ldots$ version. Another trap: calling binomial a single Bernoulli.

### Analogy for this topic only

Bernoulli is **one** yes/no coin. Indicator is a yes/no coin **tied to a specific event $B$**. Binomial is **how many yeses** in $n$ independent coins. Geometric is **how long you wait** for the first yes. Poisson is a **count with no hard cap** $n$, one knob for typical size. Same PMF rules: piles $\ge 0$ that add to $1$.

Question: **What two numbers specify a binomial family?**

In lecture words: this box closes the discrete recap.

### Local picture

```
  Bern(p)     : {0,1}           p, 1-p
  1_B         : {0,1}           P(B), 1-P(B)
  Bin(n,p)    : {0…n}           C(n,k) p^k (1-p)^{n-k}
  Poisson(λ)  : {0,1,2,…}       λ^k e^{-λ} / k!  (sums to 1)
  Geo(p)      : {1,2,3,…}       (1-p)^{k-1} p

  Next: continuous X · E · Var
```

**Notice:** every named PMF is just a legal pile of $q_i$.

### Bridge

The discrete toolkit is in place. Continuous RVs will replace piles by **densities** and will need expectation — that is the next tutorial, not this one.

---

## External references

Two layers, **both kept**.

1. **Start here** — the newer high-signal companions (famous teachers, mapped to this lecture’s hard boxes).
2. **Full topic map** — the previous per-topic list (2–3 companions each) **plus** any new entries already woven above. Use a group when one box still feels thin.

### Start here — high-signal companions

Only a few **widely used** companions — the ones people actually finish. Not a pile of random blogs. Use them after the matching topic, with this tutorial still closed.

This lecture is a **chalkboard recap**. It does **not** run Python. Do not look for a hidden Colab; check arithmetic on paper.

**If Ω, events, and $P$ still blur (Topics 1–3).** Two classroom standards: [Khan Academy’s probability unit](https://www.khanacademy.org/math/statistics-probability/probability-library) and Brown’s [Seeing Theory](https://seeing-theory.brown.edu/). Those two beat a dozen SEO probability articles.

**If “given that” and Bayes still swap in your head (Topics 2–4).** Grant Sanderson’s [3Blue1Brown — Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) is the famous geometric flip of $P(A\mid B)$. Taboga’s [Statlect — Bayes’ rule](https://www.statlect.com/fundamentals-of-probability/Bayes-rule) is the clean written version of the same formula.

**If you mix up “probability” and “likelihood” (Topics 2–4).** Josh Starmer’s [StatQuest — Probability vs Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4) is the short, famous clarification.

**If independence is still a vibe (Topics 5–6).** Khan’s [independent events](https://www.khanacademy.org/math/statistics-probability/probability-library/multiplication-rule-independent/v/compound-sample-spaces) plus Seeing Theory’s [independence](https://seeing-theory.brown.edu/compound-probability/index.html#second) keep the product test honest. Statlect’s [independent events](https://www.statlect.com/fundamentals-of-probability/independent-events) writes mutual vs pairwise in one place.

**If “$X$ is a function, $F$ is a running total” (Topics 7–10).** Khan’s [random-variables unit](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) and Seeing Theory’s [distributions](https://seeing-theory.brown.edu/probability-distributions/index.html) are the usual next stop. Statlect’s [random variables](https://www.statlect.com/fundamentals-of-probability/random-variables) is the careful $\Omega\to\mathbb{R}$ page.

**How to use.** Triplet fog → Khan or Seeing Theory *before* Topic 1. Bayes flip → 3Blue1Brown *after* Topic 4. One famous teacher per stuck idea. Do not open ten tabs.

---

### Full topic map — previous list plus new entries

**How to use:** finish the NOTES chain first (video closed if you can). When one map box still feels thin, open **only that topic’s group** — **2–3 companions each** (prefer a **teaching video + notes/blog**). All links live **here**, not inside topic bodies.

This lecture is a **chalkboard recap**. It does **not** run Python. Do not look for a hidden Colab; check arithmetic on paper.

### Topic 1 — Triplet and axioms


| Resource                                                                                      | Type         | Why it helps                    |
| ----------------------------------------------------------------------------------------------- | -------------- | --------------------------------- |
| [Khan Academy — compound sample spaces](https://www.youtube.com/watch?v=PR-A3UAO7_0)         | Video        | Ω as a list / tree of outcomes |
| [Seeing Theory — Basic probability](https://seeingtheory.io/basic-probability/introduction/) | Interactive  | Events as regions; mass as area |
| [Stat 110 home (Blitzstein)](https://stat110.hsites.harvard.edu/)                             | Course notes | Probability space language      |

### Topic 2 — Conditional probability


| Resource                                                                                                                                                                                          | Type        | Why it helps                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ----------------------------------- |
| [StatQuest — Probability vs likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4)                                                                                                             | Video       | “Given that” without new axioms |
| [Khan Academy — conditional probability](https://www.khanacademy.org/math/statistics-probability/probability-library/conditional-probability-independence/v/calculating-conditional-probability) | Lesson      | Die / card conditionals           |
| [Seeing Theory — Compound probability](https://seeingtheory.io/compound-probability/conditional-probability/)                                                                                    | Interactive | Slice$B$, then look at $A$        |

### Topic 3 — Total probability


| Resource                                                                                                       | Type        | Why it helps                          |
| ---------------------------------------------------------------------------------------------------------------- | ------------- | --------------------------------------- |
| [Stat 110 Lec 5 — total probability](https://www.youtube.com/watch?v=JzDvVgNDxo8)                             | Video       | Partition mix, same rule as the board |
| [Brilliant — Law of total probability](https://brilliant.org/wiki/law-of-total-probability/)                  | Notes       | Pie-slice derivation                  |
| [Seeing Theory — Compound probability](https://seeingtheory.io/compound-probability/conditional-probability/) | Interactive | Weighted rooms                        |

### Topic 4 — Bayes


| Resource                                                                                | Type  | Why it helps                     |
| ----------------------------------------------------------------------------------------- | ------- | ---------------------------------- |
| [3Blue1Brown — Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)             | Video | Geometry of flipping$P(A\mid B)$ |
| [3Blue1Brown — Bayes lesson (text)](https://www.3blue1brown.com/lessons/bayes-theorem) | Blog  | Same pictures, pause-able        |
| [Stat 110 Lec 4 — Bayes](https://www.youtube.com/watch?v=P7NE4WF8j-Q)                  | Video | $\{A,A^c\}$ denominator          |

### Topic 5 — Two-event independence


| Resource                                                                                                                                                                   | Type        | Why it helps                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------------------- |
| [Stat 110 Lec 4 — independence](https://www.youtube.com/watch?v=P7NE4WF8j-Q)                                                                                              | Video       | Product test as definition    |
| [Khan Academy — independent events](https://www.khanacademy.org/math/statistics-probability/probability-library/multiplication-rule-independent/v/compound-sample-spaces) | Lesson      | When you may multiply         |
| [Seeing Theory — Independence](https://seeingtheory.io/compound-probability/independence/)                                                                                | Interactive | Product vs “feel unrelated” |

### Topic 6 — Independence flavors


| Resource                                                                                                                                                  | Type  | Why it helps                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ----------------------------- |
| [Stat 110 Lec 5 — conditional independence](https://www.youtube.com/watch?v=JzDvVgNDxo8)                                                                 | Video | Tests given disease         |
| [Stat 110 notes hub](https://stat110.hsites.harvard.edu/)                                                                                                 | Notes | Pairwise ≠ mutual          |
| [Towards Data Science — conditional independence](https://towardsdatascience.com/conditional-independence-the-backbone-of-bayesian-networks-85710f1b35b) | Blog  | Same disease-test intuition |

### Topic 7 — Random variables


| Resource                                                                                                                                                                                                 | Type   | Why it helps                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------------- |
| [StatQuest — probability distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc)                                                                                                                    | Video  | What a distribution is                |
| [Khan Academy — discrete random variables](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-discrete/v/discrete-and-continuous-random-variables) | Lesson | $X$ as a numerical label              |
| [Stat 110 notes](https://stat110.hsites.harvard.edu/)                                                                                                                                                    | Notes  | RV as a function$\Omega\to\mathbb{R}$ |

### Topic 8 — CDF


| Resource                                                                                                                                                                                        | Type        | Why it helps          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ----------------------- |
| [Khan Academy — cumulative distribution](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-discrete/v/discrete-probability-distribution) | Video       | Running total of mass |
| [Seeing Theory — Distributions](https://seeingtheory.io/probability-distributions/discrete-discrete/)                                                                                          | Interactive | Stairs vs smooth$F$   |
| [Stat 110 notes](https://stat110.hsites.harvard.edu/)                                                                                                                                           | Notes       | $F$ specifies the law |

### Topic 9 — Discrete PMF / staircase


| Resource                                                                                                                                                                                                  | Type        | Why it helps                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ---------------------------------- |
| [Khan Academy — discrete probability distribution](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-discrete/v/discrete-probability-distribution) | Video       | Mass at a point vs until a point |
| [Seeing Theory — Discrete distributions](https://seeingtheory.io/probability-distributions/discrete-discrete/)                                                                                           | Interactive | Jumps = atoms                    |
| [StatQuest — distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc)                                                                                                                                 | Video       | Histogram / pile picture         |

### Topic 10 — Named families


| Resource                                                                                                                                                             | Type   | Why it helps                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------- |
| [StatQuest — Binomial](https://www.youtube.com/watch?v=J8jNoF-K8E8)                                                                                                 | Video  | $n$ independent yes/no trials   |
| [3Blue1Brown — Binomial distributions](https://www.youtube.com/watch?v=8idr1WZ1A7Q)                                                                                 | Video  | Why$\binom{n}{k}p^k(1-p)^{n-k}$ |
| [Khan Academy — binomial](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/binomial-random-variables/v/binomial-distribution) | Lesson | Parameters$n,p$                 |

### Whole-map companions


| Resource                                                                                              | Type            | Why it helps                   |
| ------------------------------------------------------------------------------------------------------- | ----------------- | -------------------------------- |
| [PREREQUISITES.md (this package)](./PREREQUISITES.md)                                                 | Warm-up         | #p1–#p8 beginner unlocks      |
| [Seeing Theory](https://seeingtheory.io/)                                                             | Interactive hub | All early probability pictures |
| [Stat 110 YouTube playlist](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) | Video course    | Same objects, slower proofs    |

---


## Sources

- Video: [Tutorial 7 : Review of Basic Probability 1](https://www.youtube.com/watch?v=owlWCCgYx50)
- Channel: NPTEL — Indian Institute of Science, Bengaluru
- Duration: ~50 min (00:03–50:06)
- Skill: `youtube-lecture-tutor` · math_technical
- 10 topics · 53 claims · coverage receipt
- Previous: [Tutorial 6 Transfer Learning](../20-Tutorial06-Transfer-Learning-PyTorch/NOTES.md)
- Next (spoken): continuous RVs, expectation, variance
- Package: `21-Tutorial07-Review-Basic-Probability-1`
