# Implementation Plan — Folder-by-Folder Upgrade for `/goal` Autonomous Execution

> **Specification Authority:** [`UPGRADE_SPEC.md`](./UPGRADE_SPEC.md)  
> **Execution Model:** **Folder-by-Folder (Module-by-Module) Autonomous Loop**. Each folder is an atomic, self-contained milestone delivering the **7-Pillar Production Learning Suite** (`PREREQUISITES.md`, `NOTES.md`, `references.md`, `examples/`, `glossary.md`, `formulae_sheet.md`, `quiz.html`). All prerequisite, sibling curriculum ([`Mathematical-foundation-ml`](../Mathematical-foundation-ml)), and lecture-specific [`MathsTerms`](./MathsTerms) are dynamically identified, built, and cross-referenced. Runnable Python simulations are executed and tested via `validate_package.py`, and a formal **Done Review with Pedagogical Confidence Scoring** is delivered before advancing to the next module.

---

## 1. Universal Pedagogical & Quality Contract

### A. Zero-Assumption Contract (10–15 Year Return Gap)
- **Target Reader:** An engineer returning to math after 10–15 years. Assume **zero prior mathematical memory** (forgotten logarithms, probability axioms, likelihood definitions, calculus chain rule, tensor shapes, and divergence properties).
- **The 5-Point Pedagogical Bridge:**
  $$\text{👶 ELI5 Intuition} \iff \text{🔍 Plain-English} \iff \text{🔢 Concrete Micro-Numbers} \iff \text{📐 Formal Math} \iff \text{💻 Runnable Code \& GenAI Systems}$$

### B. Anti-AI-Slop & Logical Rigor Contract
- **No Fluff or Hand-Waving:** Ban generic filler phrases (e.g. "In today's fast-paced AI world", "It is crucial to understand"). 
- **Logically Grounded Derivations:** Every mathematical transition must show all intermediate algebraic steps. Never say "it can easily be shown that" or "trivially".
- **Spoken English Phonetics:** Every dense formula and Greek symbol must include plain-English spoken phonetic pronunciations in `glossary.md` and `formulae_sheet.md` so the reader can speak and think through formulas fluently.
- **Deterministic Engineering Analogies:** Use concrete physical systems (kitchen scales, water pipes, gear trains, spreadsheet matrices, barcodes) rather than vague philosophical metaphors.
- **Micro-Numerical Grounding:** Every abstract formula must be backed by a hand-calculated numerical example with real numbers.

---

## 2. Dynamic `MathsTerms/` Discovery & Sibling Curriculum Bridging Rules

### A. Sibling Course Prerequisite Bridging (`Mathematical-foundation-ml`)
When a lecture relies on foundational concepts covered in the sibling course `Mathematical-foundation-ml` (or foundational `MathsTerms/`), the author **must connect the dots**:
1. Explicitly identify what was introduced or assumed from `Mathematical-foundation-ml` (e.g., linear algebra, multivariable calculus, MLE, probability foundations).
2. Add explicit cross-course bridges in `PREREQUISITES.md` under a dedicated **Curriculum & Sibling Course Prerequisite Bridges** table linking to the sibling folder (e.g., `../../Mathematical-foundation-ml/03-Lec02-Recap-Probability-Theory-Part1/NOTES.md`).
3. Cross-reference sibling lecture modules in `references.md` Category 1.

### B. Dynamic "On-Demand" `MathsTerms/` Discovery
Do **not** restrict math terms to a fixed list. During the processing of **each folder**:
1. **Aggressive Term Discovery:** Scan the lecture transcript, raw claims, notes, and prerequisites for **every single mathematical term, statistical concept, matrix operation, distribution, loss function, or optimization technique**.
2. **Creation Standard:** If a term does not exist in [`MathsTerms/`](./MathsTerms), create a dedicated markdown file following the 7-section visual gold standard of [`Softmax.md`](./MathsTerms/02-Multivariate-Calculus-and-Optimization/06-Softmax.md):
   - **§1:** Title & High-Impact 3-Stage Visual ASCII Pipeline.
   - **§2:** 👶 **ELI5 Intuition** (Physical analogy / concrete story).
   - **§3:** 🔍 **Plain-English Breakdown & Notation Rosetta Stone Table**.
   - **§4:** 📐 **Formal Mathematical Formulation, Properties & Guarantees**.
   - **§5:** 🔗 **Connecting the Dots: How this Concept Powers Modern ML & Generative AI**.
   - **§6:** 💻 **Complete Standalone Executable Python/PyTorch Verification Script**.
   - **§7:** 🩺 **Diagnostic Mini-Checks & Common Traps**.
3. **Bidirectional Linking:** Explicitly cross-link the newly created term in `PREREQUISITES.md`, `NOTES.md`, and `glossary.md`.

---

## 3. Standard Folder Upgrade Checklist (7-Pillar Learning Suite)

### A. Checklist for `PREREQUISITES.md`
1. [ ] **Math Terminology Rosetta Stone Table:** Maps every mathematical symbol to plain-English software concepts and physical metaphors.
2. [ ] **Foundational Pillars (4 to 8):** Tailored to the lecture topics. Each pillar includes:
   - 👶 Purpose & ELI5 physical analogy.
   - 🔍 Plain-English breakdown & hand-calculated concrete micro-numbers.
   - 📐 Formal math definition & guarantees.
   - 💻 Runnable standalone Python code snippet.
   - 🩺 Diagnostic Mini-Checks (self-test with clear answers).
   - Anchor tag `<a id="p1-..."></a>`.
   - 🔗 Direct markdown links to `../../MathsTerms/*.md`.
3. [ ] **Curriculum & Sibling Course Prerequisite Bridges Table:** Explicit cross-links connecting lecture prerequisites to prior foundational lectures in `../Mathematical-foundation-ml/` and `MathsTerms/`.

### B. Checklist for `NOTES.md`
1. [ ] **Title Discrepancy Notice (if applicable):** Clarifies YouTube playlist titles vs actual curriculum lecture titles.
2. [ ] **Executive Summary & Master Architecture Blueprint:**
   - 3–6 sentence plain-English lead (job $\to$ method $\to$ fork).
   - ASCII Master Architecture Blueprint diagram.
   - Comparative Feature Matrices.
   - Common Engineering & Mathematical Traps with exact fixes.
3. [ ] **Chalkboard & Mathematical Rosetta Stone Table:** Complete symbol-to-meaning reference.
4. [ ] **Complete Standalone Executable Python/PyTorch Simulation Script:** Placed at §3 (top of document), fully runnable with zero errors and rich conceptual comments.
5. [ ] **Topic Deep Dives (4 to 10 Topics matching the timeline):**
   - Exact composite screenshot links `![Caption](./screenshots/composites/...)` with timestamps.
   - Standalone **ASCII Blackboard Visual Reconstructions**.
   - Zero-leap mathematical derivations in progressive 3-layer structure (ELI5 $\to$ Plain-English $\to$ Math).
   - Contrastive "Why X, Not Y" sections explaining why naive baselines fail.
   - Explicit connections to ML, AI, and code.
   - Active comprehension checks ("Check Your Understanding": Recall, Apply, Diagnose, Vocabulary).
   - 🔗 Direct markdown links to `../../MathsTerms/*.md` and `examples/*.py`.
6. [ ] **2 Workplace Debugging Scenarios (Postmortems):** Real-world engineering incidents with Problem $\to$ Root Cause $\to$ Debugging Steps $\to$ Python Code Fix.
7. [ ] **Clean Delegation to `references.md`:** Decouple citations by embedding a clean delegation callout block to `references.md` (no inline 50+ URL clutter).
8. [ ] **Quiz Anchor Synchronization:** Matching `quiz.html` references.

### C. Checklist for `references.md`
1. [ ] **5 Structured Annotated Categories:**
   - §1: Curriculum & Sibling Course Bridges (explicit links to `../Mathematical-foundation-ml/` and `MathsTerms/`).
   - §2: Foundational & Seminal Papers (with authors, publication year, arXiv/DOI links, and exact relevance).
   - §3: Authoritative Textbooks & Lectures (Goodfellow, Murphy, Bishop, Prof. Prathosh).
   - §4: Industry & Production Implementation Guides (PyTorch, Hugging Face, Triton, postmortems).
   - §5: Interactive Visualizers & Educational Tools (Distill.pub, 3Blue1Brown, Desmos).
2. [ ] **Annotated Descriptions:** Every citation includes a 1–2 sentence description of why it matters and what concept it reinforces.

### D. Checklist for `examples/` Directory
1. [ ] **Standalone Runnable Python Scripts (`01_*.py`, `02_*.py`):** Self-contained, runnable via `python examples/01_*.py`.
2. [ ] **Zero Execution Errors:** Returns exit code 0 under standard execution.
3. [ ] **Mathematical Verification:** Proves and simulates the exact mathematical concepts discussed in `NOTES.md`.
4. [ ] **Explanatory Comments:** Comments explain tensor dimensions, mathematical steps, and numerical stability tricks.
5. [ ] **Cross-Referenced in Notes:** Linked inside `NOTES.md` and `PREREQUISITES.md`.

### E. Checklist for `glossary.md`
1. [ ] **Master Alphabetical Terminology Table:** Covers every mathematical symbol, statistical term, and architectural entity.
2. [ ] **Spoken English Phonetic Guides:** Explicit column demonstrating how practitioners pronounce terms and formulas aloud (e.g., `\mathcal{L}_{\text{ELBO}}` $\to$ *"L-sub-ELBO"*).
3. [ ] **Plain-English Definition & Math Role:** Intuitive explanation and concrete purpose in Generative AI / ML.
4. [ ] **Cross-Links:** Direct links to `NOTES.md` topics and `MathsTerms/`.

### F. Checklist for `formulae_sheet.md`
1. [ ] **Core Equations Compilation:** Clean LaTeX formulas with plain-English names and spoken phonetic pronunciations.
2. [ ] **Tensor Dimensions & Invariant Guarantees:** Exact input/output tensor shapes and mathematical bounds.
3. [ ] **Contrastive Decision Matrix ("Why X, Not Y"):** Analytical trade-offs against baseline methods.
4. [ ] **Edge Cases & Numerical Stability Checklist:** Epsilon additions, log-domain computations, gradient clipping.

### G. Checklist for `quiz.html`
1. [ ] **Interactive Quiz Application:** 15–20 high-quality diagnostic questions.
2. [ ] **Bidirectional Anchors:** Every question links directly back to the relevant section anchor in `NOTES.md`.

### Adjusted Clause: Execution & Batching Rule
- Execute one folder at a time (or max 2 adjacent folders) to ensure maximum depth, full algebraic derivation completeness, and execution verification via `validate_package.py`.
---

## 4. Sequential Folder-by-Folder Execution Queue

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               FOLDER-BY-FOLDER EXECUTION QUEUE                                   │
├────┬─────────────────────────────────────────────────┬──────────────────────────────────────────┤
│ Step │ Target Module Folder                          │ Expected Initial / Baseline MathsTerms   │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 01 │ 01-Lec01-MFGAI-Introduction                     │ • Probability_Basics_and_Axioms.md       │
│    │                                                 │ • Random_Variables_and_Distributions.md  │
│    │                                                 │ • Common_Probability_Distributions.md    │
│    │                                                 │ • Likelihood_and_Log_Likelihood.md       │
│    │                                                 │ • MLE.md (Audit & Standardize)           │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 02 │ 02-Lec02-Generative-Models-Problem-Formulation  │ • Tensors_and_Shapes.md                  │
│    │                                                 │ • Vector_Norms_and_Inner_Products.md     │
│    │                                                 │ • KL_Divergence.md                       │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 03 │ 03-Tutorial02-Introduction-to-NumPy (Tier 1)    │ • Activation_Functions.md (ReLU, Sigmoid)│
│    │                                                 │ • Loss_Functions.md (MSE, BCE, CCE, NLL) │
│    │                                                 │ • Convolution_and_Pooling.md             │
│    │                                                 │ • Recurrent_Neural_Networks.md           │
│    │                                                 │ • Softmax.md · Argmax.md · OneHot.md     │
│    │                                                 │ • Gradient_Descent.md (Audit)            │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 04 │ 04-Tutorial03-PyTorch-Basics                    │ • Derivatives_Gradients_and_Jacobians.md │
│    │                                                 │ • Tensors_and_Shapes.md                  │
│    │                                                 │ • Activation_Functions.md                │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 05 │ 05-Tutorial04-CNNs-PyTorch                      │ • Convolution_and_Pooling.md             │
│    │                                                 │ • Batch_Normalization_and_Spectral_Norm.md│
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 06 │ 06-Tutorial05-RNNs-PyTorch                      │ • Recurrent_Neural_Networks.md (LSTM/GRU)│
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 07 │ 07-Tutorial06-Transfer-Learning-PyTorch         │ • Convolution_and_Pooling.md             │
│    │                                                 │ • Loss_Functions.md                      │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 08 │ 08-Tutorial07-Review-Basic-Probability-1        │ • Probability_Basics_and_Axioms.md       │
│    │                                                 │ • Logarithms_and_Exponential_Functions.md│
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 09 │ 09-Tutorial08-Review-Basic-Probability-2        │ • Convexity_and_Jensens_Inequality.md    │
│    │                                                 │ • Random_Variables_and_Distributions.md  │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 10 │ 10-Tutorial09-Review-Basic-Probability-3        │ • Joint_Marginal_Conditional_Dist.md     │
│    │                                                 │ • Derivatives_Gradients_and_Jacobians.md │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 11 │ 11-Tutorial10-Review-Machine-Learning-1         │ • Likelihood_and_Log_Likelihood.md       │
│    │                                                 │ • Expectation_Maximization_Algorithm.md  │
│    │                                                 │ • NLL.md (Audit)                         │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 12 │ 12-Lec03-f-Divergence-Examples                  │ • f_Divergence.md                        │
│    │                                                 │ • KL_Divergence.md                       │
│    │                                                 │ • Jensen_Shannon_Divergence.md           │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 13 │ 13-Tutorial11-f-Divergence-Examples             │ • f_Divergence.md                        │
│    │                                                 │ • Convexity_and_Jensens_Inequality.md    │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 14 │ 14-Lec04-Variational-Divergence-Minimization    │ • Fenchel_Conjugate_Dual_Functions.md    │
│    │                                                 │ • Minimax_Game_and_GANs.md               │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 15 │ 15-Lec05-Generative-Adversarial-Networks        │ • Minimax_Game_and_GANs.md               │
│    │                                                 │ • Jensen_Shannon_Divergence.md           │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 16 │ 29-Tutorial12-Implementations-Vanilla-DCGAN-cGAN│ • Minimax_Game_and_GANs.md               │
│    │                                                 │ • FID_Frechet_Inception_Distance.md      │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 17 │ 17-Lec06-Wasserstein-GAN (Tier 1)               │ • Wasserstein_Distance.md                │
│    │                                                 │ • Lipschitz_Continuity.md                │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 18 │ 18-Lec07-Inversion-GANs-FID (Tier 1)            │ • FID_Frechet_Inception_Distance.md      │
│    │                                                 │ • Latent_Variable_Models.md              │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 19 │ 19-Lec08-Latent-Variable-Models-VAE (Tier 1)    │ • Latent_Variable_Models.md              │
│    │                                                 │ • ELBO_Evidence_Lower_Bound.md           │
│    │                                                 │ • Reparameterization_Trick.md            │
├────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 20 │ Master Catalog & Global System Verification     │ • Mathematical-Foundation-for-GenAI/     │
│    │                                                 │   NOTES.md Master Catalog Index          │
└────┴─────────────────────────────────────────────────┴──────────────────────────────────────────┘
```

---

## 5. Mandatory Per-Folder Completion Review & Sign-Off Gate

At the completion of each folder, the agent must output a structured **Folder Milestone Report** following this exact format:

```markdown
### 🏁 Milestone Completion Review: [Folder Name]

- **Status:** ✅ DONE / ❌ NOT DONE
- **7-Pillar Specification Compliance:**
  - [x] **Pillar 1 (`PREREQUISITES.md`):** Math Rosetta Stone, 4–8 Foundational Pillars (3-layer ELI5, concrete numbers, code & mini-checks), and Sibling Curriculum Prerequisite Bridges table.
  - [x] **Pillar 2 (`NOTES.md`):** Executive Summary, ASCII Architecture Blueprint, Chalkboard Rosetta Stone, §3 Full Simulation, Topic Deep Dives (zero-leap derivations, "Why X, Not Y", comprehension checks), 2 Workplace Debugging Postmortems, clean delegation callout to `references.md`, and quiz anchors.
  - [x] **Pillar 3 (`references.md`):** 5-category annotated citations (Curriculum Bridges, Seminal Papers, Textbooks & Lectures, Industry Guides, Visualizers).
  - [x] **Pillar 4 (`examples/`):** Standalone executable Python scripts (`01_*.py`, `02_*.py`) with rich comments, validating core math theorems.
  - [x] **Pillar 5 (`glossary.md`):** Master terminology table with spoken English phonetic pronunciation guides.
  - [x] **Pillar 6 (`formulae_sheet.md`):** Core equations, tensor dimensions, mathematical guarantees, and "Why X, Not Y" decision matrices.
  - [x] **Pillar 7 (`quiz.html`):** Interactive assessment with bidirectional anchors.
- **MathsTerms Created / Updated in this Step:**
  - `MathsTerms/[Term_Name].md` (Verified 7-section Softmax standard)
- **Automated Validation Suite (`validate_package.py`):**
  - Command: `python youtube-lecture-tutor/scripts/validate_package.py <folder_path>`
  - Output: `[PASS]` All 7 pillars verified, 0 broken relative links, all `examples/*.py` executed with exit code 0.
- **Pedagogical Understanding & Dot-Connecting Confidence:**
  - **Confidence Score:** 98% / High Confidence
  - **How Dots are Connected:** [Explain how the 10-15 year returning engineer can now seamlessly understand the physical intuition, mathematical formulas, code mechanics, sibling course foundations in `Mathematical-foundation-ml`, and how it connects to modern Generative AI].
```

---

## 6. Global System Verification Suite

At the conclusion of Step 20:
1. **Python Script Health Test:** Execute a test verifying all core AI/ML dependencies:
   ```powershell
   python -c "import torch, numpy as np, scipy.stats as stats; print('All core AI/ML dependencies healthy')"
   ```
2. **Automated 7-Pillar Package Validation:** Run `validate_package.py` across every upgraded module folder:
   ```powershell
   python youtube-lecture-tutor/scripts/validate_package.py <module_directory>
   ```
3. **Global Anchor & Relative Link Validator:** Verify zero broken links across `MathsTerms/`, `Mathematical-foundation-ml/`, and `Mathematical-Foundation-for-GenerativeAI/`.
