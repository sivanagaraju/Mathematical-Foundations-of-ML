# MathsTerms Curriculum Restructure: Progress and Evidence Ledger

## How to read this ledger

The unit of work is:

```text
folder topic → Markdown subtopic → learner task
```

Each lesson moves through `Review 1 → Plan → Implement → Review 2 → Validate`. “Inventory complete” means the file has been located and measured. It does not mean its mathematics or teaching flow has passed review.

Status values:

- `Pending` — work has not begun.
- `In progress` — evidence is being collected or changes are being made.
- `Needs work` — review found concrete defects.
- `Ready for validation` — implementation is complete but checks are outstanding.
- `Validated` — all applicable mandatory checks passed.
- `Deferred` — work is intentionally postponed with a recorded reason.

## Overall stage

| Item | Status | Evidence or next action |
| :--- | :--- | :--- |
| Preserve and inventory | Complete | 49 lessons, 12 folder navigation files, 19,955 lesson lines, and the current working tree were inventoried. |
| Folder-level structural review | Complete | All 49 lessons received an initial file-level review; every proof and code block still needs its own deep review before later implementation. |
| Lecture-package link audit | Complete | 120 course documents, 65 MathsTerms files, 4,546 relative link occurrences, and 2,053 local-anchor occurrences were inspected. |
| Entropy pilot Review 1 | Complete | Current file and Git diff were read; narrative, mathematical, rendering, and reference defects were recorded. |
| Entropy pilot plan | Complete | Narrative spine, learner-task order, scope boundaries, mathematical corrections, and validation gates were defined. |
| Entropy implementation | Complete | The user-requested cognitive-learning revision now exposes nine learner topics rather than 17, while retaining one continuous probability example and marking derivative/code depth as optional. |
| Entropy Review 2 | Complete | The earlier independent review and a fresh post-cognitive-engine review repaired reintroduced direction, optimality, attribution, URL, recall, hierarchy, and whitespace defects. |
| Entropy validation | Static checks validated; learner study pending | The pilot evaluator passes 17/17 checks; repository links, both code blocks, source rendering, math delimiters, cognitive-learning evidence, and control characters pass. A real beginner explain-back remains. |
| Remaining curriculum implementation | Deferred | Begins only after the user studies and accepts or revises the pilot approach. |

## Delegated audit register

Delegation helps gather evidence in parallel. It does not transfer editorial responsibility or allow silent completion claims.

| Workstream | Assigned scope | Edit authority | Status | Result integration |
| :--- | :--- | :--- | :--- | :--- |
| Folder audit | Six numbered folders; file structure, learner gaps, math-risk flags, and dependency order | Read only | Complete | All 49 lessons and 12 navigation files reviewed; priorities and hidden dependency cycles incorporated below. |
| Link audit | `NOTES.md`, `PREREQUISITES.md`, MathsTerms paths and anchors | Read only | Complete | Results distinguish path existence from case, anchor, semantic-target, and teaching-placement defects. |
| Entropy pilot review | Current entropy lesson, its diff, cited video/transcript availability, and proposed narrative | Read only | Complete | Exact learner drop-off points and a three-part narrative spine informed the rewrite. |
| Entropy Review 2 | Finished entropy pilot against mandatory validation gates | Read only | Complete | Fourteen actionable findings were reported and repaired; static checks were rerun afterward. |

## Repository-wide findings

| ID | Finding | Evidence | Consequence | Status |
| :--- | :--- | :--- | :--- | :--- |
| R-001 | A universal scaffold dominates the collection | 48/49 “Missing Foundation”; 49/49 confidence audit; 46/49 “Core Aha!”; 49/49 Python section | Coverage appears complete while the chapter's natural question order may be broken | Needs work |
| R-002 | Generated control characters corrupt mathematical source | 23 files contain ASCII control codes other than tab/newline/carriage return | Commands such as `\approx`, `\beta`, and `\text` can render incorrectly or change meaning | Needs work |
| R-003 | Existing changes are extensive | 60 working-tree entries at audit start | Every implementation must read the current diff and avoid overwriting useful or unrelated work | Active constraint |
| R-004 | Lesson size is substantial | 49 lessons, 19,955 lines; several exceed 5,000 words | Navigation must support a first reading without deleting the reasoning needed by beginners | Needs review |
| R-005 | Heading hierarchy is structurally flat | All 49 lessons use an H1 followed directly by H3 headings | Contents look segmented, but section relationships are not represented correctly | Needs work |
| R-006 | Folder navigation conflicts | `README.md` and `START_HERE.md` disagree in folders 02, 03, and 06 | Learners can be sent to a dependent topic before its prerequisite | Needs work |
| R-007 | Existing evals mainly count required phrases | The editorial eval searches for labels such as “Pronunciation Guide” and “Contrastive Analysis” | Passing scores certify template presence, not order, correctness, or comprehension | Needs replacement or expansion |
| R-008 | Existing progress claims exceed their evidence | The old tracker reports 49/49 scripts verified by a missing `scratch/check_code.py` and link counts that no longer match | Completion percentages cannot be trusted until checks are reproducible | Needs work |
| R-009 | MathsTerms links exist but rarely target the needed explanation | 1,052 inbound occurrences from course docs, 737 unique pairs, and zero exact-section links | A learner opens a long guide at its title rather than at the definition or proof needed now | Needs work |
| R-010 | Anchor and semantic-target defects remain | 25 confirmed stale cross-file anchors, 58 unstable emoji fragments, and several topic links pointing to the wrong guide | A resolving file path is insufficient evidence that a reference helps | Needs work |
| R-011 | Some important guides are never linked from course documents | Bounds, LOTUS, and VDM have zero inbound links from audited course documents | Newly created reference material is not yet integrated at teaching moments | Needs work |

## Folder ledger

| Topic | Lessons | Review 1 | Dependency plan | Implementation | Review 2 | Validation | Current note |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 — Primal Analysis and Foundations | 6 | In progress | In progress | Deferred | Pending | Pending | Earlier targeted corrections exist; full structural audit is underway. |
| 02 — Linear Algebra, Geometry, and Tensors | 9 | In progress | In progress | Deferred | Pending | Pending | Inventory complete; concept-level audit underway. |
| 03 — Multivariate Calculus and Optimization | 11 | In progress | In progress | Deferred | Pending | Pending | Inventory complete; concept-level audit underway. |
| 04 — Probability and Statistical Estimation | 7 | In progress | In progress | Deferred | Pending | Pending | Direct prerequisite chain into entropy requires close review. |
| 05 — Information Theory and Divergences | 6 | In progress | In progress | Entropy pilot only | Pending | Pending | Entropy chosen as pilot based on the user's reading experience. |
| 06 — Deep Architectures and Generative Models | 10 | In progress | In progress | Deferred | Pending | Pending | Requires earlier mathematics to be stable before final sequencing. |

Folder Review 1 established the following priority order:

1. **P0 — blocking defects:** corrupted LaTeX in folders 04–06; conflicting navigation in 02, 03, and 06; false or materially overstated claims in norms, broadcasting, embeddings, positional encodings, reverse mode, gradient descent, normalization, probability distributions, MLE/NLL, divergence chapters, EM/ELBO, RNNs, and FID.
2. **P1 — learning-flow defects:** overload and hidden prerequisites across foundational chapters; repeated material among derivative/Jacobian/backprop chapters; elementary and advanced levels mixed inside bounds, Fenchel, Lipschitz, and several probability chapters.
3. **P2 — refinement:** useful side topics such as label smoothing should move after the core concept rather than interrupt it.

The cross-folder route is not one linear list. Foundations lead through algebra/logarithms, probability, vectors, functions, derivatives, random variables, likelihood, and optimization. Information theory follows those foundations. Folder 06 then branches: CNNs, RNNs, autoregressive models, latent-variable models, EM/ELBO, GANs, and FID have different prerequisites.

Hidden pedagogical cycles were found even though the metadata-only graph reports no cycle:

- Fenchel conjugacy teaches an $f$-GAN application while $f$-divergence declares Fenchel conjugacy as a prerequisite.
- Lipschitz continuity teaches WGAN while Wasserstein distance declares Lipschitz continuity as a prerequisite.
- Convexity/Jensen teaches ELBO and EM while their chapters depend on Jensen.
- JSD teaches the GAN theorem while the GAN chapter declares JSD as a prerequisite.

These should become later-application links rather than part of a beginner core.

## Per-file ledger

All files below have completed only the initial inventory unless a later state is explicitly shown.

| Topic | Subtopic file | Review 1 | Plan | Implement | Review 2 | Validate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `01-Probability_Basics_and_Axioms.md` | In progress | Pending | Deferred | Pending | Pending |
| 01 | `02-Logarithms_and_Exponential_Functions.md` | In progress | Pending | Deferred | Pending | Pending |
| 01 | `03-Convexity_and_Jensens_Inequality.md` | In progress | Pending | Deferred | Pending | Pending |
| 01 | `04-Bounds_Supremum_Infimum_and_Linear_Families.md` | In progress | Pending | Deferred | Pending | Pending |
| 01 | `05-Fenchel_Conjugate_and_Dual_Representations.md` | In progress | Pending | Deferred | Pending | Pending |
| 01 | `06-Lipschitz_Continuity.md` | In progress | Pending | Deferred | Pending | Pending |
| 02 | `01-Vectors_and_Matrices.md` | In progress | Pending | Deferred | Pending | Pending |
| 02 | `02-Vector_Norms_and_Inner_Products.md` | In progress | Pending | Deferred | Pending | Pending |
| 02 | `03-Dot_Product_and_Similarity.md` | In progress | Pending | Deferred | Pending | Pending |
| 02 | `04-Tensors_and_Shapes.md` | In progress | Pending | Deferred | Pending | Pending |
| 02 | `05-Tensor_Broadcasting.md` | In progress | Pending | Deferred | Pending | Pending |
| 02 | `06-Singular_Value_Decomposition.md` | In progress | Pending | Deferred | Pending | Pending |
| 02 | `07-One_Hot_Encoding.md` | In progress | Pending | Deferred | Pending | Pending |
| 02 | `08-Encodings_Categorical_and_Embeddings.md` | In progress | Pending | Deferred | Pending | Pending |
| 02 | `09-Positional_Encodings.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `01-Functions_Derivatives_and_Rules.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `02-Derivatives_Gradients_and_Jacobians.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `03-Jacobian_Matrix.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `04-Chain_Rule_and_Backpropagation.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `05-Activation_Functions.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `06-Softmax.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `07-Argmax.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `08-Loss_Functions.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `09-Gradient_Descent.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `10-Exponential_Moving_Average_EMA.md` | In progress | Pending | Deferred | Pending | Pending |
| 03 | `11-Batch_Normalization_and_Spectral_Norm.md` | In progress | Pending | Deferred | Pending | Pending |
| 04 | `01-Random_Variables_and_Distributions.md` | In progress | Pending | Deferred | Pending | Pending |
| 04 | `02-Common_Probability_Distributions.md` | In progress | Pending | Deferred | Pending | Pending |
| 04 | `03-Joint_Marginal_Conditional_Dist.md` | In progress | Pending | Deferred | Pending | Pending |
| 04 | `04-Likelihood_and_Log_Likelihood.md` | In progress | Pending | Deferred | Pending | Pending |
| 04 | `05-MLE.md` | In progress | Pending | Deferred | Pending | Pending |
| 04 | `06-NLL.md` | In progress | Pending | Deferred | Pending | Pending |
| 04 | `07-LOTUS_and_Empirical_Expectation_Estimation.md` | In progress | Pending | Deferred | Pending | Pending |
| 05 | `01-Entropy_CrossEntropy_CCE.md` | Complete for pilot scope | Complete | Complete | Complete | Static pass; learner pending |
| 05 | `02-KL_Divergence.md` | In progress | Pending | Deferred | Pending | Pending |
| 05 | `03-Jensen_Shannon_Divergence.md` | In progress | Pending | Deferred | Pending | Pending |
| 05 | `04-f_Divergence.md` | In progress | Pending | Deferred | Pending | Pending |
| 05 | `05-Wasserstein_Distance_and_EMD.md` | In progress | Pending | Deferred | Pending | Pending |
| 05 | `06-Variational_Divergence_Minimization_VDM.md` | In progress | Pending | Deferred | Pending | Pending |
| 06 | `01-Convolution_and_Pooling.md` | In progress | Pending | Deferred | Pending | Pending |
| 06 | `02-Recurrent_Neural_Networks.md` | In progress | Pending | Deferred | Pending | Pending |
| 06 | `03-Autoencoders_and_Latent_Spaces.md` | In progress | Pending | Deferred | Pending | Pending |
| 06 | `04-Autoregressive_Models.md` | In progress | Pending | Deferred | Pending | Pending |
| 06 | `05-Latent_Variable_Models.md` | In progress | Pending | Deferred | Pending | Pending |
| 06 | `06-Expectation_Maximization_Algorithm.md` | In progress | Pending | Deferred | Pending | Pending |
| 06 | `07-ELBO_and_Variational_Inference.md` | In progress | Pending | Deferred | Pending | Pending |
| 06 | `08-Reparameterization_Trick.md` | In progress | Pending | Deferred | Pending | Pending |
| 06 | `09-Minimax_Game_and_GANs.md` | In progress | Pending | Deferred | Pending | Pending |
| 06 | `10-Frechet_Inception_Distance.md` | In progress | Pending | Deferred | Pending | Pending |

## Entropy pilot Review 1 findings

| ID | Level | Finding | Evidence | Planned response |
| :--- | :--- | :--- | :--- | :--- |
| E-001 | Structure | The central chain was interrupted by template sections | A 14-part list moved between metadata, diagrams, formulas, hardware, and code | Replaced by one route: question tree → surprisal → entropy → cross-entropy → KL → CCE → LLM use |
| E-002 | Sequence | KL appeared before entropy and cross-entropy were constructed | The identity preceded the first entropy calculation | The identity now follows both hand calculations and names an already-visible 0.25-bit gap |
| E-003 | Accuracy | Spoken KL direction was reversed | `D_KL(P || Q)` was read as “from Q to P” | Corrected to “from P to Q” or “P relative to Q” |
| E-004 | Accuracy | Optimization claim was too strong | The chapter said cross-entropy forces KL to zero | Added representation, data/objective, and optimization conditions |
| E-005 | Accuracy | Unique-logit-minimum claim was false | The old table claimed a unique global logit minimum | Added Softmax shift invariance and non-attained one-hot infimum explanation |
| E-006 | Example | Telegram analogy assigned an unsupported exact KL value | Arbitrary integer code lengths did not match the stated entropy | Replaced by one normalized four-outcome dyadic distribution used throughout |
| E-007 | Reference | Entropy video was attributed to the wrong educator | The Khan Academy video was labelled 3Blue1Brown | Corrected title/source with an explicit transcript-access limitation |
| E-008 | Rendering | Hidden characters corrupted LaTeX | Seven forbidden control characters occurred | Rewritten file has zero forbidden control characters |
| E-009 | Evidence | “Beginner confidence” boxes were pre-checked | Claims of completion contradicted the chapter | Replaced by unscored questions, reasoned answers, boundaries, and a user study gate |
| E-010 | System design | The cognitive prompt contradicted itself about chapter size | It named a 10-section architecture, described 12 sections, and supplied a 14-section template | Replaced fixed section compliance with six cognitive mechanisms and a six-to-ten-H2 default |
| E-011 | Cognitive load | The revised entropy contents exposed 17 equal-weight choices | One story was divided into separate glossary, metaphor, hardware, proof, and example blocks | Combined the material into nine learner tasks; the contents page is the tenth H2 but not a concept task |
| E-012 | Retention | The chapter tested recognition but not delayed retrieval | Answers followed questions and no closed-notes or spaced-return routine existed | Added closed-notes reconstruction, Feynman explain-back, contrast, transfer, debugging, and now/tomorrow/one-week prompts |
| E-013 | References | Some links or attributions did not match the described resource | The Khan video was labelled 3Blue1Brown; the Olah URL was stale; the textbook pointed only to a publisher home page | Added checked direct resources, exact chapters/pages, access notes, a textbook route, and a problem set with solutions |
| E-014 | Learning science | The system used an unsupported left/right-hemisphere explanation for dual coding | The mechanism was described as activating visual and verbal hemispheres | Reframed dual coding as coordinated verbal and visual representations and added scientific guardrails |
| E-015 | Accuracy regression | The later entropy revision reintroduced earlier overclaims | Reversed KL reading, “universal loss,” “forces zero,” and unique-optimum implications returned | Repaired the statements and added an evaluator regression list |

## Entropy version comparison after the cognitive-engine revision

| Dimension | Saved Gemini version | User-updated 17-topic version | Current cognitive pilot |
| :--- | :--- | :--- | :--- |
| Learner-facing structure | 14 generated H3 blocks | 17 H3 blocks with stronger derivations but more navigation choices | 9 learner tasks under a valid H2/H3 hierarchy; contents is the tenth H2 |
| Main causal chain | Interrupted by glossary, “aha,” hardware, and AI catalogue blocks | Two-machine opening is stronger, but examples and explanations are separated again later | One machine/codebook example carries probability → log → surprisal → entropy → cross-entropy → KL → CCE/NLL → LLM |
| Cognitive methods | Mostly named template ingredients | More first-principles material, but no retention loop | Prediction, integrated plain-English explanation, analogy mapping, dual-coded visuals, interleaved alternatives, recall, transfer, and spacing are observable tasks |
| Analogy quality | Multiple telegram/taxi metaphors, including unsupported exact numbers | Removes one bad number but retains an unnecessary taxi reset | One codebook/guessing model with dyadic exactness, fractional-code qualification, and explicit failure boundaries |
| Mathematical scope | Contains direction, optimum, source-coding, and hidden-character defects | Repairs characters but reintroduces direction and optimization overclaims | States log-base, support, model-family, empirical/population, non-unique-logit, and non-attainment conditions |
| Recall and practice | Answers and pre-checked confidence claims | Diagnostics and transfer solution, but no delayed retrieval | Closed-notes reconstruction, explain-back, five-level diagnostic practice, separated answers, and now/tomorrow/one-week prompts |
| Learning resources | Incorrect video attribution, stale Olah URL, publisher home page | Same attribution and URL problems | Exact creator/title, corrected visual article, exact textbook chapters/pages, formal notes, mandatory practice set/solutions, access notes, and check date |
| Size | 5,348 words | 6,277 words | 5,587 words; fewer top-level decisions and optional depth clearly marked |

## Entropy pilot task ledger

| Task | Review | Plan | Implement | Review 2 | Validate | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Upgrade the cognitive learning system | Complete | Complete | Complete | Complete | Static pass; rollout pending | Six mechanisms, flexible chapter sizing, integrated intuition, recall/spacing, exact resource roles, and evidence limits are operational requirements |
| Establish the opening problem and learner promise | Complete | Complete | Complete | Complete | Static pass; learner pending | Two message machines create the uncertainty question before notation |
| Build the yes/no-question intuition | Complete | Complete | Complete | Complete | Static pass; learner pending | Uniform and skewed trees lead to 2 and 1.75 weighted questions; coding optimality is scoped |
| Explain probability and log prerequisites | Complete | Complete | Complete | Complete | Static pass; learner pending | Required-now and later-only prerequisites are separated |
| Derive and calculate surprisal | Complete | Complete | Complete | Complete | Static pass; learner pending | Probability powers lead to `log2(1/p)`; joint-event bridge and independent additivity shown |
| Define and calculate entropy | Complete | Complete | Complete | Complete | Static pass; learner pending | Same machine probabilities produce 2 and 1.75 bits |
| Introduce cross-entropy from a wrong model | Complete | Complete | Complete | Complete | Static pass; learner pending | Uniform `Q` scores outcomes produced by skewed `P`; zero terms and support mismatch defined |
| Explain KL decomposition and conditions | Complete | Complete | Complete | Complete | Static pass; learner pending | Identity follows the 0.25-bit gap; proof dependencies and zero-KL conditions stated |
| Connect one-hot CCE and NLL | Complete | Complete | Complete | Complete | Static pass; learner pending | Population, per-example, and empirical levels and symbols distinguished |
| Connect to LLM next-token training | Complete | Complete | Complete | Complete | Static pass; learner pending | Logits-to-loss path shown without vocabulary-scale distraction |
| Add optional gradient derivation | Complete | Complete | Complete | Complete | Static pass; learner pending | Both derivative components shown; Softmax and chain-rule prerequisites linked |
| Add stable implementation example | Complete | Complete | Complete | Complete | Validated | Both extracted Python blocks execute; probability input validation is included |
| Add boundary cases and transfer checks | Complete | Complete | Complete | Complete | Static pass; learner pending | Includes zero support, log bases, coding scope, calibration, and non-dyadic transfer |
| Curate and verify learning references | Complete | Complete | Complete | Complete | Static pass with disclosed YouTube limitation | Khan attribution corrected; Olah, Goodfellow, Wiley, Stanford, MIT, Shannon, and PyTorch destinations checked; exact textbook chapters and practice set supplied |

## Activity log

| Date | Event | Result | Back-review triggered |
| :--- | :--- | :--- | :--- |
| 2026-09-14 | Read the editorial system and inventoried all six folders | Established 49-lesson scope and folder counts | Yes — template compliance cannot be used as learning evidence |
| 2026-09-14 | Scanned repeated lesson structures | Found near-universal generated scaffold | Yes — pilot will use concept-driven headings |
| 2026-09-14 | Scanned lesson text for invalid control characters | Found affected files in folders 03–06, especially 04–06 | Yes — validation expanded beyond Markdown link checks |
| 2026-09-14 | Read the entropy lesson and its current Git diff | Found useful examples mixed with sequencing, accuracy, reference, and rendering defects | Yes — implementation is held until narrative and claim plans are complete |
| 2026-09-14 | Checked public evidence around the entropy video | Direct YouTube extraction was unavailable; university/course references corroborate its entropy and yes/no-question context | Yes — reference must describe the access limitation and avoid claiming transcript review without evidence |
| 2026-09-14 | Completed read-only folder audit | Found navigation conflicts, hidden pedagogical cycles, repeated templates, and file-specific mathematical risks across all 49 lessons | Yes — curriculum rollout must follow dependencies and perform a new deep review per file |
| 2026-09-14 | Completed course-link audit | All relative targets exist locally, but case-sensitive paths, stale anchors, top-of-file links, semantic misrouting, and missing inbound links remain | Yes — link validation expanded from existence to teaching usefulness |
| 2026-09-14 | Rewrote the entropy pilot | Reduced 5,127 words to 3,377 while adding a valid H2 hierarchy and one continuous example; filename preserved | Yes — folder navigation text was adjusted to match the pilot's actual route |
| 2026-09-14 | Ran first post-edit static and numerical checks | Zero forbidden control characters; all relative MathsTerms links resolve; hand calculations and PyTorch loss checks pass | Yes — trailing whitespace was found and repaired; independent Review 2 remains |
| 2026-09-14 | Completed independent Review 2 | Found 14 issues despite the improved narrative: bit wording, optimality scope, joint-event bridge, imported proof scope, zero terms, expectation notation, gradient steps, code input safety, and transfer depth | Yes — every actionable static issue was repaired before rerunning checks |
| 2026-09-14 | Ran the final pilot evaluator | `eval_entropy_pilot.py` passed 15 checks with 0 failures; both embedded Python blocks executed | No additional static defect found |
| 2026-09-14 | Re-ran complete MathsTerms link validation | 1,561 relative links checked after adding plan navigation; 0 broken targets | No pilot link repair required |
| 2026-09-14 | Rendered Markdown to HTML in memory | 1 H1, 15 H2, 23 H3, 8 tables, and 7 code blocks rendered; no raw fences remained; both ASCII trees were retained | No source-structure defect found |
| 2026-09-14 | Checked math and encoding structure | 30 display-math opening/closing pairs, 8 aligned-environment pairs, and 0 forbidden control characters | No pilot math-source defect found |
| 2026-09-14 | Compared the cognitive-system revision, active entropy file, and saved Gemini entropy file | The active revision preserved stronger examples but reintroduced five previously recorded defects and expanded the contents to 17 learner choices | Yes — cognitive mechanisms must shape the narrative rather than appear as mandatory content blocks |
| 2026-09-14 | Upgraded the cognitive learning engine | Replaced the contradictory 10/12/14-section scheme with six operational learning mechanisms, flexible chapter sizing, an end-of-chapter retention loop, and mandatory exact textbook/practice references | Yes — the entropy pilot must be restructured against the new engine before rollout |
| 2026-09-14 | Rebuilt the entropy pilot against the upgraded engine | Condensed the contents to nine learner tasks; integrated ELI5 language, one codebook analogy, visuals, formal math, contrast, AI transfer, limits, recall, and verified resources into a single dependency chain | Yes — evaluator requirements and progress evidence were updated to test the new structure |
| 2026-09-14 | Revalidated the cognitive entropy pilot | `eval_entropy_pilot.py` passed 17 checks; both Python blocks executed; 1,608 relative links resolved; rendered structure contains 1 H1, 10 H2, 32 H3, 10 tables, and 8 code blocks | No additional static defect found; learner retrieval remains pending |

## Pilot validation evidence

| Gate | Result | Reproducible evidence |
| :--- | :--- | :--- |
| Narrative order | Pass | The evaluator verifies the nine-task chain from the guessing problem through cross-entropy, CCE/NLL, AI use, recall, and resources |
| Heading hierarchy and contents anchors | Pass | One H1, 10 H2 including the contents page, 32 H3, no skipped level, and every contents fragment maps to a heading |
| Prerequisite and notation bridge | Pass in editorial walkthrough | Required-now and later-only links are separate; joint probability, expectation, `N`, and `x_j` are decoded before dependence |
| Mathematical scope | Pass in Review 2 | Same log base, zero-weight terms, support mismatch, imported KL result, model restriction, empirical/population distinction, and logit non-uniqueness are explicit |
| Arithmetic | Pass | Main example: 1.75-bit entropy, 2-bit cross-entropy, 0.25-bit KL; transfer examples recomputed independently |
| Executable examples | Pass | Pure Python and PyTorch blocks execute; they verify the 1.75/2.00/0.25-bit identity, support mismatch, CCE/NLL equivalence, and the `q-y` gradient |
| Probability-code safety | Pass | Empty, non-finite, out-of-range, non-normalized, unequal-length, and zero-support cases are handled or rejected |
| Encoding and LaTeX structure | Pass | Zero forbidden control bytes and 62 balanced display-math delimiters; text diagrams remain below 80 columns |
| Markdown rendering | Pass for source renderer | PowerShell Markdown conversion produced the expected heading, table, and code-block structure with no raw fences |
| Relative links | Pass | Corpus evaluator checked 1,608 MathsTerms relative links and found 0 broken targets; pilot links also pass exact-case checks |
| External references | Pass with stated limitation | Olah, Goodfellow, Wiley, Stanford, MIT, Shannon, and PyTorch destinations were opened; Khan title/context was corroborated, while direct YouTube fetching remained unavailable and is disclosed in the chapter |
| Cognitive retention design | Pass for artifact; learner result pending | Closed-notes recall, explain-back, recognition, calculation, contrast, transfer, debugging, separated answers, and spaced-return prompts are present |
| Real beginner explain-back | Pending user study | Editorial review cannot substitute for a learner explaining the ideas and solving an unseen problem |

## Combined entropy reading draft — review before implementation

The user's reading feedback supersedes the earlier narrative “Pass.” Heading and keyword checks did not establish that the chapter felt understandable. In particular, “How the cognitive learning engine appears here” explains authoring machinery before teaching entropy.

Scope: create a separate `01-Entropy_CrossEntropy_CCE_combined.md`; preserve both source chapters and their existing Git changes. Tighten the cognitive prompt's learner/editor boundary. No curriculum-wide rollout or canonical-link migration is authorized by this comparison draft.

| Topic | Subtopic / task | Review finding and implementation decision | Status |
| :--- | :--- | :--- | :--- |
| Reader entry | Opening and navigation | Start with one service/message problem; move method manifests to this ledger. Teach probability and weighted averages before formal expectation. | Implemented; checks below |
| Core mathematics | Coding, logarithms, entropy | Preserve the active version's valid dyadic example; show an actual decodable code and the intermediate frequency calculation. Introduce fractional coding limits after the definition. | Implemented; checks below |
| Core mathematics | Cross-entropy and KL | Keep one distribution throughout; derive the identity and supply the missing nonnegativity proof with its prerequisites. Correct the “training forces KL to zero” claim. | Implemented; checks below |
| Training bridge | CCE, BCE, NLL, empirical averages | Introduce one observation before the batch/population distinction; define natural logs and symbols before formulas. | Implemented; checks below |
| Implementation | Gradients, competing scores, stability | Retain scoped gradient derivations; treat squared probability error as a valid alternative; remove unsupported kernel/memory claims. | Implemented; checks below |
| Breadth | Related concepts from Gemini | Retain temperature, smoothing, focal loss, mutual information, contrastive learning, RL entropy/KL, and generative-model connections as optional introductions, not implied full courses. | Implemented; checks below |
| Retention | Recall and transfer | Place prompts before a separate answer key; test the actual distinctions, including coding limits and finite-data uncertainty. | Implemented; checks below |
| Evidence | Resources and second review | Use exact textbook/practice entry points; check content and access; recompute arithmetic, execute code, inspect links and source rendering. User comprehension remains pending. | Implemented; checks below |

## Combined draft — coverage and second review

**Current status — concurrent edit detected during final validation:** The nine-section narrative draft described below passed its initial tests, but the same file was subsequently replaced by a different fourteen-section document outside this agent's edits. The replacement has different helper functions, skipped heading levels, and a failing `torch.allclose(..., abs_tol=...)` call (`atol` is the supported PyTorch keyword). Final rerun: 8 tests executed, 1 failure and 10 error reports, including incompatibility with the replaced helper API. These results do not certify the current file. The coverage and passing results below describe the earlier nine-section draft only. No attempt was made to overwrite the unexpected replacement. Await user direction before further edits to that file. Latest corpus file-target scan: 1,664 relative links, zero missing targets; this does not resolve the content conflict.

Artifact: [canonical entropy lesson](05-Information-Theory-and-Divergences/01-Entropy_CrossEntropy_CCE.md). The combined synthesis has been promoted to the canonical chapter.

| Source material | Destination in combined draft | Treatment |
| :--- | :--- | :--- |
| Active pilot's dyadic service, question trees, weighted mean | Sections 1–2 | Retained the distribution; added concrete codewords, decoding, eight-tile frequencies, and average calculation before notation. |
| Both sources' surprisal, entropy, cross-entropy, KL identity | Sections 2–3 | Integrated into one service story; decoded symbols locally; added a scoped proof of nonnegativity and the finite entropy upper bound. |
| Gemini's telegram/taxi metaphors and early grand formula map | Sections 1–3 and final recall | Replaced the false telegram arithmetic with an actual prefix code. Did not preserve invented historical quotations, unsupported usage percentages, or early advanced diagrams. |
| Active pilot's population/per-example/batch distinction | Section 4 | Reordered observation first, then counts-to-weights, then conditional population interpretation. No initial table containing unexplained parameters. |
| Both sources' CCE/NLL and LLM uses | Section 4 | Kept one-hot selection, broader NLL scope, chain-rule sequence likelihood, nats, and per-token perplexity. |
| Gemini BCE, smoothing, temperature, focal loss | Sections 4 and 6 | Defined BCE explicitly; optional changes to targets, weighting, and sampling have small examples and limitations. |
| Both sources' gradient/MSE/hardware discussion | Section 5 | Derived Softmax gradient and scoped sigmoid saturation; recognized squared probability error as valid; removed unique finite optimum and fixed-kernel claims; replaced ungrounded memory number with a shape/dtype calculation. |
| Gemini mutual information, InfoNCE, policy entropy, RLHF, information bottleneck | Section 6 | Preserved as optional introductions with the mathematical object and purpose identified. No claim to fully teach the surrounding algorithms. |
| Generative-model and continuous-entropy boundaries | Section 6 | Kept VAE/GAN connections and corrected diffusion score terminology; differential entropy's negative-value example is marked as a different object. |
| Gemini weather and cat calculations | Section 7 experiment and Section 8 transfer | Weather values recomputed; corrected Cat NLL from 0.106460 to approximately 0.106472 nats. Rounded numbers no longer called exact sums. |
| Cognitive method table, promises, prechecked confidence | Outside lesson | Engine prompt now directs editorial manifests into the ledger. Lesson implements prediction, visual/code mapping, local definitions, explain-back, separated answers, and spaced retrieval without naming the authoring machinery. |
| External reading | Section 9 and nearby research citations | Exact book chapters, university-note sections, practice problem parts, and verified versioned API page. Video access limitations remain explicit. |

Second review findings and repairs:

1. The optional proof first defined natural logarithms, leaving a dependency for readers who skipped it. Added the base and spoken notation again at first main-route use in Section 4.
2. The loss comparison omitted a direct explanation of why neither fixed data entropy nor model confidence alone checks predictions against labels. Added that distinction in Section 5.
3. The coding theorem needed a precise code class. Changed its claim to uniquely decodable binary codes under independent finite-source assumptions.
4. MIT practice was described as a probability table, but the actual problem is a prose school/test-result scenario. Corrected the description after opening the PDF.
5. An extension introduction referred to the earlier draft. Removed that editorial reference from the learner explanation.
6. Explicitly named the conditional uncertainty baseline before relying on the conditional-entropy interpretation.

Validation performed on 2026-09-14:

- Command: `python MathsTerms/tests/eval_entropy_combined.py` — **8 tests passed**. This is a new evaluator for the comparison draft; the older active-pilot evaluator is not presented as evidence for it.
- Both embedded Python programs executed, including PyTorch 2.9.1+cpu loss/NLL equivalence, logit gradients, and a finite 1000-nat extreme-logit case.
- Recomputed known entropy, cross-entropy, KL, weather, Cat NLL, saturation, perplexity, smoothing, focal multipliers, and tensor-memory values.
- Tested **240 seeded distribution pairs**, comparing bit-to-nat scores with PyTorch and checking identity, nonnegativity, entropy bounds, and self-divergence. This is numerical evidence, not a proof.
- Checked the prefix property, example decoding, average code length, and distinct encodings for all 256 sequences of four symbols.
- Finite-difference gradients agree with autograd for a soft target; tests also cover batch-mean scaling and common-logit-shift invariance.
- Rejected empty, nonfinite, negative, nonnormalized, and unequal-length distributions; checked zero weights, impossible model support, and extremely small nonzero probabilities.
- Draft contents anchors and exact-case local links resolve. Corpus link scan: **1,629 relative links, zero broken file targets**. The corpus script does not prove semantic relevance or validate all corpus anchors.
- Source conversion produced **1 H1, 9 H2, 41 H3, 4 tables, and 7 code blocks**, with no raw fences. Display-math delimiter/environment balance, control-character, and whitespace checks passed. This is source-renderer validation, not a visual inspection of mathematical typesetting in every Markdown viewer.
- External pages were opened: Olah, official Deep Learning book, Wiley contents, Stanford notes, MIT problems/solutions, versioned PyTorch documentation, and the cited original research pages. Direct YouTube fetching failed; Khan's page returned no readable content. Neither is certified as an independently verified accessible lesson in this draft.
- No new subagents were used for this revision; the second review was a fresh main-agent pass, not an independent reviewer or learner test.

**Still pending:** the user's first reading and closed-notes transfer. Earlier structure-only “Pass” labels must not be treated as evidence that the prior or combined chapter is effective for this learner. Do not replace the canonical lesson or apply this format across folders until the comparison is accepted.

## Update rule

After every implemented learner task:

1. Update the task row and record the changed section.
2. Record mathematical and teaching evidence, not only “done.”
3. Perform a transition check on the preceding and following sections.
4. Decide whether the change alters a prerequisite, roadmap, earlier lesson, or lecture-package link.
5. Add any newly discovered defect to the relevant finding table.
6. Do not mark validation complete until the independent second review and mandatory checks have passed.
