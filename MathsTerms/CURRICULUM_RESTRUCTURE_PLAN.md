# MathsTerms Curriculum Restructure: Implementation Plan

## Purpose and decision boundary

This plan applies the chapter-level [`COGNITIVE_LEARNING_ENGINE_SYSTEM_PROMPT.md`](./COGNITIVE_LEARNING_ENGINE_SYSTEM_PROMPT.md) together with the repository-level [`EDITORIAL_SYSTEM_PROMPT.md`](./EDITORIAL_SYSTEM_PROMPT.md) to the six numbered `MathsTerms` folders. The intended learner is a developer who can write basic code but may have no dependable mathematics background.

The current stage has two goals:

1. Audit the complete curriculum, its current uncommitted changes, dependencies, references, and links from lecture notes.
2. Restructure and validate one pilot lesson, [`05-Information-Theory-and-Divergences/01-Entropy_CrossEntropy_CCE.md`](./05-Information-Theory-and-Divergences/01-Entropy_CrossEntropy_CCE.md), before choosing a pattern for the other lessons.

This stage does not authorize a mechanical rewrite of all 49 lessons. The pilot must first demonstrate a clearer learning journey.

## Initial evidence

The first repository scan found:

| Folder | Topic lessons | Current lesson lines |
| :--- | ---: | ---: |
| 01 — Primal Analysis and Foundations | 6 | 3,018 |
| 02 — Linear Algebra, Geometry, and Tensors | 9 | 2,937 |
| 03 — Multivariate Calculus and Optimization | 11 | 4,646 |
| 04 — Probability and Statistical Estimation | 7 | 2,827 |
| 05 — Information Theory and Divergences | 6 | 2,536 |
| 06 — Deep Architectures and Generative Models | 10 | 3,991 |
| **Total** | **49** | **19,955** |

There were 60 working-tree entries at the start of this audit, so existing edits must be treated as work to review and preserve where useful.

The current lessons are highly templated:

- 48 of 49 use the heading “The Missing Foundation.”
- 49 of 49 contain a “Beginner Comprehension Confidence Audit.”
- 46 of 49 contain a “Core Aha!” section.
- 49 of 49 contain a “Standalone Executable Python” section.

These sections may contain useful material. Their repetition does not prove that each concept has a coherent explanation. It can make the lessons feel like filled templates rather than chapters whose structure follows the learner's questions.

A character-level scan found invalid control characters in 23 lesson files. Several appear where LaTeX commands such as `\approx`, `\beta`, or `\text` should be. This is a rendering and correctness defect and will be included in repository-wide validation.

The entropy lesson illustrates the larger structural concern. It begins with metadata, a 14-part contents list, a unified concept tree, a historical story, a pronunciation table, an identity involving KL divergence, a comparison of losses, an LLM pipeline, multiple metaphors, a large glossary, formulas, hardware notes, examples, code, checks, and references. Much of that material is relevant, but its central learning chain is interrupted repeatedly. Some claims are also too strong or incorrect, including a reversed spoken direction for `D_KL(P || Q)`, a claim that cross-entropy optimization forces KL divergence to zero, and a claim of a unique finite global minimum with respect to logits.

## What the redesign must solve

The redesign must make these three layers agree:

```text
TOPIC: the folder's learning purpose and dependency position
  └── SUBTOPIC: one Markdown lesson with one coherent promise
        └── TASK: one learner action, such as interpreting, calculating,
                  proving, comparing, implementing, or diagnosing
```

A task is complete only when the learner has been given the concepts required to perform it. A section title or formula does not count as coverage by itself.

For every lesson, the work will answer:

- What question starts this chapter?
- What does the learner already know at that point?
- What is the smallest example that creates the need for this concept?
- Which new word or symbol appears next, and has it been explained first?
- Can the learner follow every step from the example to the definition?
- What nearby concept might they confuse with this one?
- Which theorem or proof is essential, and which depth can wait?
- Where does the idea appear in AI, and what is only approximate in practice?
- Can the learner solve an unseen example and explain when the method fails?

## Structural approaches considered

### Approach A — Keep one universal lesson template

This is closest to the current generated structure. It makes coverage easy to count, but it encourages unrelated sections, duplicated metaphors, and claims of completeness based on checked boxes. It is useful as an editorial checklist, not as the visible shape of every chapter.

### Approach B — Split every concept into small independent pages

For entropy, this could create separate pages for surprisal, Shannon entropy, cross-entropy, categorical cross-entropy, and perplexity. This improves focus but creates more navigation and prerequisite links. It can also make the learner reconstruct the story across too many files.

### Approach C — One narrative spine with optional depth lanes

This is the pilot approach. One ordinary problem and one small distribution carry the reader through the core chain. Proofs, engineering details, and extensions appear after the core understanding is established. A first-reading route identifies the sections needed for conceptual understanding without hiding assumptions. The upgraded cognitive engine adds prediction, integrated ELI5/self-explanation, purposeful dual coding, analogy limits, interleaved contrasts, and an end-of-chapter recall/spacing/transfer loop without turning those mechanisms into separate generated blocks.

### Approach D — Short core chapter plus reference appendices

If the pilot remains too large after restructuring, implementation or advanced proof material can move into a companion page. This decision will be based on the second review, not word count alone.

## Recommended teaching spine for the entropy pilot

The pilot will follow one continuous question:

> If a source produces several possible messages with different probabilities, how can one number describe how uncertain the next message is, and how does that lead to the loss used to train a classifier or language model?

The chapter will then move through these learner tasks:

1. Predict the number of yes/no questions needed when outcomes are equally likely.
2. Notice that a rare outcome is more surprising than a common outcome.
3. Derive the properties required of a surprise score and explain why a logarithm satisfies them.
4. Calculate the surprisal of one outcome.
5. Average those surprisals using their probabilities to obtain entropy.
6. Compare a fair and biased source and interpret the result.
7. Use an incorrect probability model and discover cross-entropy as expected surprise under that model.
8. Separate unavoidable uncertainty from extra mismatch using `H(P,Q) = H(P) + D_KL(P || Q)`.
9. Specialize cross-entropy to a one-hot classification target and obtain negative log-likelihood for the observed class.
10. Map the same calculation to next-token training without pretending that one example equals the full population objective.
11. Verify the calculation with stable library code.
12. Diagnose boundary cases, log bases, zero probabilities, restricted models, and perplexity interpretation.

The main story will establish entropy before KL divergence or gradients appear. The CCE gradient proof will be optional depth because it requires softmax and multivariable differentiation.

## Per-file implementation protocol

Every lesson will pass through the same work states, while its visible chapter structure remains concept-specific.

### Review 1 — Understand before editing

- Read the complete current file and its Git diff.
- Record its existing promise, examples, formulas, diagrams, references, and code.
- Identify the learner's first likely point of confusion.
- Trace every prerequisite actually used, including hidden school-level knowledge.
- Mark useful content to retain, claims to correct, sections to move, and content that does not serve the lesson.

### Lesson plan — Design the explanation

- Write one central learner question.
- Define the first-reading path and optional depth.
- Choose a single main example where possible.
- Create a transition map showing why each section follows the previous one.
- List the mathematical claims and conditions that require verification.

### Implementation — Edit by learner task

- Change one coherent learner task at a time.
- Update the progress ledger immediately after the task.
- Re-read the transition into and out of the changed section.
- If a new dependency appears, revisit earlier prerequisites and the folder roadmap.
- Preserve useful existing explanations while removing duplication and unsupported claims.

### Review 2 — Read the resulting lesson as a beginner

- Start only with the advertised prerequisites.
- Stop at every new term, symbol, transformation, or change of viewpoint.
- Verify that the explanation supplies the missing bridge or links to it.
- Check whether later details have weakened or contradicted earlier intuition.
- Revisit previous lessons if the new explanation changes their dependency or terminology.

### Validation — Collect evidence

Mark the lesson complete only after all applicable validations below pass.

## Mandatory in-depth validation

### 1. Narrative and transition validation

- Write the learner question answered by every major section.
- Confirm that each section creates the need for the next one.
- Identify every transition between story, example, definition, theorem, AI application, and code.
- Confirm that the learner has the concepts required at each transition.
- Remove or relocate sections that interrupt the main question without helping answer it.

### 2. Prerequisite validation

- Build the local and cross-folder dependency graph.
- Check for cycles and links to concepts taught later.
- Separate required-now, needed-later, and optional-depth prerequisites.
- Perform a symbol-first-use scan and a terminology-first-use scan.
- Confirm that the first-reading route is possible using only its listed prerequisites.

### 3. Mathematical claim validation

- Create a claim ledger for definitions, equalities, inequalities, existence, uniqueness, optimality, and convergence claims.
- Verify assumptions, domains, quantifiers, log bases, equality conditions, and edge cases.
- Distinguish definitions, derived identities, proofs, proof sketches, numerical demonstrations, and empirical observations.
- Check each AI mapping for the difference between a population quantity, empirical estimate, restricted model, and optimization result.

### 4. Numerical validation

- Recalculate every worked example independently.
- Check rounding, units, dimensions, probability normalization, and boundary conventions such as `0 log 0 = 0`.
- Use a second unseen example to test transfer.
- Test at least one counterexample to an overgeneralized claim.

### 5. Code validation

- Run all code blocks that are intended to execute.
- Compare code output with the displayed hand calculation.
- Check tensor shapes, dtypes, library input contracts, numerical stability, and edge cases.
- Ensure the code demonstrates the stated concept rather than being included only to satisfy a template.

### 6. Markdown and repository validation

- Detect invalid control characters and malformed LaTeX commands.
- Check heading hierarchy, contents anchors, fenced blocks, tables, and relative links.
- Check inbound links when files or anchors change.
- Verify that unrelated working-tree changes were preserved.

### 7. External-reference validation

- Open each recommended destination and verify its title and availability.
- State which precise part helps the learner and what prerequisite it assumes.
- Correctly distinguish a video, transcript, course, textbook, blog, documentation page, and primary paper.
- Do not claim that a resource was watched or fully reviewed when only metadata or a secondary reference was accessible.

### 8. Lecture-package integration validation

- Audit every `NOTES.md` and `PREREQUISITES.md` link into `MathsTerms`.
- Check whether the target lesson and anchor still teach the exact concept required at that point.
- Find important MathsTerms concepts mentioned in a lecture package without a useful link.
- Treat a resolving URL as necessary but insufficient; evaluate whether it is pedagogically placed.

### 9. Beginner-learning validation

The editorial team will perform a simulated beginner walkthrough, but it will not be reported as proof that a real beginner learned the material. The strongest evidence will come from a learner who can:

- explain the idea in their own words;
- calculate an unseen small example;
- read the central formula aloud and explain every symbol;
- choose between the concept and a nearby alternative;
- identify one situation where the formula does not apply directly;
- connect the mathematical objects to an AI implementation without confusing theory with estimation.

## Rollout stages

| Stage | Scope | Required output | Exit condition |
| :--- | :--- | :--- | :--- |
| 0. Preserve and inventory | All six folders and current Git changes | Counts, paths, diffs, initial defect scan | Every file is represented in the ledger |
| 1. Curriculum review | All six folders | Dependency graph, priority map, per-file first-review notes | No folder remains unreviewed at the structural level |
| 2. Integration review | `NOTES.md`, `PREREQUISITES.md`, MathsTerms links | Link and concept-coverage report | Broken, stale, missing, and weak links are distinguished |
| 3. Entropy pilot design | One lesson | Narrative spine, claim ledger, retain/move/remove decisions | Plan explains every major transition |
| 4. Entropy pilot implementation | One lesson and directly affected navigation | Restructured lesson | First-reading route is complete |
| 5. Entropy second review | Pilot and its prerequisites/consumers | Missing-bridge review and corrections | All newly discovered gaps are resolved or explicit |
| 6. Pilot validation | Pilot | Detailed validation evidence | All mandatory applicable checks pass |
| 7. Learner decision | User studies pilot | Feedback against learning tasks | User accepts, revises, or rejects the approach |
| 8. Curriculum rollout | Remaining files, in dependency order | Reviewed and validated lessons with updated ledger | Each file completes both reviews and validation |

## Progress and evidence

All file-level states, findings, implementation events, review loops, validation results, and delegated audit results are recorded in [`CURRICULUM_RESTRUCTURE_PROGRESS.md`](./CURRICULUM_RESTRUCTURE_PROGRESS.md). That file is the operational record; this file defines the method and completion criteria.
