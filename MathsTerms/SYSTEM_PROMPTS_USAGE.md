# Using the mathematics review prompts

## Files to give another LLM

1. [Master Editorial Prompt (Recommended)](./MASTER_EDITORIAL_PROMPT.md): Single consolidated authoritative standard (~20 KB) synthesizing developer pedagogy, step-by-step math, 14 canonical sections, 5-tier references, 12 warning patterns, hardware realities, and GenAI bridges.
2. [Editorial standard (Detailed Reference)](./EDITORIAL_SYSTEM_PROMPT.md): Shared teaching rules and the default minimum of 14 substantive sections.
3. [Chapter review and revision prompt (Detailed Reference)](./CHAPTER_REVIEW_AND_REVISION_SYSTEM_PROMPT.md): Detailed per-file investigation, recurring warning patterns, and validation.
4. [Cognitive learning prompt (Detailed Reference)](./COGNITIVE_LEARNING_ENGINE_SYSTEM_PROMPT.md): Supporting guidance for prediction, analogies, recall, and transfer.
5. The actual chapter files and relevant prerequisite material.

> **Tip**: For most LLM tasks, provide `MASTER_EDITORIAL_PROMPT.md` alone. It contains everything needed without overloading context.

## Example task message

Copy this message after supplying the prompt files and target chapters. Change the mode and targets to match the work you want.

```text
Use the supplied editorial, chapter-review, and cognitive-learning system prompts.

Mode: Review and revise.
Targets: All full concept Markdown chapters in these two folders:
- MathsTerms/03-Probability-and-Statistical-Estimation/
- MathsTerms/04-Information-Theory-and-Divergences/

Reader: A developer who can write basic Python but needs mathematical concepts
built carefully from explicitly stated prerequisites.

Keep the default minimum of 14 substantive sections. Merge only genuinely
duplicated, unhelpful, or inapplicable sections, with an evidence-based coverage map.

Read each entire chapter. First build its own concept and prerequisite map;
then assess recurring warning patterns and independently discover other issues.
Do not apply identical edits to every file or stop at the sample-audit findings.

Before editing each chapter, record a file-specific plan organized as:
Main Topic -> Subtopic -> Task/Subtask.

Preserve useful explanations. Fix mathematical errors and missing learning bridges.
Include clear stories, simple English, relevant ASCII, complete derivations,
term pronunciations, fair comparisons, meaningful practice, and verified resources.

Maintain a separate progress ledger recording evidence, changes, validation,
new dependencies, reopened tasks, and unresolved work. Recheck each chapter
freshly against all supplied requirements after implementing its plan.

Leave unrelated work unchanged. Report out-of-scope dependencies explicitly.
Do not claim code, rendering, references, or learner understanding were verified
unless the corresponding check actually happened.
```

## Review without changing chapters

Set `Mode: Review only` to receive findings, a per-file implementation plan, and validation evidence without chapter edits. A sample review assesses only the selected files; it does not certify the remaining folder.

## How the documents fit together

The editorial prompt defines the common standard. The new review prompt defines how to investigate and implement it for each chapter. The cognitive prompt supplies teaching techniques within that standard. All three now use the same 14-section default and documented exception policy.
