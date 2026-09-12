# Formulae Sheet — Lecture 16: Pedagogy in the Times of AI

## 1. Epistemic Verification & Hallucination Risk
- **Multi-step Derivation Accuracy:**
  $$P(\text{Success}) = \prod_{k=1}^K (1 - p_{\text{hallucination}, k}) = (1 - \bar{p})^K$$
- **Verification Cost vs Generation Cost:**
  $$\mathcal{E}_{\text{LLM}} \sim \mathcal{O}(1), \quad \mathcal{V}_{\text{Human}} = \sum_{k=1}^K \mathcal{C}(\text{Check}_k)$$
- If verification is skipped, cumulative error probability scales as:
  $$P(\text{Failure}) = 1 - (1 - \bar{p})^K \xrightarrow{K \gg 1} 1.0$$

---

## 2. Cognitive Load & Schema Retention
- **Total Cognitive Load:**
  $$L_{\text{total}} = L_{\text{intrinsic}} + L_{\text{extraneous}} + L_{\text{germane}}$$
- **Ebbinghaus Memory Retention Decay:**
  $$R(t) = \exp\left(-\frac{t}{S}\right), \quad S = f(L_{\text{germane}})$$
  - Passive prompting: $S \approx 1 \implies R(7) \approx 0.0009$
  - Active first-principles proof: $S \approx 14 \implies R(7) \approx 0.6065$

---

## 3. Shannon Channel Capacity in Learning
- **Information Channel Transmission:**
  $$C = B \log_2 \left( 1 + \frac{S}{N} \right)$$
  - $B$: Cognitive bandwidth of human working memory ($pprox 4\text{--}7$ conceptual chunks).
  - $S/N$: Ratio of signal (load-bearing mathematical invariants) to noise (unfiltered raw AI token stream).

---

## 4. Tensor Dimensionality
- Student mental state tensor $\theta$: `(D_concepts,)`
- Socratic probe query tensor $q$: `(B_probes, D_concepts)`
- LLM generated draft solution $x_{\text{draft}}$: `(L_tokens, D_model)`
- Verification mask $M_{\text{valid}}$: `(L_tokens,)` boolean array

---

## 5. Guarantees & Invariants
- **Verification Invariant:** No AI-generated mathematical step is accepted without an independent algebraic check or assertion script.
- **Dimensionality Conservation:** All tensor code must include explicit shape comments `# Shape: (...)` at every transformation.
- **Socratic Friction:** Learning is strictly proportional to the internal cognitive friction ($L_{\text{germane}}$) exerted during derivation.

---

## 6. Contrastive Decision Table

| Metric | Passive AI Consumer | Active Mathematical Sovereign |
|---|---|---|
| **Code Generation** | Copies LLM prompts verbatim | Uses AI as drafting tool; reviews line-by-line |
| **Debugging Method** | Pastes error back to LLM blindly | Traces tensor shapes and derives root cause |
| **Mathematical Mastery** | Memorizes surface-level names | Proves theorems from axioms with zero leaps |
| **Long-Term Retention** | Near-zero ($< 1\%$ after 1 week) | High ($> 60\%$ durable mental schema) |

---

## 7. Numerical Stability & Traps
- **The Prompting Echo Chamber:** Repeatedly asking an LLM to debug its own hallucinated equation often deepens the hallucination because the model conditions on its prior flawed context tokens.
- **Remedy:** Break context, isolate the single equation, and verify it with a 10-line numerical NumPy script.
