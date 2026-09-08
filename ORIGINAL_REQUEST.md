# Original User Request

## 2026-09-04T13:15:31Z

Build a complete, standalone, interactive web learning studio that teaches differential calculus—derivatives, partial derivatives, gradients, Jacobians, and Hessians—from high school first principles to modern Generative AI. The application is specifically designed for adult learners returning to mathematics after 20 years, featuring a clean monochrome UI, vibrant interactive 3D/SVG visualizations, synchronized split-screen reading, audio pronunciations, step-by-step proofs with zero leaps, and contrastive "Why X, Not Y" explanations.

Working directory: c:/Users/sivan/Learning/Code/GenerativeAI/Mathematical-Foundations-of-ML/MathsTerms
Integrity mode: development

Reference Materials:
- Conceptual Reference: c:/Users/sivan/Learning/Code/GenerativeAI/Mathematical-Foundations-of-ML/MathsTerms/Derivatives_Gradients_and_Jacobians.md
- Prior Prototype: c:/Users/sivan/Learning/Code/GenerativeAI/Mathematical-Foundations-of-ML/MathsTerms/derivatives_gradients_jacobians_visualizer.html

## Requirements

### R1. Minimalist Monochrome Dark UI with Purposeful Colorful Visualizations
The overall UI chrome, typography, cards, and reading interface must be clean, elegant, and distraction-free black, charcoal, and white (zinc/slate dark mode). Saturated blue/cyan UI wrappers are removed. In contrast, all interactive visual elements (3D surfaces, vectors, tangent lines, contours, distortion grids, arrows) must feature distinct, vibrant pedagogical colors (e.g., emerald for level sets, crimson for steepest ascent, amber for tangents, violet for transformed coordinate vectors) so the mathematical insight stands out effortlessly.

### R2. Synchronized Parallel Split-Screen Architecture
The studio must implement a responsive split-screen layout (side-by-side on desktop/tablet, stacked on mobile). The left pane contains the story narrative, ELI5 intuition, step-by-step proofs, audio pronunciation cards, and contrastive case studies. The right pane hosts a synchronized, interactive 3D WebGL (Three.js) canvas or dynamic 2D SVG simulator with live sliders, draggable probes, and numerical telemetry that updates in real time as the user explores each section.

### R3. Story-Driven ELI5 Pedagogy with Zero Sudden Jumps
Every concept must start with a tangible real-world narrative and an "Explain Like I'm 5" physical analogy (e.g., car speedometer, hiking a foggy hill with a cane, stretching silly putty/dough, bicycle gear trains, adjusting hot and cold shower knobs) before introducing any mathematical symbols. Every mathematical formula must be derived from ground-floor primitives (arithmetic rise-over-run and geometric slopes) without skipping intermediate algebraic steps.

### R4. Pronunciation Guide with Native Web Speech Audio
Every mathematical symbol and Greek letter (∂, ∇, lim_{h→0}, J_ij, H_ij, ∈ ℝ^n) must feature:
1. Written phonetic pronunciation in plain English (e.g., ∇f: "DEL EFF" or "GRADIENT OF EFF").
2. Literal English meaning (e.g., "the vector that points directly toward the steepest uphill slope").
3. A clickable audio speaker button using the browser's native Web Speech Synthesis API (window.speechSynthesis) that clearly pronounces the term out loud.
4. Examples of how practitioners speak the formula in research and engineering discussions.

### R5. Explicit Contrastive "Why X, Not Y?" Learning
For every single mathematical concept, provide dedicated contrastive deep-dives explaining why obvious or naive alternatives fail catastrophically in machine learning:
- Why can't we just use a small fixed number like 0.01 instead of a calculus limit (h → 0)? (Approximation bias, floating-point catastrophic cancellation).
- Why can't we randomly guess weights or try grid search? (Curse of dimensionality: 10^100 combinations vs age of universe).
- Why can't we compute numerical gradients for 100 billion parameters? (Requires 100 billion forward passes per single step).
- Why do we compute Vector-Jacobian Products (VJPs) instead of the full Jacobian matrix in backprop? (Memory explosion: 10^5 × 10^5 matrix = 40 GB for a single layer).

### R6. Rigorous Step-by-Step Proofs with Full Algebraic Transparency
Include complete, unabridged line-by-line proofs for all core calculus theorems used in deep learning:
1. The Power Rule (d/dx x^n = n x^(n-1)) using the Binomial Theorem and factoring.
2. The Sum and Product Rules.
3. The Chain Rule (dz/dx = dz/dy * dy/dx) via linear approximation / limits.
4. Orthogonality of the Gradient: Proof that ∇f is strictly perpendicular (90°) to the level curves / contour lines using directional derivatives and the dot product.
5. Vector-Jacobian Product (VJP) Contraction: Algebraic proof showing that multiplying an incoming gradient vector by the Jacobian matrix produces exact sensitivities without materializing the matrix.

### R7. Interactive Multi-Level 3D WebGL (Three.js) & Responsive SVG Simulations
The application must provide high-performance, smooth interactive visualizations for every station:
- Station 1 (1D Derivative): Interactive curve (x^2, x^3, sin x) with an adjustable step slider h. Watch the secant line physically rotate and lock into the tangent line as h → 0, with real-time numeric readout of Δy/h.
- Station 2 (Partial Derivatives in 3D): 3D terrain surface with orbit controls. Toggle orthogonal cutting planes along the X-axis (freezing Y) and Y-axis (freezing X), showing how 3D surfaces decompose into two independent 1D curves.
- Station 3 (The Gradient Vector & 90° Contour Rule): Interactive 2D contour map and 3D surface. Drag an explorer puck across hills and valleys; watch the gradient vector arrow dynamically point toward the steepest ascent, maintaining exact 90° orthogonality to contour lines.
- Station 4 (The Jacobian Matrix as Space Transformation): Interactive 2D coordinate grid with circles and squares. Sliders manipulate matrix entries J = [a b; c d], demonstrating live stretching, rotation, shear, and area scaling (det J).
- Station 5 (Reverse-Mode Automatic Differentiation & VJP): Interactive multi-layer computational graph with forward values and animated backward gradient flow, showing how vector contractions bypass the Jacobian matrix bottleneck.

### R8. Standalone Self-Contained Deliverable
The application must be delivered as a single self-contained HTML file (derivatives_gradients_jacobians_masterclass.html) that loads all dependencies (Three.js, KaTeX, Tailwind CSS) via reliable public CDNs. It must run instantly by double-clicking in Chrome, Edge, Safari, or Firefox with zero local server, node, or npm requirements.

## Acceptance Criteria

### Pedagogical & Educational Criteria
- [ ] Every chapter opens with a story-first ELI5 real-world analogy before presenting any formula.
- [ ] No mathematical formulas appear without prior derivation or explicit justification of each intermediate algebraic step.
- [ ] Every single mathematical symbol and operator has an explicit phonetic guide and a working Web Speech audio button.
- [ ] Dedicated contrastive "Why X, Not Y?" sections exist for 1D limits, numerical gradients, the gradient vector, and the Jacobian matrix.
- [ ] Unabridged step-by-step proofs for the Power Rule, Chain Rule, Orthogonal Gradient, and VJP contraction are included.

### UI, UX & Visual Criteria
- [ ] The UI layout uses a parallel split-screen format with synchronized reading and visual panels.
- [ ] The general interface is styled in a clean, minimalist black-and-white / zinc dark mode without saturated blue or cyan background wrappers.
- [ ] All interactive 3D and 2D visual elements use vivid, distinct pedagogical colors to emphasize key mathematical concepts.
- [ ] KaTeX math formulas render with zero unrendered LaTeX or console errors.

### Visualizer & Simulation Criteria
- [ ] 3D scenes render smoothly at 60 FPS using Three.js with full mouse orbit, pan, and zoom controls.
- [ ] The 1D secant-to-tangent morpher updates dynamically as the slider h is dragged from 2.0 down to 0.0001.
- [ ] The 3D partial derivative viewer allows toggling and sliding X and Y slicing planes.
- [ ] The 2D contour explorer updates gradient vector arrows and contour orthogonal angles in real time.
- [ ] The Jacobian transformation grid dynamically warps unit circles and squares based on matrix slider inputs.
- [ ] The entire application works out-of-the-box as a single file opened locally via file:// protocol.
