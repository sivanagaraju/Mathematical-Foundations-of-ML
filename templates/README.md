# 🎨 Visualizer Template & Design System

A unified, high-contrast **Black & White** template system with **prominently visible borders** and **vividly colored mathematical diagrams**, designed for HTML5 visualizers and Next.js / React applications.

---

## 📁 Directory Structure

```
templates/
├── assets/
│   ├── css/
│   │   ├── tailwind.min.js      # Local standalone Tailwind CSS engine (offline capable)
│   │   └── template-theme.css   # Master high-contrast theme & visible border styles
│   └── js/
│       └── template-utils.js    # Shared Canvas, Animation & KaTeX helper utilities
├── template.html                # Boilerplate master HTML visualizer
├── TemplateComponent.jsx        # Reusable Next.js / React component template
├── tailwind.config.js           # Tailwind configuration extending dark palette
├── globals.css                  # Next.js global stylesheet with Tailwind directives
└── README.md                    # Documentation
```

---

## 🚀 How to Use in HTML Files

Link the local standalone assets directly in the `<head>` of any HTML file:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Your Visualizer Title</title>

  <!-- 1. Local Standalone Tailwind CSS Engine -->
  <script src="./assets/css/tailwind.min.js"></script>

  <!-- 2. Master High-Contrast Visible-Border Theme -->
  <link rel="stylesheet" href="./assets/css/template-theme.css">

  <!-- 3. KaTeX for Math Equations -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

  <!-- 4. Shared Canvas & Math Utilities -->
  <script src="./assets/js/template-utils.js"></script>
</head>
<body class="bg-black text-zinc-100 antialiased p-4 font-sans">
  <!-- Use .chapter-card or .template-card for visible borders -->
  <div class="template-card max-w-5xl mx-auto space-y-4">
    <h1 class="text-2xl font-bold text-white">Your Visualizer</h1>
    <canvas id="myCanvas" class="interactive-canvas" style="height: 320px;"></canvas>
  </div>
</body>
</html>
```

---

## ⚡ How to Use in Next.js & React (JavaScript)

1. Import `TemplateComponent.jsx` or `MathsTerms/DerivativesVisualizer.jsx`:
```jsx
'use client';
import TemplateComponent from '@/templates/TemplateComponent';

export default function Page() {
  return (
    <main className="min-h-screen bg-black py-8">
      <TemplateComponent 
        title="Gradient Descent Laboratory" 
        subtitle="Exploring Learning Rates and Contours"
      />
    </main>
  );
}
```

2. Add `templates/tailwind.config.js` and `templates/globals.css` to your Next.js project.

---

## 🎨 Color Palette & Design Rules

* **Shell & Cards**: Pure Black (`#000000`) base with deep charcoal cards (`#09090b`).
* **Visible Borders**: Every card and table cell uses high-contrast `border-zinc-700` (`#3f3f46`) and `border-zinc-600` (`#52525b`) with a `1.5px` border width.
* **Accent Highlight Borders**:
  * ✂️ **Secant Line**: `border-l-4 border-l-amber-500` (Amber `#f59e0b`)
  * ✨ **True Tangent Line**: `border-l-4 border-l-emerald-500` (Emerald `#10b981`)
  * ⚠️ **Error & Gradients**: `border-l-4 border-l-rose-500` (Rose `#f43f5e`)
* **Diagrams & Visualizations**: Kept in rich, vibrant colors (Sky Blue `#38bdf8` for function curves, Amber for secant chords, Emerald for tangent lines, and Rose for error deltas).
