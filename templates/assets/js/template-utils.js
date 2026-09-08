/**
 * TEMPLATE-UTILS.JS
 * 
 * Shared JavaScript Utilities for Mathematical & AI Visualizers
 * - High-DPI Canvas initialization & auto-resizing
 * - KaTeX auto-rendering helper with safe delimiter handling
 * - Smooth limit animation interpolator (exponential decay h -> 0)
 * - Color palette accessor for consistent visual styles
 */

const VisualizerTheme = {
  colors: {
    curve: '#38bdf8',      // Electric Sky Blue
    secant: '#f59e0b',     // Amber
    tangent: '#10b981',    // Emerald Green
    error: '#f43f5e',      // Rose Red
    gradient: '#ec4899',   // Hot Pink
    contour: '#a855f7',    // Violet
    background: '#09090b', // Deep Charcoal
    grid: '#27272a',       // Subtle Grid
    axis: '#52525b',       // High-Contrast Axis
    textPrimary: '#ffffff',
    textSecondary: '#a1a1aa'
  },

  /**
   * Initializes a canvas with proper high-DPI (Retina) scaling
   * @param {HTMLCanvasElement} canvas 
   * @param {number} logicalHeight 
   * @returns {CanvasRenderingContext2D}
   */
  setupCanvas(canvas, logicalHeight = 340) {
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const logicalWidth = rect.width || 700;

    canvas.width = logicalWidth * dpr;
    canvas.height = logicalHeight * dpr;
    canvas.style.height = `${logicalHeight}px`;

    ctx.scale(dpr, dpr);
    return ctx;
  },

  /**
   * Initializes KaTeX auto-rendering on the page or specific container
   * @param {HTMLElement} [element=document.body]
   */
  initKaTeX(element = document.body) {
    if (typeof renderMathInElement === 'function') {
      renderMathInElement(element, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false }
        ],
        throwOnError: false
      });
    }
  },

  /**
   * Smoothly animates a parameter approaching a limit (e.g., h -> 0.01)
   * @param {number} startVal 
   * @param {number} targetVal 
   * @param {function} onStep - Callback with current intermediate value
   * @param {function} onComplete - Callback when limit is reached
   * @param {number} [decay=0.93] 
   * @returns {function} cancel - Function to cancel the animation
   */
  animateLimit(startVal, targetVal, onStep, onComplete, decay = 0.93) {
    let currentVal = startVal;
    let animId = null;

    const loop = () => {
      currentVal = currentVal * decay;
      if (currentVal <= targetVal + 0.005) {
        onStep(targetVal);
        if (typeof onComplete === 'function') onComplete();
      } else {
        onStep(Number(currentVal.toFixed(3)));
        animId = requestAnimationFrame(loop);
      }
    };

    animId = requestAnimationFrame(loop);
    return () => {
      if (animId) cancelAnimationFrame(animId);
    };
  }
};

// Export to window object for standalone HTML scripts
if (typeof window !== 'undefined') {
  window.VisualizerTheme = VisualizerTheme;
}
