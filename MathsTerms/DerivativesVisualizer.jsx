'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';

/**
 * DerivativesVisualizer.jsx
 * 
 * High-End Next.js / React Client Component for Calculus & AI First Principles:
 * - The Secant Line (Average Rate of Change) vs. The True Tangent Line (Instantaneous Derivative)
 * - Built with Tailwind CSS, Next.js / React, and Pure JavaScript
 * - Dark high-contrast monochrome base with prominent visible borders (border-zinc-700 / border-zinc-600)
 * - High-visibility highlights (Amber for Secant, Emerald for Tangent, Rose for Error Gap, Sky Blue for Function)
 * - Deep mathematical explanations, algebraic cancellations, 0/0 paradox, and two concrete worked examples.
 */
export default function DerivativesVisualizer() {
  // --- Interactive State ---
  const [x0, setX0] = useState(1.0);
  const [h, setH] = useState(1.5);
  const [showSecant, setShowSecant] = useState(true);
  const [showTangent, setShowTangent] = useState(true);
  const [showTriangle, setShowTriangle] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [isAnimating, setIsAnimating] = useState(false);
  const [activeTab, setActiveTab] = useState('lab'); // 'lab' | 'theory' | 'examples' | 'matrix' | 'ai'

  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Parabola function f(x) = x^2
  const f = useCallback((x) => x * x, []);
  // Exact analytical derivative f'(x) = 2x
  const df = useCallback((x) => 2 * x, []);

  // Calculated values
  const y0 = f(x0);
  const x1 = x0 + h;
  const y1 = f(x1);
  const secantSlope = (y1 - y0) / h;
  const tangentSlope = df(x0);
  const absoluteError = Math.abs(secantSlope - tangentSlope);
  const relativeError = tangentSlope !== 0 
    ? Math.abs((secantSlope - tangentSlope) / tangentSlope) * 100 
    : 0;

  // --- Smooth Animation Effect for h -> 0.01 ---
  useEffect(() => {
    if (!isAnimating) return;

    let currentH = h;
    const targetH = 0.01;
    const decay = 0.94; // Smooth exponential decay

    const step = () => {
      currentH = currentH * decay;
      if (currentH <= targetH + 0.005) {
        setH(targetH);
        setIsAnimating(false);
      } else {
        setH(Number(currentH.toFixed(3)));
        animationFrameRef.current = requestAnimationFrame(step);
      }
    };

    animationFrameRef.current = requestAnimationFrame(step);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isAnimating]);

  const handleAnimate = () => {
    if (h <= 0.05) {
      setH(2.0); // Reset to large interval first
      setTimeout(() => setIsAnimating(true), 50);
    } else {
      setIsAnimating(true);
    }
  };

  // --- Canvas Rendering Loop ---
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Retina / High-DPI support
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = 340;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Coordinate space transformations
    const xMin = -2.5;
    const xMax = 4.5;
    const yMin = -1.5;
    const yMax = 12.0;

    const toCanvasX = (x) => ((x - xMin) / (xMax - xMin)) * width;
    const toCanvasY = (y) => height - ((y - yMin) / (yMax - yMin)) * height;

    // 1. Clear & Background
    ctx.fillStyle = '#09090b';
    ctx.fillRect(0, 0, width, height);

    // 2. Coordinate Grid
    if (showGrid) {
      ctx.strokeStyle = '#27272a';
      ctx.lineWidth = 1;

      // Vertical grid lines
      for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
        const cx = toCanvasX(x);
        ctx.beginPath();
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, height);
        ctx.stroke();

        ctx.fillStyle = '#52525b';
        ctx.font = '10px monospace';
        ctx.fillText(x.toString(), cx + 4, height - 8);
      }

      // Horizontal grid lines
      for (let y = 0; y <= Math.floor(yMax); y += 2) {
        const cy = toCanvasY(y);
        ctx.beginPath();
        ctx.moveTo(0, cy);
        ctx.lineTo(width, cy);
        ctx.stroke();

        ctx.fillStyle = '#52525b';
        ctx.font = '10px monospace';
        ctx.fillText(y.toString(), 6, cy - 4);
      }

      // Axis lines
      ctx.strokeStyle = '#52525b';
      ctx.lineWidth = 1.5;
      // X-Axis (y=0)
      const cy0 = toCanvasY(0);
      ctx.beginPath();
      ctx.moveTo(0, cy0);
      ctx.lineTo(width, cy0);
      ctx.stroke();

      // Y-Axis (x=0)
      const cx0 = toCanvasX(0);
      ctx.beginPath();
      ctx.moveTo(cx0, 0);
      ctx.lineTo(cx0, height);
      ctx.stroke();
    }

    // 3. Draw Parabola Function Curve f(x) = x^2 (Electric Sky Blue)
    ctx.beginPath();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 3;
    const steps = 250;
    for (let i = 0; i <= steps; i++) {
      const xVal = xMin + (i / steps) * (xMax - xMin);
      const yVal = f(xVal);
      const cx = toCanvasX(xVal);
      const cy = toCanvasY(yVal);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // 4. Draw Delta Triangle (Run = h, Rise = Delta y) & Error Wedge
    if (showTriangle && Math.abs(h) > 0.05) {
      const px0 = toCanvasX(x0);
      const py0 = toCanvasY(y0);
      const px1 = toCanvasX(x1);
      const py1 = toCanvasY(y1);

      // Shaded triangle area
      ctx.fillStyle = 'rgba(245, 158, 11, 0.08)';
      ctx.beginPath();
      ctx.moveTo(px0, py0);
      ctx.lineTo(px1, py0);
      ctx.lineTo(px1, py1);
      ctx.closePath();
      ctx.fill();

      // Horizontal run (Delta x = h)
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(px0, py0);
      ctx.lineTo(px1, py0);
      ctx.stroke();

      // Vertical rise (Delta y)
      ctx.beginPath();
      ctx.moveTo(px1, py0);
      ctx.lineTo(px1, py1);
      ctx.stroke();
      ctx.setLineDash([]);

      // Triangle labels
      ctx.fillStyle = '#fbbf24';
      ctx.font = '11px monospace';
      ctx.fillText(`Δx = h = ${h.toFixed(2)}`, px0 + (px1 - px0) / 2 - 25, py0 + 16);
      ctx.fillText(`Δy = ${(y1 - y0).toFixed(2)}`, px1 + 8, py0 - (py0 - py1) / 2);
    }

    // 5. Draw Secant Line (Amber / Orange Dashed)
    if (showSecant) {
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 4]);
      const xA = xMin;
      const yA = y0 + secantSlope * (xA - x0);
      const xB = xMax;
      const yB = y0 + secantSlope * (xB - x0);

      ctx.beginPath();
      ctx.moveTo(toCanvasX(xA), toCanvasY(yA));
      ctx.lineTo(toCanvasX(xB), toCanvasY(yB));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 6. Draw True Tangent Line (Vivid Emerald Green Solid)
    if (showTangent) {
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.5;
      const xA = xMin;
      const yA = y0 + tangentSlope * (xA - x0);
      const xB = xMax;
      const yB = y0 + tangentSlope * (xB - x0);

      ctx.beginPath();
      ctx.moveTo(toCanvasX(xA), toCanvasY(yA));
      ctx.lineTo(toCanvasX(xB), toCanvasY(yB));
      ctx.stroke();
    }

    // 7. Base Point P (x0, y0) Glowing Cyan Pin
    const pX = toCanvasX(x0);
    const pY = toCanvasY(y0);
    ctx.beginPath();
    ctx.arc(pX, pY, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#38bdf8';
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px sans-serif';
    ctx.fillText(`P (${x0.toFixed(2)}, ${y0.toFixed(2)})`, pX - 45, pY - 14);

    // 8. Secondary Point Q (x0+h, y1) Glowing Amber Pin
    const qX = toCanvasX(x1);
    const qY = toCanvasY(y1);
    ctx.beginPath();
    ctx.arc(qX, qY, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#f59e0b';
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();

    ctx.fillStyle = '#fef08a';
    ctx.font = 'bold 11px sans-serif';
    ctx.fillText(`Q (${x1.toFixed(2)}, ${y1.toFixed(2)})`, qX + 12, qY - 6);

  }, [x0, h, showSecant, showTangent, showTriangle, showGrid, f, df, secantSlope, tangentSlope, y0, x1, y1]);

  return (
    <div className="w-full max-w-7xl mx-auto p-4 sm:p-6 bg-black text-zinc-100 font-sans space-y-6">
      
      {/* ════════════════════════════════════════════════════════════════
          HEADER: CLEAN HIGH-CONTRAST MONOCHROME WITH CRISP VISIBLE BORDERS
          ════════════════════════════════════════════════════════════════ */}
      <header className="bg-zinc-950 border-2 border-zinc-700 rounded-2xl p-5 sm:p-7 shadow-2xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-3 py-1 text-xs font-mono font-bold rounded-full bg-zinc-900 text-zinc-200 border border-zinc-600 uppercase tracking-wider">
                First Principles AI & Mathematics
              </span>
              <span className="px-3 py-1 text-xs font-mono rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-600/60 font-semibold">
                Secant Line ➔ True Tangent Line Limit
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              The Speedometer Engine: Secant vs. True Tangent Line
            </h1>
            <p className="text-xs sm:text-sm text-zinc-400 max-w-3xl leading-relaxed">
              Experience the transition from a 2-point average chord (cutting through the curve) 
              to an instantaneous radar limit (grazing at a single infinitesimal point).
            </p>
          </div>

          {/* Quick Stats Pill */}
          <div className="flex items-center gap-3 bg-zinc-900/90 p-3 rounded-xl border border-zinc-600 self-start lg:self-auto">
            <div className="text-right">
              <div className="text-[10px] text-zinc-400 uppercase font-mono tracking-wider">Current Limit State</div>
              <div className="text-sm font-bold text-white font-mono">
                {h <= 0.02 ? '✨ Instant Radar Limit' : `✂️ Secant Chord (h = ${h.toFixed(2)})`}
              </div>
            </div>
            <div className="w-10 h-10 rounded-lg bg-black border border-zinc-600 flex items-center justify-center font-bold text-amber-400 font-mono text-sm">
              {h <= 0.02 ? '0.00' : h.toFixed(1)}
            </div>
          </div>
        </div>

        {/* TOP TAB NAVIGATION */}
        <nav className="flex bg-zinc-900/80 p-1.5 rounded-xl border border-zinc-700 text-xs font-semibold overflow-x-auto gap-2 mt-6">
          <button
            onClick={() => setActiveTab('lab')}
            className={`px-4 py-2 rounded-lg whitespace-nowrap transition flex items-center gap-2 ${
              activeTab === 'lab'
                ? 'bg-white text-black font-bold shadow-md'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
            }`}
          >
            <span>📈 Interactive Laboratory</span>
          </button>
          <button
            onClick={() => setActiveTab('theory')}
            className={`px-4 py-2 rounded-lg whitespace-nowrap transition flex items-center gap-2 ${
              activeTab === 'theory'
                ? 'bg-white text-black font-bold shadow-md'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
            }`}
          >
            <span>📐 Theory & Etymology</span>
          </button>
          <button
            onClick={() => setActiveTab('examples')}
            className={`px-4 py-2 rounded-lg whitespace-nowrap transition flex items-center gap-2 ${
              activeTab === 'examples'
                ? 'bg-white text-black font-bold shadow-md'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
            }`}
          >
            <span>🔢 Worked Step-by-Step Examples</span>
          </button>
          <button
            onClick={() => setActiveTab('matrix')}
            className={`px-4 py-2 rounded-lg whitespace-nowrap transition flex items-center gap-2 ${
              activeTab === 'matrix'
                ? 'bg-white text-black font-bold shadow-md'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
            }`}
          >
            <span>⚖️ 8-Point Comparison Matrix</span>
          </button>
          <button
            onClick={() => setActiveTab('ai')}
            className={`px-4 py-2 rounded-lg whitespace-nowrap transition flex items-center gap-2 ${
              activeTab === 'ai'
                ? 'bg-white text-black font-bold shadow-md'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
            }`}
          >
            <span>🚨 Why Modern AI Cannot Use Secants</span>
          </button>
        </nav>
      </header>

      {/* ════════════════════════════════════════════════════════════════
          TAB 1: INTERACTIVE LABORATORY (CANVAS + HUD + CONTROLS)
          ════════════════════════════════════════════════════════════════ */}
      {activeTab === 'lab' && (
        <section className="bg-zinc-950 border-2 border-zinc-700 rounded-2xl p-5 sm:p-7 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* Left 7 Columns: Canvas & Interactive Controls */}
            <div className="lg:col-span-7 space-y-4">
              <div className="flex justify-between items-center text-xs">
                <span className="font-bold text-white flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                  <span>Interactive Secant-to-Tangent Trajectory Graph</span>
                </span>
                <span className="font-mono text-zinc-300 bg-zinc-900 px-2.5 py-1 rounded border border-zinc-600 text-[11px]">
                  Trajectory: f(x) = x²
                </span>
              </div>

              {/* Canvas Container */}
              <div className="relative rounded-xl border-2 border-zinc-700 overflow-hidden bg-[#09090b]">
                <canvas
                  ref={canvasRef}
                  className="w-full block"
                  style={{ height: '340px' }}
                />
              </div>

              {/* Sliders Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 rounded-xl bg-zinc-900/60 border border-zinc-700 text-xs">
                <div>
                  <div className="flex justify-between font-mono mb-1.5">
                    <span className="text-zinc-400">Base Point x₀:</span>
                    <span className="text-sky-300 font-bold">{x0.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="-1.5"
                    max="2.0"
                    step="0.05"
                    value={x0}
                    onChange={(e) => setX0(parseFloat(e.target.value))}
                    className="w-full accent-white cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-zinc-400 font-mono mt-1">
                    <span>-1.50</span>
                    <span>0.00</span>
                    <span>+2.00</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between font-mono mb-1.5">
                    <span className="text-zinc-400">Interval Window h (Δx):</span>
                    <span className="text-amber-300 font-bold">{h.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="2.5"
                    step="0.01"
                    value={h}
                    onChange={(e) => setH(parseFloat(e.target.value))}
                    className="w-full accent-white cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-zinc-400 font-mono mt-1">
                    <span>h → 0.01 (Limit)</span>
                    <span>h = 1.00</span>
                    <span>h = 2.50</span>
                  </div>
                </div>
              </div>

              {/* Toggles & Animation Button */}
              <div className="flex items-center justify-between gap-3 p-3 rounded-xl bg-zinc-900/40 border border-zinc-700 flex-wrap text-xs">
                <div className="flex items-center gap-4 flex-wrap">
                  <label className="flex items-center gap-1.5 cursor-pointer text-zinc-300 hover:text-white">
                    <input
                      type="checkbox"
                      checked={showSecant}
                      onChange={(e) => setShowSecant(e.target.checked)}
                      className="accent-amber-500 rounded"
                    />
                    <span className="text-amber-400 font-semibold">Show Secant (Dashed)</span>
                  </label>

                  <label className="flex items-center gap-1.5 cursor-pointer text-zinc-300 hover:text-white">
                    <input
                      type="checkbox"
                      checked={showTangent}
                      onChange={(e) => setShowTangent(e.target.checked)}
                      className="accent-emerald-500 rounded"
                    />
                    <span className="text-emerald-400 font-semibold">Show Tangent (Solid)</span>
                  </label>

                  <label className="flex items-center gap-1.5 cursor-pointer text-zinc-300 hover:text-white">
                    <input
                      type="checkbox"
                      checked={showTriangle}
                      onChange={(e) => setShowTriangle(e.target.checked)}
                      className="accent-white rounded"
                    />
                    <span>Δx, Δy Triangle</span>
                  </label>
                </div>

                <button
                  onClick={handleAnimate}
                  className="px-3.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-white font-mono font-bold border border-zinc-600 transition flex items-center gap-1.5 shadow-sm"
                >
                  <span>{isAnimating ? '⏸ Animating...' : '▶ Animate Limit (h ➔ 0)'}</span>
                </button>
              </div>

              {/* Presets Row */}
              <div className="flex items-center gap-2 pt-2 border-t border-zinc-800 flex-wrap text-xs">
                <span className="text-zinc-400 text-[11px] font-mono">Presets:</span>
                {[
                  { label: 'h = 2.00 (Trip Avg)', val: 2.0 },
                  { label: 'h = 1.00', val: 1.0 },
                  { label: 'h = 0.50', val: 0.5 },
                  { label: 'h = 0.10', val: 0.1 },
                  { label: 'h ➔ 0.01 (Radar Limit)', val: 0.01, highlight: true }
                ].map((item) => (
                  <button
                    key={item.label}
                    onClick={() => setH(item.val)}
                    className={`px-3 py-1 rounded text-[11px] font-mono transition border ${
                      item.highlight
                        ? 'bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold border-emerald-400 shadow-md'
                        : 'bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border-zinc-700'
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Right 5 Columns: Live Telemetry HUD & Formulas */}
            <div className="lg:col-span-5 space-y-4 flex flex-col justify-between text-xs">
              
              {/* Telemetry Cards Box */}
              <div className="p-4 rounded-xl bg-zinc-900/60 border-2 border-zinc-700 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-700 pb-2">
                  <h3 className="font-bold text-white text-sm flex items-center gap-2">
                    <span>🚓 Live Sensitivity Telemetry</span>
                  </h3>
                  <span className="font-mono text-xs text-zinc-400">At x₀ = {x0.toFixed(2)}</span>
                </div>

                {/* Side-by-side Metric Badges */}
                <div className="grid grid-cols-2 gap-3">
                  {/* Secant Average Speed */}
                  <div className="p-3 rounded-lg bg-zinc-950 border-2 border-amber-500/60 shadow-lg shadow-amber-950/20">
                    <div className="text-[10px] text-amber-400 font-bold uppercase tracking-wider flex items-center gap-1">
                      <span>✂️</span> Average Speed (Secant)
                    </div>
                    <div className="text-2xl font-mono font-extrabold text-amber-300 mt-1">
                      {secantSlope.toFixed(2)}
                    </div>
                    <div className="text-[10px] text-zinc-400 font-mono mt-1">
                      m_sec = Δy / h
                    </div>
                  </div>

                  {/* Tangent Instant Radar Speed */}
                  <div className="p-3 rounded-lg bg-zinc-950 border-2 border-emerald-500/60 shadow-lg shadow-emerald-950/20">
                    <div className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider flex items-center gap-1">
                      <span>✨</span> Radar Speed (Tangent)
                    </div>
                    <div className="text-2xl font-mono font-extrabold text-emerald-300 mt-1">
                      {tangentSlope.toFixed(2)}
                    </div>
                    <div className="text-[10px] text-zinc-400 font-mono mt-1">
                      m_tan = f'(x₀) = 2x₀
                    </div>
                  </div>
                </div>

                {/* Error Metric */}
                <div className="p-3 rounded-lg bg-zinc-950 border-2 border-rose-500/50 font-mono text-[11px] flex justify-between items-center shadow-lg shadow-rose-950/20">
                  <span className="text-rose-400 font-semibold flex items-center gap-1.5">
                    <span>⚠️</span> Approximation Gap:
                  </span>
                  <span className="text-rose-300 font-bold">
                    {absoluteError.toFixed(2)} ({relativeError.toFixed(1)}% error)
                  </span>
                </div>

                {/* Live Point-Slope Equations */}
                <div className="p-3 rounded-lg bg-black border border-zinc-700 space-y-2 font-mono text-[11px]">
                  <div className="text-zinc-400 text-[10px] uppercase font-bold tracking-wider">
                    Dynamic Point-Slope Line Equations:
                  </div>
                  <div className="flex justify-between items-center text-amber-300">
                    <span className="text-zinc-400">Secant Line:</span>
                    <span>y - {y0.toFixed(2)} = <strong>{secantSlope.toFixed(2)}</strong>(x - {x0.toFixed(2)})</span>
                  </div>
                  <div className="flex justify-between items-center text-emerald-300">
                    <span className="text-zinc-400">Tangent Line:</span>
                    <span>y - {y0.toFixed(2)} = <strong>{tangentSlope.toFixed(2)}</strong>(x - {x0.toFixed(2)})</span>
                  </div>
                </div>
              </div>

              {/* Mathematical Bridge Box */}
              <div className="p-4 rounded-xl bg-zinc-900/60 border-2 border-zinc-700 space-y-2 text-xs">
                <h4 className="font-bold text-white text-sm">How the Limit Bridges the Two:</h4>
                <div className="bg-black p-3 rounded-lg border border-zinc-700 font-mono text-xs text-white text-center">
                  f'(x₀) = lim[h ➔ 0] ( f(x₀ + h) - f(x₀) ) / h
                </div>
                <p className="text-[11px] text-zinc-400 leading-relaxed">
                  As the interval window <span className="font-mono text-white">h</span> shrinks toward zero, 
                  point <span className="font-mono text-amber-300">Q</span> slides along the parabola into point <span className="font-mono text-sky-300">P</span>. 
                  The slicing chord rotates until it gently kisses the curve at that exact microsecond.
                </p>
              </div>

              {/* AI Deep Learning Connection */}
              <div className="p-3.5 rounded-xl bg-zinc-900/80 border-2 border-zinc-700 text-xs text-zinc-300 leading-relaxed">
                🔗 <strong>The Deep Learning Connection:</strong> In AI training, <span className="font-mono text-white">x</span> is a neural network weight <span className="font-mono text-white">w</span>, and <span className="font-mono text-white">f(x)</span> is the Loss function <span className="font-mono text-white">L(w)</span>. Gradient descent uses the <em>True Tangent</em> (calculated via backpropagation) to determine which way to nudge the weight knob.
              </div>

            </div>
          </div>
        </section>
      )}

      {/* ════════════════════════════════════════════════════════════════
          TAB 2: THEORY & ETYMOLOGY (THE 2-POINT CHORD VS 1-POINT TANGENT)
          ════════════════════════════════════════════════════════════════ */}
      {activeTab === 'theory' && (
        <section className="bg-zinc-950 border-2 border-zinc-700 rounded-2xl p-5 sm:p-7 space-y-6">
          <div className="border-b border-zinc-700 pb-3">
            <h3 className="text-xl font-extrabold text-white">
              What Exactly Is the Difference Between a Secant Line and a True Tangent Line?
            </h3>
            <p className="text-zinc-400 text-xs mt-1">
              To build an intuitive mental model, let us examine both lines through etymology, geometry, and physical meaning.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* The Secant Line Card */}
            <div className="p-5 rounded-xl bg-zinc-900/70 border-2 border-l-4 border-zinc-700 border-l-amber-500 space-y-3 shadow-lg shadow-amber-950/10">
              <div className="flex items-center justify-between">
                <h4 className="font-bold text-amber-300 text-base flex items-center gap-2">
                  <span>✂️ The Secant Line (Average Slope)</span>
                </h4>
                <span className="text-[10px] font-mono text-zinc-300 bg-zinc-800 px-2.5 py-1 rounded border border-zinc-600">
                  Latin: secare (&quot;to cut&quot;)
                </span>
              </div>
              <p className="text-zinc-300 text-xs leading-relaxed">
                The word <em>secant</em> comes from the Latin verb <em>secare</em>, meaning <strong>&quot;to cut or sever&quot;</strong> 
                (the linguistic origin of words like <em>dissect</em>, <em>section</em>, and <em>sector</em>).
              </p>
              <div className="p-3 rounded-lg bg-black border border-zinc-700 font-mono text-xs text-zinc-200 space-y-1">
                <div>• Point 1: P = (x₀, f(x₀))</div>
                <div>• Point 2: Q = (x₀ + h, f(x₀ + h)) &nbsp; [h ≠ 0]</div>
                <div className="text-amber-400 font-bold pt-1">
                  m_sec = ( f(x₀ + h) - f(x₀) ) / h
                </div>
                <div className="text-zinc-400">Line: y - f(x₀) = m_sec(x - x₀)</div>
              </div>
              <p className="text-zinc-400 text-xs">
                <strong>Geometric Behavior:</strong> The line plunges directly through the curve, cutting across its interior like a bridge over a valley. It only knows the starting and ending points, completely blind to what happened in between.
              </p>
            </div>

            {/* The True Tangent Line Card */}
            <div className="p-5 rounded-xl bg-zinc-900/70 border-2 border-l-4 border-zinc-700 border-l-emerald-500 space-y-3 shadow-lg shadow-emerald-950/10">
              <div className="flex items-center justify-between">
                <h4 className="font-bold text-emerald-300 text-base flex items-center gap-2">
                  <span>✨ The True Tangent Line (Instantaneous Slope)</span>
                </h4>
                <span className="text-[10px] font-mono text-zinc-300 bg-zinc-800 px-2.5 py-1 rounded border border-zinc-600">
                  Latin: tangere (&quot;to touch&quot;)
                </span>
              </div>
              <p className="text-zinc-300 text-xs leading-relaxed">
                The word <em>tangent</em> originates from the Latin verb <em>tangere</em>, meaning <strong>&quot;to touch gently&quot;</strong> 
                (the root of words like <em>tactile</em>, <em>tangible</em>, and <em>contact</em>).
              </p>
              <div className="p-3 rounded-lg bg-black border border-zinc-700 font-mono text-xs text-zinc-200 space-y-1">
                <div>• Single Base Point: P = (x₀, f(x₀))</div>
                <div>• Secondary Point: Q ➔ P as h ➔ 0</div>
                <div className="text-emerald-400 font-bold pt-1">
                  m_tan = f&apos;(x₀) = lim[h ➔ 0] ( f(x₀ + h) - f(x₀) ) / h
                </div>
                <div className="text-zinc-400">Line: y - f(x₀) = f&apos;(x₀)(x - x₀)</div>
              </div>
              <p className="text-zinc-400 text-xs">
                <strong>Geometric Behavior:</strong> The line touches the curve at exactly one infinitesimal point without cutting through its immediate neighborhood. It represents the best first-order linear approximation of the system at that instant.
              </p>
            </div>
          </div>

          {/* Physical Story: Car on Highway & Police Radar Gun */}
          <div className="p-5 rounded-xl bg-zinc-900/60 border-2 border-zinc-700 space-y-3">
            <h4 className="font-bold text-white text-base flex items-center gap-2">
              <span>🚗 The Story of the Road Trip and the Police Radar Gun</span>
            </h4>
            <p className="text-zinc-300 text-xs leading-relaxed">
              Imagine you drive from Los Angeles to San Diego—a total distance of <strong>120 miles in exactly 2 hours</strong>.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="p-4 rounded-lg bg-black border border-amber-500/40 space-y-2">
                <div className="text-amber-400 font-bold font-mono">The Secant Calculation (Highway Toll Booth):</div>
                <div className="font-mono text-zinc-300">
                  Average Speed = Total Distance / Total Time = 120 mi / 2 hr = <strong>60 mph</strong>
                </div>
                <p className="text-zinc-400">
                  The toll booth sees a safe, legal average speed of 60 mph. But does this mean you never exceeded the speed limit?
                </p>
              </div>

              <div className="p-4 rounded-lg bg-black border border-emerald-500/40 space-y-2">
                <div className="text-emerald-400 font-bold font-mono">The True Tangent (Police Radar Gun):</div>
                <div className="font-mono text-zinc-300">
                  Instantaneous Speed = lim[Δt ➔ 0] (Δs / Δt) = <strong>95 mph</strong>
                </div>
                <p className="text-zinc-400">
                  You stopped at a diner for 30 minutes, then drove 95 mph to make up time. When a police officer shoots their radar gun, it measures the <em>True Tangent</em> at that exact microsecond. The officer tickets you based on the tangent, not the secant!
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ════════════════════════════════════════════════════════════════
          TAB 3: WORKED STEP-BY-STEP EXAMPLES (WITH 0/0 PARADOX)
          ════════════════════════════════════════════════════════════════ */}
      {activeTab === 'examples' && (
        <section className="bg-zinc-950 border-2 border-zinc-700 rounded-2xl p-5 sm:p-7 space-y-6">
          <div className="border-b border-zinc-700 pb-3">
            <h3 className="text-xl font-extrabold text-white">
              Concrete Numerical Worked Examples
            </h3>
            <p className="text-zinc-400 text-xs mt-1">
              Step-by-step arithmetic and algebraic proofs showing the collapse of secant slopes into exact tangent slopes.
            </p>
          </div>

          {/* Example 1: f(x) = x^2 at x0 = 2.0 */}
          <div className="p-5 rounded-xl bg-zinc-900/60 border-2 border-zinc-700 space-y-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded text-[10px] font-bold font-mono bg-zinc-800 text-zinc-200 border border-zinc-600">
                  EXAMPLE 1
                </span>
                <h4 className="font-bold text-white text-base">
                  Parabola Trajectory: f(x) = x² at Base Point x₀ = 2.0
                </h4>
              </div>
              <span className="text-xs font-mono text-emerald-400 font-bold">
                Analytical Tangent: f&apos;(2) = 2(2) = 4.000
              </span>
            </div>

            {/* Algebraic Cancellation Proof Box */}
            <div className="p-4 rounded-xl bg-black border border-zinc-600 font-mono text-xs space-y-2">
              <div className="text-zinc-400 uppercase text-[10px] font-bold">
                Algebraic Derivation of Secant Slope:
              </div>
              <div className="text-zinc-200 leading-relaxed">
                m_sec = [ (2 + h)² - 2² ] / h<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = [ 4 + 4h + h² - 4 ] / h<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = [ 4h + h² ] / h<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = [ h(4 + h) ] / h<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = <strong className="text-amber-300">4 + h</strong> &nbsp; (for all h ≠ 0)
              </div>
              <p className="text-zinc-400 font-sans text-xs pt-1">
                💡 <strong>The Key Revelation:</strong> The secant slope for f(x) = x² at x₀ = 2 is always <strong>exactly 4 + h</strong>! 
                The error is literally equal to the interval h itself. When h = 2, error is 2. When h = 0.01, error is 0.01. As h ➔ 0, 4 + h ➔ 4.000!
              </p>
            </div>

            {/* Shrinking Intervals Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono border-collapse border border-zinc-700 rounded-lg overflow-hidden">
                <thead className="bg-zinc-900 text-zinc-300 uppercase text-[10px]">
                  <tr>
                    <th className="p-3 border border-zinc-700">Step</th>
                    <th className="p-3 border border-zinc-700">Interval h</th>
                    <th className="p-3 border border-zinc-700">Point Q (2 + h)</th>
                    <th className="p-3 border border-zinc-700">f(2 + h)</th>
                    <th className="p-3 border border-zinc-700">Rise Δy</th>
                    <th className="p-3 border border-zinc-700">Secant m_sec</th>
                    <th className="p-3 border border-zinc-700">True Tangent</th>
                    <th className="p-3 border border-zinc-700">Absolute Error</th>
                    <th className="p-3 border border-zinc-700">Rel Error %</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800 bg-black text-zinc-300">
                  {[
                    { step: '1', h: '2.0000', q: '4.0000', fq: '16.0000', dy: '12.0000', msec: '6.0000', mtan: '4.0000', err: '+2.0000', pct: '50.0%' },
                    { step: '2', h: '1.0000', q: '3.0000', fq: '9.0000', dy: '5.0000', msec: '5.0000', mtan: '4.0000', err: '+1.0000', pct: '25.0%' },
                    { step: '3', h: '0.5000', q: '2.5000', fq: '6.2500', dy: '2.2500', msec: '4.5000', mtan: '4.0000', err: '+0.5000', pct: '12.5%' },
                    { step: '4', h: '0.1000', q: '2.1000', fq: '4.4100', dy: '0.4100', msec: '4.1000', mtan: '4.0000', err: '+0.1000', pct: '2.5%' },
                    { step: '5', h: '0.0100', q: '2.0100', fq: '4.0401', dy: '0.0401', msec: '4.0100', mtan: '4.0000', err: '+0.0100', pct: '0.25%' },
                    { step: '6', h: '0.0010', q: '2.0010', fq: '4.004001', dy: '0.004001', msec: '4.0010', mtan: '4.0000', err: '+0.0010', pct: '0.025%' },
                  ].map((row) => (
                    <tr key={row.step} className="hover:bg-zinc-900/50">
                      <td className="p-2.5 border border-zinc-700 text-zinc-400">{row.step}</td>
                      <td className="p-2.5 border border-zinc-700 text-white font-bold">{row.h}</td>
                      <td className="p-2.5 border border-zinc-700">{row.q}</td>
                      <td className="p-2.5 border border-zinc-700">{row.fq}</td>
                      <td className="p-2.5 border border-zinc-700">{row.dy}</td>
                      <td className="p-2.5 border border-zinc-700 text-amber-300 font-bold">{row.msec}</td>
                      <td className="p-2.5 border border-zinc-700 text-emerald-300">{row.mtan}</td>
                      <td className="p-2.5 border border-zinc-700 text-rose-300">{row.err}</td>
                      <td className="p-2.5 border border-zinc-700 text-zinc-400">{row.pct}</td>
                    </tr>
                  ))}
                  <tr className="bg-zinc-900 font-bold">
                    <td className="p-2.5 border border-zinc-700 text-white">LIMIT</td>
                    <td className="p-2.5 border border-zinc-700 text-emerald-400">h ➔ 0</td>
                    <td className="p-2.5 border border-zinc-700 text-white">2.0000 (Q ➔ P)</td>
                    <td className="p-2.5 border border-zinc-700 text-white">4.0000</td>
                    <td className="p-2.5 border border-zinc-700 text-white">Δy ➔ 0</td>
                    <td className="p-2.5 border border-zinc-700 text-emerald-400">4.0000</td>
                    <td className="p-2.5 border border-zinc-700 text-emerald-400">4.0000</td>
                    <td className="p-2.5 border border-zinc-700 text-emerald-400">0.0000</td>
                    <td className="p-2.5 border border-zinc-700 text-emerald-400">0.00% (EXACT)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* The 0/0 Singularity Breakdown */}
          <div className="p-5 rounded-xl bg-zinc-900/60 border-2 border-zinc-700 space-y-3">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded text-[10px] font-bold font-mono bg-rose-950/80 text-rose-300 border border-rose-600">
                THE 0/0 PARADOX
              </span>
              <h4 className="font-bold text-white text-base">Why Can&apos;t We Just Substitute h = 0 Directly?</h4>
            </div>
            <p className="text-zinc-300 text-xs leading-relaxed">
              If the tangent is just the slope at an interval of zero, why not simply plug in h = 0?
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="p-4 rounded-lg bg-black border border-rose-500/50 space-y-2 font-mono">
                <div className="text-rose-400 font-bold uppercase text-[10px]">
                  ❌ Direct Substitution (Division by Zero):
                </div>
                <div className="text-zinc-200">
                  m = [ f(2 + 0) - f(2) ] / 0 = [ 4 - 4 ] / 0 = <strong className="text-rose-400">0 / 0</strong>
                </div>
                <p className="text-zinc-400 font-sans text-xs">
                  0/0 is mathematically indeterminate. It gives zero information and causes a fatal crash or NaN in computer software.
                </p>
              </div>

              <div className="p-4 rounded-lg bg-black border border-emerald-500/50 space-y-2 font-mono">
                <div className="text-emerald-400 font-bold uppercase text-[10px]">
                  ✅ Factor Cancellation Before Taking the Limit:
                </div>
                <div className="text-zinc-200">
                  m_sec = h(4 + h) / h <span className="text-emerald-300 font-bold">➔ 4 + h</span> (since h ≠ 0)
                </div>
                <p className="text-zinc-400 font-sans text-xs">
                  Because h approaches 0 without ever equaling 0 during the approach, we can legally divide out h/h = 1. The limit lim[h➔0](4+h) = 4 is completely smooth and exact!
                </p>
              </div>
            </div>
          </div>

          {/* Example 2: Braking Vehicle s(t) = 30t - 5t^2 */}
          <div className="p-5 rounded-xl bg-zinc-900/60 border-2 border-zinc-700 space-y-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded text-[10px] font-bold font-mono bg-sky-950/80 text-sky-300 border border-sky-600">
                  EXAMPLE 2
                </span>
                <h4 className="font-bold text-white text-base">
                  Braking Autonomous Vehicle: s(t) = 30t - 5t² at t₀ = 1.0 s
                </h4>
              </div>
              <span className="text-xs font-mono text-zinc-400">Distance s (meters), Time t (seconds)</span>
            </div>
            <p className="text-zinc-300 text-xs leading-relaxed">
              A self-driving car triggers emergency braking. Its position is governed by s(t) = 30t - 5t². 
              Let us compare what an average sensor window (Secant Line) computes vs what the speedometer radar actually measures (True Tangent Line).
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
              <div className="p-3.5 rounded-lg bg-black border border-zinc-700 space-y-1">
                <div className="text-zinc-400 text-[10px] uppercase font-bold">1. Base Position at t = 1.0 s</div>
                <div className="text-white text-sm font-bold">s(1) = 30(1) - 5(1)² = 25 m</div>
                <div className="text-zinc-400 text-[11px] font-sans">Car has traveled 25 meters.</div>
              </div>

              <div className="p-3.5 rounded-lg bg-black border border-amber-500/50 space-y-1">
                <div className="text-amber-400 text-[10px] uppercase font-bold">2. Secant Slope (h = 1.0 s)</div>
                <div className="text-zinc-300">s(2) = 30(2) - 5(4) = 40 m</div>
                <div className="text-amber-300 text-sm font-bold">v_avg = (40 - 25) / 1 = 15.0 m/s</div>
                <div className="text-zinc-400 text-[11px] font-sans">Average speed over 1 second (54 km/h).</div>
              </div>

              <div className="p-3.5 rounded-lg bg-black border border-emerald-500/50 space-y-1">
                <div className="text-emerald-400 text-[10px] uppercase font-bold">3. True Tangent (Instantaneous)</div>
                <div className="text-zinc-300">s&apos;(t) = 30 - 10t</div>
                <div className="text-emerald-300 text-sm font-bold">v_instant = s&apos;(1) = 20.0 m/s</div>
                <div className="text-zinc-400 text-[11px] font-sans">Actual speed at exact second 1.0 (72 km/h).</div>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-black border border-zinc-700 text-xs text-zinc-300 flex items-center gap-2">
              <span className="text-amber-400 font-bold">Critical Takeaway:</span>
              <span>Because the vehicle is decelerating, the secant line underestimates your speed by <strong>5 m/s (25%)</strong>! If an autonomous emergency braking system relied on secant calculations, it would compute insufficient stopping distance and cause a collision.</span>
            </div>
          </div>
        </section>
      )}

      {/* ════════════════════════════════════════════════════════════════
          TAB 4: SIDE-BY-SIDE 8-POINT COMPARISON MATRIX
          ════════════════════════════════════════════════════════════════ */}
      {activeTab === 'matrix' && (
        <section className="bg-zinc-950 border-2 border-zinc-700 rounded-2xl p-5 sm:p-7 space-y-6">
          <div className="border-b border-zinc-700 pb-3">
            <h3 className="text-xl font-extrabold text-white">
              Side-by-Side 8-Point Architecture Matrix
            </h3>
            <p className="text-zinc-400 text-xs mt-1">
              Direct comparison between Secant Chords and Tangent Gradients across mathematics, physics, and AI.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse border border-zinc-700 rounded-xl overflow-hidden">
              <thead className="bg-zinc-900 text-zinc-200 uppercase text-[10px] font-mono">
                <tr>
                  <th className="p-3.5 border border-zinc-700">Feature Dimension</th>
                  <th className="p-3.5 border border-zinc-700 text-amber-400">✂️ Secant Line</th>
                  <th className="p-3.5 border border-zinc-700 text-emerald-400">✨ True Tangent Line</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800 bg-black text-zinc-300 font-sans">
                <tr className="hover:bg-zinc-900/50">
                  <td className="p-3 border border-zinc-700 font-bold text-white">Etymology & Meaning</td>
                  <td className="p-3 border border-zinc-700 font-mono text-[11px]">Latin <em>secare</em> = &quot;to cut through&quot;</td>
                  <td className="p-3 border border-zinc-700 font-mono text-[11px] text-white">Latin <em>tangere</em> = &quot;to touch gently&quot;</td>
                </tr>
                <tr className="hover:bg-zinc-900/50">
                  <td className="p-3 border border-zinc-700 font-bold text-white">Points Required</td>
                  <td className="p-3 border border-zinc-700">Requires <strong>two separate points</strong> P(x₀, y₀) and Q(x₀+h, y₀+Δy)</td>
                  <td className="p-3 border border-zinc-700 text-white">Requires <strong>only one base point</strong> P(x₀, y₀) plus differentiation rule</td>
                </tr>
                <tr className="hover:bg-zinc-900/50">
                  <td className="p-3 border border-zinc-700 font-bold text-white">Slope Formula</td>
                  <td className="p-3 border border-zinc-700 font-mono text-[11px] text-amber-300">m_sec = [f(x₀+h) - f(x₀)] / h</td>
                  <td className="p-3 border border-zinc-700 font-mono text-[11px] text-emerald-300">m_tan = lim[h➔0] [f(x₀+h) - f(x₀)] / h = f&apos;(x₀)</td>
                </tr>
                <tr className="hover:bg-zinc-900/50">
                  <td className="p-3 border border-zinc-700 font-bold text-white">Point-Slope Equation</td>
                  <td className="p-3 border border-zinc-700 font-mono text-[11px]">y - f(x₀) = m_sec(x - x₀)</td>
                  <td className="p-3 border border-zinc-700 font-mono text-[11px] text-white">y - f(x₀) = f&apos;(x₀)(x - x₀)</td>
                </tr>
                <tr className="hover:bg-zinc-900/50">
                  <td className="p-3 border border-zinc-700 font-bold text-white">Physical Interpretation</td>
                  <td className="p-3 border border-zinc-700">Average velocity over elapsed interval [t₀, t₀+Δt]</td>
                  <td className="p-3 border border-zinc-700 text-white">Instantaneous speedometer reading at exact microsecond t₀</td>
                </tr>
                <tr className="hover:bg-zinc-900/50">
                  <td className="p-3 border border-zinc-700 font-bold text-white">Truncation Error</td>
                  <td className="p-3 border border-zinc-700">Error is O(h) — biased by function curvature</td>
                  <td className="p-3 border border-zinc-700 text-white">Error is <strong>0.000</strong> — exact analytical solution</td>
                </tr>
                <tr className="hover:bg-zinc-900/50">
                  <td className="p-3 border border-zinc-700 font-bold text-white">Role in Machine Learning</td>
                  <td className="p-3 border border-zinc-700">Finite differences (too slow for LLMs, causes oscillations)</td>
                  <td className="p-3 border border-zinc-700 text-white">Standard Backpropagation (Reverse-mode Automatic Differentiation)</td>
                </tr>
                <tr className="hover:bg-zinc-900/50">
                  <td className="p-3 border border-zinc-700 font-bold text-white">Computational Cost for N Parameters</td>
                  <td className="p-3 border border-zinc-700 text-rose-300 font-mono">O(N) forward evaluations (catastrophic for billions)</td>
                  <td className="p-3 border border-zinc-700 text-emerald-300 font-mono">O(1) single backward pass via computational graph</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ════════════════════════════════════════════════════════════════
          TAB 5: WHY MODERN AI CANNOT USE SECANT LINES (THE 70B DISASTER)
          ════════════════════════════════════════════════════════════════ */}
      {activeTab === 'ai' && (
        <section className="bg-zinc-950 border-2 border-zinc-700 rounded-2xl p-5 sm:p-7 space-y-6">
          <div className="border-b border-zinc-700 pb-3">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded text-[10px] font-bold font-mono bg-rose-950/80 text-rose-300 border border-rose-600">
                SYSTEM ARCHITECTURE
              </span>
              <h3 className="text-xl font-extrabold text-white">
                Why Can&apos;t Modern AI Use Secant Lines? The 70-Billion Parameter Disaster
              </h3>
            </div>
            <p className="text-zinc-400 text-xs mt-1">
              Why every modern LLM (GPT-4, Llama 3) uses analytical True Tangents (Backpropagation) instead of numerical Secants (Finite Differences).
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* The O(N) Forward Passes Nightmare */}
            <div className="p-5 rounded-xl bg-zinc-900/60 border-2 border-zinc-700 space-y-3">
              <h4 className="font-bold text-rose-400 text-sm flex items-center gap-2">
                <span>💥 1. The O(N) Computational Wall</span>
              </h4>
              <p className="text-zinc-300 text-xs leading-relaxed">
                If you attempted to compute gradients using secant lines (finite differences), you would have to perturb every single weight knob one by one:
              </p>
              <div className="p-3 rounded-lg bg-black border border-zinc-700 font-mono text-xs text-zinc-300">
                ∂L/∂w_i ≈ [ L(w₁,..., w_i + h,..., w_N) - L(w) ] / h
              </div>
              <p className="text-zinc-400 text-xs leading-relaxed">
                In a modern model like <strong>Llama 3 (70 Billion parameters)</strong>, calculating a single gradient step with secants requires <strong>70 Billion separate forward passes</strong>. On an 8x H100 GPU cluster, one training step would take over <strong>30 years</strong>!
              </p>
            </div>

            {/* Subtractive Cancellation Catastrophe */}
            <div className="p-5 rounded-xl bg-zinc-900/60 border-2 border-zinc-700 space-y-3">
              <h4 className="font-bold text-amber-400 text-sm flex items-center gap-2">
                <span>💣 2. The Numerical Precision Trap</span>
              </h4>
              <p className="text-zinc-300 text-xs leading-relaxed">
                On digital hardware (FP32 or BF16), numerical secants face a fatal dilemma:
              </p>
              <ul className="space-y-2 text-xs text-zinc-400">
                <li className="flex items-start gap-2">
                  <span className="text-rose-400 font-bold">•</span>
                  <span><strong>If h is large (h = 0.1):</strong> The secant slope is corrupted by curvature truncation error, causing gradient descent to overshoot ravines and explode loss.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-rose-400 font-bold">•</span>
                  <span><strong>If h is tiny (h = 10⁻¹⁰):</strong> f(x+h) and f(x) share the same leading bits. Subtracting them causes catastrophic <em>floating-point cancellation</em>, producing pure numerical garbage!</span>
                </li>
              </ul>
            </div>
          </div>

          {/* The Grace of Backpropagation */}
          <div className="p-5 rounded-xl bg-zinc-900/80 border-2 border-emerald-500/50 space-y-3">
            <h4 className="font-bold text-emerald-300 text-base flex items-center gap-2">
              <span>🌟 The Solution: Exact True Tangents via Backpropagation</span>
            </h4>
            <p className="text-zinc-300 text-xs leading-relaxed">
              Backpropagation (Reverse-Mode Automatic Differentiation) does not perturb weights or use secants. Instead, it applies the <strong>Chain Rule of True Tangents</strong> analytically backwards through the computational graph. 
              It computes the exact instantaneous derivatives of all 70 billion weights simultaneously in <strong>one single backward pass</strong> with zero truncation error and zero subtractive cancellation!
            </p>
          </div>
        </section>
      )}

    </div>
  );
}
