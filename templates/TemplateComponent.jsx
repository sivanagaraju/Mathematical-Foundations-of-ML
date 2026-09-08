'use client';

import React, { useState, useEffect, useRef } from 'react';

/**
 * TemplateComponent.jsx
 * 
 * Reusable Next.js / React Visualizer Template Component
 * Technology: Next.js (App Router / Pages), React 18/19, Tailwind CSS, JavaScript
 * 
 * Features:
 * - High-contrast black & white dark styling (bg-black, bg-zinc-950)
 * - Prominently visible borders (border-2 border-zinc-700, hover:border-zinc-600)
 * - Luminous colorful canvas graphics (Sky Blue, Amber, Emerald, Rose)
 * - High-DPI canvas auto-scaling
 * - Smooth mathematical limit animations
 */
export default function TemplateComponent({
  title = "Mathematical Concept Visualizer",
  subtitle = "First Principles Interactive Exploration",
  initialX = 1.0,
  initialH = 1.5,
}) {
  const [xVal, setXVal] = useState(initialX);
  const [hVal, setHVal] = useState(initialH);
  const [isAnimating, setIsAnimating] = useState(false);

  const canvasRef = useRef(null);
  const animRef = useRef(null);

  // Parabola f(x) = x^2 and derivative f'(x) = 2x
  const f = (x) => x * x;
  const df = (x) => 2 * x;

  const y0 = f(xVal);
  const y1 = f(xVal + hVal);
  const secantSlope = (y1 - y0) / hVal;
  const tangentSlope = df(xVal);
  const errorGap = Math.abs(secantSlope - tangentSlope);

  // Canvas drawing effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = 320;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const xMin = -2.5, xMax = 4.5, yMin = -1.5, yMax = 12.0;
    const toX = (x) => ((x - xMin) / (xMax - xMin)) * width;
    const toY = (y) => height - ((y - yMin) / (yMax - yMin)) * height;

    // Background
    ctx.fillStyle = '#09090b';
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = '#27272a';
    ctx.lineWidth = 1;
    for (let x = -2; x <= 4; x++) {
      ctx.beginPath();
      ctx.moveTo(toX(x), 0);
      ctx.lineTo(toX(x), height);
      ctx.stroke();
    }

    // Function Curve (Electric Sky Blue)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = xMin + (i / 200) * (xMax - xMin);
      const y = f(x);
      if (i === 0) ctx.moveTo(toX(x), toY(y));
      else ctx.lineTo(toX(x), toY(y));
    }
    ctx.stroke();

    // Secant Line (Amber Dashed)
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(toX(xMin), toY(y0 + secantSlope * (xMin - xVal)));
    ctx.lineTo(toX(xMax), toY(y0 + secantSlope * (xMax - xVal)));
    ctx.stroke();
    ctx.setLineDash([]);

    // Tangent Line (Emerald Solid)
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(toX(xMin), toY(y0 + tangentSlope * (xMin - xVal)));
    ctx.lineTo(toX(xMax), toY(y0 + tangentSlope * (xMax - xVal)));
    ctx.stroke();

    // Points P and Q
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(toX(xVal), toY(y0), 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();

    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.arc(toX(xVal + hVal), toY(y1), 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

  }, [xVal, hVal, y0, y1, secantSlope, tangentSlope]);

  // Smooth limit animation loop
  useEffect(() => {
    if (!isAnimating) return;
    let curr = hVal;
    const loop = () => {
      curr = curr * 0.94;
      if (curr <= 0.015) {
        setHVal(0.01);
        setIsAnimating(false);
      } else {
        setHVal(Number(curr.toFixed(3)));
        animRef.current = requestAnimationFrame(loop);
      }
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [isAnimating]);

  return (
    <div className="w-full max-w-6xl mx-auto p-4 sm:p-6 bg-black text-zinc-100 font-sans space-y-6">
      
      {/* Header Container */}
      <div className="bg-zinc-950 border-2 border-zinc-700 rounded-2xl p-5 sm:p-7 shadow-2xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <span className="px-2.5 py-0.5 text-xs font-mono rounded-full bg-zinc-900 text-zinc-300 border border-zinc-600">
              Next.js + Tailwind CSS Template
            </span>
            <h1 className="text-2xl font-extrabold text-white mt-2">{title}</h1>
            <p className="text-xs text-zinc-400 mt-1">{subtitle}</p>
          </div>
          <div className="p-3 bg-zinc-900 rounded-xl border border-zinc-600 font-mono text-xs">
            <span className="text-zinc-400">Interval: </span>
            <span className="text-amber-400 font-bold">h = {hVal.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* Main Grid: Canvas + Controls + HUD */}
      <div className="bg-zinc-950 border-2 border-zinc-700 rounded-2xl p-5 sm:p-7 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Canvas & Controls (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="rounded-xl border-2 border-zinc-700 overflow-hidden bg-[#09090b]">
            <canvas ref={canvasRef} className="w-full block" style={{ height: '320px' }} />
          </div>

          <div className="grid grid-cols-2 gap-4 p-4 rounded-xl bg-zinc-900/60 border border-zinc-700 text-xs">
            <div>
              <div className="flex justify-between font-mono mb-1">
                <span className="text-zinc-400">Parameter x₀:</span>
                <span className="text-sky-300 font-bold">{xVal.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-2.0"
                max="2.0"
                step="0.05"
                value={xVal}
                onChange={(e) => setXVal(parseFloat(e.target.value))}
                className="w-full accent-white"
              />
            </div>
            <div>
              <div className="flex justify-between font-mono mb-1">
                <span className="text-zinc-400">Window h:</span>
                <span className="text-amber-300 font-bold">{hVal.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="2.5"
                step="0.01"
                value={hVal}
                onChange={(e) => setHVal(parseFloat(e.target.value))}
                className="w-full accent-white"
              />
            </div>
          </div>

          <div className="flex items-center justify-between gap-2">
            <div className="flex gap-2">
              <button onClick={() => setHVal(2.0)} className="px-3 py-1 bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 rounded text-xs font-mono text-zinc-300">h = 2.0</button>
              <button onClick={() => setHVal(0.1)} className="px-3 py-1 bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 rounded text-xs font-mono text-zinc-300">h = 0.1</button>
              <button onClick={() => setHVal(0.01)} className="px-3 py-1 bg-emerald-500 hover:bg-emerald-400 text-black rounded text-xs font-mono font-bold">h ➔ 0.01</button>
            </div>
            <button
              onClick={() => {
                if (hVal <= 0.05) setHVal(2.0);
                setIsAnimating(!isAnimating);
              }}
              className="px-4 py-1.5 bg-white hover:bg-zinc-200 text-black font-bold text-xs rounded-lg transition"
            >
              {isAnimating ? '⏸ Pause' : '▶ Animate Limit'}
            </button>
          </div>
        </div>

        {/* Telemetry HUD (5 Cols) */}
        <div className="lg:col-span-5 space-y-4 flex flex-col justify-between text-xs">
          <div className="p-4 rounded-xl bg-zinc-900/60 border-2 border-zinc-700 space-y-3">
            <h3 className="font-bold text-white text-sm border-b border-zinc-700 pb-2">Live Telemetry</h3>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-zinc-950 rounded-lg border-2 border-amber-500/60 text-center">
                <div className="text-[10px] text-amber-400 font-bold uppercase">✂️ Secant Slope</div>
                <div className="text-xl font-mono font-extrabold text-amber-300 mt-1">{secantSlope.toFixed(2)}</div>
              </div>
              <div className="p-3 bg-zinc-950 rounded-lg border-2 border-emerald-500/60 text-center">
                <div className="text-[10px] text-emerald-400 font-bold uppercase">✨ Tangent Slope</div>
                <div className="text-xl font-mono font-extrabold text-emerald-300 mt-1">{tangentSlope.toFixed(2)}</div>
              </div>
            </div>

            <div className="p-2.5 bg-zinc-950 rounded-lg border-2 border-rose-500/50 flex justify-between font-mono">
              <span className="text-rose-400 font-semibold">Truncation Error:</span>
              <span className="text-rose-300 font-bold">{errorGap.toFixed(2)}</span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-zinc-900/40 border-2 border-zinc-700 text-xs text-zinc-400 leading-relaxed">
            💡 <strong>Template Guide:</strong> This component uses standard Tailwind CSS classes with visible zinc-700/600 borders. Import it into any Next.js page or React component tree.
          </div>
        </div>

      </div>

    </div>
  );
}
