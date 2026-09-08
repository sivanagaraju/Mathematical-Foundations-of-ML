/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './MathsTerms/**/*.{js,ts,jsx,tsx,html}',
    './templates/**/*.{js,ts,jsx,tsx,html}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        zinc: {
          950: '#09090b',
          900: '#18181b',
          850: '#202023',
          800: '#27272a',
          750: '#333338',
          700: '#3f3f46',
          600: '#52525b',
          500: '#71717a',
          400: '#a1a1aa',
          300: '#d4d4d8',
          200: '#e4e4e7',
          100: '#f4f4f5',
          50: '#fafafa'
        },
        ai: {
          curve: '#38bdf8',      // Sky Blue
          secant: '#f59e0b',     // Amber
          tangent: '#10b981',    // Emerald Green
          error: '#f43f5e',      // Rose
          gradient: '#ec4899',   // Pink
          contour: '#a855f7',    // Violet
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Menlo', 'monospace'],
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        'glow-secant': '0 0 25px -5px rgba(245, 158, 11, 0.3)',
        'glow-tangent': '0 0 25px -5px rgba(16, 185, 129, 0.3)',
        'glow-card': '0 10px 30px -10px rgba(0, 0, 0, 0.9)',
      }
    },
  },
  plugins: [],
};
