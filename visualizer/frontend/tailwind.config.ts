import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Robotics dark theme palette
        surface: {
          0: '#0a0a0f',    // deepest background
          1: '#111118',    // main background
          2: '#1a1a24',    // panel background
          3: '#232330',    // elevated surface
          4: '#2d2d3d',    // hover state
        },
        accent: {
          blue: '#3b82f6',
          cyan: '#06b6d4',
          green: '#22c55e',
          yellow: '#eab308',
          orange: '#f97316',
          red: '#ef4444',
          purple: '#a855f7',
        },
        border: {
          DEFAULT: '#2d2d3d',
          hover: '#3d3d50',
          active: '#3b82f6',
        },
        text: {
          primary: '#e4e4ef',
          secondary: '#9494ad',
          muted: '#5e5e76',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        'xxs': ['0.65rem', { lineHeight: '0.85rem' }],
      },
    },
  },
  plugins: [],
} satisfies Config
