/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg:      '#0d1117',
        card:    '#161b22',
        border:  '#30363d',
        muted:   '#8b949e',
        text:    '#c9d1d9',
        accent:  '#58a6ff',
        green:   '#3fb950',
        red:     '#f85149',
        orange:  '#d29922',
        purple:  '#bc8cff',
      },
      fontFamily: { mono: ['JetBrains Mono', 'Fira Code', 'monospace'] },
    },
  },
  plugins: [],
}
