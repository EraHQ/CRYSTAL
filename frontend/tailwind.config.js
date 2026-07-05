/** @type {import('tailwindcss').Config} */
// DARK THEME REMAP (2026-06-12 redesign).
// The pages were written against a light semantic gray scale (white =
// card surface, gray-50 = inset, gray-900 = primary text). Rather than
// rewriting hundreds of class names, the palette itself is remapped
// here — one merge point reaches every page. Shades 50–200 become dark
// tints (backgrounds/borders), 600–800 become light readable variants
// (they're used as TEXT colors), and 300–500 stay vivid (used for
// icons, dots, progress bars).
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // "white" is the card surface in this app.
        white: "#151823",
        gray: {
          50: "#1b1f2d",   // insets, table headers, hover rows
          100: "#222738",  // subtle borders, badge backgrounds
          200: "#2b3146",  // standard borders
          300: "#3d4459",  // strong borders, dim placeholders, idle icons
          400: "#8a92aa",  // muted text
          500: "#9aa2ba",  // secondary text
          600: "#b4bcd2",  // body text
          700: "#cdd4e6",  // emphasized text
          800: "#e2e7f4",  // headings
          900: "#f3f5fc",  // primary text
        },
        brand: {
          50: "#1c1f38",
          100: "#252a52",
          200: "#313868",
          300: "#8d92ff",
          400: "#9da1ff",
          500: "#8487fb",
          600: "#6f72f7",  // primary button bg / accent text
          700: "#a9afff",  // light variant (used as text)
          800: "#c8ccff",
          900: "#e3e5ff",
        },
        emerald: { 50: "#0f231d", 100: "#143527", 200: "#1d4d38", 300: "#34d399", 400: "#34d399", 500: "#10b981", 600: "#4ade9d", 700: "#6ee7b7", 800: "#a7f3d0" },
        green:   { 50: "#0f231d", 100: "#143527", 200: "#1d4d38", 300: "#4ade80", 400: "#4ade80", 500: "#22c55e", 600: "#5ce592", 700: "#86efac", 800: "#bbf7d0" },
        amber:   { 50: "#271d0c", 100: "#3a2c11", 200: "#54401a", 300: "#fcd34d", 400: "#fbbf24", 500: "#f59e0b", 600: "#fbbf24", 700: "#fcd34d", 800: "#fde68a" },
        yellow:  { 50: "#271d0c", 100: "#3a2c11", 200: "#54401a", 300: "#fde047", 400: "#facc15", 500: "#eab308", 600: "#facc15", 700: "#fde047", 800: "#fef08a" },
        red:     { 50: "#2a1417", 100: "#3d1c20", 200: "#58272e", 300: "#fda4af", 400: "#fb7185", 500: "#f43f5e", 600: "#fb7185", 700: "#fda4af", 800: "#fecdd3" },
        blue:    { 50: "#131c30", 100: "#1a2745", 200: "#243460", 300: "#7dafff", 400: "#60a5fa", 500: "#3b82f6", 600: "#7dafff", 700: "#a8c7ff", 800: "#cfe0ff" },
        purple:  { 50: "#1f1733", 100: "#2c2049", 200: "#3d2d66", 300: "#c4b5fd", 400: "#a78bfa", 500: "#8b5cf6", 600: "#b39bfc", 700: "#c4b5fd", 800: "#ddd6fe" },
        violet:  { 50: "#1f1733", 100: "#2c2049", 200: "#3d2d66", 300: "#c4b5fd", 400: "#a78bfa", 500: "#8b5cf6", 600: "#b39bfc", 700: "#c4b5fd", 800: "#ddd6fe" },
        orange:  { 50: "#291810", 100: "#3c2316", 200: "#573220", 300: "#fdba74", 400: "#fb923c", 500: "#f97316", 600: "#fb923c", 700: "#fdba74", 800: "#fed7aa" },
        rose:    { 50: "#2a1419", 100: "#3d1c24", 200: "#582734", 300: "#fda4af", 400: "#fb7185", 500: "#f43f5e", 600: "#fb7185", 700: "#fda4af", 800: "#fecdd3" },
        cyan:    { 50: "#0e2229", 100: "#13313c", 200: "#1b4757", 300: "#67e8f9", 400: "#22d3ee", 500: "#06b6d4", 600: "#4adef2", 700: "#67e8f9", 800: "#a5f3fc" },
        facet: {
          rose: "#fb7185",
          amber: "#fbbf24",
          emerald: "#34d399",
          cyan: "#22d3ee",
          violet: "#a78bfa",
          blue: "#60a5fa",
        },
        match: {
          high: "#34d399",
          medium: "#fbbf24",
          low: "#fb7185",
          none: "#8a92aa",
        },
        // True neutrals kept light — used where text must stay bright
        // on brand/gradient backgrounds (zinc is untouched by the remap).
      },
      fontFamily: {
        sans: ['"Inter"', "ui-sans-serif", "system-ui", "-apple-system", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px 0 rgb(0 0 0 / 0.4), 0 0 0 1px rgb(255 255 255 / 0.02)",
        "card-hover": "0 8px 24px -6px rgb(0 0 0 / 0.5), 0 0 0 1px rgb(255 255 255 / 0.04)",
        facet: "0 0 0 1px rgb(132 135 251 / 0.18), 0 4px 16px -4px rgb(132 135 251 / 0.12)",
        glow: "0 0 24px -4px rgb(132 135 251 / 0.35)",
      },
      backgroundImage: {
        "prism-subtle":
          "linear-gradient(135deg, rgba(132,135,251,0.06) 0%, rgba(34,211,238,0.04) 50%, rgba(167,139,250,0.06) 100%)",
        "prism-accent": "linear-gradient(135deg, #8487fb, #a78bfa, #22d3ee)",
        "user-bubble": "linear-gradient(135deg, #5d60ee, #7158e2)",
      },
      animation: {
        shimmer: "shimmer 2s ease-in-out infinite",
        "fade-up": "fadeUp 0.3s ease-out both",
        "dot-bounce": "dotBounce 1.2s ease-in-out infinite",
      },
      keyframes: {
        shimmer: { "0%, 100%": { opacity: "0.5" }, "50%": { opacity: "1" } },
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        dotBounce: {
          "0%, 60%, 100%": { transform: "translateY(0)", opacity: "0.4" },
          "30%": { transform: "translateY(-4px)", opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};
