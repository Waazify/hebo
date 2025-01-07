/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./templates/**/*.py",  // For any Python files that might generate HTML
    "./static/src/**/*.js",
    "./core/templates/**/*.html",
    "./knowledge/templates/**/*.html",
    "**/templates/**/*.html",  // Catch all template directories
  ],
  theme: {
    extend: {
      colors: {
        primary: "#FCD34D",    // Bumblebee Yellow
        secondary: "#F3F4F6",  // Muted Gray
        textgray: "#6B7280",   // Steel Gray
        accent: "#3B82F6",     // Muted Blue
        error: "#EF4444",      // Soft Red
        success: "#10B981",    // Soft Green
      },
      fontFamily: {
        sans: ['Inter', 'Helvetica', 'Arial', 'sans-serif'],
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
    require("daisyui"),
    require('@tailwindcss/typography'),
  ],
  daisyui: {
    themes: [
      {
        bumblebee: {
          ...require("daisyui/src/theming/themes")["[data-theme=bumblebee]"],
          "primary": "#FCD34D",
          "primary-content": "#1F2937",
          "base-100": "#FFFFFF",
          "base-200": "#F3F4F6",
          "base-300": "#E5E7EB",
        },
      },
    ],
  },
} 

