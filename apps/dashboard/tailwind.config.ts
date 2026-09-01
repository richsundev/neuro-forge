import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        paper: "#f8fafc",
        accent: "#6366f1",
        good: "#16a34a",
        bad: "#dc2626",
        warn: "#d97706",
      },
    },
  },
  plugins: [],
};

export default config;
