"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("dark");

  // Read initial theme from localStorage or system preference
  useEffect(() => {
    try {
      const saved = (localStorage.getItem("theme") as Theme | null) || null;
      if (saved === "light" || saved === "dark") {
        setThemeState(saved);
        return;
      }
    } catch {}
    // Fallback to system preference
    const prefersDark = typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    setThemeState(prefersDark ? "dark" : "light");
  }, []);

  // Apply theme to <html> tag via class
  useEffect(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    if (theme === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
  }, [theme]);

  return children;
}
