import type { Metadata } from "next";
import { Suspense } from "react";
import Link from "next/link";
import "./globals.css";
import { ThemeProvider } from "@/app/components/ThemeProvider";

export const metadata: Metadata = { title: "Pipeline | Contextual unit design space", description: "Construct and interactively explore contextual-unit organizations for tutorial system design." };

export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return <html lang="en"><body className="antialiased"><ThemeProvider>
    <nav aria-label="Main navigation" className="flex flex-wrap gap-5 px-6 py-3 text-sm border-b border-white/10 bg-neutral-950 text-neutral-200">
      <Link href="/">Pipeline</Link><Link href="/space">Explore design space</Link>
    </nav>
    <Suspense fallback={<p className="p-6">Loading interface…</p>}>{children}</Suspense>
  </ThemeProvider></body></html>;
}
