import type { Metadata } from "next";
import { Suspense } from "react";
import Link from "next/link";
import "./globals.css";
import { ThemeProvider } from "@/app/components/ThemeProvider";

export const metadata: Metadata = { title: "CIM Pipeline Explorer", description: "Explore Context Information Maps from tutorial pipelines." };

export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return <html lang="en"><body className="antialiased"><ThemeProvider>
    <nav aria-label="Main navigation" className="flex flex-wrap gap-5 px-6 py-3 text-sm border-b border-white/10 bg-neutral-950 text-neutral-200">
      <Link href="/">CIM Pipeline Explorer</Link><Link href="/space">Explore design space</Link>
    </nav>
    <Suspense fallback={<p className="p-6">Loading interface…</p>}>{children}</Suspense>
  </ThemeProvider></body></html>;
}
