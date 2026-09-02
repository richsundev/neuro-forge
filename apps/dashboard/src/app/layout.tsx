import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroForge",
  description: "Autonomous LLM System Evolution & Experimentation Engine",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen">
          <header className="border-b border-slate-200 bg-white">
            <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-4">
              <Link href="/" className="text-lg font-semibold text-ink">
                NeuroForge
              </Link>
              <nav className="flex gap-4 text-sm text-slate-600">
                <Link href="/" className="hover:text-accent">
                  Overview
                </Link>
                <Link href="/datasets" className="hover:text-accent">
                  Challenge Evolution
                </Link>
                <Link href="/promotions" className="hover:text-accent">
                  Promotion
                </Link>
              </nav>
              <span className="ml-auto text-xs text-slate-400">
                autonomous experimentation, not observability
              </span>
            </div>
          </header>
          <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
