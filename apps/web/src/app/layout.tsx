import type { Metadata, Viewport } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Andarilho", template: "%s · Andarilho" },
  description: "Capture a place. Step inside.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2 focus:text-stone-900"
        >
          Skip to content
        </a>
        <header className="border-b border-stone-200 bg-white/80 dark:border-stone-800 dark:bg-stone-900/80">
          <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
            <Link href="/" className="text-lg font-semibold tracking-tight">
              Andarilho
            </Link>
            <nav aria-label="Main">
              <Link href="/projects" className="text-sm font-medium hover:underline">
                Projects
              </Link>
            </nav>
          </div>
        </header>
        <main id="main" className="mx-auto max-w-5xl px-4 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
