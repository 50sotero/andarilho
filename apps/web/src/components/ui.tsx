import type { ButtonHTMLAttributes, ReactNode } from "react";
import type { FriendlyError } from "@/lib/api/errors";

type Variant = "primary" | "secondary" | "danger" | "ghost";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-teal-700 text-white hover:bg-teal-800 disabled:bg-stone-400 dark:disabled:bg-stone-700",
  secondary:
    "border border-stone-300 bg-white text-stone-900 hover:bg-stone-100 disabled:text-stone-400 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-100 dark:hover:bg-stone-800",
  danger: "bg-red-700 text-white hover:bg-red-800 disabled:bg-stone-400",
  ghost: "text-stone-700 hover:bg-stone-100 dark:text-stone-200 dark:hover:bg-stone-800",
};

export function Button({
  variant = "primary",
  className = "",
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      type={type}
      className={`inline-flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed ${VARIANTS[variant]} ${className}`}
      {...props}
    />
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <section
      className={`rounded-lg border border-stone-200 bg-white p-5 shadow-sm dark:border-stone-800 dark:bg-stone-900 ${className}`}
    >
      {children}
    </section>
  );
}

/** Error with title, message and suggested action. Announced assertively. */
export function ErrorNotice({
  error,
  onRetry,
}: {
  error: FriendlyError | null;
  onRetry?: () => void;
}) {
  return (
    <div role="alert" aria-live="assertive">
      {error ? (
        <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100">
          <p className="font-semibold">{error.title}</p>
          <p>{error.message}</p>
          {error.suggestedAction ? <p className="mt-1">{error.suggestedAction}</p> : null}
          {error.requestId ? (
            <p className="mt-1 text-xs opacity-75">Reference: {error.requestId}</p>
          ) : null}
          {onRetry && error.retryable ? (
            <Button variant="secondary" className="mt-2" onClick={onRetry}>
              Try again
            </Button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export function StatusBadge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad";
}) {
  const tones = {
    neutral: "bg-stone-100 text-stone-800 dark:bg-stone-800 dark:text-stone-200",
    good: "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
    warn: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200",
    bad: "bg-red-100 text-red-900 dark:bg-red-950 dark:text-red-200",
  } as const;
  return (
    <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

export const inputClass =
  "block w-full rounded-md border border-stone-300 bg-white px-3 py-2 text-sm dark:border-stone-700 dark:bg-stone-900";

export const linkButtonClass =
  "inline-flex items-center justify-center rounded-md border border-stone-300 bg-white px-3 py-2 text-sm font-medium hover:bg-stone-100 dark:border-stone-700 dark:bg-stone-900 dark:hover:bg-stone-800";

export const primaryLinkClass =
  "inline-flex items-center justify-center rounded-md bg-teal-700 px-3 py-2 text-sm font-medium text-white hover:bg-teal-800";
