"use client";

import { useState } from "react";
import { Button, ErrorNotice } from "@/components/ui";
import { api } from "@/lib/api/client";
import { ApiError, toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { RunAdvice as Advice } from "@/lib/api/types";

/**
 * After an error, keep offering the request only if trying again could help:
 * not when AI suggestions are simply not set up for this installation.
 */
export function keepOffering(error: unknown): boolean {
  return !(error instanceof ApiError && error.code === "AI_UNAVAILABLE" && !error.retryable);
}

/**
 * Optional AI explanation of a failed run (README section 16). The request is
 * explicit and says what is sent; the advice is labeled as a suggestion and
 * never replaces the run's own guidance shown above it.
 */
export function RunAdvice({ runId }: { runId: string }) {
  const [advice, setAdvice] = useState<Advice | null>(null);
  const [error, setError] = useState<FriendlyError | null>(null);
  const [offered, setOffered] = useState(true);
  const [busy, setBusy] = useState(false);

  async function ask() {
    setBusy(true);
    setError(null);
    try {
      const { data } = await api.getRunAdvice(runId);
      setAdvice(data);
    } catch (err) {
      setError(toFriendlyError(err));
      setOffered(keepOffering(err));
    } finally {
      setBusy(false);
    }
  }

  if (advice) {
    return (
      <section
        aria-label="AI suggestions"
        className="mt-3 rounded-md border border-stone-300 p-3 text-sm dark:border-stone-700"
      >
        <p className="font-semibold">AI suggestions</p>
        <p className="mt-1">{advice.advice.summary}</p>
        <ol className="mt-2 list-decimal space-y-1 pl-5">
          {advice.advice.steps.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
        <p className="mt-2 text-xs text-stone-600 dark:text-stone-400">
          Suggested by {advice.model} through {advice.provider}; it can be wrong. Sent:{" "}
          {advice.sent}
        </p>
      </section>
    );
  }
  return (
    <div className="mt-3 space-y-2">
      {offered ? (
        <div>
          <Button variant="secondary" onClick={() => void ask()} disabled={busy}>
            {busy ? "Asking..." : "Ask AI to explain this failure"}
          </Button>
          <p className="mt-1 text-xs text-stone-600 dark:text-stone-400">
            Sends the error code, its message and capture statistics (no images or files) to an
            external AI provider.
          </p>
        </div>
      ) : null}
      <ErrorNotice error={error} />
    </div>
  );
}
