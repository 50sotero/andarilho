"use client";

import { useRouter } from "next/navigation";
import { useId, useRef, useState, type FormEvent } from "react";
import { Button, inputClass } from "@/components/ui";
import { getSupabaseBrowserClient } from "@/lib/auth/browser";

type Step = "email" | "code";

export function LoginForm({ next, callbackError }: { next: string; callbackError: boolean }) {
  const router = useRouter();
  const emailId = useId();
  const codeId = useId();
  const codeRef = useRef<HTMLInputElement>(null);
  const [step, setStep] = useState<Step>("email");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(
    callbackError ? "That sign-in link is invalid or has expired. Request a new one." : null,
  );

  function appOrigin(): string {
    return process.env.NEXT_PUBLIC_APP_URL
      ? new URL(process.env.NEXT_PUBLIC_APP_URL).origin
      : window.location.origin;
  }

  async function sendLink(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const supabase = getSupabaseBrowserClient();
      const redirect = new URL("/auth/callback", appOrigin());
      redirect.searchParams.set("next", next);
      const { error: otpError } = await supabase.auth.signInWithOtp({
        email: email.trim(),
        options: { emailRedirectTo: redirect.toString() },
      });
      if (otpError) {
        setError("We couldn't send the email. Check the address and try again in a moment.");
        return;
      }
      setStep("code");
      setStatus(
        "Check your email. Open the sign-in link on this device, or enter the code from the email below.",
      );
      requestAnimationFrame(() => codeRef.current?.focus());
    } catch {
      setError("Sign-in is unavailable right now. Try again in a moment.");
    } finally {
      setBusy(false);
    }
  }

  async function verifyCode(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const supabase = getSupabaseBrowserClient();
      const { error: verifyError } = await supabase.auth.verifyOtp({
        email: email.trim(),
        token: code.trim(),
        type: "email",
      });
      if (verifyError) {
        setError("That code is invalid or has expired.");
        return;
      }
      router.replace(next);
      router.refresh();
    } catch {
      setError("Sign-in is unavailable right now. Try again in a moment.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <div aria-live="polite" className="text-sm text-stone-700 dark:text-stone-300">
        {status}
      </div>
      <div role="alert" className="text-sm text-red-700 dark:text-red-300">
        {error}
      </div>

      {step === "email" ? (
        <form onSubmit={sendLink} className="space-y-3">
          <div className="space-y-1">
            <label htmlFor={emailId} className="block text-sm font-medium">
              Email address
            </label>
            <input
              id={emailId}
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className={inputClass}
            />
          </div>
          <Button type="submit" disabled={busy || !email}>
            {busy ? "Sending..." : "Email me a sign-in link"}
          </Button>
        </form>
      ) : (
        <form onSubmit={verifyCode} className="space-y-3">
          <div className="space-y-1">
            <label htmlFor={codeId} className="block text-sm font-medium">
              Code from the email
            </label>
            <input
              id={codeId}
              ref={codeRef}
              inputMode="numeric"
              autoComplete="one-time-code"
              required
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className={inputClass}
            />
          </div>
          <div className="flex gap-2">
            <Button type="submit" disabled={busy || !code}>
              {busy ? "Checking..." : "Sign in"}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setStep("email");
                setCode("");
                setStatus(null);
              }}
            >
              Use a different email
            </Button>
          </div>
        </form>
      )}
    </div>
  );
}
