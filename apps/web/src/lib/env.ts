/**
 * Environment access. Values are read lazily (at request time) so that
 * `next build` succeeds without real credentials.
 *
 * NEXT_PUBLIC_* values are inlined into the browser bundle; they must never
 * hold secrets. API_INTERNAL_URL is server-only.
 */

function required(name: string, value: string | undefined): string {
  if (!value) {
    throw new Error(`Missing required environment variable ${name}.`);
  }
  return value;
}

export function getPublicSupabaseConfig(): { url: string; publishableKey: string } {
  // Must be referenced literally so Next.js can inline them in client bundles.
  return {
    url: required("NEXT_PUBLIC_SUPABASE_URL", process.env.NEXT_PUBLIC_SUPABASE_URL),
    publishableKey: required(
      "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY",
      process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY,
    ),
  };
}

/** The canonical public origin of the web app, e.g. `https://andarilho.example`. */
export function getAppOrigin(): string {
  const raw = required("NEXT_PUBLIC_APP_URL", process.env.NEXT_PUBLIC_APP_URL);
  return new URL(raw).origin;
}

/** Server-only: base URL of the FastAPI service. */
export function getApiInternalUrl(): string {
  return required("API_INTERNAL_URL", process.env.API_INTERNAL_URL).replace(/\/+$/, "");
}

/**
 * Server-only: the API's AI provider timeout (AI_TIMEOUT_SECONDS, the same setting
 * the API reads), so the proxy waits long enough for AI advice. Defaults to the
 * API's default; out-of-range values fall back to it as the API would reject them.
 */
export function getAiTimeoutSeconds(): number {
  const value = Number(process.env.AI_TIMEOUT_SECONDS);
  return Number.isFinite(value) && value > 0 && value <= 300 ? value : 60;
}
