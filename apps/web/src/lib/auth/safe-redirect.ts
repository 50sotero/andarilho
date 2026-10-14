/**
 * Returns a same-origin relative path for post-login redirects, or the fallback.
 * Rejects absolute URLs, protocol-relative URLs (`//evil`), backslashes and
 * control characters so the `next` parameter cannot cause an open redirect.
 */
export function safeNextPath(value: string | null | undefined, fallback = "/projects"): string {
  if (!value || typeof value !== "string") return fallback;
  if (!value.startsWith("/") || value.startsWith("//")) return fallback;
  if (/[\\\u0000-\u001f\u007f]/.test(value)) return fallback;
  try {
    const parsed = new URL(value, "http://localhost");
    if (parsed.origin !== "http://localhost") return fallback;
    return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  } catch {
    return fallback;
  }
}
