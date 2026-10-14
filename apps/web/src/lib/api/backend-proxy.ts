/**
 * Same-origin allowlisted proxy from the browser to the FastAPI service.
 *
 * - Only `/v1/<allowlisted resource>/...` paths are forwarded; anything else is 404.
 * - The bearer token comes from the server-side Supabase session, never from the client.
 * - Cookie-authenticated mutations must carry an Origin equal to NEXT_PUBLIC_APP_URL.
 * - Only a small set of request/response headers is forwarded.
 * - Responses are never cacheable by shared caches.
 *
 * Kept free of Next.js imports so it can be unit-tested with fake dependencies.
 */

export const MAX_PROXY_BODY_BYTES = 1024 * 1024; // JSON only; media goes straight to storage.
export const UPSTREAM_TIMEOUT_MS = 30_000;

const ALLOWED_RESOURCES = new Set(["projects", "uploads", "jobs", "runs", "scenes", "shares"]);
const PUBLIC_PATHS = new Set(["v1/public/scenes/delivery"]);
const SAFE_METHODS = new Set(["GET", "HEAD"]);

export const FORWARDED_REQUEST_HEADERS = [
  "content-type",
  "if-match",
  "idempotency-key",
  "x-request-id",
] as const;

export const FORWARDED_RESPONSE_HEADERS = [
  "etag",
  "retry-after",
  "x-request-id",
  "content-type",
] as const;

const SEGMENT_PATTERN = /^[A-Za-z0-9._~-]+$/;

export interface ValidatedPath {
  /** Path joined with "/" without a leading slash, e.g. `v1/projects/abc`. */
  path: string;
  isPublic: boolean;
}

/** Returns the validated path, or null when it is not allowlisted. */
export function validateProxyPath(segments: readonly string[]): ValidatedPath | null {
  if (segments.length < 2 || segments.length > 8) return null;
  for (const segment of segments) {
    if (!segment || segment === "." || segment === ".." || !SEGMENT_PATTERN.test(segment)) {
      return null;
    }
  }
  if (segments[0] !== "v1") return null;
  const path = segments.join("/");
  if (PUBLIC_PATHS.has(path)) return { path, isPublic: true };
  if (!ALLOWED_RESOURCES.has(segments[1]!)) return null;
  return { path, isPublic: false };
}

/** Mutations must come from our own origin (CSRF defence for cookie authentication). */
export function isAllowedOrigin(method: string, origin: string | null, appOrigin: string): boolean {
  if (SAFE_METHODS.has(method)) return true;
  if (!origin) return false;
  try {
    return new URL(origin).origin === appOrigin;
  } catch {
    return false;
  }
}

export function errorResponse(
  status: number,
  code: string,
  message: string,
  options: { retryable?: boolean; suggestedAction?: string; requestId?: string | null } = {},
): Response {
  return new Response(
    JSON.stringify({
      error: {
        code,
        message,
        retryable: options.retryable ?? false,
        suggestedAction: options.suggestedAction ?? null,
        requestId: options.requestId ?? null,
      },
    }),
    {
      status,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "private, no-store",
      },
    },
  );
}

async function readBoundedBody(
  request: Request,
  limit: number,
): Promise<ArrayBuffer | "too_large"> {
  const declared = request.headers.get("content-length");
  if (declared && Number(declared) > limit) return "too_large";
  if (!request.body) return new ArrayBuffer(0);
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > limit) {
      await reader.cancel().catch(() => undefined);
      return "too_large";
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

export interface BackendProxyDeps {
  getAccessToken: () => Promise<string | null>;
  getAppOrigin: () => string;
  getApiBaseUrl: () => string;
  fetch: (input: string, init: RequestInit) => Promise<Response>;
  maxBodyBytes?: number;
  timeoutMs?: number;
}

export function createBackendProxyHandler(deps: BackendProxyDeps) {
  return async function handle(request: Request, segments: readonly string[]): Promise<Response> {
    const method = request.method.toUpperCase();
    const requestId = request.headers.get("x-request-id");
    const validated = validateProxyPath(segments);
    if (!validated) {
      return errorResponse(404, "NOT_FOUND", "Not found.", { requestId });
    }

    let appOrigin: string;
    let apiBase: string;
    try {
      appOrigin = deps.getAppOrigin();
      apiBase = deps.getApiBaseUrl();
    } catch {
      return errorResponse(500, "SERVER_MISCONFIGURED", "The server is not configured.", {
        requestId,
      });
    }

    if (!isAllowedOrigin(method, request.headers.get("origin"), appOrigin)) {
      return errorResponse(403, "CROSS_ORIGIN_REJECTED", "Cross-origin request rejected.", {
        requestId,
      });
    }

    const headers = new Headers({ Accept: "application/json" });
    for (const name of FORWARDED_REQUEST_HEADERS) {
      const value = request.headers.get(name);
      if (value) headers.set(name, value);
    }

    if (!validated.isPublic) {
      const token = await deps.getAccessToken();
      if (!token) {
        return errorResponse(401, "UNAUTHENTICATED", "You are not signed in.", {
          suggestedAction: "Sign in and try again.",
          requestId,
        });
      }
      headers.set("Authorization", `Bearer ${token}`);
    }

    let body: ArrayBuffer | undefined;
    if (!SAFE_METHODS.has(method)) {
      const read = await readBoundedBody(request, deps.maxBodyBytes ?? MAX_PROXY_BODY_BYTES);
      if (read === "too_large") {
        return errorResponse(413, "PAYLOAD_TOO_LARGE", "The request body is too large.", {
          requestId,
        });
      }
      body = read.byteLength > 0 ? read : undefined;
    }

    const search = new URL(request.url).search;
    const upstreamUrl = `${apiBase}/${validated.path}${search}`;

    let upstream: Response;
    try {
      upstream = await deps.fetch(upstreamUrl, {
        method,
        headers,
        body,
        redirect: "manual",
        cache: "no-store",
        signal: AbortSignal.timeout(deps.timeoutMs ?? UPSTREAM_TIMEOUT_MS),
      });
    } catch {
      return errorResponse(
        502,
        "UPSTREAM_UNAVAILABLE",
        "The Andarilho service could not be reached.",
        {
          retryable: true,
          suggestedAction: "Check your connection and try again in a moment.",
          requestId,
        },
      );
    }

    const responseHeaders = new Headers({ "Cache-Control": "private, no-store" });
    for (const name of FORWARDED_RESPONSE_HEADERS) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }

    // The API never redirects the browser; do not leak upstream Location headers.
    if (upstream.status >= 300 && upstream.status < 400) {
      return errorResponse(502, "UPSTREAM_UNAVAILABLE", "Unexpected upstream response.", {
        retryable: true,
        requestId: upstream.headers.get("x-request-id") ?? requestId,
      });
    }

    const noBody = upstream.status === 204 || upstream.status === 304 || method === "HEAD";
    return new Response(noBody ? null : upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  };
}
