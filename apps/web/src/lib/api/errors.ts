import type { ApiErrorBody } from "./types";

/** An error returned by the API (or the same-origin proxy) in the standard envelope. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;
  readonly suggestedAction: string | null;
  readonly requestId: string | null;
  readonly retryAfterSeconds: number | null;

  constructor(
    status: number,
    body: ApiErrorBody["error"],
    options: { retryAfterSeconds?: number | null } = {},
  ) {
    super(body.message);
    this.name = "ApiError";
    this.status = status;
    this.code = body.code;
    this.retryable = body.retryable;
    this.suggestedAction = body.suggestedAction ?? null;
    this.requestId = body.requestId ?? null;
    this.retryAfterSeconds = options.retryAfterSeconds ?? null;
  }
}

function isErrorBody(value: unknown): value is ApiErrorBody {
  if (!value || typeof value !== "object") return false;
  const error = (value as { error?: unknown }).error;
  if (!error || typeof error !== "object") return false;
  const e = error as Record<string, unknown>;
  return typeof e.code === "string" && typeof e.message === "string";
}

function parseRetryAfter(header: string | null): number | null {
  if (!header) return null;
  const seconds = Number(header);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds;
  const date = Date.parse(header);
  if (Number.isNaN(date)) return null;
  return Math.max(0, Math.round((date - Date.now()) / 1000));
}

function fallbackCode(status: number): string {
  if (status === 401) return "UNAUTHENTICATED";
  if (status === 403) return "FORBIDDEN";
  if (status === 404) return "NOT_FOUND";
  if (status === 409) return "CONFLICT";
  if (status === 413) return "PAYLOAD_TOO_LARGE";
  if (status === 422) return "VALIDATION_FAILED";
  if (status === 428) return "PRECONDITION_REQUIRED";
  if (status === 429) return "RATE_LIMITED";
  if (status >= 500) return "SERVER_ERROR";
  return "UNKNOWN_ERROR";
}

/** Build an ApiError from a non-2xx response, tolerating non-envelope bodies. */
export async function apiErrorFromResponse(response: Response): Promise<ApiError> {
  const retryAfterSeconds = parseRetryAfter(response.headers.get("Retry-After"));
  let parsed: unknown = null;
  try {
    parsed = await response.json();
  } catch {
    parsed = null;
  }
  if (isErrorBody(parsed)) {
    return new ApiError(response.status, parsed.error, { retryAfterSeconds });
  }
  return new ApiError(
    response.status,
    {
      code: fallbackCode(response.status),
      message: `Request failed with status ${response.status}.`,
      retryable: response.status >= 500 || response.status === 429,
      requestId: response.headers.get("X-Request-ID"),
    },
    { retryAfterSeconds },
  );
}

export interface FriendlyError {
  title: string;
  message: string;
  suggestedAction: string | null;
  requestId: string | null;
  retryable: boolean;
}

/**
 * Stable error code -> user-facing copy. The API's own `message` and
 * `suggestedAction` are safe to show and are preferred for capture-specific
 * codes; this table supplies a consistent title and a fallback action.
 */
const ERROR_COPY: Record<string, { title: string; action: string }> = {
  UNAUTHENTICATED: { title: "Please sign in again", action: "Sign in, then retry." },
  FORBIDDEN: {
    title: "You don't have permission",
    action: "Ask the project owner for access.",
  },
  NOT_FOUND: {
    title: "Not found",
    action: "It may have been deleted, or you may not have access.",
  },
  VALIDATION_FAILED: { title: "Some input wasn't accepted", action: "Check the form and try again." },
  VERSION_CONFLICT: {
    title: "This changed since you loaded it",
    action: "Reload to get the latest version, then apply your edit again.",
  },
  ACTIVE_RUN_EXISTS: {
    title: "A reconstruction is already running",
    action: "Wait for it to finish or cancel it before starting another.",
  },
  IDEMPOTENCY_KEY_REUSED: {
    title: "This submission conflicts with an earlier one",
    action: "Reload the page and submit again.",
  },
  PROJECT_DELETING: {
    title: "This project is being deleted",
    action: "No further changes can be made to it.",
  },
  INVALID_STATE: { title: "That action isn't available right now", action: "Reload and try again." },
  PRECONDITION_REQUIRED: { title: "Please reload", action: "Reload the page, then try again." },
  PAYLOAD_TOO_LARGE: { title: "Too large", action: "Use smaller or fewer files." },
  QUOTA_EXCEEDED: {
    title: "Quota exceeded",
    action: "Remove some inputs or delete older projects.",
  },
  RATE_LIMITED: { title: "Too many requests", action: "Wait a moment and try again." },
  UPSTREAM_UNAVAILABLE: {
    title: "The service is unavailable",
    action: "Check your connection and try again in a moment.",
  },
  NETWORK_ERROR: {
    title: "Network problem",
    action: "Check your connection and try again.",
  },
  CROSS_ORIGIN_REJECTED: { title: "Request blocked", action: "Reload the page and try again." },
  INSUFFICIENT_OVERLAP: {
    title: "Not enough overlap between views",
    action: "Record a slower walkthrough with more overlap between frames.",
  },
  DISCONNECTED_CAPTURE: {
    title: "Parts of the capture couldn't be connected",
    action: "Record a slow continuous walk through the doorways between rooms.",
  },
  CAMERA_REGISTRATION_FAILED: {
    title: "Camera positions couldn't be recovered",
    action: "Capture again with good light, slow movement and textured surfaces in view.",
  },
  CAPTURE_TOO_BLURRY: {
    title: "The capture is too blurry",
    action: "Move more slowly and use better light.",
  },
  UPLOAD_INCOMPLETE: {
    title: "The upload isn't finished",
    action: "Resume the upload; missing parts will be sent again.",
  },
  UPLOAD_EXPIRED: {
    title: "The upload expired",
    action: "Remove the file and upload it again.",
  },
  UPLOAD_MODIFIED: {
    title: "The uploaded file changed",
    action: "Remove the file and upload it again.",
  },
  UPLOAD_SIZE_MISMATCH: {
    title: "The uploaded size doesn't match",
    action: "Remove the file and upload it again.",
  },
  STORAGE_UNAVAILABLE: {
    title: "Storage is temporarily unavailable",
    action: "Try again in a moment.",
  },
  AUTH_UNAVAILABLE: {
    title: "Sign-in checks are temporarily unavailable",
    action: "Try again in a moment.",
  },
  SERVER_ERROR: { title: "Something went wrong", action: "Try again in a moment." },
};

export function toFriendlyError(error: unknown): FriendlyError {
  if (error instanceof ApiError) {
    const copy = ERROR_COPY[error.code];
    let action = error.suggestedAction ?? copy?.action ?? null;
    if (error.code === "RATE_LIMITED" && error.retryAfterSeconds != null) {
      action = `Wait about ${Math.max(1, Math.ceil(error.retryAfterSeconds))} seconds and try again.`;
    }
    return {
      title: copy?.title ?? "Something went wrong",
      message: error.message,
      suggestedAction: action,
      requestId: error.requestId,
      retryable: error.retryable,
    };
  }
  if (error instanceof TypeError) {
    const copy = ERROR_COPY.NETWORK_ERROR!;
    return {
      title: copy.title,
      message: "The request could not reach the server.",
      suggestedAction: copy.action,
      requestId: null,
      retryable: true,
    };
  }
  return {
    title: "Something went wrong",
    message: error instanceof Error ? error.message : "An unexpected error occurred.",
    suggestedAction: "Try again in a moment.",
    requestId: null,
    retryable: true,
  };
}

/** Friendly copy for a run/job error object embedded in a resource (not thrown). */
export function friendlyFromErrorBody(body: ApiErrorBody["error"]): FriendlyError {
  return toFriendlyError(new ApiError(0, body));
}
