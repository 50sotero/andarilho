/**
 * Authenticated status polling (README section 12): every 3 s while the page
 * is visible, back off while hidden or after errors, pause while offline,
 * and stop on terminal states.
 */

export const POLL_VISIBLE_MS = 3_000;
export const POLL_HIDDEN_BASE_MS = 15_000;
export const POLL_MAX_MS = 60_000;

export interface PollConditions {
  visible: boolean;
  online: boolean;
  /** Consecutive polls performed while hidden. */
  hiddenPolls: number;
  /** Consecutive failed polls. */
  failures: number;
  /** Server-provided Retry-After (seconds), if the last failure had one. */
  retryAfterSeconds?: number | null;
}

/** Milliseconds until the next poll, or null to wait for the browser to come back online. */
export function nextPollDelay(c: PollConditions): number | null {
  if (!c.online) return null;
  let delay = c.visible
    ? POLL_VISIBLE_MS
    : Math.min(POLL_MAX_MS, POLL_HIDDEN_BASE_MS * 2 ** Math.max(0, c.hiddenPolls));
  if (c.failures > 0) {
    delay = Math.max(delay, Math.min(POLL_MAX_MS, POLL_VISIBLE_MS * 2 ** c.failures));
  }
  if (c.retryAfterSeconds != null && c.retryAfterSeconds > 0) {
    delay = Math.max(delay, Math.min(POLL_MAX_MS * 5, c.retryAfterSeconds * 1000));
  }
  return Math.min(delay, POLL_MAX_MS * 5);
}

export interface PollEnvironment {
  isVisible(): boolean;
  isOnline(): boolean;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
  /** Subscribe to visibility/online changes; returns an unsubscribe function. */
  onChange(fn: () => void): () => void;
}

export function browserPollEnvironment(): PollEnvironment {
  return {
    isVisible: () => typeof document === "undefined" || document.visibilityState !== "hidden",
    isOnline: () => typeof navigator === "undefined" || navigator.onLine !== false,
    setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms),
    clearTimeout: (handle) => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
    onChange: (fn) => {
      document.addEventListener("visibilitychange", fn);
      window.addEventListener("online", fn);
      window.addEventListener("offline", fn);
      return () => {
        document.removeEventListener("visibilitychange", fn);
        window.removeEventListener("online", fn);
        window.removeEventListener("offline", fn);
      };
    },
  };
}

export interface PollResult {
  /** True when the watched resource reached a terminal state: polling stops. */
  terminal: boolean;
}

export interface Poller {
  stop(): void;
  /** Poll as soon as possible (e.g. after a user action). */
  pollNow(): void;
}

/**
 * Starts polling immediately. `task` errors are counted as failures and back off;
 * they never stop polling (only `terminal: true` or `stop()` do).
 */
export function startPolling(
  task: (signal: AbortSignal) => Promise<PollResult>,
  env: PollEnvironment,
  options: { onError?: (error: unknown) => void } = {},
): Poller {
  let stopped = false;
  let running = false;
  let timer: unknown = null;
  let hiddenPolls = 0;
  let failures = 0;
  let retryAfterSeconds: number | null = null;
  let controller: AbortController | null = null;

  const clear = () => {
    if (timer !== null) {
      env.clearTimeout(timer);
      timer = null;
    }
  };

  const schedule = () => {
    clear();
    if (stopped || running) return;
    const delay = nextPollDelay({
      visible: env.isVisible(),
      online: env.isOnline(),
      hiddenPolls,
      failures,
      retryAfterSeconds,
    });
    if (delay === null) return; // wait for an online event
    timer = env.setTimeout(run, delay);
  };

  const run = async () => {
    timer = null;
    if (stopped || running) return;
    running = true;
    const visible = env.isVisible();
    controller = new AbortController();
    try {
      const result = await task(controller.signal);
      failures = 0;
      retryAfterSeconds = null;
      if (result.terminal) {
        stopped = true;
        unsubscribe();
      }
    } catch (error) {
      if (stopped) return;
      failures += 1;
      const ra = (error as { retryAfterSeconds?: unknown } | null)?.retryAfterSeconds;
      retryAfterSeconds = typeof ra === "number" ? ra : null;
      options.onError?.(error);
    } finally {
      running = false;
      controller = null;
    }
    hiddenPolls = visible ? 0 : hiddenPolls + 1;
    schedule();
  };

  const unsubscribe = env.onChange(() => {
    if (stopped) return;
    if (env.isVisible() && env.isOnline()) {
      // Became visible or reconnected: refresh promptly.
      hiddenPolls = 0;
      clear();
      if (!running) timer = env.setTimeout(run, 0);
    } else {
      schedule();
    }
  });

  timer = env.setTimeout(run, 0);

  return {
    stop() {
      stopped = true;
      clear();
      controller?.abort();
      unsubscribe();
    },
    pollNow() {
      if (stopped || running) return;
      clear();
      timer = env.setTimeout(run, 0);
    },
  };
}
