import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  POLL_HIDDEN_BASE_MS,
  POLL_MAX_MS,
  POLL_VISIBLE_MS,
  nextPollDelay,
  startPolling,
  type PollEnvironment,
} from "@/lib/polling";

describe("nextPollDelay", () => {
  const base = { visible: true, online: true, hiddenPolls: 0, failures: 0 };

  it("polls every 3 s while visible", () => {
    expect(nextPollDelay(base)).toBe(POLL_VISIBLE_MS);
    expect(POLL_VISIBLE_MS).toBe(3000);
  });

  it("backs off exponentially while hidden, capped", () => {
    expect(nextPollDelay({ ...base, visible: false, hiddenPolls: 0 })).toBe(POLL_HIDDEN_BASE_MS);
    expect(nextPollDelay({ ...base, visible: false, hiddenPolls: 1 })).toBe(
      POLL_HIDDEN_BASE_MS * 2,
    );
    expect(nextPollDelay({ ...base, visible: false, hiddenPolls: 10 })).toBe(POLL_MAX_MS);
  });

  it("pauses while offline", () => {
    expect(nextPollDelay({ ...base, online: false })).toBeNull();
  });

  it("backs off after failures and honours Retry-After", () => {
    expect(nextPollDelay({ ...base, failures: 1 })).toBe(6000);
    expect(nextPollDelay({ ...base, failures: 3 })).toBe(24000);
    expect(nextPollDelay({ ...base, failures: 20 })).toBe(POLL_MAX_MS);
    expect(nextPollDelay({ ...base, retryAfterSeconds: 30 })).toBe(30000);
  });
});

function fakeEnv() {
  const state = { visible: true, online: true };
  const listeners = new Set<() => void>();
  const env: PollEnvironment = {
    isVisible: () => state.visible,
    isOnline: () => state.online,
    setTimeout: (fn, ms) => setTimeout(fn, ms),
    clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
    onChange: (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
  return {
    env,
    state,
    listeners,
    emit: () => {
      for (const l of [...listeners]) l();
    },
  };
}

describe("startPolling", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("polls immediately, then every 3 s, and stops on a terminal state", async () => {
    const { env, listeners } = fakeEnv();
    let calls = 0;
    const task = vi.fn(async () => {
      calls += 1;
      return { terminal: calls >= 3 };
    });
    startPolling(task, env);
    await vi.advanceTimersByTimeAsync(0);
    expect(task).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2999);
    expect(task).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(task).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(3000);
    expect(task).toHaveBeenCalledTimes(3);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(task).toHaveBeenCalledTimes(3);
    expect(listeners.size).toBe(0);
  });

  it("slows down while hidden and refreshes immediately when visible again", async () => {
    const { env, state, emit } = fakeEnv();
    const task = vi.fn(async () => ({ terminal: false }));
    startPolling(task, env);
    await vi.advanceTimersByTimeAsync(0);
    state.visible = false;
    emit();
    await vi.advanceTimersByTimeAsync(3000);
    expect(task).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(POLL_HIDDEN_BASE_MS);
    expect(task).toHaveBeenCalledTimes(2);
    state.visible = true;
    emit();
    await vi.advanceTimersByTimeAsync(0);
    expect(task).toHaveBeenCalledTimes(3);
  });

  it("pauses offline and resumes on reconnect", async () => {
    const { env, state, emit } = fakeEnv();
    const task = vi.fn(async () => ({ terminal: false }));
    startPolling(task, env);
    await vi.advanceTimersByTimeAsync(0);
    state.online = false;
    emit();
    await vi.advanceTimersByTimeAsync(120_000);
    expect(task).toHaveBeenCalledTimes(1);
    state.online = true;
    emit();
    await vi.advanceTimersByTimeAsync(0);
    expect(task).toHaveBeenCalledTimes(2);
  });

  it("backs off on errors without stopping, and stop() cancels everything", async () => {
    const { env } = fakeEnv();
    const onError = vi.fn();
    const task = vi.fn(async () => {
      throw new Error("boom");
    });
    const poller = startPolling(task, env, { onError });
    await vi.advanceTimersByTimeAsync(0);
    expect(onError).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(5999);
    expect(task).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(task).toHaveBeenCalledTimes(2);
    poller.stop();
    await vi.advanceTimersByTimeAsync(600_000);
    expect(task).toHaveBeenCalledTimes(2);
  });
});
