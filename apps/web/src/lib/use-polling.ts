"use client";

import { useEffect, useRef, type RefObject } from "react";
import { browserPollEnvironment, startPolling, type Poller, type PollResult } from "./polling";

/**
 * React wrapper around startPolling. The latest `task` is always used without
 * restarting the poller; change `key` to restart (e.g. a new job id).
 */
export function usePolling(
  task: (signal: AbortSignal) => Promise<PollResult>,
  options: { enabled: boolean; key: string; onError?: (error: unknown) => void },
): RefObject<Poller | null> {
  const taskRef = useRef(task);
  const onErrorRef = useRef(options.onError);
  const pollerRef = useRef<Poller | null>(null);

  useEffect(() => {
    taskRef.current = task;
    onErrorRef.current = options.onError;
  });

  useEffect(() => {
    if (!options.enabled) return;
    const poller = startPolling((signal) => taskRef.current(signal), browserPollEnvironment(), {
      onError: (error) => onErrorRef.current?.(error),
    });
    pollerRef.current = poller;
    return () => {
      poller.stop();
      if (pollerRef.current === poller) pollerRef.current = null;
    };
  }, [options.enabled, options.key]);

  return pollerRef;
}
