"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { ProjectDetail } from "@/lib/api/types";

export interface ProjectState {
  detail: ProjectDetail | null;
  etag: string | null;
  error: FriendlyError | null;
  loading: boolean;
  reload: () => Promise<void>;
}

/** Loads GET /v1/projects/{id} and exposes a reload function. */
export function useProject(projectId: string): ProjectState {
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [etag, setEtag] = useState<string | null>(null);
  const [error, setError] = useState<FriendlyError | null>(null);
  const [loading, setLoading] = useState(true);

  // Promise callbacks (not an async body) so state updates are never synchronous in effects.
  const fetchOnce = useCallback(
    (signal?: AbortSignal): Promise<void> =>
      api.getProject(projectId, signal).then(
        (result) => {
          if (signal?.aborted) return;
          setDetail(result.data);
          setEtag(result.etag);
          setError(null);
          setLoading(false);
        },
        (err: unknown) => {
          if (signal?.aborted) return;
          setError(toFriendlyError(err));
          setLoading(false);
        },
      ),
    [projectId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void fetchOnce(controller.signal);
    return () => controller.abort();
  }, [fetchOnce]);

  const reload = useCallback(() => fetchOnce(), [fetchOnce]);

  return { detail, etag, error, loading, reload };
}
