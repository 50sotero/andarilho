"use client";

import { useCallback, useEffect, useState } from "react";
import { Card } from "@/components/ui";
import { ViewerClient } from "@/components/viewer/ViewerClient";
import { api } from "@/lib/api/client";
import { ApiError } from "@/lib/api/errors";
import type { DeliveryEnvelope } from "@/lib/api/types";

/**
 * Read-only published scene. The token is exchanged through the same-origin
 * proxy for a derivative-only delivery envelope. The token is never logged.
 */
export function PublicScene({ token }: { token: string }) {
  const [envelope, setEnvelope] = useState<DeliveryEnvelope | null>(null);
  const [problem, setProblem] = useState<"unavailable" | "error" | null>(null);

  const exchange = useCallback(
    (signal?: { cancelled: boolean }): Promise<void> =>
      api.getPublicDelivery(token).then(
        ({ data }) => {
          if (signal?.cancelled) return;
          setEnvelope(data);
          setProblem(null);
        },
        (err: unknown) => {
          if (signal?.cancelled) return;
          setProblem(
            err instanceof ApiError &&
              (err.status === 404 || err.status === 410 || err.status === 403)
              ? "unavailable"
              : "error",
          );
        },
      ),
    [token],
  );

  useEffect(() => {
    const signal = { cancelled: false };
    void exchange(signal);
    return () => {
      signal.cancelled = true;
    };
  }, [exchange]);

  const onRecoverableError = useCallback(() => void exchange(), [exchange]);

  if (problem === "unavailable") {
    return (
      <Card>
        <h2 className="text-lg font-semibold">This link is not available</h2>
        <p className="mt-1 text-sm">
          The link may be mistyped, expired, or revoked by its owner. Ask the person who shared it
          for a new link.
        </p>
      </Card>
    );
  }
  if (problem === "error") {
    return (
      <Card>
        <div role="alert">
          <h2 className="text-lg font-semibold">The scene could not be loaded</h2>
          <p className="mt-1 text-sm">Check your connection and reload the page.</p>
        </div>
      </Card>
    );
  }
  if (!envelope) return <p aria-live="polite">Opening the shared scene...</p>;
  return (
    <ViewerClient
      envelope={envelope}
      title="the shared place"
      onRecoverableError={onRecoverableError}
    />
  );
}
