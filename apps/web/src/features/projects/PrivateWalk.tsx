"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, ErrorNotice } from "@/components/ui";
import { ViewerClient } from "@/components/viewer/ViewerClient";
import { LayoutPlan } from "@/features/review/LayoutPlan";
import { api } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { DeliveryEnvelope, SceneDetail } from "@/lib/api/types";
import { useProject } from "./use-project";

const MAX_ENVELOPE_REFRESHES = 2;

/** Picks the revision to walk privately: the current draft if packaged, else the newest valid one. */
export function walkableRevisionId(scene: SceneDetail): string | null {
  const current = scene.revision;
  if (current && current.manifest && current.validationState === "valid") return current.id;
  const valid = scene.revisions
    .filter((r) => r.validationState === "valid")
    .sort((a, b) => b.revision - a.revision);
  return valid[0]?.id ?? null;
}

export function PrivateWalk({ projectId }: { projectId: string }) {
  const { detail, error: projectError } = useProject(projectId);
  const [scene, setScene] = useState<SceneDetail | null>(null);
  const [envelope, setEnvelope] = useState<DeliveryEnvelope | null>(null);
  const [error, setError] = useState<FriendlyError | null>(null);
  const [noRevision, setNoRevision] = useState(false);
  const refreshes = useRef(0);
  const sceneId = detail?.scene?.id ?? null;

  const fetchEnvelope = useCallback(
    (signal?: { cancelled: boolean }): Promise<void> => {
      if (!sceneId) return Promise.resolve();
      return api
        .getScene(sceneId)
        .then(({ data: sceneData }) => {
          if (signal?.cancelled) return;
          setScene(sceneData);
          const revisionId = walkableRevisionId(sceneData);
          if (!revisionId) {
            setNoRevision(true);
            return;
          }
          return api.getDelivery(sceneId, revisionId).then(({ data }) => {
            if (signal?.cancelled) return;
            setEnvelope(data);
            setError(null);
          });
        })
        .catch((err: unknown) => {
          if (!signal?.cancelled) setError(toFriendlyError(err));
        });
    },
    [sceneId],
  );

  useEffect(() => {
    const signal = { cancelled: false };
    void fetchEnvelope(signal);
    return () => {
      signal.cancelled = true;
    };
  }, [fetchEnvelope]);

  // Delivery URLs are short-lived: fetch a fresh envelope for the same revision when needed.
  const onRecoverableError = useCallback(() => {
    if (refreshes.current >= MAX_ENVELOPE_REFRESHES) return;
    refreshes.current += 1;
    void fetchEnvelope();
  }, [fetchEnvelope]);

  const base = `/projects/${encodeURIComponent(projectId)}`;
  const layout = scene?.revision?.layout ?? null;
  const title = detail?.project.title ?? "this place";

  if (detail && !detail.scene) {
    return (
      <Card>
        <p>There is no scene yet. It appears after processing succeeds.</p>
        <Link href={`${base}/processing`} className="text-sm underline">
          Check processing
        </Link>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <ErrorNotice error={projectError ?? error} onRetry={() => void fetchEnvelope()} />
      {noRevision ? (
        <Card>
          <p>No packaged, validated revision is available to walk yet.</p>
          {layout ? <LayoutPlan layout={layout} title={title} /> : null}
        </Card>
      ) : null}
      {envelope ? (
        <ViewerClient
          envelope={envelope}
          title={title}
          allowCollisionOverlay
          onRecoverableError={onRecoverableError}
          fallback={layout ? <LayoutPlan layout={layout} title={title} /> : null}
        />
      ) : !noRevision && !error ? (
        <p aria-live="polite">Preparing a private viewing link...</p>
      ) : null}
      <p className="text-sm text-stone-600 dark:text-stone-400">
        This private view is visible only to project members. Asset links expire after a few
        minutes and are refreshed automatically.
      </p>
    </div>
  );
}
