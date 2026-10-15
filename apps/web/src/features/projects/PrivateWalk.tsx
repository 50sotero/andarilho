"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, ErrorNotice } from "@/components/ui";
import { ViewerClient } from "@/components/viewer/ViewerClient";
import { LayoutPlan } from "@/features/review/LayoutPlan";
import { api } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { DeliveryEnvelope, Layout, SceneDetail } from "@/lib/api/types";
import { usePolling } from "@/lib/use-polling";
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

/**
 * True while nothing is walkable yet but a revision is still being packaged or
 * validated (or the first one is about to be created): worth checking again.
 */
export function awaitingRevision(scene: SceneDetail): boolean {
  if (walkableRevisionId(scene) !== null) return false;
  if (scene.revisions.length === 0) return true;
  return (
    scene.revision?.validationState === "pending" ||
    scene.revisions.some((r) => r.validationState === "pending")
  );
}

/**
 * The 2D plan shown beside (or instead of) the 3D walk. Only the current draft's
 * layout is loaded, so it is offered only when the draft is the revision being walked;
 * an older delivered revision has no matching plan here.
 */
export function fallbackLayout(
  scene: SceneDetail | null,
  envelope: DeliveryEnvelope | null,
): Layout | null {
  const draft = scene?.revision ?? null;
  if (!draft) return null;
  return !envelope || envelope.sceneRevisionId === draft.id ? draft.layout : null;
}

export function PrivateWalk({ projectId }: { projectId: string }) {
  const { detail, error: projectError } = useProject(projectId);
  const [scene, setScene] = useState<SceneDetail | null>(null);
  const [envelope, setEnvelope] = useState<DeliveryEnvelope | null>(null);
  const [error, setError] = useState<FriendlyError | null>(null);
  const [noRevision, setNoRevision] = useState(false);
  const refreshes = useRef(0);
  const sceneId = detail?.scene?.id ?? null;
  // Viewers cannot read draft review data; they walk the published (approved) revision.
  const viewerRevisionId =
    detail?.project.role === "viewer" ? (detail.project.publishedRevisionId ?? "") : null;

  const fetchEnvelope = useCallback(
    (signal?: { cancelled: boolean }): Promise<void> => {
      if (!sceneId) return Promise.resolve();
      const revisionLookup =
        viewerRevisionId !== null
          ? Promise.resolve(viewerRevisionId || null)
          : api.getScene(sceneId).then(({ data: sceneData }) => {
              if (!signal?.cancelled) setScene(sceneData);
              return walkableRevisionId(sceneData);
            });
      return revisionLookup
        .then((revisionId) => {
          if (signal?.cancelled) return;
          if (!revisionId) {
            setNoRevision(true);
            return;
          }
          return api.getDelivery(sceneId, revisionId).then(({ data }) => {
            if (signal?.cancelled) return;
            setEnvelope(data);
            setNoRevision(false);
            setError(null);
          });
        })
        .catch((err: unknown) => {
          if (!signal?.cancelled) setError(toFriendlyError(err));
        });
    },
    [sceneId, viewerRevisionId],
  );

  useEffect(() => {
    const signal = { cancelled: false };
    void fetchEnvelope(signal);
    return () => {
      signal.cancelled = true;
    };
  }, [fetchEnvelope]);

  // The first revision may still be validating when the page opens: keep checking
  // until it becomes walkable (or stops being pending) instead of waiting for a reload.
  const waiting =
    noRevision && viewerRevisionId === null && scene !== null && awaitingRevision(scene);
  usePolling(
    async (abort) => {
      if (!sceneId) return { terminal: true };
      const { data } = await api.getScene(sceneId, abort);
      setScene(data);
      if (walkableRevisionId(data) !== null) {
        await fetchEnvelope();
        return { terminal: true };
      }
      return { terminal: !awaitingRevision(data) };
    },
    { enabled: waiting, key: sceneId ?? "" },
  );

  // Delivery URLs are short-lived: fetch a fresh envelope for the same revision when needed.
  const onRecoverableError = useCallback(() => {
    if (refreshes.current >= MAX_ENVELOPE_REFRESHES) return;
    refreshes.current += 1;
    void fetchEnvelope();
  }, [fetchEnvelope]);

  const base = `/projects/${encodeURIComponent(projectId)}`;
  const layout = fallbackLayout(scene, envelope);
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
        This private view is visible only to project members. Asset links expire after a few minutes
        and are refreshed automatically.
      </p>
    </div>
  );
}
