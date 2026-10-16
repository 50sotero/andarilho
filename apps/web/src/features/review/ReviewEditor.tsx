"use client";

import Link from "next/link";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Button, Card, ErrorNotice, StatusBadge, inputClass } from "@/components/ui";
import { ViewerClient } from "@/components/viewer/ViewerClient";
import { useProject } from "@/features/projects/use-project";
import { api } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { DeliveryEnvelope, FloorPlanLink, Layout, SceneDetail } from "@/lib/api/types";
import { usePolling } from "@/lib/use-polling";
import {
  calibrationFactor,
  distance,
  layoutProblems,
  layoutViewBox,
  round,
  scaleLayout,
  type Point,
  type ViewBox,
} from "./geometry";
import { AlignmentPanel } from "./AlignmentPanel";
import { underlayFromFit } from "./alignment";
import { loadCaptureEvidence, type CaptureEvidence } from "./capture-evidence";
import { LayoutCanvas, type PlanUnderlay } from "./LayoutCanvas";
import {
  addLandmark,
  addObstacle,
  addOpening,
  addWall,
  describeSelection,
  insertFloorCorner,
  movePoint,
  pointOf,
  removeSelected,
  updateField,
  type Selection,
} from "./layout-edit";

/** Downloads of a scene's evidence tried before the editor stops asking for it. */
const EVIDENCE_DOWNLOAD_ATTEMPTS = 3;
/** How long after a failed evidence download the editor reads the scene for a new link. */
export const EVIDENCE_RETRY_DELAY_MS = 5_000;

function NumberField({
  label,
  value,
  onChange,
  step = 0.01,
  min,
  unit = "m",
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  unit?: string;
}) {
  const id = useId();
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-xs font-medium">
        {label} {unit ? `(${unit})` : ""}
      </label>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        step={step}
        min={min}
        value={Number.isFinite(value) ? value : 0}
        onChange={(e) => {
          const v = e.target.valueAsNumber;
          if (Number.isFinite(v)) onChange(v);
        }}
        className={inputClass}
      />
    </div>
  );
}

function SelectionPanel({
  layout,
  selection,
  onChange,
  onRemove,
  onAddCorner,
  onAddOpening,
}: {
  layout: Layout;
  selection: Selection | null;
  onChange: (layout: Layout) => void;
  onRemove: () => void;
  onAddCorner: () => void;
  onAddOpening: () => void;
}) {
  if (!selection) {
    return (
      <p className="text-sm text-stone-600 dark:text-stone-400">
        Select a corner, wall, opening, obstacle, furniture piece, landmark or the starting point to
        edit it.
      </p>
    );
  }
  const point = pointOf(layout, selection);
  const setPoint = (p: Point) => onChange(movePoint(layout, selection, p));
  const field = (name: string, value: number | string) =>
    onChange(updateField(layout, selection, name, value));
  const wall =
    selection.kind === "wall" || selection.kind === "wallEnd"
      ? layout.walls?.[selection.index]
      : undefined;
  const opening = selection.kind === "opening" ? layout.openings?.[selection.index] : undefined;
  const obstacle =
    selection.kind === "obstacle" || selection.kind === "obstacleCorner"
      ? layout.obstacles?.[selection.index]
      : undefined;
  const landmark = selection.kind === "landmark" ? layout.landmarks?.[selection.index] : undefined;
  const furniture =
    selection.kind === "furniture" ? layout.furniture?.[selection.index] : undefined;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold">{describeSelection(layout, selection)}</h3>
      {furniture ? (
        <p className="text-sm text-stone-700 dark:text-stone-300">
          Placed automatically. Remove it if it is wrong; moving or resizing it is not available
          yet. {furniture.solid === false ? "You can walk through it." : "It blocks walking."}
        </p>
      ) : null}
      {point ? (
        <div className="grid grid-cols-2 gap-2">
          <NumberField label="X" value={point[0]} onChange={(v) => setPoint([v, point[1]])} />
          <NumberField label="Z" value={point[1]} onChange={(v) => setPoint([point[0], v])} />
        </div>
      ) : null}
      {wall ? (
        <div className="grid grid-cols-3 gap-2">
          <NumberField
            label="Length"
            value={round(distance(wall.start, wall.end))}
            min={0.01}
            onChange={(v) => field("length", v)}
          />
          <NumberField
            label="Thickness"
            value={wall.thickness}
            min={0.01}
            onChange={(v) => field("thickness", v)}
          />
          <NumberField
            label="Height"
            value={wall.height}
            min={0.1}
            step={0.05}
            onChange={(v) => field("height", v)}
          />
        </div>
      ) : null}
      {opening ? (
        <div className="grid grid-cols-2 gap-2">
          <NumberField
            label="Offset from wall start"
            value={opening.offset}
            min={0}
            onChange={(v) => field("offset", v)}
          />
          <NumberField
            label="Width"
            value={opening.width}
            min={0.05}
            onChange={(v) => field("width", v)}
          />
          <NumberField
            label="Bottom"
            value={opening.bottom ?? 0}
            min={0}
            onChange={(v) => field("bottom", v)}
          />
          <NumberField
            label="Top"
            value={opening.top}
            min={0.1}
            onChange={(v) => field("top", v)}
          />
          <div className="col-span-2 flex items-center gap-2 text-sm">
            <input
              id={`closed-${opening.id}`}
              type="checkbox"
              checked={opening.state === "closed"}
              onChange={(e) => field("state", e.target.checked ? "closed" : "open")}
            />
            <label htmlFor={`closed-${opening.id}`}>Captured closed (blocks walking)</label>
          </div>
        </div>
      ) : null}
      {obstacle ? (
        <NumberField
          label="Obstacle height"
          value={obstacle.height}
          min={0.01}
          step={0.05}
          onChange={(v) => field("height", v)}
        />
      ) : null}
      {selection.kind === "spawn" ? (
        <NumberField
          label="Facing direction"
          unit="degrees"
          step={5}
          value={round((layout.spawn.yawRadians * 180) / Math.PI, 1)}
          onChange={(v) => field("yawDegrees", v)}
        />
      ) : null}
      {landmark ? (
        <div className="grid grid-cols-2 gap-2">
          <NumberField
            label="Height (Y)"
            value={landmark.worldPoint[1]}
            onChange={(v) => field("y", v)}
          />
          <NumberField
            label="Uncertainty"
            value={landmark.uncertainty ?? 0}
            min={0}
            onChange={(v) => field("uncertainty", v)}
          />
        </div>
      ) : null}
      <div className="flex flex-wrap gap-2">
        {selection.kind === "floorCorner" ? (
          <Button variant="secondary" onClick={onAddCorner}>
            Add corner after this one
          </Button>
        ) : null}
        {wall ? (
          <Button variant="secondary" onClick={onAddOpening}>
            Add door/opening to this wall
          </Button>
        ) : null}
        {selection.kind !== "spawn" ? (
          <Button variant="danger" onClick={onRemove}>
            Remove
          </Button>
        ) : null}
      </div>
    </div>
  );
}

const IMAGE_PLAN_TYPES = new Set(["image/png", "image/jpeg"]);

function FloorPlanPanel({
  plan,
  underlay,
  defaults,
  onChange,
  onRefresh,
}: {
  plan: FloorPlanLink;
  underlay: PlanUnderlay | null;
  defaults: PlanUnderlay;
  onChange: (next: PlanUnderlay | null) => void;
  /** Fetch a fresh short-lived link; the one loaded with the page may have expired. */
  onRefresh: () => Promise<FloorPlanLink | null>;
}) {
  const id = useId();
  if (!IMAGE_PLAN_TYPES.has(plan.contentType)) {
    return (
      <div>
        <h3 className="text-sm font-semibold">Floor plan</h3>
        <p className="mt-1 text-sm text-stone-600 dark:text-stone-400">
          PDF plans cannot be traced in the editor yet.{" "}
          <button
            type="button"
            className="underline"
            onClick={() => {
              // Open the tab synchronously (popup blockers), then point it at a fresh link.
              const tab = window.open("about:blank", "_blank");
              if (tab) tab.opener = null;
              void onRefresh().then((fresh) => {
                if (tab && fresh) tab.location.href = fresh.url;
                else tab?.close();
              });
            }}
          >
            Open the plan
          </button>{" "}
          (private link, expires in a few minutes) and trace it by eye.
        </p>
      </div>
    );
  }
  const current = underlay ?? defaults;
  const field = (key: "x" | "z" | "width" | "rotationDegrees", label: string) => (
    <label className="text-sm" htmlFor={`${id}-${key}`}>
      {label}
      <input
        id={`${id}-${key}`}
        type="number"
        step="0.1"
        className={inputClass}
        value={current[key] ?? 0}
        disabled={!underlay}
        onChange={(e) => {
          const value = Number(e.target.value);
          if (Number.isFinite(value) && (key !== "width" || value > 0)) {
            onChange({ ...current, [key]: value });
          }
        }}
      />
    </label>
  );
  return (
    <div>
      <h3 className="text-sm font-semibold">Floor plan</h3>
      <label className="mt-1 flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={Boolean(underlay)}
          onChange={(e) => {
            if (!e.target.checked) {
              onChange(null);
              return;
            }
            void onRefresh().then((fresh) => {
              if (fresh) onChange({ ...defaults, url: fresh.url });
            });
          }}
        />
        Show the plan under the layout for tracing
      </label>
      <p className="mt-1 text-xs text-stone-600 dark:text-stone-400">
        The plan is private evidence: it is never part of a shared scene. Place it so that its walls
        line up with the capture (or mark landmarks on it and let them place it), then trace or
        calibrate against it.
      </p>
      <div className="mt-2 grid grid-cols-2 gap-2">
        {field("x", "Left (m)")}
        {field("z", "Top (m)")}
        {field("width", "Width (m)")}
        {field("rotationDegrees", "Rotation (°)")}
      </div>
      <label className="mt-2 block text-sm" htmlFor={`${id}-opacity`}>
        Opacity
        <input
          id={`${id}-opacity`}
          type="range"
          min="0.1"
          max="1"
          step="0.05"
          className="w-full"
          value={current.opacity}
          disabled={!underlay}
          onChange={(e) => onChange({ ...current, opacity: Number(e.target.value) })}
        />
      </label>
    </div>
  );
}

function CalibrationPanel({
  layout,
  onApply,
}: {
  layout: Layout;
  onApply: (factor: number, description: string) => void;
}) {
  const refId = useId();
  const knownId = useId();
  const [reference, setReference] = useState("");
  const [known, setKnown] = useState("");
  const walls = layout.walls ?? [];
  const landmarks = layout.landmarks ?? [];
  const options: { value: string; label: string; length: number }[] = [
    ...walls.map((w) => ({
      value: `wall:${w.id}`,
      label: `Wall ${w.id}`,
      length: distance(w.start, w.end),
    })),
  ];
  for (let i = 0; i < landmarks.length; i += 1) {
    for (let j = i + 1; j < landmarks.length; j += 1) {
      const a = landmarks[i]!;
      const b = landmarks[j]!;
      options.push({
        value: `lm:${a.id}:${b.id}`,
        label: `Landmarks ${a.id} to ${b.id}`,
        length: Math.hypot(
          a.worldPoint[0] - b.worldPoint[0],
          a.worldPoint[1] - b.worldPoint[1],
          a.worldPoint[2] - b.worldPoint[2],
        ),
      });
    }
  }
  const chosen = options.find((o) => o.value === reference);
  const factor = chosen ? calibrationFactor(chosen.length, Number(known)) : null;

  return (
    <div className="space-y-3 text-sm">
      <h3 className="font-semibold">Known-distance calibration</h3>
      <p className="text-stone-700 dark:text-stone-300">
        Pick a wall or a pair of landmarks whose real length you measured, then enter that length.
        The whole layout is scaled uniformly; nothing is stretched in one direction only.
      </p>
      <div className="space-y-1">
        <label htmlFor={refId} className="block text-xs font-medium">
          Reference
        </label>
        <select
          id={refId}
          value={reference}
          onChange={(e) => setReference(e.target.value)}
          className={inputClass}
        >
          <option value="">Choose...</option>
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label} (currently {o.length.toFixed(2)} m)
            </option>
          ))}
        </select>
      </div>
      <div className="space-y-1">
        <label htmlFor={knownId} className="block text-xs font-medium">
          Measured real length (m)
        </label>
        <input
          id={knownId}
          type="number"
          inputMode="decimal"
          min={0.01}
          step={0.01}
          value={known}
          onChange={(e) => setKnown(e.target.value)}
          className={inputClass}
        />
      </div>
      {factor ? <p>Scale factor: {factor.toFixed(4)}</p> : null}
      <Button
        variant="secondary"
        disabled={!factor || !chosen}
        onClick={() => {
          if (factor && chosen) {
            onApply(factor, `${chosen.label} set to ${Number(known).toFixed(2)} m`);
            setKnown("");
          }
        }}
      >
        Apply scale
      </Button>
    </div>
  );
}

/**
 * True while the scene's revision is being validated and packaged: the current one,
 * or (before validation makes it current) the first revision, listed only among
 * the scene's revisions.
 */
export function revisionPending(
  scene: Pick<SceneDetail, "revision" | "revisions"> | null,
): boolean {
  if (!scene) return false;
  if (scene.revision) return scene.revision.validationState === "pending";
  return scene.revisions.some((r) => r.validationState === "pending");
}

export function ReviewEditor({ projectId }: { projectId: string }) {
  const { detail, error: projectError } = useProject(projectId);
  const sceneId = detail?.scene?.id ?? null;
  const [scene, setScene] = useState<SceneDetail | null>(null);
  const [etag, setEtag] = useState<string | null>(null);
  const [layout, setLayout] = useState<Layout | null>(null);
  const [viewBox, setViewBox] = useState<ViewBox | null>(null);
  const [dirty, setDirty] = useState(false);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [overlay, setOverlay] = useState(false);
  const [error, setError] = useState<FriendlyError | null>(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [envelope, setEnvelope] = useState<DeliveryEnvelope | null>(null);
  const [underlay, setUnderlay] = useState<PlanUnderlay | null>(null);
  const [evidence, setEvidence] = useState<CaptureEvidence | null>(null);
  const [showEvidence, setShowEvidence] = useState(true);
  // The evidence does not change for a run: once downloaded, later polls only renew its
  // link. A failed download (an expired link, a network blip) is tried again with a later
  // link, a few times at most, so a malformed file is not fetched on every poll.
  const evidenceRef = useRef({
    sceneId: null as string | null,
    loaded: false,
    pending: false,
    failures: 0,
  });
  // Failed evidence downloads that asked for a fresh link: each one schedules a scene read.
  const [evidenceRetries, setEvidenceRetries] = useState(0);
  const overlayId = useId();
  const evidenceId = useId();
  // The revision whose layout the canvas shows. Approval is only offered for it, so
  // a revision saved elsewhere can never be approved against a stale canvas.
  const [shownRevisionId, setShownRevisionId] = useState<string | null>(null);
  const shownRef = useRef<string | null>(null);
  const dirtyRef = useRef(false);
  useEffect(() => {
    dirtyRef.current = dirty;
  }, [dirty]);

  const showRevision = useCallback((rev: NonNullable<SceneDetail["revision"]>) => {
    setLayout(structuredClone(rev.layout));
    setViewBox(layoutViewBox(rev.layout, 1));
    setDirty(false);
    setSelection(null);
    shownRef.current = rev.id;
    setShownRevisionId(rev.id);
  }, []);

  /** Adopt fresh scene data; follow a newer revision unless there are unsaved edits. */
  /** Counts a failed evidence download; while tries remain, a fresh link is asked for. */
  const evidenceFailed = useCallback(() => {
    const ev = evidenceRef.current;
    ev.failures += 1;
    if (ev.failures < EVIDENCE_DOWNLOAD_ATTEMPTS) setEvidenceRetries((n) => n + 1);
  }, []);

  /** Downloads the scene's evidence, unless it is loaded, under way or given up on. */
  const requestEvidence = useCallback(
    (data: SceneDetail) => {
      // A missing, expired or failed download leaves no layer and no error.
      const ev = evidenceRef.current;
      if (ev.sceneId !== data.scene.id) {
        Object.assign(ev, { sceneId: data.scene.id, loaded: false, failures: 0 });
      }
      if (
        !data.captureEvidence ||
        ev.loaded ||
        ev.pending ||
        ev.failures >= EVIDENCE_DOWNLOAD_ATTEMPTS
      ) {
        return;
      }
      const sceneId = data.scene.id;
      ev.pending = true;
      void loadCaptureEvidence(data.captureEvidence).then((loaded) => {
        ev.pending = false;
        if (ev.sceneId !== sceneId) return;
        if (loaded) {
          ev.loaded = true;
          setEvidence(loaded);
        } else {
          evidenceFailed();
        }
      });
    },
    [evidenceFailed],
  );

  const adoptScene = useCallback(
    (data: SceneDetail, resetLayout: boolean) => {
      setScene(data);
      requestEvidence(data);
      const rev = data.revision;
      if (!rev) return;
      if (resetLayout) {
        showRevision(rev);
      } else if (rev.id !== shownRef.current && !dirtyRef.current) {
        const replacing = shownRef.current !== null;
        showRevision(rev);
        // A first revision that just finished processing was not "saved since".
        if (replacing) {
          setStatus(`Showing revision ${rev.revision}, saved since you loaded this page.`);
        }
      }
    },
    [showRevision, requestEvidence],
  );

  const refreshFloorPlan = useCallback(async (): Promise<FloorPlanLink | null> => {
    if (!sceneId) return null;
    try {
      const { data } = await api.getScene(sceneId);
      // Only the plan link is refreshed here; the editable layout is left alone.
      setScene((prev) => (prev ? { ...prev, floorPlan: data.floorPlan } : prev));
      return data.floorPlan;
    } catch (err) {
      setError(toFriendlyError(err));
      return null;
    }
  }, [sceneId]);

  const loadScene = useCallback(
    async (resetLayout: boolean) => {
      if (!sceneId) return;
      const result = await api.getScene(sceneId);
      setEtag(result.etag);
      adoptScene(result.data, resetLayout);
    },
    [sceneId, adoptScene],
  );

  useEffect(() => {
    let active = true;
    if (!sceneId) return;
    api
      .getScene(sceneId)
      .then((result) => {
        if (!active) return;
        setEtag(result.etag);
        adoptScene(result.data, true);
      })
      .catch((err: unknown) => {
        if (active) setError(toFriendlyError(err));
      });
    return () => {
      active = false;
    };
  }, [sceneId, adoptScene]);

  // Nothing else may read the scene again soon (a valid scene is not polled), so a failed
  // evidence download reads it for a fresh link itself, a little later.
  useEffect(() => {
    if (evidenceRetries === 0 || !sceneId) return;
    let active = true;
    const timer = setTimeout(() => {
      api.getScene(sceneId).then(
        ({ data }) => {
          if (active) requestEvidence(data);
        },
        () => {
          if (active) evidenceFailed();
        },
      );
    }, EVIDENCE_RETRY_DELAY_MS);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [evidenceRetries, sceneId, requestEvidence, evidenceFailed]);

  // While a saved revision is being validated/packaged, poll the scene.
  const pending = revisionPending(scene);
  usePolling(
    async (signal) => {
      if (!sceneId) return { terminal: true };
      const result = await api.getScene(sceneId, signal);
      setEtag(result.etag);
      adoptScene(result.data, false);
      const done = !revisionPending(result.data);
      const finished = result.data.revision;
      if (done) {
        setStatus(
          finished
            ? `Revision ${finished.revision} validation finished.`
            : "Validation finished without a revision to review.",
        );
      }
      return { terminal: done };
    },
    { enabled: Boolean(pending && sceneId), key: `${sceneId}:${scene?.revision?.id ?? ""}` },
  );

  const revision = scene?.revision ?? null;

  // 3D preview of the current packaged revision (with collision overlay toggle).
  const revisionId =
    revision?.manifest && revision.validationState === "valid" ? revision.id : null;
  // Bumped to fetch a fresh envelope for the same revision (a retry, or expired links).
  const [previewAttempt, setPreviewAttempt] = useState(0);
  const [previewFailure, setPreviewFailure] = useState<{
    revisionId: string;
    error: FriendlyError;
  } | null>(null);
  useEffect(() => {
    if (!sceneId || !revisionId) return;
    let active = true;
    api.getDelivery(sceneId, revisionId).then(
      ({ data }) => {
        if (!active) return;
        setEnvelope(data);
        setPreviewFailure(null);
      },
      (err: unknown) => {
        if (active) setPreviewFailure({ revisionId, error: toFriendlyError(err) });
      },
    );
    return () => {
      active = false;
    };
  }, [sceneId, revisionId, previewAttempt]);
  const retryPreview = useCallback(() => setPreviewAttempt((n) => n + 1), []);
  const previewError = previewFailure?.revisionId === revisionId ? previewFailure.error : null;
  // Never show an envelope from an earlier revision: while a new save is pending
  // or after it fails validation there is no current preview.
  const preview =
    envelope && revisionId && envelope.sceneRevisionId === revisionId ? envelope : null;

  function edit(next: Layout) {
    setLayout(next);
    setDirty(true);
  }

  const center: Point = viewBox
    ? [round(viewBox.minX + viewBox.width / 2, 2), round(viewBox.minY + viewBox.height / 2, 2)]
    : [0, 0];

  async function save() {
    // The base is the revision the canvas was edited from, never a newer one seen by
    // polling: the server then refuses a stale canvas instead of stacking it on top.
    const baseRevisionId = shownRef.current;
    if (!layout || !baseRevisionId || !sceneId || !etag) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.saveLayout(sceneId, baseRevisionId, layout, etag);
      setStatus("Saved as a new draft revision. Validating the geometry...");
      // The canvas now shows exactly the revision that was just saved.
      shownRef.current = result.data.sceneRevisionId;
      setShownRevisionId(result.data.sceneRevisionId);
      dirtyRef.current = false;
      setDirty(false);
      // The save's ETag first, then the reload's: if someone saved in between, the
      // reload adopts their revision and its newer ETag, which must win.
      if (result.etag) setEtag(result.etag);
      await loadScene(false);
    } catch (err) {
      setError(toFriendlyError(err));
    } finally {
      setBusy(false);
    }
  }

  async function approve() {
    if (!revision || !sceneId || !etag) return;
    setBusy(true);
    setError(null);
    try {
      await api.approveRevision(sceneId, revision.id, etag);
      setStatus(`Revision ${revision.revision} approved. You can now publish it.`);
      await loadScene(false);
    } catch (err) {
      setError(toFriendlyError(err));
    } finally {
      setBusy(false);
    }
  }

  async function discard() {
    setError(null);
    try {
      await loadScene(true);
      setStatus("Reloaded the latest saved layout.");
    } catch (err) {
      setError(toFriendlyError(err));
    }
  }

  const base = `/projects/${encodeURIComponent(projectId)}`;
  if (detail && !detail.scene) {
    return (
      <Card>
        <p>There is no scene to review yet. It appears after processing succeeds.</p>
        <Link href={`${base}/processing`} className="text-sm underline">
          Check processing
        </Link>
      </Card>
    );
  }
  if (scene && !scene.revision) {
    // The scene exists but its first revision is still processing, or its run
    // ended before that revision was ready.
    return (
      <Card>
        <p>This scene has no layout to review yet. It appears after processing succeeds.</p>
        <Link href={`${base}/processing`} className="text-sm underline">
          Check processing
        </Link>
      </Card>
    );
  }
  if (!layout || !viewBox || !revision) {
    return (
      <div className="space-y-2">
        <ErrorNotice error={projectError ?? error} />
        {!error && !projectError ? <p aria-live="polite">Loading the layout...</p> : null}
      </div>
    );
  }

  const problems = layoutProblems(layout);
  const stale = revision.id !== shownRevisionId;
  const canApprove =
    !dirty &&
    revision.validationState === "valid" &&
    revision.approvalState !== "approved" &&
    revision.id === shownRevisionId &&
    !busy;

  return (
    <div className="space-y-6">
      <Card>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-sm">
            Revision {revision.revision} (layout {revision.layoutRevision}){" "}
            <StatusBadge
              tone={
                revision.validationState === "valid"
                  ? "good"
                  : revision.validationState === "invalid"
                    ? "bad"
                    : "warn"
              }
            >
              {revision.validationState === "pending" ? "validating" : revision.validationState}
            </StatusBadge>{" "}
            <StatusBadge tone={revision.approvalState === "approved" ? "good" : "neutral"}>
              {revision.approvalState}
            </StatusBadge>
            {dirty ? (
              <>
                {" "}
                <StatusBadge tone="warn">unsaved changes</StatusBadge>
              </>
            ) : null}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              onClick={() => void save()}
              disabled={!dirty || busy || stale || problems.length > 0}
            >
              {busy ? "Working..." : "Save as new revision"}
            </Button>
            <Button variant="secondary" onClick={() => void discard()} disabled={busy}>
              Discard changes
            </Button>
            <Button variant="secondary" onClick={() => void approve()} disabled={!canApprove}>
              Approve this revision
            </Button>
          </div>
        </div>
        <p aria-live="polite" className="mt-2 text-sm text-teal-800 dark:text-teal-300">
          {status}
        </p>
        {stale ? (
          <p role="status" className="mt-2 text-sm text-amber-800 dark:text-amber-300">
            Someone saved revision {revision.revision} while you were editing. Discard your changes
            to see it before saving or approving.
          </p>
        ) : null}
        <div className="mt-2">
          <ErrorNotice error={error} />
        </div>
        {revision.validationFindings.length > 0 ? (
          <div className="mt-2 text-sm">
            <p className="font-medium">Validation findings</p>
            <ul className="list-disc pl-5">
              {revision.validationFindings.map((f, i) => (
                <li key={`${f.code}-${i}`}>
                  {f.message} <span className="text-xs text-stone-500">({f.code})</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        <p className="mt-2 text-xs text-stone-600 dark:text-stone-400">
          Saving creates a new draft revision and does not change a revision that is already
          published. The scene is walkable without approval; approval is needed before publishing.
        </p>
      </Card>

      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <Card>
          <div className="mb-3 flex flex-wrap gap-2">
            <Button variant="secondary" onClick={() => edit(addWall(layout, center))}>
              Add wall
            </Button>
            <Button variant="secondary" onClick={() => edit(addObstacle(layout, center))}>
              Add obstacle
            </Button>
            <Button variant="secondary" onClick={() => edit(addLandmark(layout, center))}>
              Add landmark
            </Button>
            <Button variant="ghost" onClick={() => setViewBox(layoutViewBox(layout, 1))}>
              Fit to view
            </Button>
            <span className="flex items-center gap-1 text-sm">
              <input
                id={overlayId}
                type="checkbox"
                checked={overlay}
                onChange={(e) => setOverlay(e.target.checked)}
              />
              <label htmlFor={overlayId}>Highlight collision</label>
            </span>
            {evidence ? (
              <span className="flex items-center gap-1 text-sm">
                <input
                  id={evidenceId}
                  type="checkbox"
                  checked={showEvidence}
                  onChange={(e) => setShowEvidence(e.target.checked)}
                />
                <label htmlFor={evidenceId}>Capture evidence</label>
              </span>
            ) : null}
          </div>
          <LayoutCanvas
            layout={layout}
            viewBox={viewBox}
            selection={selection}
            onSelect={setSelection}
            onChange={edit}
            showOverlay={overlay}
            underlay={
              underlay && scene?.floorPlan ? { ...underlay, url: scene.floorPlan.url } : null
            }
            evidence={evidence}
            showEvidence={showEvidence}
          />
          <p className="mt-2 text-xs text-stone-600 dark:text-stone-400">
            Grid squares are 1 m in the layout&apos;s units. Teal: walkable floor; black: walls;
            white gaps: openings; yellow: obstacles; grey: furniture (dashed where you can walk
            through it); purple: landmarks; red: starting point and facing direction.
            {evidence && showEvidence
              ? " Blue dots and the dashed purple line are the capture's points and camera path."
              : ""}
            {revision.manifest?.scaleStatus === "estimated"
              ? " Scale is estimated: confirm a known distance before relying on measurements."
              : ""}
          </p>
          <div aria-live="polite" className="mt-2 text-sm">
            {problems.length > 0 ? (
              <ul className="list-disc pl-5 text-red-700 dark:text-red-300">
                {problems.map((p, i) => (
                  <li key={i}>{p.message}</li>
                ))}
              </ul>
            ) : null}
          </div>
        </Card>

        <div className="space-y-6">
          <Card>
            <SelectionPanel
              layout={layout}
              selection={selection}
              onChange={edit}
              onRemove={() => {
                if (selection) edit(removeSelected(layout, selection));
                setSelection(null);
              }}
              onAddCorner={() => {
                if (selection?.kind === "floorCorner") {
                  edit(insertFloorCorner(layout, selection.floor, selection.ring, selection.index));
                }
              }}
              onAddOpening={() => {
                if (selection && (selection.kind === "wall" || selection.kind === "wallEnd")) {
                  edit(addOpening(layout, selection.index));
                }
              }}
            />
          </Card>
          {scene?.floorPlan ? (
            <Card>
              <FloorPlanPanel
                plan={scene.floorPlan}
                underlay={underlay}
                defaults={{
                  url: scene.floorPlan.url,
                  x: viewBox.minX,
                  z: viewBox.minY,
                  width: viewBox.width,
                  opacity: 0.5,
                }}
                onChange={setUnderlay}
                onRefresh={refreshFloorPlan}
              />
            </Card>
          ) : null}
          <Card>
            <CalibrationPanel
              layout={layout}
              onApply={(factor, description) => {
                edit(scaleLayout(layout, factor));
                setViewBox(layoutViewBox(scaleLayout(layout, factor), 1));
                setStatus(`Scaled the layout: ${description}. Save to keep it.`);
              }}
            />
          </Card>
          <Card>
            <AlignmentPanel
              layout={layout}
              selection={selection}
              plan={scene?.floorPlan ?? null}
              onSelect={setSelection}
              onChange={edit}
              onRefreshPlan={refreshFloorPlan}
              onPlace={(fit) => {
                void refreshFloorPlan().then((fresh) => {
                  if (!fresh) return;
                  setUnderlay(underlayFromFit(fit, fresh.url, underlay?.opacity ?? 0.5));
                  setStatus(
                    "Placed the plan from the landmarks. Check that its walls match the capture " +
                      "before tracing against it.",
                  );
                });
              }}
            />
          </Card>
        </div>
      </div>

      <Card>
        <h2 className="text-lg font-semibold">3D check</h2>
        <p className="mt-1 text-sm text-stone-700 dark:text-stone-300">
          Walk the current saved revision and turn on the collision overlay to confirm walls, doors
          and obstacles line up with the captured scene.
        </p>
        <div className="mt-3 space-y-3">
          <ErrorNotice error={previewError} onRetry={retryPreview} />
          {preview ? (
            <ViewerClient
              envelope={preview}
              title={detail?.project.title ?? "this place"}
              allowCollisionOverlay
              onRecoverableError={retryPreview}
            />
          ) : previewError ? null : revisionId ? (
            <p aria-live="polite" className="text-sm">
              Preparing a viewing link for the 3D preview...
            </p>
          ) : (
            <p className="text-sm">
              The 3D preview is available once this revision has been packaged and validated.
            </p>
          )}
        </div>
      </Card>
    </div>
  );
}
