"use client";

import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import type { DeliveryEnvelope } from "@andarilho/scene-contract";
import { canWalk, createViewer } from "@andarilho/viewer";
import { Button } from "@/components/ui";

/* -------------------------------------------------------------------------- */
/* Viewer adapter: the only place that knows @andarilho/viewer's API.          */
/* If the engine API changes, only this section needs to change.               */
/* -------------------------------------------------------------------------- */

export type QualityChoice = "auto" | "low" | "mobile" | "desktop";

export type EngineView =
  | { kind: "loading"; steps: { label: string; status: string }[]; renderFraction: number | null }
  | { kind: "ready"; walkable: boolean; inspectionReason: string | null; quality: string | null }
  | { kind: "walking"; quality: string | null; pointerLocked: boolean }
  | { kind: "error"; message: string; recoverable: boolean; webglUnavailable: boolean }
  | { kind: "context-lost" }
  | { kind: "idle" };

interface EngineAdapter {
  enterWalk(): boolean;
  exitWalk(): void;
  resetPosition(): void;
  setQuality(q: QualityChoice): void;
  setCollisionOverlay(on: boolean): void;
  dispose(): void;
}

type EngineState = ReturnType<ReturnType<typeof createViewer>["getState"]>;

function toEngineView(state: EngineState): EngineView {
  switch (state.status) {
    case "loading":
      return {
        kind: "loading",
        renderFraction: state.progress.renderFraction,
        steps: [
          { label: "Visual scene", status: state.progress.render },
          { label: "Collision geometry", status: state.progress.collision },
          { label: "Physics", status: state.progress.physics },
          { label: "Safe starting point", status: state.progress.spawn },
        ],
      };
    case "ready":
      return {
        kind: "ready",
        walkable: canWalk(state),
        inspectionReason: state.inspectionReason,
        quality: state.quality,
      };
    case "walking":
      return { kind: "walking", quality: state.quality, pointerLocked: state.pointerLocked };
    case "error":
      return {
        kind: "error",
        message: state.error.message,
        recoverable: state.error.recoverable,
        webglUnavailable: state.error.code === "webgl2-unavailable",
      };
    case "context-lost":
      return { kind: "context-lost" };
    default:
      return { kind: "idle" };
  }
}

function startEngine(
  container: HTMLElement,
  envelope: DeliveryEnvelope,
  quality: QualityChoice,
  ariaLabel: string,
  onView: (view: EngineView) => void,
): EngineAdapter {
  const handle = createViewer({
    container,
    source: { kind: "delivery", envelope },
    quality,
    ariaLabel,
  });
  const unsubscribe = handle.subscribe((state) => onView(toEngineView(state)));
  onView(toEngineView(handle.getState()));
  return {
    enterWalk: () => handle.enterWalk(),
    exitWalk: () => handle.exitWalk(),
    resetPosition: () => handle.resetPosition(),
    setQuality: (q) => handle.setQuality(q),
    setCollisionOverlay: (on) => handle.setCollisionOverlay(on),
    dispose: () => {
      unsubscribe();
      handle.dispose();
    },
  };
}

/* -------------------------------------------------------------------------- */

export interface ViewerCanvasProps {
  envelope: DeliveryEnvelope;
  /** Accessible name of the scene, e.g. the project title. */
  title: string;
  /** Show the collision overlay toggle (private review/walk only). */
  allowCollisionOverlay?: boolean;
  /** Non-3D alternative (e.g. a layout plan) shown on WebGL failure or on request. */
  fallback?: ReactNode;
  /** Called when the engine reports a recoverable failure (e.g. expired asset URLs). */
  onRecoverableError?: () => void;
}

const QUALITY_OPTIONS: { value: QualityChoice; label: string }[] = [
  { value: "auto", label: "Automatic" },
  { value: "low", label: "Low (battery saver)" },
  { value: "mobile", label: "Medium" },
  { value: "desktop", label: "High" },
];

export default function ViewerCanvas({
  envelope,
  title,
  allowCollisionOverlay = false,
  fallback,
  onRecoverableError,
}: ViewerCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<EngineAdapter | null>(null);
  const qualityRef = useRef<QualityChoice>("auto");
  const overlayRef = useRef(false);
  const [view, setView] = useState<EngineView>({ kind: "idle" });
  const [startFailed, setStartFailed] = useState<string | null>(null);
  const [quality, setQuality] = useState<QualityChoice>("auto");
  const [overlay, setOverlay] = useState(false);
  const [prefer2d, setPrefer2d] = useState(false);
  const qualityId = useId();
  const overlayId = useId();
  const helpId = useId();

  // Exactly one engine per mount. Strict Mode mounts, cleans up and mounts
  // again in development; the cleanup fully disposes the first engine.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || prefer2d) return;
    let disposed = false;
    let engine: EngineAdapter | null = null;
    try {
      engine = startEngine(
        container,
        envelope,
        qualityRef.current,
        `3D walkthrough of ${title}`,
        (next) => {
          if (!disposed) setView(next);
        },
      );
      engineRef.current = engine;
      if (overlayRef.current) engine.setCollisionOverlay(true);
    } catch {
      queueMicrotask(() => {
        if (!disposed) setStartFailed("The 3D viewer could not start on this device.");
      });
    }
    return () => {
      disposed = true;
      if (engineRef.current === engine) engineRef.current = null;
      engine?.dispose();
    };
  }, [envelope, title, prefer2d]);

  useEffect(() => {
    if (view.kind === "error" && view.recoverable) onRecoverableError?.();
  }, [view, onRecoverableError]);

  const walkable = view.kind === "ready" && view.walkable;
  const walking = view.kind === "walking";
  const failed = view.kind === "error" || startFailed !== null;
  const showFallback = prefer2d || failed;
  const posterUrl = envelope.assetUrls.poster?.url ?? null;

  function changeQuality(next: QualityChoice) {
    qualityRef.current = next;
    setQuality(next);
    engineRef.current?.setQuality(next);
  }

  function toggleOverlay(on: boolean) {
    overlayRef.current = on;
    setOverlay(on);
    engineRef.current?.setCollisionOverlay(on);
  }

  let statusText = "";
  if (view.kind === "loading") {
    const pct =
      view.renderFraction != null ? ` (${Math.round(view.renderFraction * 100)}% downloaded)` : "";
    statusText = `Loading${pct}: ${view.steps.map((s) => `${s.label} ${s.status}`).join(", ")}.`;
  } else if (view.kind === "ready") {
    statusText = view.walkable
      ? "Ready. Press Walk to start moving."
      : `Inspection only: ${view.inspectionReason ?? "walking is not available for this scene."}`;
  } else if (walking) {
    statusText = view.pointerLocked
      ? "Walking. Press Escape to release the mouse."
      : "Walking. Use the keyboard or touch controls to move.";
  } else if (view.kind === "context-lost") {
    statusText = "The graphics context was lost. Trying to restore it...";
  } else if (failed) {
    statusText = startFailed ?? (view.kind === "error" ? view.message : "");
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {walking ? (
          <Button onClick={() => engineRef.current?.exitWalk()}>Exit walk</Button>
        ) : (
          <Button
            onClick={() => engineRef.current?.enterWalk()}
            disabled={!walkable || showFallback}
            aria-describedby={helpId}
          >
            Walk
          </Button>
        )}
        <Button
          variant="secondary"
          onClick={() => engineRef.current?.resetPosition()}
          disabled={!(walkable || walking) || showFallback}
        >
          Reset position
        </Button>
        <label htmlFor={qualityId} className="text-sm">
          Quality
        </label>
        <select
          id={qualityId}
          value={quality}
          onChange={(e) => changeQuality(e.target.value as QualityChoice)}
          className="rounded-md border border-stone-300 bg-white px-2 py-1.5 text-sm dark:border-stone-700 dark:bg-stone-900"
          disabled={showFallback}
        >
          {QUALITY_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        {allowCollisionOverlay ? (
          <span className="flex items-center gap-1 text-sm">
            <input
              id={overlayId}
              type="checkbox"
              checked={overlay}
              onChange={(e) => toggleOverlay(e.target.checked)}
              disabled={showFallback}
            />
            <label htmlFor={overlayId}>Show collision overlay</label>
          </span>
        ) : null}
        <Button variant="ghost" onClick={() => setPrefer2d((v) => !v)} disabled={failed}>
          {prefer2d ? "Use 3D viewer" : "Show 2D view"}
        </Button>
      </div>

      <p aria-live="polite" role="status" className="text-sm text-stone-700 dark:text-stone-300">
        {statusText}
      </p>

      {showFallback ? (
        <div className="space-y-3 rounded-lg border border-stone-200 p-4 dark:border-stone-800">
          {failed ? (
            <p className="text-sm">
              {view.kind === "error" && view.webglUnavailable
                ? "This browser or device does not support WebGL2, which the 3D viewer needs. "
                : "The 3D viewer is unavailable right now. "}
              You can still see the scene&apos;s preview image and plan below.
            </p>
          ) : null}
          {posterUrl ? (
            // Signed, short-lived storage URL: next/image optimization is not appropriate here.
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={posterUrl}
              alt={`Preview image of ${title}`}
              referrerPolicy="no-referrer"
              className="w-full rounded-md"
            />
          ) : null}
          {fallback}
        </div>
      ) : null}

      <div
        ref={containerRef}
        hidden={showFallback}
        className="relative aspect-video w-full overflow-hidden rounded-lg bg-stone-900"
      />

      <details id={helpId} className="text-sm">
        <summary className="cursor-pointer font-medium">Keyboard and touch controls</summary>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          <li>Press Walk to start. On desktop the mouse is captured to look around.</li>
          <li>W A S D or the arrow keys move; the mouse looks around.</li>
          <li>Escape releases the mouse and stops walking.</li>
          <li>On touch devices, use the left pad to move and drag on the right to look.</li>
          <li>Reset position returns you to the starting point.</li>
          <li>If the scene is slow, choose a lower quality.</li>
        </ul>
      </details>
    </div>
  );
}
