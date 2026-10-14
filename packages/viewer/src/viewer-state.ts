import type { QualityProfileName } from "./render/quality";

/**
 * Typed viewer state machine:
 *
 *   idle -> loading -> ready(inspection | walkable) <-> walking
 *   any non-terminal -> error | context-lost
 *   context-lost -> loading (render rebuilt after restore)
 *   any -> disposed (terminal)
 *
 * Walking is only reachable from ready(walkable), which the engine enters only
 * after render loading, collision loading/validation, Rapier initialization and
 * spawn clearance all succeed.
 */

export type ViewerMode = "inspection" | "walkable";

export type ViewerErrorCode =
  | "webgl2-unavailable"
  | "invalid-input"
  | "render-load-failed"
  | "collision-load-failed"
  | "context-lost"
  | "init-failed";

export interface ViewerError {
  code: ViewerErrorCode;
  /** Safe, user-presentable message (no URLs or stack traces). */
  message: string;
  /** True when retrying (e.g. with a fresh delivery envelope) may succeed. */
  recoverable: boolean;
}

export type StepStatus = "pending" | "done" | "failed" | "skipped";

export interface LoadProgress {
  render: StepStatus;
  /** Fraction of the render asset downloaded, when known. */
  renderFraction: number | null;
  collision: StepStatus;
  physics: StepStatus;
  spawn: StepStatus;
}

export type ViewerState =
  | { status: "idle" }
  | { status: "loading"; progress: LoadProgress }
  | {
      status: "ready";
      mode: ViewerMode;
      /** Why walking is unavailable (inspection mode only). */
      inspectionReason: string | null;
      quality: QualityProfileName;
    }
  | { status: "walking"; quality: QualityProfileName; pointerLocked: boolean }
  | { status: "error"; error: ViewerError }
  | { status: "context-lost"; resumeMode: ViewerMode; inspectionReason: string | null }
  | { status: "disposed" };

export type ViewerStatus = ViewerState["status"];

export type ViewerEvent =
  | { type: "load-start" }
  | { type: "progress"; progress: Partial<LoadProgress> }
  | {
      type: "loaded";
      mode: ViewerMode;
      inspectionReason: string | null;
      quality: QualityProfileName;
    }
  | { type: "enter-walk"; pointerLocked: boolean }
  | { type: "exit-walk" }
  | { type: "pointer-lock"; locked: boolean }
  | { type: "quality"; quality: QualityProfileName }
  | { type: "fail"; error: ViewerError }
  | { type: "context-lost" }
  | { type: "dispose" };

export function initialLoadProgress(): LoadProgress {
  return {
    render: "pending",
    renderFraction: null,
    collision: "pending",
    physics: "pending",
    spawn: "pending",
  };
}

/** Pure transition function. Invalid events return the same state object. */
export function transition(state: ViewerState, event: ViewerEvent): ViewerState {
  if (state.status === "disposed") return state;
  if (event.type === "dispose") return { status: "disposed" };

  switch (event.type) {
    case "load-start":
      if (state.status === "idle" || state.status === "context-lost" || state.status === "error") {
        return { status: "loading", progress: initialLoadProgress() };
      }
      return state;
    case "progress":
      if (state.status !== "loading") return state;
      return { status: "loading", progress: { ...state.progress, ...event.progress } };
    case "loaded":
      if (state.status !== "loading") return state;
      return {
        status: "ready",
        mode: event.mode,
        inspectionReason: event.mode === "inspection" ? event.inspectionReason : null,
        quality: event.quality,
      };
    case "enter-walk":
      if (state.status === "ready" && state.mode === "walkable") {
        return { status: "walking", quality: state.quality, pointerLocked: event.pointerLocked };
      }
      return state;
    case "exit-walk":
      if (state.status === "walking") {
        return {
          status: "ready",
          mode: "walkable",
          inspectionReason: null,
          quality: state.quality,
        };
      }
      return state;
    case "pointer-lock":
      if (state.status === "walking" && state.pointerLocked !== event.locked) {
        return { ...state, pointerLocked: event.locked };
      }
      return state;
    case "quality":
      if (
        (state.status === "ready" || state.status === "walking") &&
        state.quality !== event.quality
      ) {
        return { ...state, quality: event.quality };
      }
      return state;
    case "fail":
      return { status: "error", error: event.error };
    case "context-lost":
      if (state.status === "ready") {
        return {
          status: "context-lost",
          resumeMode: state.mode,
          inspectionReason: state.inspectionReason,
        };
      }
      if (state.status === "walking") {
        return { status: "context-lost", resumeMode: "walkable", inspectionReason: null };
      }
      if (state.status === "loading") {
        return { status: "context-lost", resumeMode: "inspection", inspectionReason: null };
      }
      return state;
  }
}

/** True when the Walk control should be enabled. */
export function canWalk(state: ViewerState): boolean {
  return state.status === "ready" && state.mode === "walkable";
}
