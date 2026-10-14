import type { JobState, Run, RunStatus } from "@/lib/api/types";

/** Canonical user-visible stages (README section 6). */
export const STAGES = [
  { id: "validating", label: "Checking inputs" },
  { id: "preparing", label: "Preparing frames" },
  { id: "camera_reconstruction", label: "Recovering camera positions" },
  { id: "training", label: "Training the visual scene" },
  { id: "normalizing", label: "Cleaning and normalizing" },
  { id: "geometry_proposal", label: "Proposing walkable geometry" },
  { id: "packaging", label: "Packaging the scene" },
  { id: "scene_validation", label: "Validating the scene" },
] as const;

export type StageView = "done" | "active" | "waiting" | "failed" | "cancelled" | "pending";

export const TERMINAL_RUN_STATUSES: readonly RunStatus[] = ["succeeded", "failed", "cancelled"];
export const TERMINAL_JOB_STATES: readonly JobState[] = ["succeeded", "failed", "cancelled"];

export function isTerminalRun(status: RunStatus | undefined | null): boolean {
  return status != null && TERMINAL_RUN_STATUSES.includes(status);
}

export function stageLabel(stage: string): string {
  return STAGES.find((s) => s.id === stage)?.label ?? stage.replace(/_/g, " ");
}

/** Per-stage display state from the run's job summaries (latest job per stage wins). */
export function stageStates(run: Run | null): Record<string, StageView> {
  const out: Record<string, StageView> = {};
  for (const s of STAGES) out[s.id] = "pending";
  if (!run) return out;
  for (const job of run.jobs) {
    if (job.state === "succeeded") out[job.stage] = "done";
    else if (job.state === "running") out[job.stage] = "active";
    else if (job.state === "queued" || job.state === "retry_wait") out[job.stage] = "waiting";
    else if (job.state === "failed") out[job.stage] = "failed";
    else if (job.state === "cancelled") out[job.stage] = "cancelled";
  }
  return out;
}

/**
 * Formats stage-local progress if the worker reported it. Returns null when
 * nothing reliable is known (we never invent a percentage or ETA).
 */
export function formatStageProgress(progress: Record<string, unknown> | null | undefined): string | null {
  if (!progress) return null;
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const current = num(progress.current);
  const total = num(progress.total);
  const unit = typeof progress.unit === "string" ? ` ${progress.unit}` : "";
  if (current !== null && total !== null && total > 0) {
    return `${current} of ${total}${unit}`;
  }
  const fraction = num(progress.fraction);
  if (fraction !== null && fraction >= 0 && fraction <= 1) {
    return `${Math.round(fraction * 100)}% of this stage`;
  }
  return null;
}

export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h} h ${m} min`;
  if (m > 0) return `${m} min ${s} s`;
  return `${s} s`;
}
