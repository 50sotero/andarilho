import { describe, expect, it } from "vitest";
import { safeNextPath } from "@/lib/auth/safe-redirect";
import {
  calibrationFactor,
  isSelfIntersecting,
  layoutProblems,
  scaleLayout,
} from "@/features/review/geometry";
import { addOpening, movePoint, removeSelected } from "@/features/review/layout-edit";
import { revisionPending } from "@/features/review/ReviewEditor";
import {
  canRetryRun,
  formatStageProgress,
  nextEventCursor,
  runActions,
  STAGES,
  stageLabel,
  stageStates,
} from "@/features/processing/stages";
import { expiryFromDays } from "@/features/sharing/SharePanel";
import { awaitingRevision, awaitingScene, fallbackLayout } from "@/features/projects/PrivateWalk";
import { removalMode, startReadiness } from "@/features/capture/CaptureFlow";
import { rejectionMessage } from "@/features/uploads/UploadList";
import { RefreshBudget, envelopeExpired } from "@/components/viewer/envelope";
import type { DeliveryEnvelope, Layout, Run, SceneDetail } from "@/lib/api/types";
import fixture from "@andarilho/scene-contract/fixtures/synthetic-two-room.layout.json";

const layout = fixture as unknown as Layout;

describe("safeNextPath", () => {
  it("keeps same-origin relative paths", () => {
    expect(safeNextPath("/projects/abc?x=1#h")).toBe("/projects/abc?x=1#h");
  });
  it("rejects open-redirect attempts", () => {
    for (const bad of [
      "https://evil.test",
      "//evil.test",
      "/\\evil.test",
      "javascript:alert(1)",
      "evil",
      "/\u0000x",
      null,
      "",
    ]) {
      expect(safeNextPath(bad as string | null)).toBe("/projects");
    }
  });
});

describe("layout geometry", () => {
  it("accepts the synthetic fixture", () => {
    expect(layoutProblems(layout)).toEqual([]);
  });
  it("detects self-intersection", () => {
    expect(
      isSelfIntersecting([
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 1],
      ]),
    ).toBe(true);
    expect(
      isSelfIntersecting([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ]),
    ).toBe(false);
  });
  it("flags a spawn outside the floor and a bad wall", () => {
    let edited = movePoint(layout, { kind: "spawn" }, [100, 100]);
    edited = {
      ...edited,
      walls: edited.walls!.map((w, i) => (i === 0 ? { ...w, thickness: -1 } : w)),
    };
    const messages = layoutProblems(edited)
      .map((p) => p.message)
      .join(" ");
    expect(messages).toMatch(/spawn/i);
    expect(messages).toMatch(/thickness/i);
  });
  it("calibrates uniformly", () => {
    expect(calibrationFactor(2, 3)).toBe(1.5);
    expect(calibrationFactor(0, 3)).toBeNull();
    const scaled = scaleLayout(layout, 2);
    expect(scaled.floors[0]!.polygons[0]![1]).toEqual([8, -5]);
    expect(scaled.walls![0]!.thickness).toBeCloseTo(0.2);
    expect(scaled.walls![0]!.height).toBeCloseTo(layout.walls![0]!.height * 2);
    expect(scaled.calibration).toEqual({ method: "known_distance", renderScale: 2 });
    // Calibration is cumulative and absolute, so the camera poses are never rescaled twice.
    expect(scaleLayout(scaled, 1.5).calibration?.renderScale).toBeCloseTo(3);
  });
  it("adds openings inside walls and removes a wall's openings with it", () => {
    const withOpening = addOpening(layout, 0);
    expect(layoutProblems(withOpening)).toEqual([]);
    const removed = removeSelected(withOpening, { kind: "wall", index: 0 });
    expect(removed.openings!.some((o) => o.wallId === layout.walls![0]!.id)).toBe(false);
  });
});

describe("run retry offer", () => {
  it("offers retry for infrastructure failures but not for bad captures", () => {
    const failed = (code: string) => ({
      status: "failed" as const,
      error: { code, message: "", retryable: false, suggestedAction: null, requestId: "r" },
    });
    expect(canRetryRun(failed("TRANSIENT_FAILURE"))).toBe(true);
    expect(canRetryRun(failed("WORKER_LOST"))).toBe(true);
    expect(canRetryRun(failed("DISCONNECTED_CAPTURE"))).toBe(false);
    expect(canRetryRun({ status: "cancelled", error: null })).toBe(true);
    expect(canRetryRun({ status: "succeeded", error: null })).toBe(false);
  });

  it("does not retry a scene made before the current pipeline", () => {
    const outdated = {
      status: "failed" as const,
      error: {
        code: "SCENE_OUTDATED",
        message: "",
        retryable: false,
        suggestedAction: null,
        requestId: "r",
      },
    };
    expect(canRetryRun(outdated)).toBe(false);
    expect(runActions({ ...outdated, cancelRequested: false }, "owner")).toMatchObject({
      retry: false,
      newCapture: true,
    });
  });
});

describe("run actions", () => {
  const error = {
    code: "TRANSIENT_FAILURE",
    message: "",
    retryable: false,
    suggestedAction: null,
    requestId: "r",
  };
  const running = { status: "running" as const, error: null, cancelRequested: false };
  const failed = { status: "failed" as const, error, cancelRequested: false };
  const succeeded = { status: "succeeded" as const, error: null, cancelRequested: false };

  it("gives owners and editors the run controls", () => {
    for (const role of ["owner", "editor"] as const) {
      expect(runActions(running, role)).toMatchObject({ cancel: true, retry: false });
      expect(runActions(failed, role)).toMatchObject({ retry: true, newCapture: true });
      expect(runActions(succeeded, role)).toMatchObject({ review: true, cancel: false });
    }
    expect(runActions({ ...running, cancelRequested: true }, "owner").cancel).toBe(false);
  });

  it("shows read-only members no action the API would refuse", () => {
    const none = { cancel: false, retry: false, newCapture: false, review: false };
    for (const run of [running, failed, succeeded]) {
      expect(runActions(run, "viewer")).toEqual(none);
      expect(runActions(run, undefined)).toEqual(none);
    }
  });
});

describe("processing helpers", () => {
  it("never invents progress", () => {
    expect(formatStageProgress(null)).toBeNull();
    expect(formatStageProgress({ message: "x" })).toBeNull();
    expect(formatStageProgress({ current: 3, total: 10, unit: "frames" })).toBe("3 of 10 frames");
    expect(formatStageProgress({ fraction: 0.25 })).toBe("25% of this stage");
  });
  it("derives stage states from jobs", () => {
    const run = {
      jobs: [
        { id: "1", stage: "validating", state: "succeeded" },
        { id: "2", stage: "preparing", state: "running" },
      ],
    } as unknown as Run;
    const states = stageStates(run);
    expect(states.validating).toBe("done");
    expect(states.preparing).toBe("active");
    expect(states.texturing).toBe("pending");
  });

  it("lists the stages in pipeline order, with texturing after the proposal", () => {
    expect(STAGES.map((s) => s.id)).toEqual([
      "validating",
      "preparing",
      "camera_reconstruction",
      "normalizing",
      "geometry_proposal",
      "texturing",
      "packaging",
      "scene_validation",
    ]);
    expect(stageLabel("texturing")).toBe("Texturing the model from the video");
  });
});

describe("share expiry", () => {
  it("computes ISO expiry or null", () => {
    expect(expiryFromDays("", 0)).toBeNull();
    expect(expiryFromDays("1", 0)).toBe(new Date(86_400_000).toISOString());
  });
});

describe("private walk fallback plan", () => {
  const layout = fixture as unknown as Layout;
  const scene = { revision: { id: "rev-2", layout } } as unknown as SceneDetail;
  const envelopeFor = (id: string) => ({ sceneRevisionId: id }) as unknown as DeliveryEnvelope;

  it("shows the draft plan only when the draft is the walked revision", () => {
    expect(fallbackLayout(scene, envelopeFor("rev-2"))).toBe(layout);
    expect(fallbackLayout(scene, envelopeFor("rev-1"))).toBeNull();
    expect(fallbackLayout(scene, null)).toBe(layout);
    expect(fallbackLayout(null, envelopeFor("rev-1"))).toBeNull();
  });
});

describe("private walk while the first revision validates", () => {
  const rev = (id: string, validationState: string, extra: object = {}) => ({
    id,
    revision: Number(id.slice(4)),
    validationState,
    ...extra,
  });
  const sceneWith = (current: object | null, revisions: object[]) =>
    ({ revision: current, revisions }) as unknown as SceneDetail;

  it("keeps checking while a revision is pending or none exists yet", () => {
    const pending = rev("rev-1", "pending");
    expect(awaitingRevision(sceneWith(pending, [pending]))).toBe(true);
    expect(awaitingRevision(sceneWith(null, []))).toBe(true);
  });

  it("stops when a revision is walkable or nothing is pending", () => {
    const valid = rev("rev-1", "valid", { manifest: {} });
    expect(awaitingRevision(sceneWith(valid, [valid]))).toBe(false);
    const invalid = rev("rev-1", "invalid");
    expect(awaitingRevision(sceneWith(invalid, [invalid]))).toBe(false);
  });
});

describe("private walk before the run creates the scene", () => {
  const scene = { id: "scene-1", version: 1, currentDraftRevisionId: null };
  const runs = (...statuses: Run["status"][]) => statuses.map((status) => ({ status }));

  it("keeps checking while a run is still processing", () => {
    expect(awaitingScene({ scene: null, runs: runs("failed", "running") })).toBe(true);
    expect(awaitingScene({ scene: null, runs: runs("queued") })).toBe(true);
  });

  it("stops once the scene exists or no run is processing", () => {
    expect(awaitingScene({ scene, runs: runs("running") })).toBe(false);
    expect(awaitingScene({ scene: null, runs: runs("failed", "cancelled") })).toBe(false);
    expect(awaitingScene(null)).toBe(false);
  });
});

describe("review page while a revision validates", () => {
  const rev = (validationState: string) => ({ id: "rev-1", revision: 1, validationState });
  const sceneWith = (current: object | null, revisions: object[]) =>
    ({ revision: current, revisions }) as unknown as SceneDetail;

  it("waits for a first revision that validation has not made current yet", () => {
    expect(revisionPending(sceneWith(null, [rev("pending")]))).toBe(true);
    expect(revisionPending(sceneWith(rev("pending"), [rev("pending")]))).toBe(true);
  });

  it("stops once validation finished or nothing is pending", () => {
    expect(revisionPending(sceneWith(rev("valid"), [rev("valid")]))).toBe(false);
    expect(revisionPending(sceneWith(null, [rev("invalid")]))).toBe(false);
    expect(revisionPending(sceneWith(null, []))).toBe(false);
    expect(revisionPending(null)).toBe(false);
  });
});

describe("upload rejection message", () => {
  it("tells the user to re-upload when our check failed, not that the file is bad", () => {
    expect(rejectionMessage("INSPECTION_FAILED")).toMatch(/upload it again/);
    expect(rejectionMessage("INSPECTION_FAILED")).not.toMatch(/undamaged/);
    expect(rejectionMessage("UNSUPPORTED_MEDIA")).toMatch(/UNSUPPORTED_MEDIA.*undamaged/);
  });
});

describe("upload row removal", () => {
  it("waits for the server while a file is being checked, so a refusal keeps the row", () => {
    expect(removalMode("uploaded_unverified")).toBe("confirm");
    expect(removalMode("verified")).toBe("release-verified");
    expect(removalMode("initiated")).toBe("best-effort");
    expect(removalMode("rejected")).toBe("best-effort");
    expect(removalMode(null)).toBe("best-effort");
  });

  it("waits for the server while completion may be turning the upload into a checked file", () => {
    expect(removalMode(null, "completing")).toBe("confirm");
    expect(removalMode(null, "completed")).toBe("confirm");
    expect(removalMode(null, "uploading")).toBe("best-effort");
    expect(removalMode("rejected", "completed")).toBe("best-effort");
  });
});

describe("delivery envelope expiry", () => {
  const now = Date.parse("2026-10-15T12:00:00Z");
  it("treats links at or past their expiry as expired", () => {
    expect(envelopeExpired({ expiresAt: "2026-10-15T12:05:00Z" }, now)).toBe(false);
    expect(envelopeExpired({ expiresAt: "2026-10-15T12:00:00Z" }, now)).toBe(true);
    expect(envelopeExpired({ expiresAt: "2026-10-15T11:59:00Z" }, now)).toBe(true);
  });
  it("treats an unparseable expiry as expired", () => {
    expect(envelopeExpired({ expiresAt: "soon" }, now)).toBe(true);
  });
});

describe("envelope refresh budget", () => {
  it("stops a failure that keeps recurring", () => {
    const budget = new RefreshBudget(2);
    expect([budget.take(), budget.take(), budget.take()]).toEqual([true, true, false]);
  });
  it("starts over after a successful load, so a later failure can refresh again", () => {
    const budget = new RefreshBudget(2);
    budget.take();
    budget.take();
    budget.reset();
    expect([budget.take(), budget.take(), budget.take()]).toEqual([true, true, false]);
  });
});

describe("starting a run from the capture page", () => {
  const ready = {
    submitting: false,
    selectedNotUploaded: 0,
    allRowsSettled: true,
    inputCount: 40,
    hasFloorPlan: false,
    blockingErrors: 0,
  };
  it("starts when every input is uploaded and verified", () => {
    expect(startReadiness(ready)).toEqual({
      canSubmit: true,
      hint: "40 input file(s) will be used.",
    });
  });
  it("waits for files that are selected but not uploaded, even with enough inputs", () => {
    const pending = startReadiness({ ...ready, selectedNotUploaded: 2 });
    expect(pending.canSubmit).toBe(false);
    expect(pending.hint).toMatch(/Upload or clear the 2 selected file\(s\) first/);
  });
  it("waits for verification and blocking errors", () => {
    expect(startReadiness({ ...ready, allRowsSettled: false }).canSubmit).toBe(false);
    expect(startReadiness({ ...ready, blockingErrors: 1 }).canSubmit).toBe(false);
    expect(startReadiness({ ...ready, inputCount: 0 }).canSubmit).toBe(false);
  });
});

describe("nextEventCursor", () => {
  it("follows the API's cursor while more pages remain", () => {
    expect(nextEventCursor(null, [{ id: 1 }, { id: 2 }], 2)).toBe(2);
  });

  it("moves past the last event when the history fits in one page", () => {
    // nextCursor is null for a short history; refetching it every poll would be wasted.
    expect(nextEventCursor(null, [{ id: 4 }, { id: 9 }, { id: 7 }], null)).toBe(9);
    expect(nextEventCursor(9, [{ id: 10 }], null)).toBe(10);
  });

  it("keeps the cursor when nothing new arrived", () => {
    expect(nextEventCursor(9, [], null)).toBe(9);
    expect(nextEventCursor(null, [], null)).toBeNull();
  });
});
