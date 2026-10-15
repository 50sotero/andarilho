"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { Button, Card, ErrorNotice, inputClass } from "@/components/ui";
import { useProject } from "@/features/projects/use-project";
import { createLimiter } from "@/features/uploads/limiter";
import {
  DEFAULT_CONCURRENCY,
  MultipartUpload,
  UploadAbortedError,
  fetchPutPart,
  type UploadProgress,
} from "@/features/uploads/uploader";
import { UploadList, type UploadRowView } from "@/features/uploads/UploadList";
import { api, newIdempotencyKey } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { IntakePolicy, MediaKind, RunQuality, Upload } from "@/lib/api/types";
import {
  loadResumeRecord,
  pruneResumeRecords,
  removeResumeRecord,
  uploadIdentity,
} from "@/lib/upload-resume";
import { usePolling } from "@/lib/use-polling";
import { CaptureGuidance } from "./CaptureGuidance";
import { SceneAnalysisNotice, sceneAnalysisNoticeShown } from "./SceneAnalysisNotice";
import {
  ACCEPT,
  DEFAULT_INTAKE_POLICY,
  declaredContentType,
  formatBytes,
  PER_FILE_CODES,
  validateIntake,
  type CandidateFile,
} from "./validation";

const KIND_LABEL: Record<MediaKind, string> = {
  video: "Video",
  photo: "Photo",
  floor_plan: "Floor plan",
};

/** Files uploading at once; parts are additionally capped globally at 3 transfers. */
const FILE_CONCURRENCY = 3;

interface Row extends UploadRowView {
  kind: MediaKind;
  uploadId: string | null;
}

function readVideoDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    const done = (value: number | null) => {
      URL.revokeObjectURL(url);
      video.removeAttribute("src");
      resolve(value);
    };
    video.onloadedmetadata = () => done(Number.isFinite(video.duration) ? video.duration : null);
    video.onerror = () => done(null);
    video.src = url;
  });
}

/**
 * How removing an upload row involves the server. A verified file is released
 * through its own flow; while a file is being checked the server may refuse
 * (409), so the row stays until it confirms. That includes a completion still in
 * flight or just finished, before the row has seen the new status: if completion
 * wins, the file is being checked. An unfinished or rejected upload is aborted
 * best-effort (expiry and cleanup reconcile it); a row the server never saw is only
 * dropped locally.
 */
export function removalMode(
  serverStatus: UploadRowView["serverStatus"],
  phase: UploadRowView["progress"]["phase"] | null = null,
): "release-verified" | "confirm" | "best-effort" | "local" {
  if (serverStatus === "verified") return "release-verified";
  if (serverStatus === "uploaded_unverified") return "confirm";
  if (serverStatus === null && (phase === "completing" || phase === "completed")) {
    return "confirm";
  }
  return "best-effort";
}

/**
 * Whether a run can start, and the line explaining what it will use or what is
 * missing. Files still in the selection are not uploads yet: a run started now
 * would silently leave them out, so they must be uploaded (or cleared) first.
 */
export function startReadiness(state: {
  submitting: boolean;
  selectedNotUploaded: number;
  allRowsSettled: boolean;
  inputCount: number;
  hasFloorPlan: boolean;
  blockingErrors: number;
}): { canSubmit: boolean; hint: string } {
  const canSubmit =
    !state.submitting &&
    state.selectedNotUploaded === 0 &&
    state.allRowsSettled &&
    state.inputCount > 0 &&
    state.blockingErrors === 0;
  let hint: string;
  if (state.selectedNotUploaded > 0) {
    hint = `Upload or clear the ${state.selectedNotUploaded} selected file(s) first; they are not part of a run until they are uploaded and verified.`;
  } else if (state.inputCount === 0) {
    hint = "Upload a video or photos, and wait until they are verified.";
  } else if (!state.allRowsSettled) {
    hint = "Waiting for all uploads to finish and be verified. Remove rejected files to continue.";
  } else {
    hint = `${state.inputCount} input file(s)${state.hasFloorPlan ? " and a floor plan" : ""} will be used.`;
  }
  return { canSubmit, hint };
}

function initialProgress(size: number): UploadProgress {
  return {
    phase: "idle",
    uploadedBytes: 0,
    totalBytes: size,
    completedParts: 0,
    partCount: 0,
    resumed: false,
    uploadId: null,
    error: null,
  };
}

export function CaptureFlow({ projectId }: { projectId: string }) {
  const router = useRouter();
  const ids = { video: useId(), photos: useId(), plan: useId(), quality: useId(), supp: useId() };
  const { detail, error: projectError, reload } = useProject(projectId);

  const [video, setVideo] = useState<File | null>(null);
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [photos, setPhotos] = useState<File[]>([]);
  const [plan, setPlan] = useState<File | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [includePrevious, setIncludePrevious] = useState<Set<string>>(new Set());
  const [quality, setQuality] = useState<RunQuality>("preview");
  const [includeSupplementary, setIncludeSupplementary] = useState(true);
  const [error, setError] = useState<FriendlyError | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const [policy, setPolicy] = useState<IntakePolicy>(DEFAULT_INTAKE_POLICY);

  const uploadersRef = useRef(new Map<string, MultipartUpload>());
  const limiterRef = useRef(createLimiter(DEFAULT_CONCURRENCY));
  const intentRef = useRef<{ key: string; signature: string; captureSetId: string | null } | null>(
    null,
  );
  const rowSeq = useRef(0);

  useEffect(() => {
    // The deployment's limits; until they arrive (or if they cannot be read) the
    // API's defaults apply, and the API enforces its own values either way.
    let active = true;
    api
      .getIntakePolicy()
      .then(({ data }) => {
        if (active) setPolicy(data);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    pruneResumeRecords();
    const uploaders = uploadersRef.current;
    return () => {
      for (const uploader of uploaders.values()) uploader.pause();
    };
  }, []);

  const updateRow = useCallback((id: string, patch: Partial<Row>) => {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  }, []);

  const uploading = rows.some((r) =>
    ["initializing", "uploading", "completing"].includes(r.progress.phase),
  );

  useEffect(() => {
    if (!uploading) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [uploading]);

  // Previously uploaded, verified files (e.g. from before a reload) that can be reused.
  const rowUploadIds = new Set(rows.map((r) => r.uploadId).filter(Boolean));
  const previousVerified: Upload[] = (detail?.uploads ?? []).filter(
    (u) => u.status === "verified" && !rowUploadIds.has(u.id),
  );

  const candidates: CandidateFile[] = useMemo(() => {
    const list: CandidateFile[] = [];
    if (video) {
      list.push({
        name: video.name,
        size: video.size,
        type: video.type,
        mediaKind: "video",
        durationSeconds: videoDuration,
      });
    }
    for (const p of photos)
      list.push({ name: p.name, size: p.size, type: p.type, mediaKind: "photo" });
    if (plan)
      list.push({ name: plan.name, size: plan.size, type: plan.type, mediaKind: "floor_plan" });
    return list;
  }, [video, videoDuration, photos, plan]);

  const allForValidation: CandidateFile[] = [
    ...candidates,
    ...rows.map((r) => ({
      name: r.name,
      size: r.progress.totalBytes,
      type: "application/octet-stream",
      mediaKind: r.kind,
    })),
    ...previousVerified
      .filter((u) => includePrevious.has(u.id))
      .map((u) => ({
        name: u.filename,
        size: u.expectedBytes,
        type: u.contentType,
        mediaKind: u.mediaKind,
      })),
  ];

  // Per-file checks only apply to newly selected files; rows were already validated.
  const validation = validateIntake(allForValidation, policy);
  const selectionValidation = validateIntake(candidates, policy);
  const typeErrors = selectionValidation.errors.filter((e) => PER_FILE_CODES.includes(e.code));
  const aggregateErrors = validation.errors.filter((e) => !PER_FILE_CODES.includes(e.code));
  const blockingErrors = [...typeErrors, ...aggregateErrors];

  const resumableCount = useMemo(() => {
    const chosen: [File | null, MediaKind][] = [
      [video, "video"],
      ...photos.map((f): [File, MediaKind] => [f, "photo"]),
      [plan, "floor_plan"],
    ];
    return chosen.filter(([file, kind]) => {
      if (!file) return false;
      const identity = uploadIdentity(file, kind, declaredContentType(file, kind));
      return loadResumeRecord(projectId, identity) !== null;
    }).length;
  }, [projectId, video, photos, plan]);

  async function onVideoChange(file: File | null) {
    setVideo(file);
    setVideoDuration(null);
    if (file) setVideoDuration(await readVideoDuration(file));
  }

  function startUploader(id: string) {
    const uploader = uploadersRef.current.get(id);
    if (!uploader) return Promise.resolve();
    return uploader
      .start()
      .then((upload) => {
        updateRow(id, {
          uploadId: upload.id,
          serverStatus: upload.status,
          rejectionCode: upload.rejectionCode,
        });
      })
      .catch((err: unknown) => {
        if (err instanceof UploadAbortedError) return;
        setError(toFriendlyError(err));
      });
  }

  async function runQueue(idsToStart: string[]) {
    const pending = [...idsToStart];
    const worker = async () => {
      while (pending.length > 0) {
        const id = pending.shift()!;
        await startUploader(id);
      }
    };
    await Promise.all(Array.from({ length: FILE_CONCURRENCY }, () => worker()));
    setAnnouncement("Uploads finished. Files are being checked.");
  }

  function beginUpload() {
    setError(null);
    if (blockingErrors.length > 0 || candidates.length === 0) return;
    const newRows: Row[] = [];
    const add = (file: File, kind: MediaKind) => {
      rowSeq.current += 1;
      const id = `row-${rowSeq.current}`;
      const limiter = limiterRef.current;
      const uploader = new MultipartUpload(projectId, file, kind, declaredContentType(file, kind), {
        api,
        putPart: (url, body, signal) => limiter.run(() => fetchPutPart(url, body, signal)),
      });
      uploader.subscribe((progress) => {
        updateRow(id, { progress, uploadId: progress.uploadId });
      });
      uploadersRef.current.set(id, uploader);
      newRows.push({
        id,
        name: file.name,
        kind,
        kindLabel: KIND_LABEL[kind],
        progress: initialProgress(file.size),
        serverStatus: null,
        rejectionCode: null,
        uploadId: null,
      });
    };
    if (video) add(video, "video");
    for (const p of photos) add(p, "photo");
    if (plan) add(plan, "floor_plan");
    setRows((prev) => [...prev, ...newRows]);
    setVideo(null);
    setVideoDuration(null);
    setPhotos([]);
    setPlan(null);
    setAnnouncement(`Uploading ${newRows.length} file(s).`);
    void runQueue(newRows.map((r) => r.id));
  }

  function pauseRow(id: string) {
    uploadersRef.current.get(id)?.pause();
  }

  function resumeRow(id: string) {
    void startUploader(id);
  }

  function pauseAll() {
    for (const uploader of uploadersRef.current.values()) uploader.pause();
    setAnnouncement("All uploads paused.");
  }

  function resumeAll() {
    const ids = rows
      .filter((r) => r.progress.phase === "paused" || r.progress.phase === "failed")
      .map((r) => r.id);
    setAnnouncement("Resuming uploads.");
    void runQueue(ids);
  }

  /** Removes a verified file from the project, releasing its quota. False if refused. */
  async function removeVerified(uploadId: string, name: string): Promise<boolean> {
    setError(null);
    try {
      await api.abortUpload(uploadId);
    } catch (err) {
      // e.g. the file backs a reconstruction and must stay.
      setError(toFriendlyError(err));
      return false;
    }
    setIncludePrevious((prev) => {
      const next = new Set(prev);
      next.delete(uploadId);
      return next;
    });
    setAnnouncement(`${name} removed from the project.`);
    void reload();
    return true;
  }

  async function removeRow(id: string) {
    const uploader = uploadersRef.current.get(id);
    const row = rows.find((r) => r.id === id);
    const uploadId = row?.uploadId ?? uploader?.state.uploadId;
    const mode = uploadId
      ? removalMode(row?.serverStatus ?? null, row?.progress.phase ?? null)
      : "local";
    if (mode === "release-verified") {
      if (!(await removeVerified(uploadId!, row!.name))) return;
    } else if (mode === "confirm") {
      setError(null);
      try {
        await api.abortUpload(uploadId!);
      } catch (err) {
        // e.g. 409 while the file is still being checked: keep the row (and its polling).
        setError(toFriendlyError(err));
        return;
      }
    } else if (mode === "best-effort") {
      api.abortUpload(uploadId!).catch(() => undefined);
    }
    uploader?.pause();
    if (uploader) removeResumeRecord(projectId, uploader.fingerprint);
    uploadersRef.current.delete(id);
    setRows((prev) => prev.filter((r) => r.id !== id));
    if (row?.serverStatus !== "verified") setAnnouncement(`${row?.name ?? "File"} removed.`);
  }

  // Poll upload verification after storage completion.
  const awaitingVerification = rows.some((r) => r.serverStatus === "uploaded_unverified");
  usePolling(
    async (signal) => {
      const pending = rows.filter((r) => r.serverStatus === "uploaded_unverified" && r.uploadId);
      for (const row of pending) {
        const { data } = await api.getUpload(row.uploadId!, signal);
        updateRow(row.id, {
          serverStatus: data.upload.status,
          rejectionCode: data.upload.rejectionCode,
        });
      }
      return { terminal: false };
    },
    { enabled: awaitingVerification, key: "verification" },
  );

  const verifiedRows = rows.filter((r) => r.serverStatus === "verified");
  const selectedPrevious = previousVerified.filter((u) => includePrevious.has(u.id));
  const inputUploadIds = [
    ...verifiedRows.filter((r) => r.kind !== "floor_plan").map((r) => r.uploadId!),
    ...selectedPrevious.filter((u) => u.mediaKind !== "floor_plan").map((u) => u.id),
  ];
  const floorPlanUploadId =
    verifiedRows.find((r) => r.kind === "floor_plan")?.uploadId ??
    selectedPrevious.find((u) => u.mediaKind === "floor_plan")?.id ??
    null;
  const allRowsSettled = rows.every((r) => r.serverStatus === "verified");
  const { canSubmit, hint: startHint } = startReadiness({
    submitting,
    selectedNotUploaded: candidates.length,
    allRowsSettled,
    inputCount: inputUploadIds.length,
    hasFloorPlan: floorPlanUploadId !== null,
    blockingErrors: aggregateErrors.length,
  });

  async function submit() {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    const signature = JSON.stringify({
      inputUploadIds: [...inputUploadIds].sort(),
      floorPlanUploadId,
      quality,
      includeSupplementary,
    });
    // One Idempotency-Key per submit intent; retries of the same intent reuse it.
    if (!intentRef.current || intentRef.current.signature !== signature) {
      intentRef.current = { key: newIdempotencyKey(), signature, captureSetId: null };
    }
    const intent = intentRef.current;
    try {
      if (!intent.captureSetId) {
        const { data } = await api.createCaptureSet(projectId, {
          uploadIds: inputUploadIds,
          ...(floorPlanUploadId ? { floorPlanUploadId } : {}),
        });
        intent.captureSetId = data.id;
      }
      const { data: run } = await api.submitRun(
        projectId,
        {
          captureSetId: intent.captureSetId,
          quality,
          includeSupplementaryPhotos: includeSupplementary,
        },
        intent.key,
      );
      router.push(
        `/projects/${encodeURIComponent(projectId)}/processing?job=${encodeURIComponent(run.jobId)}`,
      );
    } catch (err) {
      setError(toFriendlyError(err));
      setSubmitting(false);
    }
  }

  const deleting = detail?.project.lifecycle && detail.project.lifecycle !== "active";

  return (
    <div className="space-y-6">
      <p aria-live="polite" className="sr-only">
        {announcement}
      </p>

      <Card>
        <details>
          <summary className="cursor-pointer text-sm font-medium">
            Capture guidance (read before recording)
          </summary>
          <div className="mt-3">
            <CaptureGuidance headingLevel={3} policy={policy} />
          </div>
        </details>
      </Card>

      <ErrorNotice error={projectError} onRetry={() => void reload()} />
      {deleting ? (
        <p role="alert" className="text-sm text-red-700">
          This project is being deleted; no new uploads are accepted.
        </p>
      ) : null}

      <Card>
        <h2 className="text-lg font-semibold">1. Choose files</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          <div className="space-y-1">
            <label htmlFor={ids.video} className="block text-sm font-medium">
              Walkthrough video (recommended)
            </label>
            <input
              id={ids.video}
              type="file"
              accept={ACCEPT.video}
              onChange={(e) => void onVideoChange(e.target.files?.[0] ?? null)}
              className="block w-full text-sm"
            />
            <p className="text-xs text-stone-600 dark:text-stone-400">
              One MP4 or MOV, up to 5 minutes.
            </p>
          </div>
          <div className="space-y-1">
            <label htmlFor={ids.photos} className="block text-sm font-medium">
              Supplementary photos
            </label>
            <input
              id={ids.photos}
              type="file"
              multiple
              accept={ACCEPT.photo}
              onChange={(e) => setPhotos(Array.from(e.target.files ?? []))}
              className="block w-full text-sm"
            />
            <p className="text-xs text-stone-600 dark:text-stone-400">JPEG or PNG, up to 300.</p>
          </div>
          <div className="space-y-1">
            <label htmlFor={ids.plan} className="block text-sm font-medium">
              Floor plan (optional)
            </label>
            <input
              id={ids.plan}
              type="file"
              accept={ACCEPT.floor_plan}
              onChange={(e) => setPlan(e.target.files?.[0] ?? null)}
              className="block w-full text-sm"
            />
            <p className="text-xs text-stone-600 dark:text-stone-400">
              PNG, JPEG or PDF, up to 20 MiB.
            </p>
          </div>
        </div>

        {/* The variable is named literally here so that Next.js can inline it into the bundle. */}
        <SceneAnalysisNotice
          shown={sceneAnalysisNoticeShown(process.env.NEXT_PUBLIC_AI_SCENE_ANALYSIS)}
        />

        <div aria-live="polite" className="mt-4 space-y-2 text-sm">
          {candidates.length > 0 ? (
            <p>
              Selected {candidates.length} file(s), {formatBytes(selectionValidation.totalBytes)}.
              {resumableCount > 0
                ? ` ${resumableCount} of them can resume a previous, unfinished upload.`
                : ""}
            </p>
          ) : null}
          {blockingErrors.length > 0 ? (
            <ul className="list-disc pl-5 text-red-700 dark:text-red-300">
              {blockingErrors.map((e, i) => (
                <li key={`${e.code}-${i}`}>{e.message}</li>
              ))}
            </ul>
          ) : null}
          {validation.warnings.length > 0 ? (
            <ul className="list-disc pl-5 text-amber-800 dark:text-amber-300">
              {validation.warnings.map((w, i) => (
                <li key={`${w.code}-${i}`}>{w.message}</li>
              ))}
            </ul>
          ) : null}
        </div>
        <p className="mt-2 text-xs text-stone-600 dark:text-stone-400">
          File types are checked again on the server; a file can still be rejected after upload. If
          a previous upload was interrupted, select the same file again to resume it.
        </p>
        <Button
          className="mt-4"
          onClick={beginUpload}
          disabled={candidates.length === 0 || typeErrors.length > 0 || Boolean(deleting)}
        >
          Upload selected files
        </Button>
      </Card>

      {rows.length > 0 || previousVerified.length > 0 ? (
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-lg font-semibold">2. Uploads</h2>
            {rows.length > 0 ? (
              <div className="flex gap-2">
                <Button variant="secondary" onClick={pauseAll} disabled={!uploading}>
                  Pause all
                </Button>
                <Button variant="secondary" onClick={resumeAll}>
                  Resume all
                </Button>
              </div>
            ) : null}
          </div>
          <div className="mt-3">
            <UploadList
              rows={rows}
              onPause={pauseRow}
              onResume={resumeRow}
              onRemove={(id) => void removeRow(id)}
            />
          </div>
          {previousVerified.length > 0 ? (
            <fieldset className="mt-4 space-y-2 text-sm">
              <legend className="font-medium">Previously uploaded and verified files</legend>
              {previousVerified.map((u) => (
                <div key={u.id} className="flex items-center gap-2">
                  <input
                    id={`prev-${u.id}`}
                    type="checkbox"
                    checked={includePrevious.has(u.id)}
                    onChange={(e) =>
                      setIncludePrevious((prev) => {
                        const next = new Set(prev);
                        if (e.target.checked) next.add(u.id);
                        else next.delete(u.id);
                        return next;
                      })
                    }
                  />
                  <label htmlFor={`prev-${u.id}`}>
                    {u.filename} ({KIND_LABEL[u.mediaKind]}, {formatBytes(u.expectedBytes)})
                  </label>
                  <Button
                    variant="ghost"
                    onClick={() => void removeVerified(u.id, u.filename)}
                    aria-label={`Remove ${u.filename} from the project`}
                  >
                    Remove
                  </Button>
                </div>
              ))}
            </fieldset>
          ) : null}
        </Card>
      ) : null}

      <Card>
        <h2 className="text-lg font-semibold">3. Start reconstruction</h2>
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <div className="space-y-1">
            <label htmlFor={ids.quality} className="block text-sm font-medium">
              Quality
            </label>
            <select
              id={ids.quality}
              value={quality}
              onChange={(e) => setQuality(e.target.value as RunQuality)}
              className={inputClass}
            >
              <option value="preview">Preview (faster, check the capture first)</option>
              <option value="standard">Standard</option>
            </select>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <input
              id={ids.supp}
              type="checkbox"
              checked={includeSupplementary}
              onChange={(e) => setIncludeSupplementary(e.target.checked)}
            />
            <label htmlFor={ids.supp}>Include supplementary photos</label>
          </div>
        </div>
        <p className="mt-3 text-sm text-stone-700 dark:text-stone-300">{startHint}</p>
        <div className="mt-3">
          <ErrorNotice error={error} />
        </div>
        <div className="mt-3 flex gap-2">
          <Button onClick={() => void submit()} disabled={!canSubmit}>
            {submitting ? "Submitting..." : "Start reconstruction"}
          </Button>
          <Link
            href={`/projects/${encodeURIComponent(projectId)}`}
            className="self-center text-sm underline"
          >
            Back to project
          </Link>
        </div>
      </Card>
    </div>
  );
}
