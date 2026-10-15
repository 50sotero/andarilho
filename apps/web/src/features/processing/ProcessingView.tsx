"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Button, Card, ErrorNotice, StatusBadge, primaryLinkClass } from "@/components/ui";
import { api, newIdempotencyKey } from "@/lib/api/client";
import { friendlyFromErrorBody, toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { JobDetail, JobEvent, ProjectDetail, Run } from "@/lib/api/types";
import { usePolling } from "@/lib/use-polling";
import { RunAdvice } from "./RunAdvice";
import {
  STAGES,
  formatElapsed,
  formatStageProgress,
  isTerminalRun,
  runActions,
  stageLabel,
  stageStates,
  type StageView,
} from "./stages";

const STAGE_BADGE: Record<StageView, { text: string; tone: "good" | "warn" | "bad" | "neutral" }> =
  {
    done: { text: "Done", tone: "good" },
    active: { text: "In progress", tone: "warn" },
    waiting: { text: "Queued", tone: "neutral" },
    failed: { text: "Failed", tone: "bad" },
    cancelled: { text: "Cancelled", tone: "neutral" },
    pending: { text: "Not started", tone: "neutral" },
  };

const MAX_EVENTS = 200;

function pickRun(detail: ProjectDetail | null, jobId: string | null): Run | null {
  if (!detail) return null;
  const runs = [...detail.runs].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  if (jobId) {
    const match = runs.find((r) => r.currentJobId === jobId || r.jobs.some((j) => j.id === jobId));
    if (match) return match;
  }
  return runs[0] ?? null;
}

function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

export function ProcessingView({
  projectId,
  initialJobId,
}: {
  projectId: string;
  initialJobId: string | null;
}) {
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [preferredJobId, setPreferredJobId] = useState<string | null>(initialJobId);
  const [job, setJob] = useState<JobDetail["job"] | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [pollError, setPollError] = useState<FriendlyError | null>(null);
  const [actionError, setActionError] = useState<FriendlyError | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const cursorRef = useRef<{ jobId: string | null; after: number | null }>({
    jobId: null,
    after: null,
  });
  const retryIntentRef = useRef<{ runId: string; key: string } | null>(null);

  const run = pickRun(detail, preferredJobId);
  const terminal = isTerminalRun(run?.status);
  const currentJobId = run?.currentJobId ?? preferredJobId;

  const poller = usePolling(
    async (signal) => {
      const { data } = await api.getProject(projectId, signal);
      setDetail(data);
      const latestRun = pickRun(data, preferredJobId);
      const jobId = latestRun?.currentJobId ?? preferredJobId;
      if (jobId) {
        if (cursorRef.current.jobId !== jobId) cursorRef.current = { jobId, after: null };
        const { data: jobData } = await api.getJob(jobId, cursorRef.current.after, 100, signal);
        setJob(jobData.job);
        if (jobData.events.length > 0) {
          setEvents((prev) => {
            const seen = new Set(prev.map((e) => e.id));
            const merged = [...prev, ...jobData.events.filter((e) => !seen.has(e.id))];
            return merged.slice(-MAX_EVENTS);
          });
        }
        if (jobData.nextCursor !== null) cursorRef.current.after = jobData.nextCursor;
      }
      setPollError(null);
      return { terminal: isTerminalRun(latestRun?.status) };
    },
    {
      enabled: true,
      key: `${projectId}:${preferredJobId ?? ""}`,
      onError: (err) => setPollError(toFriendlyError(err)),
    },
  );

  const now = useNow(!terminal && Boolean(job));

  async function cancel() {
    if (!run) return;
    setBusy(true);
    setActionError(null);
    try {
      await api.cancelRun(run.id);
      setConfirmCancel(false);
      poller.current?.pollNow();
    } catch (err) {
      setActionError(toFriendlyError(err));
    } finally {
      setBusy(false);
    }
  }

  async function retry() {
    if (!run) return;
    setBusy(true);
    setActionError(null);
    if (!retryIntentRef.current || retryIntentRef.current.runId !== run.id) {
      retryIntentRef.current = { runId: run.id, key: newIdempotencyKey() };
    }
    try {
      const { data } = await api.retryRun(run.id, retryIntentRef.current.key);
      retryIntentRef.current = null;
      setEvents([]);
      setJob(null);
      setPreferredJobId(data.jobId); // restarts polling for the new job
    } catch (err) {
      setActionError(toFriendlyError(err));
    } finally {
      setBusy(false);
    }
  }

  const states = stageStates(run);
  const latestEvent = events[events.length - 1];
  const stageProgress =
    latestEvent && job && latestEvent.stage === job.stage
      ? formatStageProgress(latestEvent.progress)
      : null;
  const runError = run?.error ?? job?.error ?? null;
  const friendlyRunError =
    runError && run?.status === "failed" ? friendlyFromErrorBody(runError) : null;
  const base = `/projects/${encodeURIComponent(projectId)}`;
  const canEdit = detail?.project.role === "owner" || detail?.project.role === "editor";

  if (!detail) {
    return (
      <div className="space-y-3">
        <p aria-live="polite">Loading processing status...</p>
        <ErrorNotice error={pollError} />
      </div>
    );
  }

  if (!run) {
    return (
      <Card>
        <p>No reconstruction has been started for this project.</p>
        {canEdit ? (
          <Link href={`${base}/capture`} className={`${primaryLinkClass} mt-3`}>
            Upload a capture
          </Link>
        ) : null}
      </Card>
    );
  }

  const actions = runActions(run, detail.project.role);
  return (
    <div className="space-y-6">
      <Card>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-lg font-semibold">Run status</h2>
          <StatusBadge
            tone={
              run.status === "succeeded"
                ? "good"
                : run.status === "failed"
                  ? "bad"
                  : run.status === "cancelled"
                    ? "neutral"
                    : "warn"
            }
          >
            {run.status}
          </StatusBadge>
        </div>
        <div aria-live="polite" className="mt-2 space-y-1 text-sm">
          {!terminal && job ? (
            <p>
              Current stage: <strong>{stageLabel(job.stage)}</strong>
              {job.state === "retry_wait" ? " (retrying after a temporary problem)" : ""}
              {job.attempts > 1 ? ` · attempt ${job.attempts}` : ""}
            </p>
          ) : null}
          {!terminal && stageProgress ? <p>Stage progress: {stageProgress}</p> : null}
          {!terminal && job ? (
            <p className="text-stone-600 dark:text-stone-400">
              This stage has been running for {formatElapsed(now - Date.parse(job.createdAt))}.
              Training time varies with the capture; no estimate is shown because it would not be
              reliable.
            </p>
          ) : null}
          {run.cancelRequested && !terminal ? (
            <p>Cancellation requested. Stopping work...</p>
          ) : null}
          {run.status === "succeeded" ? (
            <p>Processing finished. Review the scale, floor and walls before walking or sharing.</p>
          ) : null}
          {run.status === "cancelled" ? <p>This run was cancelled.</p> : null}
        </div>

        {friendlyRunError ? (
          <div className="mt-3 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100">
            <p className="font-semibold">{friendlyRunError.title}</p>
            <p>{friendlyRunError.message}</p>
            {friendlyRunError.suggestedAction ? (
              <p className="mt-1">
                <strong>What to do:</strong> {friendlyRunError.suggestedAction}
              </p>
            ) : null}
            {friendlyRunError.requestId ? (
              <p className="mt-1 text-xs opacity-75">Reference: {friendlyRunError.requestId}</p>
            ) : null}
          </div>
        ) : null}
        {friendlyRunError && canEdit ? <RunAdvice key={run.id} runId={run.id} /> : null}

        <div className="mt-3 space-y-2">
          <ErrorNotice error={actionError} />
          {pollError ? (
            <p className="text-sm text-amber-800 dark:text-amber-300" role="status">
              Status updates are delayed: {pollError.message} Retrying automatically.
            </p>
          ) : null}
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {actions.cancel ? (
            confirmCancel ? (
              <>
                <span className="self-center text-sm">
                  Cancel this run? Work done so far is discarded.
                </span>
                <Button variant="danger" onClick={() => void cancel()} disabled={busy}>
                  Yes, cancel
                </Button>
                <Button variant="ghost" onClick={() => setConfirmCancel(false)}>
                  Keep running
                </Button>
              </>
            ) : (
              <Button variant="secondary" onClick={() => setConfirmCancel(true)}>
                Cancel run
              </Button>
            )
          ) : null}
          {actions.retry ? (
            <Button onClick={() => void retry()} disabled={busy}>
              {busy ? "Starting..." : "Retry with the same inputs"}
            </Button>
          ) : null}
          {actions.newCapture ? (
            <Link href={`${base}/capture`} className="self-center text-sm underline">
              Upload a new capture
            </Link>
          ) : null}
          {actions.review ? (
            <Link href={`${base}/review`} className={primaryLinkClass}>
              Review the scene
            </Link>
          ) : null}
        </div>
      </Card>

      <Card>
        <h2 className="text-lg font-semibold">Stages</h2>
        <ol className="mt-3 space-y-2">
          {STAGES.map((stage, index) => {
            const view = states[stage.id] ?? "pending";
            const badge = STAGE_BADGE[view];
            return (
              <li
                key={stage.id}
                className="flex items-center justify-between gap-2 text-sm"
                aria-current={view === "active" ? "step" : undefined}
              >
                <span>
                  {index + 1}. {stage.label}
                </span>
                <StatusBadge tone={badge.tone}>{badge.text}</StatusBadge>
              </li>
            );
          })}
        </ol>
      </Card>

      {events.length > 0 ? (
        <Card>
          <h2 className="text-lg font-semibold">Activity</h2>
          <ol
            className="mt-3 max-h-72 space-y-1 overflow-y-auto text-sm"
            aria-label="Processing events"
          >
            {events.map((event) => (
              <li key={event.id} className="flex gap-2">
                <time className="shrink-0 text-stone-500" dateTime={event.createdAt}>
                  {new Date(event.createdAt).toLocaleTimeString()}
                </time>
                <span>
                  {stageLabel(event.stage)}: {event.message ?? event.state}
                  {formatStageProgress(event.progress)
                    ? ` (${formatStageProgress(event.progress)})`
                    : ""}
                </span>
              </li>
            ))}
          </ol>
        </Card>
      ) : null}
      <p className="text-xs text-stone-500">
        Run {run.id} · job {currentJobId ?? "none"}
      </p>
    </div>
  );
}
