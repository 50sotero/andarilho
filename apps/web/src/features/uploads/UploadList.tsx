"use client";

import { Button, StatusBadge } from "@/components/ui";
import { formatBytes } from "@/features/capture/validation";
import type { UploadStatus } from "@/lib/api/types";
import type { UploadProgress } from "./uploader";

export interface UploadRowView {
  id: string;
  name: string;
  kindLabel: string;
  progress: UploadProgress;
  serverStatus: UploadStatus | null;
  rejectionCode: string | null;
}

const PHASE_LABEL: Record<UploadProgress["phase"], string> = {
  idle: "Waiting",
  initializing: "Starting",
  uploading: "Uploading",
  paused: "Paused",
  completing: "Finishing",
  completed: "Uploaded",
  failed: "Failed",
};

function verificationLabel(row: UploadRowView): { text: string; tone: "good" | "warn" | "bad" | "neutral" } | null {
  switch (row.serverStatus) {
    case "verified":
      return { text: "Verified", tone: "good" };
    case "rejected":
      return { text: "Rejected", tone: "bad" };
    case "uploaded_unverified":
      return { text: "Checking file", tone: "warn" };
    case "expired":
    case "aborted":
      return { text: row.serverStatus === "expired" ? "Expired" : "Removed", tone: "neutral" };
    default:
      return null;
  }
}

export function UploadList({
  rows,
  onPause,
  onResume,
  onRemove,
}: {
  rows: UploadRowView[];
  onPause: (id: string) => void;
  onResume: (id: string) => void;
  onRemove: (id: string) => void;
}) {
  if (rows.length === 0) return null;
  return (
    <ul className="divide-y divide-stone-200 rounded-lg border border-stone-200 dark:divide-stone-800 dark:border-stone-800">
      {rows.map((row) => {
        const { progress } = row;
        const percent =
          progress.totalBytes > 0 ? Math.floor((progress.uploadedBytes / progress.totalBytes) * 100) : 0;
        const verification = verificationLabel(row);
        const active =
          progress.phase === "uploading" ||
          progress.phase === "initializing" ||
          progress.phase === "completing";
        return (
          <li key={row.id} className="space-y-2 p-3 text-sm">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0">
                <p className="truncate font-medium">{row.name}</p>
                <p className="text-xs text-stone-600 dark:text-stone-400">
                  {row.kindLabel} · {formatBytes(progress.totalBytes)}
                  {progress.resumed ? " · resumed" : ""}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <StatusBadge tone={progress.phase === "failed" ? "bad" : "neutral"}>
                  {PHASE_LABEL[progress.phase]}
                </StatusBadge>
                {verification ? <StatusBadge tone={verification.tone}>{verification.text}</StatusBadge> : null}
                {active ? (
                  <Button variant="secondary" onClick={() => onPause(row.id)} aria-label={`Pause ${row.name}`}>
                    Pause
                  </Button>
                ) : null}
                {progress.phase === "paused" || progress.phase === "failed" ? (
                  <Button variant="secondary" onClick={() => onResume(row.id)} aria-label={`Resume ${row.name}`}>
                    Resume
                  </Button>
                ) : null}
                {row.serverStatus !== "verified" ? (
                  <Button variant="ghost" onClick={() => onRemove(row.id)} aria-label={`Remove ${row.name}`}>
                    Remove
                  </Button>
                ) : null}
              </div>
            </div>
            <progress
              className="h-2 w-full"
              max={100}
              value={percent}
              aria-label={`${row.name} upload progress`}
            >
              {percent}%
            </progress>
            <p className="text-xs text-stone-600 dark:text-stone-400">
              {formatBytes(progress.uploadedBytes)} of {formatBytes(progress.totalBytes)}
              {progress.partCount > 0 ? ` · part ${progress.completedParts} of ${progress.partCount}` : ""}
            </p>
            {row.rejectionCode ? (
              <p className="text-xs text-red-700 dark:text-red-300">
                The file was rejected ({row.rejectionCode}). Check that it is a supported, undamaged
                file and try another.
              </p>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}
