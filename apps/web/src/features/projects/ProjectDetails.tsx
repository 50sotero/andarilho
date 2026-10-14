"use client";

import Link from "next/link";
import { useCallback, useEffect, useId, useState, type FormEvent } from "react";
import {
  Button,
  Card,
  ErrorNotice,
  StatusBadge,
  inputClass,
  linkButtonClass,
  primaryLinkClass,
} from "@/components/ui";
import { SharePanel } from "@/features/sharing/SharePanel";
import { api } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { Run, SceneDetail } from "@/lib/api/types";
import { DeleteProjectDialog } from "./DeleteProjectDialog";
import { MAX_TITLE_LENGTH } from "./NewProjectForm";
import { useProject } from "./use-project";

function runTone(status: Run["status"]): "good" | "warn" | "bad" | "neutral" {
  if (status === "succeeded") return "good";
  if (status === "failed") return "bad";
  if (status === "running" || status === "queued") return "warn";
  return "neutral";
}

export function latestApprovedRevisionId(scene: SceneDetail | null): string | null {
  if (!scene) return null;
  const approved = scene.revisions
    .filter((r) => r.approvalState === "approved" && r.validationState === "valid")
    .sort((a, b) => b.revision - a.revision);
  return approved[0]?.id ?? null;
}

export function ProjectDetails({ projectId }: { projectId: string }) {
  const { detail, etag, error, loading, reload } = useProject(projectId);
  const [scene, setScene] = useState<SceneDetail | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [title, setTitle] = useState("");
  const [actionError, setActionError] = useState<FriendlyError | null>(null);
  const [status, setStatus] = useState("");
  const titleId = useId();

  // Draft scene data is review material for owners/editors; viewers never request it.
  const sceneId = detail?.project.role === "viewer" ? null : (detail?.scene?.id ?? null);

  const loadScene = useCallback(async (): Promise<string | null> => {
    if (!sceneId) return null;
    const result = await api.getScene(sceneId);
    setScene(result.data);
    return result.etag;
  }, [sceneId]);

  useEffect(() => {
    if (!sceneId) return;
    let active = true;
    api
      .getScene(sceneId)
      .then((result) => {
        if (active) setScene(result.data);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [sceneId]);

  async function rename(event: FormEvent) {
    event.preventDefault();
    if (!etag || !title.trim()) return;
    setActionError(null);
    try {
      await api.renameProject(projectId, title.trim(), etag);
      setRenaming(false);
      setStatus("Project renamed.");
      await reload();
    } catch (err) {
      setActionError(toFriendlyError(err));
    }
  }

  if (loading && !detail) return <p aria-live="polite">Loading project...</p>;
  if (!detail) return <ErrorNotice error={error} onRetry={() => void reload()} />;

  const { project, uploads, runs, shares } = detail;
  const latestRun = [...runs].sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
  const deleting = project.lifecycle !== "active";
  const canEdit = !deleting && (project.role === "owner" || project.role === "editor");
  const base = `/projects/${encodeURIComponent(project.id)}`;
  const verified = uploads.filter((u) => u.status === "verified").length;
  const approvedRevisionId = latestApprovedRevisionId(scene);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          {renaming ? (
            <form onSubmit={rename} className="flex items-end gap-2">
              <div>
                <label htmlFor={titleId} className="block text-sm font-medium">
                  Project name
                </label>
                <input
                  id={titleId}
                  value={title}
                  maxLength={MAX_TITLE_LENGTH}
                  onChange={(e) => setTitle(e.target.value)}
                  className={inputClass}
                  required
                />
              </div>
              <Button type="submit">Save</Button>
              <Button variant="ghost" onClick={() => setRenaming(false)}>
                Cancel
              </Button>
            </form>
          ) : (
            <h1 className="text-2xl font-semibold">{project.title}</h1>
          )}
          <p className="text-sm text-stone-600 dark:text-stone-400">
            Created {new Date(project.createdAt).toLocaleString()}
          </p>
        </div>
        {deleting ? <StatusBadge tone="warn">Being deleted</StatusBadge> : null}
      </div>

      <p aria-live="polite" className="text-sm text-teal-800 dark:text-teal-300">
        {status}
      </p>
      <ErrorNotice error={actionError ?? error} />

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <h2 className="text-lg font-semibold">Capture</h2>
          <p className="mt-1 text-sm">
            {uploads.length === 0
              ? "No files uploaded yet."
              : `${uploads.length} file(s) uploaded, ${verified} verified.`}
          </p>
          {canEdit ? (
            <Link href={`${base}/capture`} className={`${primaryLinkClass} mt-3`}>
              {uploads.length === 0 ? "Add capture" : "Manage capture"}
            </Link>
          ) : null}
        </Card>

        <Card>
          <h2 className="text-lg font-semibold">Processing</h2>
          {latestRun ? (
            <div className="mt-1 space-y-2 text-sm">
              <p>
                Latest run:{" "}
                <StatusBadge tone={runTone(latestRun.status)}>{latestRun.status}</StatusBadge>
                {latestRun.currentStage ? ` · stage ${latestRun.currentStage}` : null}
              </p>
              <Link href={`${base}/processing`} className={linkButtonClass}>
                View progress
              </Link>
            </div>
          ) : (
            <p className="mt-1 text-sm">No reconstruction has been started.</p>
          )}
        </Card>

        <Card>
          <h2 className="text-lg font-semibold">Scene</h2>
          {detail.scene ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {canEdit ? (
                <Link href={`${base}/review`} className={linkButtonClass}>
                  Review layout
                </Link>
              ) : null}
              <Link href={`${base}/view`} className={linkButtonClass}>
                Walk privately
              </Link>
            </div>
          ) : (
            <p className="mt-1 text-sm">A scene appears after processing succeeds.</p>
          )}
        </Card>

        {detail.scene && project.role === "owner" && !deleting ? (
          <Card>
            <SharePanel
              sceneId={detail.scene.id}
              approvedRevisionId={approvedRevisionId}
              shares={shares}
              getSceneEtag={loadScene}
              onChanged={() => void reload()}
            />
          </Card>
        ) : null}
      </div>

      {project.role === "owner" && !deleting ? (
        <Card>
          <h2 className="text-lg font-semibold">Project settings</h2>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button
              variant="secondary"
              onClick={() => {
                setTitle(project.title);
                setRenaming(true);
              }}
            >
              Rename
            </Button>
            <Button variant="danger" onClick={() => setDeleteOpen(true)}>
              Delete project
            </Button>
          </div>
          <DeleteProjectDialog
            projectId={project.id}
            title={project.title}
            open={deleteOpen}
            onClose={() => setDeleteOpen(false)}
          />
        </Card>
      ) : null}
    </div>
  );
}
