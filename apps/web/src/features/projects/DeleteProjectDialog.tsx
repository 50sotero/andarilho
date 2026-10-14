"use client";

import { useRouter } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";
import { Button, ErrorNotice, inputClass } from "@/components/ui";
import { api } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";

export function DeleteProjectDialog({
  projectId,
  title,
  open,
  onClose,
}: {
  projectId: string;
  title: string;
  open: boolean;
  onClose: () => void;
}) {
  const router = useRouter();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const headingId = useId();
  const descId = useId();
  const confirmId = useId();
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<FriendlyError | null>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  async function confirm() {
    setBusy(true);
    setError(null);
    try {
      await api.deleteProject(projectId);
      router.push("/projects");
      router.refresh();
    } catch (err) {
      setError(toFriendlyError(err));
      setBusy(false);
    }
  }

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby={headingId}
      aria-describedby={descId}
      onClose={onClose}
      className="m-auto max-w-lg rounded-lg border border-stone-300 bg-white p-6 text-stone-900 backdrop:bg-black/50 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-100"
    >
      <h2 id={headingId} className="text-lg font-semibold">
        Delete &ldquo;{title}&rdquo;?
      </h2>
      <div id={descId} className="mt-3 space-y-2 text-sm">
        <p>Deleting this project:</p>
        <ul className="list-disc space-y-1 pl-5">
          <li>stops all share links immediately and blocks new uploads, runs and publications;</li>
          <li>cancels any processing that is queued or running;</li>
          <li>
            erases the original video, photos and floor plan, generated scenes, checkpoints and
            unfinished uploads from storage.
          </li>
        </ul>
        <p>
          Cleanup finishes in the background and may take a while. People who already downloaded a
          shared scene keep their copy, and database backups expire on their own schedule. This
          cannot be undone.
        </p>
      </div>
      <div className="mt-4 space-y-1">
        <label htmlFor={confirmId} className="block text-sm font-medium">
          Type the project name to confirm
        </label>
        <input
          id={confirmId}
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          className={inputClass}
          autoComplete="off"
        />
      </div>
      <div className="mt-3">
        <ErrorNotice error={error} />
      </div>
      <div className="mt-4 flex justify-end gap-2">
        <Button variant="secondary" onClick={onClose} disabled={busy}>
          Keep project
        </Button>
        <Button
          variant="danger"
          onClick={() => void confirm()}
          disabled={busy || typed.trim() !== title.trim()}
        >
          {busy ? "Deleting..." : "Delete project"}
        </Button>
      </div>
    </dialog>
  );
}
