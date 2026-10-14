"use client";

import { useRouter } from "next/navigation";
import { useId, useState, type FormEvent } from "react";
import { Button, ErrorNotice, inputClass } from "@/components/ui";
import { api } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";

export const MAX_TITLE_LENGTH = 120;

export function NewProjectForm() {
  const router = useRouter();
  const titleId = useId();
  const hintId = useId();
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<FriendlyError | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const trimmed = title.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      const { data } = await api.createProject(trimmed);
      router.push(`/projects/${encodeURIComponent(data.id)}/capture`);
    } catch (err) {
      setError(toFriendlyError(err));
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <div className="space-y-1">
        <label htmlFor={titleId} className="block text-sm font-medium">
          Name of the place
        </label>
        <input
          id={titleId}
          aria-describedby={hintId}
          required
          maxLength={MAX_TITLE_LENGTH}
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className={inputClass}
          placeholder="e.g. Rua das Flores apartment"
        />
        <p id={hintId} className="text-xs text-stone-600 dark:text-stone-400">
          Only you can see this name unless you publish a share link.
        </p>
      </div>
      <ErrorNotice error={error} />
      <Button type="submit" disabled={busy || !title.trim()}>
        {busy ? "Creating..." : "Create project"}
      </Button>
    </form>
  );
}
