"use client";

import { useId, useRef, useState } from "react";
import { Button, ErrorNotice, StatusBadge, inputClass } from "@/components/ui";
import { api } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { Share } from "@/lib/api/types";

const EXPIRY_OPTIONS = [
  { value: "", label: "No expiry (until revoked)" },
  { value: "1", label: "1 day" },
  { value: "7", label: "7 days" },
  { value: "30", label: "30 days" },
] as const;

export function expiryFromDays(days: string, now: number = Date.now()): string | null {
  const n = Number(days);
  if (!days || !Number.isFinite(n) || n <= 0) return null;
  return new Date(now + n * 24 * 60 * 60 * 1000).toISOString();
}

function shareState(share: Share): { label: string; tone: "good" | "bad" | "neutral" } {
  if (share.revokedAt) return { label: "Revoked", tone: "neutral" };
  if (share.expiresAt && Date.parse(share.expiresAt) <= Date.now()) {
    return { label: "Expired", tone: "neutral" };
  }
  return { label: "Active", tone: "good" };
}

/**
 * Publish/revoke controls for an approved scene revision.
 * The share token is shown exactly once, immediately after publishing.
 */
export function SharePanel({
  sceneId,
  approvedRevisionId,
  sceneError = null,
  onRetryScene,
  shares,
  getSceneEtag,
  onChanged,
}: {
  sceneId: string;
  /**
   * Approved revision that would be published, null when nothing is approved, or
   * undefined until the scene's revisions have been read.
   */
  approvedRevisionId: string | null | undefined;
  /** Why the scene's revisions could not be read; shown with a retry. */
  sceneError?: FriendlyError | null;
  onRetryScene?: () => void;
  shares: Share[];
  /** Fetches the current scene ETag (required by If-Match on publish). */
  getSceneEtag: () => Promise<string | null>;
  onChanged: () => void;
}) {
  const understandId = useId();
  const expiryId = useId();
  const linkId = useId();
  const linkRef = useRef<HTMLInputElement>(null);
  const [understood, setUnderstood] = useState(false);
  const [expiryDays, setExpiryDays] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<FriendlyError | null>(null);
  const [createdUrl, setCreatedUrl] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [confirmRevoke, setConfirmRevoke] = useState<string | null>(null);

  async function publish() {
    if (!approvedRevisionId) return;
    setBusy(true);
    setError(null);
    try {
      const etag = await getSceneEtag();
      if (!etag) throw new Error("The scene version could not be read. Reload and try again.");
      const { data } = await api.createShare(
        sceneId,
        approvedRevisionId,
        etag,
        expiryFromDays(expiryDays),
      );
      const url = new URL(data.path, window.location.origin).toString();
      setCreatedUrl(url);
      setUnderstood(false);
      setStatus("Share link created. Copy it now; it will not be shown again.");
      requestAnimationFrame(() => {
        linkRef.current?.focus();
        linkRef.current?.select();
      });
      onChanged();
    } catch (err) {
      setError(toFriendlyError(err));
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    if (!createdUrl) return;
    try {
      await navigator.clipboard.writeText(createdUrl);
      setStatus("Link copied to the clipboard.");
    } catch {
      linkRef.current?.select();
      setStatus("Copying failed. The link is selected; copy it with your keyboard.");
    }
  }

  async function revoke(shareId: string) {
    setBusy(true);
    setError(null);
    try {
      await api.revokeShare(shareId);
      setConfirmRevoke(null);
      setStatus("Share link revoked.");
      onChanged();
    } catch (err) {
      setError(toFriendlyError(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Sharing</h2>
      <p aria-live="polite" className="text-sm text-teal-800 dark:text-teal-300">
        {status}
      </p>
      <ErrorNotice error={error} />

      {createdUrl ? (
        <div className="space-y-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm dark:border-amber-800 dark:bg-amber-950">
          <label htmlFor={linkId} className="block font-medium">
            Your share link (shown only once)
          </label>
          <div className="flex gap-2">
            <input
              id={linkId}
              ref={linkRef}
              readOnly
              value={createdUrl}
              className={inputClass}
              onFocus={(e) => e.currentTarget.select()}
            />
            <Button variant="secondary" onClick={() => void copy()}>
              Copy
            </Button>
          </div>
          <Button variant="ghost" onClick={() => setCreatedUrl(null)}>
            I saved the link, hide it
          </Button>
        </div>
      ) : null}

      {approvedRevisionId ? (
        <div className="space-y-3 text-sm">
          <div className="space-y-1 text-stone-700 dark:text-stone-300">
            <p>
              Publishing creates an <strong>unlisted link</strong>. It is not searchable, but{" "}
              <strong>anyone who has the link can view the scene</strong> &mdash; including people
              it is forwarded to. Original video, photos and floor plan are never shared.
            </p>
            <p>
              You can revoke a link at any time. Revocation stops new visits, but it cannot recall
              anything a viewer already downloaded, and an already-issued asset link can keep
              working for up to about 5 minutes.
            </p>
          </div>
          <div className="space-y-1">
            <label htmlFor={expiryId} className="block font-medium">
              Link expiry
            </label>
            <select
              id={expiryId}
              value={expiryDays}
              onChange={(e) => setExpiryDays(e.target.value)}
              className={inputClass}
            >
              {EXPIRY_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-start gap-2">
            <input
              id={understandId}
              type="checkbox"
              checked={understood}
              onChange={(e) => setUnderstood(e.target.checked)}
              className="mt-1"
            />
            <label htmlFor={understandId}>
              I understand that anyone with the link can view this scene until it expires or I
              revoke it.
            </label>
          </div>
          <Button onClick={() => void publish()} disabled={!understood || busy}>
            {busy ? "Publishing..." : "Publish and create link"}
          </Button>
        </div>
      ) : approvedRevisionId === null ? (
        <p className="text-sm text-stone-700 dark:text-stone-300">
          Approve a scene revision in Review before publishing it.
        </p>
      ) : sceneError ? (
        <ErrorNotice error={sceneError} onRetry={onRetryScene} />
      ) : (
        <p aria-live="polite" className="text-sm text-stone-700 dark:text-stone-300">
          Checking for an approved revision...
        </p>
      )}

      {shares.length > 0 ? (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">Links</h3>
          <ul className="space-y-2 text-sm">
            {shares.map((share) => {
              const state = shareState(share);
              return (
                <li
                  key={share.id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded border border-stone-200 p-2 dark:border-stone-800"
                >
                  <div>
                    <StatusBadge tone={state.tone}>{state.label}</StatusBadge>{" "}
                    <span>Created {new Date(share.createdAt).toLocaleString()}</span>
                    {share.expiresAt ? (
                      <span> · expires {new Date(share.expiresAt).toLocaleString()}</span>
                    ) : null}
                  </div>
                  {state.label === "Active" ? (
                    confirmRevoke === share.id ? (
                      <div className="flex items-center gap-2">
                        <span>Stop this link working?</span>
                        <Button
                          variant="danger"
                          onClick={() => void revoke(share.id)}
                          disabled={busy}
                        >
                          Revoke
                        </Button>
                        <Button variant="ghost" onClick={() => setConfirmRevoke(null)}>
                          Cancel
                        </Button>
                      </div>
                    ) : (
                      <Button variant="secondary" onClick={() => setConfirmRevoke(share.id)}>
                        Revoke link
                      </Button>
                    )
                  ) : null}
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
