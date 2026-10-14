"use client";

import { useCallback, useEffect, useId, useState, type FormEvent } from "react";
import { Button, ErrorNotice, inputClass } from "@/components/ui";
import { api } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { ImportResolution, TourImport } from "@/lib/api/types";

const MODEL_ID = /^[A-Za-z0-9]{11}$/;

/**
 * The iframe URL for a Matterport tour, built only from a validated model id
 * (README section 15: never from provider-supplied HTML or a stored URL).
 */
export function matterportEmbedUrl(modelId: string): string | null {
  return MODEL_ID.test(modelId)
    ? `https://my.matterport.com/show/?m=${encodeURIComponent(modelId)}&play=0`
    : null;
}

/** The listing's host for attribution ("Found on www.homeq.se"), or null. */
export function listingHost(url: string | null): string | null {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" ? parsed.hostname : null;
  } catch {
    return null;
  }
}

function TourEmbed({ tour }: { tour: TourImport }) {
  const [shown, setShown] = useState(false);
  const src = matterportEmbedUrl(tour.modelId);
  if (!src) return null;
  return shown ? (
    <div className="mt-2 aspect-video w-full overflow-hidden rounded-md border border-stone-300 dark:border-stone-700">
      <iframe
        src={src}
        title={`Matterport tour ${tour.modelId}`}
        className="h-full w-full"
        allow="fullscreen; xr-spatial-tracking"
        allowFullScreen
        referrerPolicy="strict-origin"
        sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"
      />
    </div>
  ) : (
    <Button variant="secondary" onClick={() => setShown(true)}>
      Show the tour
    </Button>
  );
}

/**
 * Existing tours (README section 15). A Matterport tour is shown as Matterport's
 * own embed with its attribution; it is not converted into a walkable Andarilho
 * scene, which needs the owner's export.
 */
export function ImportsPanel({ projectId, canEdit }: { projectId: string; canEdit: boolean }) {
  const urlId = useId();
  const permissionId = useId();
  const [tours, setTours] = useState<TourImport[]>([]);
  const [url, setUrl] = useState("");
  const [resolution, setResolution] = useState<ImportResolution | null>(null);
  const [permission, setPermission] = useState(false);
  const [error, setError] = useState<FriendlyError | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const { data } = await api.listImports(projectId);
      setTours(data.items);
    } catch (err) {
      setError(toFriendlyError(err));
    }
  }, [projectId]);

  useEffect(() => {
    let active = true;
    api
      .listImports(projectId)
      .then(({ data }) => {
        if (active) setTours(data.items);
      })
      .catch((err: unknown) => {
        if (active) setError(toFriendlyError(err));
      });
    return () => {
      active = false;
    };
  }, [projectId]);

  async function resolve(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResolution(null);
    setPermission(false);
    try {
      const { data } = await api.resolveImport(projectId, url.trim());
      setResolution(data);
    } catch (err) {
      setError(toFriendlyError(err));
    } finally {
      setBusy(false);
    }
  }

  async function add(modelId: string) {
    setBusy(true);
    setError(null);
    try {
      await api.createImport(projectId, modelId, resolution?.listingUrl ?? null);
      setResolution(null);
      setUrl("");
      setPermission(false);
      await load();
    } catch (err) {
      setError(toFriendlyError(err));
    } finally {
      setBusy(false);
    }
  }

  async function remove(importId: string) {
    setBusy(true);
    setError(null);
    try {
      await api.deleteImport(importId);
      await load();
    } catch (err) {
      setError(toFriendlyError(err));
    } finally {
      setBusy(false);
    }
  }

  const known = new Set(tours.map((t) => t.modelId));
  return (
    <div>
      <h2 className="text-lg font-semibold">Existing tours</h2>
      <p className="mt-1 text-sm text-stone-600 dark:text-stone-400">
        A Matterport tour is shown by Matterport inside Andarilho, with its attribution. It is not
        converted into a walkable Andarilho scene; that needs the owner&apos;s model export.
      </p>

      {tours.length > 0 ? (
        <ul className="mt-3 space-y-3 text-sm">
          {tours.map((tour) => {
            const host = listingHost(tour.listingUrl);
            return (
              <li key={tour.id}>
                <p>
                  Matterport tour <strong>{tour.modelId}</strong>
                  {host && tour.listingUrl ? (
                    <>
                      {" "}
                      · found on{" "}
                      <a
                        href={tour.listingUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline"
                      >
                        {host}
                      </a>
                    </>
                  ) : null}
                </p>
                <div className="mt-1 flex flex-wrap gap-2">
                  <TourEmbed tour={tour} />
                  {canEdit ? (
                    <Button variant="ghost" onClick={() => void remove(tour.id)} disabled={busy}>
                      Remove
                    </Button>
                  ) : null}
                </div>
              </li>
            );
          })}
        </ul>
      ) : null}

      {canEdit ? (
        <form onSubmit={(e) => void resolve(e)} className="mt-4 space-y-2">
          <label htmlFor={urlId} className="block text-sm font-medium">
            Matterport tour link or HomeQ listing page
          </label>
          <div className="flex flex-wrap gap-2">
            <input
              id={urlId}
              type="url"
              inputMode="url"
              required
              placeholder="https://my.matterport.com/show/?m=..."
              className={`${inputClass} min-w-0 flex-1`}
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <Button type="submit" disabled={busy || !url.trim()}>
              {busy && !resolution ? "Looking..." : "Find tours"}
            </Button>
          </div>
        </form>
      ) : null}

      {resolution ? (
        <div aria-live="polite" className="mt-3 space-y-2 text-sm">
          {resolution.notes.map((note) => (
            <p key={note} className="text-stone-600 dark:text-stone-400">
              {note}
            </p>
          ))}
          {resolution.tours.length > 0 ? (
            <>
              <label htmlFor={permissionId} className="flex items-start gap-2">
                <input
                  id={permissionId}
                  type="checkbox"
                  checked={permission}
                  onChange={(e) => setPermission(e.target.checked)}
                />
                <span>
                  I am allowed to show this tour in Andarilho (for example, it is my listing or the
                  publisher permits embedding).
                </span>
              </label>
              <ul className="space-y-1">
                {resolution.tours.map((tour) => (
                  <li key={tour.modelId} className="flex items-center justify-between gap-2">
                    <span>Matterport tour {tour.modelId}</span>
                    {known.has(tour.modelId) ? (
                      <span className="text-stone-600 dark:text-stone-400">Already added</span>
                    ) : (
                      <Button onClick={() => void add(tour.modelId)} disabled={busy || !permission}>
                        Add tour
                      </Button>
                    )}
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </div>
      ) : null}
      <ErrorNotice error={error} />
    </div>
  );
}
