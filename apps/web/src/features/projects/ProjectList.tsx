"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button, ErrorNotice, StatusBadge } from "@/components/ui";
import { api } from "@/lib/api/client";
import { toFriendlyError, type FriendlyError } from "@/lib/api/errors";
import type { Project } from "@/lib/api/types";

const PAGE_SIZE = 20;

export function ProjectList() {
  const [items, setItems] = useState<Project[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<FriendlyError | null>(null);
  const firstNewRef = useRef<HTMLAnchorElement>(null);
  const [focusIndex, setFocusIndex] = useState<number | null>(null);

  const loadPage = useCallback(
    (cursor: string | null, signal?: AbortSignal): Promise<void> =>
      api.listProjects(cursor, PAGE_SIZE, signal).then(
        ({ data }) => {
          if (signal?.aborted) return;
          setItems((prev) => (cursor ? [...prev, ...data.items] : data.items));
          setNextCursor(data.nextCursor);
          setError(null);
          setLoaded(true);
          setLoading(false);
        },
        (err: unknown) => {
          if (signal?.aborted) return;
          setError(toFriendlyError(err));
          setLoading(false);
        },
      ),
    [],
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadPage(null, controller.signal);
    return () => controller.abort();
  }, [loadPage]);

  useEffect(() => {
    if (focusIndex !== null) firstNewRef.current?.focus();
  }, [focusIndex, items.length]);

  async function loadMore() {
    if (!nextCursor) return;
    setLoading(true);
    setFocusIndex(items.length);
    await loadPage(nextCursor);
  }

  return (
    <div className="space-y-4">
      <ErrorNotice error={error} onRetry={() => void loadPage(null)} />
      <p aria-live="polite" className="sr-only">
        {loading ? "Loading projects" : `${items.length} projects shown`}
      </p>
      {loaded && items.length === 0 ? (
        <p className="text-sm text-stone-700 dark:text-stone-300">
          No projects yet.{" "}
          <Link href="/projects/new" className="font-medium underline">
            Create your first project
          </Link>
          .
        </p>
      ) : null}
      <ul className="divide-y divide-stone-200 rounded-lg border border-stone-200 bg-white dark:divide-stone-800 dark:border-stone-800 dark:bg-stone-900">
        {items.map((project, index) => (
          <li key={project.id} className="flex items-center justify-between gap-4 p-4">
            <div>
              <Link
                href={`/projects/${encodeURIComponent(project.id)}`}
                ref={index === focusIndex ? firstNewRef : undefined}
                className="font-medium hover:underline"
              >
                {project.title}
              </Link>
              <p className="text-xs text-stone-600 dark:text-stone-400">
                Updated {new Date(project.updatedAt).toLocaleString()}
              </p>
            </div>
            <div className="flex gap-2">
              {project.lifecycle === "deleting" ? (
                <StatusBadge tone="warn">Deleting</StatusBadge>
              ) : null}
              {project.publishedRevisionId ? (
                <StatusBadge tone="good">Published</StatusBadge>
              ) : null}
              {project.role !== "owner" ? <StatusBadge>{project.role}</StatusBadge> : null}
            </div>
          </li>
        ))}
      </ul>
      {nextCursor ? (
        <Button variant="secondary" onClick={() => void loadMore()} disabled={loading}>
          {loading ? "Loading..." : "Load more projects"}
        </Button>
      ) : null}
    </div>
  );
}
