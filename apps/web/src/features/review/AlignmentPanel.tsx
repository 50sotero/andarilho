"use client";

import { useId, useMemo, useState, type MouseEvent } from "react";
import type { SimilarityFit } from "@andarilho/viewer/alignment";
import { Button, inputClass } from "@/components/ui";
import type { FloorPlanLink, Layout } from "@/lib/api/types";
import { alignPlan, describeAlignment, setPlanPoint } from "./alignment";
import type { Selection } from "./layout-edit";

const IMAGE_PLAN_TYPES = new Set(["image/png", "image/jpeg"]);

/**
 * Landmarks and plan alignment: each landmark is placed where it is in the
 * capture (on the canvas) and marked on the floor plan here; two marks place the
 * plan, three or more also check it, and disagreements are reported instead of
 * warping the scene to fit.
 */
export function AlignmentPanel({
  layout,
  selection,
  plan,
  onSelect,
  onChange,
  onRefreshPlan,
  onPlace,
}: {
  layout: Layout;
  selection: Selection | null;
  plan: FloorPlanLink | null;
  onSelect: (sel: Selection) => void;
  onChange: (layout: Layout) => void;
  /** Fetch a fresh short-lived plan link (the page's may have expired). */
  onRefreshPlan: () => Promise<FloorPlanLink | null>;
  /** Show the plan under the layout where the fit places it. */
  onPlace: (fit: SimilarityFit) => void;
}) {
  const landmarks = layout.landmarks ?? [];
  const selectedIndex = selection?.kind === "landmark" ? selection.index : null;
  const selected = selectedIndex !== null ? landmarks[selectedIndex] : undefined;
  const alignment = useMemo(() => alignPlan(layout), [layout]);
  const findings = alignment ? describeAlignment(alignment) : null;
  const marked = landmarks.filter((lm) => lm.planPoint).length;
  const imagePlan = plan && IMAGE_PLAN_TYPES.has(plan.contentType) ? plan : null;

  return (
    <div>
      <h3 className="text-sm font-semibold">Landmarks and plan alignment</h3>
      {landmarks.length === 0 ? (
        <p className="mt-1 text-sm text-stone-600 dark:text-stone-400">
          Add at least three recognizable floor points (corners, door frames) and place each where
          it is in the capture.
          {plan ? " Then mark the same points on the floor plan to align the plan." : ""}
        </p>
      ) : (
        <ul className="mt-2 space-y-1 text-sm">
          {landmarks.map((lm, i) => (
            <li key={lm.id} className="flex items-center justify-between gap-2">
              <span>
                {lm.id}: ({lm.worldPoint.map((v) => v.toFixed(2)).join(", ")}) m
                {lm.uncertainty != null ? ` ± ${lm.uncertainty} m` : ""}
                {plan ? (lm.planPoint ? " · on the plan" : " · not on the plan") : ""}
              </span>
              <Button variant="ghost" onClick={() => onSelect({ kind: "landmark", index: i })}>
                {i === selectedIndex ? "Selected" : "Edit"}
              </Button>
            </li>
          ))}
        </ul>
      )}

      {imagePlan && selected && selectedIndex !== null ? (
        <PlanPicker
          plan={imagePlan}
          layout={layout}
          index={selectedIndex}
          onMark={(point) => onChange(setPlanPoint(layout, selectedIndex, point))}
          onRefresh={onRefreshPlan}
        />
      ) : imagePlan && landmarks.length > 0 ? (
        <p className="mt-2 text-sm text-stone-600 dark:text-stone-400">
          Select a landmark to mark it on the plan.
        </p>
      ) : plan && !imagePlan ? (
        <p className="mt-2 text-sm text-stone-600 dark:text-stone-400">
          Landmarks can be marked on PNG or JPEG plans; a PDF plan can only be traced by eye.
        </p>
      ) : null}

      {imagePlan ? (
        <div aria-live="polite" className="mt-3 space-y-2 text-sm">
          {findings ? (
            <>
              {findings.summary.map((line) => (
                <p key={line}>{line}</p>
              ))}
              {findings.warnings.length > 0 ? (
                <ul className="list-disc pl-5 text-amber-800 dark:text-amber-300">
                  {findings.warnings.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              ) : null}
            </>
          ) : (
            <p className="text-stone-600 dark:text-stone-400">
              {marked === 0
                ? "No landmark is marked on the plan yet."
                : "Mark a second landmark on the plan to place it (a third to check it)."}
            </p>
          )}
          {alignment?.result.ok ? (
            <Button
              variant="secondary"
              onClick={() => {
                if (alignment.result.ok) onPlace(alignment.result.fit);
              }}
            >
              Place the plan from these landmarks
            </Button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function PlanPicker({
  plan,
  layout,
  index,
  onMark,
  onRefresh,
}: {
  plan: FloorPlanLink;
  layout: Layout;
  index: number;
  onMark: (point: [number, number] | null) => void;
  onRefresh: () => Promise<FloorPlanLink | null>;
}) {
  const id = useId();
  const [aspect, setAspect] = useState<number | null>(null);
  const [retried, setRetried] = useState<string | null>(null);
  const landmarks = layout.landmarks ?? [];
  const landmark = landmarks[index];
  if (!landmark) return null;
  const point = landmark.planPoint ?? null;

  const mark = (e: MouseEvent<HTMLImageElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return;
    // Both axes in fractions of the plan's width, from its top-left corner.
    onMark([(e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.width]);
  };
  const coordinate = (axis: 0 | 1, label: string) => (
    <label className="text-xs" htmlFor={`${id}-${axis}`}>
      {label}
      <input
        id={`${id}-${axis}`}
        type="number"
        step="0.001"
        min="0"
        className={inputClass}
        value={point ? point[axis] : ""}
        placeholder="not marked"
        onChange={(e) => {
          const value = Number(e.target.value);
          if (e.target.value === "" || !Number.isFinite(value)) return;
          const next: [number, number] = point ? [point[0], point[1]] : [0, 0];
          next[axis] = value;
          onMark(next);
        }}
      />
    </label>
  );

  return (
    <div className="mt-3">
      <p className="text-sm">
        Click the plan where <strong>{landmark.id}</strong> is.
      </p>
      <div className="relative mt-1 overflow-hidden rounded-md border border-stone-300 dark:border-stone-700">
        {/* Signed, short-lived storage URL: next/image optimization is not appropriate here. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={plan.url}
          alt={`Floor plan. Click to mark ${landmark.id}.`}
          referrerPolicy="no-referrer"
          draggable={false}
          className="block w-full cursor-crosshair select-none"
          onLoad={(e) => {
            const img = e.currentTarget;
            if (img.naturalWidth > 0) setAspect(img.naturalHeight / img.naturalWidth);
          }}
          onError={() => {
            // The link may have expired: fetch a fresh one once per link.
            if (retried === plan.url) return;
            setRetried(plan.url);
            void onRefresh();
          }}
          onClick={mark}
        />
        {aspect ? (
          <svg
            className="pointer-events-none absolute inset-0 h-full w-full"
            viewBox={`0 0 1 ${aspect}`}
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            {landmarks.map((lm, i) =>
              lm.planPoint ? (
                <circle
                  key={lm.id}
                  cx={lm.planPoint[0]}
                  cy={lm.planPoint[1]}
                  r={i === index ? 0.012 : 0.008}
                  fill={i === index ? "#dc2626" : "#7c3aed"}
                  stroke="white"
                  strokeWidth={0.003}
                />
              ) : null,
            )}
          </svg>
        ) : null}
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2">
        {coordinate(0, "Plan x (fraction of plan width)")}
        {coordinate(1, "Plan y (fraction of plan width)")}
      </div>
      {point ? (
        <Button variant="ghost" onClick={() => onMark(null)}>
          Clear the plan mark
        </Button>
      ) : null}
    </div>
  );
}
