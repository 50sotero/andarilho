"use client";

import { useRef, type KeyboardEvent, type PointerEvent } from "react";
import type { Layout } from "@/lib/api/types";
import { pointAlong, round, type Point, type ViewBox } from "./geometry";
import { describeSelection, movePoint, pointOf, type Selection } from "./layout-edit";

function sameSelection(a: Selection | null, b: Selection): boolean {
  return a !== null && JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Interactive top-down SVG editor. Point handles are focusable buttons:
 * drag with a pointer, or move with arrow keys (Shift = 0.5 m, otherwise 0.05 m).
 */
export function LayoutCanvas({
  layout,
  viewBox,
  selection,
  onSelect,
  onChange,
  showOverlay,
}: {
  layout: Layout;
  viewBox: ViewBox;
  selection: Selection | null;
  onSelect: (sel: Selection | null) => void;
  onChange: (layout: Layout) => void;
  showOverlay: boolean;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<Selection | null>(null);
  const scale = Math.max(viewBox.width, viewBox.height);
  const handleR = scale / 90;
  const stroke = scale / 400;

  function toWorld(event: PointerEvent): Point | null {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return null;
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    return [round(pt.x, 2), round(pt.y, 2)];
  }

  function startDrag(event: PointerEvent<SVGElement>, sel: Selection) {
    event.preventDefault();
    event.stopPropagation();
    onSelect(sel);
    dragRef.current = sel;
    svgRef.current?.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    const sel = dragRef.current;
    if (!sel) return;
    const p = toWorld(event);
    if (p) onChange(movePoint(layout, sel, p));
  }

  function endDrag(event: PointerEvent<SVGSVGElement>) {
    if (dragRef.current && svgRef.current?.hasPointerCapture(event.pointerId)) {
      svgRef.current.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
  }

  function onHandleKey(event: KeyboardEvent<SVGElement>, sel: Selection) {
    const step = event.shiftKey ? 0.5 : 0.05;
    const delta: Record<string, Point> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    const d = delta[event.key];
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(sel);
      return;
    }
    if (!d) return;
    event.preventDefault();
    const p = pointOf(layout, sel);
    if (!p) return;
    onSelect(sel);
    onChange(movePoint(layout, sel, [p[0] + d[0], p[1] + d[1]]));
  }

  function handle(sel: Selection, p: Point, color: string, shape: "circle" | "square" | "diamond" = "circle") {
    const selected = sameSelection(selection, sel);
    const label = `${describeSelection(layout, sel)} at x ${p[0].toFixed(2)} m, z ${p[1].toFixed(2)} m`;
    const common = {
      role: "button",
      tabIndex: 0,
      "aria-label": label,
      "aria-pressed": selected,
      onPointerDown: (e: PointerEvent<SVGElement>) => startDrag(e, sel),
      onKeyDown: (e: KeyboardEvent<SVGElement>) => onHandleKey(e, sel),
      fill: selected ? "#facc15" : color,
      stroke: "#1c1917",
      strokeWidth: stroke,
      style: { cursor: "grab", touchAction: "none" } as const,
      className: "focus:outline-none focus-visible:stroke-teal-500",
    };
    const r = selected ? handleR * 1.4 : handleR;
    if (shape === "square") {
      return <rect {...common} x={p[0] - r} y={p[1] - r} width={r * 2} height={r * 2} />;
    }
    if (shape === "diamond") {
      return (
        <polygon
          {...common}
          points={`${p[0]},${p[1] - r} ${p[0] + r},${p[1]} ${p[0]},${p[1] + r} ${p[0] - r},${p[1]}`}
        />
      );
    }
    return <circle {...common} cx={p[0]} cy={p[1]} r={r} />;
  }

  const walls = layout.walls ?? [];
  const wallById = new Map(walls.map((w) => [w.id, w]));
  const spawn: Point = [layout.spawn.feet[0], layout.spawn.feet[2]];
  // Camera forward is -Z at yaw 0; yaw rotates about +Y.
  const yaw = layout.spawn.yawRadians;
  const arrow: Point = [spawn[0] - Math.sin(yaw) * handleR * 4, spawn[1] - Math.cos(yaw) * handleR * 4];

  return (
    <svg
      ref={svgRef}
      viewBox={`${viewBox.minX} ${viewBox.minY} ${viewBox.width} ${viewBox.height}`}
      className="h-auto w-full touch-none select-none rounded-md border border-stone-300 bg-white dark:border-stone-700 dark:bg-stone-100"
      role="group"
      aria-label="Layout editor. Tab to a handle, then use arrow keys to move it."
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onPointerDown={() => onSelect(null)}
    >
      <defs>
        <pattern id="grid-1m" width="1" height="1" patternUnits="userSpaceOnUse">
          <path d="M 1 0 L 0 0 0 1" fill="none" stroke="#e7e5e4" strokeWidth={stroke} />
        </pattern>
      </defs>
      <rect x={viewBox.minX} y={viewBox.minY} width={viewBox.width} height={viewBox.height} fill="url(#grid-1m)" />

      {layout.floors.map((floor, fi) =>
        floor.polygons.map((ring, ri) => (
          <polygon
            key={`${floor.id}-${ri}`}
            points={ring.map((p) => p.join(",")).join(" ")}
            fill={showOverlay ? "rgba(16,185,129,0.25)" : "#ccfbf1"}
            stroke="#0f766e"
            strokeWidth={stroke * 2}
            data-floor={fi}
          />
        )),
      )}

      {(layout.obstacles ?? []).map((ob, oi) => (
        <polygon
          key={ob.id}
          points={ob.footprint.map((p) => p.join(",")).join(" ")}
          fill={showOverlay ? "rgba(239,68,68,0.45)" : "#fde68a"}
          stroke="#92400e"
          strokeWidth={stroke * 2}
          onPointerDown={(e) => {
            e.stopPropagation();
            onSelect({ kind: "obstacle", index: oi });
          }}
          style={{ cursor: "pointer" }}
        />
      ))}

      {walls.map((w, wi) => (
        <line
          key={w.id}
          x1={w.start[0]}
          y1={w.start[1]}
          x2={w.end[0]}
          y2={w.end[1]}
          stroke={sameSelection(selection, { kind: "wall", index: wi }) ? "#ca8a04" : showOverlay ? "#dc2626" : "#1c1917"}
          strokeWidth={Math.max(w.thickness, stroke * 3)}
          strokeLinecap="square"
          onPointerDown={(e) => {
            e.stopPropagation();
            onSelect({ kind: "wall", index: wi });
          }}
          style={{ cursor: "pointer" }}
        />
      ))}

      {(layout.openings ?? []).map((o, oi) => {
        const wall = wallById.get(o.wallId);
        if (!wall) return null;
        const a = pointAlong(wall.start, wall.end, o.offset);
        const b = pointAlong(wall.start, wall.end, o.offset + o.width);
        const selected = sameSelection(selection, { kind: "opening", index: oi });
        return (
          <line
            key={o.id}
            x1={a[0]}
            y1={a[1]}
            x2={b[0]}
            y2={b[1]}
            stroke={selected ? "#facc15" : o.state === "closed" ? "#9a3412" : "#ffffff"}
            strokeWidth={Math.max(wall.thickness, stroke * 3) * 1.3}
            onPointerDown={(e) => {
              e.stopPropagation();
              onSelect({ kind: "opening", index: oi });
            }}
            style={{ cursor: "pointer" }}
          />
        );
      })}

      {layout.floors.map((floor, fi) =>
        floor.polygons.map((ring, ri) =>
          ring.map((p, pi) => (
            <g key={`${floor.id}-${ri}-${pi}`}>
              {handle({ kind: "floorCorner", floor: fi, ring: ri, index: pi }, p as Point, "#14b8a6")}
            </g>
          )),
        ),
      )}
      {walls.map((w, wi) => (
        <g key={`${w.id}-ends`}>
          {handle({ kind: "wallEnd", index: wi, end: "start" }, w.start, "#57534e", "square")}
          {handle({ kind: "wallEnd", index: wi, end: "end" }, w.end, "#57534e", "square")}
        </g>
      ))}
      {(layout.obstacles ?? []).map((ob, oi) =>
        ob.footprint.map((p, ci) => (
          <g key={`${ob.id}-${ci}`}>{handle({ kind: "obstacleCorner", index: oi, corner: ci }, p as Point, "#f59e0b")}</g>
        )),
      )}
      {(layout.landmarks ?? []).map((lm, li) => (
        <g key={lm.id}>
          {handle({ kind: "landmark", index: li }, [lm.worldPoint[0], lm.worldPoint[2]], "#6366f1", "diamond")}
        </g>
      ))}
      <line x1={spawn[0]} y1={spawn[1]} x2={arrow[0]} y2={arrow[1]} stroke="#be123c" strokeWidth={stroke * 3} />
      {handle({ kind: "spawn" }, spawn, "#e11d48")}
    </svg>
  );
}
