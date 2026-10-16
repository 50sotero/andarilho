import { furnitureFootprint } from "@andarilho/viewer";
import type { Layout } from "@/lib/api/types";
import { layoutViewBox, pointAlong, type Point } from "./geometry";

/** Read-only top-down plan of a confirmed layout (non-3D fallback). */
export function LayoutPlan({ layout, title }: { layout: Layout; title: string }) {
  const vb = layoutViewBox(layout);
  const stroke = Math.max(vb.width, vb.height) / 300;
  const walls = new Map((layout.walls ?? []).map((w) => [w.id, w]));
  return (
    <figure className="space-y-1">
      <svg
        viewBox={`${vb.minX} ${vb.minY} ${vb.width} ${vb.height}`}
        role="img"
        aria-label={`Floor plan of ${title}`}
        className="h-auto w-full max-w-2xl rounded-md bg-white dark:bg-stone-100"
      >
        {layout.floors.map((floor) =>
          floor.polygons.map((ring, i) => (
            <polygon
              key={`${floor.id}-${i}`}
              points={ring.map((p) => p.join(",")).join(" ")}
              fill="#ccfbf1"
              stroke="#0f766e"
              strokeWidth={stroke}
            />
          )),
        )}
        {(layout.obstacles ?? []).map((o) => (
          <polygon
            key={o.id}
            points={o.footprint.map((p) => p.join(",")).join(" ")}
            fill="#fde68a"
            stroke="#92400e"
            strokeWidth={stroke}
          />
        ))}
        {(layout.furniture ?? []).map((item) => {
          const solid = item.solid !== false;
          const label = item.label || item.category;
          return (
            <polygon
              key={item.id}
              data-furniture={item.id}
              points={furnitureFootprint(item)
                .map((p) => p.join(","))
                .join(" ")}
              fill="#d6d3d1"
              fillOpacity={solid ? 1 : 0.4}
              stroke="#57534e"
              strokeWidth={stroke}
              strokeDasharray={solid ? undefined : `${stroke * 3} ${stroke * 2}`}
            >
              <title>{solid ? label : `${label} (does not block walking)`}</title>
            </polygon>
          );
        })}
        {(layout.walls ?? []).map((w) => (
          <line
            key={w.id}
            x1={w.start[0]}
            y1={w.start[1]}
            x2={w.end[0]}
            y2={w.end[1]}
            stroke="#1c1917"
            strokeWidth={Math.max(w.thickness, stroke * 2)}
          />
        ))}
        {(layout.openings ?? []).map((o) => {
          const wall = walls.get(o.wallId);
          if (!wall) return null;
          const a = pointAlong(wall.start as Point, wall.end as Point, o.offset);
          const b = pointAlong(wall.start as Point, wall.end as Point, o.offset + o.width);
          return (
            <line
              key={o.id}
              x1={a[0]}
              y1={a[1]}
              x2={b[0]}
              y2={b[1]}
              stroke={o.state === "closed" ? "#7c2d12" : "#ffffff"}
              strokeWidth={Math.max(wall.thickness, stroke * 2) * 1.2}
            />
          );
        })}
        <circle cx={layout.spawn.feet[0]} cy={layout.spawn.feet[2]} r={stroke * 5} fill="#be123c" />
      </svg>
      <figcaption className="text-xs text-stone-600 dark:text-stone-400">
        Top-down plan: walkable floor (teal), walls (black), openings (gaps), obstacles (yellow),
        furniture (grey), starting point (red dot).
      </figcaption>
    </figure>
  );
}
