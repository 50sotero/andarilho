import {
  BufferAttribute,
  BufferGeometry,
  CircleGeometry,
  ConeGeometry,
  DoubleSide,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  RingGeometry,
  WireframeGeometry,
} from "three";
import type { CollisionMesh, Vec3 } from "../types";

/**
 * Collision overlay (debug and review): a transparent fill plus wireframe built
 * directly from the world-space collision triangles, a spawn marker with a yaw
 * arrow, and an optional player-footprint ring. Geometry is already in world
 * space, so the overlay group keeps an identity transform.
 */

export interface CollisionOverlayOptions {
  spawn?: { feet: Vec3; yawRadians: number } | null;
  /** Player capsule radius for the footprint ring. */
  playerRadius?: number;
  fillColor?: number;
  lineColor?: number;
  opacity?: number;
}

export interface CollisionOverlay {
  readonly group: Group;
  setVisible(visible: boolean): void;
  readonly visible: boolean;
  /** Moves the player footprint marker (feet position). */
  setPlayerFeet(feet: Vec3 | null): void;
  dispose(): void;
}

export function createCollisionOverlay(
  mesh: CollisionMesh,
  options: CollisionOverlayOptions = {},
): CollisionOverlay {
  const group = new Group();
  group.name = "andarilho-collision-overlay";
  group.visible = false;
  group.renderOrder = 10;
  const geometries: BufferGeometry[] = [];
  const materials: (MeshBasicMaterial | LineBasicMaterial)[] = [];

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(mesh.positions), 3));
  geometry.setIndex(new BufferAttribute(new Uint32Array(mesh.indices), 1));
  geometries.push(geometry);

  const fill = new MeshBasicMaterial({
    color: options.fillColor ?? 0x22c55e,
    transparent: true,
    opacity: options.opacity ?? 0.18,
    side: DoubleSide,
    depthWrite: false,
  });
  materials.push(fill);
  group.add(new Mesh(geometry, fill));

  const wire = new WireframeGeometry(geometry);
  geometries.push(wire);
  const lineMaterial = new LineBasicMaterial({
    color: options.lineColor ?? 0x16a34a,
    transparent: true,
    opacity: 0.6,
    depthWrite: false,
  });
  materials.push(lineMaterial);
  group.add(new LineSegments(wire, lineMaterial));

  const markerMaterial = new MeshBasicMaterial({
    color: 0xf59e0b,
    side: DoubleSide,
    depthTest: false,
    transparent: true,
    opacity: 0.9,
  });
  materials.push(markerMaterial);
  const radius = options.playerRadius ?? 0.25;

  if (options.spawn) {
    const { feet, yawRadians } = options.spawn;
    const spawnGroup = new Group();
    spawnGroup.position.set(feet[0], feet[1] + 0.005, feet[2]);
    spawnGroup.rotation.set(0, yawRadians, 0, "YXZ");
    const disc = new CircleGeometry(radius, 32);
    disc.rotateX(-Math.PI / 2);
    geometries.push(disc);
    spawnGroup.add(new Mesh(disc, markerMaterial));
    const arrow = new ConeGeometry(radius * 0.4, radius * 1.2, 12);
    // Cone points +Y; rotate so it points toward camera forward (-Z) at yaw 0.
    arrow.rotateX(-Math.PI / 2);
    arrow.translate(0, 0.02, -radius * 1.4);
    geometries.push(arrow);
    spawnGroup.add(new Mesh(arrow, markerMaterial));
    group.add(spawnGroup);
  }

  const ringGeometry = new RingGeometry(radius * 0.92, radius, 32);
  ringGeometry.rotateX(-Math.PI / 2);
  geometries.push(ringGeometry);
  const ringMaterial = new MeshBasicMaterial({
    color: 0x38bdf8,
    side: DoubleSide,
    depthTest: false,
  });
  materials.push(ringMaterial);
  const ring = new Mesh(ringGeometry, ringMaterial);
  ring.visible = false;
  group.add(ring);

  let disposed = false;
  return {
    group,
    get visible() {
      return group.visible;
    },
    setVisible(visible: boolean) {
      group.visible = visible;
    },
    setPlayerFeet(feet: Vec3 | null) {
      if (!feet) {
        ring.visible = false;
        return;
      }
      ring.visible = true;
      ring.position.set(feet[0], feet[1] + 0.01, feet[2]);
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      group.removeFromParent();
      for (const g of geometries) g.dispose();
      for (const m of materials) m.dispose();
    },
  };
}
