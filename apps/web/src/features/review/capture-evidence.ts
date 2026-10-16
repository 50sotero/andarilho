import type { CaptureEvidenceLink, Layout } from "@/lib/api/types";
import type { Point } from "./geometry";

/**
 * Capture evidence: the sparse points and camera path the worker recovered from the
 * video, in the run's estimated frame. The editor draws it under the walls so the
 * layout can be checked against what the camera saw. It is never part of a saved
 * layout, and it is not calibrated: the layout's render scale maps it onto the plan.
 */

export const EVIDENCE_SCHEMA_VERSION = 1;

/**
 * Most points, and most camera positions, the editor will accept. The worker keeps
 * at most 5000 points; a larger file is not what the contract describes.
 */
export const EVIDENCE_MAX_ITEMS = 5000;

/**
 * Calibrated heights, in metres above the floor, whose points are drawn. Points
 * outside this band are floor and ceiling clutter that would hide the walls.
 */
export const EVIDENCE_HEIGHT_BAND_M = { min: 0.1, max: 2.2 } as const;

export type Vec3 = [number, number, number];

/** The validated evidence file: [x, y, z] in metres, in the estimated frame. */
export interface CaptureEvidence {
  points: Vec3[];
  cameras: Vec3[];
}

/** The evidence projected onto the plan, in layout units: [x, z] pairs. */
export interface EvidenceOnPlan {
  points: Point[];
  cameraPath: Point[];
}

function parseVectors(value: unknown): Vec3[] | null {
  if (!Array.isArray(value) || value.length > EVIDENCE_MAX_ITEMS) return null;
  const vectors: Vec3[] = [];
  for (const item of value) {
    if (!Array.isArray(item) || item.length !== 3) return null;
    if (!item.every((n) => typeof n === "number" && Number.isFinite(n))) return null;
    vectors.push([item[0], item[1], item[2]]);
  }
  return vectors;
}

/**
 * Validates the JSON behind an evidence link. A file that is malformed in any part
 * is rejected as a whole: a partly read file would show a misleading picture.
 */
export function parseCaptureEvidence(value: unknown): CaptureEvidence | null {
  if (!value || typeof value !== "object") return null;
  const body = value as Record<string, unknown>;
  if (body.schemaVersion !== EVIDENCE_SCHEMA_VERSION || body.units !== "m") return null;
  const points = parseVectors(body.points);
  const cameras = parseVectors(body.cameras);
  if (!points || !cameras) return null;
  return { points, cameras };
}

/**
 * The factor from the estimated frame to the layout's frame: the calibration's
 * cumulative render scale, or 1 before the user calibrates. A value that is not a
 * positive finite number counts as uncalibrated rather than drawing things wrongly.
 */
export function evidenceScale(calibration: Layout["calibration"]): number {
  const scale = calibration?.renderScale;
  return typeof scale === "number" && Number.isFinite(scale) && scale > 0 ? scale : 1;
}

/**
 * Projects the evidence onto the plan. Horizontal positions are multiplied by the
 * scale, because the evidence is stored uncalibrated. The height band is checked on
 * calibrated heights (y times the scale), so it means the same real-world heights
 * whatever the scale is.
 */
export function evidenceOnPlan(evidence: CaptureEvidence, scale: number): EvidenceOnPlan {
  const { min, max } = EVIDENCE_HEIGHT_BAND_M;
  const points: Point[] = [];
  for (const [x, y, z] of evidence.points) {
    const height = y * scale;
    if (height >= min && height <= max) points.push([x * scale, z * scale]);
  }
  return {
    points,
    cameraPath: evidence.cameras.map(([x, , z]): Point => [x * scale, z * scale]),
  };
}

/**
 * Downloads the evidence behind a link. Evidence only adds context, so every failure
 * (an expired link, a refused request, a malformed file) resolves to null and the
 * editor shows no layer. The request carries no credentials: the link is the grant.
 * Its expiry is left to storage, which refuses an expired link: this device's clock
 * may be off from the server's that set it.
 */
export async function loadCaptureEvidence(
  link: CaptureEvidenceLink,
  fetchImpl: typeof fetch = fetch,
): Promise<CaptureEvidence | null> {
  try {
    const response = await fetchImpl(link.url, { credentials: "omit" });
    if (!response.ok) return null;
    return parseCaptureEvidence(await response.json());
  } catch {
    return null;
  }
}
