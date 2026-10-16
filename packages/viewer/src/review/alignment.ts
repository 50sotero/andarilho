import type { Vec2 } from "../types";

/**
 * 2D similarity fit on the horizontal plane (README section 9): uniform scale,
 * rotation and translation from plan landmarks to world (x, z) landmarks via
 * closed-form least squares (Umeyama in 2D). Reflections are never produced;
 * when the data is clearly better explained by a mirror image the fit is
 * rejected instead of silently warping or flipping the scene.
 */

export interface LandmarkPair {
  id?: string;
  /** Point in plan coordinates (after the plan's distance calibration). */
  plan: Vec2;
  /** Corresponding world point projected on the floor: (x, z) in meters. */
  world: Vec2;
}

export interface SimilarityFit {
  /** Positive uniform scale (world meters per plan unit). */
  scale: number;
  /**
   * Counter-clockwise rotation in the (x, z) coordinate plane, radians:
   * world = scale * R(rotation) * plan + translation, R = [[c, -s], [s, c]].
   */
  rotationRadians: number;
  /**
   * The same rotation expressed as a right-handed yaw about +Y (the convention
   * used by spawn yaw and Three.js rotation.y). Equal to -rotationRadians.
   */
  yawRadians: number;
  translation: Vec2;
  /** Per-pair residual distance in world meters, in input order. */
  residuals: number[];
  rmsResidual: number;
  maxResidual: number;
  /** True when some anchor disagrees with the fit by more than the tolerance. */
  disagreement: boolean;
  /** Pair indices whose residual exceeds the tolerance. */
  disagreeingPairs: number[];
  /** Two anchors fit exactly and provide no independent error check. */
  underdetermined: boolean;
  /** Plan anchors are (nearly) collinear; rotation is poorly constrained. */
  collinear: boolean;
}

export type SimilarityFitFailure =
  "too-few-pairs" | "non-finite" | "degenerate-plan" | "degenerate-world" | "reflection";

export type SimilarityFitResult =
  { ok: true; fit: SimilarityFit } | { ok: false; reason: SimilarityFitFailure; message: string };

export interface SimilarityFitOptions {
  /** Residual tolerance in meters for flagging disagreement (default 0.1). */
  tolerance?: number;
  /** Optional per-pair weights (e.g. inverse uncertainty). */
  weights?: number[];
}

export function applySimilarity(
  fit: Pick<SimilarityFit, "scale" | "rotationRadians" | "translation">,
  p: Vec2,
): Vec2 {
  const c = Math.cos(fit.rotationRadians);
  const s = Math.sin(fit.rotationRadians);
  return [
    fit.scale * (c * p[0] - s * p[1]) + fit.translation[0],
    fit.scale * (s * p[0] + c * p[1]) + fit.translation[1],
  ];
}

function fail(reason: SimilarityFitFailure, message: string): SimilarityFitResult {
  return { ok: false, reason, message };
}

export function fitSimilarity2D(
  pairs: readonly LandmarkPair[],
  options: SimilarityFitOptions = {},
): SimilarityFitResult {
  if (pairs.length < 2) return fail("too-few-pairs", "At least two landmark pairs are required.");
  const finite = pairs.every(
    (p) => [...p.plan, ...p.world].every((v) => Number.isFinite(v)) && p.plan.length === 2,
  );
  if (!finite) return fail("non-finite", "Landmark coordinates must be finite numbers.");
  const weights = pairs.map((_, i) => {
    const w = options.weights?.[i];
    return w !== undefined && Number.isFinite(w) && w > 0 ? w : 1;
  });
  const wSum = weights.reduce((a, b) => a + b, 0);

  let pcx = 0;
  let pcy = 0;
  let wcx = 0;
  let wcy = 0;
  pairs.forEach((pair, i) => {
    const w = weights[i]!;
    pcx += w * pair.plan[0];
    pcy += w * pair.plan[1];
    wcx += w * pair.world[0];
    wcy += w * pair.world[1];
  });
  pcx /= wSum;
  pcy /= wSum;
  wcx /= wSum;
  wcy /= wSum;

  let varPlan = 0;
  let varWorld = 0;
  let a = 0; // sum p . q
  let b = 0; // sum p x q
  let ar = 0; // reflected-fit terms
  let br = 0;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  pairs.forEach((pair, i) => {
    const w = weights[i]!;
    const px = pair.plan[0] - pcx;
    const py = pair.plan[1] - pcy;
    const qx = pair.world[0] - wcx;
    const qy = pair.world[1] - wcy;
    varPlan += w * (px * px + py * py);
    varWorld += w * (qx * qx + qy * qy);
    a += w * (px * qx + py * qy);
    b += w * (px * qy - py * qx);
    ar += w * (px * qx - py * qy);
    br += w * (px * qy + py * qx);
    sxx += w * px * px;
    sxy += w * px * py;
    syy += w * py * py;
  });

  const spread = Math.sqrt(varPlan / wSum);
  if (!(spread > 1e-9)) return fail("degenerate-plan", "Plan landmarks coincide.");
  if (!(Math.sqrt(varWorld / wSum) > 1e-9)) {
    return fail("degenerate-world", "World landmarks coincide.");
  }

  const proper = Math.hypot(a, b);
  const mirrored = Math.hypot(ar, br);
  // Collinearity from the plan covariance eigenvalues.
  const trace = sxx + syy;
  const det = sxx * syy - sxy * sxy;
  const disc = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));
  const minEig = trace / 2 - disc;
  const maxEig = trace / 2 + disc;
  const collinear = maxEig > 0 ? minEig / maxEig < 1e-4 : true;

  if (pairs.length >= 3 && !collinear && mirrored > proper * 1.05 + 1e-12) {
    // A proper similarity explains less variance than a mirror image: the plan
    // is flipped relative to the capture. Reject rather than fit a reflection.
    return fail(
      "reflection",
      "Landmarks imply a mirrored plan. Check the plan orientation or landmark order.",
    );
  }

  const rotation = Math.atan2(b, a);
  const scale = proper / varPlan;
  if (!(scale > 0) || !Number.isFinite(scale)) {
    return fail("degenerate-world", "Could not determine a positive scale.");
  }
  const c = Math.cos(rotation);
  const s = Math.sin(rotation);
  const translation: Vec2 = [wcx - scale * (c * pcx - s * pcy), wcy - scale * (s * pcx + c * pcy)];
  const partial = { scale, rotationRadians: rotation, translation };
  const tolerance = options.tolerance ?? 0.1;
  const residuals = pairs.map((pair) => {
    const m = applySimilarity(partial, pair.plan);
    return Math.hypot(m[0] - pair.world[0], m[1] - pair.world[1]);
  });
  const rms = Math.sqrt(residuals.reduce((sum, r) => sum + r * r, 0) / residuals.length);
  const disagreeingPairs = residuals.flatMap((r, i) => (r > tolerance ? [i] : []));

  return {
    ok: true,
    fit: {
      scale,
      rotationRadians: rotation,
      yawRadians: rotation === 0 ? 0 : -rotation,
      translation,
      residuals,
      rmsResidual: rms,
      maxResidual: Math.max(...residuals),
      disagreement: disagreeingPairs.length > 0,
      disagreeingPairs,
      underdetermined: pairs.length < 3,
      collinear,
    },
  };
}
