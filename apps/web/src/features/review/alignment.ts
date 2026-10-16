import {
  applySimilarity,
  fitSimilarity2D,
  type LandmarkPair,
  type SimilarityFit,
  type SimilarityFitResult,
} from "@andarilho/viewer/alignment";
import type { Layout } from "@/lib/api/types";
import { round } from "./geometry";
import type { PlanUnderlay } from "./LayoutCanvas";

/**
 * Plan-to-capture landmark alignment (README section 9, user-assisted alignment).
 *
 * A landmark's `planPoint` is stored in plan-image units: fractions of the
 * image's width on both axes (x right, y down from the top-left corner), so the
 * fitted scale is the plan's width in meters and the underlay can be placed
 * from the fit directly. Its `worldPoint` is the same feature in the capture.
 */

/** Residual (m) above which a landmark is reported as disagreeing with the fit. */
export const ALIGNMENT_TOLERANCE_M = 0.1;

export interface PlanAlignment {
  /** Landmark ids in fit order (only landmarks marked on the plan take part). */
  ids: string[];
  result: SimilarityFitResult;
  /**
   * The one landmark whose removal makes all the others agree, with its distance
   * from where that fit puts it (needs four or more landmarks: three still check).
   */
  suspect: { id: string; residual: number } | null;
}

export function landmarkPairs(layout: Layout): (LandmarkPair & { uncertainty: number })[] {
  return (layout.landmarks ?? []).flatMap((lm) =>
    lm.planPoint
      ? [
          {
            id: lm.id,
            plan: [lm.planPoint[0], lm.planPoint[1]] as [number, number],
            world: [lm.worldPoint[0], lm.worldPoint[2]] as [number, number],
            uncertainty: lm.uncertainty ?? 0,
          },
        ]
      : [],
  );
}

/** The similarity fit from plan to capture, or null with fewer than two marked landmarks. */
export function alignPlan(layout: Layout): PlanAlignment | null {
  const pairs = landmarkPairs(layout);
  if (pairs.length < 2) return null;
  // Inverse-variance weights: a landmark placed with less certainty pulls the fit less.
  const weights = pairs.map((p) => 1 / Math.max(p.uncertainty, 0.01) ** 2);
  const options = { tolerance: ALIGNMENT_TOLERANCE_M };
  const result = fitSimilarity2D(pairs, { ...options, weights });
  let suspect: PlanAlignment["suspect"] = null;
  if (result.ok && result.fit.disagreement && pairs.length >= 4) {
    // Least squares spreads one bad mark over its neighbours. Refit without each
    // landmark in turn: leaving out the bad one makes the rest agree far better
    // than leaving out any other. Name it only when that is clear-cut.
    const refits = pairs
      .map((pair, i) => {
        const rest = fitSimilarity2D(
          pairs.filter((_, j) => j !== i),
          { ...options, weights: weights.filter((_, j) => j !== i) },
        );
        if (!rest.ok) return null;
        const [x, z] = applySimilarity(rest.fit, pair.plan);
        const residual = Math.hypot(x - pair.world[0], z - pair.world[1]);
        return { id: pair.id ?? "", residual, rest: rest.fit };
      })
      .filter((r) => r !== null)
      .sort((a, b) => a.rest.rmsResidual - b.rest.rmsResidual);
    const [best, next] = refits;
    if (
      best &&
      !best.rest.disagreement &&
      best.residual > ALIGNMENT_TOLERANCE_M &&
      (!next || best.rest.rmsResidual < 0.5 * next.rest.rmsResidual)
    ) {
      suspect = { id: best.id, residual: best.residual };
    }
  }
  return { ids: pairs.map((p) => p.id ?? ""), result, suspect };
}

/** Places the plan image so its landmarks land on their capture positions. */
export function underlayFromFit(fit: SimilarityFit, url: string, opacity: number): PlanUnderlay {
  return {
    url,
    x: round(fit.translation[0]),
    z: round(fit.translation[1]),
    width: round(fit.scale),
    rotationDegrees: round((fit.rotationRadians * 180) / Math.PI, 2),
    opacity,
  };
}

export function setPlanPoint(
  layout: Layout,
  index: number,
  point: [number, number] | null,
): Layout {
  const landmarks = layout.landmarks ?? [];
  if (!landmarks[index]) return layout;
  return {
    ...layout,
    landmarks: landmarks.map((lm, i) =>
      i === index
        ? { ...lm, planPoint: point ? [round(point[0], 4), round(point[1], 4)] : null }
        : lm,
    ),
  };
}

/** Plain-language findings for the review panel: the fit, then anything to check. */
export function describeAlignment(alignment: PlanAlignment): {
  summary: string[];
  warnings: string[];
} {
  const { result, ids } = alignment;
  if (!result.ok) {
    const reason =
      result.reason === "reflection"
        ? "The landmarks only fit a mirror image of the plan. Check that each landmark is " +
          "marked on the matching plan feature; the plan is never flipped to fit."
        : result.message;
    return { summary: [], warnings: [reason] };
  }
  const fit = result.fit;
  const summary = [
    `The plan is ${fit.scale.toFixed(2)} m wide and turned ${round(
      (fit.rotationRadians * 180) / Math.PI,
      1,
    )}° against the capture.`,
    `Landmarks disagree by ${fit.rmsResidual.toFixed(2)} m on average (RMS), ` +
      `${fit.maxResidual.toFixed(2)} m at most.`,
  ];
  const warnings: string[] = [];
  if (ids.length === 2) {
    warnings.push(
      "Two landmarks place the plan but cannot check it: mark a third, away from the line " +
        "through the first two.",
    );
  } else if (fit.collinear) {
    warnings.push(
      "The landmarks lie almost on one line, so the fit is weak across it: mark one away " +
        "from that line.",
    );
  }
  if (fit.disagreement && alignment.suspect) {
    const { id, residual } = alignment.suspect;
    warnings.push(
      `${id} looks misplaced: the other landmarks agree without it, and it is ` +
        `${residual.toFixed(2)} m from where they put it. Check its marks on the plan and ` +
        "in the capture.",
    );
  } else if (fit.disagreement) {
    const names = fit.disagreeingPairs.map((i) => ids[i]).join(", ");
    warnings.push(
      `${names} ${fit.disagreeingPairs.length === 1 ? "is" : "are"} more than ` +
        `${ALIGNMENT_TOLERANCE_M.toFixed(2)} m from where the plan puts ` +
        `${fit.disagreeingPairs.length === 1 ? "it" : "them"}. Check the marks; if they are ` +
        "right, the plan does not match the captured space there.",
    );
  }
  return { summary, warnings };
}
