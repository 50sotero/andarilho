import type { DeliveryEnvelope, Layout, SceneManifest } from "@andarilho/scene-contract";
import { decomposeRenderTransform } from "./render/transform";
import type { RenderFormat } from "./render/spark-scene";

/**
 * Viewer-side checks of the persistent manifest and the short-lived delivery
 * envelope (README section 10). The API validates against the JSON Schema;
 * these checks protect the engine against values that would break rendering or
 * movement (non-finite numbers, wrong frames, mismatched revisions).
 */

/** Viewer input: a validated delivery envelope, or explicit URLs in dev/fixture mode. */
export type ViewerSource =
  | { kind: "delivery"; envelope: DeliveryEnvelope }
  | {
      kind: "urls";
      manifest: SceneManifest;
      renderUrl: string;
      /** World-space collision GLB URL. */
      collisionUrl?: string | null;
      /**
       * Confirmed layout used to build collision in the browser when no GLB is
       * given (development only; production walking uses the packaged GLB).
       */
      layout?: Layout | null;
    };

export interface ResolvedSource {
  manifest: SceneManifest;
  renderUrl: string;
  renderFormat: RenderFormat;
  collisionUrl: string | null;
  /** Expected SHA-256 (hex) of the collision asset, verified when present. */
  collisionSha256: string | null;
  layout: Layout | null;
  posterUrl: string | null;
}

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const finiteTuple = (v: unknown, n: number): boolean =>
  Array.isArray(v) && v.length === n && v.every(finite);

export function sceneManifestIssues(manifest: SceneManifest): string[] {
  const issues: string[] = [];
  if (!manifest || typeof manifest !== "object") return ["manifest is missing"];
  if (manifest.schemaVersion !== "1.0") issues.push("unsupported manifest schemaVersion");
  if (manifest.units !== "m") issues.push("units must be meters");
  if (manifest.upAxis !== "+Y") issues.push("upAxis must be +Y");
  if (manifest.cameraForward !== "-Z") issues.push("cameraForward must be -Z");
  if (manifest.geometrySpace !== "world") issues.push("geometrySpace must be world");
  try {
    decomposeRenderTransform(manifest.worldFromDecodedRender);
  } catch (error) {
    issues.push(error instanceof Error ? error.message : "invalid worldFromDecodedRender");
  }
  const b = manifest.bounds;
  if (!b || !finiteTuple(b.min, 3) || !finiteTuple(b.max, 3)) {
    issues.push("bounds must be finite");
  } else if (!b.min.every((v, i) => v < (b.max[i] as number))) {
    issues.push("bounds min must be below max");
  }
  const s = manifest.spawn;
  if (!s || !finiteTuple(s.feet, 3) || !finite(s.yawRadians)) {
    issues.push("spawn must be finite");
  } else if (b && finiteTuple(b.min, 3) && finiteTuple(b.max, 3)) {
    const inside = s.feet.every(
      (v, i) => v >= (b.min[i] as number) - 1e-6 && v <= (b.max[i] as number) + 1e-6,
    );
    if (!inside) issues.push("spawn is outside the scene bounds");
  }
  const nav = manifest.navigation;
  if (!nav) {
    issues.push("navigation is missing");
  } else {
    for (const key of ["playerRadius", "playerHeight", "eyeHeight"] as const) {
      const v = nav[key];
      if (v !== undefined && !(finite(v) && v > 0 && v < 5))
        issues.push(`navigation.${key} is invalid`);
    }
  }
  return issues;
}

export function deliveryEnvelopeIssues(envelope: DeliveryEnvelope, nowMs = Date.now()): string[] {
  if (!envelope || typeof envelope !== "object") return ["delivery envelope is missing"];
  const issues = sceneManifestIssues(envelope.manifest);
  if (envelope.manifest && envelope.sceneRevisionId !== envelope.manifest.sceneRevisionId) {
    issues.push("envelope and manifest refer to different scene revisions");
  }
  const expires = Date.parse(envelope.expiresAt);
  if (!Number.isFinite(expires)) issues.push("expiresAt is invalid");
  else if (expires <= nowMs) issues.push("delivery envelope has expired");
  const render = envelope.assetUrls?.render;
  if (!render || !isAllowedUrl(render.url)) issues.push("render asset URL is missing or invalid");
  const collision = envelope.assetUrls?.collision;
  if (collision && !isAllowedUrl(collision.url)) issues.push("collision asset URL is invalid");
  return issues;
}

export function isAllowedUrl(url: unknown): url is string {
  if (typeof url !== "string" || url.length === 0) return false;
  if (url.startsWith("/") || url.startsWith("./") || url.startsWith("../")) return true;
  try {
    const parsed = new URL(url);
    return ["https:", "http:", "blob:"].includes(parsed.protocol);
  } catch {
    return false;
  }
}

export class ViewerInputError extends Error {
  readonly issues: string[];
  constructor(issues: string[]) {
    super(`Invalid viewer input: ${issues.join("; ")}`);
    this.name = "ViewerInputError";
    this.issues = issues;
  }
}

/** Validates the source and normalizes it into the URLs the engine loads. */
export function resolveViewerSource(source: ViewerSource, nowMs = Date.now()): ResolvedSource {
  if (source.kind === "delivery") {
    const issues = deliveryEnvelopeIssues(source.envelope, nowMs);
    if (issues.length > 0) throw new ViewerInputError(issues);
    const { envelope } = source;
    const render = envelope.assetUrls.render!;
    const collision = envelope.assetUrls.collision;
    return {
      manifest: envelope.manifest,
      renderUrl: render.url,
      renderFormat: envelope.manifest.assets.render.format,
      collisionUrl: collision?.url ?? null,
      collisionSha256: collision?.sha256 ?? null,
      layout: null,
      posterUrl: envelope.assetUrls.poster?.url ?? null,
    };
  }
  const issues = sceneManifestIssues(source.manifest);
  if (!isAllowedUrl(source.renderUrl)) issues.push("renderUrl is missing or invalid");
  if (source.collisionUrl && !isAllowedUrl(source.collisionUrl))
    issues.push("collisionUrl is invalid");
  if (issues.length > 0) throw new ViewerInputError(issues);
  return {
    manifest: source.manifest,
    renderUrl: source.renderUrl,
    renderFormat: source.manifest.assets?.render?.format ?? "ply",
    collisionUrl: source.collisionUrl ?? null,
    collisionSha256: null,
    layout: source.layout ?? null,
    posterUrl: null,
  };
}

/** Hex SHA-256 using WebCrypto (browser and Node >= 20). */
export async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const digest = await subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}
