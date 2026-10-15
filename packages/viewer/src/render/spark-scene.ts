import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import type { Scene, WebGLRenderer } from "three";
import { whenSparkIdle } from "./spark-idle";
import { decomposeRenderTransform } from "./transform";

/**
 * Spark bridge (README section 11, "Rendering bridge"). One SparkRenderer per
 * Three.js renderer, one SplatMesh per scene revision, and the manifest's
 * `worldFromDecodedRender` applied exactly once to the splat object.
 */

export type RenderFormat = "sog" | "spz" | "ply";

export interface SparkSceneOptions {
  renderer: WebGLRenderer;
  scene: Scene;
  url: string;
  /** Manifest render format; passed to Spark because signed URLs hide extensions. */
  format?: RenderFormat;
  worldFromDecodedRender: readonly number[];
  /** Initial LoD splat budget from the quality profile. */
  splatBudget?: number;
  onProgress?: (loadedFraction: number | null) => void;
  /** Called when Spark needs a redraw (sort/LoD update finished). */
  onDirty?: () => void;
}

export interface SparkSceneHandle {
  readonly splats: SplatMesh;
  readonly spark: SparkRenderer;
  /** Resolves once the splat file is downloaded and decoded. */
  readonly initialized: Promise<void>;
  setSplatBudget(count: number): void;
  dispose(): void;
}

export function renderFileType(format: RenderFormat | undefined): SplatFileType | undefined {
  switch (format) {
    case "ply":
      return SplatFileType.PLY;
    case "spz":
      return SplatFileType.SPZ;
    case "sog":
      // Bundled SOG is a zip container of WebP component images.
      return SplatFileType.PCSOGSZIP;
    default:
      return undefined;
  }
}

export function createSparkScene(options: SparkSceneOptions): SparkSceneHandle {
  const transform = decomposeRenderTransform(options.worldFromDecodedRender);
  const spark = new SparkRenderer({
    renderer: options.renderer,
    ...(options.onDirty ? { onDirty: options.onDirty } : {}),
    ...(options.splatBudget ? { lodSplatCount: options.splatBudget } : {}),
  });
  let splats: SplatMesh | null = null;
  try {
    options.scene.add(spark);
    const fileType = renderFileType(options.format);
    splats = new SplatMesh({
      url: options.url,
      ...(fileType ? { fileType } : {}),
      onProgress: (event: ProgressEvent) => {
        options.onProgress?.(
          event.lengthComputable && event.total > 0 ? event.loaded / event.total : null,
        );
      },
    });
    splats.position.copy(transform.position);
    splats.quaternion.copy(transform.quaternion);
    splats.scale.setScalar(transform.scale);
    options.scene.add(splats);
  } catch (error) {
    if (splats) {
      options.scene.remove(splats);
      splats.dispose();
    }
    options.scene.remove(spark);
    spark.dispose();
    throw error;
  }
  const mesh = splats;
  let disposed = false;
  return {
    splats: mesh,
    spark,
    initialized: mesh.initialized.then(() => undefined),
    setSplatBudget(count: number) {
      if (Number.isFinite(count) && count > 0) spark.lodSplatCount = Math.round(count);
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      options.scene.remove(mesh);
      options.scene.remove(spark);
      // Out of the scene nothing starts new work; let a sort in flight finish first.
      whenSparkIdle(spark, () => {
        mesh.dispose();
        spark.dispose();
      });
    },
  };
}
