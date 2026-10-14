import { Matrix4, Quaternion, Vector3 } from "three";

/**
 * `worldFromDecodedRender` contract (README section 9): a column-major 4x4
 * matrix compatible with `THREE.Matrix4.fromArray`, containing only a proper
 * rotation, translation and positive uniform scale. It is applied once, to the
 * splat render object only; world-space collision/layout assets never receive it.
 */

export interface RenderTransform {
  position: Vector3;
  quaternion: Quaternion;
  /** Uniform scale factor. */
  scale: number;
  matrix: Matrix4;
}

export class RenderTransformError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RenderTransformError";
  }
}

export function decomposeRenderTransform(
  values: readonly number[],
  tolerance = 1e-4,
): RenderTransform {
  if (!Array.isArray(values) || values.length !== 16) {
    throw new RenderTransformError("worldFromDecodedRender must have 16 elements");
  }
  if (!values.every((v) => typeof v === "number" && Number.isFinite(v))) {
    throw new RenderTransformError("worldFromDecodedRender must contain finite numbers");
  }
  // Column-major: the last row is elements 3, 7, 11, 15.
  if (
    Math.abs(values[3] as number) > tolerance ||
    Math.abs(values[7] as number) > tolerance ||
    Math.abs(values[11] as number) > tolerance ||
    Math.abs((values[15] as number) - 1) > tolerance
  ) {
    throw new RenderTransformError("worldFromDecodedRender must be affine (last row 0 0 0 1)");
  }
  const matrix = new Matrix4().fromArray(values as number[]);
  if (!(matrix.determinant() > 0)) {
    throw new RenderTransformError("worldFromDecodedRender must not contain a reflection");
  }
  const position = new Vector3();
  const quaternion = new Quaternion();
  const scale = new Vector3();
  matrix.decompose(position, quaternion, scale);
  const mean = (scale.x + scale.y + scale.z) / 3;
  if (!(mean > 0)) throw new RenderTransformError("worldFromDecodedRender scale must be positive");
  const maxDev = Math.max(
    Math.abs(scale.x - mean),
    Math.abs(scale.y - mean),
    Math.abs(scale.z - mean),
  );
  if (maxDev > tolerance * Math.max(1, mean)) {
    throw new RenderTransformError("worldFromDecodedRender scale must be uniform");
  }
  // Reject shear: the rebuilt matrix must match the original.
  const rebuilt = new Matrix4().compose(position, quaternion, new Vector3(mean, mean, mean));
  for (let i = 0; i < 16; i++) {
    if (
      Math.abs((rebuilt.elements[i] as number) - (values[i] as number)) >
      tolerance * Math.max(1, mean)
    ) {
      throw new RenderTransformError("worldFromDecodedRender must not contain shear");
    }
  }
  return { position, quaternion, scale: mean, matrix };
}
