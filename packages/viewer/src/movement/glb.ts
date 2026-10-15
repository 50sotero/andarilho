import type { Aabb, CollisionMesh } from "../types";
import { meshBounds } from "../types";

/**
 * Minimal, dependency-free GLB (glTF 2.0 binary) reader for collision assets.
 *
 * Collision GLBs are a product contract (README section 9): vertex positions are
 * baked in canonical world coordinates and every node transform is identity.
 * This reader validates that contract and rejects anything else instead of
 * guessing or applying the splat render transform.
 */

const GLB_MAGIC = 0x46546c67; // "glTF"
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;
const IDENTITY_EPSILON = 1e-6;

const COMPONENT_SIZES: Readonly<Record<number, number>> = {
  5120: 1,
  5121: 1,
  5122: 2,
  5123: 2,
  5125: 4,
  5126: 4,
};
const TYPE_COMPONENTS: Readonly<Record<string, number>> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
};

export class CollisionAssetError extends Error {
  readonly issues: string[];
  constructor(issues: string[]) {
    super(`Invalid collision asset: ${issues.slice(0, 5).join("; ")}`);
    this.name = "CollisionAssetError";
    this.issues = issues;
  }
}

/** Subset of the glTF JSON schema used by the collision reader. */
export interface GltfJson {
  asset?: { version?: string };
  extensionsRequired?: string[];
  scene?: number;
  scenes?: { nodes?: number[] }[];
  nodes?: GltfNode[];
  meshes?: { primitives?: GltfPrimitive[]; weights?: number[] }[];
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: { byteLength?: number; uri?: string }[];
}

export interface GltfNode {
  name?: string;
  mesh?: number;
  children?: number[];
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
  skin?: number;
  weights?: number[];
}

export interface GltfPrimitive {
  attributes?: Record<string, number>;
  indices?: number;
  mode?: number;
  targets?: unknown[];
}

export interface GltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType?: number;
  normalized?: boolean;
  count?: number;
  type?: string;
  sparse?: unknown;
}

export interface GltfBufferView {
  buffer?: number;
  byteOffset?: number;
  byteLength?: number;
  byteStride?: number;
}

export interface ParsedGlb {
  json: GltfJson;
  bin: Uint8Array | null;
}

function toBytes(data: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (data instanceof Uint8Array) return data;
  if (ArrayBuffer.isView(data))
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return new Uint8Array(data);
}

/** Splits a GLB container into its JSON and BIN chunks. Throws on a malformed container. */
export function parseGlb(data: ArrayBuffer | ArrayBufferView): ParsedGlb {
  const bytes = toBytes(data);
  if (bytes.byteLength < 20) throw new CollisionAssetError(["file is too small to be a GLB"]);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== GLB_MAGIC) throw new CollisionAssetError(["missing glTF magic"]);
  const version = view.getUint32(4, true);
  if (version !== 2) throw new CollisionAssetError([`unsupported GLB version ${version}`]);
  const length = view.getUint32(8, true);
  if (length > bytes.byteLength) throw new CollisionAssetError(["GLB length exceeds data"]);

  let offset = 12;
  let json: GltfJson | null = null;
  let bin: Uint8Array | null = null;
  while (offset + 8 <= length) {
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    const start = offset + 8;
    const end = start + chunkLength;
    if (end > length) throw new CollisionAssetError(["GLB chunk exceeds file length"]);
    const chunk = bytes.subarray(start, end);
    if (chunkType === CHUNK_JSON && json === null) {
      try {
        const parsed: unknown = JSON.parse(new TextDecoder().decode(chunk));
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("not an object");
        }
        json = parsed as GltfJson;
      } catch {
        throw new CollisionAssetError(["GLB JSON chunk is not valid JSON"]);
      }
    } else if (chunkType === CHUNK_BIN && bin === null) {
      bin = chunk;
    }
    offset = end + ((4 - (chunkLength % 4)) % 4);
  }
  if (!json) throw new CollisionAssetError(["GLB has no JSON chunk"]);
  return { json, bin };
}

function near(a: number, b: number): boolean {
  return Math.abs(a - b) <= IDENTITY_EPSILON;
}

/** Returns a description of why a node transform is not identity, or null when it is. */
export function nodeTransformIssue(node: GltfNode): string | null {
  if (node.matrix !== undefined) {
    const m = node.matrix;
    if (!Array.isArray(m) || m.length !== 16) return "matrix must have 16 elements";
    for (let i = 0; i < 16; i++) {
      const expected = i % 5 === 0 ? 1 : 0;
      const v = m[i];
      if (typeof v !== "number" || !near(v, expected)) return "matrix is not identity";
    }
  }
  if (node.translation !== undefined) {
    const t = node.translation;
    if (
      !Array.isArray(t) ||
      t.length !== 3 ||
      !t.every((v) => typeof v === "number" && near(v, 0))
    ) {
      return "translation is not zero";
    }
  }
  if (node.rotation !== undefined) {
    const r = node.rotation;
    if (
      !Array.isArray(r) ||
      r.length !== 4 ||
      !r.every((v) => typeof v === "number" && Number.isFinite(v)) ||
      !near(r[0] as number, 0) ||
      !near(r[1] as number, 0) ||
      !near(r[2] as number, 0) ||
      !near(Math.abs(r[3] as number), 1)
    ) {
      return "rotation is not identity";
    }
  }
  if (node.scale !== undefined) {
    const s = node.scale;
    if (
      !Array.isArray(s) ||
      s.length !== 3 ||
      !s.every((v) => typeof v === "number" && near(v, 1))
    ) {
      return "scale is not one";
    }
  }
  if (node.skin !== undefined) return "skinned nodes are not allowed";
  if (node.weights !== undefined) return "morph weights are not allowed";
  return null;
}

interface AccessorView {
  view: DataView;
  count: number;
  componentType: number;
  stride: number;
  components: number;
}

function resolveAccessor(
  json: GltfJson,
  bin: Uint8Array | null,
  index: number,
  label: string,
  issues: string[],
): AccessorView | null {
  const accessor = json.accessors?.[index];
  if (!accessor) {
    issues.push(`${label}: accessor ${index} does not exist`);
    return null;
  }
  if (accessor.sparse !== undefined) {
    issues.push(`${label}: sparse accessors are not allowed`);
    return null;
  }
  const componentType = accessor.componentType ?? -1;
  const componentSize = COMPONENT_SIZES[componentType];
  const components = TYPE_COMPONENTS[accessor.type ?? ""];
  const count = accessor.count ?? -1;
  if (!componentSize || !components || !Number.isInteger(count) || count < 0) {
    issues.push(`${label}: accessor ${index} has invalid type/componentType/count`);
    return null;
  }
  if (accessor.bufferView === undefined) {
    issues.push(`${label}: accessor ${index} has no bufferView`);
    return null;
  }
  const bufferView = json.bufferViews?.[accessor.bufferView];
  if (!bufferView) {
    issues.push(`${label}: bufferView ${accessor.bufferView} does not exist`);
    return null;
  }
  const buffer = json.buffers?.[bufferView.buffer ?? -1];
  if (!buffer || buffer.uri !== undefined || (bufferView.buffer ?? -1) !== 0 || !bin) {
    issues.push(`${label}: only the embedded GLB BIN buffer is allowed`);
    return null;
  }
  const elementSize = componentSize * components;
  const stride = bufferView.byteStride ?? elementSize;
  if (stride < elementSize || stride % componentSize !== 0) {
    issues.push(`${label}: invalid byteStride`);
    return null;
  }
  const viewOffset = bufferView.byteOffset ?? 0;
  const viewLength = bufferView.byteLength ?? -1;
  if (viewOffset < 0 || viewLength < 0 || viewOffset + viewLength > bin.byteLength) {
    issues.push(`${label}: bufferView exceeds BIN chunk`);
    return null;
  }
  const accessorOffset = accessor.byteOffset ?? 0;
  const needed = count === 0 ? 0 : accessorOffset + stride * (count - 1) + elementSize;
  if (accessorOffset < 0 || needed > viewLength) {
    issues.push(`${label}: accessor ${index} exceeds its bufferView`);
    return null;
  }
  if ((bin.byteOffset + viewOffset + accessorOffset) % componentSize !== 0) {
    issues.push(`${label}: accessor ${index} is misaligned`);
    return null;
  }
  const view = new DataView(
    bin.buffer,
    bin.byteOffset + viewOffset + accessorOffset,
    needed - accessorOffset,
  );
  return { view, count, componentType, stride, components };
}

export interface CollisionValidationOptions {
  /** Approved scene bounds; vertices must lie within them (plus tolerance). */
  bounds?: { min: readonly number[]; max: readonly number[] } | null;
  /** Bounds tolerance in meters. */
  boundsTolerance?: number;
  /** Upper bound on accepted triangles. */
  maxTriangles?: number;
}

export interface CollisionMeshStats {
  vertexCount: number;
  triangleCount: number;
  degenerateTriangles: number;
  bounds: Aabb | null;
}

export interface ValidatedCollisionMesh {
  mesh: CollisionMesh;
  stats: CollisionMeshStats;
}

/**
 * Validates a world-space triangle soup: finite vertices, triangle-multiple
 * indices, in-range indices, non-empty, and (optionally) inside approved bounds.
 * Returns the list of issues; an empty list means the mesh is acceptable.
 */
export function collisionMeshIssues(
  mesh: CollisionMesh,
  options: CollisionValidationOptions = {},
): string[] {
  const issues: string[] = [];
  const { positions, indices } = mesh;
  if (positions.length === 0 || positions.length % 3 !== 0) {
    issues.push("positions must be a non-empty multiple of 3");
  }
  for (let i = 0; i < positions.length; i++) {
    if (!Number.isFinite(positions[i] as number)) {
      issues.push(`vertex ${Math.floor(i / 3)} has a non-finite coordinate`);
      break;
    }
  }
  if (indices.length === 0 || indices.length % 3 !== 0) {
    issues.push("indices must be a non-empty multiple of 3");
  }
  const vertexCount = Math.floor(positions.length / 3);
  for (let i = 0; i < indices.length; i++) {
    if ((indices[i] as number) >= vertexCount) {
      issues.push(`index ${indices[i]} at ${i} is out of range (vertexCount ${vertexCount})`);
      break;
    }
  }
  const maxTriangles = options.maxTriangles ?? 2_000_000;
  if (indices.length / 3 > maxTriangles) {
    issues.push(`triangle count ${indices.length / 3} exceeds limit ${maxTriangles}`);
  }
  if (options.bounds && issues.length === 0) {
    const tol = options.boundsTolerance ?? 0.05;
    const b = meshBounds(mesh);
    const { min, max } = options.bounds;
    if (b) {
      for (let k = 0; k < 3; k++) {
        if (b.min[k]! < (min[k] as number) - tol || b.max[k]! > (max[k] as number) + tol) {
          issues.push("collision vertices fall outside the approved scene bounds");
          break;
        }
      }
    }
  }
  return issues;
}

function countDegenerate(mesh: CollisionMesh): number {
  const p = mesh.positions;
  const idx = mesh.indices;
  let degenerate = 0;
  for (let t = 0; t + 2 < idx.length; t += 3) {
    const a = idx[t]! * 3;
    const b = idx[t + 1]! * 3;
    const c = idx[t + 2]! * 3;
    const ux = p[b]! - p[a]!;
    const uy = p[b + 1]! - p[a + 1]!;
    const uz = p[b + 2]! - p[a + 2]!;
    const vx = p[c]! - p[a]!;
    const vy = p[c + 1]! - p[a + 1]!;
    const vz = p[c + 2]! - p[a + 2]!;
    const cx = uy * vz - uz * vy;
    const cy = uz * vx - ux * vz;
    const cz = ux * vy - uy * vx;
    if (cx * cx + cy * cy + cz * cz < 1e-18) degenerate++;
  }
  return degenerate;
}

export function validateCollisionMesh(
  mesh: CollisionMesh,
  options: CollisionValidationOptions = {},
): ValidatedCollisionMesh {
  const issues = collisionMeshIssues(mesh, options);
  if (issues.length > 0) throw new CollisionAssetError(issues);
  return {
    mesh,
    stats: {
      vertexCount: mesh.positions.length / 3,
      triangleCount: mesh.indices.length / 3,
      degenerateTriangles: countDegenerate(mesh),
      bounds: meshBounds(mesh),
    },
  };
}

function collectMeshNodes(json: GltfJson): number[] {
  const nodes = json.nodes ?? [];
  const scenes = json.scenes ?? [];
  const roots: number[] = [];
  if (scenes.length > 0) {
    for (const scene of scenes) roots.push(...(scene.nodes ?? []));
  } else {
    nodes.forEach((_, i) => roots.push(i));
  }
  const visited = new Set<number>();
  const stack = [...roots];
  while (stack.length > 0) {
    const index = stack.pop() as number;
    if (visited.has(index)) continue;
    visited.add(index);
    const node = nodes[index];
    if (node) stack.push(...(node.children ?? []));
  }
  return [...visited].filter((i) => nodes[i]?.mesh !== undefined).sort((a, b) => a - b);
}

/**
 * Extracts and validates a world-space collision mesh from a GLB container.
 * Rejects non-identity node transforms (every node, referenced or not), non-
 * triangle primitives, morph/skin data, external buffers, required extensions,
 * non-finite vertices, out-of-range indices and out-of-bounds geometry.
 */
export function readCollisionGlb(
  data: ArrayBuffer | ArrayBufferView,
  options: CollisionValidationOptions = {},
): ValidatedCollisionMesh {
  const { json, bin } = parseGlb(data);
  const issues: string[] = [];

  if (json.asset?.version !== undefined && !String(json.asset.version).startsWith("2.")) {
    issues.push(`unsupported glTF version ${json.asset.version}`);
  }
  if (json.extensionsRequired && json.extensionsRequired.length > 0) {
    issues.push(`required extensions are not supported: ${json.extensionsRequired.join(", ")}`);
  }
  const nodes = json.nodes ?? [];
  nodes.forEach((node, i) => {
    const issue = nodeTransformIssue(node);
    if (issue) issues.push(`node ${i}${node.name ? ` (${node.name})` : ""}: ${issue}`);
  });
  if (issues.length > 0) throw new CollisionAssetError(issues);

  const meshIndices = new Set<number>();
  for (const nodeIndex of collectMeshNodes(json)) {
    meshIndices.add(nodes[nodeIndex]!.mesh as number);
  }
  if (meshIndices.size === 0) throw new CollisionAssetError(["no mesh nodes in collision asset"]);

  const positionChunks: Float32Array[] = [];
  const indexChunks: Uint32Array[] = [];
  let vertexBase = 0;

  for (const meshIndex of [...meshIndices].sort((a, b) => a - b)) {
    const mesh = json.meshes?.[meshIndex];
    if (!mesh) {
      issues.push(`mesh ${meshIndex} does not exist`);
      continue;
    }
    if (mesh.weights !== undefined) issues.push(`mesh ${meshIndex}: morph weights are not allowed`);
    (mesh.primitives ?? []).forEach((primitive, p) => {
      const label = `mesh ${meshIndex} primitive ${p}`;
      if (primitive.mode !== undefined && primitive.mode !== 4) {
        issues.push(`${label}: only TRIANGLES (mode 4) is allowed`);
        return;
      }
      if (primitive.targets !== undefined) {
        issues.push(`${label}: morph targets are not allowed`);
        return;
      }
      const posIndex = primitive.attributes?.POSITION;
      if (posIndex === undefined) {
        issues.push(`${label}: missing POSITION`);
        return;
      }
      const pos = resolveAccessor(json, bin, posIndex, `${label} POSITION`, issues);
      if (!pos) return;
      if (pos.componentType !== 5126 || pos.components !== 3) {
        issues.push(`${label}: POSITION must be float VEC3`);
        return;
      }
      if (json.accessors?.[posIndex]?.normalized) {
        issues.push(`${label}: POSITION must not be normalized`);
        return;
      }
      const positions = new Float32Array(pos.count * 3);
      for (let i = 0; i < pos.count; i++) {
        for (let k = 0; k < 3; k++) {
          positions[i * 3 + k] = pos.view.getFloat32(i * pos.stride + k * 4, true);
        }
      }
      let indices: Uint32Array;
      if (primitive.indices !== undefined) {
        const acc = resolveAccessor(json, bin, primitive.indices, `${label} indices`, issues);
        if (!acc) return;
        if (acc.components !== 1 || ![5121, 5123, 5125].includes(acc.componentType)) {
          issues.push(`${label}: indices must be unsigned SCALAR`);
          return;
        }
        indices = new Uint32Array(acc.count);
        for (let i = 0; i < acc.count; i++) {
          const at = i * acc.stride;
          const v =
            acc.componentType === 5121
              ? acc.view.getUint8(at)
              : acc.componentType === 5123
                ? acc.view.getUint16(at, true)
                : acc.view.getUint32(at, true);
          if (v >= pos.count) {
            issues.push(`${label}: index ${v} out of range (vertex count ${pos.count})`);
            return;
          }
          indices[i] = v + vertexBase;
        }
      } else {
        indices = new Uint32Array(pos.count);
        for (let i = 0; i < pos.count; i++) indices[i] = i + vertexBase;
      }
      if (indices.length % 3 !== 0) {
        issues.push(`${label}: index count is not a multiple of 3`);
        return;
      }
      positionChunks.push(positions);
      indexChunks.push(indices);
      vertexBase += pos.count;
    });
  }
  if (issues.length > 0) throw new CollisionAssetError(issues);

  const merged: CollisionMesh = {
    positions: concatFloat32(positionChunks),
    indices: concatUint32(indexChunks),
  };
  return validateCollisionMesh(merged, options);
}

function concatFloat32(chunks: Float32Array[]): Float32Array {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

function concatUint32(chunks: Uint32Array[]): Uint32Array {
  const out = new Uint32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

export interface GlbWriteNode {
  name?: string;
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
}

/**
 * Per-axis minimum and maximum of the vertices, which glTF requires on POSITION
 * accessors (GLTFLoader warns without them). An empty mesh has no bounds to give.
 */
function positionBounds(positions: Float32Array): { min?: number[]; max?: number[] } {
  if (positions.length === 0) return {};
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = positions[i + k]!;
      min[k] = Math.min(min[k]!, v);
      max[k] = Math.max(max[k]!, v);
    }
  }
  return { min, max };
}

/**
 * Writes a minimal single-mesh GLB (one node, float positions with their bounds,
 * uint32 indices). Used by tests and fixtures; the optional node transform lets tests
 * produce deliberately invalid assets.
 */
export function writeCollisionGlb(mesh: CollisionMesh, node: GlbWriteNode = {}): Uint8Array {
  const posBytes = mesh.positions.byteLength;
  const idxBytes = mesh.indices.byteLength;
  const binLength = posBytes + idxBytes;
  const json: GltfJson & Record<string, unknown> = {
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, ...node }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, mode: 4 }] }],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: mesh.positions.length / 3,
        type: "VEC3",
        ...positionBounds(mesh.positions),
      },
      { bufferView: 1, componentType: 5125, count: mesh.indices.length, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: posBytes },
      { buffer: 0, byteOffset: posBytes, byteLength: idxBytes },
    ],
    buffers: [{ byteLength: binLength }],
  };
  let jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  if (jsonPad) {
    const padded = new Uint8Array(jsonBytes.length + jsonPad).fill(0x20);
    padded.set(jsonBytes);
    jsonBytes = padded;
  }
  const binPad = (4 - (binLength % 4)) % 4;
  const total = 12 + 8 + jsonBytes.length + 8 + binLength + binPad;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, GLB_MAGIC, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonBytes.length, true);
  view.setUint32(16, CHUNK_JSON, true);
  out.set(jsonBytes, 20);
  const binStart = 20 + jsonBytes.length;
  view.setUint32(binStart, binLength + binPad, true);
  view.setUint32(binStart + 4, CHUNK_BIN, true);
  out.set(new Uint8Array(mesh.positions.buffer, mesh.positions.byteOffset, posBytes), binStart + 8);
  out.set(
    new Uint8Array(mesh.indices.buffer, mesh.indices.byteOffset, idxBytes),
    binStart + 8 + posBytes,
  );
  return out;
}
