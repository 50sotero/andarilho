import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  Mesh,
  MeshBasicMaterial,
  SRGBColorSpace,
  Texture,
  type BufferGeometry,
  type Color,
  type Material,
  type Object3D,
  type Scene,
  type WebGLRenderer,
} from "three";

/**
 * Textured-mesh bridge (layout texturing). A GLB whose geometry and textures
 * were baked by the worker is drawn with plain three.js, next to the Spark splat
 * bridge (spark-scene.ts); the engine picks one per render format. The scene has
 * no lights, so every material is unlit: anything that is not already a
 * MeshBasicMaterial is replaced by one that keeps its colour map.
 */

export interface MeshSceneOptions {
  renderer: WebGLRenderer;
  scene: Scene;
  url: string;
  /** Requested anisotropic filtering level, capped by the GPU limit. */
  maxAnisotropy: number;
  /** Download progress in [0, 1], or null while the total length is unknown. */
  onProgress?: (loadedFraction: number | null) => void;
  /** Test seam: turns the downloaded GLB into a scene graph. */
  parse?: (data: ArrayBuffer) => Promise<Object3D>;
  /** Test seam: the download function. */
  fetch?: typeof fetch;
}

export interface MeshSceneHandle {
  /** The loaded scene graph; null until it loads and after disposal. */
  readonly root: Object3D | null;
  /** Resolves once the GLB is downloaded, parsed and added to the scene. */
  readonly initialized: Promise<void>;
  setMaxAnisotropy(count: number): void;
  dispose(): void;
}

function isAnisotropy(value: number): boolean {
  return Number.isFinite(value) && value >= 1;
}

async function parseGlb(data: ArrayBuffer): Promise<Object3D> {
  const gltf = await new GLTFLoader().parseAsync(data, "");
  return gltf.scene;
}

/**
 * The Content-Length of the download, or null when it does not measure the bytes
 * that arrive: the server did not declare one, or the body is content-encoded
 * (the declared length is then the encoded size, smaller than the stream).
 */
function declaredLength(headers: Headers): number | null {
  const encoding = headers.get("content-encoding");
  if (encoding !== null && encoding.trim().toLowerCase() !== "identity") return null;
  const raw = headers.get("content-length");
  const length = raw === null ? Number.NaN : Number(raw);
  return Number.isFinite(length) && length > 0 ? length : null;
}

/**
 * Reads a body of known length straight into one buffer of that size, so a large
 * GLB is not copied on the way in. The buffer grows only if the body is longer
 * than declared, and a shorter body is copied once to drop the unused tail.
 */
async function readSizedBody(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  total: number,
  onProgress: (fraction: number | null) => void,
): Promise<ArrayBuffer> {
  let bytes = new Uint8Array(total);
  let loaded = 0;
  for (let next = await reader.read(); !next.done; next = await reader.read()) {
    const chunk = next.value;
    if (loaded + chunk.byteLength > bytes.byteLength) {
      const capacity = Math.max(loaded + chunk.byteLength, bytes.byteLength * 2, 65_536);
      const grown = new Uint8Array(capacity);
      grown.set(bytes.subarray(0, loaded));
      bytes = grown;
    }
    bytes.set(chunk, loaded);
    loaded += chunk.byteLength;
    onProgress(Math.min(1, loaded / total));
  }
  return loaded === bytes.byteLength ? bytes.buffer : bytes.buffer.slice(0, loaded);
}

/**
 * Reads a body of unknown length. The chunks are kept until the end, then copied
 * once into an exact-size buffer; that copy needs a second buffer of the same size.
 */
async function readUnsizedBody(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onProgress: (fraction: null) => void,
): Promise<ArrayBuffer> {
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (let next = await reader.read(); !next.done; next = await reader.read()) {
    chunks.push(next.value);
    size += next.value.byteLength;
    onProgress(null);
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes.buffer;
}

/** Reads the body as it arrives, reporting progress per chunk. */
function readBody(
  response: Response,
  onProgress: (fraction: number | null) => void,
): Promise<ArrayBuffer> {
  const body = response.body;
  if (!body) return response.arrayBuffer();
  const reader = body.getReader();
  const total = declaredLength(response.headers);
  return total === null
    ? readUnsizedBody(reader, onProgress)
    : readSizedBody(reader, total, onProgress);
}

function cancelledError(): Error {
  const error = new Error("Mesh scene load was cancelled");
  error.name = "AbortError";
  return error;
}

/** Cancels a response body that will not be read, so its connection is released. */
function discardBody(response: Response): void {
  void response.body?.cancel().catch(() => undefined);
}

/**
 * Downloads and draws one textured GLB. Disposal before the download finishes
 * aborts it; nothing is added to the scene and `initialized` rejects with an
 * AbortError that is not reported as unhandled.
 */
export function createMeshScene(options: MeshSceneOptions): MeshSceneHandle {
  const { renderer, scene, url } = options;
  const fetchBytes = options.fetch ?? fetch;
  const parse = options.parse ?? parseGlb;
  const controller = new AbortController();
  const geometries = new Set<BufferGeometry>();
  const materials = new Set<Material>();
  // Every texture the scene owns, drawn or not, so each one is disposed.
  const textures = new Set<Texture>();
  // The textures the drawn materials bind: only these take the anisotropy.
  const drawnTextures = new Set<Texture>();
  // GLTFLoader gives every primitive that uses one glTF material the same instance;
  // the replacement is shared the same way, and the source is disposed once.
  const replacements = new Map<Material, MeshBasicMaterial>();
  let anisotropy = isAnisotropy(options.maxAnisotropy) ? Math.floor(options.maxAnisotropy) : 1;
  let root: Object3D | null = null;
  let disposed = false;
  let lastProgress: number | null | undefined;

  const reportProgress = (fraction: number | null): void => {
    if (disposed || fraction === lastProgress) return;
    lastProgress = fraction;
    options.onProgress?.(fraction);
  };

  /** Records every texture a material binds, and returns them. */
  const trackTextures = (material: Material): Texture[] => {
    const bound = (Object.values(material) as unknown[]).filter(
      (value): value is Texture => value instanceof Texture,
    );
    for (const texture of bound) textures.add(texture);
    return bound;
  };

  const track = (material: MeshBasicMaterial): MeshBasicMaterial => {
    materials.add(material);
    for (const texture of trackTextures(material)) drawnTextures.add(texture);
    if (material.map) material.map.colorSpace = SRGBColorSpace;
    return material;
  };

  const unlit = (material: Material): Material => {
    if (material instanceof MeshBasicMaterial) return track(material);
    const known = replacements.get(material);
    if (known) return known;
    // GLTFLoader maps KHR_materials_unlit to MeshBasicMaterial; anything else is lit.
    const lit = material as Material & { map?: Texture | null; color?: Color };
    const map = lit.map ?? null;
    const replacement = new MeshBasicMaterial({
      map,
      color: map ? 0xffffff : (lit.color ?? 0xffffff),
      side: lit.side,
      transparent: lit.transparent,
      opacity: lit.opacity,
    });
    // The source's other textures (normal, roughness, ...) are no longer drawn but
    // still belong to the scene, so they are released with it.
    trackTextures(lit);
    lit.dispose();
    replacements.set(material, replacement);
    return track(replacement);
  };

  // Node transforms are identity by contract; freezing them skips per-frame work.
  const prepare = (object: Object3D): void => {
    object.traverse((node) => {
      node.updateMatrix();
      node.matrixAutoUpdate = false;
      if (!(node instanceof Mesh)) return;
      geometries.add(node.geometry);
      node.material = Array.isArray(node.material)
        ? node.material.map(unlit)
        : unlit(node.material);
    });
  };

  /** Takes ownership of a parsed graph that will not be drawn, so disposal still frees it. */
  const adoptGraph = (object: Object3D): void => {
    object.traverse((node) => {
      if (!(node instanceof Mesh)) return;
      geometries.add(node.geometry);
      for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
        materials.add(material);
        trackTextures(material);
      }
    });
  };

  const applyAnisotropy = (): void => {
    const count = Math.min(anisotropy, renderer.capabilities.getMaxAnisotropy());
    for (const texture of drawnTextures) {
      texture.anisotropy = count;
      texture.needsUpdate = true;
    }
  };

  const releaseResources = (): void => {
    for (const geometry of geometries) geometry.dispose();
    for (const material of materials) material.dispose();
    for (const texture of textures) texture.dispose();
    geometries.clear();
    materials.clear();
    textures.clear();
    drawnTextures.clear();
    replacements.clear();
  };

  const load = async (): Promise<void> => {
    const response = await fetchBytes(url, { signal: controller.signal });
    if (!response.ok) {
      discardBody(response);
      throw new Error(`Mesh scene request failed (${response.status})`);
    }
    if (disposed) {
      discardBody(response);
      throw cancelledError();
    }
    const data = await readBody(response, reportProgress);
    if (disposed) throw cancelledError();
    reportProgress(1);
    const object = await parse(data);
    if (disposed) {
      // Parsed after disposal: nothing draws it, but GLTFLoader allocated it, so free it.
      adoptGraph(object);
      releaseResources();
      throw cancelledError();
    }
    prepare(object);
    root = object;
    scene.add(object);
    applyAnisotropy();
  };

  const initialized = load().catch((error: unknown) => {
    throw disposed ? cancelledError() : error;
  });

  return {
    get root() {
      return root;
    },
    initialized,
    setMaxAnisotropy(count) {
      if (disposed || !isAnisotropy(count)) return;
      anisotropy = Math.floor(count);
      applyAnisotropy();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      controller.abort();
      // A caller that never awaits `initialized` must not see the cancellation as unhandled.
      void initialized.catch(() => undefined);
      if (root) {
        scene.remove(root);
        root = null;
      }
      releaseResources();
    },
  };
}
