import {
  BoxGeometry,
  BufferGeometry,
  DoubleSide,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  NoColorSpace,
  Points,
  PointsMaterial,
  Scene,
  SRGBColorSpace,
  Texture,
  type Object3D,
  type WebGLRenderer,
} from "three";
import { describe, expect, it, vi } from "vitest";
import { writeCollisionGlb } from "../src/movement/glb";
import { createMeshScene, type MeshSceneOptions } from "../src/render/mesh-scene";
import type { CollisionMesh } from "../src/types";

const SCENE_URL = "https://cdn.example.test/scene.glb?sig=1";
type Fetch = typeof fetch;

function fakeRenderer(maxAnisotropy = 16): WebGLRenderer {
  return { capabilities: { getMaxAnisotropy: () => maxAnisotropy } } as unknown as WebGLRenderer;
}

/** A response whose body arrives in the given chunks, like a streamed download. */
function streamed(
  chunks: Uint8Array[],
  headers: Record<string, string> = {},
  status = 200,
): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
  return new Response(body, { status, headers });
}

/** A download that answers every request with a fresh response. */
function download(respond: () => Response): Fetch {
  return vi.fn(async () => respond()) as unknown as Fetch;
}

/** A download that only fails when its signal aborts, as fetch does. */
function abortableDownload(): { fetch: Fetch; signals: AbortSignal[] } {
  const signals: AbortSignal[] = [];
  const fetch = (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> =>
    new Promise<Response>((_resolve, reject) => {
      const signal = init?.signal;
      if (!signal) return;
      signals.push(signal);
      signal.addEventListener("abort", () =>
        reject(new DOMException("The download was aborted", "AbortError")),
      );
    });
  return { fetch: fetch as Fetch, signals };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** The scene GLTFLoader would produce for a textured, lit cube. */
function texturedCube() {
  const texture = new Texture();
  const geometry = new BoxGeometry();
  const material = new MeshStandardMaterial({ map: texture });
  const mesh: Mesh = new Mesh(geometry, material);
  const group = new Group();
  group.add(mesh);
  return { texture, geometry, material, mesh, group };
}

function options(overrides: Partial<MeshSceneOptions> = {}): MeshSceneOptions {
  return {
    renderer: fakeRenderer(),
    scene: new Scene(),
    url: SCENE_URL,
    maxAnisotropy: 8,
    fetch: download(() => streamed([new Uint8Array(3)], { "content-length": "3" })),
    ...overrides,
  };
}

describe("mesh scene", () => {
  it("draws the GLB unlit, with sRGB textures at the requested anisotropy", async () => {
    const scene = new Scene();
    const cube = texturedCube();
    const handle = createMeshScene(options({ scene, parse: async () => cube.group }));
    await handle.initialized;

    expect(handle.root).toBe(cube.group);
    expect(scene.children).toContain(cube.group);
    const replaced = cube.mesh.material;
    expect(replaced).toBeInstanceOf(MeshBasicMaterial);
    expect(replaced).not.toBe(cube.material);
    expect((replaced as MeshBasicMaterial).map).toBe(cube.texture);
    expect((replaced as MeshBasicMaterial).color.getHex()).toBe(0xffffff);
    expect(cube.texture.colorSpace).toBe(SRGBColorSpace);
    expect(cube.texture.anisotropy).toBe(8);
    expect(cube.group.matrixAutoUpdate).toBe(false);
    expect(cube.mesh.matrixAutoUpdate).toBe(false);
    handle.dispose();
  });

  it("keeps the colour, side, transparency and opacity of a material without a texture", async () => {
    const group = new Group();
    const source = new MeshStandardMaterial({
      color: 0x336699,
      side: DoubleSide,
      transparent: true,
      opacity: 0.5,
    });
    const sourceDispose = vi.spyOn(source, "dispose");
    group.add(new Mesh(new BoxGeometry(), source));
    const handle = createMeshScene(options({ parse: async () => group }));
    await handle.initialized;

    const material = (group.children[0] as Mesh).material as MeshBasicMaterial;
    expect(material).toBeInstanceOf(MeshBasicMaterial);
    expect(material.map).toBeNull();
    expect(material.color.getHex()).toBe(0x336699);
    expect(material.side).toBe(DoubleSide);
    expect(material.transparent).toBe(true);
    expect(material.opacity).toBe(0.5);
    expect(sourceDispose).toHaveBeenCalledTimes(1);
    handle.dispose();
  });

  it("keeps the alpha cutoff of a MASK material, so its cut-outs stay cut out", async () => {
    const group = new Group();
    const source = new MeshStandardMaterial({ map: new Texture(), alphaTest: 0.5 });
    group.add(new Mesh(new BoxGeometry(), source));
    const handle = createMeshScene(options({ parse: async () => group }));
    await handle.initialized;

    const material = (group.children[0] as Mesh).material as MeshBasicMaterial;
    expect(material).toBeInstanceOf(MeshBasicMaterial);
    expect(material.alphaTest).toBe(0.5);
    handle.dispose();
  });

  it("keeps materials that are already unlit", async () => {
    const texture = new Texture();
    const basic = new MeshBasicMaterial({ map: texture });
    const group = new Group();
    group.add(new Mesh(new BoxGeometry(), basic));
    const handle = createMeshScene(options({ parse: async () => group }));
    await handle.initialized;

    expect((group.children[0] as Mesh).material).toBe(basic);
    expect(texture.colorSpace).toBe(SRGBColorSpace);
    handle.dispose();
  });

  it("replaces a material shared by several primitives once and disposes it once", async () => {
    const shared = new MeshStandardMaterial({ color: 0x112233 });
    const sourceDispose = vi.spyOn(shared, "dispose");
    const group = new Group();
    const first = new Mesh(new BoxGeometry(), shared);
    const second = new Mesh(new BoxGeometry(), shared);
    group.add(first, second);
    const handle = createMeshScene(options({ parse: async () => group }));
    await handle.initialized;

    expect(first.material).toBeInstanceOf(MeshBasicMaterial);
    expect(second.material).toBe(first.material);
    expect(sourceDispose).toHaveBeenCalledTimes(1);
    handle.dispose();
  });

  it("releases the other textures of a lit material, though they are not drawn", async () => {
    const normal = new Texture();
    const normalDispose = vi.spyOn(normal, "dispose");
    const group = new Group();
    const source = new MeshStandardMaterial({ map: new Texture(), normalMap: normal });
    group.add(new Mesh(new BoxGeometry(), source));
    const handle = createMeshScene(options({ parse: async () => group }));
    await handle.initialized;

    // Not drawn, so it keeps the defaults: no anisotropy and no colour space.
    expect(normal.anisotropy).toBe(1);
    expect(normal.colorSpace).toBe(NoColorSpace);
    handle.dispose();
    expect(normalDispose).toHaveBeenCalledTimes(1);
  });

  it("reports download progress up to 1, or null while the length is unknown", async () => {
    const chunks = [new Uint8Array(4), new Uint8Array(4), new Uint8Array(2)];

    const known: (number | null)[] = [];
    const sized = createMeshScene(
      options({
        fetch: download(() => streamed(chunks, { "content-length": "10" })),
        parse: async () => new Group(),
        onProgress: (fraction) => known.push(fraction),
      }),
    );
    await sized.initialized;
    expect(known).toEqual([0.4, 0.8, 1]);
    sized.dispose();

    const unknown: (number | null)[] = [];
    const streaming = createMeshScene(
      options({
        fetch: download(() => streamed(chunks)),
        parse: async () => new Group(),
        onProgress: (fraction) => unknown.push(fraction),
      }),
    );
    await streaming.initialized;
    expect(unknown).toEqual([null, 1]);
    streaming.dispose();
  });

  it("reports unknown progress when Content-Length measures an encoded body", async () => {
    const progress: (number | null)[] = [];
    const handle = createMeshScene(
      options({
        fetch: download(() =>
          streamed([new Uint8Array(4)], { "content-length": "4", "content-encoding": "gzip" }),
        ),
        parse: async () => new Group(),
        onProgress: (fraction) => progress.push(fraction),
      }),
    );
    await handle.initialized;
    expect(progress).toEqual([null, 1]);
    handle.dispose();
  });

  it("hands the parser the whole body, even when Content-Length is wrong", async () => {
    const received: number[] = [];
    const parse = async (data: ArrayBuffer) => {
      received.push(data.byteLength);
      return new Group();
    };
    // Longer than declared: the buffer grows to hold it.
    const longer = createMeshScene(
      options({
        fetch: download(() =>
          streamed([new Uint8Array(4), new Uint8Array(4)], { "content-length": "3" }),
        ),
        parse,
      }),
    );
    await longer.initialized;
    // Shorter than declared: the unused tail is dropped.
    const shorter = createMeshScene(
      options({
        fetch: download(() => streamed([new Uint8Array(4)], { "content-length": "10" })),
        parse,
      }),
    );
    await shorter.initialized;

    // No length at all: the chunks are joined.
    const unsized = createMeshScene(
      options({
        fetch: download(() => streamed([new Uint8Array(4), new Uint8Array(4), new Uint8Array(2)])),
        parse,
      }),
    );
    await unsized.initialized;

    expect(received).toEqual([8, 4, 10]);
    longer.dispose();
    shorter.dispose();
    unsized.dispose();
  });

  it("hands the parser the exact bytes of a body read with a correct Content-Length", async () => {
    const received: number[][] = [];
    const handle = createMeshScene(
      options({
        // Three chunks that together are exactly the declared 9 bytes.
        fetch: download(() =>
          streamed([Uint8Array.of(1, 2, 3), Uint8Array.of(4, 5), Uint8Array.of(6, 7, 8, 9)], {
            "content-length": "9",
          }),
        ),
        parse: async (data) => {
          received.push(Array.from(new Uint8Array(data)));
          return new Group();
        },
      }),
    );
    await handle.initialized;

    expect(received).toEqual([[1, 2, 3, 4, 5, 6, 7, 8, 9]]);
    handle.dispose();
  });

  it("caps anisotropy at the GPU limit and updates the loaded textures", async () => {
    const cube = texturedCube();
    const handle = createMeshScene(
      options({ renderer: fakeRenderer(4), maxAnisotropy: 8, parse: async () => cube.group }),
    );
    await handle.initialized;
    expect(cube.texture.anisotropy).toBe(4);

    const version = cube.texture.version;
    handle.setMaxAnisotropy(2);
    expect(cube.texture.anisotropy).toBe(2);
    expect(cube.texture.version).toBeGreaterThan(version);

    handle.setMaxAnisotropy(Number.NaN);
    expect(cube.texture.anisotropy).toBe(2);
    handle.dispose();
  });

  it("disposes each geometry, material and texture once and removes the root", async () => {
    const scene = new Scene();
    const cube = texturedCube();
    const sourceDispose = vi.spyOn(cube.material, "dispose");
    const handle = createMeshScene(options({ scene, parse: async () => cube.group }));
    await handle.initialized;

    const replaced = cube.mesh.material as MeshBasicMaterial;
    const geometryDispose = vi.spyOn(cube.geometry, "dispose");
    const materialDispose = vi.spyOn(replaced, "dispose");
    const textureDispose = vi.spyOn(cube.texture, "dispose");
    handle.dispose();
    handle.dispose();

    expect(scene.children).not.toContain(cube.group);
    expect(handle.root).toBeNull();
    expect(geometryDispose).toHaveBeenCalledTimes(1);
    expect(materialDispose).toHaveBeenCalledTimes(1);
    expect(textureDispose).toHaveBeenCalledTimes(1);
    expect(sourceDispose).toHaveBeenCalledTimes(1);
  });

  /** A group with one point cloud and one line set, each with its own texture-bearing material. */
  function pointsAndLines() {
    const texture = new Texture();
    const group = new Group();
    const points = new Points(new BufferGeometry(), new PointsMaterial({ map: texture }));
    const lines = new LineSegments(new BufferGeometry(), new LineBasicMaterial());
    group.add(points, lines);
    return { texture, group, points, lines };
  }

  it("disposes the points and lines it draws, with their geometry, material and textures", async () => {
    const { texture, group, points, lines } = pointsAndLines();
    const scene = new Scene();
    const handle = createMeshScene(options({ scene, parse: async () => group }));
    await handle.initialized;

    // Already unlit, so they keep their own materials.
    expect(points.matrixAutoUpdate).toBe(false);
    expect(lines.matrixAutoUpdate).toBe(false);
    const pointsGeometry = vi.spyOn(points.geometry, "dispose");
    const pointsMaterial = vi.spyOn(points.material as PointsMaterial, "dispose");
    const linesGeometry = vi.spyOn(lines.geometry, "dispose");
    const linesMaterial = vi.spyOn(lines.material as LineBasicMaterial, "dispose");
    const textureDispose = vi.spyOn(texture, "dispose");
    handle.dispose();

    expect(scene.children).toHaveLength(0);
    expect(pointsGeometry).toHaveBeenCalledTimes(1);
    expect(pointsMaterial).toHaveBeenCalledTimes(1);
    expect(linesGeometry).toHaveBeenCalledTimes(1);
    expect(linesMaterial).toHaveBeenCalledTimes(1);
    expect(textureDispose).toHaveBeenCalledTimes(1);
  });

  it("frees the points and lines of a scene that finishes after disposal", async () => {
    const scene = new Scene();
    const { texture, group, points, lines } = pointsAndLines();
    const parsed = deferred<Object3D>();
    const parse = vi.fn(() => parsed.promise);
    const handle = createMeshScene(options({ scene, parse }));
    await vi.waitFor(() => expect(parse).toHaveBeenCalledTimes(1));
    const pointsGeometry = vi.spyOn(points.geometry, "dispose");
    const linesMaterial = vi.spyOn(lines.material as LineBasicMaterial, "dispose");
    const textureDispose = vi.spyOn(texture, "dispose");

    handle.dispose();
    parsed.resolve(group);

    await expect(handle.initialized).rejects.toMatchObject({ name: "AbortError" });
    expect(scene.children).toHaveLength(0);
    expect(pointsGeometry).toHaveBeenCalledTimes(1);
    expect(linesMaterial).toHaveBeenCalledTimes(1);
    expect(textureDispose).toHaveBeenCalledTimes(1);
  });

  it("aborts a download disposed early, adds nothing and rejects with AbortError", async () => {
    const scene = new Scene();
    const parse = vi.fn(async () => new Group());
    const { fetch, signals } = abortableDownload();
    const handle = createMeshScene(options({ scene, fetch, parse }));
    handle.dispose();

    await expect(handle.initialized).rejects.toMatchObject({ name: "AbortError" });
    expect(signals).toHaveLength(1);
    expect(signals[0]?.aborted).toBe(true);
    expect(parse).not.toHaveBeenCalled();
    expect(scene.children).toHaveLength(0);
    expect(handle.root).toBeNull();
  });

  it("aborts a download disposed while its body is still streaming", async () => {
    const scene = new Scene();
    let push!: (chunk: Uint8Array) => void;
    let finish!: () => void;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        push = (chunk) => controller.enqueue(chunk);
        finish = () => controller.close();
      },
    });
    const progress: (number | null)[] = [];
    const parse = vi.fn(async () => new Group());
    const handle = createMeshScene(
      options({
        scene,
        fetch: download(() => new Response(body, { headers: { "content-length": "10" } })),
        parse,
        onProgress: (fraction) => progress.push(fraction),
      }),
    );
    push(new Uint8Array(4));
    await vi.waitFor(() => expect(progress).toEqual([0.4]));

    handle.dispose();
    // The stream ends after disposal, as a real aborted download's would.
    finish();
    await expect(handle.initialized).rejects.toMatchObject({ name: "AbortError" });
    expect(parse).not.toHaveBeenCalled();
    expect(scene.children).toHaveLength(0);
    expect(handle.root).toBeNull();
  });

  it("aborts a download whose body errors with AbortError once disposed", async () => {
    const scene = new Scene();
    const progress: (number | null)[] = [];
    const parse = vi.fn(async () => new Group());
    // As fetch does, aborting the request errors the body stream while it is read.
    const fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new Uint8Array(4));
          init?.signal?.addEventListener("abort", () =>
            controller.error(new DOMException("The download was aborted", "AbortError")),
          );
        },
      });
      return new Response(body, { headers: { "content-length": "10" } });
    }) as unknown as Fetch;
    const handle = createMeshScene(
      options({
        scene,
        fetch,
        parse,
        onProgress: (fraction) => progress.push(fraction),
      }),
    );
    await vi.waitFor(() => expect(progress).toEqual([0.4]));

    handle.dispose();
    await expect(handle.initialized).rejects.toMatchObject({ name: "AbortError" });
    expect(parse).not.toHaveBeenCalled();
    expect(scene.children).toHaveLength(0);
    expect(handle.root).toBeNull();
  });

  it("does not parse a response that arrives after disposal", async () => {
    const scene = new Scene();
    const response = deferred<Response>();
    const parse = vi.fn(async () => new Group());
    // A fetch that ignores the abort signal and still resolves.
    const fetch = vi.fn(() => response.promise) as unknown as Fetch;
    const handle = createMeshScene(options({ scene, fetch, parse }));
    handle.dispose();
    response.resolve(streamed([new Uint8Array(3)], { "content-length": "3" }));

    await expect(handle.initialized).rejects.toMatchObject({ name: "AbortError" });
    expect(parse).not.toHaveBeenCalled();
    expect(scene.children).toHaveLength(0);
    expect(handle.root).toBeNull();
  });

  it("frees a parsed scene that finishes after disposal without drawing it", async () => {
    const scene = new Scene();
    const cube = texturedCube();
    const parsed = deferred<Object3D>();
    const parse = vi.fn(() => parsed.promise);
    const geometryDispose = vi.spyOn(cube.geometry, "dispose");
    const materialDispose = vi.spyOn(cube.material, "dispose");
    const textureDispose = vi.spyOn(cube.texture, "dispose");
    const handle = createMeshScene(options({ scene, parse }));
    await vi.waitFor(() => expect(parse).toHaveBeenCalledTimes(1));

    handle.dispose();
    parsed.resolve(cube.group);

    await expect(handle.initialized).rejects.toMatchObject({ name: "AbortError" });
    expect(scene.children).toHaveLength(0);
    expect(handle.root).toBeNull();
    expect(geometryDispose).toHaveBeenCalledTimes(1);
    expect(materialDispose).toHaveBeenCalledTimes(1);
    expect(textureDispose).toHaveBeenCalledTimes(1);
  });

  it("does not report an unawaited cancellation as an unhandled rejection", async () => {
    const unhandled = vi.fn();
    process.on("unhandledRejection", unhandled);
    try {
      const { fetch } = abortableDownload();
      createMeshScene(options({ fetch })).dispose();
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(unhandled).not.toHaveBeenCalled();
    } finally {
      process.off("unhandledRejection", unhandled);
    }
  });

  it("rejects with the HTTP status when the download fails", async () => {
    const scene = new Scene();
    const handle = createMeshScene(
      options({
        scene,
        fetch: download(() => new Response("missing", { status: 404 })),
        parse: async () => new Group(),
      }),
    );
    await expect(handle.initialized).rejects.toThrow(/404/);
    expect(scene.children).toHaveLength(0);
    handle.dispose();
  });

  it("cancels the body of a failed response, which is never read", async () => {
    const cancel = vi.fn();
    const handle = createMeshScene(
      options({
        fetch: download(
          () => new Response(new ReadableStream<Uint8Array>({ cancel }), { status: 500 }),
        ),
        parse: async () => new Group(),
      }),
    );
    await expect(handle.initialized).rejects.toThrow(/500/);
    expect(cancel).toHaveBeenCalledTimes(1);
    handle.dispose();
  });

  it("cancels the body of a response that arrives after disposal", async () => {
    const response = deferred<Response>();
    const cancel = vi.fn();
    const parse = vi.fn(async () => new Group());
    // A fetch that ignores the abort signal and still resolves.
    const fetch = vi.fn(() => response.promise) as unknown as Fetch;
    const handle = createMeshScene(options({ fetch, parse }));
    handle.dispose();
    response.resolve(
      new Response(new ReadableStream<Uint8Array>({ cancel }), {
        headers: { "content-length": "3" },
      }),
    );

    await expect(handle.initialized).rejects.toMatchObject({ name: "AbortError" });
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(parse).not.toHaveBeenCalled();
  });

  it("parses the GLB with the GLTF loader by default", async () => {
    const quad: CollisionMesh = {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 2, 1, 0, 3, 2]),
    };
    const scene = new Scene();
    const glb = Uint8Array.from(writeCollisionGlb(quad));
    const handle = createMeshScene(options({ scene, fetch: download(() => new Response(glb)) }));
    await handle.initialized;

    const root = handle.root;
    expect(root).not.toBeNull();
    expect(scene.children).toContain(root);
    const mesh = root?.children.find((child) => child instanceof Mesh) as Mesh | undefined;
    expect(mesh?.material).toBeInstanceOf(MeshBasicMaterial);
    handle.dispose();
  });
});
