import type { SceneManifest } from "@andarilho/scene-contract";
import { PerspectiveCamera, Scene, WebGLRenderer } from "three";
import { createCollisionOverlay, type CollisionOverlay } from "./debug/overlay";
import {
  resolveViewerSource,
  sha256Hex,
  ViewerInputError,
  type ResolvedSource,
  type ViewerSource,
} from "./manifest";
import {
  CollisionAssetError,
  CollisionWorld,
  FixedStepAccumulator,
  initRapier,
  readCollisionGlb,
  validateCollisionMesh,
  type CollisionMeshStats,
  type ControllerConfig,
  type RapierModule,
  type SpawnValidation,
} from "./movement/collision";
import { InputController } from "./movement/input";
import { resolvePlayerConfig, type PlayerConfig } from "./movement/player";
import {
  classifyDevice,
  detectDeviceInfo,
  effectivePixelRatio,
  isQualityProfileName,
  QUALITY_PROFILES,
  QualityGovernor,
  selectQualityProfile,
  type QualityProfileName,
} from "./render/quality";
import type { RenderFormat, SparkSceneHandle } from "./render/spark-scene";
import { layoutToCollisionMesh } from "./review/floorplan";
import type { Aabb, CollisionMesh, Vec2, Vec3 } from "./types";
import {
  canWalk,
  transition,
  type LoadProgress,
  type ViewerError,
  type ViewerEvent,
  type ViewerMode,
  type ViewerState,
} from "./viewer-state";

// ---------------------------------------------------------------------------
// Dependency seams (real implementations by default; mocks in tests)
// ---------------------------------------------------------------------------

/** The subset of THREE.WebGLRenderer the engine uses. */
export interface RendererLike {
  readonly domElement: HTMLCanvasElement;
  setAnimationLoop(callback: ((time: number) => void) | null): void;
  setPixelRatio(value: number): void;
  setSize(width: number, height: number, updateStyle?: boolean): void;
  render(scene: Scene, camera: PerspectiveCamera): void;
  dispose(): void;
}

/** The engine's view of a collision world (see CollisionWorld). */
export interface CollisionWorldLike {
  validateSpawn(feet: Vec3): SpawnValidation;
  setSpawn(feet: Vec3): void;
  resetToSpawn(): void;
  step(direction: Vec2, dt: number): { feet: Vec3; grounded: boolean; recovered: boolean };
  readonly state: { interpolatedFeet(alpha: number): Vec3; feet: Vec3 };
  dispose(): void;
}

export interface CreateSparkSceneArgs {
  renderer: RendererLike;
  scene: Scene;
  url: string;
  format: RenderFormat;
  worldFromDecodedRender: readonly number[];
  splatBudget: number;
  onProgress: (fraction: number | null) => void;
}

export interface CreateCollisionWorldArgs {
  rapier: RapierModule;
  mesh: CollisionMesh;
  player: PlayerConfig;
  controller?: Partial<ControllerConfig>;
  bounds: Aabb;
  step: number;
}

export interface ViewerDependencies {
  isWebGL2Available(): boolean;
  createRenderer(): RendererLike;
  createSparkScene(args: CreateSparkSceneArgs): Promise<SparkSceneHandle> | SparkSceneHandle;
  initRapier(): Promise<RapierModule>;
  createCollisionWorld(args: CreateCollisionWorldArgs): CollisionWorldLike;
  fetchBytes(url: string, signal: AbortSignal): Promise<ArrayBuffer>;
  now(): number;
  ResizeObserver: typeof ResizeObserver | undefined;
}

function defaultIsWebGL2Available(): boolean {
  try {
    if (typeof document === "undefined") return false;
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2");
    if (!gl) return false;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return true;
  } catch {
    return false;
  }
}

async function defaultFetchBytes(url: string, signal: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal, credentials: "omit" });
  if (!response.ok) throw new Error(`Collision asset request failed (${response.status})`);
  return response.arrayBuffer();
}

/**
 * Resolves engine dependencies. Spark and Rapier (large, browser/WASM-heavy)
 * are imported lazily; nothing touches WebGL until createViewer runs.
 */
function resolveDependencies(partial: Partial<ViewerDependencies> | undefined): ViewerDependencies {
  return {
    isWebGL2Available: defaultIsWebGL2Available,
    async createSparkScene(args) {
      const { createSparkScene } = await import("./render/spark-scene");
      return createSparkScene({
        // The default renderer factory produces a real WebGLRenderer.
        renderer: args.renderer as WebGLRenderer,
        scene: args.scene,
        url: args.url,
        format: args.format,
        worldFromDecodedRender: args.worldFromDecodedRender,
        splatBudget: args.splatBudget,
        onProgress: args.onProgress,
      });
    },
    // Rapier's WASM module is imported lazily inside initRapier().
    initRapier,
    createRenderer: () =>
      new WebGLRenderer({ antialias: false, powerPreference: "high-performance" }),
    createCollisionWorld: (args) =>
      new CollisionWorld(args.rapier, args.mesh, {
        player: args.player,
        bounds: args.bounds,
        step: args.step,
        ...(args.controller ? { controller: args.controller } : {}),
      }),
    fetchBytes: defaultFetchBytes,
    now: () => (typeof performance !== "undefined" ? performance.now() : Date.now()),
    ResizeObserver: typeof ResizeObserver !== "undefined" ? ResizeObserver : undefined,
    ...partial,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export type QualitySetting = QualityProfileName | "auto";

export interface CreateViewerOptions {
  /** Element that will own the viewer canvas. Sized by CSS; observed for resizes. */
  container: HTMLElement;
  /** Validated delivery envelope, or explicit URLs for dev/fixture mode. */
  source: ViewerSource;
  /** Initial quality: a fixed profile or "auto" (device heuristic + downgrade). */
  quality?: QualitySetting;
  /** Overrides for the capsule (manifest navigation values apply first). */
  player?: Partial<PlayerConfig>;
  controller?: Partial<ControllerConfig>;
  /** Mouse look sensitivity (radians per pixel). */
  lookSensitivity?: number;
  invertY?: boolean;
  /** Reduced-motion behavior; defaults to the OS preference. */
  reducedMotion?: boolean;
  /** Use pointer lock when walking: true, false or "auto" (desktop only). */
  pointerLock?: boolean | "auto";
  /** Called on every state change (also available through subscribe()). */
  onStateChange?: (state: ViewerState) => void;
  /** Accessible label of the canvas. */
  ariaLabel?: string;
  /** Test/advanced seam: replace engine dependencies. */
  dependencies?: Partial<ViewerDependencies>;
}

export interface ViewerDiagnostics {
  quality: QualityProfileName;
  qualitySetting: QualitySetting;
  collision: CollisionMeshStats | null;
  spawn: Vec3 | null;
  feet: Vec3 | null;
  inspectionReason: string | null;
}

export interface ViewerHandle {
  /** Current state snapshot (stable object between changes; useSyncExternalStore-friendly). */
  getState(): ViewerState;
  /** Subscribes to state changes; returns an unsubscribe function. */
  subscribe(listener: (state: ViewerState) => void): () => void;
  /** Enters walking (call from a user gesture so pointer lock can engage). Returns success. */
  enterWalk(): boolean;
  exitWalk(): void;
  /** Returns the player to the validated spawn and spawn yaw. */
  resetPosition(): void;
  setQuality(setting: QualitySetting): void;
  setCollisionOverlay(visible: boolean): void;
  setLookSensitivity(radiansPerPixel: number): void;
  setReducedMotion(enabled: boolean): void;
  getDiagnostics(): ViewerDiagnostics;
  /** Stops the loop, removes listeners/observers, frees Rapier/Spark/WebGL and removes the canvas. */
  dispose(): void;
}

class DisposerStack {
  private readonly items: (() => void)[] = [];
  push(fn: () => void): void {
    this.items.push(fn);
  }
  disposeAll(): void {
    while (this.items.length > 0) {
      const fn = this.items.pop()!;
      try {
        fn();
      } catch {
        // Keep releasing the remaining resources.
      }
    }
  }
}

function safeMessage(error: unknown, fallback: string): string {
  if (error instanceof ViewerInputError) return "The scene data is invalid.";
  if (error instanceof CollisionAssetError) return "The collision asset is invalid.";
  if (error instanceof Error && error.name === "RenderTransformError")
    return "The scene transform is invalid.";
  return fallback;
}

function navigationPlayer(
  manifest: SceneManifest,
  overrides?: Partial<PlayerConfig>,
): PlayerConfig {
  const nav = manifest.navigation;
  const fromManifest: Partial<PlayerConfig> = {};
  if (nav?.playerRadius !== undefined) fromManifest.radius = nav.playerRadius;
  if (nav?.playerHeight !== undefined) fromManifest.height = nav.playerHeight;
  if (nav?.eyeHeight !== undefined) fromManifest.eyeHeight = nav.eyeHeight;
  return resolvePlayerConfig({ ...fromManifest, ...overrides });
}

/**
 * Creates one viewer engine bound to `container`. The engine owns its canvas,
 * animation loop, loading state and GPU resources; React should keep only the
 * returned handle (e.g. in a ref) and render controls from `subscribe()`.
 */
export function createViewer(options: CreateViewerOptions): ViewerHandle {
  let state: ViewerState = { status: "idle" };
  const listeners = new Set<(s: ViewerState) => void>();
  const resources = new DisposerStack();
  const abort = new AbortController();
  let disposed = false;

  const dispatch = (event: ViewerEvent): void => {
    const next = transition(state, event);
    if (next === state) return;
    state = next;
    options.onStateChange?.(state);
    for (const listener of [...listeners]) listener(state);
  };
  const fail = (error: ViewerError): void => dispatch({ type: "fail", error });

  // Engine state shared by the closures below.
  let deps: ViewerDependencies | null = null;
  let source: ResolvedSource | null = null;
  let player: PlayerConfig | null = null;
  let renderer: RendererLike | null = null;
  let sparkScene: SparkSceneHandle | null = null;
  let world: CollisionWorldLike | null = null;
  let collisionMesh: CollisionMesh | null = null;
  let collisionStats: CollisionMeshStats | null = null;
  let overlay: CollisionOverlay | null = null;
  let overlayWanted = false;
  let input: InputController | null = null;
  let spawnFeet: Vec3 | null = null;
  let inspectionReason: string | null = null;
  let paused = false;
  let contextLost = false;
  let lastFrame: number | null = null;
  let qualitySetting: QualitySetting = options.quality ?? "auto";
  let qualityName: QualityProfileName = "desktop";
  let governor: QualityGovernor | null = null;
  const scene = new Scene();
  const camera = new PerspectiveCamera(65, 1, 0.05, 250);
  const accumulator = new FixedStepAccumulator({
    step: 1 / 60,
    maxFrameDelta: 0.1,
    maxSubsteps: 6,
  });
  const usePointerLock =
    options.pointerLock === "auto" || options.pointerLock === undefined
      ? classifyDevice(detectDeviceInfo()) === "desktop"
      : options.pointerLock;

  const applyQuality = (name: QualityProfileName): void => {
    qualityName = name;
    const profile = QUALITY_PROFILES[name];
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio : 1;
    renderer?.setPixelRatio(effectivePixelRatio(dpr, profile));
    sparkScene?.setSplatBudget(profile.splatBudget);
    resize();
    dispatch({ type: "quality", quality: name });
  };

  const resize = (): void => {
    if (!renderer) return;
    const width = Math.max(1, Math.floor(options.container.clientWidth));
    const height = Math.max(1, Math.floor(options.container.clientHeight));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    governor?.resetWindow();
  };

  const placeCamera = (feet: Vec3): void => {
    if (!player || !input) return;
    camera.position.set(feet[0], feet[1] + player.eyeHeight, feet[2]);
    camera.rotation.set(input.look.pitch, input.look.yaw, 0, "YXZ");
  };

  const ensureOverlay = (): void => {
    if (overlay || !collisionMesh || !overlayWanted || disposed) return;
    overlay = createCollisionOverlay(collisionMesh, {
      spawn: source
        ? { feet: source.manifest.spawn.feet, yawRadians: source.manifest.spawn.yawRadians }
        : null,
      playerRadius: player?.radius ?? 0.25,
    });
    scene.add(overlay.group);
    const created = overlay;
    resources.push(() => created.dispose());
  };

  const frame = (): void => {
    if (disposed || contextLost || !deps || !renderer) return;
    const now = deps.now();
    if (paused) {
      lastFrame = now;
      return;
    }
    const dt = lastFrame === null ? 0 : Math.max(0, (now - lastFrame) / 1000);
    lastFrame = now;
    input?.update(dt);

    let feet: Vec3 | null = spawnFeet;
    if (world) {
      if (state.status === "walking" && input) {
        const { steps, alpha } = accumulator.advance(dt);
        for (let i = 0; i < steps; i++) world.step(input.getMoveDirection(), accumulator.step);
        feet = world.state.interpolatedFeet(alpha);
      } else {
        feet = world.state.feet;
      }
    }
    if (feet) placeCamera(feet);
    overlay?.setPlayerFeet(state.status === "walking" ? feet : null);

    if (governor && dt > 0) {
      const change = governor.sample(dt * 1000, now);
      if (change) applyQuality(change.to);
    }
    renderer.render(scene, camera);
  };

  const setPaused = (value: boolean): void => {
    paused = value;
    lastFrame = null;
    accumulator.reset();
    governor?.resetWindow();
  };

  // --- loading -------------------------------------------------------------

  const loadRender = async (): Promise<void> => {
    if (!deps || !source || !renderer) return;
    const handle = await deps.createSparkScene({
      renderer,
      scene,
      url: source.renderUrl,
      format: source.renderFormat,
      worldFromDecodedRender: source.manifest.worldFromDecodedRender,
      splatBudget: QUALITY_PROFILES[qualityName].splatBudget,
      onProgress: (fraction) => {
        if (!disposed) dispatch({ type: "progress", progress: { renderFraction: fraction } });
      },
    });
    if (disposed) {
      handle.dispose();
      return;
    }
    sparkScene = handle;
    resources.push(() => {
      if (sparkScene === handle) sparkScene = null;
      handle.dispose();
    });
    await handle.initialized;
  };

  const loadCollisionMesh = async (): Promise<CollisionMesh | null> => {
    if (!deps || !source) return null;
    const bounds = source.manifest.bounds;
    if (source.collisionUrl) {
      const bytes = await deps.fetchBytes(source.collisionUrl, abort.signal);
      if (source.collisionSha256) {
        const digest = await sha256Hex(bytes);
        if (digest !== null && digest !== source.collisionSha256.toLowerCase()) {
          throw new CollisionAssetError(["collision asset checksum mismatch"]);
        }
      }
      const validated = readCollisionGlb(bytes, { bounds });
      collisionStats = validated.stats;
      return validated.mesh;
    }
    if (source.layout) {
      const mesh = layoutToCollisionMesh(source.layout);
      const validated = validateCollisionMesh(mesh, { bounds });
      collisionStats = validated.stats;
      return validated.mesh;
    }
    return null;
  };

  const loadPhysics = async (): Promise<{ mode: ViewerMode; reason: string | null }> => {
    if (!deps || !source || !player)
      return { mode: "inspection", reason: "Viewer not initialized." };
    const nav = source.manifest.navigation;
    const [rapierResult, meshResult] = await Promise.allSettled([
      deps.initRapier(),
      loadCollisionMesh(),
    ]);
    if (disposed) return { mode: "inspection", reason: null };
    dispatch({
      type: "progress",
      progress: {
        physics: rapierResult.status === "fulfilled" ? "done" : "failed",
        collision:
          meshResult.status === "rejected" ? "failed" : meshResult.value ? "done" : "skipped",
      },
    });
    if (meshResult.status === "rejected") {
      return {
        mode: "inspection",
        reason: safeMessage(meshResult.reason, "Collision geometry could not be loaded."),
      };
    }
    if (!meshResult.value)
      return { mode: "inspection", reason: "No collision geometry is available." };
    collisionMesh = meshResult.value;
    ensureOverlay();
    if (rapierResult.status === "rejected") {
      return { mode: "inspection", reason: "The physics engine could not start." };
    }
    const created = deps.createCollisionWorld({
      rapier: rapierResult.value,
      mesh: collisionMesh,
      player,
      ...(options.controller ? { controller: options.controller } : {}),
      bounds: { min: [...source.manifest.bounds.min], max: [...source.manifest.bounds.max] },
      step: accumulator.step,
    });
    if (disposed) {
      created.dispose();
      return { mode: "inspection", reason: null };
    }
    resources.push(() => {
      if (world === created) world = null;
      created.dispose();
    });
    const spawn = created.validateSpawn([...source.manifest.spawn.feet]);
    if (!spawn.ok || !spawn.feet) {
      dispatch({ type: "progress", progress: { spawn: "failed" } });
      created.dispose();
      return {
        mode: "inspection",
        reason:
          spawn.reason === "blocked"
            ? "The spawn point does not have enough clearance."
            : "The spawn point is not supported by the floor.",
      };
    }
    created.setSpawn(spawn.feet);
    spawnFeet = spawn.feet;
    world = created;
    dispatch({ type: "progress", progress: { spawn: "done" } });
    if (nav?.mode !== "approved_collision" || !nav.approved) {
      return { mode: "inspection", reason: "Navigation geometry is not approved for walking." };
    }
    return { mode: "walkable", reason: null };
  };

  const start = async (): Promise<void> => {
    try {
      deps = resolveDependencies(options.dependencies);
      // Yield once so the caller receives the handle (and can subscribe)
      // before the first state transition.
      await Promise.resolve();
      if (disposed) return;
      if (!deps.isWebGL2Available()) {
        fail({
          code: "webgl2-unavailable",
          message: "This device or browser does not support WebGL2.",
          recoverable: false,
        });
        return;
      }
      try {
        source = resolveViewerSource(options.source);
        player = navigationPlayer(source.manifest, options.player);
      } catch (error) {
        fail({
          code: "invalid-input",
          message: safeMessage(error, "The scene data is invalid."),
          recoverable: true,
        });
        return;
      }
      dispatch({ type: "load-start" });
      setupEngine(deps);
      if (disposed) return;

      const [renderResult, physicsResult] = await Promise.allSettled([loadRender(), loadPhysics()]);
      if (disposed) return;
      if (renderResult.status === "rejected") {
        dispatch({ type: "progress", progress: { render: "failed" } });
        fail({
          code: "render-load-failed",
          message: safeMessage(renderResult.reason, "The scene could not be loaded."),
          recoverable: true,
        });
        return;
      }
      dispatch({ type: "progress", progress: { render: "done" } });
      const physics =
        physicsResult.status === "fulfilled"
          ? physicsResult.value
          : { mode: "inspection" as const, reason: "Collision setup failed." };
      inspectionReason = physics.reason;
      if (contextLost) return;
      dispatch({
        type: "loaded",
        mode: physics.mode,
        inspectionReason: physics.reason,
        quality: qualityName,
      });
    } catch (error) {
      if (disposed) return;
      resources.disposeAll();
      renderer = null;
      fail({
        code: "init-failed",
        message: safeMessage(error, "The viewer could not start."),
        recoverable: true,
      });
    }
  };

  const setupEngine = (d: ViewerDependencies): void => {
    if (!source || !player) return;
    const created = d.createRenderer();
    renderer = created;
    resources.push(() => {
      created.dispose();
      if (renderer === created) renderer = null;
    });
    const canvas = created.domElement;
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.touchAction = "none";
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "application");
    canvas.setAttribute("aria-label", options.ariaLabel ?? "3D scene viewer");
    options.container.appendChild(canvas);
    resources.push(() => canvas.remove());

    // Quality profile.
    if (qualitySetting === "auto") {
      qualityName = selectQualityProfile(detectDeviceInfo());
      governor = new QualityGovernor({ initial: qualityName });
    } else {
      qualityName = qualitySetting;
    }
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio : 1;
    created.setPixelRatio(effectivePixelRatio(dpr, QUALITY_PROFILES[qualityName]));

    // Camera at the manifest spawn; collision may refine the feet later.
    const spawn = source.manifest.spawn;
    spawnFeet = [...spawn.feet];
    const controller = new InputController({
      element: canvas,
      ...(options.lookSensitivity !== undefined
        ? { lookSensitivity: options.lookSensitivity }
        : {}),
      ...(options.invertY !== undefined ? { invertY: options.invertY } : {}),
      ...(options.reducedMotion !== undefined ? { reducedMotion: options.reducedMotion } : {}),
      onPause: () => accumulator.reset(),
      onPointerLockChange: (locked) => {
        dispatch({ type: "pointer-lock", locked });
        // Escape releases pointer lock; treat that as leaving walk mode.
        if (!locked && state.status === "walking" && usePointerLock) handle.exitWalk();
      },
    });
    controller.setLook({ yaw: spawn.yawRadians, pitch: 0 });
    controller.attach();
    input = controller;
    resources.push(() => {
      controller.detach();
      if (input === controller) input = null;
    });
    placeCamera(spawnFeet);

    // Resize handling.
    const RO = d.ResizeObserver;
    if (RO) {
      const observer = new RO(() => resize());
      observer.observe(options.container);
      resources.push(() => observer.disconnect());
    } else if (typeof window !== "undefined") {
      window.addEventListener("resize", resize);
      resources.push(() => window.removeEventListener("resize", resize));
    }
    resize();

    // Pause when hidden; resume with a fresh clock (no huge catch-up step).
    if (typeof document !== "undefined") {
      const onVisibility = () => setPaused(document.visibilityState === "hidden");
      document.addEventListener("visibilitychange", onVisibility);
      resources.push(() => document.removeEventListener("visibilitychange", onVisibility));
      if (document.visibilityState === "hidden") paused = true;
    }

    // WebGL context loss/restore.
    const onLost = (event: Event) => {
      event.preventDefault();
      contextLost = true;
      input?.exitPointerLock();
      if (input) input.movementEnabled = false;
      dispatch({ type: "context-lost" });
    };
    const onRestored = () => {
      if (disposed) return;
      contextLost = false;
      void recoverRender();
    };
    canvas.addEventListener("webglcontextlost", onLost);
    canvas.addEventListener("webglcontextrestored", onRestored);
    resources.push(() => {
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("webglcontextrestored", onRestored);
    });

    created.setAnimationLoop(frame);
    resources.push(() => created.setAnimationLoop(null));
  };

  const recoverRender = async (): Promise<void> => {
    if (state.status !== "context-lost") return;
    const resume = state.resumeMode;
    const reason = state.inspectionReason;
    dispatch({ type: "load-start" });
    try {
      sparkScene?.dispose();
      sparkScene = null;
      await loadRender();
      if (disposed) return;
      const mode: ViewerMode = resume === "walkable" && world ? "walkable" : "inspection";
      dispatch({
        type: "loaded",
        mode,
        inspectionReason: mode === "inspection" ? reason : null,
        quality: qualityName,
      });
    } catch {
      if (disposed) return;
      fail({
        code: "context-lost",
        message: "The graphics context was lost and the scene could not be restored.",
        recoverable: true,
      });
    }
  };

  const handle: ViewerHandle = {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    enterWalk() {
      if (disposed || !canWalk(state) || !input) return false;
      input.movementEnabled = true;
      accumulator.reset();
      lastFrame = null;
      if (usePointerLock) input.requestPointerLock();
      dispatch({ type: "enter-walk", pointerLocked: input.pointerLocked });
      return true;
    },
    exitWalk() {
      if (state.status !== "walking") return;
      if (input) {
        input.movementEnabled = false;
        input.clear();
        input.exitPointerLock();
      }
      dispatch({ type: "exit-walk" });
    },
    resetPosition() {
      if (disposed || !source) return;
      world?.resetToSpawn();
      accumulator.reset();
      input?.setLook({ yaw: source.manifest.spawn.yawRadians, pitch: 0 });
      if (spawnFeet) placeCamera(spawnFeet);
    },
    setQuality(setting) {
      if (disposed) return;
      if (setting !== "auto" && !isQualityProfileName(setting)) return;
      qualitySetting = setting;
      if (setting === "auto") {
        const initial = selectQualityProfile(detectDeviceInfo());
        governor = new QualityGovernor({ initial });
        applyQuality(initial);
      } else {
        governor = null;
        applyQuality(setting);
      }
    },
    setCollisionOverlay(visible) {
      overlayWanted = visible;
      if (visible) ensureOverlay();
      overlay?.setVisible(visible);
    },
    setLookSensitivity(value) {
      if (input && Number.isFinite(value) && value > 0) input.lookSensitivity = value;
    },
    setReducedMotion(enabled) {
      if (input) input.reducedMotion = enabled;
    },
    getDiagnostics() {
      return {
        quality: qualityName,
        qualitySetting,
        collision: collisionStats,
        spawn: spawnFeet ? [...spawnFeet] : null,
        feet: world ? [...world.state.feet] : null,
        inspectionReason,
      };
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      abort.abort();
      resources.disposeAll();
      world = null;
      sparkScene = null;
      overlay = null;
      input = null;
      renderer = null;
      dispatch({ type: "dispose" });
      listeners.clear();
    },
  };

  void start();
  return handle;
}

export type { LoadProgress, ViewerError, ViewerMode, ViewerState };
