import type { Vec2 } from "../types";

/** Maximum look pitch magnitude (radians). Keeps the view from flipping over. */
export const DEFAULT_PITCH_LIMIT = (85 * Math.PI) / 180;
/** Mouse look sensitivity in radians per pixel of pointer-lock movement. */
export const DEFAULT_LOOK_SENSITIVITY = 0.0022;
/** Touch drag-to-look sensitivity in radians per CSS pixel. */
export const DEFAULT_TOUCH_LOOK_SENSITIVITY = 0.005;
/** Virtual move pad radius in CSS pixels. */
export const DEFAULT_MOVE_PAD_RADIUS = 60;

export type MoveAction = "forward" | "backward" | "left" | "right";

export interface MoveKeys {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
}

/** Local movement axes: x = strafe right (+), y = forward (+). */
export interface MoveAxes {
  x: number;
  y: number;
}

export interface LookAngles {
  /** Rotation about +Y in radians; yaw 0 looks down -Z. */
  yaw: number;
  /** Rotation about the camera X axis in radians; positive looks up. */
  pitch: number;
}

const KEY_ACTIONS: Readonly<Record<string, MoveAction>> = {
  KeyW: "forward",
  ArrowUp: "forward",
  KeyS: "backward",
  ArrowDown: "backward",
  KeyA: "left",
  ArrowLeft: "left",
  KeyD: "right",
  ArrowRight: "right",
};

export function keyCodeToAction(code: string): MoveAction | null {
  return KEY_ACTIONS[code] ?? null;
}

export function emptyMoveKeys(): MoveKeys {
  return { forward: false, backward: false, left: false, right: false };
}

export function keysToAxes(keys: MoveKeys): MoveAxes {
  return {
    x: (keys.right ? 1 : 0) - (keys.left ? 1 : 0),
    y: (keys.forward ? 1 : 0) - (keys.backward ? 1 : 0),
  };
}

/** Clamps axes to the unit disc so diagonals are never faster than straight moves. */
export function clampAxes(axes: MoveAxes): MoveAxes {
  const len = Math.hypot(axes.x, axes.y);
  if (!Number.isFinite(len) || len === 0) return { x: 0, y: 0 };
  if (len <= 1) return { x: axes.x, y: axes.y };
  return { x: axes.x / len, y: axes.y / len };
}

/**
 * Converts local movement axes into a world XZ direction relative to the
 * camera yaw (rotation about +Y, camera forward -Z at yaw 0). The result has
 * length <= 1: full-deflection diagonals are normalized.
 */
export function moveDirectionFromAxes(axes: MoveAxes, yaw: number): Vec2 {
  const a = clampAxes(axes);
  const sin = Math.sin(yaw);
  const cos = Math.cos(yaw);
  // forward = (-sin, -cos), right = (cos, -sin) in (x, z).
  const x = -sin * a.y + cos * a.x;
  const z = -cos * a.y - sin * a.x;
  return [x === 0 ? 0 : x, z === 0 ? 0 : z];
}

export function moveDirectionFromKeys(keys: MoveKeys, yaw: number): Vec2 {
  return moveDirectionFromAxes(keysToAxes(keys), yaw);
}

export function clampPitch(pitch: number, limit = DEFAULT_PITCH_LIMIT): number {
  if (!Number.isFinite(pitch)) return 0;
  const l = Math.abs(limit);
  return Math.min(l, Math.max(-l, pitch));
}

/** Wraps an angle into (-PI, PI]. */
export function wrapAngle(angle: number): number {
  if (!Number.isFinite(angle)) return 0;
  const twoPi = Math.PI * 2;
  let a = angle % twoPi;
  if (a <= -Math.PI) a += twoPi;
  if (a > Math.PI) a -= twoPi;
  return a;
}

export interface LookOptions {
  sensitivity: number;
  invertY?: boolean;
  pitchLimit?: number;
}

/**
 * Applies a pointer delta (pixels, screen y down) to look angles. Moving the
 * pointer right turns right (yaw decreases); moving it up looks up.
 */
export function applyLookDelta(
  look: LookAngles,
  dx: number,
  dy: number,
  options: LookOptions,
): LookAngles {
  const sx = Number.isFinite(dx) ? dx : 0;
  const sy = Number.isFinite(dy) ? dy : 0;
  const sign = options.invertY ? 1 : -1;
  return {
    yaw: wrapAngle(look.yaw - sx * options.sensitivity),
    pitch: clampPitch(look.pitch + sign * sy * options.sensitivity, options.pitchLimit),
  };
}

/** Virtual move pad: pointer offset from the touch origin mapped to unit-disc axes. */
export function padAxes(
  originX: number,
  originY: number,
  x: number,
  y: number,
  radius = DEFAULT_MOVE_PAD_RADIUS,
  deadZone = 0.12,
): MoveAxes {
  if (!(radius > 0)) return { x: 0, y: 0 };
  const ax = (x - originX) / radius;
  const ay = -(y - originY) / radius;
  const clamped = clampAxes({ x: ax, y: ay });
  const len = Math.hypot(clamped.x, clamped.y);
  if (len < deadZone) return { x: 0, y: 0 };
  return clamped;
}

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export interface InputControllerOptions {
  /** Element receiving pointer input (normally the viewer canvas). */
  element: HTMLElement;
  lookSensitivity?: number;
  touchLookSensitivity?: number;
  invertY?: boolean;
  pitchLimit?: number;
  /** Disables look smoothing (and any camera motion effects). */
  reducedMotion?: boolean;
  movePadRadius?: number;
  /** Called when the input pauses because the page was hidden or lost focus. */
  onPause?: () => void;
  /** Called when pointer lock is acquired or released (Escape releases it). */
  onPointerLockChange?: (locked: boolean) => void;
  /** Called on any look change so a non-walking (inspection) view can redraw. */
  onLook?: () => void;
}

export interface TouchPadState {
  active: boolean;
  originX: number;
  originY: number;
  x: number;
  y: number;
}

interface TrackedPointer {
  role: "move" | "look";
  originX: number;
  originY: number;
  lastX: number;
  lastY: number;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!target || typeof (target as HTMLElement).tagName !== "string") return false;
  const el = target as HTMLElement;
  const tag = el.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable === true;
}

/**
 * DOM input adapter: keyboard (WASD/arrows), pointer-lock mouse look (desktop),
 * left virtual move pad + right drag-to-look (touch, no pointer lock), and
 * pause on blur/visibility change. All listeners are removed by `detach()`.
 */
export class InputController {
  look: LookAngles = { yaw: 0, pitch: 0 };
  movementEnabled = false;
  lookEnabled = true;
  lookSensitivity: number;
  touchLookSensitivity: number;
  invertY: boolean;
  reducedMotion: boolean;
  readonly pitchLimit: number;

  private readonly element: HTMLElement;
  private readonly options: InputControllerOptions;
  private keys: MoveKeys = emptyMoveKeys();
  private padAxesState: MoveAxes = { x: 0, y: 0 };
  private readonly pointers = new Map<number, TrackedPointer>();
  private pendingLook = { dx: 0, dy: 0, sensitivity: 0 };
  private attached = false;
  private locked = false;
  private readonly removers: (() => void)[] = [];

  constructor(options: InputControllerOptions) {
    this.options = options;
    this.element = options.element;
    this.lookSensitivity = options.lookSensitivity ?? DEFAULT_LOOK_SENSITIVITY;
    this.touchLookSensitivity = options.touchLookSensitivity ?? DEFAULT_TOUCH_LOOK_SENSITIVITY;
    this.invertY = options.invertY ?? false;
    this.reducedMotion = options.reducedMotion ?? prefersReducedMotion();
    this.pitchLimit = options.pitchLimit ?? DEFAULT_PITCH_LIMIT;
  }

  get pointerLocked(): boolean {
    return this.locked;
  }

  attach(): void {
    if (this.attached) return;
    this.attached = true;
    const win = this.element.ownerDocument.defaultView ?? window;
    const doc = this.element.ownerDocument;
    this.listen(win, "keydown", (e) => this.onKey(e as KeyboardEvent, true));
    this.listen(win, "keyup", (e) => this.onKey(e as KeyboardEvent, false));
    this.listen(win, "blur", () => this.pause());
    this.listen(doc, "visibilitychange", () => {
      if (doc.visibilityState === "hidden") this.pause();
    });
    this.listen(doc, "pointerlockchange", () => this.onPointerLockChange());
    this.listen(this.element, "mousemove", (e) => this.onMouseMove(e as MouseEvent));
    this.listen(this.element, "pointerdown", (e) => this.onPointerDown(e as PointerEvent));
    this.listen(this.element, "pointermove", (e) => this.onPointerMove(e as PointerEvent));
    this.listen(this.element, "pointerup", (e) => this.onPointerEnd(e as PointerEvent));
    this.listen(this.element, "pointercancel", (e) => this.onPointerEnd(e as PointerEvent));
    this.listen(this.element, "contextmenu", (e) => e.preventDefault());
  }

  detach(): void {
    while (this.removers.length > 0) this.removers.pop()?.();
    this.attached = false;
    this.clear();
    if (this.locked) this.exitPointerLock();
  }

  /** Requests pointer lock (must be called from a user gesture, e.g. the Walk button). */
  requestPointerLock(): void {
    const el = this.element as HTMLElement & { requestPointerLock?: () => unknown };
    if (typeof el.requestPointerLock !== "function") return;
    try {
      const result = el.requestPointerLock();
      if (result && typeof (result as Promise<void>).catch === "function") {
        (result as Promise<void>).catch(() => undefined);
      }
    } catch {
      // Pointer lock is optional; drag-to-look still works.
    }
  }

  exitPointerLock(): void {
    const doc = this.element.ownerDocument;
    if (doc.pointerLockElement === this.element && typeof doc.exitPointerLock === "function") {
      doc.exitPointerLock();
    }
  }

  /** Clears held keys, touch pads and pending look (used on pause/exit). */
  clear(): void {
    this.keys = emptyMoveKeys();
    this.padAxesState = { x: 0, y: 0 };
    this.pointers.clear();
    this.pendingLook = { dx: 0, dy: 0, sensitivity: 0 };
  }

  /** Current world XZ move direction (length <= 1) relative to the camera yaw. */
  getMoveDirection(): Vec2 {
    if (!this.movementEnabled) return [0, 0];
    const k = keysToAxes(this.keys);
    const axes = clampAxes({ x: k.x + this.padAxesState.x, y: k.y + this.padAxesState.y });
    return moveDirectionFromAxes(axes, this.look.yaw);
  }

  getTouchPadState(): TouchPadState {
    for (const p of this.pointers.values()) {
      if (p.role === "move") {
        return { active: true, originX: p.originX, originY: p.originY, x: p.lastX, y: p.lastY };
      }
    }
    return { active: false, originX: 0, originY: 0, x: 0, y: 0 };
  }

  /**
   * Applies pending look deltas. With reduced motion off, touch look is lightly
   * smoothed; with reduced motion on, deltas apply immediately and fully.
   */
  update(dt: number): void {
    const { dx, dy, sensitivity } = this.pendingLook;
    if (dx === 0 && dy === 0) return;
    const smoothing = this.reducedMotion ? 1 : 1 - Math.exp(-dt * 30);
    const fraction = Math.min(1, Math.max(0, smoothing));
    const ax = dx * fraction;
    const ay = dy * fraction;
    this.look = applyLookDelta(this.look, ax, ay, {
      sensitivity,
      invertY: this.invertY,
      pitchLimit: this.pitchLimit,
    });
    this.pendingLook = {
      dx: Math.abs(dx - ax) < 0.01 ? 0 : dx - ax,
      dy: Math.abs(dy - ay) < 0.01 ? 0 : dy - ay,
      sensitivity,
    };
    this.options.onLook?.();
  }

  setLook(look: LookAngles): void {
    this.look = { yaw: wrapAngle(look.yaw), pitch: clampPitch(look.pitch, this.pitchLimit) };
    this.pendingLook = { dx: 0, dy: 0, sensitivity: 0 };
  }

  private listen(target: EventTarget, type: string, handler: (e: Event) => void): void {
    target.addEventListener(type, handler);
    this.removers.push(() => target.removeEventListener(type, handler));
  }

  private pause(): void {
    this.clear();
    this.options.onPause?.();
  }

  private onKey(event: KeyboardEvent, down: boolean): void {
    if (down && (event.ctrlKey || event.metaKey || event.altKey)) return;
    if (isEditableTarget(event.target)) return;
    const action = keyCodeToAction(event.code);
    if (!action) return;
    if (!this.movementEnabled) {
      if (!down) this.keys[action] = false;
      return;
    }
    this.keys[action] = down;
    if (event.code.startsWith("Arrow")) event.preventDefault();
  }

  private onPointerLockChange(): void {
    const locked = this.element.ownerDocument.pointerLockElement === this.element;
    if (locked === this.locked) return;
    this.locked = locked;
    if (!locked) this.keys = emptyMoveKeys();
    this.options.onPointerLockChange?.(locked);
  }

  private queueLook(dx: number, dy: number, sensitivity: number, immediate: boolean): void {
    if (!this.lookEnabled) return;
    if (immediate) {
      this.look = applyLookDelta(this.look, dx, dy, {
        sensitivity,
        invertY: this.invertY,
        pitchLimit: this.pitchLimit,
      });
      this.options.onLook?.();
      return;
    }
    this.pendingLook = {
      dx: this.pendingLook.dx + dx,
      dy: this.pendingLook.dy + dy,
      sensitivity,
    };
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.locked) return;
    this.queueLook(event.movementX, event.movementY, this.lookSensitivity, true);
  }

  private onPointerDown(event: PointerEvent): void {
    if (this.locked) return;
    const rect = this.element.getBoundingClientRect();
    const isTouch = event.pointerType === "touch" || event.pointerType === "pen";
    const leftHalf = event.clientX - rect.left < rect.width / 2;
    const role: TrackedPointer["role"] =
      isTouch && leftHalf && this.movementEnabled ? "move" : "look";
    if (role === "move") {
      for (const p of this.pointers.values()) if (p.role === "move") return;
    }
    this.pointers.set(event.pointerId, {
      role,
      originX: event.clientX,
      originY: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
    });
    try {
      this.element.setPointerCapture?.(event.pointerId);
    } catch {
      // Capture is best effort.
    }
    if (isTouch) event.preventDefault();
  }

  private onPointerMove(event: PointerEvent): void {
    const p = this.pointers.get(event.pointerId);
    if (!p) return;
    const dx = event.clientX - p.lastX;
    const dy = event.clientY - p.lastY;
    p.lastX = event.clientX;
    p.lastY = event.clientY;
    if (p.role === "move") {
      this.padAxesState = padAxes(
        p.originX,
        p.originY,
        event.clientX,
        event.clientY,
        this.options.movePadRadius ?? DEFAULT_MOVE_PAD_RADIUS,
      );
    } else {
      const touch = event.pointerType !== "mouse";
      this.queueLook(
        dx,
        dy,
        touch ? this.touchLookSensitivity : this.lookSensitivity * 1.5,
        this.reducedMotion || !touch,
      );
    }
  }

  private onPointerEnd(event: PointerEvent): void {
    const p = this.pointers.get(event.pointerId);
    if (!p) return;
    this.pointers.delete(event.pointerId);
    if (p.role === "move") this.padAxesState = { x: 0, y: 0 };
  }
}
