/**
 * Device quality profiles and a hysteresis-based governor (README section 11,
 * "Device budgets and fallback"). Pure logic: no DOM access except the optional
 * `detectDeviceInfo()` helper, which callers may replace in tests.
 */

export type QualityProfileName = "low" | "mobile" | "desktop";

export interface QualityProfile {
  name: QualityProfileName;
  /** Device pixel ratio cap. */
  maxDpr: number;
  /** Target visible splat budget (used as Spark's LoD splat count). */
  splatBudget: number;
  /** Frame-time budget in milliseconds (30 FPS mobile, 60 FPS desktop). */
  frameBudgetMs: number;
}

export const QUALITY_PROFILES: Readonly<Record<QualityProfileName, Readonly<QualityProfile>>> =
  Object.freeze({
    low: Object.freeze({
      name: "low",
      maxDpr: 0.75,
      splatBudget: 400_000,
      frameBudgetMs: 1000 / 30,
    }),
    mobile: Object.freeze({
      name: "mobile",
      maxDpr: 1.0,
      splatBudget: 750_000,
      frameBudgetMs: 1000 / 30,
    }),
    desktop: Object.freeze({
      name: "desktop",
      maxDpr: 1.5,
      splatBudget: 1_500_000,
      frameBudgetMs: 1000 / 60,
    }),
  });

/** Order used when downgrading: each step reduces DPR and detail. */
export const DOWNGRADE_ORDER: readonly QualityProfileName[] = ["desktop", "mobile", "low"];

export function isQualityProfileName(value: unknown): value is QualityProfileName {
  return value === "low" || value === "mobile" || value === "desktop";
}

export function nextLowerProfile(name: QualityProfileName): QualityProfileName | null {
  const i = DOWNGRADE_ORDER.indexOf(name);
  return i >= 0 && i < DOWNGRADE_ORDER.length - 1 ? (DOWNGRADE_ORDER[i + 1] ?? null) : null;
}

export function nextHigherProfile(name: QualityProfileName): QualityProfileName | null {
  const i = DOWNGRADE_ORDER.indexOf(name);
  return i > 0 ? (DOWNGRADE_ORDER[i - 1] ?? null) : null;
}

export interface DeviceInfo {
  userAgent?: string;
  maxTouchPoints?: number;
  hardwareConcurrency?: number;
  /** navigator.deviceMemory in GiB, when exposed. */
  deviceMemory?: number;
  /** Shortest screen side in CSS pixels. */
  screenMinSide?: number;
  /** True when the primary pointer is coarse (touch). */
  coarsePointer?: boolean;
}

export type DeviceClass = "mobile" | "desktop";

const MOBILE_UA = /Android|iPhone|iPad|iPod|Mobile|Silk|Kindle|Opera Mini|IEMobile/i;

/** Heuristic device classification; err on the side of the lighter profile. */
export function classifyDevice(info: DeviceInfo): DeviceClass {
  const ua = info.userAgent ?? "";
  if (MOBILE_UA.test(ua)) return "mobile";
  // iPadOS reports a desktop Safari UA but has touch points.
  if (/Macintosh/.test(ua) && (info.maxTouchPoints ?? 0) > 1) return "mobile";
  if (info.coarsePointer && (info.screenMinSide ?? Infinity) < 820) return "mobile";
  return "desktop";
}

/** Picks the initial profile for a device; weak hardware starts lower. */
export function selectQualityProfile(info: DeviceInfo): QualityProfileName {
  const device = classifyDevice(info);
  const weak =
    (info.deviceMemory !== undefined && info.deviceMemory <= 2) ||
    (info.hardwareConcurrency !== undefined && info.hardwareConcurrency <= 2);
  if (device === "mobile") return weak ? "low" : "mobile";
  return weak ? "mobile" : "desktop";
}

export function effectivePixelRatio(devicePixelRatio: number, profile: QualityProfile): number {
  const dpr = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0 ? devicePixelRatio : 1;
  return Math.min(dpr, profile.maxDpr);
}

/** Reads the current browser's device information (safe during SSR). */
export function detectDeviceInfo(): DeviceInfo {
  if (typeof navigator === "undefined") return {};
  const nav = navigator as Navigator & { deviceMemory?: number };
  const info: DeviceInfo = {
    userAgent: nav.userAgent,
    maxTouchPoints: nav.maxTouchPoints,
    hardwareConcurrency: nav.hardwareConcurrency,
  };
  if (typeof nav.deviceMemory === "number") info.deviceMemory = nav.deviceMemory;
  if (typeof window !== "undefined") {
    if (window.screen) info.screenMinSide = Math.min(window.screen.width, window.screen.height);
    if (typeof window.matchMedia === "function") {
      info.coarsePointer = window.matchMedia("(pointer: coarse)").matches;
    }
  }
  return info;
}

export interface QualityGovernorOptions {
  initial: QualityProfileName;
  /** Sustained window length in ms before a downgrade is considered. */
  sustainMs?: number;
  /** Ratio over the frame budget that counts as "over budget". */
  overBudgetRatio?: number;
  /** Ratio under the budget required to consider an upgrade. */
  underBudgetRatio?: number;
  /** Cooldown after any change during which no change happens, ms. */
  cooldownMs?: number;
  /** Allow automatic upgrades (off by default; never back to a level we left). */
  allowUpgrade?: boolean;
  /** Upgrade window length, ms. */
  upgradeSustainMs?: number;
  /** Frames longer than this are treated as hitches (tab switch) and ignored. */
  outlierMs?: number;
  /** Lowest profile the governor may select. */
  floor?: QualityProfileName;
}

export interface QualityChange {
  from: QualityProfileName;
  to: QualityProfileName;
  reason: "sustained-over-budget" | "sustained-under-budget";
  averageFrameMs: number;
}

/**
 * Reduces DPR/detail when the average frame time stays over budget for a
 * sustained window, with a cooldown and window reset after each change. A level
 * that was downgraded from is never automatically re-entered, so quality cannot
 * oscillate.
 */
export class QualityGovernor {
  private current: QualityProfileName;
  private readonly opts: Required<Omit<QualityGovernorOptions, "initial">>;
  private windowStart: number | null = null;
  private windowSum = 0;
  private windowFrames = 0;
  private overStart: number | null = null;
  private underStart: number | null = null;
  private cooldownUntil = -Infinity;
  private readonly blocked = new Set<QualityProfileName>();

  constructor(options: QualityGovernorOptions) {
    this.current = options.initial;
    this.opts = {
      sustainMs: options.sustainMs ?? 3000,
      overBudgetRatio: options.overBudgetRatio ?? 1.25,
      underBudgetRatio: options.underBudgetRatio ?? 0.5,
      cooldownMs: options.cooldownMs ?? 8000,
      allowUpgrade: options.allowUpgrade ?? false,
      upgradeSustainMs: options.upgradeSustainMs ?? 20000,
      outlierMs: options.outlierMs ?? 250,
      floor: options.floor ?? "low",
    };
  }

  get profile(): QualityProfileName {
    return this.current;
  }

  /** Manual override (user quality selector): resets history and blocks. */
  set(profile: QualityProfileName, nowMs = -Infinity): void {
    this.current = profile;
    this.blocked.clear();
    this.resetWindow();
    this.cooldownUntil = nowMs + this.opts.cooldownMs;
  }

  /** Clears the measurement window (e.g. after a pause or resize). */
  resetWindow(): void {
    this.windowStart = null;
    this.windowSum = 0;
    this.windowFrames = 0;
    this.overStart = null;
    this.underStart = null;
  }

  /**
   * Records one frame time. Returns a change when the profile should switch.
   * Averages are computed over ~0.5 s buckets to smooth out single spikes.
   */
  sample(frameMs: number, nowMs: number): QualityChange | null {
    if (!Number.isFinite(frameMs) || frameMs <= 0 || frameMs > this.opts.outlierMs) return null;
    if (nowMs < this.cooldownUntil) return null;
    if (this.windowStart === null) this.windowStart = nowMs;
    this.windowSum += frameMs;
    this.windowFrames++;
    if (nowMs - this.windowStart < 500) return null;

    const average = this.windowSum / this.windowFrames;
    const bucketStart = this.windowStart;
    this.windowStart = nowMs;
    this.windowSum = 0;
    this.windowFrames = 0;

    const budget = QUALITY_PROFILES[this.current].frameBudgetMs;
    if (average > budget * this.opts.overBudgetRatio) {
      this.underStart = null;
      if (this.overStart === null) this.overStart = bucketStart;
      if (nowMs - this.overStart >= this.opts.sustainMs) {
        const lower = nextLowerProfile(this.current);
        const floorIndex = DOWNGRADE_ORDER.indexOf(this.opts.floor);
        if (lower && DOWNGRADE_ORDER.indexOf(lower) <= floorIndex) {
          return this.change(lower, "sustained-over-budget", average, nowMs, true);
        }
        this.overStart = null;
      }
      return null;
    }
    this.overStart = null;
    if (this.opts.allowUpgrade && average < budget * this.opts.underBudgetRatio) {
      if (this.underStart === null) this.underStart = bucketStart;
      if (nowMs - this.underStart >= this.opts.upgradeSustainMs) {
        const higher = nextHigherProfile(this.current);
        if (higher && !this.blocked.has(higher)) {
          return this.change(higher, "sustained-under-budget", average, nowMs, false);
        }
        this.underStart = null;
      }
    } else {
      this.underStart = null;
    }
    return null;
  }

  private change(
    to: QualityProfileName,
    reason: QualityChange["reason"],
    averageFrameMs: number,
    nowMs: number,
    downgrade: boolean,
  ): QualityChange {
    const from = this.current;
    if (downgrade) this.blocked.add(from);
    this.current = to;
    this.resetWindow();
    this.cooldownUntil = nowMs + this.opts.cooldownMs;
    return { from, to, reason, averageFrameMs };
  }
}
