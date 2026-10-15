import { describe, expect, it } from "vitest";
import {
  QUALITY_PROFILES,
  QualityGovernor,
  classifyDevice,
  effectivePixelRatio,
  nextLowerProfile,
  selectQualityProfile,
} from "../src/render/quality";

const IPHONE =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148";
const WINDOWS =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36";
const IPAD_DESKTOP_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15";

describe("quality profiles", () => {
  it("caps DPR at 1.0 mobile and 1.5 desktop with README budgets", () => {
    expect(QUALITY_PROFILES.mobile.maxDpr).toBe(1.0);
    expect(QUALITY_PROFILES.desktop.maxDpr).toBe(1.5);
    expect(QUALITY_PROFILES.mobile.splatBudget).toBe(750_000);
    expect(QUALITY_PROFILES.desktop.splatBudget).toBe(1_500_000);
    expect(QUALITY_PROFILES.low.maxDpr).toBeLessThan(1);
    expect(effectivePixelRatio(3, QUALITY_PROFILES.mobile)).toBe(1);
    expect(effectivePixelRatio(2, QUALITY_PROFILES.desktop)).toBe(1.5);
    expect(effectivePixelRatio(1, QUALITY_PROFILES.desktop)).toBe(1);
    expect(effectivePixelRatio(Number.NaN, QUALITY_PROFILES.desktop)).toBe(1);
  });

  it("sets the texture anisotropy per profile", () => {
    expect(QUALITY_PROFILES.low.maxAnisotropy).toBe(1);
    expect(QUALITY_PROFILES.mobile.maxAnisotropy).toBe(4);
    expect(QUALITY_PROFILES.desktop.maxAnisotropy).toBe(8);
  });

  it("classifies devices", () => {
    expect(classifyDevice({ userAgent: IPHONE })).toBe("mobile");
    expect(classifyDevice({ userAgent: WINDOWS })).toBe("desktop");
    expect(classifyDevice({ userAgent: IPAD_DESKTOP_UA, maxTouchPoints: 5 })).toBe("mobile");
    expect(classifyDevice({ userAgent: IPAD_DESKTOP_UA, maxTouchPoints: 0 })).toBe("desktop");
  });

  it("selects profiles, starting weak hardware lower", () => {
    expect(selectQualityProfile({ userAgent: IPHONE })).toBe("mobile");
    expect(selectQualityProfile({ userAgent: IPHONE, deviceMemory: 2 })).toBe("low");
    expect(selectQualityProfile({ userAgent: WINDOWS, hardwareConcurrency: 16 })).toBe("desktop");
    expect(selectQualityProfile({ userAgent: WINDOWS, hardwareConcurrency: 2 })).toBe("mobile");
    expect(nextLowerProfile("desktop")).toBe("mobile");
    expect(nextLowerProfile("low")).toBeNull();
  });
});

/** Feeds frames of a fixed duration and returns the changes observed. */
function run(governor: QualityGovernor, frameMs: number, durationMs: number, startMs: number) {
  const changes = [];
  let t = startMs;
  while (t < startMs + durationMs) {
    t += frameMs;
    const c = governor.sample(frameMs, t);
    if (c) changes.push({ ...c, at: t });
  }
  return { changes, end: t };
}

describe("QualityGovernor hysteresis", () => {
  it("ignores brief spikes", () => {
    const g = new QualityGovernor({ initial: "desktop", sustainMs: 3000 });
    let r = run(g, 16, 2000, 0);
    r = run(g, 40, 1500, r.end); // 1.5 s over budget is not sustained
    r = run(g, 16, 2000, r.end);
    expect(r.changes).toHaveLength(0);
    expect(g.profile).toBe("desktop");
  });

  it("downgrades after sustained over-budget frames", () => {
    const g = new QualityGovernor({ initial: "desktop", sustainMs: 3000 });
    const r = run(g, 30, 5000, 0);
    expect(r.changes).toHaveLength(1);
    expect(r.changes[0]?.to).toBe("mobile");
    expect(r.changes[0]!.at).toBeGreaterThanOrEqual(3000);
    expect(g.profile).toBe("mobile");
  });

  it("respects the cooldown and steps down one level at a time", () => {
    const g = new QualityGovernor({ initial: "desktop", sustainMs: 3000, cooldownMs: 8000 });
    const r = run(g, 60, 30_000, 0); // over budget for every profile
    expect(r.changes.map((c) => c.to)).toEqual(["mobile", "low"]);
    const gap = r.changes[1]!.at - r.changes[0]!.at;
    expect(gap).toBeGreaterThanOrEqual(8000 + 3000 - 1000);
  });

  it("never oscillates back to a level it left, even with upgrades enabled", () => {
    const g = new QualityGovernor({
      initial: "desktop",
      sustainMs: 2000,
      cooldownMs: 1000,
      allowUpgrade: true,
      upgradeSustainMs: 5000,
    });
    let r = run(g, 30, 4000, 0);
    expect(g.profile).toBe("mobile");
    r = run(g, 5, 60_000, r.end); // very fast frames for a minute
    expect(r.changes).toHaveLength(0);
    expect(g.profile).toBe("mobile");
  });

  it("upgrades only to levels never downgraded from", () => {
    const g = new QualityGovernor({
      initial: "low",
      cooldownMs: 0,
      allowUpgrade: true,
      upgradeSustainMs: 5000,
    });
    const r = run(g, 5, 7000, 0);
    expect(r.changes.map((c) => c.to)).toEqual(["mobile"]);
  });

  it("ignores hitches from tab switches and resets on manual selection", () => {
    const g = new QualityGovernor({ initial: "desktop", sustainMs: 1000 });
    for (let i = 0; i < 20; i++) expect(g.sample(5000, i * 5000)).toBeNull();
    g.set("low", 0);
    expect(g.profile).toBe("low");
  });
});
