import { describe, expect, it } from "vitest";
import {
  GiB,
  MiB,
  declaredContentType,
  isAcceptedType,
  validateIntake,
  type CandidateFile,
} from "@/features/capture/validation";

const video = (o: Partial<CandidateFile> = {}): CandidateFile => ({
  name: "walk.mp4",
  size: 100 * MiB,
  type: "video/mp4",
  mediaKind: "video",
  ...o,
});
const photo = (i: number, o: Partial<CandidateFile> = {}): CandidateFile => ({
  name: `p${i}.jpg`,
  size: 2 * MiB,
  type: "image/jpeg",
  mediaKind: "photo",
  ...o,
});
const codes = (r: { errors: { code: string }[] }) => r.errors.map((e) => e.code);

describe("validateIntake", () => {
  it("accepts a typical capture", () => {
    const result = validateIntake([video({ durationSeconds: 120 }), photo(1), photo(2)]);
    expect(result.ok).toBe(true);
    expect(result.warnings).toEqual([]);
  });

  it("requires a video or photos", () => {
    const plan: CandidateFile = { name: "plan.pdf", size: MiB, type: "application/pdf", mediaKind: "floor_plan" };
    expect(codes(validateIntake([plan]))).toContain("NO_CAPTURE_MEDIA");
    expect(codes(validateIntake([]))).toContain("NO_CAPTURE_MEDIA");
  });

  it("warns (not errors) when photos are used without a video", () => {
    const result = validateIntake([photo(1)]);
    expect(result.ok).toBe(true);
    expect(result.warnings.map((w) => w.code)).toContain("NO_VIDEO");
  });

  it("allows only one video", () => {
    expect(codes(validateIntake([video(), video({ name: "b.mov", type: "video/quicktime" })]))).toContain(
      "TOO_MANY_VIDEOS",
    );
  });

  it("treats video duration over 5 minutes as advisory", () => {
    const result = validateIntake([video({ durationSeconds: 6 * 60 })]);
    expect(result.ok).toBe(true);
    expect(result.warnings.map((w) => w.code)).toContain("VIDEO_TOO_LONG");
  });

  it("limits photos to 300", () => {
    const ok = Array.from({ length: 300 }, (_, i) => photo(i, { size: 1000 }));
    expect(validateIntake(ok).ok).toBe(true);
    expect(codes(validateIntake([...ok, photo(301, { size: 1000 })]))).toContain("TOO_MANY_PHOTOS");
  });

  it("limits the total to 2 GiB", () => {
    expect(validateIntake([video({ size: 2 * GiB })]).ok).toBe(true);
    expect(codes(validateIntake([video({ size: 2 * GiB }), photo(1, { size: 1 })]))).toContain(
      "TOTAL_TOO_LARGE",
    );
  });

  it("limits floor plans to 20 MiB and one file", () => {
    const plan = (size: number, name = "plan.png"): CandidateFile => ({
      name,
      size,
      type: "image/png",
      mediaKind: "floor_plan",
    });
    expect(validateIntake([video(), plan(20 * MiB)]).ok).toBe(true);
    expect(codes(validateIntake([video(), plan(20 * MiB + 1)]))).toContain("FLOOR_PLAN_TOO_LARGE");
    expect(codes(validateIntake([video(), plan(1), plan(1, "b.png")]))).toContain("TOO_MANY_FLOOR_PLANS");
  });

  it("rejects empty files and unsupported types", () => {
    expect(codes(validateIntake([video({ size: 0 })]))).toContain("EMPTY_FILE");
    expect(codes(validateIntake([video({ name: "walk.webm", type: "video/webm" })]))).toContain(
      "UNSUPPORTED_TYPE",
    );
    expect(codes(validateIntake([video(), photo(1, { name: "x.heic", type: "image/heic" })]))).toContain(
      "UNSUPPORTED_TYPE",
    );
    expect(
      codes(validateIntake([video(), { name: "plan.svg", size: 10, type: "image/svg+xml", mediaKind: "floor_plan" }])),
    ).toContain("UNSUPPORTED_TYPE");
  });
});

describe("type hints", () => {
  it("accepts when either the extension or the MIME type matches", () => {
    expect(isAcceptedType({ name: "WALK.MOV", type: "" }, "video")).toBe(true);
    expect(isAcceptedType({ name: "walk", type: "video/quicktime" }, "video")).toBe(true);
    expect(isAcceptedType({ name: "plan.pdf", type: "" }, "floor_plan")).toBe(true);
    expect(isAcceptedType({ name: "plan.pdf", type: "" }, "photo")).toBe(false);
  });

  it("derives a declared content type from the extension when the browser reports none", () => {
    expect(declaredContentType({ name: "a.MOV", type: "" }, "video")).toBe("video/quicktime");
    expect(declaredContentType({ name: "a.jpeg", type: "" }, "photo")).toBe("image/jpeg");
    expect(declaredContentType({ name: "a.png", type: "image/png" }, "floor_plan")).toBe("image/png");
    expect(declaredContentType({ name: "a.bin", type: "" }, "video")).toBe("application/octet-stream");
  });
});
