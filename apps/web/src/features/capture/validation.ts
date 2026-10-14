/**
 * Client-side intake checks (README section 2 "Suggested beta intake limits"
 * and section 7 "Supported intake"). Extension and MIME type are hints only:
 * the API and the inspection worker verify file signatures and decodability.
 * These checks exist to give fast, friendly feedback before uploading.
 */
import type { MediaKind } from "@/lib/api/types";

export const MiB = 1024 * 1024;
export const GiB = 1024 * MiB;

export const INTAKE_LIMITS = {
  maxVideos: 1,
  maxVideoSeconds: 5 * 60, // advisory
  maxPhotos: 300,
  maxTotalBytes: 2 * GiB,
  maxFloorPlanBytes: 20 * MiB,
} as const;

interface TypeRule {
  extensions: readonly string[];
  mimeTypes: readonly string[];
  /** Content type to declare when the browser reports none/unknown, by extension. */
  byExtension: Readonly<Record<string, string>>;
}

export const TYPE_RULES: Record<MediaKind, TypeRule> = {
  video: {
    extensions: [".mp4", ".mov"],
    mimeTypes: ["video/mp4", "video/quicktime"],
    byExtension: { ".mp4": "video/mp4", ".mov": "video/quicktime" },
  },
  photo: {
    extensions: [".jpg", ".jpeg", ".png"],
    mimeTypes: ["image/jpeg", "image/png"],
    byExtension: { ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png" },
  },
  floor_plan: {
    extensions: [".png", ".jpg", ".jpeg", ".pdf"],
    mimeTypes: ["image/png", "image/jpeg", "application/pdf"],
    byExtension: {
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".pdf": "application/pdf",
    },
  },
};

/** `accept` attribute values for file inputs. */
export const ACCEPT: Record<MediaKind, string> = {
  video: [...TYPE_RULES.video.extensions, ...TYPE_RULES.video.mimeTypes].join(","),
  photo: [...TYPE_RULES.photo.extensions, ...TYPE_RULES.photo.mimeTypes].join(","),
  floor_plan: [...TYPE_RULES.floor_plan.extensions, ...TYPE_RULES.floor_plan.mimeTypes].join(","),
};

export interface CandidateFile {
  name: string;
  size: number;
  type: string;
  mediaKind: MediaKind;
  /** Video duration in seconds when the browser could read it. */
  durationSeconds?: number | null;
}

export interface IntakeIssue {
  code:
    | "NO_CAPTURE_MEDIA"
    | "TOO_MANY_VIDEOS"
    | "TOO_MANY_PHOTOS"
    | "TOO_MANY_FLOOR_PLANS"
    | "UNSUPPORTED_TYPE"
    | "EMPTY_FILE"
    | "FLOOR_PLAN_TOO_LARGE"
    | "TOTAL_TOO_LARGE"
    | "VIDEO_TOO_LONG"
    | "NO_VIDEO";
  message: string;
  fileName?: string;
}

export interface IntakeResult {
  ok: boolean;
  errors: IntakeIssue[];
  warnings: IntakeIssue[];
  totalBytes: number;
}

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
}

/** Accepts when either the extension or the reported MIME type is allowed for the kind. */
export function isAcceptedType(
  file: Pick<CandidateFile, "name" | "type">,
  kind: MediaKind,
): boolean {
  const rule = TYPE_RULES[kind];
  const mime = file.type.toLowerCase();
  return rule.extensions.includes(extensionOf(file.name)) || rule.mimeTypes.includes(mime);
}

/** The content type to declare to the API (a hint; the server verifies). */
export function declaredContentType(
  file: Pick<CandidateFile, "name" | "type">,
  kind: MediaKind,
): string {
  const rule = TYPE_RULES[kind];
  const mime = file.type.toLowerCase();
  if (rule.mimeTypes.includes(mime)) return mime;
  return rule.byExtension[extensionOf(file.name)] ?? "application/octet-stream";
}

export function formatBytes(bytes: number): string {
  if (bytes >= GiB) return `${(bytes / GiB).toFixed(2)} GiB`;
  if (bytes >= MiB) return `${(bytes / MiB).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KiB`;
  return `${bytes} B`;
}

export function validateIntake(files: readonly CandidateFile[]): IntakeResult {
  const errors: IntakeIssue[] = [];
  const warnings: IntakeIssue[] = [];
  const videos = files.filter((f) => f.mediaKind === "video");
  const photos = files.filter((f) => f.mediaKind === "photo");
  const plans = files.filter((f) => f.mediaKind === "floor_plan");
  const totalBytes = files.reduce((sum, f) => sum + f.size, 0);

  if (videos.length === 0 && photos.length === 0) {
    errors.push({
      code: "NO_CAPTURE_MEDIA",
      message: "Add a walkthrough video or overlapping photos. A floor plan alone is not enough.",
    });
  } else if (videos.length === 0) {
    warnings.push({
      code: "NO_VIDEO",
      message:
        "A slow continuous video is the recommended input. Photos alone often fail to connect rooms.",
    });
  }
  if (videos.length > INTAKE_LIMITS.maxVideos) {
    errors.push({ code: "TOO_MANY_VIDEOS", message: "Use one continuous walkthrough video." });
  }
  if (photos.length > INTAKE_LIMITS.maxPhotos) {
    errors.push({
      code: "TOO_MANY_PHOTOS",
      message: `Use at most ${INTAKE_LIMITS.maxPhotos} photos (you selected ${photos.length}).`,
    });
  }
  if (plans.length > 1) {
    errors.push({ code: "TOO_MANY_FLOOR_PLANS", message: "Use at most one floor plan." });
  }

  for (const file of files) {
    if (file.size <= 0) {
      errors.push({ code: "EMPTY_FILE", message: `${file.name} is empty.`, fileName: file.name });
      continue;
    }
    if (!isAcceptedType(file, file.mediaKind)) {
      const expected =
        file.mediaKind === "video"
          ? "MP4 or MOV"
          : file.mediaKind === "photo"
            ? "JPEG or PNG"
            : "PNG, JPEG or PDF";
      errors.push({
        code: "UNSUPPORTED_TYPE",
        message: `${file.name} doesn't look like ${expected}.`,
        fileName: file.name,
      });
    }
    if (file.mediaKind === "floor_plan" && file.size > INTAKE_LIMITS.maxFloorPlanBytes) {
      errors.push({
        code: "FLOOR_PLAN_TOO_LARGE",
        message: `${file.name} is ${formatBytes(file.size)}; floor plans can be up to 20 MiB.`,
        fileName: file.name,
      });
    }
    if (
      file.mediaKind === "video" &&
      file.durationSeconds != null &&
      file.durationSeconds > INTAKE_LIMITS.maxVideoSeconds
    ) {
      warnings.push({
        code: "VIDEO_TOO_LONG",
        message: `${file.name} is about ${Math.round(file.durationSeconds / 60)} minutes. Videos up to 5 minutes are supported in the beta; longer ones may be rejected.`,
        fileName: file.name,
      });
    }
  }

  if (totalBytes > INTAKE_LIMITS.maxTotalBytes) {
    errors.push({
      code: "TOTAL_TOO_LARGE",
      message: `The selection is ${formatBytes(totalBytes)}; the limit is 2 GiB in total.`,
    });
  }

  return { ok: errors.length === 0, errors, warnings, totalBytes };
}
