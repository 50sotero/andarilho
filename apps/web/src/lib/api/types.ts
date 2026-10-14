/**
 * JSON shapes of the Andarilho /v1 API (docs/api-contract.md).
 * Scene/layout/delivery types come from the generated scene contract.
 */
import type { DeliveryEnvelope, Layout, SceneManifest } from "@andarilho/scene-contract";

export type { DeliveryEnvelope, Layout, SceneManifest };

export type Role = "owner" | "editor" | "viewer";

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    retryable: boolean;
    suggestedAction?: string | null;
    requestId?: string | null;
  };
}

export interface Project {
  id: string;
  title: string;
  lifecycle: "active" | "deleting" | "deleted";
  version: number;
  role: Role;
  activeDraftRevisionId: string | null;
  publishedRevisionId: string | null;
  createdAt: string;
  updatedAt: string;
}

export type MediaKind = "video" | "photo" | "floor_plan";

export type UploadStatus =
  | "initiated"
  | "uploaded_unverified"
  | "verified"
  | "rejected"
  | "aborted"
  | "expired";

export interface Upload {
  id: string;
  projectId: string;
  mediaKind: MediaKind;
  filename: string;
  contentType: string;
  expectedBytes: number;
  status: UploadStatus;
  partSize: number;
  partCount: number;
  rejectionCode: string | null;
  metadata: Record<string, unknown>;
  expiresAt: string;
  createdAt: string;
}

export interface CaptureSet {
  id: string;
  projectId: string;
  inputs: { uploadId: string; mediaKind: string; sha256: string; bytes: number }[];
  floorPlanUploadId: string | null;
  createdAt: string;
}

export type JobState = "queued" | "running" | "retry_wait" | "succeeded" | "failed" | "cancelled";

export interface JobSummary {
  id: string;
  stage: string;
  state: JobState;
}

export type RunQuality = "preview" | "standard" | "synthetic";
export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface Run {
  id: string;
  captureSetId: string;
  quality: RunQuality;
  status: RunStatus;
  cancelRequested: boolean;
  error: ApiErrorBody["error"] | null;
  currentJobId: string | null;
  currentStage: string | null;
  jobs: JobSummary[];
  createdAt: string;
  finishedAt: string | null;
}

export interface SceneRevision {
  id: string;
  revision: number;
  layoutRevision: number;
  layout: Layout;
  manifest: SceneManifest | null;
  validationState: "pending" | "valid" | "invalid";
  validationFindings: { code: string; message: string }[];
  approvalState: "draft" | "approved";
  approvedAt: string | null;
  createdAt: string;
}

export interface Share {
  id: string;
  sceneRevisionId: string;
  expiresAt: string | null;
  revokedAt: string | null;
  createdAt: string;
}

export interface ProjectList {
  items: Project[];
  nextCursor: string | null;
}

export interface ProjectDetail {
  project: Project;
  uploads: Upload[];
  captureSets: CaptureSet[];
  runs: Run[];
  scene: { id: string; version: number; currentDraftRevisionId: string | null } | null;
  shares: Share[];
}

export interface CreateUploadRequest {
  filename: string;
  bytes: number;
  mediaKind: MediaKind;
  contentType: string;
}

export interface CreateUploadResponse {
  upload: Upload;
  partSize: number;
  partCount: number;
}

export interface CompletedPart {
  partNumber: number;
  etag: string;
  size?: number;
}

export interface UploadStatusResponse {
  upload: Upload;
  completedParts: { partNumber: number; etag: string; size: number }[];
}

export interface PartUrlsResponse {
  urls: { partNumber: number; url: string }[];
  expiresAt: string;
}

export interface RunAccepted {
  runId: string;
  jobId: string;
  state: JobState;
  stage: string;
}

export interface JobEvent {
  id: number;
  stage: string;
  state: string;
  message: string | null;
  progress: Record<string, unknown> | null;
  createdAt: string;
}

export interface JobDetail {
  job: {
    id: string;
    runId: string;
    stage: string;
    state: JobState;
    attempts: number;
    error: ApiErrorBody["error"] | null;
    createdAt: string;
    updatedAt: string;
  };
  events: JobEvent[];
  nextCursor: number | null;
}

export interface SceneDetail {
  scene: {
    id: string;
    projectId: string;
    runId: string;
    version: number;
    currentDraftRevisionId: string | null;
  };
  revision: SceneRevision | null;
  revisions: {
    id: string;
    revision: number;
    layoutRevision: number;
    validationState: SceneRevision["validationState"];
    approvalState: SceneRevision["approvalState"];
    createdAt: string;
  }[];
}

export interface LayoutSaved {
  sceneRevisionId: string;
  jobId: string;
  version: number;
}

export interface ShareCreated {
  share: Share;
  token: string;
  path: string;
}

/** A response body plus the (quoted) ETag header when present. */
export interface Versioned<T> {
  data: T;
  etag: string | null;
}
