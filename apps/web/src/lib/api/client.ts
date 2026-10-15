import { ApiError, apiErrorFromResponse } from "./errors";
import type {
  CaptureSet,
  CreateUploadRequest,
  CreateUploadResponse,
  DeliveryEnvelope,
  JobDetail,
  Layout,
  LayoutSaved,
  PartUrlsResponse,
  Project,
  ProjectDetail,
  ProjectList,
  ImportResolution,
  IntakePolicy,
  RunAccepted,
  RunAdvice,
  TourImport,
  RunQuality,
  SceneDetail,
  ShareCreated,
  Upload,
  UploadStatusResponse,
  Versioned,
} from "./types";

/** Every browser request goes through the same-origin allowlisted proxy. */
export const API_BASE = "/api/backend";

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  ifMatch?: string | null;
  idempotencyKey?: string;
  signal?: AbortSignal;
  query?: Record<string, string | number | null | undefined>;
}

function buildPath(path: string, query?: RequestOptions["query"]): string {
  const url = `${API_BASE}${path}`;
  if (!query) return url;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
  }
  const qs = params.toString();
  return qs ? `${url}?${qs}` : url;
}

/** Generate a fresh Idempotency-Key. Call once per user submit intent and reuse on retry. */
export function newIdempotencyKey(): string {
  return crypto.randomUUID();
}

export function createApiClient(fetchImpl: FetchLike = (input, init) => fetch(input, init)) {
  async function request<T>(path: string, options: RequestOptions = {}): Promise<Versioned<T>> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (options.body !== undefined) headers["Content-Type"] = "application/json";
    if (options.ifMatch) headers["If-Match"] = options.ifMatch;
    if (options.idempotencyKey) headers["Idempotency-Key"] = options.idempotencyKey;

    const response = await fetchImpl(buildPath(path, options.query), {
      method: options.method ?? "GET",
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      credentials: "same-origin",
      cache: "no-store",
      signal: options.signal,
    });

    if (!response.ok) throw await apiErrorFromResponse(response);

    const etag = response.headers.get("ETag");
    if (response.status === 204) return { data: undefined as T, etag };
    const text = await response.text();
    if (!text) return { data: undefined as T, etag };
    try {
      return { data: JSON.parse(text) as T, etag };
    } catch {
      throw new ApiError(response.status, {
        code: "INVALID_RESPONSE",
        message: "The server returned an unreadable response.",
        retryable: true,
        requestId: response.headers.get("X-Request-ID"),
      });
    }
  }

  const id = (value: string) => encodeURIComponent(value);

  return {
    // Projects
    listProjects: (cursor?: string | null, limit = 20, signal?: AbortSignal) =>
      request<ProjectList>("/v1/projects", { query: { cursor, limit }, signal }),
    createProject: (title: string) =>
      request<Project>("/v1/projects", { method: "POST", body: { title } }),
    getProject: (projectId: string, signal?: AbortSignal) =>
      request<ProjectDetail>(`/v1/projects/${id(projectId)}`, { signal }),
    renameProject: (projectId: string, title: string, ifMatch: string) =>
      request<Project>(`/v1/projects/${id(projectId)}`, {
        method: "PATCH",
        body: { title },
        ifMatch,
      }),
    deleteProject: (projectId: string) =>
      request<{ projectId: string; lifecycle: "deleting" }>(`/v1/projects/${id(projectId)}`, {
        method: "DELETE",
      }),

    // Uploads
    createUpload: (projectId: string, body: CreateUploadRequest) =>
      request<CreateUploadResponse>(`/v1/projects/${id(projectId)}/uploads`, {
        method: "POST",
        body,
      }),
    getUpload: (uploadId: string, signal?: AbortSignal) =>
      request<UploadStatusResponse>(`/v1/uploads/${id(uploadId)}`, { signal }),
    requestPartUrls: (uploadId: string, partNumbers: number[], signal?: AbortSignal) =>
      request<PartUrlsResponse>(`/v1/uploads/${id(uploadId)}/parts`, {
        method: "POST",
        body: { partNumbers },
        signal,
      }),
    completeUpload: (uploadId: string) =>
      request<{ upload: Upload }>(`/v1/uploads/${id(uploadId)}/complete`, {
        method: "POST",
        body: {},
      }),
    abortUpload: (uploadId: string) =>
      request<void>(`/v1/uploads/${id(uploadId)}`, { method: "DELETE" }),

    // Capture sets and runs
    createCaptureSet: (
      projectId: string,
      body: { uploadIds: string[]; floorPlanUploadId?: string; metadata?: Record<string, unknown> },
    ) =>
      request<CaptureSet>(`/v1/projects/${id(projectId)}/capture-sets`, {
        method: "POST",
        body,
      }),
    submitRun: (
      projectId: string,
      body: { captureSetId: string; quality: RunQuality; includeSupplementaryPhotos: boolean },
      idempotencyKey: string,
    ) =>
      request<RunAccepted>(`/v1/projects/${id(projectId)}/runs`, {
        method: "POST",
        body,
        idempotencyKey,
      }),
    getJob: (jobId: string, after?: number | null, limit = 100, signal?: AbortSignal) =>
      request<JobDetail>(`/v1/jobs/${id(jobId)}`, { query: { after, limit }, signal }),
    cancelRun: (runId: string) =>
      request<{ runId: string; status: string; cancelRequested: true }>(
        `/v1/runs/${id(runId)}/cancel`,
        { method: "POST", body: {} },
      ),
    // Existing tours
    resolveImport: (projectId: string, url: string) =>
      request<ImportResolution>(`/v1/projects/${id(projectId)}/imports/resolve`, {
        method: "POST",
        body: { url },
      }),
    createImport: (projectId: string, modelId: string, listingUrl: string | null) =>
      request<TourImport>(`/v1/projects/${id(projectId)}/imports`, {
        method: "POST",
        body: { kind: "matterport_embed", modelId, listingUrl, permissionConfirmed: true },
      }),
    listImports: (projectId: string) =>
      request<{ items: TourImport[] }>(`/v1/projects/${id(projectId)}/imports`),
    deleteImport: (importId: string) =>
      request<void>(`/v1/imports/${id(importId)}`, { method: "DELETE" }),

    getIntakePolicy: () => request<IntakePolicy>("/v1/intake-policy"),

    getRunAdvice: (runId: string) =>
      request<RunAdvice>(`/v1/runs/${id(runId)}/advice`, { method: "POST", body: {} }),
    retryRun: (runId: string, idempotencyKey: string) =>
      request<RunAccepted>(`/v1/runs/${id(runId)}/retry`, {
        method: "POST",
        body: {},
        idempotencyKey,
      }),

    // Scenes
    getScene: (sceneId: string, signal?: AbortSignal) =>
      request<SceneDetail>(`/v1/scenes/${id(sceneId)}`, { signal }),
    saveLayout: (sceneId: string, baseRevisionId: string, layout: Layout, ifMatch: string) =>
      request<LayoutSaved>(`/v1/scenes/${id(sceneId)}/layout`, {
        method: "PUT",
        body: { baseRevisionId, layout },
        ifMatch,
      }),
    approveRevision: (sceneId: string, sceneRevisionId: string, ifMatch: string) =>
      request<{ sceneRevisionId: string; approvalState: "approved"; version: number }>(
        `/v1/scenes/${id(sceneId)}/approve`,
        { method: "POST", body: { sceneRevisionId }, ifMatch },
      ),
    getDelivery: (sceneId: string, sceneRevisionId: string) =>
      request<DeliveryEnvelope>(`/v1/scenes/${id(sceneId)}/delivery`, {
        method: "POST",
        body: { sceneRevisionId },
      }),

    // Sharing
    createShare: (
      sceneId: string,
      sceneRevisionId: string,
      ifMatch: string,
      expiresAt?: string | null,
    ) =>
      request<ShareCreated>(`/v1/scenes/${id(sceneId)}/shares`, {
        method: "POST",
        body: expiresAt ? { sceneRevisionId, expiresAt } : { sceneRevisionId },
        ifMatch,
      }),
    revokeShare: (shareId: string) =>
      request<void>(`/v1/shares/${id(shareId)}`, { method: "DELETE" }),
    getPublicDelivery: (token: string) =>
      request<DeliveryEnvelope>("/v1/public/scenes/delivery", {
        method: "POST",
        body: { token },
      }),
  };
}

export type ApiClient = ReturnType<typeof createApiClient>;

/** Default browser client. */
export const api: ApiClient = createApiClient();
