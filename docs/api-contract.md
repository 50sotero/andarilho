# Andarilho HTTP API contract (v1)

This document pins the JSON shapes of the `/v1` API (README section 12) so the
FastAPI service and the Next.js client agree. FastAPI also serves the live
OpenAPI document at `/openapi.json`.

## Conventions

- JSON bodies use **camelCase**. Resource IDs are UUID strings. Timestamps are
  ISO-8601 UTC strings.
- Authentication: `Authorization: Bearer <Supabase access token>`; the API
  verifies the JWT against the configured JWKS and checks project membership.
- A resource that does not exist and a resource the caller cannot access both
  return `404 NOT_FOUND`.
- Versioned resources return `ETag: "<version>"`. Mutations that require
  optimistic concurrency need `If-Match: "<version>"`: a missing header returns
  `428 PRECONDITION_REQUIRED`; a stale one returns `409 VERSION_CONFLICT`.
- Asynchronous work returns `202 Accepted`.
- Errors always use this shape:

```json
{
  "error": {
    "code": "VERSION_CONFLICT",
    "message": "The scene changed since you loaded it.",
    "retryable": false,
    "suggestedAction": "Reload the scene and apply your edit again.",
    "requestId": "8a6e0c..."
  }
}
```

Common codes: `UNAUTHENTICATED` (401), `NOT_FOUND` (404), `FORBIDDEN` (403, only
for an authenticated member whose role is insufficient), `VALIDATION_FAILED`
(422), `VERSION_CONFLICT` / `ACTIVE_RUN_EXISTS` / `IDEMPOTENCY_KEY_REUSED` /
`PROJECT_DELETING` / `INVALID_STATE` (409), `PRECONDITION_REQUIRED` (428),
`PAYLOAD_TOO_LARGE` / `QUOTA_EXCEEDED` (413), `RATE_LIMITED` (429, with
`Retry-After`). Optional AI advice adds `AI_UNAVAILABLE` (503; not retryable when
AI is not configured), `AI_RATE_LIMITED` (429, with `Retry-After` when known) and
`AI_INVALID_OUTPUT` (502). Existing-tour imports add `IMPORT_UNSUPPORTED` (422) and
`IMPORT_SOURCE_UNAVAILABLE` (502).

## Shapes

```ts
type Role = "owner" | "editor" | "viewer";

interface Project {
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

interface Upload {
  id: string;
  projectId: string;
  mediaKind: "video" | "photo" | "floor_plan";
  filename: string; // display metadata only
  contentType: string;
  expectedBytes: number;
  status: "initiated" | "uploaded_unverified" | "verified" | "rejected" | "aborted" | "expired";
  partSize: number;
  partCount: number;
  rejectionCode: string | null;
  metadata: Record<string, unknown>; // duration, width, height, codec... after inspection
  expiresAt: string;
  createdAt: string;
}

interface CaptureSet {
  id: string;
  projectId: string;
  inputs: { uploadId: string; mediaKind: string; sha256: string; bytes: number }[];
  floorPlanUploadId: string | null;
  createdAt: string;
}

// An existing provider tour shown as the provider's embed (README section 15); the
// client builds the iframe from modelId, never from provider HTML.
interface TourImport {
  id: string;
  projectId: string;
  kind: "matterport_embed";
  provider: "matterport";
  modelId: string; // 11 alphanumerics
  canonicalUrl: string; // https://my.matterport.com/show/?m=<modelId>
  listingUrl: string | null; // the listing it was found on (attribution)
  createdAt: string;
}

interface JobSummary {
  id: string;
  stage: string;
  state: "queued" | "running" | "retry_wait" | "succeeded" | "failed" | "cancelled";
}

interface Run {
  id: string;
  captureSetId: string;
  quality: "preview" | "standard" | "synthetic";
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  cancelRequested: boolean;
  error: ApiErrorBody["error"] | null;
  currentJobId: string | null;
  currentStage: string | null;
  jobs: JobSummary[];
  createdAt: string;
  finishedAt: string | null;
}

interface SceneRevision {
  id: string;
  revision: number;
  layoutRevision: number;
  layout: Layout; // packages/scene-contract
  manifest: SceneManifest | null;
  validationState: "pending" | "valid" | "invalid";
  validationFindings: { code: string; message: string }[];
  approvalState: "draft" | "approved";
  approvedAt: string | null;
  createdAt: string;
}

interface Share {
  id: string;
  sceneRevisionId: string;
  expiresAt: string | null;
  revokedAt: string | null;
  createdAt: string;
}
```

## Routes

| Route                                    | Request                                                                                                            | Response                                                                                                                                                                                                                                                                                                                                             |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /v1/projects`                      | `{title}`                                                                                                          | `201 Project` + `ETag`                                                                                                                                                                                                                                                                                                                               |
| `GET /v1/projects?cursor=&limit=`        |                                                                                                                    | `{items: Project[], nextCursor: string \| null}`                                                                                                                                                                                                                                                                                                     |
| `GET /v1/projects/{id}`                  |                                                                                                                    | `{project, uploads: Upload[], captureSets: CaptureSet[], runs: Run[], scene: {id, version, currentDraftRevisionId} \| null, shares: Share[]}` + `ETag`                                                                                                                                                                                               |
| `PATCH /v1/projects/{id}`                | `If-Match`, `{title}`                                                                                              | `Project` + `ETag`                                                                                                                                                                                                                                                                                                                                   |
| `DELETE /v1/projects/{id}`               |                                                                                                                    | `202 {projectId, lifecycle: "deleting"}`                                                                                                                                                                                                                                                                                                             |
| `POST /v1/projects/{id}/uploads`         | `{filename, bytes, mediaKind, contentType}`                                                                        | `201 {upload, partSize, partCount}`                                                                                                                                                                                                                                                                                                                  |
| `GET /v1/uploads/{id}`                   |                                                                                                                    | `{upload, completedParts: {partNumber, etag, size}[]}`                                                                                                                                                                                                                                                                                               |
| `POST /v1/uploads/{id}/parts`            | `{partNumbers: number[]}` (1-20 numbers)                                                                           | `{urls: {partNumber, url}[], expiresAt}`                                                                                                                                                                                                                                                                                                             |
| `POST /v1/uploads/{id}/complete`         | `{}` (the API lists parts itself)                                                                                  | `202 {upload}`                                                                                                                                                                                                                                                                                                                                       |
| `DELETE /v1/uploads/{id}`                |                                                                                                                    | `204`; `409 INVALID_STATE` while the upload is being checked or when an active or succeeded run uses it                                                                                                                                                                                                                                              |
| `POST /v1/projects/{id}/capture-sets`    | `{uploadIds: string[], floorPlanUploadId?: string, metadata?: object}`                                             | `201 CaptureSet`                                                                                                                                                                                                                                                                                                                                     |
| `POST /v1/projects/{id}/runs`            | `Idempotency-Key`, `{captureSetId, quality, includeSupplementaryPhotos}`                                           | `202 {runId, jobId, state, stage}`                                                                                                                                                                                                                                                                                                                   |
| `GET /v1/jobs/{id}?after=&limit=`        |                                                                                                                    | `{job: {id, runId, stage, state, attempts, error, createdAt, updatedAt}, events: {id, stage, state, message, progress, createdAt}[], nextCursor: number \| null}`                                                                                                                                                                                    |
| `POST /v1/runs/{id}/cancel`              |                                                                                                                    | `202 {runId, status, cancelRequested: true}`                                                                                                                                                                                                                                                                                                         |
| `POST /v1/runs/{id}/retry`               | `Idempotency-Key`, `{}`                                                                                            | `202 {runId, jobId, state, stage}`                                                                                                                                                                                                                                                                                                                   |
| `POST /v1/runs/{id}/advice`              | `{}` (owner/editor; failed runs only; optional AI, README section 16)                                              | `{runId, provider, model, advice: {summary, steps: string[]}, sent}`; sends only the error code, its message, the failed stage and capture statistics                                                                                                                                                                                                |
| `POST /v1/projects/{id}/imports/resolve` | `{url}` (owner/editor): a Matterport Showcase link or an allowlisted HomeQ listing page (fetched under SSRF rules) | `{source, listingUrl, tours: {provider, modelId, canonicalUrl, modes}[], notes}`; `422 IMPORT_UNSUPPORTED`, `502 IMPORT_SOURCE_UNAVAILABLE`                                                                                                                                                                                                          |
| `POST /v1/projects/{id}/imports`         | `{kind: "matterport_embed", modelId, listingUrl?, permissionConfirmed: true}`                                      | `201 TourImport`; `409 INVALID_STATE` if the tour is already attached                                                                                                                                                                                                                                                                                |
| `GET /v1/projects/{id}/imports`          |                                                                                                                    | `{items: TourImport[]}`                                                                                                                                                                                                                                                                                                                              |
| `DELETE /v1/imports/{id}`                | (owner/editor)                                                                                                     | `204`                                                                                                                                                                                                                                                                                                                                                |
| `GET /v1/scenes/{id}`                    |                                                                                                                    | `{scene: {id, projectId, runId, version, currentDraftRevisionId}, revision: SceneRevision \| null, revisions: {id, revision, layoutRevision, validationState, approvalState, createdAt}[], floorPlan: {uploadId, contentType, url, expiresAt} \| null}` + `ETag` (owners/editors only; viewers get `403`. `floorPlan` is a short-lived private link) |
| `PUT /v1/scenes/{id}/layout`             | `If-Match`, `{baseRevisionId, layout}`                                                                             | `202 {sceneRevisionId, jobId, version}` + `ETag`                                                                                                                                                                                                                                                                                                     |
| `POST /v1/scenes/{id}/approve`           | `If-Match`, `{sceneRevisionId}`                                                                                    | `{sceneRevisionId, approvalState: "approved", version}`                                                                                                                                                                                                                                                                                              |
| `POST /v1/scenes/{id}/delivery`          | `{sceneRevisionId}`                                                                                                | `DeliveryEnvelope`                                                                                                                                                                                                                                                                                                                                   |
| `POST /v1/scenes/{id}/shares`            | `If-Match`, `{sceneRevisionId, expiresAt?}`                                                                        | `201 {share, token, path: "/s/<token>"}` (token shown once)                                                                                                                                                                                                                                                                                          |
| `DELETE /v1/shares/{id}`                 |                                                                                                                    | `204`                                                                                                                                                                                                                                                                                                                                                |
| `POST /v1/public/scenes/delivery`        | `{token}` (no auth)                                                                                                | derivative-only `DeliveryEnvelope`                                                                                                                                                                                                                                                                                                                   |
| `GET /healthz` / `GET /readyz`           |                                                                                                                    | liveness / readiness (DB + storage)                                                                                                                                                                                                                                                                                                                  |

Viewers may request delivery only for approved revisions (the published one); drafts
are limited to owners and editors.

`DeliveryEnvelope` and `SceneManifest` are defined in
`packages/scene-contract/scene.ts` (generated). Delivery URLs expire after
`DELIVERY_URL_TTL_SECONDS` (default 300 s); fetch a fresh envelope for the same
`sceneRevisionId` when they expire.
