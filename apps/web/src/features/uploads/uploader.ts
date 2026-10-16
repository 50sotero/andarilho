/**
 * Resumable multipart uploader (README section 7, steps 1-4).
 *
 * - Initializes an upload through the API (which chooses the key and part size).
 * - Requests short-lived part URLs in small batches.
 * - PUTs parts directly to storage with bounded concurrency (3 by default) and
 *   reads the ETag response header (storage CORS must expose it).
 * - Retries failed parts with exponential backoff and jitter.
 * - Persists upload id, fingerprint, content sample, completed parts/ETags and
 *   expiry locally, so the same file can resume after a reload once the user
 *   re-selects it. A file whose sampled bytes differ starts a new upload.
 * - Supports pause (abort in-flight requests) and resume (call start() again).
 */
import { ApiError } from "@/lib/api/errors";
import type { ApiClient } from "@/lib/api/client";
import type { MediaKind, Upload } from "@/lib/api/types";
import {
  RESUME_RECORD_VERSION,
  clearCreationKey,
  contentSample,
  creationIdentity,
  creationKey,
  fileFingerprint,
  getDefaultStorage,
  isExpired,
  loadResumeRecord,
  partRange,
  remainingParts,
  removeResumeRecord,
  saveResumeRecord,
  withCompletedPart,
  type CompletedPartRecord,
  type FileIdentity,
  type KeyValueStorage,
  type UploadResumeRecord,
  uploadIdentity,
} from "@/lib/upload-resume";

export const DEFAULT_CONCURRENCY = 3;
export const DEFAULT_URL_BATCH = 10; // API accepts 1-20 part numbers per request
export const DEFAULT_MAX_ATTEMPTS = 5;
export const DEFAULT_PART_SIZE = 16 * 1024 * 1024;
/** Refresh a signed part URL when it expires within this margin. */
const URL_EXPIRY_MARGIN_MS = 30_000;

export interface UploadableFile {
  name: string;
  size: number;
  lastModified: number;
  type: string;
  slice(start: number, end: number): Blob;
}

export type UploaderApi = Pick<
  ApiClient,
  "createUpload" | "getUpload" | "requestPartUrls" | "completeUpload" | "abortUpload"
>;

export class StoragePutError extends Error {
  readonly status: number;
  readonly retryable: boolean;
  constructor(status: number, message: string, retryable: boolean) {
    super(message);
    this.name = "StoragePutError";
    this.status = status;
    this.retryable = retryable;
  }
}

export class UploadAbortedError extends Error {
  constructor() {
    super("Upload paused.");
    this.name = "UploadAbortedError";
  }
}

export type PutPart = (url: string, body: Blob, signal: AbortSignal) => Promise<string>;

/** Default storage PUT. Never sends cookies or app credentials to storage. */
export const fetchPutPart: PutPart = async (url, body, signal) => {
  const response = await fetch(url, {
    method: "PUT",
    body,
    signal,
    credentials: "omit",
    cache: "no-store",
  });
  if (!response.ok) {
    const retryable = response.status >= 500 || response.status === 408 || response.status === 429;
    // 403 typically means the signed URL expired: refresh it and retry.
    throw new StoragePutError(
      response.status,
      `Storage rejected part (HTTP ${response.status}).`,
      retryable || response.status === 403,
    );
  }
  const etag = response.headers.get("ETag");
  if (!etag) {
    throw new StoragePutError(
      0,
      "Storage did not expose the ETag header. Check the bucket CORS configuration.",
      false,
    );
  }
  return etag;
};

export type UploadPhase =
  "idle" | "initializing" | "uploading" | "paused" | "completing" | "completed" | "failed";

export interface UploadProgress {
  phase: UploadPhase;
  uploadedBytes: number;
  totalBytes: number;
  completedParts: number;
  partCount: number;
  resumed: boolean;
  uploadId: string | null;
  error: unknown;
}

export interface UploaderOptions {
  api: UploaderApi;
  putPart?: PutPart;
  storage?: KeyValueStorage | null;
  concurrency?: number;
  urlBatchSize?: number;
  maxAttempts?: number;
  baseRetryDelayMs?: number;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  random?: () => number;
  now?: () => number;
  /** Creation keys held by this page's uploaders; a page (or a reload) starts empty. */
  heldCreationKeys?: Set<string>;
}

function abortableSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new UploadAbortedError());
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new UploadAbortedError());
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function newCreationKey(): string {
  return crypto.randomUUID();
}

function isRetryable(error: unknown): boolean {
  if (error instanceof StoragePutError) return error.retryable;
  if (error instanceof ApiError) return error.retryable || error.status >= 500;
  // Network failures surface as TypeError from fetch.
  return error instanceof TypeError;
}

/**
 * Creation keys held by uploaders on this page. The same file selected twice has one
 * creation identity: a second uploader gets a key of its own, or the API would answer
 * both with one upload and both would upload into it.
 */
const pageCreationKeys = new Set<string>();

/** Server states meaning the bytes were already completed and frozen. */
const COMPLETED_STATUSES = new Set<Upload["status"]>(["uploaded_unverified", "verified"]);

export class MultipartUpload {
  readonly projectId: string;
  readonly file: UploadableFile;
  readonly mediaKind: MediaKind;
  readonly contentType: string;
  /** The file sent as this media kind and content type: what finds its saved state. */
  readonly identity: FileIdentity;
  readonly fingerprint: string;

  private readonly api: UploaderApi;
  private readonly putPart: PutPart;
  private readonly storage: KeyValueStorage | null;
  private readonly concurrency: number;
  private readonly urlBatchSize: number;
  private readonly maxAttempts: number;
  private readonly baseRetryDelayMs: number;
  private readonly sleep: (ms: number, signal: AbortSignal) => Promise<void>;
  private readonly random: () => number;
  private readonly now: () => number;
  private readonly heldCreationKeys: Set<string>;

  private record: UploadResumeRecord | null = null;
  private controller: AbortController | null = null;
  private running: Promise<Upload> | null = null;
  private listeners = new Set<(p: UploadProgress) => void>();
  private progress: UploadProgress;
  // Signed part URLs belong to one multipart upload: urlCacheUploadId names it.
  private urlCache = new Map<number, { url: string; expiresAt: number }>();
  private pendingUrls = new Map<number, Promise<void>>();
  private urlCacheUploadId: string | null = null;
  // Idempotency-Key of a creation that has not had a definite answer yet.
  private creationKey: string | null = null;
  // The key this uploader holds (pageCreationKeys), until it is done with its upload.
  private heldKey: string | null = null;
  // Whether that key is the one saved for this file, which only its holder may clear.
  private heldKeySaved = false;
  // Fingerprint and content sample: what the saved creation key is found by.
  private creationIdentity: string | null = null;
  // Set by discard(): an upload created after it is aborted (when asked) and never kept.
  private discarded: { abortOnServer: boolean } | null = null;
  // An upload whose parts storage already joined: it has no part left to send.
  private assembledUploadId: string | null = null;

  constructor(
    projectId: string,
    file: UploadableFile,
    mediaKind: MediaKind,
    contentType: string,
    options: UploaderOptions,
  ) {
    this.projectId = projectId;
    this.file = file;
    this.mediaKind = mediaKind;
    this.contentType = contentType;
    this.identity = uploadIdentity(file, mediaKind, contentType);
    this.fingerprint = fileFingerprint(this.identity);
    this.api = options.api;
    this.putPart = options.putPart ?? fetchPutPart;
    this.storage = options.storage === undefined ? getDefaultStorage() : options.storage;
    this.concurrency = Math.max(1, options.concurrency ?? DEFAULT_CONCURRENCY);
    this.urlBatchSize = Math.min(20, Math.max(1, options.urlBatchSize ?? DEFAULT_URL_BATCH));
    this.maxAttempts = Math.max(1, options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS);
    this.baseRetryDelayMs = options.baseRetryDelayMs ?? 1000;
    this.sleep = options.sleep ?? abortableSleep;
    this.random = options.random ?? Math.random;
    this.now = options.now ?? Date.now;
    this.heldCreationKeys = options.heldCreationKeys ?? pageCreationKeys;
    this.progress = {
      phase: "idle",
      uploadedBytes: 0,
      totalBytes: file.size,
      completedParts: 0,
      partCount: 0,
      resumed: false,
      uploadId: null,
      error: null,
    };
  }

  get state(): UploadProgress {
    return this.progress;
  }

  subscribe(listener: (p: UploadProgress) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(patch: Partial<UploadProgress>) {
    this.progress = { ...this.progress, ...patch };
    for (const listener of this.listeners) listener(this.progress);
  }

  /** Starts or resumes. Resolves with the completed upload (status uploaded_unverified). */
  start(): Promise<Upload> {
    if (this.running) return this.running;
    if (this.discarded) return Promise.reject(new UploadAbortedError());
    const controller = new AbortController();
    this.controller = controller;
    this.running = this.run(controller.signal).finally(() => {
      this.running = null;
      if (this.controller === controller) this.controller = null;
    });
    return this.running;
  }

  /** Aborts in-flight requests. Completed parts are kept; call start() to resume. */
  pause(): void {
    this.controller?.abort();
  }

  /**
   * Stops for good and forgets the upload, for a file the user removed. With
   * `abortOnServer` the upload is aborted best-effort (expiry and cleanup reconcile a
   * failure), including one whose creation is still awaiting its answer: it is aborted
   * once its ID arrives, so it does not keep holding the project's quota.
   */
  discard(options: { abortOnServer: boolean }): void {
    this.discarded = options;
    this.pause();
    const uploadId = this.record?.uploadId ?? this.progress.uploadId;
    this.record = null;
    this.resetUrlCache(null);
    removeResumeRecord(this.projectId, this.fingerprint, this.storage);
    this.forgetCreationKey();
    if (options.abortOnServer && uploadId) this.abortQuietly(uploadId);
  }

  private abortQuietly(uploadId: string) {
    this.api.abortUpload(uploadId).catch(() => undefined);
  }

  private async run(signal: AbortSignal): Promise<Upload> {
    try {
      this.emit({ phase: "initializing", error: null });
      const initialized = await this.initialize(signal);
      if ("completedUpload" in initialized) {
        // The server already completed this upload (e.g. the completion response was
        // lost before a reload): treat it as done instead of uploading a duplicate.
        this.forgetFinishedUpload();
        this.emit({ phase: "completed", uploadedBytes: this.file.size });
        return initialized.completedUpload;
      }
      const record = initialized;
      this.record = record;
      if (this.urlCacheUploadId !== record.uploadId) this.resetUrlCache(record.uploadId);
      // A pause while the upload was being created must stop it before any part.
      if (signal.aborted) throw new UploadAbortedError();
      this.emitPartProgress("uploading");

      const queue =
        this.assembledUploadId === record.uploadId
          ? []
          : remainingParts(record.partCount, record.completedParts);
      await this.uploadParts(queue, signal);

      if (signal.aborted) throw new UploadAbortedError();
      this.emit({ phase: "completing" });
      const { data } = await this.api.completeUpload(record.uploadId);
      this.forgetFinishedUpload();
      this.emit({ phase: "completed", uploadedBytes: this.file.size });
      return data.upload;
    } catch (error) {
      if (signal.aborted || error instanceof UploadAbortedError) {
        this.emit({ phase: "paused" });
        throw new UploadAbortedError();
      }
      if (
        error instanceof ApiError &&
        (error.code === "UPLOAD_EXPIRED" || error.status === 404 || error.status === 410)
      ) {
        // The server no longer accepts this upload: the next Resume starts a new one.
        this.discardRecord();
      }
      this.emit({ phase: "failed", error });
      throw error;
    }
  }

  private async initialize(
    signal: AbortSignal,
  ): Promise<UploadResumeRecord | { completedUpload: Upload }> {
    const inSession = this.record;
    let saved: UploadResumeRecord | null;
    if (inSession) {
      // Resuming within the same page session. Near its expiry the server decides: the
      // upload holds the project's quota until it lets the upload go, so a replacement
      // could be refused while this one still takes parts.
      if (!isExpired(inSession, this.now())) return inSession;
      saved = inSession;
    } else {
      saved = loadResumeRecord(this.projectId, this.identity, {
        storage: this.storage,
        now: this.now(),
      });
      // Name, size and date only find the record: another file can share all three, and
      // reusing its upload would keep the old bytes or mix the two files.
      if (saved && (await contentSample(this.file, saved.partSize)) !== saved.sample) {
        removeResumeRecord(this.projectId, this.fingerprint, this.storage);
        saved = null;
      }
    }
    if (signal.aborted) throw new UploadAbortedError();
    if (saved) {
      try {
        const { data } = await this.api.getUpload(saved.uploadId, signal);
        if (COMPLETED_STATUSES.has(data.upload.status)) {
          return { completedUpload: data.upload };
        }
        if (data.assembled) {
          // A completion joined the parts and then failed: completing again finishes it.
          this.assembledUploadId = saved.uploadId;
          this.emit({ resumed: true, uploadId: saved.uploadId });
          return saved;
        }
        // The server decides by its own clock: this device's may be off, and a replacement
        // could be refused while this upload still holds the project's quota.
        if (data.acceptingParts) {
          // The server's part listing is authoritative; local ETags fill gaps only.
          const serverParts: CompletedPartRecord[] = data.completedParts.map((p) => ({
            partNumber: p.partNumber,
            etag: p.etag,
          }));
          const record: UploadResumeRecord = {
            ...saved,
            completedParts: serverParts.sort((a, b) => a.partNumber - b.partNumber),
            expiresAt: data.upload.expiresAt ?? saved.expiresAt,
          };
          saveResumeRecord(record, this.storage);
          this.emit({ resumed: true, uploadId: record.uploadId });
          return record;
        }
      } catch (error) {
        if (signal.aborted) throw new UploadAbortedError();
        if (!(error instanceof ApiError) || error.status !== 404) {
          if (isRetryable(error)) throw error;
        }
      }
      if (inSession) this.discardRecord();
      else removeResumeRecord(this.projectId, this.fingerprint, this.storage);
    }

    // A lost response may hide an upload the server created: a retry with the same
    // key gets it back instead of creating (and charging quota for) another.
    if (this.creationKey === null) {
      this.creationIdentity ??= await creationIdentity(this.file, this.fingerprint);
      if (signal.aborted) throw new UploadAbortedError();
      const saved = creationKey(this.projectId, this.creationIdentity, {
        storage: this.storage,
        now: this.now(),
      });
      // Another uploader on this page holds it: the same file, selected again.
      const held = this.heldCreationKeys.has(saved);
      this.holdCreationKey(held ? newCreationKey() : saved, !held);
    }
    let data: Awaited<ReturnType<UploaderApi["createUpload"]>>["data"];
    try {
      ({ data } = await this.api.createUpload(
        this.projectId,
        {
          filename: this.file.name,
          bytes: this.file.size,
          mediaKind: this.mediaKind,
          contentType: this.contentType,
        },
        this.creationKey!,
      ));
    } catch (error) {
      if (!isRetryable(error)) this.forgetCreationKey();
      throw error;
    }
    // Known at once, before anything below can fail (the file may no longer be readable):
    // a file removed without a record still gets its upload aborted (discard).
    this.emit({ uploadId: data.upload.id, resumed: false });
    const record: UploadResumeRecord = {
      version: RESUME_RECORD_VERSION,
      uploadId: data.upload.id,
      projectId: this.projectId,
      fingerprint: this.fingerprint,
      // Empty where it cannot be computed: the record then stays in this session only.
      sample: (await contentSample(this.file, data.partSize)) ?? "",
      partSize: data.partSize,
      partCount: data.partCount,
      completedParts: [],
      expiresAt: data.upload.expiresAt,
    };
    if (this.discarded) {
      // The file was removed while the upload was being created: nothing may keep it.
      if (this.discarded.abortOnServer) this.abortQuietly(record.uploadId);
      this.forgetCreationKey();
      throw new UploadAbortedError();
    }
    // This session now holds the upload in its record. Until a saved record can find it
    // after a reload, the stored creation key does: sending it again gets this same upload
    // back rather than creating another that also takes quota.
    this.creationKey = null;
    if (saveResumeRecord(record, this.storage)) this.clearSavedCreationKey();
    return record;
  }

  private holdCreationKey(key: string, saved: boolean) {
    this.releaseCreationKey();
    this.heldCreationKeys.add(key);
    this.heldKey = key;
    this.heldKeySaved = saved;
    this.creationKey = key;
  }

  private releaseCreationKey() {
    if (this.heldKey !== null) this.heldCreationKeys.delete(this.heldKey);
    this.heldKey = null;
  }

  /** Clears the saved key, unless it belongs to another uploader of the same file. */
  private clearSavedCreationKey() {
    if (this.heldKey !== null && !this.heldKeySaved) return;
    clearCreationKey(this.projectId, this.creationIdentity ?? this.fingerprint, this.storage);
  }

  private forgetCreationKey() {
    this.creationKey = null;
    this.clearSavedCreationKey();
    this.releaseCreationKey();
  }

  private discardRecord() {
    this.record = null;
    this.resetUrlCache(null);
    removeResumeRecord(this.projectId, this.fingerprint, this.storage);
    // A creation key kept because the record could not be saved would get this upload back.
    this.forgetCreationKey();
  }

  /** The upload is complete: neither its record nor its creation key may find it again. */
  private forgetFinishedUpload() {
    this.record = null;
    removeResumeRecord(this.projectId, this.fingerprint, this.storage);
    this.forgetCreationKey();
  }

  /** Forget part URLs signed for another upload: they would upload into it. */
  private resetUrlCache(uploadId: string | null) {
    this.urlCache.clear();
    this.pendingUrls.clear();
    this.urlCacheUploadId = uploadId;
  }

  private emitPartProgress(phase?: UploadPhase) {
    const record = this.record;
    if (!record) return;
    const uploadedBytes = record.completedParts.reduce((sum, p) => {
      const [start, end] = partRange(p.partNumber, record.partSize, this.file.size);
      return sum + (end - start);
    }, 0);
    this.emit({
      ...(phase ? { phase } : {}),
      uploadedBytes,
      completedParts: record.completedParts.length,
      partCount: record.partCount,
    });
  }

  private async uploadParts(queue: number[], signal: AbortSignal): Promise<void> {
    const pending = [...queue];
    let failure: unknown = null;
    const inner = new AbortController();
    const onAbort = () => inner.abort();
    // An abort that already happened fires no event: honour it before any worker starts.
    if (signal.aborted) inner.abort();
    else signal.addEventListener("abort", onAbort, { once: true });

    const worker = async () => {
      while (pending.length > 0 && !failure && !inner.signal.aborted) {
        const partNumber = pending.shift()!;
        try {
          await this.uploadPartWithRetry(partNumber, pending, inner.signal);
        } catch (error) {
          if (!failure) failure = error;
          inner.abort();
        }
      }
    };

    try {
      await Promise.all(
        Array.from({ length: Math.min(this.concurrency, queue.length) }, () => worker()),
      );
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
    if (signal.aborted) throw new UploadAbortedError();
    if (failure) throw failure;
  }

  private async uploadPartWithRetry(
    partNumber: number,
    upcoming: readonly number[],
    signal: AbortSignal,
  ): Promise<void> {
    const record = this.record!;
    const [start, end] = partRange(partNumber, record.partSize, this.file.size);
    let attempt = 0;
    for (;;) {
      attempt += 1;
      try {
        const url = await this.urlFor(partNumber, upcoming, signal);
        const etag = await this.putPart(url, this.file.slice(start, end), signal);
        this.record = withCompletedPart(this.record!, { partNumber, etag });
        saveResumeRecord(this.record, this.storage);
        this.emitPartProgress();
        return;
      } catch (error) {
        if (signal.aborted) throw new UploadAbortedError();
        if (error instanceof StoragePutError && error.status === 403) {
          this.urlCache.delete(partNumber);
        }
        if (!isRetryable(error) || attempt >= this.maxAttempts) throw error;
        const backoff = this.baseRetryDelayMs * 2 ** (attempt - 1);
        const jitter = backoff * 0.5 * this.random();
        await this.sleep(Math.min(30_000, backoff + jitter), signal);
      }
    }
  }

  /** Returns a valid signed URL for the part, fetching a small batch when needed. */
  private async urlFor(
    partNumber: number,
    upcoming: readonly number[],
    signal: AbortSignal,
  ): Promise<string> {
    for (let i = 0; i < 3; i += 1) {
      const cached = this.urlCache.get(partNumber);
      if (cached && cached.expiresAt - URL_EXPIRY_MARGIN_MS > this.now()) return cached.url;
      const inFlight = this.pendingUrls.get(partNumber);
      if (inFlight) {
        await inFlight;
        continue;
      }
      const batch = [partNumber];
      for (const n of upcoming) {
        if (batch.length >= this.urlBatchSize) break;
        const c = this.urlCache.get(n);
        const fresh = c && c.expiresAt - URL_EXPIRY_MARGIN_MS > this.now();
        if (!fresh && !this.pendingUrls.has(n) && !batch.includes(n)) batch.push(n);
      }
      const uploadId = this.record!.uploadId;
      const request = this.api
        .requestPartUrls(uploadId, batch, signal)
        .then(({ data }) => {
          if (this.urlCacheUploadId !== uploadId) return; // the upload was replaced
          const expiresAt = Date.parse(data.expiresAt);
          for (const { partNumber: n, url } of data.urls) {
            this.urlCache.set(n, { url, expiresAt: Number.isNaN(expiresAt) ? 0 : expiresAt });
          }
        })
        .finally(() => {
          if (this.urlCacheUploadId !== uploadId) return;
          for (const n of batch) this.pendingUrls.delete(n);
        });
      for (const n of batch) this.pendingUrls.set(n, request);
      await request;
    }
    const cached = this.urlCache.get(partNumber);
    if (cached) return cached.url;
    throw new StoragePutError(0, `No upload URL was issued for part ${partNumber}.`, true);
  }
}
