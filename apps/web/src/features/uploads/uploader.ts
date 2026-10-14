/**
 * Resumable multipart uploader (README section 7, steps 1-4).
 *
 * - Initializes an upload through the API (which chooses the key and part size).
 * - Requests short-lived part URLs in small batches.
 * - PUTs parts directly to storage with bounded concurrency (3 by default) and
 *   reads the ETag response header (storage CORS must expose it).
 * - Retries failed parts with exponential backoff and jitter.
 * - Persists upload id, fingerprint, completed parts/ETags and expiry locally,
 *   so the same file can resume after a reload once the user re-selects it.
 * - Supports pause (abort in-flight requests) and resume (call start() again).
 */
import { ApiError } from "@/lib/api/errors";
import type { ApiClient } from "@/lib/api/client";
import type { MediaKind, Upload } from "@/lib/api/types";
import {
  fileFingerprint,
  getDefaultStorage,
  loadResumeRecord,
  partRange,
  remainingParts,
  removeResumeRecord,
  saveResumeRecord,
  withCompletedPart,
  type CompletedPartRecord,
  type KeyValueStorage,
  type UploadResumeRecord,
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
  "createUpload" | "getUpload" | "requestPartUrls" | "completeUpload"
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

function isRetryable(error: unknown): boolean {
  if (error instanceof StoragePutError) return error.retryable;
  if (error instanceof ApiError) return error.retryable || error.status >= 500;
  // Network failures surface as TypeError from fetch.
  return error instanceof TypeError;
}

export class MultipartUpload {
  readonly projectId: string;
  readonly file: UploadableFile;
  readonly mediaKind: MediaKind;
  readonly contentType: string;
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

  private record: UploadResumeRecord | null = null;
  private controller: AbortController | null = null;
  private running: Promise<Upload> | null = null;
  private listeners = new Set<(p: UploadProgress) => void>();
  private progress: UploadProgress;
  private urlCache = new Map<number, { url: string; expiresAt: number }>();
  private pendingUrls = new Map<number, Promise<void>>();

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
    this.fingerprint = fileFingerprint(file);
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

  private async run(signal: AbortSignal): Promise<Upload> {
    try {
      this.emit({ phase: "initializing", error: null });
      const record = await this.initialize(signal);
      this.record = record;
      this.emitPartProgress("uploading");

      const queue = remainingParts(record.partCount, record.completedParts);
      await this.uploadParts(queue, signal);

      if (signal.aborted) throw new UploadAbortedError();
      this.emit({ phase: "completing" });
      const { data } = await this.api.completeUpload(record.uploadId);
      removeResumeRecord(this.projectId, this.fingerprint, this.storage);
      this.record = null;
      this.emit({ phase: "completed", uploadedBytes: this.file.size });
      return data.upload;
    } catch (error) {
      if (signal.aborted || error instanceof UploadAbortedError) {
        this.emit({ phase: "paused" });
        throw new UploadAbortedError();
      }
      this.emit({ phase: "failed", error });
      throw error;
    }
  }

  private async initialize(signal: AbortSignal): Promise<UploadResumeRecord> {
    if (this.record) return this.record; // resuming within the same page session

    const saved = loadResumeRecord(this.projectId, this.file, {
      storage: this.storage,
      now: this.now(),
    });
    if (saved) {
      try {
        const { data } = await this.api.getUpload(saved.uploadId, signal);
        if (data.upload.status === "initiated") {
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
      removeResumeRecord(this.projectId, this.fingerprint, this.storage);
    }

    const { data } = await this.api.createUpload(this.projectId, {
      filename: this.file.name,
      bytes: this.file.size,
      mediaKind: this.mediaKind,
      contentType: this.contentType,
    });
    const record: UploadResumeRecord = {
      version: 1,
      uploadId: data.upload.id,
      projectId: this.projectId,
      fingerprint: this.fingerprint,
      partSize: data.partSize,
      partCount: data.partCount,
      completedParts: [],
      expiresAt: data.upload.expiresAt,
    };
    saveResumeRecord(record, this.storage);
    this.emit({ uploadId: record.uploadId, resumed: false });
    return record;
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
    signal.addEventListener("abort", onAbort, { once: true });

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
      const request = this.api
        .requestPartUrls(this.record!.uploadId, batch, signal)
        .then(({ data }) => {
          const expiresAt = Date.parse(data.expiresAt);
          for (const { partNumber: n, url } of data.urls) {
            this.urlCache.set(n, { url, expiresAt: Number.isNaN(expiresAt) ? 0 : expiresAt });
          }
        })
        .finally(() => {
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
