/**
 * Local resume state for multipart uploads (README section 7, step 4).
 *
 * The browser cannot reopen a file after reload, so the user re-selects it;
 * we find its persisted record by a fingerprint built from the file's name,
 * size and lastModified. That is only a hint: another file can share all
 * three, so the record also holds a hash of sampled bytes (`contentSample`),
 * and the uploader reuses the upload only when the re-selected file's sample
 * matches. Storage access is always wrapped in try/catch: private windows and
 * blocked storage must not break uploading, they only disable resume.
 */

export const RESUME_STORAGE_PREFIX = "andarilho:upload:";
/** Version 2 added `sample`; older records cannot be checked, so they are dropped. */
export const RESUME_RECORD_VERSION = 2;
/** Bytes hashed at the start and the middle of every part, and at the end of the file. */
export const SAMPLE_BYTES = 16 * 1024;

/**
 * What a resume record and a creation key belong to: the file, and how it is uploaded.
 * The same file chosen as a photo and as a floor plan is two uploads, and neither may
 * take over the other's.
 */
export interface FileIdentity {
  name: string;
  size: number;
  lastModified: number;
  mediaKind: string;
  /** The declared content type sent when creating the upload. */
  contentType: string;
}

/** A file's identity for an upload of `mediaKind` (a DOM `File` cannot be spread). */
export function uploadIdentity(
  file: { name: string; size: number; lastModified: number },
  mediaKind: string,
  contentType: string,
): FileIdentity {
  return {
    name: file.name,
    size: file.size,
    lastModified: file.lastModified,
    mediaKind,
    contentType,
  };
}

export interface CompletedPartRecord {
  partNumber: number;
  etag: string;
}

export interface UploadResumeRecord {
  version: typeof RESUME_RECORD_VERSION;
  uploadId: string;
  projectId: string;
  fingerprint: string;
  /** `contentSample` of the file at this record's part size. */
  sample: string;
  partSize: number;
  partCount: number;
  completedParts: CompletedPartRecord[];
  expiresAt: string;
}

/** Minimal Storage subset so tests can inject fakes. */
export interface KeyValueStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  readonly length: number;
  key(index: number): string | null;
}

export function fileFingerprint(file: FileIdentity): string {
  const kind = `${encodeURIComponent(file.mediaKind)}:${encodeURIComponent(file.contentType)}`;
  return `v2:${kind}:${file.size}:${file.lastModified}:${encodeURIComponent(file.name)}`;
}

/** The part of a file `contentSample` reads. */
export interface SampledFile {
  size: number;
  slice(start: number, end: number): Blob;
}

/**
 * SHA-256 (hex) of `SAMPLE_BYTES` at the start and the middle of every part
 * and at the end of the file, with the size and part size. Every part a
 * resumed upload keeps is sampled, so a different file with the same name,
 * size and date is caught without reading it all (at most a few MiB for the
 * largest upload). Null where Web Crypto is unavailable (an insecure origin);
 * resume is then off.
 */
export async function contentSample(file: SampledFile, partSize: number): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle || !(partSize > 0)) return null;
  const windows: [number, number][] = [];
  const add = (start: number) => {
    const from = Math.max(0, Math.min(start, file.size - 1));
    windows.push([from, Math.min(from + SAMPLE_BYTES, file.size)]);
  };
  const partCount = Math.ceil(file.size / partSize);
  for (let n = 1; n <= partCount; n += 1) {
    const [start, end] = partRange(n, partSize, file.size);
    add(start);
    add(start + Math.floor((end - start) / 2));
  }
  add(file.size - SAMPLE_BYTES);
  const header = new TextEncoder().encode(`v1:${file.size}:${partSize}:`);
  const chunks = await Promise.all(
    windows.map(async ([start, end]) => new Uint8Array(await file.slice(start, end).arrayBuffer())),
  );
  const joined = new Uint8Array(header.length + chunks.reduce((n, c) => n + c.length, 0));
  joined.set(header, 0);
  let offset = header.length;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.length;
  }
  const digest = new Uint8Array(await subtle.digest("SHA-256", joined));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

export function resumeKey(projectId: string, fingerprint: string): string {
  return `${RESUME_STORAGE_PREFIX}${projectId}:${fingerprint}`;
}

export function getDefaultStorage(): KeyValueStorage | null {
  try {
    if (typeof window === "undefined") return null;
    return window.localStorage;
  } catch {
    return null;
  }
}

function isPositiveInt(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value > 0;
}

/** Validates an untrusted parsed record. Returns null when it is malformed. */
export function parseResumeRecord(raw: unknown): UploadResumeRecord | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  if (r.version !== RESUME_RECORD_VERSION) return null;
  if (typeof r.uploadId !== "string" || !r.uploadId) return null;
  if (typeof r.projectId !== "string" || !r.projectId) return null;
  if (typeof r.fingerprint !== "string" || !r.fingerprint) return null;
  if (typeof r.sample !== "string" || !r.sample) return null;
  if (!isPositiveInt(r.partSize) || !isPositiveInt(r.partCount)) return null;
  if (typeof r.expiresAt !== "string" || Number.isNaN(Date.parse(r.expiresAt))) return null;
  if (!Array.isArray(r.completedParts)) return null;
  const parts: CompletedPartRecord[] = [];
  const seen = new Set<number>();
  for (const p of r.completedParts) {
    if (!p || typeof p !== "object") return null;
    const part = p as Record<string, unknown>;
    if (!isPositiveInt(part.partNumber) || part.partNumber > r.partCount) return null;
    if (typeof part.etag !== "string" || !part.etag) return null;
    if (seen.has(part.partNumber)) continue;
    seen.add(part.partNumber);
    parts.push({ partNumber: part.partNumber, etag: part.etag });
  }
  parts.sort((a, b) => a.partNumber - b.partNumber);
  return {
    version: RESUME_RECORD_VERSION,
    uploadId: r.uploadId,
    projectId: r.projectId,
    fingerprint: r.fingerprint,
    sample: r.sample,
    partSize: r.partSize,
    partCount: r.partCount,
    completedParts: parts,
    expiresAt: r.expiresAt,
  };
}

/**
 * Within this margin of its expiry a record is no longer trusted on its own: the
 * uploader asks the server whether the upload still takes parts.
 */
export const EXPIRY_SAFETY_MARGIN_MS = 5 * 60 * 1000;

/** True near or past the record's expiry: the server must confirm the upload. */
export function isExpired(record: UploadResumeRecord, now: number = Date.now()): boolean {
  return Date.parse(record.expiresAt) - EXPIRY_SAFETY_MARGIN_MS <= now;
}

/**
 * True once the record's expiry has passed. Until then the server may still accept the
 * upload, and the upload holds the project's quota, so a replacement could be refused.
 */
export function hasLapsed(record: UploadResumeRecord, now: number = Date.now()): boolean {
  return Date.parse(record.expiresAt) <= now;
}

export function saveResumeRecord(
  record: UploadResumeRecord,
  storage: KeyValueStorage | null = getDefaultStorage(),
): boolean {
  // A record without a content sample could not be checked against the file on resume.
  if (!storage || !record.sample) return false;
  try {
    storage.setItem(resumeKey(record.projectId, record.fingerprint), JSON.stringify(record));
    return true;
  } catch {
    return false;
  }
}

export function removeResumeRecord(
  projectId: string,
  fingerprint: string,
  storage: KeyValueStorage | null = getDefaultStorage(),
): void {
  if (!storage) return;
  try {
    storage.removeItem(resumeKey(projectId, fingerprint));
  } catch {
    // ignore
  }
}

/**
 * Loads a record for this project/file. Corrupted, mismatched or expired
 * records are deleted and ignored. The file's identity only finds the record:
 * check its `sample` against the file (`contentSample`) before reusing it.
 */
export function loadResumeRecord(
  projectId: string,
  file: FileIdentity,
  options: { storage?: KeyValueStorage | null; now?: number } = {},
): UploadResumeRecord | null {
  const storage = options.storage === undefined ? getDefaultStorage() : options.storage;
  if (!storage) return null;
  const fingerprint = fileFingerprint(file);
  const key = resumeKey(projectId, fingerprint);
  let text: string | null;
  try {
    text = storage.getItem(key);
  } catch {
    return null;
  }
  if (!text) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    removeResumeRecord(projectId, fingerprint, storage);
    return null;
  }
  const record = parseResumeRecord(parsed);
  const expectedParts = record ? Math.ceil(file.size / record.partSize) : 0;
  if (
    !record ||
    record.projectId !== projectId ||
    record.fingerprint !== fingerprint ||
    expectedParts !== record.partCount ||
    hasLapsed(record, options.now)
  ) {
    removeResumeRecord(projectId, fingerprint, storage);
    return null;
  }
  return record;
}

/** Returns a new record with this part marked complete (idempotent). */
export function withCompletedPart(
  record: UploadResumeRecord,
  part: CompletedPartRecord,
): UploadResumeRecord {
  const others = record.completedParts.filter((p) => p.partNumber !== part.partNumber);
  return {
    ...record,
    completedParts: [...others, part].sort((a, b) => a.partNumber - b.partNumber),
  };
}

/** Part numbers (1-based, ascending) that still need uploading. */
export function remainingParts(
  partCount: number,
  completed: readonly { partNumber: number }[],
): number[] {
  const done = new Set(completed.map((p) => p.partNumber));
  const out: number[] = [];
  for (let n = 1; n <= partCount; n += 1) {
    if (!done.has(n)) out.push(n);
  }
  return out;
}

/** Byte range [start, end) for a 1-based part number. */
export function partRange(
  partNumber: number,
  partSize: number,
  fileSize: number,
): [number, number] {
  const start = (partNumber - 1) * partSize;
  return [start, Math.min(start + partSize, fileSize)];
}

/*
 * Upload creation keys. A POST that creates an upload can succeed on the server
 * and still lose its response; retrying with the same Idempotency-Key replays that
 * upload instead of creating (and charging quota for) a second one. The key is kept
 * until the creation gets a definite answer, across reloads where storage allows.
 */

export const CREATION_STORAGE_PREFIX = "andarilho:upload-create:";
/**
 * The API keeps idempotency records at least this long: it refuses an
 * IDEMPOTENCY_TTL_SECONDS below a day, and its settings test checks this value.
 */
export const CREATION_KEY_TTL_MS = 24 * 60 * 60 * 1000;
const CREATION_KEY = /^[A-Za-z0-9._:-]{8,255}$/;

function creationSlot(projectId: string, fingerprint: string): string {
  return `${CREATION_STORAGE_PREFIX}${projectId}:${fingerprint}`;
}

function parseCreationKey(text: string | null, now: number): string | null {
  if (!text) return null;
  try {
    const saved = JSON.parse(text) as { key?: unknown; createdAt?: unknown };
    if (typeof saved.key !== "string" || !CREATION_KEY.test(saved.key)) return null;
    if (typeof saved.createdAt !== "number" || now - saved.createdAt >= CREATION_KEY_TTL_MS) {
      return null;
    }
    return saved.key;
  } catch {
    return null;
  }
}

/** The pending creation key for this file, or a new one (saved where storage allows). */
export function creationKey(
  projectId: string,
  fingerprint: string,
  options: { storage?: KeyValueStorage | null; now?: number; newKey?: () => string } = {},
): string {
  const storage = options.storage === undefined ? getDefaultStorage() : options.storage;
  const now = options.now ?? Date.now();
  const slot = creationSlot(projectId, fingerprint);
  try {
    const saved = parseCreationKey(storage?.getItem(slot) ?? null, now);
    if (saved) return saved;
  } catch {
    // Fall through to a new key.
  }
  const key = (options.newKey ?? (() => crypto.randomUUID()))();
  try {
    storage?.setItem(slot, JSON.stringify({ key, createdAt: now }));
  } catch {
    // Without storage the key lasts as long as the uploader that holds it.
  }
  return key;
}

/** Forgets the key once the creation got a definite answer. */
export function clearCreationKey(
  projectId: string,
  fingerprint: string,
  storage: KeyValueStorage | null = getDefaultStorage(),
): void {
  try {
    storage?.removeItem(creationSlot(projectId, fingerprint));
  } catch {
    // ignore
  }
}

/** Removes expired or corrupted records and creation keys for any project; returns how many. */
export function pruneResumeRecords(
  storage: KeyValueStorage | null = getDefaultStorage(),
  now: number = Date.now(),
): number {
  if (!storage) return 0;
  const doomed: string[] = [];
  try {
    for (let i = 0; i < storage.length; i += 1) {
      const key = storage.key(i);
      if (key?.startsWith(CREATION_STORAGE_PREFIX)) {
        if (!parseCreationKey(storage.getItem(key), now)) doomed.push(key);
        continue;
      }
      if (!key || !key.startsWith(RESUME_STORAGE_PREFIX)) continue;
      let record: UploadResumeRecord | null = null;
      try {
        record = parseResumeRecord(JSON.parse(storage.getItem(key) ?? "null"));
      } catch {
        record = null;
      }
      if (!record || hasLapsed(record, now)) doomed.push(key);
    }
    for (const key of doomed) storage.removeItem(key);
  } catch {
    return 0;
  }
  return doomed.length;
}
