/**
 * Local resume state for multipart uploads (README section 7, step 4).
 *
 * The browser cannot reopen a file after reload, so the user re-selects it;
 * we match it to a persisted record by a fingerprint built from the file's
 * name, size and lastModified (a hint, not a content hash). Storage access is
 * always wrapped in try/catch: private windows and blocked storage must not
 * break uploading, they only disable resume.
 */

export const RESUME_STORAGE_PREFIX = "andarilho:upload:";
export const RESUME_RECORD_VERSION = 1;

export interface FileIdentity {
  name: string;
  size: number;
  lastModified: number;
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
  return `v1:${file.size}:${file.lastModified}:${encodeURIComponent(file.name)}`;
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
    partSize: r.partSize,
    partCount: r.partCount,
    completedParts: parts,
    expiresAt: r.expiresAt,
  };
}

/** Records within this margin of expiry are treated as expired (time to finish uploading). */
export const EXPIRY_SAFETY_MARGIN_MS = 5 * 60 * 1000;

export function isExpired(record: UploadResumeRecord, now: number = Date.now()): boolean {
  return Date.parse(record.expiresAt) - EXPIRY_SAFETY_MARGIN_MS <= now;
}

export function saveResumeRecord(
  record: UploadResumeRecord,
  storage: KeyValueStorage | null = getDefaultStorage(),
): boolean {
  if (!storage) return false;
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
 * records are deleted and ignored.
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
    isExpired(record, options.now)
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
/** The API keeps idempotency records for a day by default (IDEMPOTENCY_TTL_SECONDS). */
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
      if (!record || isExpired(record, now)) doomed.push(key);
    }
    for (const key of doomed) storage.removeItem(key);
  } catch {
    return 0;
  }
  return doomed.length;
}
