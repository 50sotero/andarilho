import { describe, expect, it } from "vitest";
import {
  CREATION_KEY_TTL_MS,
  EXPIRY_SAFETY_MARGIN_MS,
  RESUME_RECORD_VERSION,
  SAMPLE_BYTES,
  clearCreationKey,
  contentSample,
  creationKey,
  fileFingerprint,
  isExpired,
  loadResumeRecord,
  partRange,
  pruneResumeRecords,
  remainingParts,
  removeResumeRecord,
  resumeKey,
  saveResumeRecord,
  withCompletedPart,
  type UploadResumeRecord,
} from "@/lib/upload-resume";
import { MemoryStorage, ThrowingStorage } from "./memory-storage";

const MiB = 1024 * 1024;
const NOW = Date.parse("2026-10-14T12:00:00Z");
const file = {
  name: "walk through.mp4",
  size: 40 * MiB,
  lastModified: 1_700_000_000_000,
  mediaKind: "video",
  contentType: "video/mp4",
};

function record(overrides: Partial<UploadResumeRecord> = {}): UploadResumeRecord {
  return {
    version: RESUME_RECORD_VERSION,
    uploadId: "upload-1",
    projectId: "project-1",
    fingerprint: fileFingerprint(file),
    sample: "ab".repeat(32),
    partSize: 16 * MiB,
    partCount: 3,
    completedParts: [{ partNumber: 1, etag: '"a"' }],
    expiresAt: new Date(NOW + 24 * 3600 * 1000).toISOString(),
    ...overrides,
  };
}

describe("fileFingerprint", () => {
  it("is stable for the same name/size/lastModified", () => {
    expect(fileFingerprint(file)).toBe(fileFingerprint({ ...file }));
  });
  it("changes when any identity field changes", () => {
    const base = fileFingerprint(file);
    expect(fileFingerprint({ ...file, size: file.size + 1 })).not.toBe(base);
    expect(fileFingerprint({ ...file, lastModified: file.lastModified + 1 })).not.toBe(base);
    expect(fileFingerprint({ ...file, name: "other.mp4" })).not.toBe(base);
  });
  it("tells apart the same file uploaded as another media kind or content type", () => {
    // A JPEG chosen as a photo and as a floor plan is two uploads.
    const photo = { ...file, mediaKind: "photo", contentType: "image/jpeg" };
    expect(fileFingerprint({ ...photo, mediaKind: "floor_plan" })).not.toBe(fileFingerprint(photo));
    expect(fileFingerprint({ ...photo, contentType: "image/png" })).not.toBe(
      fileFingerprint(photo),
    );
  });
  it("does not let ':' in names collide with other fields", () => {
    expect(fileFingerprint({ ...file, name: "a:1", size: 2, lastModified: 3 })).not.toBe(
      fileFingerprint({ ...file, name: "a", size: 2, lastModified: 3 }),
    );
    expect(fileFingerprint({ ...file, mediaKind: "a:b", contentType: "c" })).not.toBe(
      fileFingerprint({ ...file, mediaKind: "a", contentType: "b:c" }),
    );
  });
});

/** A file of `size` bytes, each `fill`, except `changed` bytes set to 255. */
function bytesFile(size: number, fill = 7, changed: number[] = []) {
  const bytes = new Uint8Array(size).fill(fill);
  for (const at of changed) bytes[at] = 255;
  return { size, slice: (start: number, end: number) => new Blob([bytes.slice(start, end)]) };
}

describe("contentSample", () => {
  const PART = 4 * SAMPLE_BYTES;
  const size = 3 * PART + 100;

  it("is a stable SHA-256 hex digest of the same bytes", async () => {
    const sample = await contentSample(bytesFile(size), PART);
    expect(sample).toMatch(/^[0-9a-f]{64}$/);
    expect(await contentSample(bytesFile(size), PART)).toBe(sample);
  });

  it("changes when a part's start, its middle or the file's end differs", async () => {
    const base = await contentSample(bytesFile(size), PART);
    for (const at of [0, PART, 2 * PART + PART / 2, size - 1]) {
      expect(await contentSample(bytesFile(size, 7, [at]), PART)).not.toBe(base);
    }
    expect(await contentSample(bytesFile(size, 8), PART)).not.toBe(base);
  });

  it("depends on the part size, which decides what a resumed upload keeps", async () => {
    const file = bytesFile(size);
    expect(await contentSample(file, PART)).not.toBe(await contentSample(file, 2 * PART));
  });

  it("reads only samples, not the whole file", async () => {
    const big = 64 * 1024 * 1024;
    let read = 0;
    const file = {
      size: big,
      slice: (start: number, end: number) => {
        read += end - start;
        return new Blob([new Uint8Array(end - start)]);
      },
    };
    expect(await contentSample(file, 16 * 1024 * 1024)).not.toBeNull();
    expect(read).toBeLessThanOrEqual(9 * SAMPLE_BYTES);
  });
});

describe("persistence", () => {
  it("drops records written before the content sample", () => {
    const storage = new MemoryStorage();
    const { sample: _omitted, ...old } = record();
    storage.setItem(
      resumeKey("project-1", fileFingerprint(file)),
      JSON.stringify({ ...old, version: 1 }),
    );
    expect(loadResumeRecord("project-1", file, { storage, now: NOW })).toBeNull();
    expect(storage.length).toBe(0);
  });

  it("does not save a record without a content sample", () => {
    const storage = new MemoryStorage();
    expect(saveResumeRecord(record({ sample: "" }), storage)).toBe(false);
    expect(storage.length).toBe(0);
  });

  it("round-trips a record", () => {
    const storage = new MemoryStorage();
    const rec = record();
    expect(saveResumeRecord(rec, storage)).toBe(true);
    expect(loadResumeRecord("project-1", file, { storage, now: NOW })).toEqual(rec);
  });

  it("is scoped per project", () => {
    const storage = new MemoryStorage();
    saveResumeRecord(record(), storage);
    expect(loadResumeRecord("project-2", file, { storage, now: NOW })).toBeNull();
  });

  it("returns null for a different file", () => {
    const storage = new MemoryStorage();
    saveResumeRecord(record(), storage);
    expect(
      loadResumeRecord("project-1", { ...file, lastModified: 1 }, { storage, now: NOW }),
    ).toBeNull();
  });

  it("ignores and removes corrupted JSON", () => {
    const storage = new MemoryStorage();
    const key = resumeKey("project-1", fileFingerprint(file));
    storage.setItem(key, "{not json");
    expect(loadResumeRecord("project-1", file, { storage, now: NOW })).toBeNull();
    expect(storage.getItem(key)).toBeNull();
  });

  it("ignores structurally invalid records", () => {
    const storage = new MemoryStorage();
    const key = resumeKey("project-1", fileFingerprint(file));
    const bad = [
      { ...record(), version: 99 },
      { ...record(), partSize: -1 },
      { ...record(), completedParts: [{ partNumber: 0, etag: "x" }] },
      { ...record(), completedParts: [{ partNumber: 9, etag: "x" }] },
      { ...record(), completedParts: [{ partNumber: 1, etag: "" }] },
      { ...record(), expiresAt: "not a date" },
      { ...record(), uploadId: 7 },
      null,
      [],
    ];
    for (const value of bad) {
      storage.setItem(key, JSON.stringify(value));
      expect(loadResumeRecord("project-1", file, { storage, now: NOW })).toBeNull();
    }
  });

  it("ignores a record whose part count does not match the file size", () => {
    const storage = new MemoryStorage();
    saveResumeRecord(record({ partCount: 5 }), storage);
    expect(loadResumeRecord("project-1", file, { storage, now: NOW })).toBeNull();
  });

  it("ignores and removes expired records, but keeps one within the safety margin", () => {
    const storage = new MemoryStorage();
    saveResumeRecord(record({ expiresAt: new Date(NOW - 1000).toISOString() }), storage);
    expect(loadResumeRecord("project-1", file, { storage, now: NOW })).toBeNull();
    expect(storage.length).toBe(0);

    // The server may still accept it, and it holds quota: the uploader asks the server.
    const nearExpiry = record({
      expiresAt: new Date(NOW + EXPIRY_SAFETY_MARGIN_MS - 1000).toISOString(),
    });
    saveResumeRecord(nearExpiry, storage);
    expect(loadResumeRecord("project-1", file, { storage, now: NOW })).not.toBeNull();
    expect(isExpired(nearExpiry, NOW)).toBe(true);
  });

  it("deduplicates and sorts completed parts", () => {
    const storage = new MemoryStorage();
    const key = resumeKey("project-1", fileFingerprint(file));
    storage.setItem(
      key,
      JSON.stringify(
        record({
          completedParts: [
            { partNumber: 3, etag: "c" },
            { partNumber: 1, etag: "a" },
            { partNumber: 3, etag: "c2" },
          ],
        }),
      ),
    );
    const loaded = loadResumeRecord("project-1", file, { storage, now: NOW });
    expect(loaded?.completedParts).toEqual([
      { partNumber: 1, etag: "a" },
      { partNumber: 3, etag: "c" },
    ]);
  });

  it("survives storage that throws on every access", () => {
    const storage = new ThrowingStorage();
    expect(saveResumeRecord(record(), storage)).toBe(false);
    expect(loadResumeRecord("project-1", file, { storage, now: NOW })).toBeNull();
    expect(() => removeResumeRecord("project-1", "x", storage)).not.toThrow();
    expect(pruneResumeRecords(storage, NOW)).toBe(0);
  });

  it("works without storage at all", () => {
    expect(saveResumeRecord(record(), null)).toBe(false);
    expect(loadResumeRecord("project-1", file, { storage: null, now: NOW })).toBeNull();
  });

  it("prunes only expired or corrupted upload records", () => {
    const storage = new MemoryStorage();
    saveResumeRecord(record(), storage);
    saveResumeRecord(
      record({ fingerprint: "old", expiresAt: new Date(NOW - 1).toISOString() }),
      storage,
    );
    storage.setItem("andarilho:upload:broken", "nope");
    storage.setItem("unrelated", "keep");
    expect(pruneResumeRecords(storage, NOW)).toBe(2);
    expect(storage.getItem("unrelated")).toBe("keep");
    expect(loadResumeRecord("project-1", file, { storage, now: NOW })).not.toBeNull();
  });
});

describe("upload creation keys", () => {
  const fp = fileFingerprint(file);
  let n = 0;
  const newKey = () => `key-${(n += 1).toString().padStart(4, "0")}`;

  it("keeps one key until the creation is answered, across uploader instances", () => {
    const storage = new MemoryStorage();
    const first = creationKey("project-1", fp, { storage, now: NOW, newKey });
    expect(creationKey("project-1", fp, { storage, now: NOW + 1000, newKey })).toBe(first);
    expect(creationKey("project-2", fp, { storage, now: NOW, newKey })).not.toBe(first);
    clearCreationKey("project-1", fp, storage);
    expect(creationKey("project-1", fp, { storage, now: NOW, newKey })).not.toBe(first);
  });

  it("starts over once the server would no longer replay the key", () => {
    const storage = new MemoryStorage();
    const first = creationKey("project-1", fp, { storage, now: NOW, newKey });
    const later = NOW + CREATION_KEY_TTL_MS;
    expect(creationKey("project-1", fp, { storage, now: later, newKey })).not.toBe(first);
  });

  it("works without storage and prunes stale or broken keys", () => {
    expect(creationKey("project-1", fp, { storage: null, newKey })).toMatch(/^key-/);
    expect(creationKey("project-1", fp, { storage: new ThrowingStorage(), newKey })).toMatch(
      /^key-/,
    );
    const storage = new MemoryStorage();
    creationKey("project-1", fp, { storage, now: NOW, newKey });
    creationKey("project-1", "old", { storage, now: NOW - CREATION_KEY_TTL_MS, newKey });
    storage.setItem("andarilho:upload-create:broken", "nope");
    expect(pruneResumeRecords(storage, NOW)).toBe(2);
    expect(storage.length).toBe(1);
  });
});

describe("part bookkeeping", () => {
  it("computes remaining parts", () => {
    expect(remainingParts(5, [{ partNumber: 2 }, { partNumber: 4 }])).toEqual([1, 3, 5]);
    expect(remainingParts(3, [])).toEqual([1, 2, 3]);
    expect(remainingParts(2, [{ partNumber: 1 }, { partNumber: 2 }])).toEqual([]);
  });

  it("marks parts complete idempotently", () => {
    let rec = record({ completedParts: [] });
    rec = withCompletedPart(rec, { partNumber: 2, etag: "b" });
    rec = withCompletedPart(rec, { partNumber: 1, etag: "a" });
    rec = withCompletedPart(rec, { partNumber: 2, etag: "b2" });
    expect(rec.completedParts).toEqual([
      { partNumber: 1, etag: "a" },
      { partNumber: 2, etag: "b2" },
    ]);
  });

  it("computes byte ranges with a short last part", () => {
    expect(partRange(1, 16, 40)).toEqual([0, 16]);
    expect(partRange(3, 16, 40)).toEqual([32, 40]);
  });
});
