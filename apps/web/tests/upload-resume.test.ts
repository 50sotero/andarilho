import { describe, expect, it } from "vitest";
import {
  EXPIRY_SAFETY_MARGIN_MS,
  fileFingerprint,
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
const file = { name: "walk through.mp4", size: 40 * MiB, lastModified: 1_700_000_000_000 };

function record(overrides: Partial<UploadResumeRecord> = {}): UploadResumeRecord {
  return {
    version: 1,
    uploadId: "upload-1",
    projectId: "project-1",
    fingerprint: fileFingerprint(file),
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
  it("does not let ':' in names collide with other fields", () => {
    expect(fileFingerprint({ name: "a:1", size: 2, lastModified: 3 })).not.toBe(
      fileFingerprint({ name: "a", size: 2, lastModified: 3 }),
    );
  });
});

describe("persistence", () => {
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
    expect(loadResumeRecord("project-1", { ...file, lastModified: 1 }, { storage, now: NOW })).toBeNull();
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

  it("ignores and removes expired records, including within the safety margin", () => {
    const storage = new MemoryStorage();
    saveResumeRecord(record({ expiresAt: new Date(NOW - 1000).toISOString() }), storage);
    expect(loadResumeRecord("project-1", file, { storage, now: NOW })).toBeNull();
    expect(storage.length).toBe(0);

    saveResumeRecord(
      record({ expiresAt: new Date(NOW + EXPIRY_SAFETY_MARGIN_MS - 1000).toISOString() }),
      storage,
    );
    expect(loadResumeRecord("project-1", file, { storage, now: NOW })).toBeNull();
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
