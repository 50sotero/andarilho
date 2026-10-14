import { describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api/errors";
import type { Upload } from "@/lib/api/types";
import {
  MultipartUpload,
  StoragePutError,
  UploadAbortedError,
  type PutPart,
  type UploadableFile,
  type UploaderApi,
} from "@/features/uploads/uploader";
import { createLimiter } from "@/features/uploads/limiter";
import {
  fileFingerprint,
  loadResumeRecord,
  resumeKey,
  saveResumeRecord,
} from "@/lib/upload-resume";
import { MemoryStorage } from "./memory-storage";

const PART = 100;
const FUTURE = new Date(Date.now() + 24 * 3600 * 1000).toISOString();

function fakeFile(size: number, name = "walk.mp4"): UploadableFile {
  return {
    name,
    size,
    lastModified: 1_700_000_000_000,
    type: "video/mp4",
    slice: (start, end) => new Blob([new Uint8Array(end - start)]),
  };
}

function uploadRecord(id: string, status: Upload["status"] = "initiated"): Upload {
  return {
    id,
    projectId: "p1",
    mediaKind: "video",
    filename: "walk.mp4",
    contentType: "video/mp4",
    expectedBytes: 0,
    status,
    partSize: PART,
    partCount: 0,
    rejectionCode: null,
    metadata: {},
    expiresAt: FUTURE,
    createdAt: FUTURE,
  };
}

function fakeApi(fileSize: number, overrides: Partial<UploaderApi> = {}) {
  const partCount = Math.ceil(fileSize / PART);
  const batches: number[][] = [];
  const api = {
    createUpload: vi.fn(async () => ({
      data: { upload: uploadRecord("u-new"), partSize: PART, partCount },
      etag: null,
    })),
    getUpload: vi.fn(async (): Promise<never> => {
      throw new ApiError(404, { code: "NOT_FOUND", message: "Not found.", retryable: false });
    }),
    requestPartUrls: vi.fn(async (uploadId: string, partNumbers: number[]) => {
      batches.push([...partNumbers]);
      return {
        data: {
          urls: partNumbers.map((n) => ({
            partNumber: n,
            url: `https://storage.test/${uploadId}/${n}`,
          })),
          expiresAt: FUTURE,
        },
        etag: null,
      };
    }),
    completeUpload: vi.fn(async (uploadId: string) => ({
      data: { upload: uploadRecord(uploadId, "uploaded_unverified") },
      etag: null,
    })),
    ...overrides,
  };
  return { api: api as unknown as UploaderApi & typeof api, batches, partCount };
}

const partOf = (url: string) => Number(url.split("/").pop());
const noSleep = vi.fn(async () => undefined);

describe("MultipartUpload", () => {
  it("requests URLs in small batches and uploads with at most 3 concurrent PUTs", async () => {
    const size = 25 * PART - 7;
    const { api, batches, partCount } = fakeApi(size);
    let active = 0;
    let maxActive = 0;
    const uploaded: number[] = [];
    const putPart: PutPart = async (url, body) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise((r) => setTimeout(r, 2));
      active -= 1;
      expect(body.size).toBe(partOf(url) === partCount ? size - (partCount - 1) * PART : PART);
      uploaded.push(partOf(url));
      return `"etag-${partOf(url)}"`;
    };
    const storage = new MemoryStorage();
    const upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api,
      putPart,
      storage,
      urlBatchSize: 10,
    });
    const result = await upload.start();

    expect(result.status).toBe("uploaded_unverified");
    expect(maxActive).toBe(3);
    expect([...uploaded].sort((a, b) => a - b)).toEqual(
      Array.from({ length: partCount }, (_, i) => i + 1),
    );
    expect(batches.every((b) => b.length >= 1 && b.length <= 10)).toBe(true);
    expect(new Set(batches.flat()).size).toBe(partCount);
    expect(batches.length).toBeLessThanOrEqual(6);
    expect(api.completeUpload).toHaveBeenCalledTimes(1);
    expect(upload.state.phase).toBe("completed");
    expect(upload.state.uploadedBytes).toBe(size);
    // The resume record is removed after completion.
    expect(storage.length).toBe(0);
  });

  it("shares a global transfer limit across files", async () => {
    const limiter = createLimiter(3);
    let active = 0;
    let maxActive = 0;
    const put: PutPart = (url) =>
      limiter.run(async () => {
        active += 1;
        maxActive = Math.max(maxActive, active);
        await new Promise((r) => setTimeout(r, 2));
        active -= 1;
        return `"e${partOf(url)}"`;
      });
    const uploads = [0, 1, 2].map((i) => {
      const { api } = fakeApi(6 * PART);
      return new MultipartUpload("p1", fakeFile(6 * PART, `f${i}.mp4`), "video", "video/mp4", {
        api,
        putPart: put,
        storage: null,
      });
    });
    await Promise.all(uploads.map((u) => u.start()));
    expect(maxActive).toBe(3);
  });

  it("retries a failed part with exponential backoff", async () => {
    const { api } = fakeApi(3 * PART);
    const failures = new Map<number, number>([[2, 2]]);
    const putPart: PutPart = async (url) => {
      const n = partOf(url);
      const left = failures.get(n) ?? 0;
      if (left > 0) {
        failures.set(n, left - 1);
        throw new StoragePutError(503, "busy", true);
      }
      return `"e${n}"`;
    };
    const sleep = vi.fn(async () => undefined);
    const upload = new MultipartUpload("p1", fakeFile(3 * PART), "video", "video/mp4", {
      api,
      putPart,
      storage: null,
      sleep,
      random: () => 0,
      baseRetryDelayMs: 100,
    });
    await upload.start();
    expect(sleep.mock.calls.map((c) => (c as unknown[])[0])).toEqual([100, 200]);
    expect(api.completeUpload).toHaveBeenCalledTimes(1);
  });

  it("treats network TypeErrors as retryable", async () => {
    const { api } = fakeApi(PART);
    let calls = 0;
    const putPart: PutPart = async () => {
      calls += 1;
      if (calls === 1) throw new TypeError("Failed to fetch");
      return '"e1"';
    };
    const upload = new MultipartUpload("p1", fakeFile(PART), "video", "video/mp4", {
      api,
      putPart,
      storage: null,
      sleep: noSleep,
    });
    await upload.start();
    expect(calls).toBe(2);
  });

  it("gives up after maxAttempts and does not complete", async () => {
    const { api } = fakeApi(2 * PART);
    const putPart: PutPart = async () => {
      throw new StoragePutError(500, "boom", true);
    };
    const upload = new MultipartUpload("p1", fakeFile(2 * PART), "video", "video/mp4", {
      api,
      putPart,
      storage: null,
      sleep: noSleep,
      maxAttempts: 3,
    });
    await expect(upload.start()).rejects.toBeInstanceOf(StoragePutError);
    expect(api.completeUpload).not.toHaveBeenCalled();
    expect(upload.state.phase).toBe("failed");
  });

  it("does not retry non-retryable errors such as a missing ETag header", async () => {
    const { api } = fakeApi(PART);
    const putPart = vi.fn<PutPart>(async () => {
      throw new StoragePutError(0, "no etag", false);
    });
    const upload = new MultipartUpload("p1", fakeFile(PART), "video", "video/mp4", {
      api,
      putPart,
      storage: null,
      sleep: noSleep,
    });
    await expect(upload.start()).rejects.toThrow("no etag");
    expect(putPart).toHaveBeenCalledTimes(1);
  });

  it("refreshes the signed URL after a 403 from storage", async () => {
    const { api, batches } = fakeApi(PART);
    let calls = 0;
    const putPart: PutPart = async () => {
      calls += 1;
      if (calls === 1) throw new StoragePutError(403, "expired", true);
      return '"e1"';
    };
    const upload = new MultipartUpload("p1", fakeFile(PART), "video", "video/mp4", {
      api,
      putPart,
      storage: null,
      sleep: noSleep,
    });
    await upload.start();
    expect(batches).toEqual([[1], [1]]);
  });

  it("persists progress, pauses, and resumes after a reload with the same file", async () => {
    const size = 6 * PART;
    const storage = new MemoryStorage();
    const first = fakeApi(size);
    const done: number[] = [];
    let upload: MultipartUpload | null = null;
    const pausingPut: PutPart = async (url, _body, signal) => {
      const n = partOf(url);
      if (done.length >= 2) {
        upload!.pause();
        if (signal.aborted) throw new DOMException("Aborted", "AbortError");
      }
      done.push(n);
      return `"e${n}"`;
    };
    upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api: first.api,
      putPart: pausingPut,
      storage,
      concurrency: 1,
    });
    await expect(upload.start()).rejects.toBeInstanceOf(UploadAbortedError);
    expect(upload.state.phase).toBe("paused");

    const saved = loadResumeRecord("p1", fakeFile(size), { storage });
    expect(saved?.uploadId).toBe("u-new");
    expect(saved?.completedParts.map((p) => p.partNumber)).toEqual([1, 2]);

    // "Reload": a fresh uploader, same storage, same file re-selected.
    const second = fakeApi(size, {
      getUpload: vi.fn(async () => ({
        data: {
          upload: uploadRecord("u-new", "initiated"),
          completedParts: [
            { partNumber: 1, etag: '"e1"', size: PART },
            { partNumber: 2, etag: '"e2"', size: PART },
          ],
        },
        etag: null,
      })),
    });
    const uploadedAfterReload: number[] = [];
    const resumed = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api: second.api,
      putPart: async (url) => {
        uploadedAfterReload.push(partOf(url));
        return `"e${partOf(url)}"`;
      },
      storage,
    });
    await resumed.start();
    expect(second.api.createUpload).not.toHaveBeenCalled();
    expect(second.api.getUpload).toHaveBeenCalledWith("u-new", expect.anything());
    expect(uploadedAfterReload.sort()).toEqual([3, 4, 5, 6]);
    expect(resumed.state.resumed).toBe(true);
    expect(second.api.completeUpload).toHaveBeenCalledWith("u-new");
    expect(storage.length).toBe(0);
  });

  it("starts a new upload when the saved one no longer exists on the server", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const file = fakeFile(size);
    saveResumeRecord(
      {
        version: 1,
        uploadId: "u-gone",
        projectId: "p1",
        fingerprint: fileFingerprint(file),
        partSize: PART,
        partCount: 2,
        completedParts: [{ partNumber: 1, etag: '"x"' }],
        expiresAt: FUTURE,
      },
      storage,
    );
    const { api } = fakeApi(size);
    const upload = new MultipartUpload("p1", file, "video", "video/mp4", {
      api,
      putPart: async (url) => `"e${partOf(url)}"`,
      storage,
    });
    await upload.start();
    expect(api.getUpload).toHaveBeenCalledTimes(1);
    expect(api.createUpload).toHaveBeenCalledTimes(1);
    expect(storage.getItem(resumeKey("p1", fileFingerprint(file)))).toBeNull();
  });

  it("treats an upload the server already completed as done after a reload", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const file = fakeFile(size);
    saveResumeRecord(
      {
        version: 1,
        uploadId: "u-done",
        projectId: "p1",
        fingerprint: fileFingerprint(file),
        partSize: PART,
        partCount: 2,
        completedParts: [{ partNumber: 1, etag: '"x"' }],
        expiresAt: FUTURE,
      },
      storage,
    );
    const { api } = fakeApi(size, {
      getUpload: vi.fn(async () => ({
        data: { upload: uploadRecord("u-done", "uploaded_unverified"), completedParts: [] },
        etag: null,
      })),
    });
    const putPart = vi.fn<PutPart>(async () => '"e"');
    const upload = new MultipartUpload("p1", file, "video", "video/mp4", {
      api,
      putPart,
      storage,
    });
    const result = await upload.start();
    expect(result.id).toBe("u-done");
    expect(upload.state.phase).toBe("completed");
    expect(api.createUpload).not.toHaveBeenCalled();
    expect(api.completeUpload).not.toHaveBeenCalled();
    expect(putPart).not.toHaveBeenCalled();
    expect(storage.getItem(resumeKey("p1", fileFingerprint(file)))).toBeNull();
  });

  it("can resume in the same session after pause without re-initializing", async () => {
    const size = 3 * PART;
    const { api } = fakeApi(size);
    let block = true;
    let upload: MultipartUpload | null = null;
    const putPart: PutPart = async (url, _b, signal) => {
      if (block && partOf(url) === 2) {
        upload!.pause();
        if (signal.aborted) throw new DOMException("Aborted", "AbortError");
      }
      return `"e${partOf(url)}"`;
    };
    upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api,
      putPart,
      storage: null,
      concurrency: 1,
    });
    await expect(upload.start()).rejects.toBeInstanceOf(UploadAbortedError);
    block = false;
    await upload.start();
    expect(api.createUpload).toHaveBeenCalledTimes(1);
    expect(upload.state.phase).toBe("completed");
  });
  it("starts a new upload when a paused upload expired in the same session", async () => {
    const size = 3 * PART;
    const { api } = fakeApi(size);
    let clock = Date.now();
    let block = true;
    let upload: MultipartUpload | null = null;
    const putPart: PutPart = async (url, _b, signal) => {
      if (block && partOf(url) === 2) {
        upload!.pause();
        if (signal.aborted) throw new DOMException("Aborted", "AbortError");
      }
      return `"e${partOf(url)}"`;
    };
    upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api,
      putPart,
      storage: null,
      concurrency: 1,
      now: () => clock,
    });
    await expect(upload.start()).rejects.toBeInstanceOf(UploadAbortedError);
    clock += 2 * 24 * 3600 * 1000; // paused past the upload's expiry
    block = false;
    await upload.start();
    expect(api.createUpload).toHaveBeenCalledTimes(2);
    expect(upload.state.phase).toBe("completed");
  });

  it("drops an upload the server reports expired so Resume starts over", async () => {
    const size = 2 * PART;
    let expired = true;
    const signer = fakeApi(size).api;
    const { api } = fakeApi(size, {
      requestPartUrls: async (id: string, parts: number[]) => {
        if (expired) {
          throw new ApiError(410, {
            code: "UPLOAD_EXPIRED",
            message: "Expired.",
            retryable: false,
          });
        }
        return signer.requestPartUrls(id, parts);
      },
    });
    const upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api,
      putPart: async (url) => `"e${partOf(url)}"`,
      storage: null,
      sleep: noSleep,
    });
    await expect(upload.start()).rejects.toBeInstanceOf(ApiError);
    expired = false;
    await upload.start();
    expect(api.createUpload).toHaveBeenCalledTimes(2);
    expect(upload.state.phase).toBe("completed");
  });
});
