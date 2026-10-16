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
  RESUME_RECORD_VERSION,
  contentSample,
  fileFingerprint,
  loadResumeRecord,
  resumeKey,
  saveResumeRecord,
  uploadIdentity,
} from "@/lib/upload-resume";
import { MemoryStorage } from "./memory-storage";

const PART = 100;
const FUTURE = new Date(Date.now() + 24 * 3600 * 1000).toISOString();

function fakeFile(size: number, name = "walk.mp4", fill = 0): UploadableFile {
  return {
    name,
    size,
    lastModified: 1_700_000_000_000,
    type: "video/mp4",
    slice: (start, end) => new Blob([new Uint8Array(end - start).fill(fill)]),
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
            bytes: Math.min(PART, fileSize - (n - 1) * PART),
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
    abortUpload: vi.fn(async () => ({ data: undefined, etag: null })),
    ...overrides,
  };
  return { api: api as unknown as UploaderApi & typeof api, batches, partCount };
}

const partOf = (url: string) => Number(url.split("/").pop());
/** The identity of a file sent as these tests' uploaders send it, as a video. */
const video = (file: UploadableFile) => uploadIdentity(file, "video", "video/mp4");
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

    const saved = loadResumeRecord("p1", video(fakeFile(size)), { storage });
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
          acceptingParts: true,
          assembled: false,
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

  it("never resumes the same file's upload made as another media kind", async () => {
    // One JPEG chosen as a photo and as the floor plan: a photo upload cannot serve as
    // the capture set's floor plan.
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const created: string[] = [];
    const { api } = fakeApi(size, {
      createUpload: vi.fn(async (_p: string, body: { mediaKind: string }) => {
        created.push(body.mediaKind);
        const upload = uploadRecord(`u-${body.mediaKind}`);
        return { data: { upload, partSize: PART, partCount: 2 }, etag: null };
      }),
      getUpload: vi.fn(async (id: string) => ({
        data: {
          upload: uploadRecord(id, "initiated"),
          completedParts: [],
          acceptingParts: true,
          assembled: false,
        },
        etag: null,
      })),
    });
    const file = fakeFile(size, "plan.jpg");
    const photo = new MultipartUpload("p1", file, "photo", "image/jpeg", {
      api,
      putPart: async () => {
        photo.pause();
        throw new DOMException("Aborted", "AbortError");
      },
      storage,
      concurrency: 1,
    });
    await expect(photo.start()).rejects.toBeInstanceOf(UploadAbortedError);

    const plan = new MultipartUpload("p1", file, "floor_plan", "image/jpeg", {
      api,
      putPart: async (url) => `"e${partOf(url)}"`,
      storage,
    });
    const uploaded = await plan.start();
    expect(created).toEqual(["photo", "floor_plan"]);
    expect(uploaded.id).toBe("u-floor_plan");
    // The paused photo upload keeps its own record, and resumes as the photo it is.
    expect(
      loadResumeRecord("p1", uploadIdentity(file, "photo", "image/jpeg"), { storage })?.uploadId,
    ).toBe("u-photo");
  });

  it("starts a new upload when the saved one no longer exists on the server", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const file = fakeFile(size);
    saveResumeRecord(
      {
        version: RESUME_RECORD_VERSION,
        uploadId: "u-gone",
        projectId: "p1",
        fingerprint: fileFingerprint(video(file)),
        sample: (await contentSample(file, PART))!,
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
    expect(storage.getItem(resumeKey("p1", fileFingerprint(video(file))))).toBeNull();
  });

  it("treats an upload the server already completed as done after a reload", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const file = fakeFile(size);
    saveResumeRecord(
      {
        version: RESUME_RECORD_VERSION,
        uploadId: "u-done",
        projectId: "p1",
        fingerprint: fileFingerprint(video(file)),
        sample: (await contentSample(file, PART))!,
        partSize: PART,
        partCount: 2,
        completedParts: [{ partNumber: 1, etag: '"x"' }],
        expiresAt: FUTURE,
      },
      storage,
    );
    const { api } = fakeApi(size, {
      getUpload: vi.fn(async () => ({
        data: {
          upload: uploadRecord("u-done", "uploaded_unverified"),
          completedParts: [],
          acceptingParts: false,
          assembled: false,
        },
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
    expect(storage.getItem(resumeKey("p1", fileFingerprint(video(file))))).toBeNull();
  });

  it("starts a new upload for another file with the same name, size and date", async () => {
    const size = 3 * PART;
    const storage = new MemoryStorage();
    const first = fakeFile(size, "walk.mp4", 1);
    saveResumeRecord(
      {
        version: RESUME_RECORD_VERSION,
        uploadId: "u-old",
        projectId: "p1",
        fingerprint: fileFingerprint(video(first)),
        sample: (await contentSample(first, PART))!,
        partSize: PART,
        partCount: 3,
        completedParts: [{ partNumber: 1, etag: '"x"' }],
        expiresAt: FUTURE,
      },
      storage,
    );
    // The same identity, different bytes: its upload must be neither reused nor mixed.
    const other = fakeFile(size, "walk.mp4", 2);
    expect(loadResumeRecord("p1", video(other), { storage })?.uploadId).toBe("u-old");
    const { api } = fakeApi(size);
    const uploaded: number[] = [];
    const upload = new MultipartUpload("p1", other, "video", "video/mp4", {
      api,
      putPart: async (url) => {
        uploaded.push(partOf(url));
        return `"e${partOf(url)}"`;
      },
      storage,
    });
    const result = await upload.start();
    expect(result.id).toBe("u-new");
    expect(api.getUpload).not.toHaveBeenCalled();
    expect(api.createUpload).toHaveBeenCalledTimes(1);
    expect(uploaded.sort()).toEqual([1, 2, 3]);
    expect(upload.state.resumed).toBe(false);
  });

  it("does not take an upload the server completed for another file as done", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const first = fakeFile(size, "walk.mp4", 1);
    saveResumeRecord(
      {
        version: RESUME_RECORD_VERSION,
        uploadId: "u-done",
        projectId: "p1",
        fingerprint: fileFingerprint(video(first)),
        sample: (await contentSample(first, PART))!,
        partSize: PART,
        partCount: 2,
        completedParts: [],
        expiresAt: FUTURE,
      },
      storage,
    );
    const { api } = fakeApi(size, {
      getUpload: vi.fn(async () => ({
        data: {
          upload: uploadRecord("u-done", "uploaded_unverified"),
          completedParts: [],
          acceptingParts: false,
          assembled: false,
        },
        etag: null,
      })),
    });
    const upload = new MultipartUpload("p1", fakeFile(size, "walk.mp4", 2), "video", "video/mp4", {
      api,
      putPart: async (url) => `"e${partOf(url)}"`,
      storage,
    });
    expect((await upload.start()).id).toBe("u-new");
    expect(api.getUpload).not.toHaveBeenCalled();
    expect(api.completeUpload).toHaveBeenCalledWith("u-new");
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
  it("honours a pause made while the upload is still being created", async () => {
    const size = 3 * PART;
    const server = fakeApi(size).api;
    let upload: MultipartUpload | null = null;
    const { api } = fakeApi(size, {
      createUpload: vi.fn(async (...args: Parameters<UploaderApi["createUpload"]>) => {
        upload!.pause(); // the user pauses before the server answers
        return server.createUpload(...args);
      }),
    });
    const putPart = vi.fn<PutPart>(async (url) => `"e${partOf(url)}"`);
    upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api,
      putPart,
      storage: null,
      concurrency: 2,
    });
    await expect(upload.start()).rejects.toBeInstanceOf(UploadAbortedError);
    expect(upload.state.phase).toBe("paused");
    expect(putPart).not.toHaveBeenCalled();
    await upload.start(); // resuming uses the upload that was created
    expect(api.createUpload).toHaveBeenCalledTimes(1);
    expect(putPart).toHaveBeenCalledTimes(3);
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

  it("keeps a paused upload the server still accepts during the expiry margin", async () => {
    const size = 3 * PART;
    const { api } = fakeApi(size, {
      getUpload: vi.fn(async (uploadId: string) => ({
        data: {
          upload: uploadRecord(uploadId),
          completedParts: [{ partNumber: 1, etag: '"e1"', size: PART }],
          acceptingParts: true,
          assembled: false,
        },
        etag: null,
      })),
    });
    let clock = Date.now();
    let block = true;
    let upload: MultipartUpload | null = null;
    const putPart = vi.fn<PutPart>(async (url, _b, signal) => {
      if (block && partOf(url) === 2) {
        upload!.pause();
        if (signal.aborted) throw new DOMException("Aborted", "AbortError");
      }
      return `"e${partOf(url)}"`;
    });
    upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api,
      putPart,
      storage: null,
      concurrency: 1,
      now: () => clock,
    });
    await expect(upload.start()).rejects.toBeInstanceOf(UploadAbortedError);
    // Four minutes before its expiry the record alone is no longer trusted, but the
    // server still accepts the upload, which holds the project's quota: it is kept.
    clock = Date.parse(FUTURE) - 4 * 60 * 1000;
    block = false;
    await upload.start();
    expect(api.getUpload).toHaveBeenCalledTimes(1);
    expect(api.createUpload).toHaveBeenCalledTimes(1);
    expect(upload.state.phase).toBe("completed");
  });

  it("resumes an upload the server still accepts when this device's clock is fast", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const file = fakeFile(size);
    const expiresAt = new Date(Date.now() + 3600 * 1000).toISOString();
    saveResumeRecord(
      {
        version: RESUME_RECORD_VERSION,
        uploadId: "u-open",
        projectId: "p1",
        fingerprint: fileFingerprint(video(file)),
        sample: (await contentSample(file, PART))!,
        partSize: PART,
        partCount: 2,
        completedParts: [{ partNumber: 1, etag: '"e1"' }],
        expiresAt,
      },
      storage,
    );
    const { api } = fakeApi(size, {
      getUpload: vi.fn(async () => ({
        data: {
          upload: { ...uploadRecord("u-open"), expiresAt },
          completedParts: [{ partNumber: 1, etag: '"e1"', size: PART }],
          acceptingParts: true,
          assembled: false,
        },
        etag: null,
      })),
    });
    const uploaded: number[] = [];
    const upload = new MultipartUpload("p1", file, "video", "video/mp4", {
      api,
      putPart: async (url) => {
        uploaded.push(partOf(url));
        return `"e${partOf(url)}"`;
      },
      storage,
      // By this device's clock the upload expired a minute ago; the server, which decides,
      // still accepts it, and it holds the project's quota.
      now: () => Date.parse(expiresAt) + 60 * 1000,
    });
    const result = await upload.start();
    expect(result.id).toBe("u-open");
    expect(api.createUpload).not.toHaveBeenCalled();
    expect(uploaded).toEqual([2]);
    expect(upload.state.resumed).toBe(true);
  });

  it("completes a saved upload whose parts storage already joined", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const file = fakeFile(size);
    saveResumeRecord(
      {
        version: RESUME_RECORD_VERSION,
        uploadId: "u-joined",
        projectId: "p1",
        fingerprint: fileFingerprint(video(file)),
        sample: (await contentSample(file, PART))!,
        partSize: PART,
        partCount: 2,
        completedParts: [{ partNumber: 1, etag: '"e1"' }],
        expiresAt: FUTURE,
      },
      storage,
    );
    // A completion joined the parts and then failed before the server recorded it.
    const { api } = fakeApi(size, {
      getUpload: vi.fn(async () => ({
        data: {
          upload: uploadRecord("u-joined"),
          completedParts: [],
          acceptingParts: false,
          assembled: true,
        },
        etag: null,
      })),
    });
    const putPart = vi.fn<PutPart>(async () => '"e"');
    const upload = new MultipartUpload("p1", file, "video", "video/mp4", {
      api,
      putPart,
      storage,
    });
    expect((await upload.start()).id).toBe("u-joined");
    expect(api.createUpload).not.toHaveBeenCalled();
    expect(api.requestPartUrls).not.toHaveBeenCalled();
    expect(putPart).not.toHaveBeenCalled();
    expect(api.completeUpload).toHaveBeenCalledWith("u-joined");
    expect(upload.state.phase).toBe("completed");
  });

  it("replaces a saved upload the server no longer accepts, whatever this clock says", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const file = fakeFile(size);
    saveResumeRecord(
      {
        version: RESUME_RECORD_VERSION,
        uploadId: "u-over",
        projectId: "p1",
        fingerprint: fileFingerprint(video(file)),
        sample: (await contentSample(file, PART))!,
        partSize: PART,
        partCount: 2,
        completedParts: [{ partNumber: 1, etag: '"e1"' }],
        expiresAt: FUTURE,
      },
      storage,
    );
    // Still "initiated" until the worker marks it expired, but over by the server's clock.
    const { api } = fakeApi(size, {
      getUpload: vi.fn(async () => ({
        data: {
          upload: uploadRecord("u-over"),
          completedParts: [{ partNumber: 1, etag: '"e1"', size: PART }],
          acceptingParts: false,
          assembled: false,
        },
        etag: null,
      })),
    });
    const upload = new MultipartUpload("p1", file, "video", "video/mp4", {
      api,
      putPart: async (url) => `"e${partOf(url)}"`,
      storage,
    });
    expect((await upload.start()).id).toBe("u-new");
    expect(api.getUpload).toHaveBeenCalledTimes(1);
    expect(api.createUpload).toHaveBeenCalledTimes(1);
  });

  it("aborts an upload whose creation answers after its file was removed", async () => {
    const size = 3 * PART;
    const server = fakeApi(size).api;
    const storage = new MemoryStorage();
    let upload: MultipartUpload | null = null;
    const { api } = fakeApi(size, {
      createUpload: vi.fn(async (...args: Parameters<UploaderApi["createUpload"]>) => {
        upload!.discard({ abortOnServer: true }); // the user removes the row first
        return server.createUpload(...args);
      }),
    });
    const putPart = vi.fn<PutPart>(async (url) => `"e${partOf(url)}"`);
    upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api,
      putPart,
      storage,
    });
    await expect(upload.start()).rejects.toBeInstanceOf(UploadAbortedError);
    // Created once the row was gone: aborted, so it does not hold quota until it
    // expires, and nothing that could find it again is kept.
    expect(api.abortUpload).toHaveBeenCalledWith("u-new");
    expect(putPart).not.toHaveBeenCalled();
    expect(storage.length).toBe(0);
    await expect(upload.start()).rejects.toBeInstanceOf(UploadAbortedError);
    expect(api.createUpload).toHaveBeenCalledTimes(1);
  });

  it("aborts a removed file's unfinished upload only when asked", async () => {
    const size = 3 * PART;
    for (const abortOnServer of [true, false]) {
      const { api } = fakeApi(size);
      const storage = new MemoryStorage();
      let upload: MultipartUpload | null = null;
      const putPart: PutPart = async (url, _b, signal) => {
        if (partOf(url) === 2) {
          upload!.pause();
          if (signal.aborted) throw new DOMException("Aborted", "AbortError");
        }
        return `"e${partOf(url)}"`;
      };
      upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
        api,
        putPart,
        storage,
        concurrency: 1,
      });
      await expect(upload.start()).rejects.toBeInstanceOf(UploadAbortedError);
      expect(storage.length).toBe(1); // the resume record
      upload.discard({ abortOnServer });
      expect(api.abortUpload).toHaveBeenCalledTimes(abortOnServer ? 1 : 0);
      expect(storage.length).toBe(0);
    }
  });

  it("never reuses part URLs signed for an upload it replaced", async () => {
    const size = 3 * PART;
    const start = Date.now();
    let clock = start;
    let created = 0;
    const urlsExpire = new Date(start + 3600 * 1000).toISOString(); // still fresh later
    const { api } = fakeApi(size, {
      // Each upload expires in 10 minutes and gets its own ID.
      createUpload: vi.fn(async () => {
        created += 1;
        const record = uploadRecord(`u-${created}`);
        record.expiresAt = new Date(clock + 10 * 60 * 1000).toISOString();
        return { data: { upload: record, partSize: PART, partCount: 3 }, etag: null };
      }),
      requestPartUrls: vi.fn(async (uploadId: string, partNumbers: number[]) => ({
        data: {
          urls: partNumbers.map((n) => ({
            partNumber: n,
            url: `https://storage.test/${uploadId}/${n}`,
            bytes: Math.min(PART, size - (n - 1) * PART),
          })),
          expiresAt: urlsExpire,
        },
        etag: null,
      })),
    });
    let block = true;
    let upload: MultipartUpload | null = null;
    const putUrls: string[] = [];
    const putPart: PutPart = async (url, _b, signal) => {
      if (block && partOf(url) === 2) {
        upload!.pause();
        if (signal.aborted) throw new DOMException("Aborted", "AbortError");
      }
      putUrls.push(url);
      return `"e${partOf(url)}"`;
    };
    upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api,
      putPart,
      storage: null,
      concurrency: 1,
      urlBatchSize: 3,
      now: () => clock,
    });
    await expect(upload.start()).rejects.toBeInstanceOf(UploadAbortedError);
    // Within five minutes of the upload's expiry, though its part URLs are fresh.
    clock += 6 * 60 * 1000;
    block = false;
    putUrls.length = 0;
    await upload.start();
    expect(api.createUpload).toHaveBeenCalledTimes(2);
    expect(putUrls.sort()).toEqual([1, 2, 3].map((n) => `https://storage.test/u-2/${n}`));
    expect(api.completeUpload).toHaveBeenCalledWith("u-2");
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

  it("retries a creation whose response was lost with the same key", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const keys: (string | undefined)[] = [];
    let lose = true;
    const { api } = fakeApi(size, {
      createUpload: vi.fn(async (_p: string, _b: unknown, key?: string) => {
        keys.push(key);
        // The server created the upload, but the response never arrived.
        if (lose) throw new TypeError("Failed to fetch");
        return { data: { upload: uploadRecord("u-1"), partSize: PART, partCount: 2 }, etag: null };
      }),
    });
    // Each uploader is on a page of its own, as after a reload.
    const make = () =>
      new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
        api,
        putPart: async (url) => `"e${partOf(url)}"`,
        storage,
        maxAttempts: 1,
        sleep: noSleep,
        heldCreationKeys: new Set(),
      });
    await expect(make().start()).rejects.toBeInstanceOf(TypeError);
    lose = false;
    // A reload builds a new uploader; the saved key still replays the same upload.
    await make().start();
    expect(keys.length).toBe(2);
    expect(keys[0]).toMatch(/^[A-Za-z0-9._:-]{8,255}$/);
    expect(keys[1]).toBe(keys[0]);
    // Answered: the next upload of this file gets a new key.
    removeAllResumeState(storage);
    await make().start();
    expect(keys[2]).not.toBe(keys[0]);
  });

  it("gives files that share a name, size and date keys of their own", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const page = new Set<string>();
    const keys: (string | undefined)[] = [];
    const { api } = fakeApi(size, {
      createUpload: vi.fn(async (_p: string, _b: unknown, key?: string) => {
        keys.push(key);
        const upload = uploadRecord(`u-${keys.length}`);
        return { data: { upload, partSize: PART, partCount: 2 }, etag: null };
      }),
    });
    const make = (fill: number) =>
      new MultipartUpload("p1", fakeFile(size, "walk.mp4", fill), "video", "video/mp4", {
        api,
        putPart: async (url) => `"e${partOf(url)}"`,
        storage,
        sleep: noSleep,
        heldCreationKeys: page,
      });
    // Different bytes behind one name, size and date, then the same file selected twice:
    // one key would make the API answer them with one upload, and all would upload into it.
    const uploads = await Promise.all([make(1), make(2), make(1), make(1)].map((u) => u.start()));
    expect(new Set(keys).size).toBe(4);
    expect(new Set(uploads.map((u) => u.id)).size).toBe(4);
  });

  it("replays a lost creation after a reload only for the same bytes", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const keys: (string | undefined)[] = [];
    let lose = true;
    const { api } = fakeApi(size, {
      createUpload: vi.fn(async (_p: string, _b: unknown, key?: string) => {
        keys.push(key);
        if (lose) throw new TypeError("Failed to fetch");
        return { data: { upload: uploadRecord("u-1"), partSize: PART, partCount: 2 }, etag: null };
      }),
    });
    const make = (fill: number) =>
      new MultipartUpload("p1", fakeFile(size, "walk.mp4", fill), "video", "video/mp4", {
        api,
        putPart: async (url) => `"e${partOf(url)}"`,
        storage,
        maxAttempts: 1,
        sleep: noSleep,
        heldCreationKeys: new Set(), // each on a page of its own, as after a reload
      });
    await expect(make(1).start()).rejects.toBeInstanceOf(TypeError);
    lose = false;
    await make(2).start(); // another file with the same name, size and date
    expect(keys[1]).not.toBe(keys[0]);
    await make(1).start(); // the same file: its saved key gets the lost upload back
    expect(keys[2]).toBe(keys[0]);
  });

  it("aborts a removed file's upload created just before its sample failed", async () => {
    const size = 2 * PART;
    let unreadable = false;
    const { api } = fakeApi(size, {
      createUpload: vi.fn(async () => {
        unreadable = true; // the file can no longer be read once the upload exists
        return { data: { upload: uploadRecord("u-1"), partSize: PART, partCount: 2 }, etag: null };
      }),
    });
    const file = fakeFile(size);
    const upload = new MultipartUpload(
      "p1",
      {
        ...file,
        slice: (start, end) => {
          if (unreadable) throw new DOMException("The file could not be read.", "NotReadableError");
          return file.slice(start, end);
        },
      },
      "video",
      "video/mp4",
      { api, storage: new MemoryStorage(), sleep: noSleep, heldCreationKeys: new Set() },
    );
    await expect(upload.start()).rejects.toBeInstanceOf(DOMException);
    expect(upload.state.uploadId).toBe("u-1");
    upload.discard({ abortOnServer: true }); // the user removes the failed row
    expect(api.abortUpload).toHaveBeenCalledWith("u-1");
  });

  it("keeps the creation key until the new upload's record is saved", async () => {
    const size = 2 * PART;
    const storage = new MemoryStorage();
    const keys: (string | undefined)[] = [];
    let unreadable = false;
    const { api } = fakeApi(size, {
      createUpload: vi.fn(async (_p: string, _b: unknown, key?: string) => {
        keys.push(key);
        unreadable = keys.length === 1; // unreadable just after the first creation
        return { data: { upload: uploadRecord("u-1"), partSize: PART, partCount: 2 }, etag: null };
      }),
    });
    const file = fakeFile(size);
    const make = () =>
      new MultipartUpload(
        "p1",
        {
          ...file,
          // The file cannot be read for the record's sample (or the page closes then).
          slice: (start, end) => {
            if (unreadable)
              throw new DOMException("The file could not be read.", "NotReadableError");
            return file.slice(start, end);
          },
        },
        "video",
        "video/mp4",
        {
          api,
          putPart: async (url) => `"e${partOf(url)}"`,
          storage,
          sleep: noSleep,
          heldCreationKeys: new Set(), // a page of its own, as after a reload
        },
      );
    await expect(make().start()).rejects.toBeInstanceOf(DOMException);
    unreadable = false;
    // The server created u-1, and the same key gets it back instead of another upload.
    const upload = await make().start();
    expect(upload.id).toBe("u-1");
    expect(keys[1]).toBe(keys[0]);
    expect([...storage.map.keys()].some((k) => k.startsWith("andarilho:upload-create:"))).toBe(
      false,
    );
  });

  it("forgets a creation key it could not replace with a record once the upload is done", async () => {
    const size = PART;
    const keys: (string | undefined)[] = [];
    const { api } = fakeApi(size, {
      createUpload: vi.fn(async (_p: string, _b: unknown, key?: string) => {
        keys.push(key);
        return { data: { upload: uploadRecord("u-1"), partSize: PART, partCount: 1 }, etag: null };
      }),
    });
    // Resume records cannot be stored, so the creation key stays while the upload runs.
    const storage = new MemoryStorage();
    const setItem = storage.setItem.bind(storage);
    storage.setItem = (key: string, value: string) => {
      if (key.startsWith("andarilho:upload:")) throw new Error("QuotaExceededError");
      setItem(key, value);
    };
    const make = () =>
      new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
        api,
        putPart: async (url) => `"e${partOf(url)}"`,
        storage,
        sleep: noSleep,
      });
    await make().start();
    // Done: the next upload of this file is a new upload with a new key.
    await make().start();
    expect(keys.length).toBe(2);
    expect(keys[1]).not.toBe(keys[0]);
  });

  it("does not reuse the key of a creation the server refused", async () => {
    const size = PART;
    const keys: (string | undefined)[] = [];
    let refuse = true;
    const { api } = fakeApi(size, {
      createUpload: vi.fn(async (_p: string, _b: unknown, key?: string) => {
        keys.push(key);
        if (refuse) {
          throw new ApiError(413, { code: "QUOTA_EXCEEDED", message: "Full.", retryable: false });
        }
        return { data: { upload: uploadRecord("u-1"), partSize: PART, partCount: 1 }, etag: null };
      }),
    });
    const upload = new MultipartUpload("p1", fakeFile(size), "video", "video/mp4", {
      api,
      putPart: async (url) => `"e${partOf(url)}"`,
      storage: new MemoryStorage(),
      sleep: noSleep,
    });
    await expect(upload.start()).rejects.toBeInstanceOf(ApiError);
    refuse = false;
    await upload.start();
    expect(keys[1]).not.toBe(keys[0]);
  });
});

/** Drops resume records so the next start creates a new upload. */
function removeAllResumeState(storage: MemoryStorage) {
  for (let i = storage.length - 1; i >= 0; i -= 1) {
    const key = storage.key(i);
    if (key?.startsWith("andarilho:upload:")) storage.removeItem(key);
  }
}
