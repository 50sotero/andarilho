// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CaptureFlow } from "@/features/capture/CaptureFlow";
import {
  SCENE_ANALYSIS_NOTICE,
  sceneAnalysisNoticeShown,
} from "@/features/capture/SceneAnalysisNotice";
import { useProject } from "@/features/projects/use-project";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));
vi.mock("@/features/projects/use-project", () => ({ useProject: vi.fn() }));
vi.mock("@/lib/api/client", () => ({
  api: { getIntakePolicy: vi.fn(() => new Promise(() => undefined)) },
  newIdempotencyKey: () => "key",
}));

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  vi.mocked(useProject).mockReturnValue({
    detail: null,
    etag: null,
    error: null,
    loading: false,
    reload: async () => {},
  });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  vi.unstubAllEnvs();
  await act(async () => {
    root.unmount();
  });
  container.remove();
});

async function renderCapture() {
  await act(async () => {
    root.render(<CaptureFlow projectId="project-1" />);
  });
}

describe("sceneAnalysisNoticeShown", () => {
  it("shows the notice only for the exact flag value true", () => {
    expect(sceneAnalysisNoticeShown("true")).toBe(true);
    expect(sceneAnalysisNoticeShown("false")).toBe(false);
    expect(sceneAnalysisNoticeShown("1")).toBe(false);
    expect(sceneAnalysisNoticeShown("")).toBe(false);
    expect(sceneAnalysisNoticeShown(undefined)).toBe(false);
  });
});

describe("the AI scene analysis notice on the capture page", () => {
  it("is shown after the file inputs and before the upload button when scene analysis is on", async () => {
    vi.stubEnv("NEXT_PUBLIC_AI_SCENE_ANALYSIS", "true");
    await renderCapture();
    const notice = [...container.querySelectorAll("p")].find(
      (p) => p.textContent === SCENE_ANALYSIS_NOTICE,
    );
    expect(notice).toBeDefined();
    expect(SCENE_ANALYSIS_NOTICE).toContain("DeepSeek, through OpenCode");
    const input = container.querySelector('input[type="file"]')!;
    const upload = [...container.querySelectorAll("button")].find(
      (b) => b.textContent === "Upload selected files",
    )!;
    // DOCUMENT_POSITION_FOLLOWING: the later node comes after the earlier one.
    expect(input.compareDocumentPosition(notice!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(notice!.compareDocumentPosition(upload) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("is not shown when scene analysis is off", async () => {
    vi.stubEnv("NEXT_PUBLIC_AI_SCENE_ANALYSIS", "false");
    await renderCapture();
    expect(container.textContent).not.toContain("sent to an AI model");
    expect(container.textContent).toContain("Walkthrough video");
  });
});

describe("the upload button", () => {
  it("is disabled while a selection-wide error is listed, not only a file's", async () => {
    await renderCapture();
    const photos = container.querySelector<HTMLInputElement>('input[type="file"][multiple]')!;
    // Well-formed photos, but fewer than a set without a video needs.
    const files = Array.from(
      { length: 5 },
      (_, i) => new File(["x"], `photo-${i}.jpg`, { type: "image/jpeg" }),
    );
    Object.defineProperty(photos, "files", { value: files, configurable: true });
    await act(async () => {
      photos.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(container.textContent).toContain("Without a video, add at least");
    const upload = [...container.querySelectorAll("button")].find(
      (b) => b.textContent === "Upload selected files",
    )!;
    expect(upload.disabled).toBe(true);
  });
});
