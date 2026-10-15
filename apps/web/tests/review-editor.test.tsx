// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useProject } from "@/features/projects/use-project";
import { ReviewEditor } from "@/features/review/ReviewEditor";
import { api } from "@/lib/api/client";
import type {
  CaptureEvidenceLink,
  Furniture,
  Layout,
  ProjectDetail,
  SceneDetail,
} from "@/lib/api/types";
import fixture from "@andarilho/scene-contract/fixtures/synthetic-two-room.layout.json";

vi.mock("@/features/projects/use-project", () => ({ useProject: vi.fn() }));
vi.mock("@/lib/api/client", () => ({
  api: { getScene: vi.fn(), getDelivery: vi.fn(), saveLayout: vi.fn() },
}));
vi.mock("@/components/viewer/ViewerClient", () => ({ ViewerClient: () => null }));

const layout = fixture as unknown as Layout;
const evidenceLink: CaptureEvidenceLink = {
  url: "https://storage.test/evidence.json",
  expiresAt: new Date(Date.now() + 10 * 60_000).toISOString(),
};
// Two points are inside the 0.1 to 2.2 m band at scale 1; at scale 2 only the first is.
// The camera path is three positions along x.
const evidenceBody = {
  schemaVersion: 1,
  units: "m",
  points: [
    [1, 0.5, 2],
    [1, 0.02, 2],
    [3, 1.2, 4],
    [5, 2.5, 6],
  ],
  cameras: [
    [0, 1.5, 0],
    [1, 1.5, 0],
    [2, 1.5, 0],
  ],
};

function sceneDetail(
  captureEvidence: CaptureEvidenceLink | null,
  { renderScale, furniture }: { renderScale?: number; furniture?: Furniture[] } = {},
): SceneDetail {
  let shown: Layout = layout;
  if (renderScale !== undefined) {
    shown = { ...shown, calibration: { method: "known_distance", renderScale } };
  }
  if (furniture !== undefined) shown = { ...shown, furniture };
  const createdAt = "2026-10-15T12:00:00Z";
  return {
    scene: {
      id: "scene-1",
      projectId: "project-1",
      runId: "run-1",
      version: 1,
      currentDraftRevisionId: "rev-1",
    },
    revision: {
      id: "rev-1",
      revision: 1,
      layoutRevision: 1,
      layout: shown,
      manifest: null,
      validationState: "valid",
      validationFindings: [],
      approvalState: "draft",
      approvedAt: null,
      createdAt,
    },
    revisions: [
      {
        id: "rev-1",
        revision: 1,
        layoutRevision: 1,
        validationState: "valid",
        approvalState: "draft",
        createdAt,
      },
    ],
    floorPlan: null,
    captureEvidence,
  } as unknown as SceneDetail;
}

const projectState = {
  detail: {
    project: { id: "project-1", title: "Flat", role: "owner" },
    scene: { id: "scene-1", version: 1, currentDraftRevisionId: "rev-1" },
  } as unknown as ProjectDetail,
  etag: null,
  error: null,
  loading: false,
  reload: async () => {},
};

let container: HTMLDivElement;
let root: Root;

/** Let the editor's scene and evidence requests settle. */
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

async function renderEditor(scene: SceneDetail) {
  vi.mocked(api.getScene).mockResolvedValue({ data: scene, etag: '"1"' });
  await act(async () => {
    root.render(<ReviewEditor projectId="project-1" />);
  });
  await settle();
}

function button(text: string): HTMLButtonElement | undefined {
  return [...container.querySelectorAll("button")].find((b) => b.textContent === text);
}

/** The checkbox whose label reads `text`; the editor pairs them by id. */
function checkbox(text: string): HTMLInputElement | null {
  const label = [...container.querySelectorAll("label")].find((l) => l.textContent === text);
  return label ? (document.getElementById(label.htmlFor) as HTMLInputElement | null) : null;
}

const evidenceLayer = () => container.querySelector('[data-layer="capture-evidence"]');
const dots = () =>
  container.querySelectorAll('[data-layer="capture-evidence"] circle[fill="#0284c7"]');

async function click(element: HTMLElement) {
  await act(async () => {
    element.click();
  });
}

beforeEach(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  vi.mocked(useProject).mockReturnValue(projectState);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  container.remove();
  vi.unstubAllGlobals();
});

describe("capture evidence in the review editor", () => {
  it("draws the evidence by default and hides it while the toggle is off", async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, json: async () => evidenceBody }));
    vi.stubGlobal("fetch", fetchMock);
    await renderEditor(sceneDetail(evidenceLink));

    expect(fetchMock).toHaveBeenCalledWith(evidenceLink.url, { credentials: "omit" });
    expect(evidenceLayer()).not.toBeNull();
    expect(dots()).toHaveLength(2);
    expect(checkbox("Capture evidence")?.checked).toBe(true);

    await click(checkbox("Capture evidence")!);
    expect(evidenceLayer()).toBeNull();
    expect(checkbox("Capture evidence")?.checked).toBe(false);

    await click(checkbox("Capture evidence")!);
    expect(evidenceLayer()).not.toBeNull();
  });

  it("scales the drawn evidence by the layout's render scale", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, json: async () => evidenceBody })),
    );
    await renderEditor(sceneDetail(evidenceLink, { renderScale: 2 }));

    const plotted = dots();
    expect(plotted).toHaveLength(1);
    expect(plotted[0]!.getAttribute("cx")).toBe("2");
    expect(plotted[0]!.getAttribute("cy")).toBe("4");
    const path = evidenceLayer()!.querySelector("polyline");
    expect(path?.getAttribute("points")).toBe("0,0 2,0 4,0");
  });

  it("shows no layer and no toggle, and downloads nothing, without captureEvidence", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    await renderEditor(sceneDetail(null));

    expect(evidenceLayer()).toBeNull();
    expect(checkbox("Capture evidence")).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("stays quiet when the evidence download fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    await renderEditor(sceneDetail(evidenceLink));

    expect(evidenceLayer()).toBeNull();
    expect(checkbox("Capture evidence")).toBeNull();
    expect(container.textContent).toContain("Revision 1");
    expect(container.textContent).not.toMatch(/went wrong|Network problem/);
  });

  it("downloads the evidence once per scene, not again when the layout reloads", async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, json: async () => evidenceBody }));
    vi.stubGlobal("fetch", fetchMock);
    await renderEditor(sceneDetail(evidenceLink));
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      button("Discard changes")!.click();
    });
    await settle();
    expect(api.getScene).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(evidenceLayer()).not.toBeNull();
  });
});

describe("furniture in the review editor", () => {
  const chair: Furniture = {
    id: "chair-0",
    floorId: "floor-0",
    category: "chair",
    label: "Wooden chair",
    position: [-1, 1.5],
    yawRadians: 0,
    size: [0.5, 0.9, 0.5],
    solid: true,
    parts: [{ shape: "box", min: [0, 0, 0], max: [1, 1, 1], color: "#aa3333" }],
  };
  const rug: Furniture = { ...chair, id: "rug-0", label: "Rug", solid: false, position: [2, -1.5] };

  it("draws every piece and lets the user select one and remove it", async () => {
    await renderEditor(sceneDetail(null, { furniture: [chair, rug] }));

    expect(container.querySelector('[data-furniture="chair-0"]')).not.toBeNull();
    expect(container.querySelector('[data-furniture="rug-0"]')).not.toBeNull();

    const chairPolygon = container.querySelector('[data-furniture="chair-0"]')!;
    await act(async () => {
      chairPolygon.dispatchEvent(new Event("pointerdown", { bubbles: true }));
    });
    expect(container.textContent).toContain("Furniture Wooden chair");

    await click(button("Remove")!);
    expect(container.querySelector('[data-furniture="chair-0"]')).toBeNull();
    expect(container.querySelector('[data-furniture="rug-0"]')).not.toBeNull();
  });

  it("saves the layout without the removed piece", async () => {
    vi.stubGlobal("fetch", vi.fn());
    vi.mocked(api.saveLayout).mockResolvedValue({
      data: { sceneRevisionId: "rev-2", jobId: "job-2", version: 2 },
      etag: '"2"',
    });
    await renderEditor(sceneDetail(null, { furniture: [chair, rug] }));

    await act(async () => {
      container
        .querySelector('[data-furniture="chair-0"]')!
        .dispatchEvent(new Event("pointerdown", { bubbles: true }));
    });
    await click(button("Remove")!);
    await click(button("Save as new revision")!);

    expect(api.saveLayout).toHaveBeenCalledTimes(1);
    const saved = vi.mocked(api.saveLayout).mock.calls[0]![2];
    expect(saved.furniture?.map((f) => f.id)).toEqual(["rug-0"]);
  });

  it("explains that the scene is walkable and that approval comes before publishing", async () => {
    vi.stubGlobal("fetch", vi.fn());
    await renderEditor(sceneDetail(null));
    expect(container.textContent).toContain("approval is needed before publishing");
    expect(container.textContent).not.toContain("walking with collision");
  });
});
