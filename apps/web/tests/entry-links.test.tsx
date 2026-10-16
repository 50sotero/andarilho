// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { linkButtonClass, primaryLinkClass } from "@/components/ui";
import { ProcessingView } from "@/features/processing/ProcessingView";
import { ProjectDetails } from "@/features/projects/ProjectDetails";
import { useProject } from "@/features/projects/use-project";
import { api } from "@/lib/api/client";
import type { ProjectDetail, Role, Run } from "@/lib/api/types";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));
vi.mock("@/features/projects/use-project", () => ({ useProject: vi.fn() }));
vi.mock("@/features/imports/ImportsPanel", () => ({ ImportsPanel: () => null }));
vi.mock("@/features/sharing/SharePanel", () => ({ SharePanel: () => null }));
vi.mock("@/lib/api/client", () => ({
  api: {
    getProject: vi.fn(),
    getScene: vi.fn(() => new Promise(() => undefined)),
  },
  newIdempotencyKey: () => "key",
}));

const succeededRun = {
  id: "run-1",
  status: "succeeded",
  error: null,
  cancelRequested: false,
  currentJobId: null,
  currentStage: null,
  createdAt: "2026-10-15T12:00:00Z",
  jobs: [],
} as unknown as Run;

function detailFor(role: Role, runs: Run[] = []): ProjectDetail {
  return {
    project: {
      id: "project-1",
      title: "Flat",
      lifecycle: "active",
      version: 1,
      role,
      activeDraftRevisionId: "rev-1",
      publishedRevisionId: null,
      createdAt: "2026-10-15T12:00:00Z",
      updatedAt: "2026-10-15T12:00:00Z",
    },
    uploads: [],
    captureSets: [],
    runs,
    scene: { id: "scene-1", version: 1, currentDraftRevisionId: "rev-1" },
    shares: [],
  } as unknown as ProjectDetail;
}

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  container.remove();
});

/** Let the first poll and any state it sets settle. */
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

function link(text: string): HTMLAnchorElement | undefined {
  return [...container.querySelectorAll("a")].find((a) => a.textContent === text);
}

describe("processing page for a finished run", () => {
  it("makes walking the scene the primary action and adjusting the layout secondary", async () => {
    const detail = detailFor("owner", [succeededRun]);
    vi.mocked(api.getProject).mockResolvedValue({ data: detail, etag: '"1"' } as never);
    await act(async () => {
      root.render(<ProcessingView projectId="project-1" initialJobId={null} />);
    });
    await settle();

    const walk = link("Walk the scene");
    const adjust = link("Adjust layout");
    expect(walk?.getAttribute("href")).toBe("/projects/project-1/view");
    expect(walk?.className).toBe(primaryLinkClass);
    expect(adjust?.getAttribute("href")).toBe("/projects/project-1/review");
    expect(adjust?.className).toBe(linkButtonClass);
    expect(link("Review the scene")).toBeUndefined();
  });

  it("offers read-only members neither link", async () => {
    const detail = detailFor("viewer", [succeededRun]);
    vi.mocked(api.getProject).mockResolvedValue({ data: detail, etag: '"1"' } as never);
    await act(async () => {
      root.render(<ProcessingView projectId="project-1" initialJobId={null} />);
    });
    await settle();

    expect(link("Walk the scene")).toBeUndefined();
    expect(link("Adjust layout")).toBeUndefined();
  });
});

describe("project scene card", () => {
  it("walks privately first, and adjusts the layout second, for owners", async () => {
    vi.mocked(useProject).mockReturnValue({
      detail: detailFor("owner"),
      etag: '"1"',
      error: null,
      loading: false,
      reload: async () => {},
    } as never);
    await act(async () => {
      root.render(<ProjectDetails projectId="project-1" />);
    });
    await settle();

    const walk = link("Walk privately");
    const adjust = link("Adjust layout");
    expect(walk?.getAttribute("href")).toBe("/projects/project-1/view");
    expect(walk?.className).toBe(primaryLinkClass);
    expect(adjust?.getAttribute("href")).toBe("/projects/project-1/review");
    expect(adjust?.className).toBe(linkButtonClass);
    // The walk link comes before the adjust link in the card.
    const walkFirst = walk!.compareDocumentPosition(adjust!) & Node.DOCUMENT_POSITION_FOLLOWING;
    expect(walkFirst).toBeTruthy();
  });

  it("offers viewers the walk only", async () => {
    vi.mocked(useProject).mockReturnValue({
      detail: detailFor("viewer"),
      etag: null,
      error: null,
      loading: false,
      reload: async () => {},
    } as never);
    await act(async () => {
      root.render(<ProjectDetails projectId="project-1" />);
    });
    await settle();

    expect(link("Walk privately")).toBeDefined();
    expect(link("Adjust layout")).toBeUndefined();
  });
});
