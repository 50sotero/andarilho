/**
 * Viewer mode (README section 14): loads the tiny synthetic fixture straight
 * from packages/scene-contract/fixtures and lets you walk it. No API, GPU or
 * cloud credentials are involved.
 */
import type { Layout, SceneManifest } from "@andarilho/scene-contract";
import layoutJson from "@andarilho/scene-contract/fixtures/synthetic-two-room.layout.json";
import manifestJson from "@andarilho/scene-contract/fixtures/synthetic-two-room.manifest.json";
import collisionUrl from "@andarilho/scene-contract/fixtures/synthetic-two-room.collision.glb?url";
import renderUrl from "@andarilho/scene-contract/fixtures/synthetic-two-room.splat.ply?url";
import { createViewer, isQualityProfileName, type ViewerHandle, type ViewerState } from "../src";

const manifest = manifestJson as unknown as SceneManifest;
const layout = layoutJson as unknown as Layout;

function el<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing #${id}`);
  return found as T;
}

const stage = el<HTMLDivElement>("stage");
const walkButton = el<HTMLButtonElement>("walk");
const exitButton = el<HTMLButtonElement>("exit");
const resetButton = el<HTMLButtonElement>("reset");
const qualitySelect = el<HTMLSelectElement>("quality");
const collisionSelect = el<HTMLSelectElement>("collision");
const overlayToggle = el<HTMLInputElement>("overlay");
const status = el<HTMLSpanElement>("status");

function describe(state: ViewerState): string {
  switch (state.status) {
    case "loading": {
      const p = state.progress;
      const pct = p.renderFraction === null ? "" : ` ${Math.round(p.renderFraction * 100)}%`;
      return `loading render:${p.render}${pct} collision:${p.collision} physics:${p.physics} spawn:${p.spawn}`;
    }
    case "ready":
      return state.mode === "walkable"
        ? `ready (walkable, ${state.quality})`
        : `inspection only: ${state.inspectionReason ?? "unknown reason"}`;
    case "walking":
      return `walking (${state.quality}${state.pointerLocked ? ", pointer locked" : ""})`;
    case "error":
      return `error: ${state.error.message}`;
    case "context-lost":
      return "graphics context lost; waiting for the browser to restore it";
    default:
      return state.status;
  }
}

let viewer: ViewerHandle | null = null;

function mount(): void {
  viewer?.dispose();
  const useLayout = collisionSelect.value === "layout";
  const quality = qualitySelect.value;
  viewer = createViewer({
    container: stage,
    source: {
      kind: "urls",
      manifest,
      renderUrl,
      collisionUrl: useLayout ? null : collisionUrl,
      layout: useLayout ? layout : null,
    },
    quality: isQualityProfileName(quality) ? quality : "auto",
    onStateChange: (state) => {
      status.textContent = describe(state);
      walkButton.disabled = !(state.status === "ready" && state.mode === "walkable");
      exitButton.disabled = state.status !== "walking";
    },
  });
  viewer.setCollisionOverlay(overlayToggle.checked);
}

walkButton.addEventListener("click", () => viewer?.enterWalk());
exitButton.addEventListener("click", () => viewer?.exitWalk());
resetButton.addEventListener("click", () => viewer?.resetPosition());
qualitySelect.addEventListener("change", () => {
  const value = qualitySelect.value;
  viewer?.setQuality(isQualityProfileName(value) ? value : "auto");
});
overlayToggle.addEventListener("change", () => viewer?.setCollisionOverlay(overlayToggle.checked));
collisionSelect.addEventListener("change", mount);

// Virtual pad indicator for touch input.
const pad = el<HTMLDivElement>("pad");
stage.addEventListener("pointerdown", (event) => {
  const rect = stage.getBoundingClientRect();
  if (event.pointerType === "mouse" || event.clientX - rect.left > rect.width / 2) return;
  pad.style.display = "block";
  pad.style.left = `${event.clientX - rect.left}px`;
  pad.style.top = `${event.clientY - rect.top}px`;
});
const hidePad = () => {
  pad.style.display = "none";
};
stage.addEventListener("pointerup", hidePad);
stage.addEventListener("pointercancel", hidePad);

// Position readout for manual testing.
const position = document.createElement("span");
position.style.fontSize = "0.8rem";
position.style.color = "#9ca3af";
status.after(position);
setInterval(() => {
  const feet = viewer?.getDiagnostics().feet;
  position.textContent = feet ? `feet ${feet.map((v) => v.toFixed(2)).join(", ")}` : "";
}, 250);

mount();

if (import.meta.hot) {
  import.meta.hot.dispose(() => viewer?.dispose());
}
