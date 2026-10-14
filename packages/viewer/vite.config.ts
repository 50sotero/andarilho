import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const packageDir = fileURLToPath(new URL(".", import.meta.url));
const fixturesDir = fileURLToPath(new URL("../scene-contract/fixtures", import.meta.url));

// Viewer mode (README section 14): `corepack pnpm --filter @andarilho/viewer dev`
// serves index.html + dev/main.ts and the synthetic fixture straight from
// packages/scene-contract/fixtures (imported through the package's exports).
export default defineConfig({
  server: {
    fs: {
      allow: [packageDir, fixturesDir],
    },
  },
  assetsInclude: ["**/*.ply", "**/*.glb", "**/*.sog", "**/*.spz"],
  optimizeDeps: {
    // Rapier's compat build inlines its WASM; Spark ships its own workers/WASM.
    exclude: ["@dimforge/rapier3d-compat", "@sparkjsdev/spark"],
  },
  build: {
    target: "es2022",
  },
});
