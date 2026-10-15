import { createReadStream, existsSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const packageDir = fileURLToPath(new URL(".", import.meta.url));
const fixturesDir = fileURLToPath(new URL("../scene-contract/fixtures", import.meta.url));

const BUNDLE_TYPES: Record<string, string> = {
  ".json": "application/json",
  ".glb": "model/gltf-binary",
};

/**
 * Dev only: serves a smoke-run scene bundle (manifest, layout, render asset,
 * collision GLB) from VIEWER_BUNDLE_DIR at /bundle/, for `?bundle=/bundle`.
 * Plain file names only; nothing outside that directory is reachable.
 */
function sceneBundle(dir: string | undefined): Plugin {
  return {
    name: "andarilho-scene-bundle",
    apply: "serve",
    configureServer(server) {
      if (!dir) return;
      server.middlewares.use("/bundle", (req, res, next) => {
        const pathname = (req.url ?? "/").split("?")[0] ?? "";
        const name = decodeURIComponent(pathname).replace(/^\/+/, "");
        if (!/^[\w.-]+$/.test(name) || name.startsWith(".")) return next();
        const file = path.join(dir, name);
        if (!existsSync(file) || !statSync(file).isFile()) return next();
        res.setHeader(
          "Content-Type",
          BUNDLE_TYPES[path.extname(name)] ?? "application/octet-stream",
        );
        res.setHeader("Cache-Control", "no-store");
        createReadStream(file).pipe(res);
      });
    },
  };
}

// Viewer mode (README section 14): `corepack pnpm --filter @andarilho/viewer dev`
// serves index.html + dev/main.ts and the synthetic fixture straight from
// packages/scene-contract/fixtures (imported through the package's exports).
export default defineConfig({
  plugins: [sceneBundle(process.env.VIEWER_BUNDLE_DIR)],
  server: {
    fs: {
      allow: [packageDir, fixturesDir],
    },
  },
  // The textured model and the collision mesh are imported as URLs (`?url`).
  assetsInclude: ["**/*.glb"],
  optimizeDeps: {
    // Rapier's compat build inlines its WASM, so Vite must not pre-bundle it.
    exclude: ["@dimforge/rapier3d-compat"],
  },
  build: {
    target: "es2022",
  },
});
