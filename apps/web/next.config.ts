import path from "node:path";
import type { NextConfig } from "next";

// The workspace root (two levels up from apps/web) so workspace packages are
// traced into the standalone output and resolvable by Turbopack.
const workspaceRoot = path.join(process.cwd(), "..", "..");

const baseSecurityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  {
    key: "Content-Security-Policy",
    value: "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'",
  },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), payment=()" },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
];

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
  transpilePackages: ["@andarilho/viewer", "@andarilho/scene-contract"],
  outputFileTracingRoot: workspaceRoot,
  turbopack: {
    root: workspaceRoot,
  },
  async headers() {
    return [
      { source: "/:path*", headers: baseSecurityHeaders },
      // Share pages carry a bearer capability in the URL: never leak it via Referer.
      // Declared last so it overrides the general Referrer-Policy above.
      {
        source: "/s/:path*",
        headers: [
          { key: "Referrer-Policy", value: "no-referrer" },
          { key: "X-Robots-Tag", value: "noindex, nofollow, noarchive" },
          { key: "Cache-Control", value: "private, no-store" },
        ],
      },
    ];
  },
};

export default nextConfig;
