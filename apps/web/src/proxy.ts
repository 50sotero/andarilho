import type { NextRequest } from "next/server";
import { updateSession } from "@/lib/auth/session-proxy";

export async function proxy(request: NextRequest) {
  return updateSession(request);
}

export const config = {
  // Application pages and the auth flow. Static assets, the backend proxy
  // (which reads the session itself) and public share pages are excluded.
  matcher: ["/projects/:path*", "/projects", "/login", "/auth/:path*"],
};
