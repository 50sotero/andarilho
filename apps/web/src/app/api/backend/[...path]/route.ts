import { adviceTimeoutMs, createBackendProxyHandler } from "@/lib/api/backend-proxy";
import { getAccessToken } from "@/lib/auth/server";
import { getAiTimeoutSeconds, getApiInternalUrl, getAppOrigin } from "@/lib/env";

export const dynamic = "force-dynamic";

const handle = createBackendProxyHandler({
  getAccessToken,
  getAppOrigin,
  getApiBaseUrl: getApiInternalUrl,
  fetch: (input, init) => fetch(input, init),
  adviceTimeoutMs: adviceTimeoutMs(getAiTimeoutSeconds()),
});

type RouteContext = { params: Promise<{ path: string[] }> };

async function handler(request: Request, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  return handle(request, path ?? []);
}

export const GET = handler;
export const POST = handler;
export const PUT = handler;
export const PATCH = handler;
export const DELETE = handler;
