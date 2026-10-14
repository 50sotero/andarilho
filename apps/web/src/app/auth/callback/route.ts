import { NextResponse, type NextRequest } from "next/server";
import { createSupabaseServerClient } from "@/lib/auth/server";
import { safeNextPath } from "@/lib/auth/safe-redirect";
import { getAppOrigin } from "@/lib/env";

export const dynamic = "force-dynamic";

function originFor(request: NextRequest): string {
  try {
    return getAppOrigin();
  } catch {
    return request.nextUrl.origin;
  }
}

/** Exchanges the magic-link/PKCE code for a session, then redirects to a safe relative path. */
export async function GET(request: NextRequest) {
  const origin = originFor(request);
  const code = request.nextUrl.searchParams.get("code");
  const next = safeNextPath(request.nextUrl.searchParams.get("next"));

  if (code) {
    try {
      const supabase = await createSupabaseServerClient();
      const { error } = await supabase.auth.exchangeCodeForSession(code);
      if (!error) {
        const response = NextResponse.redirect(new URL(next, origin));
        response.headers.set("Cache-Control", "private, no-store");
        return response;
      }
    } catch {
      // fall through to the error redirect
    }
  }

  const failure = new URL("/login", origin);
  failure.searchParams.set("error", "callback");
  return NextResponse.redirect(failure);
}
