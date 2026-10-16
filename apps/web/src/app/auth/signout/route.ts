import { NextResponse, type NextRequest } from "next/server";
import { isAllowedOrigin } from "@/lib/api/backend-proxy";
import { createSupabaseServerClient } from "@/lib/auth/server";
import { getAppOrigin } from "@/lib/env";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  let origin: string;
  try {
    origin = getAppOrigin();
  } catch {
    origin = request.nextUrl.origin;
  }
  if (!isAllowedOrigin("POST", request.headers.get("origin"), origin)) {
    return new NextResponse("Cross-origin request rejected.", { status: 403 });
  }
  try {
    const supabase = await createSupabaseServerClient();
    await supabase.auth.signOut();
  } catch {
    // Cookies are cleared best-effort; the redirect still happens.
  }
  const response = NextResponse.redirect(new URL("/login", origin), { status: 303 });
  response.headers.set("Cache-Control", "private, no-store");
  return response;
}
