import { createServerClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";
import { cookies } from "next/headers";
import { getPublicSupabaseConfig } from "@/lib/env";

/**
 * Supabase server client bound to the current request's cookies.
 * Create a new one per request; never share it.
 */
export async function createSupabaseServerClient(): Promise<SupabaseClient> {
  const cookieStore = await cookies();
  const { url, publishableKey } = getPublicSupabaseConfig();
  return createServerClient(url, publishableKey, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll(cookiesToSet) {
        try {
          for (const { name, value, options } of cookiesToSet) {
            cookieStore.set(name, value, options);
          }
        } catch {
          // Called from a Server Component, where cookies are read-only.
          // The proxy (src/proxy.ts) refreshes the session cookies instead.
        }
      },
    },
  });
}

/**
 * The current user's Supabase access token, or null when signed out.
 * The token is forwarded to the API, which verifies the JWT signature,
 * issuer, audience and expiry itself; this function never authorizes anything.
 */
export async function getAccessToken(): Promise<string | null> {
  try {
    const supabase = await createSupabaseServerClient();
    const { data } = await supabase.auth.getSession();
    return data.session?.access_token ?? null;
  } catch {
    return null;
  }
}

/** Verified user id (subject) for server-rendered UI decisions, or null. */
export async function getUserSubject(): Promise<string | null> {
  try {
    const supabase = await createSupabaseServerClient();
    const { data, error } = await supabase.auth.getClaims();
    if (error || !data) return null;
    return typeof data.claims.sub === "string" ? data.claims.sub : null;
  } catch {
    return null;
  }
}
