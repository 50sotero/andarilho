"use client";

import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";
import { getPublicSupabaseConfig } from "@/lib/env";

let client: SupabaseClient | undefined;

/** Supabase browser client (singleton per tab). Used for login/logout only. */
export function getSupabaseBrowserClient(): SupabaseClient {
  if (!client) {
    const { url, publishableKey } = getPublicSupabaseConfig();
    client = createBrowserClient(url, publishableKey);
  }
  return client;
}
