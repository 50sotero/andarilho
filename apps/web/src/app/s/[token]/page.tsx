import type { Metadata } from "next";
import { PublicScene } from "@/features/sharing/PublicScene";

export const dynamic = "force-dynamic";

// Share URLs are bearer capabilities: keep them out of search engines and Referer headers.
// next.config.ts also sends `Referrer-Policy: no-referrer` and `X-Robots-Tag` for /s/*.
export const metadata: Metadata = {
  title: "Shared scene",
  robots: { index: false, follow: false, nocache: true },
  referrer: "no-referrer",
};

export default async function SharedScenePage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = await params;
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Shared scene</h1>
      <p className="text-sm text-stone-600 dark:text-stone-400">
        A read-only 3D walkthrough shared through Andarilho. It shows an approved reconstruction,
        not a surveyed measurement of the place.
      </p>
      <PublicScene token={token} />
    </div>
  );
}
