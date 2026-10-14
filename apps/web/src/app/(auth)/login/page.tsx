import type { Metadata } from "next";
import { Card } from "@/components/ui";
import { LoginForm } from "@/features/auth/LoginForm";
import { safeNextPath } from "@/lib/auth/safe-redirect";

export const metadata: Metadata = { title: "Sign in" };

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ next?: string | string[]; error?: string | string[] }>;
}) {
  const params = await searchParams;
  const next = safeNextPath(typeof params.next === "string" ? params.next : null);
  return (
    <div className="mx-auto max-w-md space-y-4">
      <h1 className="text-2xl font-semibold">Sign in</h1>
      <Card>
        <LoginForm next={next} callbackError={params.error === "callback"} />
      </Card>
      <p className="text-sm text-stone-600 dark:text-stone-400">
        We email you a one-time link and code. No password is needed.
      </p>
    </div>
  );
}
