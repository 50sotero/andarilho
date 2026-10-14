import Link from "next/link";

export const dynamic = "force-dynamic";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4 text-sm">
        <nav aria-label="Projects">
          <Link href="/projects" className="font-medium hover:underline">
            All projects
          </Link>
        </nav>
        <form action="/auth/signout" method="post">
          <button type="submit" className="text-stone-600 hover:underline dark:text-stone-300">
            Sign out
          </button>
        </form>
      </div>
      {children}
    </div>
  );
}
