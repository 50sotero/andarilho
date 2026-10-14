import type { Metadata } from "next";
import Link from "next/link";
import { primaryLinkClass } from "@/components/ui";
import { ProjectList } from "@/features/projects/ProjectList";

export const metadata: Metadata = { title: "Projects" };

export default function ProjectsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Projects</h1>
        <Link href="/projects/new" className={primaryLinkClass}>
          New project
        </Link>
      </div>
      <ProjectList />
    </div>
  );
}
