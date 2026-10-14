import type { Metadata } from "next";
import { ProjectDetails } from "@/features/projects/ProjectDetails";

export const metadata: Metadata = { title: "Project" };

export default async function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ProjectDetails projectId={id} />;
}
