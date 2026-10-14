import type { Metadata } from "next";
import { Card } from "@/components/ui";
import { NewProjectForm } from "@/features/projects/NewProjectForm";

export const metadata: Metadata = { title: "New project" };

export default function NewProjectPage() {
  return (
    <div className="mx-auto max-w-lg space-y-4">
      <h1 className="text-2xl font-semibold">New project</h1>
      <Card>
        <NewProjectForm />
      </Card>
    </div>
  );
}
