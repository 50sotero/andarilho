import type { Metadata } from "next";
import { PrivateWalk } from "@/features/projects/PrivateWalk";

export const metadata: Metadata = { title: "Walk" };

export default async function ViewPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Walk privately</h1>
      <PrivateWalk projectId={id} />
    </div>
  );
}
