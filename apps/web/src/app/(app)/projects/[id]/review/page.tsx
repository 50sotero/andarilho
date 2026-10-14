import type { Metadata } from "next";
import { ReviewEditor } from "@/features/review/ReviewEditor";

export const metadata: Metadata = { title: "Review" };

export default async function ReviewPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Review the scene</h1>
      <p className="max-w-3xl text-sm text-stone-700 dark:text-stone-300">
        Confirm the walkable floor, walls, doors, obstacles and a safe starting point. Drag the
        handles or select them with Tab and move them with the arrow keys. Only areas the camera
        actually saw should be marked walkable.
      </p>
      <ReviewEditor projectId={id} />
    </div>
  );
}
