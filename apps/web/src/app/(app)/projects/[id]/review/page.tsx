import type { Metadata } from "next";
import { ReviewEditor } from "@/features/review/ReviewEditor";

export const metadata: Metadata = { title: "Adjust layout" };

export default async function ReviewPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Adjust the layout</h1>
      <p className="max-w-3xl text-sm text-stone-700 dark:text-stone-300">
        The scene is already walkable. Adjust anything the automatic placement got wrong: walls,
        doors, furniture or the scale. Drag the handles or select them with Tab and move them with
        the arrow keys. To remove a piece of furniture, select it and choose Remove. Saving makes a
        new revision.
      </p>
      <ReviewEditor projectId={id} />
    </div>
  );
}
