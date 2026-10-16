import type { Metadata } from "next";
import { CaptureFlow } from "@/features/capture/CaptureFlow";

export const metadata: Metadata = { title: "Capture" };

export default async function CapturePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Capture and upload</h1>
      <CaptureFlow projectId={id} />
    </div>
  );
}
