import type { Metadata } from "next";
import { ProcessingView } from "@/features/processing/ProcessingView";

export const metadata: Metadata = { title: "Processing" };

export default async function ProcessingPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ job?: string | string[] }>;
}) {
  const [{ id }, query] = await Promise.all([params, searchParams]);
  const job = typeof query.job === "string" ? query.job : null;
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Processing</h1>
      <ProcessingView projectId={id} initialJobId={job} />
    </div>
  );
}
