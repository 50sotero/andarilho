import Link from "next/link";
import { Card, linkButtonClass, primaryLinkClass } from "@/components/ui";
import { CaptureGuidance } from "@/features/capture/CaptureGuidance";

const STEPS: readonly [string, string][] = [
  [
    "Capture",
    "Record one slow, continuous walkthrough. Add photos and a floor plan if you have them.",
  ],
  ["Upload", "Files go straight to private storage and can resume if your connection drops."],
  [
    "Process",
    "The video's camera path places the rooms, and the walls, floor and ceiling are coloured from its frames. If a capture cannot be connected, you get a clear reason and a suggested fix.",
  ],
  [
    "Adjust (optional)",
    "We place rooms, doors and furniture automatically; adjust anything you like afterwards.",
  ],
  [
    "Walk",
    "Explore privately with collision, then publish a specific revision with a revocable link.",
  ],
  ["Control", "Revoke links or delete the project, including its original capture, at any time."],
];

export default function HomePage() {
  return (
    <div className="space-y-10">
      <section className="space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">Capture a place. Step inside.</h1>
        <p className="max-w-2xl text-lg text-stone-700 dark:text-stone-300">
          Andarilho turns a continuous walkthrough video and overlapping photos of a real place into
          a walkable 3D scene in your browser. A floor plan is optional and helps establish scale
          and align rooms.
        </p>
        <div className="flex gap-3">
          <Link href="/projects/new" className={primaryLinkClass}>
            Start a project
          </Link>
          <Link href="/projects" className={linkButtonClass}>
            Your projects
          </Link>
        </div>
      </section>

      <section aria-labelledby="flow-heading" className="space-y-3">
        <h2 id="flow-heading" className="text-lg font-semibold">
          How it works
        </h2>
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {STEPS.map(([title, body], i) => (
            <li
              key={title}
              className="rounded-lg border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900"
            >
              <p className="text-sm font-semibold">
                {i + 1}. {title}
              </p>
              <p className="mt-1 text-sm text-stone-700 dark:text-stone-300">{body}</p>
            </li>
          ))}
        </ol>
      </section>

      <Card>
        <CaptureGuidance />
      </Card>

      <section className="space-y-2 text-sm text-stone-700 dark:text-stone-300">
        <h2 className="text-lg font-semibold text-stone-900 dark:text-stone-100">
          What it cannot do
        </h2>
        <p>
          Andarilho does not invent rooms it never saw. Missing surfaces, mirrors, glass, moving
          people and blank walls can make reconstruction ambiguous. Parts the video never saw stay
          plain grey rather than being invented. Measurements are only shown as real when you
          confirmed a physical distance.
        </p>
      </section>
    </div>
  );
}
