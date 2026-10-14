import { INTAKE_LIMITS } from "./validation";

export const CAPTURE_TIPS: readonly string[] = [
  "Use a normal perspective camera, good light, and a consistent lens/zoom.",
  "Walk slowly with translation; avoid recording only a stationary panorama.",
  "Keep doors open and film the transition between rooms continuously.",
  "Look into corners and revisit the start to help connect the camera trajectory.",
  "Capture furniture from several sides and add photos for poorly covered details.",
  "Avoid people moving through the frame, rapid turns, and long views into mirrors.",
  "Include at least one physical distance you can confirm if real-world scale matters.",
  "Capture only spaces and material you have permission to process and share.",
];

export function CaptureGuidance({ headingLevel = 2 }: { headingLevel?: 2 | 3 }) {
  const Heading = headingLevel === 2 ? "h2" : "h3";
  return (
    <div className="space-y-4">
      <Heading className="text-lg font-semibold">How to capture a place</Heading>
      <ul className="list-disc space-y-1 pl-5 text-sm">
        {CAPTURE_TIPS.map((tip) => (
          <li key={tip}>{tip}</li>
        ))}
      </ul>
      <div className="text-sm text-stone-700 dark:text-stone-300">
        <p className="font-medium">Beta intake limits</p>
        <ul className="list-disc pl-5">
          <li>One video, up to {INTAKE_LIMITS.maxVideoSeconds / 60} minutes (MP4 or MOV).</li>
          <li>Up to {INTAKE_LIMITS.maxPhotos} supplementary photos (JPEG or PNG).</li>
          <li>Up to 2 GiB in total.</li>
          <li>An optional floor plan up to 20 MiB (PNG, JPEG or PDF).</li>
        </ul>
        <p className="mt-2">
          These are starting limits, not reconstruction guarantees. Areas the camera never saw
          cannot be reconstructed, and rooms that were not filmed connected may come out
          separate.
        </p>
      </div>
    </div>
  );
}
