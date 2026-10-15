/**
 * What leaves the server when AI scene analysis is switched on. The deployment
 * decides that with NEXT_PUBLIC_AI_SCENE_ANALYSIS, so the notice is shown only then:
 * when analysis is off, nothing is sent to an AI model and nothing needs saying.
 */
export const SCENE_ANALYSIS_NOTICE =
  "To place rooms, doors and furniture automatically, a few still frames from your video, your photos and your floor plan are sent to an AI model (DeepSeek, through OpenCode). Nothing else leaves our servers.";

/** True only for the exact flag value "true" (the deployment's switch is on). */
export function sceneAnalysisNoticeShown(flag: string | undefined): boolean {
  return flag === "true";
}

export function SceneAnalysisNotice({ shown }: { shown: boolean }) {
  if (!shown) return null;
  return (
    <p className="mt-4 rounded-md border border-stone-300 bg-stone-50 p-3 text-sm text-stone-800 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200">
      {SCENE_ANALYSIS_NOTICE}
    </p>
  );
}
