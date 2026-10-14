/**
 * @andarilho/viewer public API.
 *
 * `createViewer` is browser-only (call it from a client component after mount).
 * The other exports are pure modules that are safe to import on the server and
 * in Node tests. Spark and Rapier are loaded lazily by the engine.
 */

export {
  createViewer,
  type CreateViewerOptions,
  type CreateSparkSceneArgs,
  type CreateCollisionWorldArgs,
  type CollisionWorldLike,
  type QualitySetting,
  type RendererLike,
  type ViewerDependencies,
  type ViewerDiagnostics,
  type ViewerHandle,
} from "./create-viewer";

export {
  canWalk,
  initialLoadProgress,
  transition,
  type LoadProgress,
  type StepStatus,
  type ViewerError,
  type ViewerErrorCode,
  type ViewerEvent,
  type ViewerMode,
  type ViewerState,
  type ViewerStatus,
} from "./viewer-state";

export {
  deliveryEnvelopeIssues,
  isAllowedUrl,
  resolveViewerSource,
  sceneManifestIssues,
  sha256Hex,
  ViewerInputError,
  type ResolvedSource,
  type ViewerSource,
} from "./manifest";

export {
  DEFAULT_PLAYER_CONFIG,
  PlayerState,
  capsuleCenterY,
  capsuleHalfHeight,
  centerToFeet,
  desiredTranslation,
  eyeYFromFeetY,
  feetToCenter,
  feetToEye,
  feetYFromCenterY,
  integrateVertical,
  isOutOfBounds,
  resolvePlayerConfig,
  validatePlayerConfig,
  type PlayerConfig,
  type VerticalStep,
} from "./movement/player";

export {
  DEFAULT_LOOK_SENSITIVITY,
  DEFAULT_PITCH_LIMIT,
  DEFAULT_TOUCH_LOOK_SENSITIVITY,
  InputController,
  applyLookDelta,
  clampAxes,
  clampPitch,
  keyCodeToAction,
  keysToAxes,
  moveDirectionFromAxes,
  moveDirectionFromKeys,
  padAxes,
  prefersReducedMotion,
  wrapAngle,
  type InputControllerOptions,
  type LookAngles,
  type MoveAxes,
  type MoveKeys,
} from "./movement/input";

export {
  CollisionAssetError,
  CollisionWorld,
  DEFAULT_CONTROLLER_CONFIG,
  FixedStepAccumulator,
  collisionMeshIssues,
  initRapier,
  nodeTransformIssue,
  parseGlb,
  readCollisionGlb,
  validateCollisionMesh,
  writeCollisionGlb,
  type CollisionMeshStats,
  type CollisionValidationOptions,
  type CollisionWorldOptions,
  type ControllerConfig,
  type FixedStepOptions,
  type FixedStepResult,
  type RapierModule,
  type SpawnValidation,
  type StepResult,
  type ValidatedCollisionMesh,
} from "./movement/collision";

export {
  DOWNGRADE_ORDER,
  QUALITY_PROFILES,
  QualityGovernor,
  classifyDevice,
  detectDeviceInfo,
  effectivePixelRatio,
  isQualityProfileName,
  nextHigherProfile,
  nextLowerProfile,
  selectQualityProfile,
  type DeviceClass,
  type DeviceInfo,
  type QualityChange,
  type QualityGovernorOptions,
  type QualityProfile,
  type QualityProfileName,
} from "./render/quality";

export {
  decomposeRenderTransform,
  RenderTransformError,
  type RenderTransform,
} from "./render/transform";
export type { RenderFormat, SparkSceneHandle } from "./render/spark-scene";

export {
  LayoutValidationError,
  MAX_ELEMENT_HEIGHT,
  groupProjectedArea,
  isDegeneratePolygon,
  isSelfIntersecting,
  layoutBounds,
  layoutToCollisionMesh,
  normalizeRing,
  pointInPolygon,
  polygonArea,
  polygonSignedArea,
  segmentsIntersect,
  splitWall,
  validateLayout,
  wallPieceFootprint,
  type FloorplanGroup,
  type FloorplanMesh,
  type FloorplanOptions,
  type FloorplanPartKind,
  type LayoutIssue,
  type WallPiece,
} from "./review/floorplan";

export {
  applySimilarity,
  fitSimilarity2D,
  type LandmarkPair,
  type SimilarityFit,
  type SimilarityFitFailure,
  type SimilarityFitOptions,
  type SimilarityFitResult,
} from "./review/alignment";

export {
  createCollisionOverlay,
  type CollisionOverlay,
  type CollisionOverlayOptions,
} from "./debug/overlay";

export type { Aabb, CollisionMesh, Vec2, Vec3 } from "./types";
