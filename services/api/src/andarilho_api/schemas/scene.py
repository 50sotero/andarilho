"""Scene manifest and layout contracts.

These Pydantic models are the single source of truth for the persistent scene
manifest (README section 10) and the editable layout (section 9).
``scripts/generate-contracts.py`` emits the JSON Schema and TypeScript types
from them; do not edit the generated files by hand.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

# The manifest and the layout are versioned separately. Manifest 2.0 replaced the
# Gaussian-splat render (1.0) with a textured mesh; the layout format is unchanged.
SCHEMA_VERSION = "2.0"
LAYOUT_SCHEMA_VERSION = "1.0"

Id = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
# Viewer derivatives must live under their own scene revision prefix.
SceneAssetKey = Annotated[
    str,
    Field(
        pattern=r"^tenants/[A-Za-z0-9_-]+/scenes/[A-Za-z0-9_-]+/[A-Za-z0-9._-]+$", max_length=512
    ),
]
Finite = Annotated[float, Field(allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Vec3 = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Vec2 = Annotated[list[Finite], Field(min_length=2, max_length=2)]
Matrix4 = Annotated[list[Finite], Field(min_length=16, max_length=16)]


class Contract(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
    )


class Bounds(Contract):
    min: Vec3
    max: Vec3

    @model_validator(mode="after")
    def _ordered(self) -> Bounds:
        if any(lo >= hi for lo, hi in zip(self.min, self.max, strict=True)):
            raise ValueError("bounds.min must be strictly less than bounds.max on every axis")
        return self

    def contains(self, point: list[float]) -> bool:
        return all(lo <= p <= hi for lo, p, hi in zip(self.min, point, self.max, strict=True))


class Spawn(Contract):
    feet: Vec3
    yaw_radians: Finite
    floor_id: Id


class FloorRef(Contract):
    id: Id
    elevation: Finite


class Asset(Contract):
    key: SceneAssetKey
    bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256


class RenderAsset(Asset):
    """The textured layout model (binary glTF, unlit materials, world space)."""

    format: Literal["glb"]


class CollisionAsset(Asset):
    format: Literal["glb"]


class PosterAsset(Asset):
    format: Literal["webp", "png", "jpeg"]


class SceneAssets(Contract):
    render: RenderAsset
    collision: CollisionAsset
    poster: PosterAsset | None = None


class Navigation(Contract):
    mode: Literal["approved_collision", "inspection_only"]
    layout_revision: Annotated[int, Field(ge=1)]
    approved: bool
    player_radius: Positive = 0.25
    player_height: Positive = 1.7
    eye_height: Positive = 1.6

    @model_validator(mode="after")
    def _capsule(self) -> Navigation:
        if self.player_height <= 2 * self.player_radius:
            raise ValueError("playerHeight must exceed twice playerRadius")
        if self.eye_height > self.player_height:
            raise ValueError("eyeHeight cannot exceed playerHeight")
        return self


class Quality(Contract):
    registered_views: Annotated[int, Field(ge=0)]
    selected_views: Annotated[int, Field(ge=0)]
    disconnected_components: Annotated[int, Field(ge=0)]
    warnings: list[Annotated[str, Field(max_length=500)]] = Field(
        default_factory=list, max_length=100
    )


class Provenance(Contract):
    recipe_digest: Annotated[str, Field(min_length=1, max_length=128)]
    capture_set_id: Id
    geometry_source: Literal[
        "user_confirmed_trace", "floor_plan_trace", "proposal", "synthetic_fixture"
    ]


class SceneManifest(Contract):
    """Persistent scene manifest. Contains object keys, never signed URLs or tokens."""

    schema_version: Literal["2.0"]
    scene_id: Id
    scene_revision_id: Id
    revision: Annotated[int, Field(ge=1)]
    run_id: Id
    render_mode: Literal["mesh"]
    units: Literal["m"]
    up_axis: Literal["+Y"]
    camera_forward: Literal["-Z"]
    scale_status: Literal["user_measured", "estimated"]
    geometry_space: Literal["world"]
    bounds: Bounds
    spawn: Spawn
    floors: Annotated[list[FloorRef], Field(min_length=1, max_length=16)]
    assets: SceneAssets
    navigation: Navigation
    quality: Quality
    provenance: Provenance

    @model_validator(mode="after")
    def _consistency(self) -> SceneManifest:
        floor_ids = {floor.id for floor in self.floors}
        if len(floor_ids) != len(self.floors):
            raise ValueError("floor ids must be unique")
        if self.spawn.floor_id not in floor_ids:
            raise ValueError("spawn.floorId must reference a declared floor")
        if not self.bounds.contains(self.spawn.feet):
            raise ValueError("spawn.feet must lie within bounds")
        prefix = None
        for asset in (self.assets.render, self.assets.collision, self.assets.poster):
            if asset is None:
                continue
            parts = asset.key.split("/")
            if parts[3] != self.scene_revision_id:
                raise ValueError("every asset key must belong to this sceneRevisionId")
            if prefix is None:
                prefix = parts[1]
            elif parts[1] != prefix:
                raise ValueError("every asset key must belong to the same project prefix")
        return self


# ---------------------------------------------------------------------------
# Editable layout (section 9). World coordinates: meters, +Y up, XZ horizontal.
# 2D points are [x, z] on the horizontal plane.
# ---------------------------------------------------------------------------

Polygon = Annotated[list[Vec2], Field(min_length=3, max_length=512)]


class Provenanced(Contract):
    confidence: Annotated[float, Field(ge=0, le=1)] = 1.0
    provenance: Literal["user", "floor_plan", "proposal", "synthetic"] = "user"


class LayoutFloor(Contract):
    id: Id
    elevation: Finite
    polygons: Annotated[list[Polygon], Field(min_length=1, max_length=64)]
    holes: Annotated[list[Polygon], Field(max_length=64)] = Field(default_factory=list)


class Wall(Provenanced):
    id: Id
    floor_id: Id
    start: Vec2
    end: Vec2
    thickness: Positive
    height: Positive


class Opening(Provenanced):
    id: Id
    wall_id: Id
    offset: NonNegative
    width: Positive
    bottom: NonNegative = 0.0
    top: Positive
    state: Literal["open", "closed"] = "open"

    @model_validator(mode="after")
    def _heights(self) -> Opening:
        if self.top <= self.bottom:
            raise ValueError("opening top must be above bottom")
        return self


class Obstacle(Provenanced):
    id: Id
    floor_id: Id
    footprint: Polygon
    height: Positive


class Landmark(Contract):
    id: Id
    plan_point: Vec2 | None = None
    world_point: Vec3
    uncertainty: NonNegative = 0.0


class LayoutSpawn(Contract):
    feet: Vec3
    yaw_radians: Finite
    floor_id: Id


class Calibration(Contract):
    """User-confirmed metric calibration.

    ``render_scale`` is the cumulative uniform scale, about the world origin, applied
    to the run's estimated frame (its camera poses and capture evidence). It is
    absolute rather than relative to the previous revision, so re-saving a layout
    never compounds it.
    """

    method: Literal["known_distance"]
    render_scale: Positive
    reference_length: Positive | None = None


class Layout(Contract):
    schema_version: Literal["1.0"]
    floors: Annotated[list[LayoutFloor], Field(min_length=1, max_length=16)]
    walls: Annotated[list[Wall], Field(max_length=2048)] = Field(default_factory=list)
    openings: Annotated[list[Opening], Field(max_length=1024)] = Field(default_factory=list)
    obstacles: Annotated[list[Obstacle], Field(max_length=1024)] = Field(default_factory=list)
    landmarks: Annotated[list[Landmark], Field(max_length=256)] = Field(default_factory=list)
    spawn: LayoutSpawn
    calibration: Calibration | None = None

    @model_validator(mode="after")
    def _references(self) -> Layout:
        floor_ids = {f.id for f in self.floors}
        wall_ids = {w.id for w in self.walls}
        if len(floor_ids) != len(self.floors) or len(wall_ids) != len(self.walls):
            raise ValueError("layout ids must be unique")
        if self.spawn.floor_id not in floor_ids:
            raise ValueError("spawn.floorId must reference a floor")
        for item in [*self.walls, *self.obstacles]:
            if item.floor_id not in floor_ids:
                raise ValueError(f"{item.id} references unknown floor {item.floor_id}")
        walls = {w.id: w for w in self.walls}
        for opening in self.openings:
            wall = walls.get(opening.wall_id)
            if wall is None:
                raise ValueError(f"opening {opening.id} references unknown wall")
            length = sum((e - s) ** 2 for s, e in zip(wall.start, wall.end, strict=True)) ** 0.5
            if opening.offset + opening.width > length + 1e-9:
                raise ValueError(f"opening {opening.id} extends past its wall")
            if opening.top > wall.height + 1e-9:
                raise ValueError(f"opening {opening.id} is taller than its wall")
        for wall in self.walls:
            if wall.start == wall.end:
                raise ValueError(f"wall {wall.id} is degenerate")
        return self


# ---------------------------------------------------------------------------
# Short-lived delivery envelope (section 10). Never persisted.
# ---------------------------------------------------------------------------


class AssetUrl(Contract):
    url: str
    sha256: Sha256
    bytes: Annotated[int, Field(gt=0)]
    format: str


class DeliveryEnvelope(Contract):
    scene_revision_id: Id
    manifest_sha256: Sha256
    expires_at: str
    manifest: SceneManifest
    asset_urls: dict[Literal["render", "collision", "poster"], AssetUrl]
