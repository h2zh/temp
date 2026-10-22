"""Algorithm tunables of reconstruct.py, validation.py and qa.py, overridable in one place.

Product thresholds (photo counts, validation limits, quality-gate limits) live in
config.Settings. The values here are the algorithms' internal knobs. The modules
read them at call time through current(), so an evaluation can run the same code
with other values (scripts/benchmark.py configs) while the defaults below stay
exactly what the product ships.

The active set is process-wide (the job runner's worker thread sees it too):
activate() for a process that runs one configuration, override() for a scope.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .config import Settings


@dataclass(frozen=True)
class Params:
    # --- reconstruct: GPS (real phone fixes near buildings are ~10 m off, Florence Baptistry eval)
    gps_sigma_m: float = 6.0                 # 1 sigma per axis for a photo that reports no accuracy
    gps_sigma_min_m: float = 2.0             # floor for a reported GPSHPositioningError
    gps_inlier_sigmas: float = 3.0           # RANSAC inlier threshold is max(this x sigma, gps_inlier_min_m)
    gps_inlier_min_m: float = 10.0
    max_gps_scale_error: float = 0.35        # predicted relative scale error (= yaw error in rad) GPS may leave
    # --- reconstruct: compass and the pseudo-compass ("every photo looks at the landmark centre")
    compass_sigma_deg: float = 15.0          # phone compasses are poor next to steel and concrete
    compass_yaw_min_sigma_deg: float = 3.0   # readings at one site share magnetic disturbances
    compass_outlier_sigmas: float = 3.0      # compass_yaw drops headings this many sigmas off the mean
    pseudo_compass_sigma_deg: float = 25.0
    pseudo_compass_min_sigma_deg: float = 10.0   # facade photos aim at parts of the building, not its centre
    max_yaw_conflict_deg: float = 40.0       # GPS yaw this far from the (pseudo-)compass yaw is not trusted
    # --- reconstruct: gravity (estimate_up)
    up_prior_weight: float = 0.01            # pull of the (pitch-biased) mean camera up
    up_min_eigen_gap: float = 0.02           # below this the tilt is unobservable: mean up
    up_common_height_fraction: float = 0.25  # cameras this far (x horizontal spread) off the median height are elevated
    # --- reconstruct: matching and mapping
    exhaustive_max_images: int = 60          # larger sets use GPS-guided spatial matching
    spatial_max_neighbors: int = 30
    spatial_max_distance_m: float = 80.0     # photos of the same side of a landmark
    focal_refine_min_images: int = 12        # smaller sets keep the EXIF focal length fixed
    mapping_attempts: int = 3                # sets of at most small_set_images always get all of them
    mapping_attempts_large: int = 2          # larger sets only retry a suspicious result
    small_set_images: int = 30
    min_candidate_points: int = 100          # a mapped model with fewer points is degenerate
    retry_flagged_fraction: float = 0.05     # retry when more cameras than this disagree with their sensors ...
    retry_compass_error_deg: float = 20.0    # ... or the mean compass disagreement (capped at 90 deg) exceeds this
    # --- reconstruct: per-camera sanity checks against the photo's own sensors
    gross_compass_sigmas: float = 3.0        # this many compass sigmas off its compass: wrong, not noisy
    far_from_gps_factor: float = 3.0         # this many GPS inlier thresholds from its own fix: wrong depth or facade
    focal_ratio_min: float = 0.25            # focal / longer image side outside [min, max]: degenerate camera
    focal_ratio_max: float = 12.0
    # --- validation
    analysis_max_side_px: int = 800          # blur and dHash are measured on a copy this size
    # --- qa
    qa_max_facing_offset_deg: float = 90.0   # no compass and the estimate looks this far from the landmark: flagged
    qa_full_sector_min_registered: int = 2   # registered photos that make a face-ready sector "covered"

    @property
    def gross_compass_deg(self) -> float:
        return self.gross_compass_sigmas * self.compass_sigma_deg

    @property
    def focal_ratio_range(self) -> tuple[float, float]:
        return self.focal_ratio_min, self.focal_ratio_max

    def gps_inlier_threshold_m(self, gps_accuracy_m: float | None) -> float:
        """georegister's inlier threshold for one photo: max(gps_inlier_sigmas x sigma, gps_inlier_min_m)."""
        sigma = self.gps_sigma_m if gps_accuracy_m is None or not math.isfinite(gps_accuracy_m) else gps_accuracy_m
        return max(self.gps_inlier_sigmas * max(sigma, self.gps_sigma_min_m), self.gps_inlier_min_m)


DEFAULTS = Params()
PARAM_NAMES = tuple(f.name for f in dataclasses.fields(Params))

# Settings fields a pipeline configuration may change. The rest is where data lives and how the
# process runs (data dir, inline jobs, kept work dirs, crash retries), or chosen separately (reconstructor).
SETTINGS_NOT_TUNABLE = frozenset({"data_dir", "reconstructor", "inline_jobs", "keep_work_dirs", "max_job_attempts"})
SETTINGS_NAMES = tuple(f.name for f in dataclasses.fields(Settings) if f.name not in SETTINGS_NOT_TUNABLE)

_active = DEFAULTS
_guard = threading.Lock()


def current() -> Params:
    """The parameter set the algorithms use right now."""
    return _active


def activate(params: Params) -> Params:
    """Make params the process-wide set; returns the previous one."""
    global _active
    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    with _guard:
        previous, _active = _active, params
    return previous


@contextmanager
def using(params: Params) -> Iterator[Params]:
    """Use params inside the block, then restore the previous set."""
    previous = activate(params)
    try:
        yield params
    finally:
        activate(previous)


def override(**changes: Any):
    """Context manager: the current set with changes (unknown names raise ValueError)."""
    return using(with_overrides(changes, base=current()))


def _coerce(name: str, value: Any, default: Any) -> Any:
    """value checked against the type of the default (ints stay ints; an int is fine for a float)."""
    if isinstance(default, bool) or isinstance(value, bool):
        raise ValueError(f"{name}: booleans are not valid here")
    if isinstance(default, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(f"{name}: expected an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name}: expected a finite number, got {value!r}")
        return float(value)
    raise ValueError(f"{name}: unsupported type {type(default).__name__}")


def with_overrides(overrides: Mapping[str, Any], base: Params = DEFAULTS) -> Params:
    """base with the named fields replaced; names outside PARAM_NAMES raise ValueError."""
    unknown = sorted(set(overrides) - set(PARAM_NAMES))
    if unknown:
        raise ValueError(f"unknown algorithm parameter(s) {unknown}; tunable: {', '.join(PARAM_NAMES)}")
    return dataclasses.replace(base, **{k: _coerce(k, v, getattr(DEFAULTS, k)) for k, v in overrides.items()})


def settings_with(settings: Settings, overrides: Mapping[str, Any]) -> Settings:
    """settings with whitelisted fields replaced; anything outside SETTINGS_NAMES raises ValueError."""
    unknown = sorted(set(overrides) - set(SETTINGS_NAMES))
    if unknown:
        raise ValueError(f"unknown or non-tunable setting(s) {unknown}; tunable: {', '.join(SETTINGS_NAMES)}")
    default = Settings(data_dir=settings.data_dir)
    return dataclasses.replace(settings, **{k: _coerce(k, v, getattr(default, k)) for k, v in overrides.items()})


def effective(settings: Settings, params: Params | None = None) -> dict[str, Any]:
    """Everything a pipeline run depends on: reconstructor, every tunable Settings field, every Params field."""
    params = current() if params is None else params
    return {
        "reconstructor": settings.reconstructor,
        "settings": {name: getattr(settings, name) for name in SETTINGS_NAMES},
        "params": dataclasses.asdict(params),
    }
