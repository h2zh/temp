"""Structure first in one call: a landmark's OSM building:part snapshot, judged with one model's photos, then stylized
(docs/STRUCTURE_FIRST.md). scripts/stylize_model.py and the benchmark's stylize step call the same functions:

    source = find_source(slug, lat, lon)          # the catalogue's snapshot; None when the landmark has none
    model, srun = run(scene, source, PhotoCache(dir))   # stylized model (summary["structure_run"]), StructureRun

With no source, run() is stylize_landmark() unchanged (footprint massing). With one:
1. the structure is loaded from the snapshot (memoised; every run gets its own copy);
2. the photos are registered on the outline by stylize.register_scene, the code stylize_landmark itself runs;
3. the judge (D15) runs only when that registration is accepted and the structure has at most
   Limits.max_parts_judged built parts. Unregistered, nothing measured from the photos could be published (stylize
   withdraws it), so nothing is measured; above the part limit the model keeps the OSM-parts massing as mapped;
4. D25 ("auto") runs only with at least correct.G5_MIN_MASKS licensed sky masks: never today. "survey" runs it
   without applying anything (benchmark and research); "off" skips it;
5. the publish gate's checks, then stylize_landmark with the judged structure, the same snap, masks and 3-D lines.

Masks: PRODUCT_MASKS holds the only sky mask crowd3d computes or accepts, A+R (crowd3d.skyline.sky_mask), the one
cleared for commercial use. Any other name is refused and there is no override: reproductions with other masks are
research drivers outside crowd3d. With one mask the skyline channel, the D15 frame scale check and D25's G5 have
nothing to compare, so D15 can adopt only where SfM points and 3-D lines agree with a reading of the part's own roof
(D27), and D25 does not run.

Limits (per model): max_seconds of wall time (checked before each photo stage: a stage that would start late is
skipped and the model is stylized with what was measured so far) and max_rss_mb of peak memory; the record says
whether both held."""

from __future__ import annotations

import copy
import hashlib
import json
import resource
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import correct as C
from . import correct_measure as M
from . import judge as J
from . import lines3d as L3
from . import skyline as K
from . import structure as S
from . import stylize as st

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "eval" / "landmarks.json"
PRODUCT_MASKS = {"A+R": K.sky_mask}           # the licensed sky masks: name -> mask of an RGB photo
MASK_VERSION = {"A+R": 1}                     # in the cache key: bump when the mask's code changes its output
D25_MODES = ("auto", "survey", "off")


def check_mask_names(names) -> None:
    """Refuse any sky mask that is not a licensed product mask, and one mask under two names."""
    names = list(names)
    bad = sorted({str(n) for n in names} - set(PRODUCT_MASKS))
    if bad:
        raise ValueError(f"sky mask(s) {bad} are not licensed product masks (only {sorted(PRODUCT_MASKS)}): "
                         "crowd3d refuses them")
    if len(set(names)) != len(names):
        raise ValueError(f"sky masks {names}: a mask may be used once (one mask under two names counts twice)")


@dataclass(frozen=True)
class Limits:
    max_seconds: float = 1200.0       # wall time per model: the structure step and its stylization
    max_rss_mb: float = 4096.0        # peak resident memory per model
    max_parts_judged: int = 60        # above this many built parts the photos judge nothing (OSM-parts massing only)
    max_cameras_per_part: int = 60
    d25: str = "auto"                 # "auto" | "survey" | "off" (see the module docstring)
    masks: tuple[str, ...] = ("A+R",)
    lines: L3.LinesParams = field(default_factory=lambda: replace(L3.DEFAULT_LINES, detector="lsd", threads=2))


DEFAULT_LIMITS = Limits()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class StructureSource:
    """A landmark's OSM parts snapshot and main outline, and the ENU frame (the scene's: the landmark's centre)."""
    slug: str
    osm_parts: Path
    osm_parts_sha256: str
    outline: Path
    lat: float
    lon: float

    def to_meta(self) -> dict[str, Any]:
        def rel(p: Path) -> str:
            p = Path(p).resolve()
            return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
        return {"slug": self.slug, "osm_parts": rel(self.osm_parts), "osm_parts_sha256": self.osm_parts_sha256,
                "outline": rel(self.outline), "outline_sha256": _sha256(self.outline), "lat": self.lat, "lon": self.lon}


def catalog_entry(catalog: Path, slug: str) -> dict[str, Any] | None:
    """The catalogue's entry of this slug (a list of entries, or {"landmarks": [...]}); None when it has none."""
    data = json.loads(Path(catalog).read_text(encoding="utf-8"))
    entries = data["landmarks"] if isinstance(data, dict) else data
    return next((e for e in entries if e.get("slug") == slug), None)


def find_source(slug: str, lat: float, lon: float, catalog: Path = CATALOG, osm_parts: Path | None = None,
                outline: Path | None = None) -> StructureSource | None:
    """The landmark's snapshot (the catalogue's osm_parts block, its sha256 checked; osm_parts: another file) and
    outline (the catalogue's osm file unless given), in the ENU frame at (lat, lon). None when the landmark has no
    snapshot or no outline. The binding is by slug only."""
    entry = catalog_entry(catalog, slug) or {}
    root = Path(catalog).resolve().parent.parent
    if osm_parts is None:
        block = entry.get("osm_parts")
        if not block:
            return None
        osm_parts = root / block["file"]
        sha = _sha256(osm_parts) if osm_parts.is_file() else None
        if sha != block["sha256"]:
            what = "missing" if sha is None else "not the snapshot the catalogue lists"
            raise ValueError(f"{slug}: {osm_parts} is {what}")
    else:
        sha = _sha256(osm_parts)
    if outline is None:
        file = (entry.get("osm") or {}).get("file")
        if not file:
            return None
        outline = root / file
    return StructureSource(slug, Path(osm_parts), sha, Path(outline), float(lat), float(lon))


_LOADED: dict[tuple, S.StructureModel] = {}


def load_structure(source: StructureSource) -> S.StructureModel:
    """The structure of the snapshot (memoised by snapshot, outline and frame: Pisa's 284 parts take 14 s); a copy."""
    key = (source.slug, source.osm_parts_sha256, _sha256(source.outline), source.lat, source.lon)
    if key not in _LOADED:
        if len(_LOADED) >= 4:
            _LOADED.clear()
        _LOADED[key] = S.load_osm_structure(source.osm_parts, source.outline, source.lat, source.lon, source.slug)
    return copy.deepcopy(_LOADED[key])


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


class PhotoCache:
    """Per-photo results reused across models and runs: sky masks (PNG, by photo id, size and mask version) and the
    3-D lines' segments (crowd3d.lines3d's cache). root None: nothing is kept."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else None
        self.stats = {"masks_read": 0, "masks_computed": 0}

    @property
    def segments_dir(self) -> Path | None:
        return None if self.root is None else self.root / "segments-1"

    def mask_file(self, cam: st.PosedCamera, name: str = "A+R") -> Path | None:
        if self.root is None:
            return None
        return (self.root / f"mask-{_safe(name)}-{MASK_VERSION[name]}"
                / f"{_safe(cam.photo_id or cam.name)}-{cam.width}x{cam.height}.png")

    def masks(self, cameras: list[st.PosedCamera], name: str = "A+R") -> dict[int, np.ndarray]:
        """{camera index: bool sky mask (camera height x width)}; photos that cannot be read are left out."""
        check_mask_names([name])
        out: dict[int, np.ndarray] = {}
        for i, cam in enumerate(cameras):
            f = self.mask_file(cam, name)
            m = None
            if f is not None and f.is_file():
                im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if im is not None and im.shape == (cam.height, cam.width):
                    m = im > 127
                    self.stats["masks_read"] += 1
            if m is None:
                try:
                    rgb = cam.load()
                except (OSError, ValueError, TypeError, AttributeError):
                    continue
                m = np.asarray(PRODUCT_MASKS[name](rgb), bool)
                if m.shape != (cam.height, cam.width):
                    m = cv2.resize(m.astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST) > 0
                self.stats["masks_computed"] += 1
                if f is not None:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(f), m.astype(np.uint8) * 255)
            out[i] = m
        return out


def _status_mb(key: str) -> float | None:
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def _reset_peak() -> None:
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as f:
            f.write("5")                       # VmHWM back to the current size (Linux)
    except OSError:
        pass


def _peak_mb() -> float:
    hwm = _status_mb("VmHWM")
    return round(hwm if hwm is not None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _plain(o: Any) -> Any:
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def _adoption(a: dict[str, Any]) -> dict[str, Any]:
    return _plain({"id": a.get("id"), "rule": a.get("rule"), "kind": a.get("kind"), "part": a.get("part"),
                   "param": a.get("param"), "value": a.get("value"), "map_value": a.get("map_value"),
                   "channels": sorted(a.get("channels") or {}), "active": a.get("active", True)})


def _params_of(model: S.StructureModel) -> dict[tuple[str, str], tuple]:
    return {(p.id, k): (v.value, v.source) for p in model.specs for k, v in p.params.items()}


def revert_all_equals_map(model: S.StructureModel, mapped: S.StructureModel) -> bool:
    """Reverting every active adoption (newest first) gives back the mapped structure's values and their sources (the
    states keep what the judge found: a reverted adoption leaves its part disputed, its ticket open)."""
    rv = copy.deepcopy(model)
    for a in reversed([a for a in rv.adoptions if a.get("active", True)]):
        J.revert_adoption(rv, a["id"])
    return _params_of(rv) == _params_of(mapped)


@dataclass
class StructureRun:
    structure: S.StructureModel              # judged (and corrected) when the photos could measure it
    registration: st.SceneRegistration
    masks: dict[str, dict[int, np.ndarray]]  # {mask name: {camera index: mask}}; empty when nothing was judged
    lines: Any                               # crowd3d.lines3d.LinesResult or None
    record: dict[str, Any]                   # JSON-safe: what ran, what it decided, timings, memory, limits


def build_structure(scene: st.SceneInput, source: StructureSource, cache: PhotoCache | None = None,
                    limits: Limits = DEFAULT_LIMITS, params: st.StyleParams = st.DEFAULT_PARAMS,
                    log=None) -> StructureRun:
    """Steps 1-4 of the module docstring for one model (scene: the model's points, cameras and outline)."""
    check_mask_names(limits.masks)
    if limits.d25 not in D25_MODES:
        raise ValueError(f"d25 must be one of {D25_MODES}, not {limits.d25!r}")
    cache = cache or PhotoCache(None)
    stats0 = dict(cache.stats)
    _reset_peak()                       # the process's VmHWM: peak_rss_mb is this model's (a process-wide reset)
    t0, c0 = time.perf_counter(), time.process_time()
    T: dict[str, float] = {}
    t = [t0]

    def lap(stage: str) -> None:
        now = time.perf_counter()
        T[stage] = round(now - t[0], 2)
        t[0] = now

    def late(stage: str) -> str | None:
        spent = time.perf_counter() - t0
        return f"time limit: {spent:.0f} s of {limits.max_seconds:.0f} s spent before {stage}" \
            if spent > limits.max_seconds else None

    structure = load_structure(source)
    mapped = copy.deepcopy(structure)
    lap("load")
    reg = st.register_scene(scene, params)
    lap("register")
    n_built = len(structure.built)
    quality = (np.asarray(scene.point_quality, float) if reg.points is not None and scene.point_quality is not None
               and len(scene.point_quality) == len(reg.points) else None)
    snap_scale = float(reg.snap.scale) if reg.snap is not None and reg.snap.accepted else 1.0
    masks: dict[str, dict[int, np.ndarray]] = {}
    lines = None
    judge: dict[str, Any] = {"status": "skipped"}
    d25: dict[str, Any] = {"mode": limits.d25, "ran": False, "records": []}
    gate = None
    if not reg.photo_ok:
        judge["reason"] = (f"registration not accepted ({reg.registration.get('reason') or 'unknown'}): nothing "
                           "measured from these photos could be published")
    elif n_built > limits.max_parts_judged:
        judge["reason"] = f"{n_built} built parts > {limits.max_parts_judged}: OSM-parts massing as mapped"
    elif not n_built:
        judge["reason"] = "the snapshot builds no part"
    else:
        judge["reason"] = late("the sky masks")
        if not judge["reason"]:
            masks = {name: cache.masks(reg.cameras, name) for name in limits.masks}
            lap("masks")
            judge["reason"] = late("the 3-D lines")
        if not judge["reason"]:
            lines = L3.lines_channel(structure, reg.cameras, reg.points, quality, snap_scale, limits.lines,
                                     cache_dir=cache.segments_dir, log=log)
            lap("lines3d")
            judge["reason"] = late("the judge")
        if not judge["reason"]:
            ev = J.PhotoEvidence(reg.cameras, masks, reg.points, quality, lines=lines, snap_scale=snap_scale,
                                 max_cameras_per_part=limits.max_cameras_per_part, lines_params=limits.lines)
            res = J.judge_landmark(structure, ev, log=log)
            lap("judge")
            fd = res.get("frame_decision") or {}
            judge = {"status": "ran", "reason": None, "parts_judged": len(res["decisions"]),
                     "frame": {"adopt": bool(fd.get("adopt")), "adopted": (res.get("frame_adoption") or {}).get("id"),
                               "why": fd.get("why") or ""},
                     "skyline": sorted({str((getattr(d, "channels", None) or {}).get(J.SKYLINE, {}).get("why", ""))[:80]
                                        for d in res["decisions"].values()} - {""}),
                     "timings_s": res["timings_s"]}
            why_late = late("D25") if limits.d25 != "off" else None
            if why_late:
                d25["why"] = why_late
            else:
                d25 = _d25(structure, limits, reg, scene, masks, lines, quality, snap_scale, log)
                lap("d25")
            gate = {"check_decisions": bool(J.check_decisions(structure)["passed"]),
                    "check_structure": bool(S.check_structure(structure)["passed"]),
                    "revert_all_equals_map": revert_all_equals_map(structure, mapped)}
            lap("checks")
    if not d25["ran"] and not d25.get("why"):
        d25["why"] = "off" if limits.d25 == "off" else "nothing was judged"
    states = Counter(structure.part(b.id).params["height"].state for b in structure.built)
    seconds = round(time.perf_counter() - t0, 2)
    record = {
        "source": source.to_meta(),
        "registration": {"accepted": bool(reg.registration.get("accepted")), "reason": reg.registration.get("reason")},
        "parts_built": n_built,
        "masks": {name: len(m) for name, m in masks.items()},
        "lines3d": None if lines is None else {"detector": lines.detector, "stats": _plain(lines.stats)},
        "judge": judge,
        "adoptions": [_adoption(a) for a in structure.adoptions],
        "height_states": dict(states),
        "tickets_open": sum(1 for x in structure.tickets if x.get("open", True)),
        "d25": d25,
        "gate": gate,
        "cache": {k: v - stats0.get(k, 0) for k, v in cache.stats.items()},     # this model's reads / computations
        "timings_s": T,
        "seconds": seconds,
        "cpu_s": round(time.process_time() - c0, 2),
        "peak_rss_mb": _peak_mb(),
    }
    return StructureRun(structure, reg, masks, lines, record)


def _d25(structure: S.StructureModel, limits: Limits, reg: st.SceneRegistration, scene: st.SceneInput,
         masks: dict[str, dict[int, np.ndarray]], lines: Any, quality: np.ndarray | None, snap_scale: float,
         log=None) -> dict[str, Any]:
    mode = limits.d25
    if mode == "off":
        return {"mode": mode, "ran": False, "records": []}
    if mode == "auto" and len(masks) < C.G5_MIN_MASKS:
        return {"mode": mode, "ran": False, "records": [],
                "why": f"G5 needs {C.G5_MIN_MASKS} sky masks and {len(masks)} is licensed ({', '.join(masks)}): "
                       "D25 not run"}
    grade = C.publish_grade(reg.snap, scene.snap_prior, reg.points)
    ev = M.Evidence25(reg.cameras, masks, reg.points, quality, lines=lines, lines_params=limits.lines,
                      snap_scale=snap_scale, grade=grade, snap_meta=reg.snap.to_meta() if reg.snap else None,
                      max_cams=limits.max_cameras_per_part)
    apply = mode == "auto"
    res = M.correct_landmark(structure if apply else copy.deepcopy(structure), ev, log=log, apply=apply)
    fires = sorted(pid for pid, d in (res.get("decisions") or {}).items() if (d.get("final") or {}).get("fires"))
    return _plain({"mode": mode, "ran": True, "applied": apply, "grade": grade, "records": res.get("records") or [],
                   "reverted_by_g5": res.get("records_reverted_by_g5"), "fires": fires, "why": res.get("why")})


def stylize(scene: st.SceneInput, srun: StructureRun, params: st.StyleParams = st.DEFAULT_PARAMS) -> st.StylizedModel:
    """stylize_landmark with the judged structure, on the same snap, sky masks and 3-D lines."""
    sc = replace(scene, structure=srun.structure, snap_override=srun.registration.snap,
                 sky_masks=srun.masks.get("A+R"), lines3d=srun.lines)
    return st.stylize_landmark(sc, params)


def run(scene: st.SceneInput, source: StructureSource | None, cache: PhotoCache | None = None,
        limits: Limits = DEFAULT_LIMITS, params: st.StyleParams = st.DEFAULT_PARAMS,
        log=None) -> tuple[st.StylizedModel, StructureRun | None]:
    """One model, structure first when the landmark has a snapshot: (stylized model, its StructureRun), the model's
    summary["structure_run"] the record with the whole run's time and peak memory and whether the limits held.
    (stylize_landmark unchanged, None) without one."""
    if source is None:
        return st.stylize_landmark(scene, params), None
    t0 = time.perf_counter()
    srun = build_structure(scene, source, cache, limits, params, log)
    model = stylize(scene, srun, params)
    rec = srun.record
    total_s, peak = round(time.perf_counter() - t0, 2), _peak_mb()
    over = ([f"time {total_s:.0f} s > {limits.max_seconds:.0f} s"] if total_s > limits.max_seconds else []) + \
           ([f"memory {peak:.0f} MB > {limits.max_rss_mb:.0f} MB"] if peak > limits.max_rss_mb else [])
    rec["total"] = {"seconds": total_s, "peak_rss_mb": peak}
    rec["limits"] = {"max_seconds": limits.max_seconds, "max_rss_mb": limits.max_rss_mb,
                     "max_parts_judged": limits.max_parts_judged, "d25": limits.d25, "within": not over, "over": over}
    model.summary["structure_run"] = rec
    return model, srun


# ---------------------------------------------------------------- the product's stylize step (one child process)

PHOTO_ATTRIBUTION = "Facade colours derived from this landmark's check-in photos (Crowd3D contributors)"
STAGE_SHARE = 0.75        # the photo stages may start until this share of the child's time (the stylization: the rest)


def _qa_failures(qa: dict[str, Any]) -> list[str]:
    gates = ("audit", "glb", "structure")
    return ([k for k in gates if isinstance(qa.get(k), dict) and not qa[k].get("passed", True)]
            + [k for k, v in qa.items() if k not in gates and k != "passed" and isinstance(v, (int, float))
               and not isinstance(v, bool) and v])


def run_model(spec: dict[str, Any]) -> dict[str, Any]:
    """The product's stylize step for one published model; crowd3d.jobs runs it in a child process
    (`python -m crowd3d.structure_pipeline run --spec spec.json`). Inputs: the model's saved map (its sparse model; the
    photos' processed copies by image name) and its model JSON, the landmark (name, centre) and its catalogue slug.
    Structure first on the slug's OSM parts snapshot (run()), the GeoAnchor and snap prior as scripts/stylize_model.py
    takes them. Writes into spec["out"] only, plus the photo cache spec["cache"]: <model id>.glb when the stylized model
    passed its QA, <model id>.style.json (its summary) and result.json (what the job runner records). It never opens
    the product database. Returns the result: ok, error, registration {accepted, reason}, and what the structure step
    did."""
    from .geo import LocalENU
    from .scene_io import facts_from_entry, load_footprint_rings, load_job, model_anchor, snap_prior

    t0, c0 = time.perf_counter(), time.process_time()
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    lm, m = spec["landmark"], spec["model"]
    catalog = Path(spec["catalog"])
    result: dict[str, Any] = {"ok": False, "model_id": m["id"], "error": None}
    try:
        source = find_source(spec["slug"], lm["lat"], lm["lon"], catalog=catalog)
        if source is None:
            raise ValueError(f"{spec['slug']}: the catalogue {catalog} has no OSM parts snapshot and outline for it")
        entry = catalog_entry(catalog, spec["slug"]) or {}
        footprint, holes = load_footprint_rings(source.outline, LocalENU(lm["lat"], lm["lon"]))
        meta_path = Path(m["meta"])
        anchor = model_anchor(meta_path) if st.DEFAULT_PARAMS.snap_use_anchor else None
        images = {name: Path(path) for name, path in (m.get("images") or {}).items()}
        points, colors, cams, loaded_from, quality, tracks = load_job(
            None, meta_path, with_tracks=True, anchor=anchor, sparse_dir=Path(m["sparse"]), image_path_of=images.get)
        osm = [(entry.get(k) or {}).get("attribution") for k in ("osm", "osm_parts")]
        scene = st.SceneInput(lm["name"], footprint, points, colors, cams, facts_from_entry(entry, source.outline),
                              [a for a in dict.fromkeys(osm) if a] + [PHOTO_ATTRIBUTION], snap=True,
                              holes=tuple(holes), point_quality=quality,
                              snap_prior=snap_prior(meta_path, anchor=anchor), point_tracks=tracks)
        seconds = float(spec.get("max_seconds") or DEFAULT_LIMITS.max_seconds)
        model, _ = run(scene, source, PhotoCache(spec.get("cache")),
                       replace(DEFAULT_LIMITS, max_seconds=STAGE_SHARE * seconds))
    except (SystemExit, st.StylizeError, ValueError, OSError) as e:      # (load_job says what is wrong by SystemExit)
        result["error"] = f"{type(e).__name__}: {e}"
        result["seconds"] = round(time.perf_counter() - t0, 2)
        (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        return result
    s = model.summary
    rec = s.get("structure_run") or {}
    reg = s.get("registration") or {}
    seconds = round(time.perf_counter() - t0, 2)
    summary = _plain({**s, "source": loaded_from, "total_s": seconds})
    (out / f"{m['id']}.style.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    passed = bool(s["qa"]["passed"])
    if passed:
        (out / f"{m['id']}.glb").write_bytes(model.glb)
    judge = rec.get("judge") or {}
    result.update(
        ok=passed, error=None if passed else "the stylized model failed its QA: " + ", ".join(_qa_failures(s["qa"])),
        registration={"accepted": bool(reg.get("accepted")), "reason": reg.get("reason")},
        massing_source=s.get("massing_source"), parts_built=rec.get("parts_built"),
        judge={"status": judge.get("status"), "reason": judge.get("reason")},
        adoptions=sum(1 for a in rec.get("adoptions") or [] if a.get("active")),
        observed_walls=s.get("observed_walls"), num_walls=s.get("num_walls"),
        glb_bytes=len(model.glb) if passed else None, seconds=seconds, cpu_s=round(time.process_time() - c0, 2),
        peak_rss_mb=_peak_mb(), within_limits=(rec.get("limits") or {}).get("within"),
        source={"slug": source.slug, "osm_parts_sha256": source.osm_parts_sha256})
    (out / "result.json").write_text(json.dumps(_plain(result), indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    """`python -m crowd3d.structure_pipeline run --spec spec.json`: run_model in this process (the job runner's child);
    exit status 0 when it wrote a GLB."""
    import argparse
    import signal

    ap = argparse.ArgumentParser(prog="python -m crowd3d.structure_pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="stylize one published model from a job runner spec (JSON)")
    r.add_argument("--spec", type=Path, required=True)
    args = ap.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("kill_after_s") and hasattr(signal, "alarm"):
        signal.alarm(int(spec["kill_after_s"]))      # outlives no dead server by long (its runner kills it earlier)
    cv2.setNumThreads(int(spec.get("threads") or 2))
    return 0 if run_model(spec)["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
