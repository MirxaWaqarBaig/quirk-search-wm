"""Eight-layer ATLAS barrel used by arXiv:1708.02243.

The paper follows the pixel and SCT barrels and keeps eight layers, so the
geometry is the Run-2 layout: the insertable B-layer plus three pixel layers
and four SCT layers. Pixel hits are placed at pixel centres. SCT hits are
smeared by twice the intrinsic resolution quoted for the 2009 inner-detector
performance note. The primary vertex is taken as known and moved to the origin
before the plane fit, as assumed in Sec. III.C.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quirk_search.dynamics import C_MM_PER_NS, PairTrajectory, Trajectory
from quirk_search.yaml_util import load_yaml_file
from quirk_search import CONFIG_DIR


@dataclass
class Layer:
    id: int
    radius_mm: float
    z_half_mm: float
    kind: str
    pitch_rphi_mm: float = 0.05
    pitch_z_mm: float = 0.4
    sigma_rphi_mm: float = 0.034
    sigma_z_mm: float = 1.16


@dataclass
class Hit:
    x: float
    y: float
    z: float
    layer: int
    source: str
    time_ns: float

    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class Detector:
    layers: list[Layer]
    b_field_t: float = 2.0
    beamspot_sigma_z_mm: float = 45.0
    conversion_radius_mm: float = 28.0
    timing_ns: float = 25.0

    def layer_by_id(self, layer_id: int) -> Layer:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(layer_id)

    @property
    def outer_radius_mm(self) -> float:
        return max(layer.radius_mm for layer in self.layers)

    @property
    def n_layers(self) -> int:
        return len(self.layers)


@dataclass
class MeasuredEvent:
    reconstructed: list[Hit]
    quirk_layers: set[int] = field(default_factory=set)
    antiquirk_layers: set[int] = field(default_factory=set)
    vertex_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def fiducial(self, n_layers: int = 8) -> bool:
        return len(self.quirk_layers) >= n_layers and len(self.antiquirk_layers) >= n_layers


def default_detector() -> Detector:
    """Run-2 pixel+SCT barrel radii and the paper's hit resolutions."""
    pixel = dict(kind="pixel", pitch_rphi_mm=0.05, pitch_z_mm=0.40, sigma_rphi_mm=0.05, sigma_z_mm=0.40)
    sct = dict(kind="sct", pitch_rphi_mm=0.08, pitch_z_mm=1.16, sigma_rphi_mm=0.034, sigma_z_mm=1.16)
    specs = [
        (1, 33.25, 330.0, {**pixel, "pitch_z_mm": 0.25}),
        (2, 50.5, 400.5, pixel),
        (3, 88.5, 400.5, pixel),
        (4, 122.5, 400.5, pixel),
        (5, 299.0, 746.0, sct),
        (6, 371.0, 746.0, sct),
        (7, 443.0, 746.0, sct),
        (8, 514.0, 746.0, sct),
    ]
    layers = [Layer(i, r, z, **kw) for i, r, z, kw in specs]
    return Detector(layers=layers)


def detector_from_config(path=None) -> Detector:
    path = CONFIG_DIR / "detector_atlas_barrel.yaml" if path is None else path
    raw = load_yaml_file(path)
    layers = []
    for item in raw["layers"]:
        layers.append(
            Layer(
                id=int(item["id"]),
                radius_mm=float(item["radius_mm"]),
                z_half_mm=float(item["z_half_mm"]),
                kind=str(item["kind"]),
                pitch_rphi_mm=float(item["pitch_rphi_mm"]),
                pitch_z_mm=float(item["pitch_z_mm"]),
                sigma_rphi_mm=float(item["sigma_rphi_mm"]),
                sigma_z_mm=float(item["sigma_z_mm"]),
            )
        )
    return Detector(
        layers=layers,
        b_field_t=float(raw["magnetic_field_t"]),
        beamspot_sigma_z_mm=float(raw["beamspot_sigma_z_mm"]),
        conversion_radius_mm=float(raw["conversion_radius_mm"]),
        timing_ns=float(raw.get("timing_ns", 25.0)),
    )


def _angle_difference(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _cylinder_crossings(traj: Trajectory, radius: float) -> list[tuple[float, np.ndarray]]:
    xyz = traj.xyz
    time = traj.time_mm
    radius2 = radius * radius
    found = []
    for i in range(1, len(xyz)):
        x0, y0, z0 = xyz[i - 1]
        x1, y1, z1 = xyz[i]
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        a = dx * dx + dy * dy
        if a < 1e-18:
            continue
        b = 2.0 * (x0 * dx + y0 * dy)
        c = x0 * x0 + y0 * y0 - radius2
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            continue
        root = np.sqrt(disc)
        for sign in (-1.0, 1.0):
            u = (-b + sign * root) / (2.0 * a)
            if 0.0 < u <= 1.0:
                point = np.array([x0 + u * dx, y0 + u * dy, z0 + u * dz], dtype=float)
                when = time[i - 1] + u * (time[i] - time[i - 1])
                found.append((float(when), point))
    return found


def _dedupe_times(crossings: list[tuple[float, np.ndarray]], min_gap_mm: float) -> list[tuple[float, np.ndarray]]:
    if not crossings:
        return []
    ordered = sorted(crossings, key=lambda item: item[0])
    kept = [ordered[0]]
    for when, point in ordered[1:]:
        if when - kept[-1][0] < min_gap_mm:
            continue
        kept.append((when, point))
    return kept


def _apply_resolution(point: np.ndarray, layer: Layer, rng: np.random.Generator) -> np.ndarray:
    """Place the hit using the coordinate that actually resolves the plane.

    Pixels are digitised to the pixel centre in r-phi, which is the 50 um
    pitch. SCT hits are smeared by twice the 17 um intrinsic r-phi accuracy.
    The SCT stereo z uncertainty is about 1 mm, larger than the paper's
    0.1 mm thickness cut, so it is not applied as a shift of the hit before
    the unweighted plane fit. Pixel z is still digitised to the pixel length.
    """
    radius = layer.radius_mm
    phi = float(np.arctan2(point[1], point[0]))
    z = float(point[2])
    if layer.kind == "pixel":
        pitch_phi = layer.pitch_rphi_mm / radius
        phi = float(np.round(phi / pitch_phi) * pitch_phi)
        z = float(np.round(z / layer.pitch_z_mm) * layer.pitch_z_mm)
    else:
        phi = phi + float(rng.normal(0.0, layer.sigma_rphi_mm / radius))
    return np.array([radius * np.cos(phi), radius * np.sin(phi), z], dtype=float)


def _hits_close(a: Hit, b: Hit, layer: Layer) -> bool:
    dphi = _angle_difference(np.arctan2(a.y, a.x), np.arctan2(b.y, b.x))
    drphi = layer.radius_mm * dphi
    dz = abs(a.z - b.z)
    if layer.kind == "pixel":
        return drphi <= layer.pitch_rphi_mm and dz <= layer.pitch_z_mm
    return drphi <= 2.0 * layer.sigma_rphi_mm and dz <= 2.0 * layer.sigma_z_mm


def _average_hit(cluster: list[Hit]) -> Hit:
    sources = {hit.source for hit in cluster}
    if sources <= {"quirk", "antiquirk", "signal_merged"}:
        source = "quirk" if sources == {"quirk"} else "antiquirk" if sources == {"antiquirk"} else "signal_merged"
    elif "pileup" in sources and len(sources) == 1:
        source = "pileup"
    elif "conversion" in sources and len(sources) == 1:
        source = "conversion"
    else:
        source = "mixed"
    return Hit(
        x=float(np.mean([hit.x for hit in cluster])),
        y=float(np.mean([hit.y for hit in cluster])),
        z=float(np.mean([hit.z for hit in cluster])),
        layer=cluster[0].layer,
        source=source,
        time_ns=float(np.min([hit.time_ns for hit in cluster])),
    )


def merge_hits(hits: list[Hit], detector: Detector) -> list[Hit]:
    """Merge hits that land in the same or a neighbouring pixel, as in Sec. III.A."""
    grouped: dict[int, list[Hit]] = {}
    for hit in hits:
        grouped.setdefault(hit.layer, []).append(hit)
    merged: list[Hit] = []
    for layer_id, group in grouped.items():
        layer = detector.layer_by_id(layer_id)
        used = [False] * len(group)
        for i, hit in enumerate(group):
            if used[i]:
                continue
            cluster = [hit]
            used[i] = True
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                if _hits_close(hit, group[j], layer):
                    cluster.append(group[j])
                    used[j] = True
            merged.append(_average_hit(cluster))
    return merged


def intersect_trajectory(
    detector: Detector,
    traj: Trajectory,
    source: str,
    rng: np.random.Generator,
    vertex_mm: np.ndarray | None = None,
) -> tuple[list[Hit], set[int]]:
    """Exact cylinder crossings, then pixel quantisation or SCT smearing.

    Acceptance is applied in the lab frame. Returned coordinates are relative
    to the primary vertex.
    """
    vertex = np.zeros(3) if vertex_mm is None else np.asarray(vertex_mm, dtype=float)
    hits: list[Hit] = []
    layers_hit: set[int] = set()
    for layer in detector.layers:
        crossings = _dedupe_times(_cylinder_crossings(traj, layer.radius_mm), min_gap_mm=1.0)
        for time_mm, point in crossings:
            if abs(point[2]) > layer.z_half_mm:
                continue
            time_ns = time_mm / C_MM_PER_NS
            if time_ns > detector.timing_ns:
                continue
            measured = _apply_resolution(point, layer, rng)
            hits.append(
                Hit(
                    x=float(measured[0] - vertex[0]),
                    y=float(measured[1] - vertex[1]),
                    z=float(measured[2] - vertex[2]),
                    layer=layer.id,
                    source=source,
                    time_ns=float(time_ns),
                )
            )
            layers_hit.add(layer.id)
    return hits, layers_hit


def measure_pair(
    detector: Detector,
    trajectory: PairTrajectory,
    rng: np.random.Generator,
) -> MeasuredEvent:
    quirk_hits, quirk_layers = intersect_trajectory(
        detector, trajectory.quirk, "quirk", rng, trajectory.vertex_mm
    )
    anti_hits, anti_layers = intersect_trajectory(
        detector, trajectory.antiquirk, "antiquirk", rng, trajectory.vertex_mm
    )
    reconstructed = merge_hits(quirk_hits + anti_hits, detector)
    return MeasuredEvent(
        reconstructed=reconstructed,
        quirk_layers=quirk_layers,
        antiquirk_layers=anti_layers,
        vertex_mm=np.asarray(trajectory.vertex_mm, dtype=float),
    )
