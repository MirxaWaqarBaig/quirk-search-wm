"""Plane finder of Knapen et al., arXiv:1708.02243, Sec. III.C.

Hits are fit to an uncentred tensor

    T_ij = 1/(N-1) sum_a x_i^a x_j^a

whose eigenvalues are the squared thickness Delta s, the squared width
Delta w, and the longitudinal scale. Seeding uses layers 8 and 7. The
remaining layers are added from the outside in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quirk_search.detector import Hit


@dataclass
class PaperCuts:
    """Cuts from Sec. III.C and the signal region in Sec. IV. Values are mm."""

    delta_phi: float = 0.1
    delta_z_mm: float = 20.0
    delta_s_seed_mm: float = 0.5
    delta_w_seed_mm: float = 10.0
    delta_s_iter_mm: float = 0.5
    delta_w_iter_mm: float = 10.0
    grow_factor: float = 3.0
    delta_s_final_mm: float = 0.1
    delta_w_final_mm: float = 10.0
    # Pairs on the two seed layers that are this far apart in azimuth or z
    # cannot satisfy the 10 mm width cut on the outer barrel. The cut removes
    # combinations the paper's fit would reject.
    seed_pair_phi: float = 0.5
    seed_pair_z_mm: float = 80.0
    n_layers: int = 8
    pt_trigger_gev: float = 200.0


@dataclass
class PlaneFit:
    delta_s_mm: float
    delta_w_mm: float
    n1: np.ndarray
    n2: np.ndarray
    n3: np.ndarray
    eigenvalues: np.ndarray


@dataclass
class PlaneResult:
    found: bool
    delta_s_mm: float = np.inf
    delta_w_mm: float = np.inf
    indices: list[int] = field(default_factory=list)
    layer_counts: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=int))
    n1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n3: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_seeds_tested: int = 0
    cut_flags: dict = field(default_factory=dict)


def stack_hits(hits: list[Hit]) -> np.ndarray:
    if not hits:
        return np.zeros((0, 3))
    return np.array([[hit.x, hit.y, hit.z] for hit in hits], dtype=float)


def fit_plane(points: np.ndarray) -> PlaneFit:
    """Smallest eigenvector of T is the plane normal. Eq. (10) and Eq. (11)."""
    points = np.asarray(points, dtype=float)
    n_points = len(points)
    if n_points < 2:
        raise ValueError("a plane fit needs at least two hits")
    tensor = (points.T @ points) / (n_points - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    n1 = eigenvectors[:, 0]
    n2 = eigenvectors[:, 1]
    n3 = eigenvectors[:, 2]
    if float(np.mean(points @ n3)) < 0.0:
        n3 = -n3
    return PlaneFit(
        delta_s_mm=float(np.sqrt(eigenvalues[0])),
        delta_w_mm=float(np.sqrt(eigenvalues[1])),
        n1=n1,
        n2=n2,
        n3=n3,
        eigenvalues=eigenvalues,
    )


def _angle_difference(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _circular_mean(phi_a: float, phi_b: float) -> float:
    return float(np.arctan2(np.sin(phi_a) + np.sin(phi_b), np.cos(phi_a) + np.cos(phi_b)))


def close_pairs(indices: list[int], xyz: np.ndarray, delta_phi: float, delta_z_mm: float) -> list[tuple[int, int]]:
    """Hit pairs on one layer inside the paper's Delta phi and Delta z window."""
    if len(indices) < 2:
        return []
    idx = np.asarray(indices, dtype=int)
    phi = np.arctan2(xyz[idx, 1], xyz[idx, 0])
    z = xyz[idx, 2]
    order = np.argsort(phi)
    pairs: list[tuple[int, int]] = []
    n = len(order)
    for a in range(n):
        i = int(order[a])
        for b in range(a + 1, n):
            j = int(order[b])
            if phi[j] - phi[i] > delta_phi:
                break
            if abs(z[i] - z[j]) < delta_z_mm:
                pairs.append((int(idx[i]), int(idx[j])))
    # Pairs that cross the phi = ±pi boundary.
    for a in range(n):
        i = int(order[a])
        if phi[i] > -np.pi + delta_phi:
            break
        for b in range(n - 1, -1, -1):
            j = int(order[b])
            if j == i:
                continue
            wrapped = (phi[i] + 2.0 * np.pi) - phi[j]
            if wrapped > delta_phi:
                break
            if wrapped <= 0.0:
                continue
            if abs(z[i] - z[j]) < delta_z_mm:
                pairs.append((int(idx[i]), int(idx[j])))
    return pairs


def _accept_growth(new: PlaneFit, current_s: float, current_w: float, cuts: PaperCuts) -> bool:
    """Accept a hit only when neither Delta s nor Delta w grows by more than 3."""
    if current_s > 1e-8:
        if new.delta_s_mm > cuts.grow_factor * current_s:
            return False
    elif new.delta_s_mm > cuts.grow_factor * cuts.delta_s_iter_mm:
        return False
    if current_w > 1e-8:
        if new.delta_w_mm > cuts.grow_factor * current_w:
            return False
    elif new.delta_w_mm > cuts.grow_factor * cuts.delta_w_iter_mm:
        return False
    return True


def _layer_index_lists(hits: list[Hit], n_layers: int) -> dict[int, list[int]]:
    grouped = {layer: [] for layer in range(1, n_layers + 1)}
    for i, hit in enumerate(hits):
        if hit.layer in grouped:
            grouped[hit.layer].append(i)
    return grouped


def _counts(indices: list[int], hits: list[Hit], n_layers: int) -> np.ndarray:
    counts = np.zeros(n_layers, dtype=int)
    for i in indices:
        layer = hits[i].layer
        if 1 <= layer <= n_layers:
            counts[layer - 1] += 1
    return counts


def find_plane(hits: list[Hit], cuts: PaperCuts | None = None) -> PlaneResult:
    """Run the two-stage plane finder. The returned plane has Delta w < 1 cm."""
    cuts = PaperCuts() if cuts is None else cuts
    xyz = stack_hits(hits)
    by_layer = _layer_index_lists(hits, cuts.n_layers)
    outer = cuts.n_layers
    second = cuts.n_layers - 1
    pairs_outer = close_pairs(by_layer[outer], xyz, cuts.delta_phi, cuts.delta_z_mm)
    pairs_second = close_pairs(by_layer[second], xyz, cuts.delta_phi, cuts.delta_z_mm)
    flags = {
        "seed_pair_layer8": bool(pairs_outer),
        "seed_pair_layer7": bool(pairs_second),
        "seed_prefilter": False,
        "seed_delta_s": False,
        "seed_delta_w": False,
        "seed_forward": False,
        "inward_walk": False,
    }
    best: PlaneResult | None = None
    n_tested = 0
    phi = np.arctan2(xyz[:, 1], xyz[:, 0]) if len(xyz) else np.zeros(0)

    for i1, i2 in pairs_outer:
        phi_outer = _circular_mean(float(phi[i1]), float(phi[i2]))
        z_outer = 0.5 * (xyz[i1, 2] + xyz[i2, 2])
        for j1, j2 in pairs_second:
            phi_second = _circular_mean(float(phi[j1]), float(phi[j2]))
            z_second = 0.5 * (xyz[j1, 2] + xyz[j2, 2])
            if _angle_difference(phi_outer, phi_second) > cuts.seed_pair_phi:
                continue
            if abs(z_outer - z_second) > cuts.seed_pair_z_mm:
                continue
            flags["seed_prefilter"] = True
            seed = [i1, i2, j1, j2]
            n_tested += 1
            fit = fit_plane(xyz[seed])
            if fit.delta_s_mm >= cuts.delta_s_seed_mm:
                continue
            flags["seed_delta_s"] = True
            if fit.delta_w_mm >= cuts.delta_w_seed_mm:
                continue
            flags["seed_delta_w"] = True
            if np.any(xyz[seed] @ fit.n3 <= 0.0):
                continue
            flags["seed_forward"] = True
            selected = list(seed)
            current_s = fit.delta_s_mm
            current_w = fit.delta_w_mm
            for layer in range(second - 1, 0, -1):
                reference = fit_plane(xyz[selected])
                candidates = []
                for index in by_layer[layer]:
                    if index in selected:
                        continue
                    point = xyz[index]
                    if float(point @ reference.n3) <= 0.0:
                        continue
                    if abs(float(point @ reference.n1)) >= cuts.delta_s_iter_mm:
                        continue
                    if abs(float(point @ reference.n2)) >= cuts.delta_w_iter_mm:
                        continue
                    candidates.append(index)
                candidates.sort(key=lambda index: abs(float(xyz[index] @ reference.n1)))
                for index in candidates:
                    trial = fit_plane(xyz[selected + [index]])
                    if not _accept_growth(trial, current_s, current_w, cuts):
                        continue
                    selected.append(index)
                    current_s = trial.delta_s_mm
                    current_w = trial.delta_w_mm
            final = fit_plane(xyz[selected])
            if final.delta_w_mm >= cuts.delta_w_final_mm:
                continue
            flags["inward_walk"] = True
            if best is None or final.delta_s_mm < best.delta_s_mm:
                best = PlaneResult(
                    found=True,
                    delta_s_mm=final.delta_s_mm,
                    delta_w_mm=final.delta_w_mm,
                    indices=selected,
                    layer_counts=_counts(selected, hits, cuts.n_layers),
                    n1=final.n1,
                    n2=final.n2,
                    n3=final.n3,
                    n_seeds_tested=n_tested,
                )
    if best is None:
        return PlaneResult(found=False, n_seeds_tested=n_tested, cut_flags=flags)
    best.n_seeds_tested = n_tested
    best.cut_flags = flags
    return best
