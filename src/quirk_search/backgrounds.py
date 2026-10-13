"""Pileup hits and photon conversions, the two backgrounds in Sec. III.D.

The pileup model places unassociated hits uniformly on each barrel layer.
The paper quotes O(1000) such hits per layer at <mu> = 50, which is the
default 20 unassociated hits per pileup interaction per layer. Conversions
are an equal-energy e+e- pair born on the beampipe and bent by the solenoid.
This is a toy of the Pythia-plus-detector chain in the paper, not that chain.
"""

from __future__ import annotations

import numpy as np

from quirk_search.detector import Detector, Hit, merge_hits
from quirk_search.dynamics import (
    ELECTRON_MASS_GEV,
    PairState,
    propagate_pair,
)
from quirk_search.detector import intersect_trajectory


def pileup_hits(
    detector: Detector,
    rng: np.random.Generator,
    mu: float = 50.0,
    unassociated_per_interaction: int = 20,
) -> list[Hit]:
    n_per_layer = int(round(float(mu) * int(unassociated_per_interaction)))
    hits: list[Hit] = []
    for layer in detector.layers:
        for _ in range(n_per_layer):
            phi = float(rng.uniform(-np.pi, np.pi))
            z = float(rng.uniform(-layer.z_half_mm, layer.z_half_mm))
            hits.append(
                Hit(
                    x=float(layer.radius_mm * np.cos(phi)),
                    y=float(layer.radius_mm * np.sin(phi)),
                    z=z,
                    layer=layer.id,
                    source="pileup",
                    time_ns=0.0,
                )
            )
    return hits


def _direction(eta: float, phi: float) -> np.ndarray:
    theta = 2.0 * np.arctan(np.exp(-eta))
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        dtype=float,
    )


def photon_conversion_hits(
    detector: Detector,
    rng: np.random.Generator,
    photon_energy_gev: float = 10.0,
    eta: float = 0.2,
    phi: float = 0.4,
    opening_rad: float = 0.01,
) -> list[Hit]:
    """Collimated e+e- from a beampipe conversion. Energy is shared equally."""
    direction = _direction(eta, phi)
    radius = detector.conversion_radius_mm
    d_xy = float(np.linalg.norm(direction[:2]))
    if d_xy < 1e-8:
        direction = np.array([1.0, 0.0, 0.0])
        d_xy = 1.0
    origin = direction * (radius / d_xy)
    each_energy = 0.5 * photon_energy_gev
    mass = ELECTRON_MASS_GEV
    momentum = np.sqrt(max(each_energy * each_energy - mass * mass, 0.0))

    def tilted(sign: float) -> np.ndarray:
        # Small opening in the z-x plane of the photon direction.
        axis = np.cross(direction, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        # Rodrigues rotation by sign * opening/2.
        angle = sign * 0.5 * opening_rad
        rotated = (
            direction * np.cos(angle)
            + np.cross(axis, direction) * np.sin(angle)
            + axis * np.dot(axis, direction) * (1.0 - np.cos(angle))
        )
        return momentum * rotated

    state = PairState(
        x1=origin.copy(),
        x2=origin.copy(),
        p1=tilted(+1.0),
        p2=tilted(-1.0),
        mass_gev=mass,
        lambda_ev=0.0,
        charge1=-1.0,
        charge2=+1.0,
    )
    trajectory = propagate_pair(
        state,
        b_field_t=detector.b_field_t,
        t_max_ns=detector.timing_ns,
        outer_radius_mm=detector.outer_radius_mm,
        dt_mm=1.0,
    )
    electron_hits, _ = intersect_trajectory(detector, trajectory.quirk, "conversion", rng, vertex_mm=np.zeros(3))
    positron_hits, _ = intersect_trajectory(
        detector, trajectory.antiquirk, "conversion", rng, vertex_mm=np.zeros(3)
    )
    return merge_hits(electron_hits + positron_hits, detector)


def overlay(signal_hits: list[Hit], background_hits: list[Hit], detector: Detector) -> list[Hit]:
    return merge_hits(list(signal_hits) + list(background_hits), detector)
