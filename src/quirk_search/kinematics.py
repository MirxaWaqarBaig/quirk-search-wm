"""Quirk-pair kinematics with the paper's pT > 200 GeV trigger.

MadGraph and Pythia are not required. Events are back-to-back in the pair
rest frame, boosted by an ISR-like transverse momentum, and passed through
the straight-string equation of motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quirk_search.detector import Detector, MeasuredEvent, measure_pair
from quirk_search.dynamics import PairState, PairTrajectory, initial_pair_state, propagate_pair


@dataclass
class SignalEvent:
    state: PairState
    trajectory: PairTrajectory
    measured: MeasuredEvent
    pair_pt_gev: float
    pair_mass_gev: float
    triggered: bool
    mode: str
    lambda_ev: float
    mass_gev: float


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def random_axis(rng: np.random.Generator) -> np.ndarray:
    z = float(rng.uniform(-1.0, 1.0))
    phi = float(rng.uniform(-np.pi, np.pi))
    rho = np.sqrt(max(0.0, 1.0 - z * z))
    return np.array([rho * np.cos(phi), rho * np.sin(phi), z], dtype=float)


def pair_four_momentum(mass_gev: float, v_internal: float, beta_lab: np.ndarray) -> tuple[float, np.ndarray]:
    gamma = 1.0 / np.sqrt(1.0 - v_internal * v_internal)
    pair_mass = 2.0 * gamma * mass_gev
    beta = np.asarray(beta_lab, dtype=float)
    b2 = float(np.dot(beta, beta))
    gamma_lab = 1.0 / np.sqrt(1.0 - b2)
    momentum = gamma_lab * pair_mass * beta
    return pair_mass, momentum


def sample_beta_lab(
    rng: np.random.Generator,
    pair_mass_gev: float,
    pt_scale_gev: float,
    pz_scale_gev: float,
) -> tuple[np.ndarray, float]:
    """Falling pT spectrum. The transverse component is the trigger variable."""
    pt = float(rng.exponential(pt_scale_gev))
    pz = float(rng.normal(0.0, pz_scale_gev))
    phi = float(rng.uniform(-np.pi, np.pi))
    momentum = np.array([pt * np.cos(phi), pt * np.sin(phi), pz], dtype=float)
    energy = float(np.sqrt(pair_mass_gev * pair_mass_gev + np.dot(momentum, momentum)))
    beta = momentum / energy
    return beta, pt


def make_state(
    mass_gev: float,
    lambda_ev: float,
    v_internal: float,
    axis: np.ndarray,
    beta_lab: np.ndarray,
    vertex_mm: np.ndarray | None = None,
) -> PairState:
    return initial_pair_state(
        mass_gev=mass_gev,
        lambda_ev=lambda_ev,
        v_internal=v_internal,
        axis=axis,
        beta_lab=beta_lab,
        vertex_mm=vertex_mm,
    )


def simulate_from_state(
    state: PairState,
    detector: Detector,
    rng: np.random.Generator,
    pair_pt_gev: float,
    mode: str = "dy",
    pt_trigger_gev: float = 200.0,
    t_max_ns: float = 25.0,
) -> SignalEvent:
    trajectory = propagate_pair(
        state,
        b_field_t=detector.b_field_t,
        t_max_ns=t_max_ns,
        outer_radius_mm=detector.outer_radius_mm,
    )
    measured = measure_pair(detector, trajectory, rng)
    gamma = None
    # Pair mass follows from the initial momenta, before propagation.
    e1 = float(np.sqrt(np.dot(state.p1, state.p1) + state.mass_gev**2))
    e2 = float(np.sqrt(np.dot(state.p2, state.p2) + state.mass_gev**2))
    total_p = state.p1 + state.p2
    pair_mass = float(np.sqrt(max(0.0, (e1 + e2) ** 2 - np.dot(total_p, total_p))))
    del gamma
    return SignalEvent(
        state=state,
        trajectory=trajectory,
        measured=measured,
        pair_pt_gev=float(pair_pt_gev),
        pair_mass_gev=pair_mass,
        triggered=float(pair_pt_gev) > float(pt_trigger_gev),
        mode=mode,
        lambda_ev=state.lambda_ev,
        mass_gev=state.mass_gev,
    )


def simulate_signal_event(
    rng: np.random.Generator,
    detector: Detector,
    mass_gev: float,
    lambda_ev: float,
    mode: str = "dy",
    pt_trigger_gev: float = 200.0,
    v_range: tuple[float, float] = (0.35, 0.85),
) -> SignalEvent:
    """One event. Trigger is recorded; the event is still fully simulated."""
    v_internal = float(rng.uniform(*v_range))
    gamma = 1.0 / np.sqrt(1.0 - v_internal * v_internal)
    pair_mass = 2.0 * gamma * mass_gev
    pt_scale = 160.0 if mode == "dy" else 260.0
    beta, pair_pt = sample_beta_lab(rng, pair_mass, pt_scale_gev=pt_scale, pz_scale_gev=180.0)
    vertex = np.array([0.0, 0.0, float(rng.normal(0.0, detector.beamspot_sigma_z_mm))])
    state = make_state(mass_gev, lambda_ev, v_internal, random_axis(rng), beta, vertex)
    return simulate_from_state(
        state,
        detector,
        rng,
        pair_pt_gev=pair_pt,
        mode=mode,
        pt_trigger_gev=pt_trigger_gev,
    )
