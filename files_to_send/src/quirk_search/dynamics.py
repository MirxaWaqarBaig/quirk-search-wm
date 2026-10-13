"""Quirk propagation for the straight-string regime of arXiv:1708.02243.

Units inside the integrator are GeV, millimetres, and time in millimetres
(c = 1). The string tension is Lambda^2, with Lambda the confinement scale.
A 2 T solenoid along z supplies the Lorentz force used in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# hbar c = 197.3269804 MeV fm.
HBARC_GEV_MM = 1.973269804e-13
# pT [GeV] = 0.299792458 * |q| * B [T] * R [m].
KAPPA_GEV_PER_MM_T = 2.99792458e-4
C_MM_PER_NS = 299.792458
ELECTRON_MASS_GEV = 5.1099895e-4


def string_tension_gev_per_mm(lambda_ev: float) -> float:
    """String tension Lambda^2, converted from eV to GeV/mm."""
    lam_gev = float(lambda_ev) * 1e-9
    return (lam_gev * lam_gev) / HBARC_GEV_MM


def energy_from_momentum(momentum: np.ndarray, mass_gev: float) -> float:
    momentum = np.asarray(momentum, dtype=float)
    return float(np.sqrt(np.dot(momentum, momentum) + mass_gev * mass_gev))


def beta_from_momentum(momentum: np.ndarray, mass_gev: float) -> np.ndarray:
    momentum = np.asarray(momentum, dtype=float)
    return momentum / energy_from_momentum(momentum, mass_gev)


def lorentz_boost(momentum: np.ndarray, energy: float, beta: np.ndarray):
    """Boost a four-momentum by a velocity beta (in units of c)."""
    beta = np.asarray(beta, dtype=float)
    b2 = float(np.dot(beta, beta))
    momentum = np.asarray(momentum, dtype=float)
    if b2 < 1e-18:
        return momentum.copy(), float(energy)
    if b2 >= 1.0:
        raise ValueError("boost velocity must be slower than light")
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bdotp = float(np.dot(beta, momentum))
    boosted_p = momentum + beta * (((gamma - 1.0) / b2) * bdotp + gamma * energy)
    boosted_e = gamma * (energy + bdotp)
    return boosted_p, float(boosted_e)


def string_force(velocity: np.ndarray, n_out: np.ndarray, tension: float) -> np.ndarray:
    """Straight-string force in Eq. (3).

    ``n_out`` points from the partner toward this quirk. With that orientation
    the collinear force is ``-tension * n_out``, so dp/dt = -Lambda^2 on the
    outward axis and the motion reproduces Eq. (4).
    """
    velocity = np.asarray(velocity, dtype=float)
    n_out = np.asarray(n_out, dtype=float)
    v_parallel = float(np.dot(velocity, n_out))
    v_perp = velocity - v_parallel * n_out
    v_perp2 = float(np.dot(v_perp, v_perp))
    v_perp2 = min(max(v_perp2, 0.0), 1.0 - 1e-12)
    root = np.sqrt(1.0 - v_perp2)
    parenthesis = root * n_out + (v_parallel / root) * v_perp
    return -tension * parenthesis


def magnetic_force(velocity: np.ndarray, charge: float, b_field_t: float) -> np.ndarray:
    """dp/dt for B = (0, 0, B). ``velocity`` is in units of c."""
    vx, vy, _vz = np.asarray(velocity, dtype=float)
    scale = KAPPA_GEV_PER_MM_T * charge * b_field_t
    return scale * np.array([vy, -vx, 0.0])


@dataclass
class PairState:
    x1: np.ndarray
    x2: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    mass_gev: float
    lambda_ev: float
    charge1: float = 1.0
    charge2: float = -1.0

    def energies(self) -> tuple[float, float]:
        return (
            energy_from_momentum(self.p1, self.mass_gev),
            energy_from_momentum(self.p2, self.mass_gev),
        )


def initial_pair_state(
    mass_gev: float,
    lambda_ev: float,
    v_internal: float,
    axis: np.ndarray,
    beta_lab: np.ndarray,
    vertex_mm: np.ndarray | None = None,
    separation_mm: float = 1e-3,
    charge1: float = 1.0,
    charge2: float = -1.0,
) -> PairState:
    """Back-to-back quirks in their CM, then boosted into the lab.

    ``v_internal`` is the speed of each quirk in the pair CM, the ``v`` of
    Eq. (4). ``axis`` is that back-to-back axis. ``beta_lab`` is the velocity
    of the pair CM.
    """
    if not 0.0 < float(v_internal) < 1.0:
        raise ValueError("internal speed must lie between 0 and 1")
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        raise ValueError("axis must be non-zero")
    axis = axis / axis_norm
    gamma = 1.0 / np.sqrt(1.0 - v_internal * v_internal)
    energy_cm = gamma * mass_gev
    momentum_cm = gamma * mass_gev * v_internal
    beta = np.asarray(beta_lab, dtype=float)
    p1, energy1 = lorentz_boost(momentum_cm * axis, energy_cm, beta)
    p2, energy2 = lorentz_boost(-momentum_cm * axis, energy_cm, beta)
    del energy1, energy2
    relative = beta_from_momentum(p1, mass_gev) - beta_from_momentum(p2, mass_gev)
    relative_norm = np.linalg.norm(relative)
    direction = axis if relative_norm < 1e-12 else relative / relative_norm
    vertex = np.zeros(3) if vertex_mm is None else np.asarray(vertex_mm, dtype=float)
    half = 0.5 * float(separation_mm) * direction
    return PairState(
        x1=vertex + half,
        x2=vertex - half,
        p1=p1,
        p2=p2,
        mass_gev=float(mass_gev),
        lambda_ev=float(lambda_ev),
        charge1=float(charge1),
        charge2=float(charge2),
    )


def analytic_distance_from_cm_mm(t_mm, mass_gev: float, lambda_ev: float, v_internal: float):
    """Eq. (4): one quirk's distance from the CM for collinear motion and B = 0."""
    time = np.asarray(t_mm, dtype=float)
    speed = float(v_internal)
    gamma = 1.0 / np.sqrt(1.0 - speed * speed)
    tension = string_tension_gev_per_mm(lambda_ev)
    u = tension * time / mass_gev - speed * gamma
    return (mass_gev / tension) * (gamma - np.sqrt(1.0 + u * u))


def paper_amplitude_mm(mass_gev: float, lambda_ev: float, gamma: float) -> float:
    """Eq. (1) in exact form: (gamma - 1) * m_Q / Lambda^2."""
    return float((gamma - 1.0) * mass_gev / string_tension_gev_per_mm(lambda_ev))


def collinear_period_mm(mass_gev: float, lambda_ev: float, v_internal: float) -> float:
    """Round-trip time in Eq. (4), 2 v gamma m / Lambda^2, in mm."""
    speed = float(v_internal)
    gamma = 1.0 / np.sqrt(1.0 - speed * speed)
    return 2.0 * speed * gamma * mass_gev / string_tension_gev_per_mm(lambda_ev)


@dataclass
class Trajectory:
    time_mm: np.ndarray
    xyz: np.ndarray

    @property
    def time_ns(self) -> np.ndarray:
        return self.time_mm / C_MM_PER_NS


@dataclass
class PairTrajectory:
    quirk: Trajectory
    antiquirk: Trajectory
    vertex_mm: np.ndarray
    p_quirk: np.ndarray
    p_antiquirk: np.ndarray


def _derivatives(state_vec: np.ndarray, mass: float, tension: float, q1: float, q2: float, b_field: float):
    x1 = state_vec[0:3]
    x2 = state_vec[3:6]
    p1 = state_vec[6:9]
    p2 = state_vec[9:12]
    e1 = np.sqrt(np.dot(p1, p1) + mass * mass)
    e2 = np.sqrt(np.dot(p2, p2) + mass * mass)
    v1 = p1 / e1
    v2 = p2 / e2
    separation = x1 - x2
    distance = float(np.linalg.norm(separation))
    if distance < 1e-8 or tension <= 0.0:
        force1 = np.zeros(3)
        force2 = np.zeros(3)
    else:
        n_out = separation / distance
        force1 = string_force(v1, n_out, tension)
        force2 = string_force(v2, -n_out, tension)
    force1 = force1 + magnetic_force(v1, q1, b_field)
    force2 = force2 + magnetic_force(v2, q2, b_field)
    return np.concatenate([v1, v2, force1, force2])


def propagate_pair(
    state: PairState,
    b_field_t: float = 2.0,
    t_max_ns: float = 25.0,
    outer_radius_mm: float = 520.0,
    max_steps: int = 250_000,
    dt_mm: float | None = None,
) -> PairTrajectory:
    """Integrate Eq. (3) until both quirks leave the tracker, or 25 ns."""
    tension = string_tension_gev_per_mm(state.lambda_ev) if state.lambda_ev > 0.0 else 0.0
    mass = float(state.mass_gev)
    if dt_mm is None:
        dt = 0.25 if tension <= 0.0 else float(np.clip(0.04 * mass / tension, 0.02, 1.0))
    else:
        dt = float(dt_mm)
    t_max_mm = float(t_max_ns) * C_MM_PER_NS
    y = np.concatenate(
        [
            np.asarray(state.x1, dtype=float),
            np.asarray(state.x2, dtype=float),
            np.asarray(state.p1, dtype=float),
            np.asarray(state.p2, dtype=float),
        ]
    )
    vertex = 0.5 * (y[0:3] + y[3:6])
    times = [0.0]
    xs1 = [y[0:3].copy()]
    xs2 = [y[3:6].copy()]
    ps1 = [y[6:9].copy()]
    ps2 = [y[9:12].copy()]
    t = 0.0
    margin = float(outer_radius_mm) + 15.0
    for _ in range(int(max_steps)):
        if t >= t_max_mm:
            break
        step = min(dt, t_max_mm - t)
        k1 = _derivatives(y, mass, tension, state.charge1, state.charge2, b_field_t)
        k2 = _derivatives(y + 0.5 * step * k1, mass, tension, state.charge1, state.charge2, b_field_t)
        k3 = _derivatives(y + 0.5 * step * k2, mass, tension, state.charge1, state.charge2, b_field_t)
        k4 = _derivatives(y + step * k3, mass, tension, state.charge1, state.charge2, b_field_t)
        y = y + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += step
        times.append(t)
        xs1.append(y[0:3].copy())
        xs2.append(y[3:6].copy())
        ps1.append(y[6:9].copy())
        ps2.append(y[9:12].copy())
        r1 = float(np.hypot(y[0], y[1]))
        r2 = float(np.hypot(y[3], y[4]))
        if r1 > margin and r2 > margin:
            e1 = np.sqrt(y[6] * y[6] + y[7] * y[7] + y[8] * y[8] + mass * mass)
            e2 = np.sqrt(y[9] * y[9] + y[10] * y[10] + y[11] * y[11] + mass * mass)
            outward1 = (y[0] * y[6] + y[1] * y[7]) / e1
            outward2 = (y[3] * y[9] + y[4] * y[10]) / e2
            if outward1 > 0.0 and outward2 > 0.0:
                break
    return PairTrajectory(
        quirk=Trajectory(np.asarray(times), np.vstack(xs1)),
        antiquirk=Trajectory(np.asarray(times), np.vstack(xs2)),
        vertex_mm=np.asarray(vertex, dtype=float),
        p_quirk=np.vstack(ps1),
        p_antiquirk=np.vstack(ps2),
    )


def particle_energy_sum(state_or_momenta, mass_gev: float) -> float:
    if isinstance(state_or_momenta, PairState):
        e1, e2 = state_or_momenta.energies()
        return e1 + e2
    p1, p2 = state_or_momenta
    return energy_from_momentum(p1, mass_gev) + energy_from_momentum(p2, mass_gev)


def mechanical_energy_gev(x1, x2, p1, p2, mass_gev: float, lambda_ev: float) -> float:
    """Quirk energies plus string energy tension * separation. Conserved at B = 0."""
    tension = string_tension_gev_per_mm(lambda_ev)
    separation = float(np.linalg.norm(np.asarray(x1) - np.asarray(x2)))
    return particle_energy_sum((p1, p2), mass_gev) + tension * separation
