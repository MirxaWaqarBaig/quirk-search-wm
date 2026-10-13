"""Equation of motion, Eq. (1) and Eq. (4), and cylinder crossings."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quirk_search.detector import default_detector, intersect_trajectory
from quirk_search.dynamics import (
    C_MM_PER_NS,
    PairState,
    analytic_distance_from_cm_mm,
    collinear_period_mm,
    initial_pair_state,
    mechanical_energy_gev,
    paper_amplitude_mm,
    propagate_pair,
    string_tension_gev_per_mm,
)


class DynamicsTest(unittest.TestCase):
    def test_amplitude_matches_the_paper_two_centimetre_rule(self):
        # m = 100 GeV, Lambda = 1 keV, gamma - 1 = 1 -> about 2 cm.
        amplitude = paper_amplitude_mm(100.0, 1000.0, gamma=2.0)
        self.assertAlmostEqual(amplitude, 20.0, delta=1.0)
        tension = string_tension_gev_per_mm(1000.0)
        self.assertAlmostEqual(amplitude, 1.0 * 100.0 / tension)

    def test_collinear_motion_follows_equation_4(self):
        mass = 100.0
        lam = 1000.0
        speed = math.sqrt(0.75)  # gamma = 2
        gamma = 2.0
        momentum = gamma * mass * speed
        eps = 1e-3
        state = PairState(
            x1=np.array([eps, 0.0, 0.0]),
            x2=np.array([-eps, 0.0, 0.0]),
            p1=np.array([momentum, 0.0, 0.0]),
            p2=np.array([-momentum, 0.0, 0.0]),
            mass_gev=mass,
            lambda_ev=lam,
        )
        period = collinear_period_mm(mass, lam, speed)
        trajectory = propagate_pair(
            state,
            b_field_t=0.0,
            t_max_ns=0.85 * period / C_MM_PER_NS,
            outer_radius_mm=1.0e9,
            dt_mm=0.05,
        )
        predicted = analytic_distance_from_cm_mm(trajectory.quirk.time_mm, mass, lam, speed)
        residual = trajectory.quirk.xyz[:, 0] - predicted
        self.assertLess(float(np.max(np.abs(residual))), 0.5)
        # The outward peak is (gamma - 1) m / tension.
        self.assertAlmostEqual(float(np.max(trajectory.quirk.xyz[:, 0])), paper_amplitude_mm(mass, lam, gamma), delta=0.5)

    def test_string_energy_is_conserved_without_a_magnetic_field(self):
        mass = 100.0
        lam = 1000.0
        speed = 0.6
        gamma = 1.0 / math.sqrt(1.0 - speed * speed)
        momentum = gamma * mass * speed
        state = PairState(
            x1=np.array([1e-3, 0.0, 0.0]),
            x2=np.array([-1e-3, 0.0, 0.0]),
            p1=np.array([momentum, 0.0, 0.0]),
            p2=np.array([-momentum, 0.0, 0.0]),
            mass_gev=mass,
            lambda_ev=lam,
        )
        initial = mechanical_energy_gev(state.x1, state.x2, state.p1, state.p2, mass, lam)
        period = collinear_period_mm(mass, lam, speed)
        trajectory = propagate_pair(
            state,
            b_field_t=0.0,
            t_max_ns=0.4 * period / C_MM_PER_NS,
            outer_radius_mm=1.0e9,
            dt_mm=0.05,
        )
        mid = len(trajectory.quirk.time_mm) // 2
        later = mechanical_energy_gev(
            trajectory.quirk.xyz[mid],
            trajectory.antiquirk.xyz[mid],
            trajectory.p_quirk[mid],
            trajectory.p_antiquirk[mid],
            mass,
            lam,
        )
        self.assertAlmostEqual(later, initial, delta=0.02 * initial)

    def test_opposite_charges_bend_in_opposite_directions(self):
        state = PairState(
            x1=np.array([0.0, 0.0, 0.0]),
            x2=np.array([0.0, 0.0, 1.0]),
            p1=np.array([200.0, 0.0, 0.0]),
            p2=np.array([200.0, 0.0, 0.0]),
            mass_gev=100.0,
            lambda_ev=0.0,
            charge1=1.0,
            charge2=-1.0,
        )
        trajectory = propagate_pair(
            state,
            b_field_t=2.0,
            t_max_ns=0.2,
            outer_radius_mm=1.0e9,
            dt_mm=0.2,
        )
        self.assertLess(trajectory.quirk.xyz[-1, 1], -1.0e-3)
        self.assertGreater(trajectory.antiquirk.xyz[-1, 1], 1.0e-3)

    def test_boost_preserves_the_pair_invariant_mass(self):
        mass = 800.0
        speed = 0.5
        gamma = 1.0 / math.sqrt(1.0 - speed * speed)
        beta = np.array([0.4, 0.1, 0.2])
        state = initial_pair_state(mass, 2000.0, speed, np.array([0.0, 0.0, 1.0]), beta)
        e1 = math.sqrt(float(np.dot(state.p1, state.p1)) + mass * mass)
        e2 = math.sqrt(float(np.dot(state.p2, state.p2)) + mass * mass)
        total_p = state.p1 + state.p2
        pair_mass = math.sqrt((e1 + e2) ** 2 - float(np.dot(total_p, total_p)))
        self.assertAlmostEqual(pair_mass, 2.0 * gamma * mass, delta=1e-4)

    def test_a_straight_track_crosses_every_barrel_layer(self):
        detector = default_detector()
        state = PairState(
            x1=np.zeros(3),
            x2=np.array([0.0, 0.0, 1.0]),
            p1=np.array([80.0, 0.0, 0.0]),
            p2=np.array([0.0, 80.0, 0.0]),
            mass_gev=50.0,
            lambda_ev=0.0,
        )
        trajectory = propagate_pair(
            state,
            b_field_t=0.0,
            t_max_ns=20.0,
            outer_radius_mm=detector.outer_radius_mm,
            dt_mm=0.5,
        )
        hits, layers = intersect_trajectory(
            detector,
            trajectory.quirk,
            "quirk",
            np.random.default_rng(1),
            vertex_mm=np.zeros(3),
        )
        self.assertEqual(layers, set(range(1, 9)))
        self.assertEqual(len(hits), 8)


if __name__ == "__main__":
    unittest.main()
