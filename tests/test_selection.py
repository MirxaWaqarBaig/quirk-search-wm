"""Detector config, backgrounds, one propagated quirk event, and yields."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quirk_search.backgrounds import overlay, photon_conversion_hits, pileup_hits
from quirk_search.detector import default_detector, detector_from_config
from quirk_search.dynamics import initial_pair_state, propagate_pair
from quirk_search.detector import measure_pair
from quirk_search.kinematics import simulate_from_state
from quirk_search.plane_finder import PaperCuts, find_plane
from quirk_search.selection import (
    CUT_FLOW,
    CutFlow,
    EfficiencyStats,
    cuts_from_config,
    passes_signal_region,
    run_sample_parallel,
)
from quirk_search.sensitivity import (
    TOP_CROSS_SECTION_PB,
    contour_label,
    cross_section_pb,
    dy_cross_section_pb,
    expected_events,
    qcd_cross_section_pb,
)
from quirk_search.yaml_util import load_yaml_file
from quirk_search import CONFIG_DIR


class SelectionTest(unittest.TestCase):
    def test_configs_match_the_paper_cuts_and_the_eight_layers(self):
        cuts = cuts_from_config()
        self.assertEqual(cuts.delta_phi, 0.1)
        self.assertEqual(cuts.delta_z_mm, 20.0)
        self.assertEqual(cuts.delta_s_seed_mm, 0.5)
        self.assertEqual(cuts.delta_w_seed_mm, 10.0)
        self.assertEqual(cuts.delta_s_final_mm, 0.1)
        self.assertEqual(cuts.delta_w_final_mm, 10.0)
        self.assertEqual(cuts.grow_factor, 3.0)
        self.assertEqual(cuts.pt_trigger_gev, 200.0)
        detector = detector_from_config()
        reference = default_detector()
        self.assertEqual([layer.radius_mm for layer in detector.layers], [layer.radius_mm for layer in reference.layers])
        self.assertEqual(detector.b_field_t, 2.0)
        self.assertEqual(detector.beamspot_sigma_z_mm, 45.0)
        raw = load_yaml_file(CONFIG_DIR / "benchmarks.yaml")
        self.assertEqual(raw["luminosity_fb"], 300.0)
        self.assertEqual(raw["benchmarks"][0]["mass_gev"], 800.0)
        self.assertEqual(raw["benchmarks"][1]["charged_fraction"], 0.3)

    def test_pileup_count_and_conversion_hits(self):
        detector = default_detector()
        rng = np.random.default_rng(2)
        pileup = pileup_hits(detector, rng, mu=2, unassociated_per_interaction=3)
        self.assertEqual(len(pileup), 2 * 3 * 8)
        self.assertTrue(all(hit.source == "pileup" for hit in pileup))
        conversions = photon_conversion_hits(detector, rng, photon_energy_gev=8.0, eta=0.1, phi=0.3)
        self.assertGreaterEqual(len(conversions), 4)
        self.assertTrue(all(hit.source == "conversion" for hit in conversions))
        mixed = overlay(conversions[:2], pileup[:4], detector)
        self.assertGreaterEqual(len(mixed), 4)

    def test_propagated_quirk_pair_is_reconstructed(self):
        # Boosted pair, oscillation along z, width inside the 1 cm cut.
        detector = default_detector()
        detector.beamspot_sigma_z_mm = 0.0
        mass = 800.0
        lam = 2000.0
        speed = 0.53
        beta = np.array([0.90, 0.0, 0.0])
        state = initial_pair_state(
            mass_gev=mass,
            lambda_ev=lam,
            v_internal=speed,
            axis=np.array([0.0, 0.0, 1.0]),
            beta_lab=beta,
            vertex_mm=np.zeros(3),
        )
        self.assertGreater(state.p1[0], 0.0)
        self.assertGreater(state.p2[0], 0.0)
        event = simulate_from_state(
            state,
            detector,
            np.random.default_rng(7),
            pair_pt_gev=float(np.hypot(state.p1[0] + state.p2[0], state.p1[1] + state.p2[1])),
            mode="dy",
            pt_trigger_gev=200.0,
        )
        self.assertTrue(event.triggered)
        self.assertTrue(
            event.measured.fiducial(),
            msg=f"quirk layers {sorted(event.measured.quirk_layers)} "
            f"anti {sorted(event.measured.antiquirk_layers)} "
            f"hits {len(event.measured.reconstructed)}",
        )
        result = find_plane(event.measured.reconstructed, PaperCuts())
        self.assertTrue(
            result.found,
            msg=f"ds={result.delta_s_mm:.4f} dw={result.delta_w_mm:.4f} seeds={result.n_seeds_tested} "
            f"counts={result.layer_counts.tolist()} nhits={len(event.measured.reconstructed)}",
        )
        self.assertTrue(
            passes_signal_region(result),
            msg=f"ds={result.delta_s_mm:.4f} dw={result.delta_w_mm:.4f} counts={result.layer_counts.tolist()}",
        )

    def test_cutflow_is_sequential_and_defines_the_efficiencies(self):
        flow = CutFlow()
        full = {name: True for name, _definition in CUT_FLOW}
        dropped = dict(full)
        dropped["fiducial"] = False
        flow.add(full)
        flow.add(dropped)
        self.assertEqual(flow.counts["generated"], 2)
        self.assertEqual(flow.counts["trigger"], 2)
        self.assertEqual(flow.counts["fiducial"], 1)
        self.assertEqual(flow.counts["hit_pattern"], 1)
        summary = flow.as_dict()
        self.assertAlmostEqual(summary["epsilon_trigger"], 1.0)
        self.assertAlmostEqual(summary["epsilon_fiducial"], 0.5)
        self.assertAlmostEqual(summary["epsilon_reco"], 1.0)
        self.assertIn("epsilon_total", summary["definitions"])
        self.assertEqual([row["cut"] for row in summary["cuts"]], [name for name, _definition in CUT_FLOW])

    def test_parallel_sample_covers_every_event(self):
        sample = run_sample_parallel(
            mass_gev=800.0,
            lambda_ev=4000.0,
            mode="dy",
            n_events=4,
            seed=11,
            workers=2,
        )
        self.assertEqual(sample.stats.n_generated, 4)
        self.assertEqual(sample.cutflow.counts["generated"], 4)
        self.assertEqual(sorted(row["event"] for row in sample.events), [0, 1, 2, 3])

    def test_efficiency_factorisation_and_event_yield(self):
        stats = EfficiencyStats()
        stats.add(True, True, True)
        stats.add(True, True, False)
        stats.add(True, False, False)
        stats.add(False, False, False)
        self.assertAlmostEqual(stats.epsilon_trigger, 0.75)
        self.assertAlmostEqual(stats.epsilon_fiducial, 2.0 / 3.0)
        self.assertAlmostEqual(stats.epsilon_reco, 0.5)
        self.assertAlmostEqual(stats.epsilon_total, 0.75 * (2.0 / 3.0) * 0.5)
        # 1 pb * 1 fb^-1 = 1000 events before efficiency.
        self.assertAlmostEqual(expected_events(1.0, 1.0, 1.0, 1.0), 1000.0)
        self.assertAlmostEqual(expected_events(0.0014, 300.0, 0.04, 0.3), 5.04, places=6)
        self.assertEqual(contour_label(5.0), "discovery")
        self.assertEqual(contour_label(3.0), "exclusion")
        self.assertEqual(contour_label(1.0), "below")

    def test_cross_sections_fall_and_top_is_normalised(self):
        self.assertAlmostEqual(qcd_cross_section_pb(173.0), TOP_CROSS_SECTION_PB, delta=1.0)
        self.assertGreater(qcd_cross_section_pb(500.0), qcd_cross_section_pb(1600.0))
        self.assertGreater(qcd_cross_section_pb(1600.0), 0.0)
        low = dy_cross_section_pb(200.0)
        high = dy_cross_section_pb(800.0)
        self.assertGreater(low, high)
        self.assertGreater(high, 0.0)
        self.assertAlmostEqual(cross_section_pb(173.0, "qcd"), TOP_CROSS_SECTION_PB, delta=1.0)

    def test_propagation_helper_is_used_by_the_event(self):
        # Guards the public propagation entry point used above.
        detector = default_detector()
        state = initial_pair_state(200.0, 2000.0, 0.5, np.array([1.0, 0.0, 0.0]), np.array([0.8, 0.0, 0.0]))
        trajectory = propagate_pair(state, b_field_t=0.0, t_max_ns=5.0, outer_radius_mm=detector.outer_radius_mm)
        measured = measure_pair(detector, trajectory, np.random.default_rng(0))
        self.assertGreater(len(measured.reconstructed), 0)


if __name__ == "__main__":
    unittest.main()
