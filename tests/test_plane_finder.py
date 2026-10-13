"""Seeding, the uncentred tensor, and the signal-region hit pattern."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quirk_search.detector import Hit
from quirk_search.plane_finder import PaperCuts, close_pairs, find_plane, fit_plane, stack_hits
from quirk_search.selection import hit_pattern_ok, passes_signal_region


RADII = [33.25, 50.5, 88.5, 122.5, 299.0, 371.0, 443.0, 514.0]


def _planted_strip(half_angle: float = 0.004) -> list[Hit]:
    """Two rays in the x-z plane. Every hit has y = 0, so Delta s is zero."""
    hits = []
    for layer, radius in enumerate(RADII, start=1):
        for sign in (-1.0, 1.0):
            theta = sign * half_angle
            hits.append(
                Hit(
                    x=radius * math.cos(theta),
                    y=0.0,
                    z=radius * math.sin(theta),
                    layer=layer,
                    source="quirk" if sign < 0 else "antiquirk",
                    time_ns=1.0,
                )
            )
    return hits


class PlaneFinderTest(unittest.TestCase):
    def test_tensor_is_uncentred(self):
        # y = 5 mm with independent x and z does not lie on a plane through the origin.
        points = np.array([[radius, 5.0, sign * 20.0] for radius in RADII for sign in (-1.0, 1.0)])
        fit = fit_plane(points)
        self.assertGreater(fit.delta_s_mm, 1.0)
        planar = np.array([[radius, 0.0, 0.1 * radius] for radius in RADII])
        self.assertLess(fit_plane(planar).delta_s_mm, 1e-6)

    def test_seed_window_rejects_a_wide_pair(self):
        xyz = np.array(
            [
                [500.0, 0.0, 0.0],
                [500.0 * math.cos(0.4), 500.0 * math.sin(0.4), 0.0],
            ]
        )
        pairs = close_pairs([0, 1], xyz, delta_phi=0.1, delta_z_mm=20.0)
        self.assertEqual(pairs, [])
        tight = np.array([[500.0, 0.0, 0.0], [499.0, 5.0, 2.0]])
        self.assertEqual(len(close_pairs([0, 1], tight, 0.1, 20.0)), 1)

    def test_phi_wrap_pairs_hits_across_the_branch_cut(self):
        xyz = np.array(
            [
                [math.cos(3.13), math.sin(3.13), 0.0],
                [math.cos(-3.13), math.sin(-3.13), 1.0],
            ]
        )
        self.assertEqual(len(close_pairs([0, 1], xyz, 0.1, 20.0)), 1)

    def test_planted_coplanar_hits_pass_the_signal_region(self):
        hits = _planted_strip()
        # A far hit must not be pulled onto the plane.
        hits.append(Hit(100.0, 40.0, 0.0, layer=3, source="pileup", time_ns=0.0))
        result = find_plane(hits, PaperCuts())
        self.assertTrue(result.found, msg=f"seeds={result.n_seeds_tested} ds={result.delta_s_mm} dw={result.delta_w_mm}")
        self.assertLess(result.delta_s_mm, 0.1)
        self.assertLess(result.delta_w_mm, 10.0)
        self.assertTrue(passes_signal_region(result, PaperCuts()))
        self.assertNotIn(len(hits) - 1, result.indices)
        self.assertTrue(result.cut_flags["seed_pair_layer8"])
        self.assertTrue(result.cut_flags["seed_delta_s"])
        self.assertTrue(result.cut_flags["inward_walk"])

    def test_hits_out_of_the_plane_are_not_reconstructed(self):
        hits = []
        for layer, radius in enumerate(RADII, start=1):
            for y in (-20.0, 20.0):
                hits.append(Hit(math.sqrt(radius**2 - y**2), y, 0.0, layer, "pileup", 0.0))
        result = find_plane(hits, PaperCuts())
        self.assertFalse(result.found)
        self.assertFalse(passes_signal_region(result))

    def test_growth_factor_keeps_a_distant_hit_off_the_plane(self):
        hits = _planted_strip()
        # Inside the iterative distance gates this hit is not: |y| = 30 mm.
        distant = Hit(80.0, 30.0, 0.0, layer=3, source="pileup", time_ns=0.0)
        hits.append(distant)
        result = find_plane(hits, PaperCuts())
        self.assertTrue(result.found)
        self.assertNotIn(len(hits) - 1, result.indices)

    def test_hit_pattern_matches_section_iv(self):
        self.assertTrue(hit_pattern_ok(np.full(8, 2)))
        counts = np.full(8, 2)
        counts[3] = 1
        self.assertTrue(hit_pattern_ok(counts))
        counts[4] = 1
        self.assertFalse(hit_pattern_ok(counts))
        counts = np.full(8, 2)
        counts[0] = 0
        self.assertFalse(hit_pattern_ok(counts))

    def test_stack_preserves_order(self):
        hits = _planted_strip()
        xyz = stack_hits(hits)
        self.assertEqual(xyz.shape, (16, 3))
        self.assertAlmostEqual(xyz[0, 0], RADII[0] * math.cos(-0.004))


if __name__ == "__main__":
    unittest.main()
