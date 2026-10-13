#!/usr/bin/env python3
"""Scan mass and string tension and label the 3-event and 5-event contours."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quirk_search import CONFIG_DIR, RESULTS_DIR
from quirk_search.selection import load_detector_and_cuts, run_sample
from quirk_search.sensitivity import contour_label, cross_section_pb, expected_events
from quirk_search.yaml_util import load_yaml_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=None, help="Override events per grid point")
    args = parser.parse_args()
    config = load_yaml_file(CONFIG_DIR / "benchmarks.yaml")
    scan = config["scan"]
    n_events = int(scan["events_per_point"] if args.events is None else args.events)
    detector, cuts = load_detector_and_cuts()
    rows = []
    seed = int(config["seed"])
    point = 0
    for mode in scan["modes"]:
        fraction = 0.3 if mode == "qcd" else 1.0
        for mass in scan["masses_gev"]:
            sigma_pb = cross_section_pb(float(mass), str(mode))
            for lam in scan["lambdas_ev"]:
                stats = run_sample(
                    detector,
                    cuts,
                    mass_gev=float(mass),
                    lambda_ev=float(lam),
                    mode=str(mode),
                    n_events=n_events,
                    seed=seed + point,
                ).stats
                point += 1
                n_expected = expected_events(
                    sigma_pb,
                    float(config["luminosity_fb"]),
                    stats.epsilon_total,
                    fraction,
                )
                row = {
                    "mode": mode,
                    "mass_gev": float(mass),
                    "lambda_ev": float(lam),
                    "cross_section_pb": sigma_pb,
                    "charged_fraction": fraction,
                    "expected_events": n_expected,
                    "contour": contour_label(
                        n_expected,
                        discovery=float(config["discovery_events"]),
                        exclusion=float(config["exclusion_events"]),
                    ),
                    **stats.as_dict(),
                }
                rows.append(row)
                print(
                    f"{mode:4} m={float(mass):7.1f} GeV  Lambda={float(lam):8.1f} eV  "
                    f"eps={stats.epsilon_total:.4f}  N={n_expected:.3f}  {row['contour']}"
                )
    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sensitivity.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
