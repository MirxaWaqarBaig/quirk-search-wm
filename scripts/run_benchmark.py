#!/usr/bin/env python3
"""Simulate one Table I benchmark point and write the efficiency breakdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quirk_search import RESULTS_DIR
from quirk_search.selection import load_detector_and_cuts, run_sample
from quirk_search.sensitivity import contour_label, cross_section_pb, expected_events
from quirk_search.yaml_util import load_yaml_file
from quirk_search import CONFIG_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="dy_800", help="Benchmark name in configs/benchmarks.yaml")
    parser.add_argument("--lambda-ev", type=float, default=4000.0, help="Confinement scale in eV")
    parser.add_argument("--events", type=int, default=8, help="Generated events, before the trigger")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml_file(CONFIG_DIR / "benchmarks.yaml")
    chosen = next(item for item in config["benchmarks"] if item["name"] == args.name)
    detector, cuts = load_detector_and_cuts()
    sample = run_sample(
        detector,
        cuts,
        mass_gev=float(chosen["mass_gev"]),
        lambda_ev=float(args.lambda_ev),
        mode=str(chosen["mode"]),
        n_events=int(args.events),
        seed=int(config["seed"] if args.seed is None else args.seed),
    )
    stats = sample.stats
    sigma_pb = cross_section_pb(float(chosen["mass_gev"]), str(chosen["mode"]))
    n_expected = expected_events(
        sigma_pb,
        float(config["luminosity_fb"]),
        stats.epsilon_total,
        float(chosen["charged_fraction"]),
    )
    payload = {
        "benchmark": chosen["name"],
        "mass_gev": float(chosen["mass_gev"]),
        "lambda_ev": float(args.lambda_ev),
        "mode": chosen["mode"],
        "charged_fraction": float(chosen["charged_fraction"]),
        "cross_section_pb": sigma_pb,
        "luminosity_fb": float(config["luminosity_fb"]),
        "expected_events_300fb": n_expected,
        "contour": contour_label(
            n_expected,
            discovery=float(config["discovery_events"]),
            exclusion=float(config["exclusion_events"]),
        ),
        **stats.as_dict(),
        "cutflow": sample.cutflow.as_dict(),
    }
    out_dir = RESULTS_DIR / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{chosen['name']}_lambda{args.lambda_ev:g}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
